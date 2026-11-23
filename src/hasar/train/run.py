"""Glue between the CLI and the training loop and pre-training gates.

This module holds no logic of its own: it loads the configuration, builds the model and the
datasets, calls :func:`hasar.train.loop.train` or the gates of :mod:`hasar.preflight.gates`, and
prints a short Turkish summary. Everything it needs already lives in those modules.

Automatic fallback (design section 5)
-------------------------------------
When the main model is ChangeMamba and the compiled selective-scan kernel is not available on a
CUDA device, the run switches to ``cfg.model.fallback`` (DamageFormer Swin-T) with a Turkish
warning instead of silently training the slow pure-PyTorch scan.

torch is imported at module top: this module is only imported where torch exists.
"""

from __future__ import annotations

import dataclasses
import gc
import hashlib
import json
import math
import subprocess
import tempfile
import traceback
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from hasar.config import Config, config_hash, load_config, run_id
from hasar.data import io
from hasar.data.dataset import BrightDataset
from hasar.data.events import Split, check_no_leak, load_standard_lists, make_zero_shot_split
from hasar.data.normalize import compute_sar_stats
from hasar.data.summary import label_histogram
from hasar.fewshot.split import load_split
from hasar.guard import TurkeyLockError, require_fewshot_access
from hasar.models import registry
from hasar.preflight.gates import (
    EXPECTED_SPLIT,
    _result,
    clip_from_preflight,
    failed_gate,
    gate_grad_clip,
    gate_initial_loss,
    gate_kernel,
    gate_loaded_checkpoint,
    gate_metric_edges,
    gate_overfit,
    gate_reproduction,
    gate_resume,
    gate_sampler_length,
    gate_sar_connected,
    gate_speed,
    gate_split,
    preflight_is_current,
    write_preflight,
)
from hasar.train.loop import VAL_TAG, _train_sampler, train

#: Default folder of the Phase 3 reproduction records (gate 1).
DEFAULT_REPRO_DIR = Path("results") / "yeniden-uretim"
ANA_VAL13_RECORD = Path("results") / "ana" / "val13.json"
ANA_FINAL_STEP = 4000
ANA_PRETRAINED_REQUIRED = "Ana koşunun best.pt dosyasını --pretrained ile verin."

#: Approximate Colab CU price per hour by GPU (gate 8); an unknown GPU uses the A100 price.
CU_PER_HOUR = {"A100": 13.0, "L4": 5.0}


def cu_per_hour(device_name: str) -> float:
    """Return the CU price per hour for a GPU name such as ``"NVIDIA L4"`` (A100 price if unknown)."""
    for key, price in CU_PER_HOUR.items():
        if key in device_name:
            return price
    return CU_PER_HOUR["A100"]


def _git_commit() -> str:
    """Return the current git commit SHA, or ``"unknown"`` when git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _git_root() -> Path:
    """Find the current Git worktree root, falling back to the working directory."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            encoding="utf-8",
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return Path.cwd()
    root = result.stdout.strip()
    return Path(root) if root else Path.cwd()


def _verify_ana_sha256(handle: BinaryIO, checkpoint: Path, expected: str | None) -> None:
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(f"Checkpoint sha256 uyuşmazlığı: {checkpoint}; beklenen sha256: {expected}.")


def _require_pretrained(cfg: Config, pretrained: Path | str | None) -> None:
    if pretrained is None:
        if cfg.model.init == "ana_run":
            raise ValueError(ANA_PRETRAINED_REQUIRED)
        if cfg.model.init == "bright_zero_shot":
            raise ValueError(
                "DamageFormer zero-shot checkpoint'ini `hasar agirlik --config ...` ile indirip --pretrained ile verin."
            )


def build_model(cfg: Config, pretrained: Path | str | None, device: str) -> torch.nn.Module:
    """Build the configured model on ``device``, applying the automatic fallback rule."""
    from hasar.models import changemamba  # registers the model factories

    name = cfg.model.name
    if name == "changemamba_mmbda_tiny" and changemamba.scan_backend() != "cuda":
        if str(device).startswith("cuda"):
            print(
                "Uyarı: ChangeMamba selective_scan çekirdeği CUDA'da derlenemedi; "
                f"ana eğitim yedek modele geçiyor ({cfg.model.fallback})."
            )
            name = cfg.model.fallback

    if name == "changemamba_mmbda_tiny":
        model = registry.build(name, pretrained=str(pretrained) if pretrained is not None else None)
    elif name == "damageformer_swin_t":
        model = registry.build(name, pretrained=True)
    elif name == "bright_damageformer":
        from hasar.models import bright_baselines

        if cfg.model.init == "ana_run":
            if pretrained is None:
                raise ValueError(ANA_PRETRAINED_REQUIRED)
            checkpoint = Path(pretrained)
            with checkpoint.open("rb") as handle:
                _verify_ana_sha256(handle, checkpoint, cfg.model.init_sha256)
                handle.seek(0)
                payload = torch.load(handle, map_location="cpu", weights_only=True)
            step = payload.get("step") if isinstance(payload, dict) else None
            if step != ANA_FINAL_STEP:
                raise ValueError(f"Ana checkpoint adımı {ANA_FINAL_STEP} olmalı, okunan adım: {step}.")
            model = registry.build(name)
            model.load_state_dict(payload["model"])
            print(f"Ana checkpoint sha256: {cfg.model.init_sha256}")
        else:
            model = registry.build(name)
            if pretrained is not None:
                baseline = bright_baselines.BASELINES["damageformer"]
                if bright_baselines.file_md5(pretrained) != baseline.md5:
                    raise ValueError(
                        f"Checkpoint md5 uyuşmazlığı: {Path(pretrained).name}; "
                        f"beklenen checkpoint: {baseline.checkpoint}"
                    )
                bright_baselines.load_checkpoint(model, pretrained)
    else:
        model = registry.build(name)
    return model.to(device)


def build_datasets(
    cfg: Config, data_root: Path | str, repo_root: Path | str | None = None
) -> tuple[Split, BrightDataset, BrightDataset]:
    """Build the 13-event datasets, appending the locked adapt pool for few-shot training."""
    split = make_zero_shot_split(*load_standard_lists())
    check_no_leak(split)

    stats = None
    if cfg.norm.sar == "train_stats":

        def load(name: str) -> tuple[np.ndarray, np.ndarray]:
            pre, post, _ = io.read_tile(data_root, name)
            return post, ~io.nodata_mask(pre, post)

        stats = compute_sar_stats(list(split.train), load)

    access = None
    n_primary = len(split.train)
    adapt = []
    if cfg.data.fewshot_split is not None:
        root = Path(repo_root) if repo_root is not None else _git_root()
        access = require_fewshot_access(root, "adapt")
        adapt = sorted(load_split(root / cfg.data.fewshot_split)["adapt"])
        if not adapt or frozenset(adapt) != access.allowed:
            raise TurkeyLockError("Few-shot uyarlama listesi boş veya kilitli izin listesiyle uyuşmuyor.")
        split = dataclasses.replace(split, train=(*split.train, *adapt))

    train_set = BrightDataset(
        data_root,
        split.train,
        mode="train",
        crop=cfg.data.crop,
        flips=cfg.aug.flips,
        rot90=cfg.aug.rot90,
        nodata_ignore=cfg.data.nodata_ignore,
        sar_mode=cfg.norm.sar,
        sar_stats=stats,
        seed=cfg.train.seed,
        sar_p=cfg.aug.sar_p,
        sar_gamma=cfg.aug.sar_gamma,
        sar_gain=cfg.aug.sar_gain,
        access=access,
    )
    if access is not None:
        train_set.n_primary = n_primary
        train_set.n_secondary = len(adapt)
    val_set = BrightDataset(
        data_root,
        split.val,
        mode="eval",
        sar_mode=cfg.norm.sar,
        sar_stats=stats,
    )
    return split, train_set, val_set


def run_training(
    config: Path | str,
    data: Path | str,
    out: Path | str,
    device: str = "cuda",
    pretrained: Path | str | None = None,
    commit: str | None = None,
    steps: int | None = None,
    preflight: Path | str | None = None,
) -> int:
    """Run the single full training and return 0.

    When the YAML leaves ``train.clip`` null, the clipping threshold is read from gate 9 of the
    preflight record (``preflight``, default ``<out>/preflight.json``) with the hash of the YAML
    configuration, before the ``steps`` override. The run folder is named after the configuration
    with the ``steps`` override but the clip still as in the YAML, so a short run lands in the same
    folder as the full one.
    """
    cfg = load_config(config)
    _require_pretrained(cfg, pretrained)
    if cfg.train.clip is None:
        preflight_path = Path(preflight) if preflight is not None else Path(out) / "preflight.json"
        clip = clip_from_preflight(preflight_path, config_hash(cfg))
        print(f"Gradyan kırpma eşiği: {clip:.4f} (9. kapıdan)")
    else:
        clip = cfg.train.clip
    if steps is not None:
        cfg = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, steps=steps))
    if commit is None:
        commit = _git_commit()

    out_dir = Path(out) / run_id(cfg)
    cfg = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, clip=clip))
    _, train_set, val_set = build_datasets(cfg, data)
    model = build_model(cfg, pretrained, device)

    state = train(cfg, model, train_set, val_set, out_dir, device=device, commit=commit)
    print(f"Eğitim bitti: {state.step} adım, en iyi doğrulama mIoU {state.best_miou * 100.0:.2f} {VAL_TAG} ({out_dir})")
    return 0


def _train_batch(train_set: BrightDataset, cfg: Config) -> dict[str, torch.Tensor]:
    """Return one collated train batch, mixing both pools in few-shot runs."""
    batch = cfg.train.batch if getattr(train_set, "n_secondary", 0) else 8
    sampler = _train_sampler(train_set, batch, cfg.train.seed, total_steps=1)
    loader = DataLoader(train_set, batch_size=batch, sampler=sampler, num_workers=0)
    return next(iter(loader))


def _fixed_batch(train_set: BrightDataset, cfg: Config) -> dict[str, torch.Tensor]:
    """Return a collated batch of the fixed samples 0..7 with augmentation off."""
    noaug = BrightDataset(
        train_set.root,
        train_set.names,
        mode="train",
        crop=cfg.data.crop,
        flips=False,
        rot90=False,
        nodata_ignore=cfg.data.nodata_ignore,
        sar_mode=cfg.norm.sar,
        sar_stats=train_set.sar_stats,
        seed=cfg.train.seed,
        access=train_set.access,
    )
    return default_collate([noaug[(index, index)] for index in range(8)])


def _read_log(path: Path) -> list[dict[str, Any]]:
    """Read a JSON-lines training log into a list of dicts."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _short_train(
    cfg: Config,
    model: torch.nn.Module,
    train_set: BrightDataset,
    val_set: BrightDataset,
    device: str,
    steps: int,
) -> list[dict[str, Any]]:
    """Run a short training of ``steps`` steps in a temp dir and return its train log.

    The run is configured one step longer than it runs (``stop_at=steps``), so the loop never reaches
    its final step and skips the full 337-tile validation pass that only costs CU here.
    """
    short = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, steps=steps + 1))
    with tempfile.TemporaryDirectory() as tmp:
        train(short, model, train_set, val_set, tmp, device=device, stop_at=steps)
        return _read_log(Path(tmp) / "train_log.jsonl")


def run_preflight(
    config: Path | str,
    data: Path | str,
    out: Path | str,
    device: str = "cuda",
    pretrained: Path | str | None = None,
    gates: list[int] | None = None,
    repro_dir: Path | str | None = None,
    repo_root: Path | str | None = None,
) -> int:
    """Run the selected pre-training gates and return 0 only if every one passed.

    ``preflight.json`` is rewritten after every gate with ``complete=False``, so an interrupted run
    is never taken as current. A gate that crashes is recorded as failed (with its traceback in the
    log) and the run continues with the next gate.
    """
    cfg = load_config(config)
    _require_pretrained(cfg, pretrained)
    out_dir = Path(out)
    preflight_path = out_dir / "preflight.json"
    cfg_hash = config_hash(cfg)
    from_ana = cfg.model.init == "ana_run"
    fewshot = cfg.data.fewshot_split is not None
    root = Path(repo_root) if repo_root is not None else _git_root() if from_ana else None
    dataset_kwargs = {"repo_root": root} if fewshot else {}

    if gates is None and preflight_is_current(preflight_path, cfg_hash):
        if from_ana:
            checkpoint = Path(pretrained)
            with checkpoint.open("rb") as handle:
                _verify_ana_sha256(handle, checkpoint, cfg.model.init_sha256)
        if fewshot:
            require_fewshot_access(root, "adapt")
        print(f"Ön kontrol güncel ({cfg_hash}): kapılar atlandı.")
        return 0

    selected = list(range(1, 12)) if gates is None else list(gates)
    repro = Path(repro_dir) if repro_dir is not None else DEFAULT_REPRO_DIR

    split, train_set, val_set = build_datasets(cfg, data, **dataset_kwargs)
    results = []
    commit = _git_commit()

    for gate in selected:
        result = None
        try:
            if gate == 1:
                result = gate_reproduction(repro)
            elif gate == 2:
                if fewshot:
                    n_train, n_val, n_test = EXPECTED_SPLIT
                    result = gate_split(
                        split,
                        expected=(n_train + train_set.n_secondary, n_val, n_test),
                        allowed_adapt=train_set.access.allowed,
                    )
                else:
                    result = gate_split(split)
            elif gate == 3:
                n_primary = train_set.n_primary if fewshot else len(train_set)
                mixed = {"n_secondary": train_set.n_secondary} if fewshot else {}
                result = gate_sampler_length(n_primary, cfg.train.batch, cfg.train.steps, seed=cfg.train.seed, **mixed)
            elif gate == 4:
                if from_ana or cfg.model.init == "bright_zero_shot":
                    record_path = root / ANA_VAL13_RECORD if from_ana else repro / "damageformer_val.json"
                    key = "best_miou_pct" if from_ana else "miou_pct"
                    label = "Ana doğrulama kaydı" if from_ana else "Yeniden üretim kaydı"
                    try:
                        record = json.loads(record_path.read_text(encoding="utf-8"))
                        expected = record["values"][key]
                        if from_ana and (type(expected) not in (int, float) or not math.isfinite(expected)):
                            raise ValueError("Geçersiz mIoU değeri.")
                    except (OSError, ValueError, KeyError, TypeError):
                        result = _result(
                            4,
                            False,
                            {"path": str(record_path)},
                            f"{label} eksik veya values.{key} okunamadı: {record_path}",
                        )
                    else:
                        result = gate_loaded_checkpoint(build_model(cfg, pretrained, device), val_set, expected, device)
                else:
                    batch = _train_batch(train_set, cfg)
                    result = gate_initial_loss(build_model(cfg, pretrained, device), batch)
            elif gate == 5:
                batch = _fixed_batch(train_set, cfg)
                result = gate_overfit(build_model(cfg, pretrained, device), batch)
            elif gate == 6:
                batch = _train_batch(train_set, cfg)
                result = gate_sar_connected(build_model(cfg, pretrained, device), batch)
            elif gate == 7:
                histogram = label_histogram(data, split.val)
                counts = np.array([int(histogram["pixels"][str(index)]) for index in range(4)], dtype=np.int64)
                result = gate_metric_edges(counts)
            elif gate == 8:
                log = _short_train(cfg, build_model(cfg, pretrained, device), train_set, val_set, device, 200)
                # Wall time per step (data loading included); the first steps are warm-up (cuDNN
                # autotuning, worker start) and would inflate the projection.
                seconds = [float(entry.get("wall", entry["seconds"])) for entry in log][10:]
                gpu = torch.cuda.get_device_name(0) if str(device).startswith("cuda") else ""
                result = gate_speed(seconds, cfg.train.steps, cu_per_hour(gpu), cfg.train.budget_cu)
            elif gate == 9:
                doubled = dataclasses.replace(cfg, optim=dataclasses.replace(cfg.optim, lr=cfg.optim.lr * 2.0))
                log = _short_train(doubled, build_model(cfg, pretrained, device), train_set, val_set, device, 500)
                result = gate_grad_clip([float(entry["grad_norm"]) for entry in log])
            elif gate == 10:
                # The resume test uses a two-sample batch, so its micro batch must fit it.
                resume_train_kwargs = (
                    {"micro_batch": min(cfg.train.micro_batch, 2)} if cfg.train.micro_batch is not None else {}
                )
                resume_cfg = dataclasses.replace(
                    cfg,
                    data=dataclasses.replace(cfg.data, crop=256),
                    train=dataclasses.replace(cfg.train, steps=80, batch=2, **resume_train_kwargs),
                )
                _, resume_train, resume_val = build_datasets(resume_cfg, data, **dataset_kwargs)
                # Both runs validate once at their final step; eight tiles are enough to compare them.
                resume_val = BrightDataset(
                    data,
                    list(resume_val.names)[:8],
                    mode="eval",
                    sar_mode=resume_cfg.norm.sar,
                    sar_stats=resume_val.sar_stats,
                )
                with tempfile.TemporaryDirectory() as tmp:
                    result = gate_resume(
                        resume_cfg,
                        lambda: build_model(cfg, pretrained, device),
                        resume_train,
                        resume_val,
                        tmp,
                        stop=60,
                        device=device,
                    )
            elif gate == 11:
                result = gate_kernel(lambda: build_model(cfg, pretrained, device), device)
        except Exception as exc:
            traceback.print_exc()
            result = failed_gate(gate, exc)

        if result is None:
            continue
        results.append(result)
        write_preflight(preflight_path, results, cfg_hash, commit, complete=False)
        verdict = "geçti" if result.passed else "kaldı"
        print(f"{result.gate:>4}  {result.name:<24}  {verdict:<6}  {result.detail}", flush=True)
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()

    write_preflight(preflight_path, results, cfg_hash, commit, complete=True)

    print(f"{'Kapı':>4}  {'Ad':<24}  {'Sonuç':<6}  Ayrıntı")
    for result in results:
        verdict = "geçti" if result.passed else "kaldı"
        print(f"{result.gate:>4}  {result.name:<24}  {verdict:<6}  {result.detail}")

    return 0 if all(result.passed for result in results) else 1
