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
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from hasar.config import Config, config_hash, load_config, run_id
from hasar.data import io
from hasar.data.dataset import BrightDataset
from hasar.data.events import check_no_leak, load_standard_lists, make_zero_shot_split
from hasar.data.normalize import compute_sar_stats
from hasar.data.sampler import StepSampler
from hasar.data.summary import label_histogram
from hasar.models import registry
from hasar.preflight.gates import (
    gate_grad_clip,
    gate_initial_loss,
    gate_kernel,
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
from hasar.train.loop import VAL_TAG, train

#: Default folder of the Phase 3 reproduction records (gate 1).
DEFAULT_REPRO_DIR = Path("results") / "yeniden-uretim"

#: Approximate Colab CU price per hour by GPU (gate 8); an unknown GPU uses the A100 price.
CU_PER_HOUR = {"A100": 13.0, "L4": 5.0}


def cu_per_hour(device_name: str) -> float:
    """Return the CU price per hour for a GPU name such as ``"NVIDIA L4"`` (A100 price if unknown)."""
    for key, price in CU_PER_HOUR.items():
        if key in device_name:
            return price
    return CU_PER_HOUR["A100"]


#: CU budget of the single full training (gate 8).
CU_BUDGET = 75.0


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
    else:
        model = registry.build(name)
    return model.to(device)


def build_datasets(cfg: Config, data_root: Path | str) -> tuple[Any, BrightDataset, BrightDataset]:
    """Build the zero-shot split and its train and val datasets."""
    split = make_zero_shot_split(*load_standard_lists())
    check_no_leak(split)

    stats = None
    if cfg.norm.sar == "train_stats":

        def load(name: str) -> tuple[np.ndarray, np.ndarray]:
            pre, post, _ = io.read_tile(data_root, name)
            return post, ~io.nodata_mask(pre, post)

        stats = compute_sar_stats(list(split.train), load)

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
    )
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
) -> int:
    """Run the single full training and return 0."""
    cfg = load_config(config)
    if steps is not None:
        cfg = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, steps=steps))
    if commit is None:
        commit = _git_commit()

    out_dir = Path(out) / run_id(cfg)
    _, train_set, val_set = build_datasets(cfg, data)
    model = build_model(cfg, pretrained, device)

    state = train(cfg, model, train_set, val_set, out_dir, device=device, commit=commit)
    print(f"Eğitim bitti: {state.step} adım, en iyi doğrulama mIoU {state.best_miou * 100.0:.2f} {VAL_TAG} ({out_dir})")
    return 0


def _train_batch(train_set: BrightDataset, cfg: Config) -> dict[str, torch.Tensor]:
    """Return one collated batch of eight train samples from a DataLoader."""
    sampler = StepSampler(len(train_set), 8, cfg.train.seed, total_steps=1)
    loader = DataLoader(train_set, batch_size=8, sampler=sampler, num_workers=0)
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
) -> int:
    """Run the selected pre-training gates and return 0 only if every one passed."""
    cfg = load_config(config)
    out_dir = Path(out)
    preflight_path = out_dir / "preflight.json"
    cfg_hash = config_hash(cfg)

    if gates is None and preflight_is_current(preflight_path, cfg_hash):
        print(f"Ön kontrol güncel ({cfg_hash}): kapılar atlandı.")
        return 0

    selected = list(range(1, 12)) if gates is None else list(gates)
    repro = Path(repro_dir) if repro_dir is not None else DEFAULT_REPRO_DIR

    split, train_set, val_set = build_datasets(cfg, data)
    results = []

    for gate in selected:
        if gate == 1:
            results.append(gate_reproduction(repro))
        elif gate == 2:
            results.append(gate_split(split))
        elif gate == 3:
            results.append(gate_sampler_length(len(train_set), cfg.train.batch, cfg.train.steps, seed=cfg.train.seed))
        elif gate == 4:
            batch = _train_batch(train_set, cfg)
            results.append(gate_initial_loss(build_model(cfg, pretrained, device), batch))
        elif gate == 5:
            batch = _fixed_batch(train_set, cfg)
            results.append(gate_overfit(build_model(cfg, pretrained, device), batch))
        elif gate == 6:
            batch = _train_batch(train_set, cfg)
            results.append(gate_sar_connected(build_model(cfg, pretrained, device), batch))
        elif gate == 7:
            histogram = label_histogram(data, split.val)
            counts = np.array([int(histogram["pixels"][str(index)]) for index in range(4)], dtype=np.int64)
            results.append(gate_metric_edges(counts))
        elif gate == 8:
            log = _short_train(cfg, build_model(cfg, pretrained, device), train_set, val_set, device, 200)
            # Wall time per step (data loading included); the first steps are warm-up (cuDNN
            # autotuning, worker start) and would inflate the projection.
            seconds = [float(entry.get("wall", entry["seconds"])) for entry in log][10:]
            gpu = torch.cuda.get_device_name(0) if str(device).startswith("cuda") else ""
            results.append(gate_speed(seconds, cfg.train.steps, cu_per_hour(gpu), CU_BUDGET))
        elif gate == 9:
            doubled = dataclasses.replace(cfg, optim=dataclasses.replace(cfg.optim, lr=cfg.optim.lr * 2.0))
            log = _short_train(doubled, build_model(cfg, pretrained, device), train_set, val_set, device, 500)
            results.append(gate_grad_clip([float(entry["grad_norm"]) for entry in log]))
        elif gate == 10:
            resume_cfg = dataclasses.replace(
                cfg,
                data=dataclasses.replace(cfg.data, crop=256),
                train=dataclasses.replace(cfg.train, steps=80, batch=2),
            )
            _, resume_train, resume_val = build_datasets(resume_cfg, data)
            # Both runs validate once at their final step; eight tiles are enough to compare them.
            resume_val = BrightDataset(
                data,
                list(resume_val.names)[:8],
                mode="eval",
                sar_mode=resume_cfg.norm.sar,
                sar_stats=resume_val.sar_stats,
            )
            with tempfile.TemporaryDirectory() as tmp:
                results.append(
                    gate_resume(
                        resume_cfg,
                        lambda: build_model(cfg, pretrained, device),
                        resume_train,
                        resume_val,
                        tmp,
                        stop=60,
                        device=device,
                    )
                )
        elif gate == 11:
            results.append(gate_kernel(lambda: build_model(cfg, pretrained, device), device))

    write_preflight(preflight_path, results, cfg_hash, _git_commit())

    print(f"{'Kapı':>4}  {'Ad':<24}  {'Sonuç':<6}  Ayrıntı")
    for result in results:
        verdict = "geçti" if result.passed else "kaldı"
        print(f"{result.gate:>4}  {result.name:<24}  {verdict:<6}  {result.detail}")

    return 0 if all(result.passed for result in results) else 1
