"""Glue between the CLI and the single Turkey evaluation (``final`` and ``sicaklik``).

This module holds no logic of its own: it loads the configuration, finds the best checkpoint of
the single full training, builds the model, and calls :func:`hasar.eval.final.run_final` or the
temperature fit of :mod:`hasar.eval.calibration`. Everything it needs already lives in those
modules.

The temperature is chosen on the 13-event validation split only (protocol section 10, rule 1) and
merely measured on Turkey, so :func:`fit_temperature_cli` never touches Turkey imagery.

torch is imported at module top: this module is only imported where torch exists.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from hasar.config import config_hash, load_config, run_id
from hasar.eval.bootstrap import turkey_blocks
from hasar.eval.calibration import NllAccumulator
from hasar.eval.final import run_final
from hasar.eval.metrics import ConfusionAccumulator, miou
from hasar.eval.predict import predict
from hasar.guard import require_final_access
from hasar.results import ResultRecord, read_result, write_result
from hasar.train.checkpoint import load_checkpoint
from hasar.train.run import _git_commit, build_datasets, build_model

#: The temperature record written by ``sicaklik`` and read by ``final``.
TEMPERATURE_FILE = "sicaklik.json"
#: The metric tag of the temperature record.
TEMPERATURE_TAG = "[VAL13 mIoU, sicaklik]"
#: Default cache folder on Colab; the local fallback is used when its parent is missing.
COLAB_CACHE = Path("/content/data/final_cache")
LOCAL_CACHE = Path("data/final_cache")


def _best_checkpoint(runs: Path | str, cfg) -> Path:
    """Return ``runs / run_id(cfg) / 'best.pt'``; require the sibling ``DONE`` file.

    ``DONE`` is written only when the single full training finished, so a missing file means the
    checkpoint cannot be trusted for the locked Turkey evaluation.
    """
    run_dir = Path(runs) / run_id(cfg)
    if not (run_dir / "DONE").is_file():
        raise RuntimeError(f"eğitim bitmedi: {run_dir / 'DONE'} yok, Türkiye değerlendirmesi başlatılmadı.")
    return run_dir / "best.pt"


def model_label(cfg, model: torch.nn.Module) -> str:
    """Return the name of the model that really runs: the fallback name when it took over.

    :func:`hasar.train.run.build_model` silently switches to ``cfg.model.fallback`` when the
    ChangeMamba CUDA kernel is missing, so the configured name alone could mislabel a record.
    """
    if cfg.model.name == "changemamba_mmbda_tiny" and type(model).__name__ != "ChangeMambaWrapper":
        return cfg.model.fallback
    return cfg.model.name


def _load_model(cfg, runs: Path | str, device: str) -> torch.nn.Module:
    """Build the configured model and load the EMA weights of the best checkpoint."""
    model = build_model(cfg, None, device)
    state = load_checkpoint(_best_checkpoint(runs, cfg))
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()
    return model


def fit_temperature_cli(
    config: Path | str,
    data: Path | str,
    runs: Path | str,
    device: str = "cuda",
    out: Path | str = Path("results") / "ana",
) -> int:
    """Fit the single ``clf`` temperature on the 13-event validation split and write it.

    The model runs once over the validation tiles; the same pass feeds the negative log-likelihood
    grid of :class:`hasar.eval.calibration.NllAccumulator` and the confusion matrix whose mIoU is
    written next to the chosen temperature.
    """
    cfg = load_config(config)
    model = _load_model(cfg, runs, device)
    _, _, val_set = build_datasets(cfg, data)

    nll = NllAccumulator()
    accumulator = ConfusionAccumulator()
    for item in val_set:
        name = str(item["name"])
        label = np.asarray(item["clf"])
        pre = torch.from_numpy(np.asarray(item["pre"])).unsqueeze(0).to(device)
        post = torch.from_numpy(np.asarray(item["post"])).unsqueeze(0).to(device)
        clf_prob, _loc_prob = predict(model, pre, post, tta=False)
        clf = clf_prob[0].float().cpu().numpy()

        nll.add(clf.reshape(clf.shape[0], -1), label.reshape(-1))
        accumulator.add(label, clf.argmax(axis=0), name)

    temperature = nll.best()
    miou_pct = miou(accumulator.total) * 100.0

    write_result(
        Path(out) / TEMPERATURE_FILE,
        ResultRecord(
            metric_tag=TEMPERATURE_TAG,
            values={"temperature": temperature, "miou_pct": miou_pct},
            model=model_label(cfg, model),
            split="zero_shot/val",
            commit=_git_commit(),
            config_hash=config_hash(cfg),
        ),
    )
    print(f"Sıcaklık {temperature:.2f}, doğrulama mIoU {miou_pct:.2f} {TEMPERATURE_TAG}")
    return 0


def _read_temperature(out_dir: Path) -> float | None:
    """Return the temperature of ``out_dir/sicaklik.json``, or ``None`` with a warning."""
    path = out_dir / TEMPERATURE_FILE
    if not path.is_file():
        print(f"Uyarı: sıcaklık dosyası yok ({path}); Türkiye değerlendirmesi T=1 ile yapılıyor.")
        return None
    return float(read_result(path).values["temperature"])


def _resolve_cache(cache: Path | str | None, cfg) -> Path:
    """Return the per-model cache folder ``<base>/<run_id(cfg)>``.

    ``base`` is the given folder, or the Colab default when its parent exists. The run id
    subfolder keeps the main model's and the variant's caches apart: :func:`run_final` reuses any
    cached tile, so a shared folder would silently hand one model's probabilities to the other.
    """
    base = Path(cache) if cache is not None else (COLAB_CACHE if COLAB_CACHE.parent.is_dir() else LOCAL_CACHE)
    return base / run_id(cfg)


def run_final_cli(
    config: Path | str,
    data: Path | str,
    runs: Path | str,
    device: str = "cuda",
    out: Path | str = Path("results") / "ana",
    cache: Path | str | None = None,
    instances: Path | str | None = None,
    blocks: Path | str | None = Path("results") / "arastirma" / "turkiye_bloklar.json",
) -> int:
    """Run the single Turkey evaluation and print the headline with its metric tag."""
    cfg = load_config(config)
    out_dir = Path(out)
    model = _load_model(cfg, runs, device)

    temperature = _read_temperature(out_dir)

    block_map = None
    if blocks is not None and Path(blocks).is_file():
        block_map = turkey_blocks(blocks)

    records = run_final(
        model,
        data,
        out_dir,
        _resolve_cache(cache, cfg),
        model_name=model_label(cfg, model),
        config_hash=config_hash(cfg),
        temperature=temperature,
        blocks=block_map,
        instances_dir=instances,
        device=device,
        access_fn=lambda repo_root: require_final_access(repo_root),
    )
    headline = records["headline"]
    print(f"Türkiye {headline.values['miou_pct']:.2f} {headline.metric_tag}")
    return 0
