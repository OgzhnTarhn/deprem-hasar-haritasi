"""Resumable training loop for the single full run (design section 7).

Design section 7 fixes one training recipe: AdamW, 40.000 optimizer steps, linear warmup then
cosine decay, bf16 autocast, EMA 0.999, validation every 2.000 steps and a checkpoint every 1.000
steps. This module is that recipe as code, and nothing else.

Validation only on the 13 events
--------------------------------
``val_set`` is the zero-shot validation split of the 13 BRIGHT events; Turkey is never part of it.
The guard at the top of :func:`train` rejects any Turkey tile name in either split before the first
step, so the training code has no path to Turkey (project rule 1).

Selection by EMA val mIoU (weak selection, reported)
----------------------------------------------------
The EMA weights, not the live ones, are evaluated and the best checkpoint is the one with the
highest validation mIoU. That is a *weak* selection: the validation events are the same 13 events
the model is trained on, so the number is optimistic and is reported as ``[VAL13 mIoU, ema]``, never
as the headline ``[ZS-EVENT mIoU]``.

Colab resume semantics
----------------------
A Colab session can die at any step. Every checkpoint holds the model, EMA, optimizer, scheduler,
step, sample counter and every RNG, and is written atomically, so a resumed run continues exactly
where it stopped: the same samples, the same dropout draws, the same losses. A finished run is a
no-op (``DONE`` is read and returned), and a checkpoint from a different configuration is refused
instead of silently mixing two runs.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from hasar.config import Config, config_hash
from hasar.data.sampler import StepSampler
from hasar.eval.metrics import summary
from hasar.eval.predict import evaluate
from hasar.guard import TurkeyLockError, is_turkey_tile
from hasar.results import ResultRecord, write_result
from hasar.train.checkpoint import capture_rng, load_checkpoint, prune_checkpoints, restore_rng, save_checkpoint
from hasar.train.ema import ModelEMA
from hasar.train.losses import damage_loss
from hasar.train.schedule import make_scheduler

#: One JSON line per optimizer step. The log is small (a few hundred bytes per step) and a
#: per-step line makes a disconnect easy to diagnose.
LOG_EVERY = 1

#: Metric tag of the validation number: weak selection on the 13 training events, EMA weights.
VAL_TAG = "[VAL13 mIoU, ema]"


@dataclass(frozen=True)
class TrainState:
    """Outcome of :func:`train`: where the run stopped and how good the EMA got."""

    step: int
    done: bool
    best_step: int
    best_miou: float
    skipped: int = 0
    samples: int = 0


def _check_guard(train_set: Any, val_set: Any) -> None:
    """Raise :class:`TurkeyLockError` if either split holds a Turkey tile name."""
    for split_name, dataset in (("train", train_set), ("val", val_set)):
        for name in dataset.names:
            if is_turkey_tile(str(name)):
                raise TurkeyLockError(
                    f"{split_name} bölmesinde Türkiye karesi var ({name}): eğitim kodunun Türkiye'ye yolu yok."
                )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` as JSON to ``path`` atomically (temporary file, then rename)."""
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _append_jsonl(path: Path, entry: dict[str, Any]) -> None:
    """Append one JSON object as a line to ``path`` and flush it to disk."""
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
        handle.flush()


def _autocast(cfg: Config, device: str) -> Any:
    """Return a bf16 autocast context on CUDA, otherwise a no-op context."""
    if cfg.train.amp == "bf16" and device.startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def train(
    cfg: Config,
    model: torch.nn.Module,
    train_set: Any,
    val_set: Any,
    out_dir: Path | str,
    device: str = "cuda",
    commit: str = "unknown",
    stop_at: int | None = None,
) -> TrainState:
    """Train ``model`` on ``train_set``, validating on ``val_set``, and return the final state.

    ``out_dir`` holds every artefact: ``train_log.jsonl``, ``val_log.jsonl``, ``best.pt``,
    ``last.pt``, ``step_*.pt``, ``val13.json`` and finally ``DONE``. ``stop_at`` stops after the
    checkpoint of that step and returns ``done=False``, which simulates a Colab disconnect.
    """
    _check_guard(train_set, val_set)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_hash = config_hash(cfg)
    total_steps = cfg.train.steps
    batch = cfg.train.batch
    ignore = cfg.loss.ignore_index

    done_path = out / "DONE"
    if done_path.exists():
        done = json.loads(done_path.read_text(encoding="utf-8"))
        return TrainState(
            step=int(done["steps"]),
            done=True,
            best_step=int(done["best_step"]),
            best_miou=float(done["best_miou"]),
            skipped=int(done.get("skipped", 0)),
            samples=int(done.get("samples", 0)),
        )

    model.to(device)
    ema = ModelEMA(model, decay=cfg.train.ema)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    scheduler = make_scheduler(optimizer, cfg.sched, cfg.optim.lr, total_steps)

    step = 0
    samples = 0
    skipped = 0
    best_step = 0
    best_miou = -1.0
    last_miou = 0.0
    last_eval_step = -1
    last_ckpt_step = -1

    last_path = out / "last.pt"
    resumed = last_path.exists()
    if resumed:
        state = load_checkpoint(last_path)
        if state["config_hash"] != run_hash:
            raise RuntimeError(
                f"Kayıt noktası farklı bir yapılandırmaya ait ({state['config_hash']} != {run_hash}): "
                f"aynı çalıştırma sürdürülemez, çıktı klasörünü değiştirin ({out})."
            )
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step = int(state["step"])
        samples = int(state["samples"])
        best_step = int(state["best_step"])
        best_miou = float(state["best_miou"])
        skipped = int(state["skipped"])
        restore_rng(state["rng"])
    else:
        torch.manual_seed(cfg.train.seed)

    sampler = StepSampler(
        len(train_set),
        batch,
        cfg.train.seed,
        total_steps=None,
        start_step=samples // batch,
    )
    # A dedicated generator keeps the loader's base-seed draw out of the global torch RNG, so a
    # resumed run sees exactly the RNG state the checkpoint captured (dropout draws included).
    loader_generator = torch.Generator()
    loader_generator.manual_seed(cfg.train.seed)
    loader = DataLoader(
        train_set,
        batch_size=batch,
        sampler=sampler,
        num_workers=cfg.data.workers,
        drop_last=True,
        pin_memory=device.startswith("cuda"),
        persistent_workers=cfg.data.workers > 0,
        generator=loader_generator,
    )

    train_log = out / "train_log.jsonl"
    val_log = out / "val_log.jsonl"
    model.train()

    def run_eval(at_step: int) -> None:
        nonlocal best_step, best_miou, last_miou, last_eval_step
        accumulator = evaluate(ema.module, val_set, device=device)
        metrics = summary(accumulator.total)
        last_miou = float(metrics["miou"])
        last_eval_step = at_step
        _append_jsonl(val_log, {"step": at_step, **metrics})
        if last_miou > best_miou:
            best_step = at_step
            best_miou = last_miou
            save_checkpoint(
                out / "best.pt",
                {
                    "model": ema.module.state_dict(),
                    "step": at_step,
                    "miou": last_miou,
                    "config_hash": run_hash,
                },
            )

    def save_state(at_step: int) -> None:
        nonlocal last_ckpt_step
        payload = {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": at_step,
            "samples": samples,
            "best_step": best_step,
            "best_miou": best_miou,
            "skipped": skipped,
            "config_hash": run_hash,
            "rng": capture_rng(),
        }
        save_checkpoint(out / f"step_{at_step:06d}.pt", payload)
        save_checkpoint(last_path, payload)
        prune_checkpoints(out, keep=cfg.ckpt.keep_last)
        last_ckpt_step = at_step

    last_tick = time.monotonic()
    for batch_items in loader:
        if step >= total_steps:
            # A run resumed after its last step has nothing left to do.
            break
        pre = batch_items["pre"].to(device, non_blocking=True)
        post = batch_items["post"].to(device, non_blocking=True)
        loc = batch_items["loc"].to(device, non_blocking=True)
        clf = batch_items["clf"].to(device, non_blocking=True)
        samples += batch

        if int((clf != ignore).sum()) == 0:
            # Official behaviour: an all-void batch contributes nothing and costs no optimizer step.
            skipped += 1
            continue

        started = time.monotonic()
        lr = float(optimizer.param_groups[0]["lr"])

        optimizer.zero_grad(set_to_none=True)
        with _autocast(cfg, device):
            outputs = model(pre, post)
        loss, parts = damage_loss(outputs, loc, clf, cfg.loss)
        loss.backward()
        max_norm = cfg.train.clip if cfg.train.clip is not None else float("inf")
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm))
        optimizer.step()
        scheduler.step()
        ema.update(model, step)
        step += 1

        if step % LOG_EVERY == 0:
            _append_jsonl(
                train_log,
                {
                    "step": step,
                    "lr": lr,
                    "loss": parts["total"],
                    "ce_loc": parts["ce_loc"],
                    "ce_clf": parts["ce_clf"],
                    "lovasz_loc": parts["lovasz_loc"],
                    "lovasz_clf": parts["lovasz_clf"],
                    "grad_norm": grad_norm,
                    "samples": samples,
                    "seconds": time.monotonic() - started,
                    # Wall time since the previous logged step, data loading included (gate 8).
                    "wall": time.monotonic() - last_tick,
                },
            )
            last_tick = time.monotonic()

        if step % cfg.eval.every == 0 or step == total_steps:
            run_eval(step)

        if step % cfg.ckpt.every == 0 or step == total_steps or step == stop_at:
            save_state(step)

        if stop_at is not None and step == stop_at:
            return TrainState(
                step=step,
                done=False,
                best_step=best_step,
                best_miou=best_miou,
                skipped=skipped,
                samples=samples,
            )

        if step >= total_steps:
            break

    # A run resumed after its last step never enters the loop body; still close it out.
    if last_eval_step != total_steps:
        run_eval(total_steps)
    if last_ckpt_step != total_steps:
        save_state(total_steps)

    write_result(
        out / "val13.json",
        ResultRecord(
            metric_tag=VAL_TAG,
            model=type(model).__name__,
            split="zero_shot/val",
            commit=commit,
            config_hash=run_hash,
            values={
                "best_miou_pct": best_miou * 100.0,
                "last_miou_pct": last_miou * 100.0,
            },
            extra={"best_step": best_step, "steps": step, "skipped": skipped},
        ),
    )

    _write_json_atomic(
        done_path,
        {
            "steps": step,
            "samples": samples,
            "best_step": best_step,
            "best_miou": best_miou,
            "skipped": skipped,
            "config_hash": run_hash,
        },
    )

    return TrainState(
        step=step,
        done=True,
        best_step=best_step,
        best_miou=best_miou,
        skipped=skipped,
        samples=samples,
    )
