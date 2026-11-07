"""Pre-training gates that must pass before any GPU time is spent (design section 8).

The eleven gates of design section 8 are implemented here, one function each, plus the
``preflight.json`` helpers that record their outcome:

1. ``gate_reproduction`` -- the published SiamCRNN, UNet and DamageFormer checkpoints reproduce
   their Turkey ``[ZS-EVENT mIoU, repro]`` values within tolerance.
2. ``gate_split`` -- the tile counts and event lists match the expected split and no Turkey tile
   leaks into train or validation.
3. ``gate_sampler_length`` -- the loader length equals ``steps * batch`` (the ``max_iters`` trap).
4. ``gate_initial_loss`` -- the initial losses are close to ln 4 (damage head) and ln 2 (building
   head).
5. ``gate_overfit`` -- a fixed batch of eight tiles is memorised with augmentation off.
6. ``gate_sar_connected`` -- zeroing the SAR input changes the damage output.
7. ``gate_metric_edges`` -- a perfect prediction scores 1.0 and an always-background prediction
   scores the background fraction / 4.
8. ``gate_speed`` -- the measured step time projects to a CU cost within budget.
9. ``gate_grad_clip`` -- the p90 gradient norm is finite and becomes the clipping threshold.
10. ``gate_resume`` -- a run killed mid-training resumes to exactly the uninterrupted result.
11. ``gate_kernel`` -- the ChangeMamba selective-scan kernel compiles and survives a short
    forward-backward test.

Gates 1, 8, 9 and 11 need Colab (an A100, the downloaded data and the Zenodo results); the rest
also run on CPU with tiny fakes, which is how the test suite pins them.

A gate never raises for a failed check: it returns ``passed=False`` with a Turkish detail. Only a
programming error (a wrong argument type, a broken import) raises. Every value stored in a
``GateResult`` is a plain Python type so that ``preflight.json`` is valid JSON.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hasar.config import Config
from hasar.data.events import TEST_EVENT, Split, SplitLeakError, check_no_leak
from hasar.data.sampler import StepSampler
from hasar.eval.metrics import always_background_miou, confusion_matrix, miou
from hasar.train.loop import train
from hasar.train.losses import damage_loss

#: Metric tag of a Phase 3 reproduction record on the Turkey test split.
REPRO_TAG = "[ZS-EVENT mIoU, repro]"

#: The three published checkpoints gate 1 requires (design section 8, gate 1).
REQUIRED_REPRO: tuple[str, ...] = ("bright_siamcrnn", "bright_unet", "bright_damageformer")

#: Expected zero-shot split sizes: train / val / test (design section 6).
EXPECTED_SPLIT: tuple[int, int, int] = (2795, 337, 1114)

#: Short Turkish labels of the eleven gates, indexed by gate number minus one.
GATE_NAMES: tuple[str, ...] = (
    "Yeniden üretim",
    "Bölme sayıları",
    "Örnekleyici uzunluğu",
    "Başlangıç kaybı",
    "Ezberleme",
    "SAR dalı bağlı",
    "Metrik uç durumları",
    "Hız ve CU projeksiyonu",
    "Gradyan kırpma eşiği",
    "Devam testi",
    "Çekirdek",
)


@dataclass(frozen=True)
class GateResult:
    """Outcome of one gate: its number, label, verdict, measured values and Turkish detail."""

    gate: int
    name: str
    passed: bool
    value: dict[str, Any] = field(default_factory=dict)
    detail: str = ""


def _result(gate: int, passed: bool, value: dict[str, Any], detail: str) -> GateResult:
    """Build a :class:`GateResult` with the gate's canonical Turkish label."""
    return GateResult(gate=gate, name=GATE_NAMES[gate - 1], passed=passed, value=value, detail=detail)


# --- gate 1: reproduction ---------------------------------------------------


def gate_reproduction(results_dir: Path | str) -> GateResult:
    """Check that the three required published checkpoints reproduce within tolerance.

    Every ``*.json`` under ``results_dir`` whose ``metric_tag`` is ``[ZS-EVENT mIoU, repro]`` is
    read; validation records (``[VAL13 mIoU, repro]``) are ignored. A required model that is
    missing or out of tolerance fails the gate. An extra model that is out of tolerance (for
    example a documented negative finding) is reported but does not fail the gate.
    """
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(Path(results_dir).rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or payload.get("metric_tag") != REPRO_TAG:
            continue
        model = payload.get("model")
        if isinstance(model, str):
            records[model] = payload

    def within(model: str) -> bool:
        extra = records[model].get("extra")
        return isinstance(extra, dict) and extra.get("within_tolerance") is True

    missing = [model for model in REQUIRED_REPRO if model not in records]
    out_of_tolerance = [model for model in REQUIRED_REPRO if model in records and not within(model)]
    extras = sorted(model for model in records if model not in REQUIRED_REPRO)
    failing_extras = [model for model in extras if not within(model)]

    value: dict[str, Any] = {
        "models": len(records),
        "required": list(REQUIRED_REPRO),
        "missing": missing,
        "out_of_tolerance": out_of_tolerance,
        "extra": extras,
    }

    if missing or out_of_tolerance:
        parts = []
        if missing:
            parts.append(f"eksik kayıt: {', '.join(missing)}")
        if out_of_tolerance:
            parts.append(f"tolerans dışı: {', '.join(out_of_tolerance)}")
        return _result(1, False, value, "Yeniden üretim başarısız (" + "; ".join(parts) + ").")

    detail = f"{len(REQUIRED_REPRO)} zorunlu model tolerans içinde."
    if failing_extras:
        detail += f" Tolerans dışı ek kayıt (belgelenmiş): {', '.join(failing_extras)}."
    return _result(1, True, value, detail)


# --- gate 2: split ----------------------------------------------------------


def gate_split(
    split: Split,
    expected: tuple[int, int, int] = EXPECTED_SPLIT,
    test_event: str = TEST_EVENT,
) -> GateResult:
    """Check the split sizes and that no test-event tile leaks into train or validation."""
    counts = split.counts()
    actual = (counts["train"], counts["val"], counts["test"])
    value: dict[str, Any] = {"counts": counts, "expected": list(expected)}

    problems: list[str] = []
    if actual != tuple(expected):
        problems.append(f"sayılar {actual}, beklenen {tuple(expected)}")
    try:
        check_no_leak(split, test_event=test_event)
    except SplitLeakError as exc:
        problems.append(str(exc))

    if problems:
        return _result(2, False, value, "Bölme sayıları hatalı: " + "; ".join(problems))
    return _result(2, True, value, f"Bölme sayıları beklenenle aynı: {actual}.")


# --- gate 3: sampler length -------------------------------------------------


def gate_sampler_length(n_items: int, batch: int, steps: int, seed: int = 0) -> GateResult:
    """Check that the step-bounded sampler yields exactly ``steps * batch`` samples."""
    sampler = StepSampler(n_items, batch, seed, total_steps=steps)
    length = len(sampler)
    expected = steps * batch
    value: dict[str, Any] = {"length": int(length), "expected": int(expected), "steps": int(steps)}
    if length != expected:
        return _result(3, False, value, f"Örnekleyici uzunluğu {length}, beklenen {expected} (max_iters tuzağı).")
    return _result(3, True, value, f"Örnekleyici uzunluğu {length} = {steps} adım × {batch}.")


# --- gate 4: initial loss ---------------------------------------------------


def _on_model_device(model: torch.nn.Module, batch: dict[str, Any]) -> dict[str, Any]:
    """Return ``batch`` with every tensor moved to the device of ``model``'s parameters."""
    device = next(model.parameters()).device
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _batch_loss(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    """Return the total loss tensor and its float parts for one fixed batch."""
    batch = _on_model_device(model, batch)
    outputs = model(batch["pre"], batch["post"])
    return damage_loss(outputs, batch["loc"], batch["clf"])


def gate_initial_loss(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    tolerance: float = 0.25,
) -> GateResult:
    """Check that the initial losses are close to ln 4 (damage) and ln 2 (building)."""
    model.eval()
    with torch.no_grad():
        _, parts = _batch_loss(model, batch)

    expected = {"ce_clf": math.log(4.0), "ce_loc": math.log(2.0)}
    diffs = {name: abs(parts[name] - target) for name, target in expected.items()}
    value: dict[str, Any] = {
        "ce_clf": float(parts["ce_clf"]),
        "ce_loc": float(parts["ce_loc"]),
        "expected_ce_clf": float(expected["ce_clf"]),
        "expected_ce_loc": float(expected["ce_loc"]),
        "tolerance": float(tolerance),
    }
    bad = [name for name, diff in diffs.items() if diff > tolerance]
    if bad:
        detail = ", ".join(f"{name}={parts[name]:.4f} (beklenen {expected[name]:.4f})" for name in bad)
        return _result(4, False, value, f"Başlangıç kaybı beklenenden uzak: {detail}.")
    return _result(4, True, value, "Başlangıç kayıpları ln 4 ve ln 2'ye yakın.")


# --- gate 5: overfit --------------------------------------------------------


def gate_overfit(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    steps: int = 300,
    lr: float = 1e-3,
    ratio: float = 0.2,
) -> GateResult:
    """Check that a fixed batch is memorised: the final loss falls below ``ratio`` of the initial."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.eval()
    with torch.no_grad():
        _, initial_parts = _batch_loss(model, batch)
    initial = initial_parts["total"]

    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = _batch_loss(model, batch)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        _, final_parts = _batch_loss(model, batch)
    final = final_parts["total"]

    achieved = final / initial if initial > 0 else float("inf")
    value: dict[str, Any] = {
        "initial": float(initial),
        "final": float(final),
        "ratio": float(achieved),
        "threshold": float(ratio),
        "steps": int(steps),
    }
    if achieved < ratio:
        return _result(5, True, value, f"Sabit parti ezberlendi: kayıp oranı {achieved:.4f} < {ratio}.")
    return _result(5, False, value, f"Sabit parti ezberlenmedi: kayıp oranı {achieved:.4f} ≥ {ratio}.")


# --- gate 6: SAR branch connected -------------------------------------------


def gate_sar_connected(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> GateResult:
    """Check that zeroing the SAR input changes the damage output (the SAR branch is connected)."""
    batch = _on_model_device(model, batch)
    model.eval()
    with torch.no_grad():
        base = model(batch["pre"], batch["post"])["clf"]
        zeroed = model(batch["pre"], torch.zeros_like(batch["post"]))["clf"]

    difference = float((base - zeroed).abs().max())
    value: dict[str, Any] = {"max_diff": difference}
    if difference > 1e-6:
        return _result(6, True, value, f"SAR dalı bağlı: en büyük çıktı farkı {difference:.6f}.")
    return _result(6, False, value, "SAR dalı bağlı değil: SAR sıfırlanınca hasar çıktısı değişmiyor.")


# --- gate 7: metric edge cases ----------------------------------------------


def gate_metric_edges(class_counts: np.ndarray, tolerance: float = 1e-6) -> GateResult:
    """Check the metric on a perfect prediction and on an always-background prediction.

    ``class_counts`` holds the pixel count of each of the four classes. A perfect prediction must
    score 1.0; an always-background prediction must score ``background_fraction / 4``.
    """
    counts = np.asarray(class_counts, dtype=np.int64)
    labels = np.repeat(np.arange(counts.size, dtype=np.int64), counts)

    perfect = miou(confusion_matrix(labels, labels))
    always_background = miou(confusion_matrix(labels, np.zeros_like(labels)))
    expected_background = always_background_miou(float(counts[0]) / float(counts.sum()))

    value: dict[str, Any] = {
        "perfect": float(perfect),
        "always_background": float(always_background),
        "expected_background": float(expected_background),
    }
    problems = []
    if abs(perfect - 1.0) > tolerance:
        problems.append(f"kusursuz tahmin {perfect:.6f} ≠ 1,0")
    if abs(always_background - expected_background) > tolerance:
        problems.append(f"hep arka plan {always_background:.6f} ≠ {expected_background:.6f}")
    if problems:
        return _result(7, False, value, "Metrik uç durumları hatalı: " + "; ".join(problems))
    return _result(7, True, value, "Metrik uç durumları doğru (kusursuz 1,0; hep arka plan oran/4).")


# --- gate 8: speed projection -----------------------------------------------


def gate_speed(
    step_seconds: list[float] | tuple[float, ...],
    steps: int,
    cu_per_hour: float,
    budget_cu: float,
) -> GateResult:
    """Project the CU cost of ``steps`` from the measured step times and compare it with budget."""
    seconds = float(np.mean(np.asarray(step_seconds, dtype=np.float64)))
    projected = seconds * steps / 3600.0 * cu_per_hour
    value: dict[str, Any] = {
        "seconds_per_step": seconds,
        "projected_cu": float(projected),
        "budget_cu": float(budget_cu),
    }
    if projected <= budget_cu:
        return _result(8, True, value, f"CU projeksiyonu {projected:.2f} ≤ bütçe {budget_cu:.2f}.")
    return _result(8, False, value, f"CU projeksiyonu {projected:.2f} > bütçe {budget_cu:.2f}: eğitim başlamaz.")


# --- gate 9: gradient clipping threshold ------------------------------------


def gate_grad_clip(norms: list[float] | tuple[float, ...], quantile: float = 0.9) -> GateResult:
    """Check that the gradient norms are finite and return their ``quantile`` as the clip value."""
    values = np.asarray(norms, dtype=np.float64)
    finite = bool(np.all(np.isfinite(values)))
    clip = float(np.quantile(values, quantile)) if finite else None
    value: dict[str, Any] = {
        "clip": clip,
        "quantile": float(quantile),
        "norms": int(values.size),
    }
    if not finite:
        return _result(9, False, value, "Gradyan normları arasında sonlu olmayan değer var.")
    if clip is None or clip <= 0:
        return _result(9, False, value, f"Kırpma eşiği pozitif değil: {clip}.")
    return _result(9, True, value, f"Kırpma eşiği p{quantile * 100:.0f} = {clip:.4f}.")


# --- gate 10: resume --------------------------------------------------------


def gate_resume(
    cfg: Config,
    model_fn: Any,
    train_set: Any,
    val_set: Any,
    out_dir: Path | str,
    stop: int = 60,
    device: str = "cpu",
    atol: float | None = None,
) -> GateResult:
    """Check that a run stopped at ``stop`` and resumed matches the uninterrupted run.

    On CPU the comparison is exact (``torch.equal``). On CUDA some backward kernels (atomic adds in
    upsampling and the selective scan) are not bitwise deterministic, so two *uninterrupted* runs
    already differ (measured locally: parameters within 1e-4, BatchNorm running statistics up to a
    few percent, losses within ~1e-3). There the parameters are compared with ``atol`` (default
    :data:`CUDA_RESUME_ATOL`, ``rtol`` 1e-3) and the post-resume losses with :data:`LOSS_RTOL`; BatchNorm buffers and
    the validation mIoU that depends on them are only reported. A resume bug (wrong step, sample offset,
    optimizer or RNG state) moves the weights and the loss curve far more than that.
    """
    if atol is None and str(device).startswith("cuda"):
        atol = CUDA_RESUME_ATOL
    out = Path(out_dir)
    interrupted_dir = out / "interrupted"
    reference_dir = out / "reference"

    def fresh_model() -> torch.nn.Module:
        # Same initial weights for every run: model_fn may build randomly initialised heads.
        torch.manual_seed(cfg.train.seed)
        return model_fn()

    first = train(cfg, fresh_model(), train_set, val_set, interrupted_dir, device=device, stop_at=stop)
    if first.done:
        value: dict[str, Any] = {"stop": int(stop), "steps": int(cfg.train.steps)}
        return _result(10, False, value, f"Devam testi kurulamadı: durma adımı {stop} toplam adımdan küçük değil.")

    resumed = train(cfg, fresh_model(), train_set, val_set, interrupted_dir, device=device)
    reference = train(cfg, fresh_model(), train_set, val_set, reference_dir, device=device)

    resumed_state = torch.load(interrupted_dir / "last.pt", map_location="cpu", weights_only=False)
    reference_state = torch.load(reference_dir / "last.pt", map_location="cpu", weights_only=False)

    buffer_names = {name for name, _ in fresh_model().named_buffers()} if atol is not None else set()

    def _same(name: str, a: torch.Tensor, b: torch.Tensor) -> bool:
        if atol is None or not a.is_floating_point():
            return torch.equal(a, b)
        return torch.allclose(a, b, atol=atol, rtol=1e-3)

    # On CUDA the BatchNorm running statistics (momentum 0.1, so mostly the last few batches)
    # amplify the GPU noise: two uninterrupted runs already differ there by up to tens of percent.
    # They are reported but do not decide; the parameters and the loss curve do.
    mismatched = [
        name
        for name, tensor in reference_state["model"].items()
        if name not in buffer_names and not _same(name, tensor, resumed_state["model"][name])
    ]
    buffer_diff = max(
        (
            float((reference_state["model"][name].float() - resumed_state["model"][name].float()).abs().max())
            for name in buffer_names
            if name in reference_state["model"] and reference_state["model"][name].is_floating_point()
        ),
        default=0.0,
    )
    losses_ok = True
    if atol is not None:
        tail = int(cfg.train.steps) - int(stop)
        resumed_losses = _log_losses(interrupted_dir / "train_log.jsonl")[-tail:]
        reference_losses = _log_losses(reference_dir / "train_log.jsonl")[-tail:]
        losses_ok = len(resumed_losses) == len(reference_losses) and all(
            math.isclose(a, b, rel_tol=LOSS_RTOL, abs_tol=1e-4)
            for a, b in zip(resumed_losses, reference_losses, strict=True)
        )
    max_diff = max(
        (
            float((tensor.float() - resumed_state["model"][name].float()).abs().max())
            for name, tensor in reference_state["model"].items()
            if tensor.is_floating_point() and tensor.numel() and name not in buffer_names
        ),
        default=0.0,
    )
    same_step = int(resumed.step) == int(reference.step)
    # On CUDA the validation mIoU depends on the BatchNorm buffers above, so it is reported only.
    same_miou = atol is not None or math.isclose(
        float(resumed.best_miou), float(reference.best_miou), rel_tol=0.0, abs_tol=1e-9
    )

    value = {
        "stop": int(stop),
        "resumed_step": int(resumed.step),
        "reference_step": int(reference.step),
        "mismatched_tensors": len(mismatched),
        "max_abs_diff": max_diff,
        "buffer_max_abs_diff": buffer_diff,
        "mismatched_names": mismatched[:5],
        "resumed_best_miou": float(resumed.best_miou),
        "reference_best_miou": float(reference.best_miou),
        "atol": atol,
    }
    value["losses_match"] = losses_ok
    if mismatched or not same_step or not same_miou or not losses_ok:
        detail = (
            f"Devam testi başarısız: {len(mismatched)} tensör farklı (en büyük fark {max_diff:.2e}), "
            f"adım {resumed.step}/{reference.step}."
        )
        return _result(10, False, value, detail)
    how = "birebir aynı" if atol is None else f"atol {atol} içinde aynı"
    return _result(10, True, value, f"Devam testi geçti: {resumed.step} adım {how}.")


#: Relative tolerance of the post-resume losses on CUDA (gate 10).
LOSS_RTOL = 1e-2

#: Absolute parameter tolerance on CUDA (gate 10). Measured noise floor between two uninterrupted
#: DamageFormer runs on an RTX 4060 after 80 steps: ~1.3e-4; this leaves headroom for the selective
#: scan kernel on an A100.
CUDA_RESUME_ATOL = 1e-3


def _log_losses(path: Path) -> list[float]:
    """Return the ``loss`` column of a JSON-lines training log."""
    if not path.exists():
        return []
    return [float(json.loads(line)["loss"]) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# --- gate 11: kernel --------------------------------------------------------


def gate_kernel(model_fn: Any, device: str = "cuda", steps: int = 20) -> GateResult:
    """Check that the model's custom kernel compiles and survives a short forward-backward test."""
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        value: dict[str, Any] = {"device": str(device)}
        return _result(11, False, value, f"CUDA yok: çekirdek kapısı yalnız cuda cihazında çalışır ({device}).")

    try:
        model = model_fn().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        pre = torch.rand(2, 3, 64, 64, device=device)
        post = torch.rand(2, 3, 64, 64, device=device)
        loc = torch.zeros(2, 64, 64, dtype=torch.long, device=device)
        clf = torch.zeros(2, 64, 64, dtype=torch.long, device=device)

        backend = _scan_backend_of(model)
        equivalence = _kernel_equivalence(model, device) if backend == "cuda" else None

        losses: list[float] = []
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            outputs = model(pre, post)
            loss, parts = damage_loss(outputs, loc, clf)
            loss.backward()
            optimizer.step()
            losses.append(parts["total"])
    except Exception as exc:  # noqa: BLE001 - a failed kernel compile is a gate failure, not a crash
        value = {"device": str(device), "steps": int(steps)}
        return _result(11, False, value, f"Çekirdek derlenemedi veya ileri-geri testi düştü: {exc}")

    finite = all(math.isfinite(loss) for loss in losses)
    value = {
        "device": str(device),
        "steps": int(steps),
        "final_loss": float(losses[-1]),
        "model": type(model).__name__,
        "scan_backend": backend,
        "max_abs_diff_vs_torch": equivalence,
    }
    if not finite:
        return _result(11, False, value, "Çekirdek testinde sonlu olmayan kayıp üretildi.")
    if equivalence is not None and not equivalence < KERNEL_TOLERANCE:
        detail = f"CUDA çekirdeği torch yoluyla eşdeğer değil: en büyük fark {equivalence:.2e} >= {KERNEL_TOLERANCE}."
        return _result(11, False, value, detail)
    return _result(11, True, value, f"{type(model).__name__} ({backend}) {steps} adımlık ileri-geri testini geçti.")


#: Largest allowed |clf_cuda - clf_torch| for one forward pass (gate 11).
KERNEL_TOLERANCE = 1e-2


def _scan_backend_of(model: torch.nn.Module) -> str:
    """Return the selective-scan backend for a ChangeMamba model, ``"none"`` for other models."""
    if type(model).__name__ != "ChangeMambaWrapper":
        return "none"
    from hasar.models.changemamba import scan_backend

    return scan_backend()


def _kernel_equivalence(model: torch.nn.Module, device: str) -> float:
    """Return max |clf| difference between the CUDA kernel and the forced pure-PyTorch scan."""
    torch.manual_seed(0)
    pre = torch.rand(1, 3, 64, 64, device=device)
    post = torch.rand(1, 3, 64, 64, device=device)
    was_training = model.training
    model.eval()
    previous = os.environ.get("HASAR_SCAN_BACKEND")
    try:
        with torch.no_grad():
            kernel = model(pre, post)["clf"].float()
            os.environ["HASAR_SCAN_BACKEND"] = "torch"
            reference = model(pre, post)["clf"].float()
    finally:
        if previous is None:
            os.environ.pop("HASAR_SCAN_BACKEND", None)
        else:
            os.environ["HASAR_SCAN_BACKEND"] = previous
        model.train(was_training)
    return float((kernel - reference).abs().max())


# --- preflight.json ---------------------------------------------------------


def write_preflight(
    path: Path | str,
    results: list[GateResult],
    config_hash: str,
    commit: str,
) -> Path:
    """Write the gate results and their provenance to ``path`` as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config_hash": config_hash,
        "commit": commit,
        "all_passed": all(result.passed for result in results),
        "gates": [
            {
                "gate": result.gate,
                "name": result.name,
                "passed": result.passed,
                "value": result.value,
                "detail": result.detail,
            }
            for result in results
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def read_preflight(path: Path | str) -> dict[str, Any]:
    """Read a ``preflight.json`` file written by :func:`write_preflight`."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def preflight_is_current(path: Path | str, config_hash: str) -> bool:
    """Return ``True`` if ``path`` exists, matches ``config_hash`` and every gate passed."""
    path = Path(path)
    if not path.exists():
        return False
    try:
        payload = read_preflight(path)
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("config_hash") == config_hash and payload.get("all_passed") is True
