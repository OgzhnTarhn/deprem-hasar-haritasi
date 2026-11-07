"""Phase 3 reproduction runner for the published BRIGHT zero-shot baselines.

Phase 3 exit criterion
----------------------
The published zero-shot checkpoints are re-run on the Turkey earthquake and the resulting
``[ZS-EVENT mIoU]`` must land within :data:`TOLERANCE_PCT` (0.3 points) of the value published
with the checkpoint. This is the only check that the evaluation path itself is faithful before
any training starts.

Project rule 1, first exception
-------------------------------
Turkey is normally read only in the ``final`` step behind the protocol tag. Re-running the
ready-made zero-shot checkpoints is the first of the two exceptions: it does no training, and its
result is tagged ``repro`` (``[ZS-EVENT mIoU, repro]``) so it is never the headline number. Turkey
imagery and labels are still read only through
:func:`hasar.guard.require_reproduction_access`, which accepts the known checkpoint names and
records the commit behind the numbers.

The ``val`` split (the 13 non-Turkey events) needs no Turkey access and is used to sanity-check
the pipeline without touching the test event.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

from hasar.data.dataset import BrightDataset
from hasar.data.events import load_standard_lists, make_zero_shot_split
from hasar.eval.metrics import building_miou, iou_per_class, miou
from hasar.eval.predict import evaluate
from hasar.guard import TurkeyAccess, require_reproduction_access
from hasar.models.bright_baselines import BASELINES
from hasar.results import ResultRecord, write_result

#: Phase 3 exit criterion: the reproduced mIoU must be within this many points of the published one.
TOLERANCE_PCT = 0.3

SPLITS: tuple[str, ...] = ("test", "val")

ModelFn = Callable[[str], object]
AccessFn = Callable[[Path | str, str], TurkeyAccess]

_DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[3]


def _default_model_fn(key: str, checkpoint_path: str | Path | None) -> object:
    """Build the published baseline ``key`` and load its checkpoint.

    ``hasar.models.bright_baselines`` is imported for its registration side effect, then the model
    is built from the registry and the checkpoint is loaded strictly.
    """
    if checkpoint_path is None:
        raise ValueError("varsayılan model_fn için checkpoint_path zorunlu")
    import hasar.models.bright_baselines  # noqa: F401  (registers the baseline factories)
    from hasar.models.bright_baselines import load_checkpoint
    from hasar.models.registry import build

    model = build(BASELINES[key].registry_name)
    load_checkpoint(model, checkpoint_path)
    return model


def _head_commit(repo_root: Path) -> str:
    """Return ``git rev-parse HEAD`` for ``repo_root``, or ``"unknown"`` when git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "unknown"


def _metric_values(cm) -> dict[str, float]:
    """Return the percent metrics derived from one event-level confusion matrix."""
    ious = iou_per_class(cm)
    return {
        "miou_pct": miou(cm) * 100,
        "iou_background_pct": float(ious[0]) * 100,
        "iou_intact_pct": float(ious[1]) * 100,
        "iou_damaged_pct": float(ious[2]) * 100,
        "iou_destroyed_pct": float(ious[3]) * 100,
        "building_miou_pct": building_miou(cm) * 100,
    }


def reproduce(
    key: str,
    data_root: Path | str,
    out_dir: Path | str,
    split: str = "test",
    names: list[str] | tuple[str, ...] | None = None,
    checkpoint_path: str | Path | None = None,
    repo_root: Path | str | None = None,
    tta: bool = False,
    device: str = "cpu",
    access_fn: AccessFn | None = None,
    model_fn: ModelFn | None = None,
) -> ResultRecord:
    """Reproduce one published baseline on the ``test`` (Turkey) or ``val`` split.

    ``names`` defaults to the zero-shot split built from the ``standard_ML`` lists. The ``test``
    split requires Turkey access through ``access_fn`` (default
    :func:`hasar.guard.require_reproduction_access`) and compares the result with the published
    value; the ``val`` split never requests access. The confusion matrices are written to
    ``<out_dir>/<key>_<split>_matrices.npz`` and the record to ``<out_dir>/<key>_<split>.json``.
    """
    if split not in SPLITS:
        raise ValueError(f"bilinmeyen bölme: {split!r} (izinli: {SPLITS})")
    if key not in BASELINES:
        raise KeyError(f"bilinmeyen taban model: {key!r} (izinli: {sorted(BASELINES)})")

    baseline = BASELINES[key]
    repo_root = Path(repo_root) if repo_root is not None else _DEFAULT_REPO_ROOT
    out_dir = Path(out_dir)

    if names is None:
        zero_shot = make_zero_shot_split(*load_standard_lists())
        names = zero_shot.test if split == "test" else zero_shot.val
    names = list(names)

    checkpoint_name = Path(checkpoint_path).name if checkpoint_path is not None else baseline.checkpoint

    access: TurkeyAccess | None = None
    if split == "test":
        access_fn = access_fn if access_fn is not None else require_reproduction_access
        access = access_fn(repo_root, checkpoint_name)

    if model_fn is None:
        model = _default_model_fn(key, checkpoint_path)
    else:
        model = model_fn(key)

    dataset = BrightDataset(data_root, names, mode="eval", sar_mode="official", access=access)
    accumulator = evaluate(model, dataset, device=device, tta=tta)

    npz_path = out_dir / f"{key}_{split}_matrices.npz"
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    accumulator.save_npz(npz_path)

    values = _metric_values(accumulator.total)
    extra: dict[str, object] = {
        "checkpoint": checkpoint_name,
        "tiles": len(names),
        "tta": tta,
    }

    if split == "test":
        published = baseline.published_turkey
        diff = values["miou_pct"] - published
        values["published_pct"] = published
        values["diff_pct"] = diff
        extra["tolerance_pct"] = TOLERANCE_PCT
        extra["within_tolerance"] = bool(abs(diff) <= TOLERANCE_PCT)
        metric_tag = "[ZS-EVENT mIoU, repro]"
        commit = access.commit if access is not None else _head_commit(repo_root)
    else:
        metric_tag = "[VAL13 mIoU, repro]"
        commit = _head_commit(repo_root)

    record = ResultRecord(
        metric_tag=metric_tag,
        values=values,
        model=baseline.registry_name,
        split=f"zero_shot/{split}",
        commit=commit,
        config_hash="none",
        extra=extra,
    )
    write_result(out_dir / f"{key}_{split}.json", record)
    return record
