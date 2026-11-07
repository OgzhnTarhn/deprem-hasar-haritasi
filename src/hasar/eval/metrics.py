"""Segmentation metrics for building damage mapping (numpy only).

Locked headline path (protocol section 3)
-----------------------------------------
The headline number is computed exactly as follows:

* every tile is evaluated at its full 1024x1024 resolution;
* the prediction is ``pred = argmax(clf)`` over the four classes;
* one single 4x4 confusion matrix is accumulated over *all* tiles of the event;
* labels outside ``[0, 4)`` (for example 255 nodata) are dropped;
* nodata is *not* excluded separately, so the SAR nodata mask does not change the metric;
* the IoU denominator is ``row + col - diag + 1e-7``;
* the mean is taken over all four classes, background included.

Only this number is compared with 41.28 ``[ZS-EVENT mIoU]``. Per-tile confusion matrices are
kept as well, so bootstrap resampling and per-event breakdowns never have to rerun the model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

NUM_CLASSES = 4
EPS = 1e-7
CLASS_NAMES = ("background", "intact", "damaged", "destroyed")

TOTAL_KEY = "__total__"


def confusion_matrix(label: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Return the 4x4 confusion matrix of ``label`` versus ``pred``.

    Both arrays may have any shape as long as they match; they are flattened first. Labels
    outside ``[0, NUM_CLASSES)`` are dropped, predictions outside the range raise ``ValueError``.
    """
    label = np.asarray(label)
    pred = np.asarray(pred)
    if label.shape != pred.shape:
        raise ValueError(f"label and pred shapes differ: {label.shape} vs {pred.shape}")

    label = label.reshape(-1)
    pred = pred.reshape(-1)

    mask = (label >= 0) & (label < NUM_CLASSES)
    label = label[mask].astype(np.int64)
    pred = pred[mask].astype(np.int64)

    if pred.size and (pred.min() < 0 or pred.max() >= NUM_CLASSES):
        raise ValueError(f"pred values must be in [0, {NUM_CLASSES}), got range [{pred.min()}, {pred.max()}]")

    index = NUM_CLASSES * label + pred
    count = np.bincount(index, minlength=NUM_CLASSES**2)
    return count.reshape(NUM_CLASSES, NUM_CLASSES).astype(np.int64)


def iou_per_class(cm: np.ndarray) -> np.ndarray:
    """Return the per-class IoU vector of a confusion matrix (float64)."""
    cm = np.asarray(cm, dtype=np.int64)
    diag = np.diag(cm)
    denominator = cm.sum(axis=1) + cm.sum(axis=0) - diag + EPS
    return diag / denominator


def miou(cm: np.ndarray) -> float:
    """Return the mean IoU over all four classes, background included."""
    return float(np.mean(iou_per_class(cm)))


def pixel_accuracy(cm: np.ndarray) -> float:
    """Return the overall pixel accuracy of a confusion matrix."""
    cm = np.asarray(cm, dtype=np.int64)
    return float(np.diag(cm).sum() / (cm.sum() + EPS))


def building_miou(cm: np.ndarray) -> float:
    """Return the mean IoU over the three building classes (background excluded)."""
    return float(np.mean(iou_per_class(cm)[1:]))


def summary(cm: np.ndarray) -> dict[str, float | int]:
    """Return the headline metrics of a confusion matrix as a flat dictionary."""
    ious = iou_per_class(cm)
    out: dict[str, float | int] = {
        "miou": float(np.mean(ious)),
        "building_miou": float(np.mean(ious[1:])),
        "pixel_accuracy": pixel_accuracy(cm),
    }
    for name, value in zip(CLASS_NAMES, ious, strict=True):
        out[f"iou_{name}"] = float(value)
    out["pixels"] = int(np.asarray(cm, dtype=np.int64).sum())
    return out


def always_background_miou(background_fraction: float) -> float:
    """Return the mIoU of a model that predicts background everywhere.

    Only background has a non-zero IoU (equal to its share of the pixels), the three building
    classes score zero, so the mean over four classes is ``background_fraction / 4``.
    """
    return float(background_fraction) / NUM_CLASSES


class ConfusionAccumulator:
    """Accumulate per-tile confusion matrices into one event-level matrix.

    ``total`` is the sum over all tiles, ``per_tile`` keeps each tile's own matrix so bootstrap
    and breakdowns can reuse them without rerunning the model.
    """

    def __init__(self) -> None:
        self.total = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        self.per_tile: dict[str, np.ndarray] = {}

    def add(self, label: np.ndarray, pred: np.ndarray, key: str) -> None:
        """Add one tile's confusion matrix under ``key``; a repeated key is summed."""
        cm = confusion_matrix(label, pred)
        self.total += cm
        if key in self.per_tile:
            self.per_tile[key] = self.per_tile[key] + cm
        else:
            self.per_tile[key] = cm

    def save_npz(self, path: str | Path) -> None:
        """Save the total matrix under ``__total__`` and every tile under its own key."""
        arrays: dict[str, np.ndarray] = {TOTAL_KEY: self.total}
        arrays.update(self.per_tile)
        np.savez(path, **arrays)

    @classmethod
    def load_npz(cls, path: str | Path) -> ConfusionAccumulator:
        """Rebuild an accumulator from an npz file written by :meth:`save_npz`."""
        acc = cls()
        with np.load(path) as data:
            if TOTAL_KEY not in data:
                raise ValueError(f"npz file has no {TOTAL_KEY!r} entry: {path}")
            acc.total = np.array(data[TOTAL_KEY], dtype=np.int64)
            for key in data.files:
                if key == TOTAL_KEY:
                    continue
                acc.per_tile[key] = np.array(data[key], dtype=np.int64)
        return acc
