"""Data summaries: label histograms and split reports.

The summary helpers are data-free in the sense that they only read the damage labels of the
tiles they are given; no imagery is decoded. Two rules shape the code.

**Turkey guard.** Every label read goes through :func:`hasar.data.io.read_label`, which asks
:func:`hasar.guard.is_turkey_tile` before opening any file. A Turkey tile therefore raises
:class:`hasar.guard.TurkeyLockError` unless a :class:`hasar.guard.TurkeyAccess` token is passed.

**Test labels stay unread.** :func:`build_split_report` counts pixels for the train and val
parts only; the test part is summarised by tile count, never by label content.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np

from hasar.data.events import Split, split_summary
from hasar.data.io import read_label
from hasar.guard import TurkeyAccess

DEFAULT_NUM_CLASSES = 4


def label_histogram(
    root: Path | str,
    names: Iterable[str],
    num_classes: int = DEFAULT_NUM_CLASSES,
    access: TurkeyAccess | None = None,
) -> dict[str, object]:
    """Count label pixels per class over ``names`` under ``root``.

    Returns ``{"pixels": {"0": n, ...}, "other": n, "tiles": n}``. ``pixels`` holds one
    string-keyed entry per class in ``range(num_classes)``; every pixel outside that range is
    counted in ``other``. The Turkey gate runs before any disk access.
    """
    names = list(names)
    counts = np.zeros(256, dtype=np.int64)
    for name in names:
        label = read_label(root, name, access=access)
        counts += np.bincount(label.ravel(), minlength=256)

    pixels = {str(index): int(counts[index]) for index in range(num_classes)}
    other = int(counts[num_classes:].sum())
    return {"pixels": pixels, "other": other, "tiles": len(names)}


def build_split_report(
    split: Split,
    root: Path | str | None = None,
    access: TurkeyAccess | None = None,
) -> dict[str, object]:
    """Return the split summary, adding per-class pixel counts when ``root`` is given.

    Only the train and val parts are counted; the test labels are never read.
    """
    report = split_summary(split)
    if root is None:
        return report

    class_pixels: dict[str, dict[str, object]] = {}
    for part in ("train", "val"):
        class_pixels[part] = label_histogram(root, list(getattr(split, part)), access=access)
    report["class_pixels"] = class_pixels
    return report
