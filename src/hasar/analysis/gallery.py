"""SAR error gallery (Capella/Umbra breakdown, layover and shadow, isolated damaged buildings).

The gallery is built from cached predictions and per-tile confusion matrices only; the model is
never rerun. ``panel`` lays the pre-disaster optical tile, the post-disaster SAR tile, the label
and the prediction side by side so a reader can see *why* a tile failed. ``rank_tiles`` picks the
worst (or best) tiles of an event by a chosen metric, and ``group_summary`` sums the per-tile
matrices per SAR source (Capella, Umbra, ...) so the breakdown never touches the model either.

Everything here works on numpy arrays and cached predictions; no model is imported.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from hasar.eval.metrics import summary

#: Colour of every label class in the gallery panels (RGB).
#: 0 background black, 1 intact green, 2 damaged orange, 3 destroyed red, 255 nodata grey.
PALETTE: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 0),
    1: (0, 200, 0),
    2: (255, 140, 0),
    3: (220, 0, 0),
    255: (128, 128, 128),
}

#: Group name used for tiles that are missing from the ``groups`` mapping.
UNKNOWN_GROUP = "bilinmiyor"

#: Default side length of one panel tile in pixels.
DEFAULT_SIZE = 256


def colorize(label: np.ndarray) -> np.ndarray:
    """Return the ``(H, W, 3)`` uint8 RGB image of a label map using :data:`PALETTE`.

    Values outside the palette fall back to the nodata grey, so an unexpected class never crashes
    the gallery.
    """
    label = np.asarray(label)
    rgb = np.zeros(label.shape + (3,), dtype=np.uint8)
    for value, colour in PALETTE.items():
        rgb[label == value] = colour
    return rgb


def _resize(array: np.ndarray, size: int) -> np.ndarray:
    """Resize an ``(H, W)`` or ``(H, W, 3)`` uint8 array to ``size x size`` (nearest neighbour)."""
    image = Image.fromarray(np.asarray(array, dtype=np.uint8))
    return np.asarray(image.resize((size, size), Image.NEAREST))


def panel(
    pre: np.ndarray,
    post: np.ndarray,
    label: np.ndarray,
    pred: np.ndarray,
    size: int = DEFAULT_SIZE,
) -> Image.Image:
    """Return a ``(4 * size, size)`` RGB panel of one tile.

    The four columns are the pre-disaster optical image, the post-disaster SAR image rendered as
    grey, the colourised label and the colourised prediction. Every input is resized to
    ``size x size`` with nearest-neighbour sampling.
    """
    pre_rgb = _resize(pre, size)
    post_grey = _resize(post, size)
    post_rgb = np.repeat(post_grey[:, :, None], 3, axis=2)
    label_rgb = _resize(colorize(label), size)
    pred_rgb = _resize(colorize(pred), size)

    columns = np.concatenate([pre_rgb, post_rgb, label_rgb, pred_rgb], axis=1)
    return Image.fromarray(columns)


def _has_buildings(cm: np.ndarray) -> bool:
    """Return whether a confusion matrix has any building pixels (rows 1..3 non-empty)."""
    return bool(np.asarray(cm, dtype=np.int64)[1:, :].sum() > 0)


def rank_tiles(
    per_tile: dict[str, np.ndarray],
    key: str = "building_miou",
    worst: bool = True,
    k: int = 10,
) -> list[tuple[str, float]]:
    """Return ``(tile, score)`` pairs sorted by ``key``, worst first when ``worst`` is true.

    Tiles without any building pixel are skipped: their building IoU is undefined and they would
    dominate the ranking for no reason. At most ``k`` pairs are returned.
    """
    scored: list[tuple[str, float]] = []
    for name, cm in per_tile.items():
        if not _has_buildings(cm):
            continue
        scored.append((name, float(summary(cm)[key])))

    scored.sort(key=lambda item: item[1], reverse=not worst)
    return scored[:k]


def group_summary(per_tile: dict[str, np.ndarray], groups: dict[str, str]) -> dict[str, dict[str, float | int]]:
    """Sum the per-tile matrices per group and return each group's :func:`summary` plus ``tiles``.

    Tiles missing from ``groups`` land in the :data:`UNKNOWN_GROUP` bucket instead of being
    dropped, so the breakdown always accounts for every tile.
    """
    totals: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for name, cm in per_tile.items():
        group = groups.get(name, UNKNOWN_GROUP)
        matrix = np.asarray(cm, dtype=np.int64)
        totals[group] = totals[group] + matrix if group in totals else matrix
        counts[group] = counts.get(group, 0) + 1

    result: dict[str, dict[str, float | int]] = {}
    for group, matrix in totals.items():
        entry: dict[str, float | int] = dict(summary(matrix))
        entry["tiles"] = counts[group]
        result[group] = entry
    return result
