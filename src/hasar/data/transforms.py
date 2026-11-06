"""Paired geometric augmentation for optical, SAR and label tiles.

The official BRIGHT/ChangeMamba recipe is geometric only: a random crop, random horizontal and
vertical flips and a random 90 degree rotation. No photometric augmentation is applied, so the
same window and the same flip/rotation are applied to every array of a sample (optical, SAR,
label and any extra mask) and the label stays aligned with the inputs.

Random numbers are drawn in a fixed order (crop offset ``y`` then ``x``, then the horizontal flip,
the vertical flip and the rotation count ``k``), so a seed fully determines the result. Every
returned array is a fresh C-contiguous copy that shares no memory with its input.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _fresh(array: np.ndarray) -> np.ndarray:
    """Return a C-contiguous copy of ``array`` that shares no memory with it."""
    return np.ascontiguousarray(np.array(array, copy=True))


def _check_arrays(arrays: Sequence[np.ndarray]) -> tuple[int, int]:
    """Validate that every array is at least 2-D and shares the same ``(H, W)``."""
    if not arrays:
        raise ValueError("en az bir dizi gerekli")
    first = np.asarray(arrays[0])
    if first.ndim < 2:
        raise ValueError(f"diziler en az 2 boyutlu olmalı, alınan: {first.shape}")
    height, width = first.shape[:2]
    for array in arrays[1:]:
        shape = np.asarray(array).shape
        if shape[:2] != (height, width):
            raise ValueError(f"bütün diziler aynı (H, W) olmalı, alınan: {first.shape[:2]} ve {shape[:2]}")
    return height, width


def random_crop(arrays: Sequence[np.ndarray], size: int, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    """Cut the same ``size x size`` window from every array.

    The top-left offset is drawn as ``y`` then ``x`` from ``rng``. ``size`` may not exceed the
    height or the width of the arrays.
    """
    arrays = [np.asarray(array) for array in arrays]
    height, width = _check_arrays(arrays)
    if size <= 0:
        raise ValueError(f"kırpma boyutu pozitif olmalı, alınan: {size}")
    if size > height:
        raise ValueError(f"kırpma boyutu yükseklikten büyük olamaz: {size} > {height}")
    if size > width:
        raise ValueError(f"kırpma boyutu genişlikten büyük olamaz: {size} > {width}")

    top = int(rng.integers(0, height - size + 1))
    left = int(rng.integers(0, width - size + 1))
    return tuple(_fresh(array[top : top + size, left : left + size]) for array in arrays)


def random_flip_rot(
    arrays: Sequence[np.ndarray],
    rng: np.random.Generator,
    flips: bool = True,
    rot90: bool = True,
) -> tuple[np.ndarray, ...]:
    """Apply the same random flips and 90 degree rotation to every array.

    The horizontal flip, the vertical flip and the rotation count ``k`` are drawn in that order
    from ``rng``. The channel axis (the last one) is never touched.
    """
    arrays = [np.asarray(array) for array in arrays]
    _check_arrays(arrays)

    horizontal = bool(rng.integers(0, 2)) if flips else False
    vertical = bool(rng.integers(0, 2)) if flips else False
    k = int(rng.integers(0, 4)) if rot90 else 0

    out: list[np.ndarray] = []
    for array in arrays:
        result = array
        if horizontal:
            result = np.flip(result, axis=1)
        if vertical:
            result = np.flip(result, axis=0)
        if k:
            result = np.rot90(result, k=k, axes=(0, 1))
        out.append(_fresh(result))
    return tuple(out)


def augment(
    pre: np.ndarray,
    post: np.ndarray,
    label: np.ndarray,
    rng: np.random.Generator,
    crop: int | None = None,
    flips: bool = True,
    rot90: bool = True,
    extra: Sequence[np.ndarray] = (),
) -> tuple[np.ndarray, ...]:
    """Apply the paired geometric augmentation to a sample.

    ``pre``, ``post``, ``label`` and every array in ``extra`` get the same crop and the same
    flips/rotation. Returns ``(pre, post, label, *extra)`` as fresh C-contiguous arrays.
    """
    arrays = [pre, post, label, *extra]
    if crop is not None:
        arrays = list(random_crop(arrays, size=crop, rng=rng))
    return random_flip_rot(arrays, rng, flips=flips, rot90=rot90)
