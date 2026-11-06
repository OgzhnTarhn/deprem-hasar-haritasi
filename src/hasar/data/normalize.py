"""Optical and SAR input normalization.

Optical pre-disaster tiles are normalized with the ImageNet channel statistics, the same values
the ChangeMamba backbone was pretrained with.

SAR post-disaster tiles have two modes:

* ``official`` (default): the official BRIGHT/ChangeMamba input format. The single SAR band is
  copied to three channels and normalized with the ImageNet statistics (ADR-0004 option C).
* ``train_stats``: idea A3, per-dataset SAR statistics. The statistics may only be computed from
  the 13 training events; project rule 1 forbids using Turkey imagery or labels for any
  normalization statistic, so :func:`compute_sar_stats` rejects Turkey tile names before it loads
  anything.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from hasar.guard import TurkeyLockError, is_turkey_tile

#: ImageNet per-channel mean in 0-255 scale (RGB order).
IMAGENET_MEAN: tuple[float, float, float] = (123.675, 116.28, 103.53)
#: ImageNet per-channel standard deviation in 0-255 scale (RGB order).
IMAGENET_STD: tuple[float, float, float] = (58.395, 57.12, 57.375)

SAR_MODES: tuple[str, ...] = ("official", "train_stats")


@dataclass(frozen=True)
class SarStats:
    """Mean and standard deviation of the valid SAR pixels of the training events."""

    mean: float
    std: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.mean):
            raise ValueError(f"SAR ortalaması sonlu olmalı, alınan: {self.mean!r}")
        if not np.isfinite(self.std) or self.std <= 0:
            raise ValueError(f"SAR standart sapması pozitif olmalı, alınan: {self.std!r}")

    def to_dict(self) -> dict[str, float]:
        """Return the statistics as a JSON-serializable payload."""
        return {"mean": float(self.mean), "std": float(self.std)}

    @classmethod
    def from_dict(cls, payload: dict[str, float]) -> SarStats:
        """Rebuild the statistics from a :meth:`to_dict` payload."""
        return cls(mean=float(payload["mean"]), std=float(payload["std"]))


def _imagenet_arrays() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1),
        np.asarray(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1),
    )


def normalize_optical(pre: np.ndarray) -> np.ndarray:
    """Normalize an ``(H, W, 3)`` uint8 optical tile to ``(3, H, W)`` float32.

    The tile is converted to channels-first and scaled with the ImageNet statistics.
    """
    array = np.asarray(pre)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"optik karo (H, W, 3) olmalı, alınan: {array.shape}")
    chw = np.transpose(array.astype(np.float32), (2, 0, 1))
    mean, std = _imagenet_arrays()
    return ((chw - mean) / std).astype(np.float32)


def sar_to_input(
    post: np.ndarray,
    mode: str = "official",
    stats: SarStats | None = None,
) -> np.ndarray:
    """Turn a SAR tile into a ``(C, H, W)`` float32 network input.

    ``mode="official"`` copies the single band to three channels and applies the ImageNet
    statistics (ADR-0004 option C). ``mode="train_stats"`` normalizes with the given
    :class:`SarStats` (idea A3) and keeps a single channel.
    """
    array = np.asarray(post, dtype=np.float32)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"SAR karosu (H, W) olmalı, alınan: {array.shape}")

    if mode == "official":
        chw = np.repeat(array[np.newaxis, :, :], 3, axis=0)
        mean, std = _imagenet_arrays()
        return ((chw - mean) / std).astype(np.float32)

    if mode == "train_stats":
        if stats is None:
            raise ValueError("mode='train_stats' için SarStats gerekli")
        return ((array[np.newaxis, :, :] - np.float32(stats.mean)) / np.float32(stats.std)).astype(np.float32)

    raise ValueError(f"bilinmeyen SAR normalizasyon modu: {mode!r} (izinli: {SAR_MODES})")


def compute_sar_stats(
    names: list[str],
    load: Callable[[str], tuple[np.ndarray, np.ndarray]],
) -> SarStats:
    """Compute SAR statistics from the valid pixels of the given training tiles.

    ``load(name)`` returns ``(array, valid_mask)``. Every name is checked against the Turkey lock
    before any tile is loaded, so a Turkey tile can never contribute to a normalization statistic
    (project rule 1).
    """
    if not names:
        raise ValueError("SAR istatistikleri için en az bir karo adı gerekli")

    for name in names:
        if is_turkey_tile(name):
            raise TurkeyLockError(
                f"'{name}' Türkiye karosu: normalizasyon istatistikleri Türkiye'yi kullanmaz (proje kuralı 1)."
            )

    # Streaming (Chan et al.) combination of per-tile mean and M2: memory stays O(one tile), which
    # matters on Colab with High-RAM off (2.795 tiles x 1M pixels would not fit as one array).
    count = 0
    mean = 0.0
    m2 = 0.0
    for name in names:
        array, mask = load(name)
        values = np.asarray(array, dtype=np.float64)[np.asarray(mask, dtype=bool)]
        n = int(values.size)
        if n == 0:
            continue
        tile_mean = float(values.mean())
        tile_m2 = float(((values - tile_mean) ** 2).sum())
        total = count + n
        delta = tile_mean - mean
        mean += delta * n / total
        m2 += tile_m2 + delta * delta * count * n / total
        count = total
    if count == 0:
        raise ValueError("SAR istatistikleri için geçerli piksel yok")
    return SarStats(mean=mean, std=float(np.sqrt(m2 / count)))
