"""Transductive Y3 analysis: re-standardise Turkey SAR with Turkey's unlabeled statistics.

This module implements rule 1's second exception. Phase 7's Y3 analysis may use the *unlabeled*
SAR statistics of the Turkey scenes to re-standardise Turkey SAR before the model sees it. The
result is reported in a separate "transductive" table only: it never enters the headline number
and it makes no novelty claim. The target moments are the SAR moments of the 13 training events
(computed from the training split only); the source moments are Turkey's. Running the model with
re-standardised inputs is done by the Faz 7 runner after the headline ``final`` result exists.

Everything here works on numpy arrays and cached predictions; no model is imported.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Qualifier appended to the metric tag of every transductive number (rule 4).
TRANSDUCTIVE_QUALIFIER = "transduktif-y3"

#: SAR values are stored as uint8, so re-standardised pixels are clipped to this range.
SAR_MIN = 0.0
SAR_MAX = 255.0


@dataclass(frozen=True)
class Moments:
    """Population mean and standard deviation (``ddof=0``) over a set of valid pixels."""

    mean: float
    std: float
    count: int


class MomentAccumulator:
    """Streaming (Chan et al.) accumulation of mean and standard deviation.

    Memory stays O(one chunk), which matters on Colab with High-RAM off. Only pixels selected by
    the ``valid`` mask contribute; nodata pixels are ignored.
    """

    def __init__(self) -> None:
        self._count = 0
        self._mean = 0.0
        self._m2 = 0.0

    def add(self, post: np.ndarray, valid: np.ndarray) -> None:
        """Add the valid pixels of one SAR chunk."""
        values = np.asarray(post, dtype=np.float64)[np.asarray(valid, dtype=bool)]
        n = int(values.size)
        if n == 0:
            return
        chunk_mean = float(values.mean())
        chunk_m2 = float(((values - chunk_mean) ** 2).sum())
        total = self._count + n
        delta = chunk_mean - self._mean
        self._mean += delta * n / total
        self._m2 += chunk_m2 + delta * delta * self._count * n / total
        self._count = total

    def result(self) -> Moments:
        """Return the accumulated moments; raises :class:`ValueError` if nothing was added."""
        if self._count == 0:
            raise ValueError("moment birikimi boş: geçerli piksel yok")
        return Moments(mean=self._mean, std=float(np.sqrt(self._m2 / self._count)), count=self._count)


def restandardize(post: np.ndarray, source: Moments, target: Moments) -> np.ndarray:
    """Map ``post`` from the ``source`` moments to the ``target`` moments.

    ``post == 0`` marks nodata and stays ``0.0`` in the output. The result is float32 and clipped
    to ``[0, 255]`` so it can be fed to the network as a SAR tile.
    """
    if not np.isfinite(source.std) or source.std <= 0:
        raise ValueError(f"kaynak standart sapması pozitif olmalı, alınan: {source.std!r}")

    array = np.asarray(post, dtype=np.float64)
    valid = array != 0
    out = np.zeros(array.shape, dtype=np.float64)
    scaled = (array[valid] - source.mean) / source.std * target.std + target.mean
    out[valid] = np.clip(scaled, SAR_MIN, SAR_MAX)
    return out.astype(np.float32)


def metric_tag() -> str:
    """Return the metric tag of the transductive table (rule 4: the type is always named)."""
    return f"[ZS-EVENT mIoU, {TRANSDUCTIVE_QUALIFIER}]"
