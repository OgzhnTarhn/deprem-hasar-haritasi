"""Calibration and uncertainty helpers (numpy only).

Protocol section 10 fixes the rules implemented here:

* one single temperature ``T`` for the ``clf`` logits, chosen by negative log-likelihood on the
  validation split of the 13 training events;
* ECE over building pixels (labels 1-3) using the confidence of the most likely class and 15
  equal-width bins;
* entropy of the averaged flip-TTA probabilities as the building uncertainty.

``T`` is chosen on the 13 validation events only and merely measured on Turkey (protocol
section 10, rule 1): nothing in this module may look at Turkey to pick a value.

All accumulators are streaming: they keep one running sum per temperature or per bin, never the
pixels themselves, so a whole validation split can be fed tile by tile.
"""

from __future__ import annotations

import numpy as np

DEFAULT_GRID = np.round(np.arange(0.5, 3.01, 0.05), 2)
DEFAULT_BINS = 15
BUILDING_LABELS = (1, 2, 3)
_LOG_FLOOR = 1e-12


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax along the class axis (axis 0)."""
    shifted = logits - logits.max(axis=0, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=0, keepdims=True))


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Return ``probs`` rescaled by a single temperature ``temperature``.

    The probabilities are turned back into logits with a logarithm, divided by ``temperature``
    and passed through a softmax along axis 0, so ``T = 1`` is the identity, ``T > 1`` softens
    and ``T < 1`` sharpens the distribution. The result sums to one along axis 0.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    logits = np.log(np.clip(np.asarray(probs, dtype=np.float64), _LOG_FLOOR, None))
    return np.exp(_log_softmax(logits / temperature))


class NllAccumulator:
    """Streaming negative log-likelihood over a fixed temperature grid.

    ``sums`` holds the summed NLL of every grid value over all pixels added so far; ``best()``
    returns the grid temperature with the lowest sum. Pixels whose label is outside
    ``[0, num_classes)`` (for example 255 nodata) are ignored.
    """

    def __init__(self, grid: np.ndarray | None = None) -> None:
        self.grid = DEFAULT_GRID.copy() if grid is None else np.asarray(grid, dtype=np.float64)
        self.sums = np.zeros(self.grid.shape[0], dtype=np.float64)

    def add(self, probs: np.ndarray, labels: np.ndarray) -> None:
        """Add one batch of class probabilities ``(C, N)`` and integer labels ``(N,)``."""
        probs = np.asarray(probs, dtype=np.float64)
        labels = np.asarray(labels).reshape(-1)
        if probs.ndim != 2 or probs.shape[1] != labels.shape[0]:
            raise ValueError(f"probs {probs.shape} and labels {labels.shape} do not match")

        valid = (labels >= 0) & (labels < probs.shape[0])
        if not valid.any():
            return
        probs = probs[:, valid]
        labels = labels[valid].astype(np.int64)
        columns = np.arange(labels.shape[0])

        logits = np.log(np.clip(probs, _LOG_FLOOR, None))
        for index, temperature in enumerate(self.grid):
            log_probs = _log_softmax(logits / temperature)
            self.sums[index] += -log_probs[labels, columns].sum()

    def best(self) -> float:
        """Return the grid temperature with the lowest accumulated NLL."""
        return float(self.grid[int(np.argmin(self.sums))])


class EceAccumulator:
    """Streaming expected calibration error over building pixels.

    Only pixels whose label is a building class (1-3) count. The confidence is the probability
    of the most likely class and the bins are equal-width over ``[0, 1]``; the last bin is
    closed so that a confidence of exactly 1.0 lands in it.
    """

    def __init__(self, bins: int = DEFAULT_BINS) -> None:
        if bins < 1:
            raise ValueError(f"bins must be positive, got {bins}")
        self.bins = bins
        self.counts = np.zeros(bins, dtype=np.float64)
        self.confidence_sums = np.zeros(bins, dtype=np.float64)
        self.correct_sums = np.zeros(bins, dtype=np.float64)

    def add(self, probs: np.ndarray, labels: np.ndarray) -> None:
        """Add one batch of class probabilities ``(C, N)`` and integer labels ``(N,)``."""
        probs = np.asarray(probs, dtype=np.float64)
        labels = np.asarray(labels).reshape(-1)
        if probs.ndim != 2 or probs.shape[1] != labels.shape[0]:
            raise ValueError(f"probs {probs.shape} and labels {labels.shape} do not match")

        valid = np.isin(labels, BUILDING_LABELS)
        if not valid.any():
            return
        probs = probs[:, valid]
        labels = labels[valid].astype(np.int64)

        confidence = probs.max(axis=0)
        predicted = probs.argmax(axis=0)
        correct = (predicted == labels).astype(np.float64)

        index = np.minimum((confidence * self.bins).astype(np.int64), self.bins - 1)
        self.counts += np.bincount(index, minlength=self.bins)
        self.confidence_sums += np.bincount(index, weights=confidence, minlength=self.bins)
        self.correct_sums += np.bincount(index, weights=correct, minlength=self.bins)

    def value(self) -> float:
        """Return the ECE, or 0.0 when no building pixel has been added yet."""
        total = self.counts.sum()
        if total == 0:
            return 0.0
        accuracy = np.divide(self.correct_sums, self.counts, out=np.zeros_like(self.counts), where=self.counts > 0)
        mean_confidence = np.divide(
            self.confidence_sums, self.counts, out=np.zeros_like(self.counts), where=self.counts > 0
        )
        return float((np.abs(accuracy - mean_confidence) * self.counts).sum() / total)


def entropy(probs: np.ndarray) -> np.ndarray:
    """Return the Shannon entropy (nats) of each pixel's class distribution.

    ``probs`` is ``(C, N)``; the result is ``(N,)``. Zero probabilities contribute zero, so a
    one-hot distribution has entropy exactly 0.
    """
    probs = np.asarray(probs, dtype=np.float64)
    terms = np.where(probs > 0, probs * np.log(np.where(probs > 0, probs, 1.0)), 0.0)
    return -terms.sum(axis=0)
