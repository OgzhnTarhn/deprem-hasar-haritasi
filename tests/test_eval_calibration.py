"""Tests for calibration and uncertainty helpers (``hasar.eval.calibration``).

Protocol section 10: a single temperature chosen by negative log-likelihood on the 13-event
validation set, ECE over building pixels (labels 1-3) with 15 equal-width bins, and entropy of
the averaged flip-TTA probabilities. Every array is tiny synthetic numpy, so no data files and
no network are needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from hasar.eval.calibration import (
    EceAccumulator,
    NllAccumulator,
    apply_temperature,
    entropy,
)

DEFAULT_GRID = np.round(np.arange(0.5, 3.01, 0.05), 2)


def softmax(logits: np.ndarray, axis: int = 0) -> np.ndarray:
    """Numerically stable softmax along ``axis``."""
    shifted = logits - logits.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


def sample_labels(probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw one class per pixel from the categorical distribution ``probs`` (C, N)."""
    cumulative = np.cumsum(probs, axis=0)
    draws = rng.random(probs.shape[1])[None, :]
    labels = (draws > cumulative).sum(axis=0)
    return np.clip(labels, 0, probs.shape[0] - 1)


# ------------------------------------------------------------------- apply_temperature


def test_apply_temperature_identity_at_one() -> None:
    rng = np.random.default_rng(0)
    probs = softmax(rng.normal(size=(4, 8, 8)))

    out = apply_temperature(probs, 1.0)

    assert out.shape == probs.shape
    np.testing.assert_allclose(out, probs, atol=1e-6)


def test_apply_temperature_sums_to_one() -> None:
    rng = np.random.default_rng(1)
    probs = softmax(rng.normal(size=(4, 16)))

    for temperature in (0.5, 1.0, 2.0, 10.0):
        out = apply_temperature(probs, temperature)
        np.testing.assert_allclose(out.sum(axis=0), np.ones(16), atol=1e-9)


def test_apply_temperature_large_t_approaches_uniform() -> None:
    rng = np.random.default_rng(2)
    probs = softmax(rng.normal(size=(4, 32)) * 3.0)

    out = apply_temperature(probs, 1000.0)

    np.testing.assert_allclose(out, np.full_like(out, 0.25), atol=5e-3)


def test_apply_temperature_low_t_sharpens() -> None:
    probs = np.array([[0.6], [0.3], [0.1], [0.0]])

    out = apply_temperature(probs, 0.5)

    assert out[0, 0] > probs[0, 0]
    assert out[2, 0] < probs[2, 0]


def test_apply_temperature_matches_logit_scaling() -> None:
    rng = np.random.default_rng(3)
    logits = rng.normal(size=(4, 10))
    probs = softmax(logits)

    out = apply_temperature(probs, 1.7)

    np.testing.assert_allclose(out, softmax(logits / 1.7), atol=1e-6)


# --------------------------------------------------------------------- NllAccumulator


def test_nll_default_grid_spans_half_to_three() -> None:
    rng = np.random.default_rng(9)
    probs = softmax(rng.normal(size=(4, 200)))
    labels = rng.integers(0, 4, size=200)

    acc = NllAccumulator()
    acc.add(probs, labels)

    assert acc.best() in set(DEFAULT_GRID.tolist())
    assert DEFAULT_GRID[0] == pytest.approx(0.5)
    assert DEFAULT_GRID[-1] == pytest.approx(3.0)


def test_nll_ignores_out_of_range_labels() -> None:
    probs = np.array([[0.7, 0.2], [0.2, 0.3], [0.05, 0.3], [0.05, 0.2]])
    label = np.array([0, 255])

    with_ignored = NllAccumulator()
    with_ignored.add(probs, label)

    only_valid = NllAccumulator()
    only_valid.add(probs[:, :1], label[:1])

    np.testing.assert_allclose(with_ignored.sums, only_valid.sums, atol=1e-12)


def test_nll_accumulates_across_calls() -> None:
    rng = np.random.default_rng(4)
    probs = softmax(rng.normal(size=(4, 50)))
    label = rng.integers(0, 4, size=50)

    split = NllAccumulator()
    split.add(probs[:, :20], label[:20])
    split.add(probs[:, 20:], label[20:])

    whole = NllAccumulator()
    whole.add(probs, label)

    np.testing.assert_allclose(split.sums, whole.sums, atol=1e-9)
    assert split.best() == pytest.approx(whole.best())


def test_nll_too_soft_model_prefers_low_temperature() -> None:
    rng = np.random.default_rng(5)
    logits = rng.normal(size=(4, 20000)) * 2.0
    true_probs = softmax(logits)
    labels = sample_labels(true_probs, rng)
    # The model is too soft by a factor 2: probs = softmax(z / 2).
    probs = softmax(logits / 2.0)

    acc = NllAccumulator()
    acc.add(probs, labels)

    assert acc.best() == pytest.approx(0.5, abs=0.1)


def test_nll_calibrated_model_prefers_temperature_one() -> None:
    rng = np.random.default_rng(6)
    logits = rng.normal(size=(4, 20000)) * 2.0
    probs = softmax(logits)
    labels = sample_labels(probs, rng)

    acc = NllAccumulator()
    acc.add(probs, labels)

    assert acc.best() == pytest.approx(1.0, abs=0.15)


def test_nll_best_is_a_grid_value() -> None:
    rng = np.random.default_rng(7)
    probs = softmax(rng.normal(size=(4, 100)))
    labels = rng.integers(0, 4, size=100)

    acc = NllAccumulator()
    acc.add(probs, labels)

    assert acc.best() in set(DEFAULT_GRID.tolist())


# --------------------------------------------------------------------- EceAccumulator


def test_ece_perfect_one_hot_predictions_are_zero() -> None:
    probs = np.zeros((4, 6))
    label = np.array([1, 2, 3, 1, 2, 3])
    probs[label, np.arange(6)] = 1.0

    acc = EceAccumulator()
    acc.add(probs, label)

    assert acc.value() == pytest.approx(0.0, abs=1e-12)


def test_ece_hand_case_two_bins() -> None:
    # Bin 1 ([0, 0.5)): one pixel, confidence 0.4, wrong.
    # Bin 2 ([0.5, 1]): three pixels, mean confidence 0.7, one of three correct.
    probs = np.array(
        [
            [0.4, 0.1, 0.1, 0.1],
            [0.3, 0.8, 0.1, 0.1],
            [0.2, 0.05, 0.7, 0.1],
            [0.1, 0.05, 0.1, 0.6],
        ]
    )
    label = np.array([1, 1, 1, 1])

    acc = EceAccumulator(bins=2)
    acc.add(probs, label)

    bin1 = abs(0.0 - 0.4) * 1 / 4
    bin2 = abs(1 / 3 - 0.7) * 3 / 4
    assert acc.value() == pytest.approx(bin1 + bin2, abs=1e-9)
    assert acc.value() == pytest.approx(0.375, abs=1e-9)


def test_ece_uses_only_building_labels() -> None:
    probs = np.array(
        [
            [0.1, 0.9, 0.1],
            [0.8, 0.05, 0.1],
            [0.05, 0.05, 0.4],
            [0.05, 0.0, 0.4],
        ]
    )
    label = np.array([1, 0, 255])

    with_background = EceAccumulator(bins=2)
    with_background.add(probs, label)

    only_building = EceAccumulator(bins=2)
    only_building.add(probs[:, :1], label[:1])

    assert with_background.value() == pytest.approx(only_building.value(), abs=1e-12)


def test_ece_accumulates_across_calls() -> None:
    rng = np.random.default_rng(8)
    probs = softmax(rng.normal(size=(4, 60)))
    label = rng.integers(1, 4, size=60)

    split = EceAccumulator(bins=5)
    split.add(probs[:, :30], label[:30])
    split.add(probs[:, 30:], label[30:])

    whole = EceAccumulator(bins=5)
    whole.add(probs, label)

    assert split.value() == pytest.approx(whole.value(), abs=1e-12)


# ---------------------------------------------------------------------------- entropy


def test_entropy_one_hot_is_zero() -> None:
    probs = np.zeros((4, 3))
    probs[2, :] = 1.0

    out = entropy(probs)

    assert out.shape == (3,)
    np.testing.assert_allclose(out, np.zeros(3), atol=1e-9)


def test_entropy_uniform_is_log_classes() -> None:
    probs = np.full((4, 5), 0.25)

    out = entropy(probs)

    np.testing.assert_allclose(out, np.full(5, np.log(4)), atol=1e-12)


def test_entropy_matches_manual_formula() -> None:
    probs = np.array([[0.5, 0.1], [0.25, 0.2], [0.15, 0.3], [0.1, 0.4]])

    out = entropy(probs)

    expected = -(probs * np.log(probs)).sum(axis=0)
    np.testing.assert_allclose(out, expected, atol=1e-12)
    assert out[0] == pytest.approx(1.207974, abs=1e-6)
