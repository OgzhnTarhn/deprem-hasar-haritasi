"""Tests for the transductive Y3 analysis (``hasar.analysis.transductive``).

Rule 1's second exception: Turkey's *unlabeled* SAR statistics may re-standardise Turkey SAR for
the separate "transductive" table. Everything here is tiny synthetic numpy data; no dataset, no
network, no Turkey imagery or labels.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from hasar.analysis.transductive import (
    TRANSDUCTIVE_QUALIFIER,
    MomentAccumulator,
    Moments,
    metric_tag,
    restandardize,
)
from hasar.results import validate_metric_tag


def _moments(values: np.ndarray) -> tuple[float, float]:
    """Population mean and standard deviation (ddof=0) of a 1-D array."""
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=0))


def test_moments_is_frozen_dataclass() -> None:
    assert dataclasses.is_dataclass(Moments)
    moments = Moments(mean=1.5, std=2.5, count=7)
    assert moments.mean == 1.5
    assert moments.std == 2.5
    assert moments.count == 7
    with pytest.raises(dataclasses.FrozenInstanceError):
        moments.mean = 0.0  # type: ignore[misc]


def test_accumulator_matches_numpy_over_two_chunks() -> None:
    rng = np.random.default_rng(0)
    first = rng.integers(1, 256, size=(4, 5), dtype=np.uint8)
    second = rng.integers(1, 256, size=(3, 7), dtype=np.uint8)

    accumulator = MomentAccumulator()
    accumulator.add(first, np.ones_like(first, dtype=bool))
    accumulator.add(second, np.ones_like(second, dtype=bool))
    result = accumulator.result()

    expected_mean, expected_std = _moments(np.concatenate([first.ravel(), second.ravel()]))
    assert result.count == first.size + second.size
    assert result.mean == pytest.approx(expected_mean, rel=1e-9)
    assert result.std == pytest.approx(expected_std, rel=1e-9)


def test_accumulator_ignores_invalid_pixels() -> None:
    post = np.array([[10, 20, 0], [30, 0, 40]], dtype=np.uint8)
    valid = np.array([[True, True, False], [True, False, True]])

    accumulator = MomentAccumulator()
    accumulator.add(post, valid)
    result = accumulator.result()

    expected_mean, expected_std = _moments(np.array([10, 20, 30, 40], dtype=np.float64))
    assert result.count == 4
    assert result.mean == pytest.approx(expected_mean, rel=1e-9)
    assert result.std == pytest.approx(expected_std, rel=1e-9)


def test_accumulator_empty_raises() -> None:
    accumulator = MomentAccumulator()
    with pytest.raises(ValueError):
        accumulator.result()


def test_accumulator_all_invalid_raises() -> None:
    post = np.zeros((2, 2), dtype=np.uint8)
    accumulator = MomentAccumulator()
    accumulator.add(post, np.zeros_like(post, dtype=bool))
    with pytest.raises(ValueError):
        accumulator.result()


def test_restandardize_matches_target_moments() -> None:
    rng = np.random.default_rng(1)
    post = rng.integers(10, 71, size=(8, 9), dtype=np.uint8)
    post[0, :3] = 0  # nodata pixels
    valid = post != 0

    source_mean, source_std = _moments(post[valid])
    source = Moments(mean=source_mean, std=source_std, count=int(valid.sum()))
    target = Moments(mean=100.0, std=30.0, count=int(valid.sum()))

    out = restandardize(post, source, target)

    assert out.shape == post.shape
    assert out.dtype == np.float32
    assert np.all(out[~valid] == 0.0)
    out_mean, out_std = _moments(out[valid])
    assert out_mean == pytest.approx(target.mean, rel=0.01)
    assert out_std == pytest.approx(target.std, rel=0.01)


def test_restandardize_formula_and_clipping() -> None:
    post = np.array([[0, 10, 50, 255]], dtype=np.uint8)
    source = Moments(mean=50.0, std=20.0, count=3)
    target = Moments(mean=100.0, std=40.0, count=3)

    out = restandardize(post, source, target)

    assert out[0, 0] == 0.0
    assert out[0, 1] == pytest.approx(20.0)
    assert out[0, 2] == pytest.approx(100.0)
    assert out[0, 3] == pytest.approx(255.0)  # clipped from 510


def test_transductive_qualifier_and_metric_tag() -> None:
    assert TRANSDUCTIVE_QUALIFIER == "transduktif-y3"
    assert metric_tag() == "[ZS-EVENT mIoU, transduktif-y3]"
    validate_metric_tag(metric_tag())
