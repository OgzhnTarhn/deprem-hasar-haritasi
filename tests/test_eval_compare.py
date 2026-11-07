"""Tests for the paired model comparison (``hasar.eval.compare``).

Protocol section 8 compares two models with a *paired* block bootstrap and section 9 decides
whether the difference is "eşit" or "farklı". Every test builds tiny 4x4 confusion matrices with
numpy, so no data files and no network are needed; ``n`` is kept at 300 so the file runs fast.
"""

from __future__ import annotations

import numpy as np
import pytest

from hasar.eval.compare import compare_models, compare_record, ensemble_tile
from hasar.eval.metrics import miou
from hasar.results import ResultRecord

N = 300

RESULT_KEYS = {"a_miou_pct", "b_miou_pct", "diff_pct", "low_pct", "high_pct", "verdict"}


def tile_matrix(seed: int, scale: int = 10) -> np.ndarray:
    """Return a deterministic 4x4 int64 confusion matrix with a strong diagonal."""
    rng = np.random.default_rng(seed)
    matrix = rng.integers(0, scale, size=(4, 4), dtype=np.int64)
    np.fill_diagonal(matrix, rng.integers(scale, 3 * scale, size=4))
    return matrix


def tiles(count: int = 12) -> dict[str, np.ndarray]:
    """Return ``count`` distinct tiles named ``t00``, ``t01``, ..."""
    return {f"t{i:02d}": tile_matrix(i) for i in range(count)}


def better_tiles(base: dict[str, np.ndarray], boost: int = 20) -> dict[str, np.ndarray]:
    """Return a copy of ``base`` with a much stronger diagonal on every tile."""
    out = {}
    for name, matrix in base.items():
        improved = matrix.copy()
        np.fill_diagonal(improved, np.diag(improved) + boost)
        out[name] = improved
    return out


# ------------------------------------------------------------------------- compare_models


def test_compare_models_returns_the_declared_keys() -> None:
    per_tile = tiles()

    result = compare_models(per_tile, dict(per_tile), n=N, seed=0)

    assert set(result) == RESULT_KEYS


def test_identical_models_have_zero_difference_and_are_equal() -> None:
    per_tile = tiles()

    result = compare_models(per_tile, dict(per_tile), n=N, seed=0)

    assert result["diff_pct"] == pytest.approx(0.0)
    assert result["a_miou_pct"] == pytest.approx(result["b_miou_pct"])
    assert result["low_pct"] == pytest.approx(0.0)
    assert result["high_pct"] == pytest.approx(0.0)
    assert result["verdict"] == "eşit"


def test_point_estimates_are_miou_in_percent() -> None:
    per_tile_a = tiles()
    per_tile_b = better_tiles(per_tile_a)

    result = compare_models(per_tile_a, per_tile_b, n=N, seed=0)

    expected_a = miou(sum(per_tile_a.values())) * 100.0
    expected_b = miou(sum(per_tile_b.values())) * 100.0
    assert result["a_miou_pct"] == pytest.approx(expected_a)
    assert result["b_miou_pct"] == pytest.approx(expected_b)
    assert result["diff_pct"] == pytest.approx(expected_b - expected_a)


def test_better_model_has_positive_difference_and_is_different() -> None:
    per_tile_a = tiles()
    per_tile_b = better_tiles(per_tile_a)

    result = compare_models(per_tile_a, per_tile_b, n=N, seed=0)

    assert result["diff_pct"] > 1.0
    assert result["low_pct"] > 0.0
    assert result["verdict"] == "farklı"


def test_worse_model_has_negative_difference() -> None:
    per_tile_a = tiles()
    per_tile_b = better_tiles(per_tile_a)

    result = compare_models(per_tile_b, per_tile_a, n=N, seed=0)

    assert result["diff_pct"] < -1.0
    assert result["high_pct"] < 0.0
    assert result["verdict"] == "farklı"


def test_interval_brackets_the_difference() -> None:
    per_tile_a = tiles()
    per_tile_b = better_tiles(per_tile_a, boost=2)

    result = compare_models(per_tile_a, per_tile_b, n=N, seed=0)

    assert result["low_pct"] <= result["diff_pct"] <= result["high_pct"]


def test_same_seed_gives_identical_result() -> None:
    per_tile_a = tiles()
    per_tile_b = better_tiles(per_tile_a, boost=2)

    first = compare_models(per_tile_a, per_tile_b, n=N, seed=7)
    second = compare_models(per_tile_a, per_tile_b, n=N, seed=7)

    assert first == second


def test_mismatched_tile_sets_raise_value_error() -> None:
    per_tile_a = tiles()
    per_tile_b = dict(per_tile_a)
    per_tile_b.pop("t00")

    with pytest.raises(ValueError):
        compare_models(per_tile_a, per_tile_b, n=N, seed=0)


def test_blocks_none_treats_every_tile_as_its_own_block() -> None:
    per_tile_a = tiles()
    per_tile_b = better_tiles(per_tile_a, boost=2)
    own_blocks = {name: ("all", name) for name in per_tile_a}

    default = compare_models(per_tile_a, per_tile_b, n=N, seed=0)
    explicit = compare_models(per_tile_a, per_tile_b, blocks=own_blocks, n=N, seed=0)

    assert default == explicit


def test_explicit_blocks_keep_the_point_estimate() -> None:
    per_tile_a = tiles()
    per_tile_b = better_tiles(per_tile_a, boost=2)
    blocks = {name: ("all", f"block-{i // 3}") for i, name in enumerate(sorted(per_tile_a))}

    result = compare_models(per_tile_a, per_tile_b, blocks=blocks, n=N, seed=0)

    expected_a = miou(sum(per_tile_a.values())) * 100.0
    expected_b = miou(sum(per_tile_b.values())) * 100.0
    assert result["a_miou_pct"] == pytest.approx(expected_a)
    assert result["b_miou_pct"] == pytest.approx(expected_b)
    assert result["diff_pct"] == pytest.approx(expected_b - expected_a)


# -------------------------------------------------------------------------- ensemble_tile


def constant_cache(value: float) -> dict[str, np.ndarray]:
    """Return a cache whose four probability maps are all ``value``."""
    return {
        "clf": np.full((4, 2, 2), value, dtype=np.float32),
        "loc": np.full((2, 2), value, dtype=np.float32),
        "clf_tta": np.full((4, 2, 2), value, dtype=np.float32),
        "loc_tta": np.full((2, 2), value, dtype=np.float32),
    }


def test_ensemble_tile_returns_weighted_mean_of_every_map() -> None:
    cache_a = constant_cache(0.2)
    cache_b = constant_cache(0.6)

    out = ensemble_tile(cache_a, cache_b, weight=0.5)

    assert set(out) == {"clf", "loc", "clf_tta", "loc_tta"}
    for name in out:
        np.testing.assert_allclose(out[name], np.full_like(cache_a[name], 0.4), rtol=0, atol=1e-6)


def test_ensemble_tile_output_is_float32() -> None:
    out = ensemble_tile(constant_cache(0.2), constant_cache(0.6), weight=0.5)

    for name, array in out.items():
        assert array.dtype == np.float32, name


def test_ensemble_tile_weight_zero_is_model_a() -> None:
    cache_a = constant_cache(0.2)
    cache_b = constant_cache(0.6)

    out = ensemble_tile(cache_a, cache_b, weight=0.0)

    for name in out:
        np.testing.assert_allclose(out[name], cache_a[name], rtol=0, atol=1e-6)


def test_ensemble_tile_weight_one_is_model_b() -> None:
    cache_a = constant_cache(0.2)
    cache_b = constant_cache(0.6)

    out = ensemble_tile(cache_a, cache_b, weight=1.0)

    for name in out:
        np.testing.assert_allclose(out[name], cache_b[name], rtol=0, atol=1e-6)


def test_ensemble_tile_default_weight_is_half() -> None:
    cache_a = constant_cache(0.2)
    cache_b = constant_cache(0.6)

    default = ensemble_tile(cache_a, cache_b)
    half = ensemble_tile(cache_a, cache_b, weight=0.5)

    for name in default:
        np.testing.assert_array_equal(default[name], half[name])


# ------------------------------------------------------------------------- compare_record


def comparison() -> dict[str, object]:
    """Return a comparison result of two models on the same tiles."""
    per_tile_a = tiles()
    per_tile_b = better_tiles(per_tile_a)
    return compare_models(per_tile_a, per_tile_b, n=N, seed=0)


def test_compare_record_uses_the_paired_difference_tag() -> None:
    record = compare_record(comparison(), "changemamba", "varyant", "abc123", "deadbeef")

    assert isinstance(record, ResultRecord)
    assert record.metric_tag == "[ZS-EVENT mIoU, eslestirilmis-fark]"


def test_compare_record_values_are_the_numeric_entries() -> None:
    result = comparison()

    record = compare_record(result, "changemamba", "varyant", "abc123", "deadbeef")

    numeric = {key: value for key, value in result.items() if isinstance(value, (int, float))}
    assert set(record.values) == set(numeric)
    for key, value in numeric.items():
        assert record.values[key] == pytest.approx(float(value))


def test_compare_record_extra_carries_verdict_and_models() -> None:
    result = comparison()

    record = compare_record(result, "changemamba", "varyant", "abc123", "deadbeef")

    assert record.extra["verdict"] == result["verdict"]
    assert record.extra["model_a"] == "changemamba"
    assert record.extra["model_b"] == "varyant"


def test_compare_record_keeps_provenance() -> None:
    record = compare_record(comparison(), "changemamba", "varyant", "abc123", "deadbeef")

    assert record.commit == "abc123"
    assert record.config_hash == "deadbeef"
    assert isinstance(record.split, str) and record.split
