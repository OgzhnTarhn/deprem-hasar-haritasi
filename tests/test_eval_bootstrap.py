"""Tests for the block bootstrap (``hasar.eval.bootstrap``).

Every test builds tiny 4x4 confusion matrices with numpy, so no data files and no network are
needed. ``n`` is kept small (200-500) so the whole file runs well under a second.
"""

from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from hasar.eval.bootstrap import (
    BootstrapResult,
    anchor_verdict,
    block_bootstrap,
    equal_verdict,
    paired_block_bootstrap,
    turkey_blocks,
)
from hasar.eval.metrics import miou


def tile_matrix(seed: int, scale: int = 10) -> np.ndarray:
    """Return a deterministic 4x4 int64 confusion matrix with a strong diagonal."""
    rng = np.random.default_rng(seed)
    matrix = rng.integers(0, scale, size=(4, 4), dtype=np.int64)
    np.fill_diagonal(matrix, rng.integers(scale, 3 * scale, size=4))
    return matrix


def heterogeneous_tiles(count: int = 12) -> dict[str, np.ndarray]:
    """Return ``count`` distinct tiles spread over two strata and several blocks."""
    tiles = {}
    for i in range(count):
        stratum = "A" if i % 2 == 0 else "B"
        tiles[f"{stratum}_{i:02d}"] = tile_matrix(i)
    return tiles


def heterogeneous_blocks(tiles: dict[str, np.ndarray]) -> dict[str, tuple[str, str]]:
    """Group tiles into blocks of two inside each stratum."""
    blocks = {}
    for i, name in enumerate(sorted(tiles)):
        stratum = name.split("_")[0]
        blocks[name] = (stratum, f"{stratum}-block-{i // 2}")
    return blocks


def total_pixels(matrix: np.ndarray) -> float:
    """A trivial metric: the number of pixels in class 0."""
    return float(matrix[0, 0])


# ------------------------------------------------------------------------ BootstrapResult


def test_bootstrap_result_is_frozen_dataclass() -> None:
    result = BootstrapResult(estimate=1.0, low=0.5, high=1.5, n=10, seed=3)

    assert result.estimate == 1.0
    assert result.low == 0.5
    assert result.high == 1.5
    assert result.n == 10
    assert result.seed == 3
    with pytest.raises(FrozenInstanceError):
        result.estimate = 2.0  # type: ignore[misc]


# ------------------------------------------------------------------------- block_bootstrap


def test_estimate_is_metric_of_summed_matrices() -> None:
    tiles = heterogeneous_tiles()
    blocks = heterogeneous_blocks(tiles)

    result = block_bootstrap(tiles, blocks, n=200, seed=0)

    expected = miou(sum(tiles.values()))
    assert result.estimate == pytest.approx(expected)
    assert result.n == 200
    assert result.seed == 0


def test_estimate_uses_custom_metric() -> None:
    tiles = heterogeneous_tiles()
    blocks = heterogeneous_blocks(tiles)

    result = block_bootstrap(tiles, blocks, metric=total_pixels, n=200, seed=0)

    assert result.estimate == pytest.approx(float(sum(tiles.values())[0, 0]))


def test_interval_brackets_estimate_on_heterogeneous_set() -> None:
    tiles = heterogeneous_tiles()
    blocks = heterogeneous_blocks(tiles)

    result = block_bootstrap(tiles, blocks, n=400, seed=1)

    assert result.low <= result.estimate <= result.high
    assert result.low < result.high


def test_same_seed_gives_identical_result() -> None:
    tiles = heterogeneous_tiles()
    blocks = heterogeneous_blocks(tiles)

    first = block_bootstrap(tiles, blocks, n=300, seed=7)
    second = block_bootstrap(tiles, blocks, n=300, seed=7)

    assert first == second


def test_different_seed_changes_interval() -> None:
    tiles = heterogeneous_tiles()
    blocks = heterogeneous_blocks(tiles)

    first = block_bootstrap(tiles, blocks, n=300, seed=0)
    second = block_bootstrap(tiles, blocks, n=300, seed=1)

    assert (first.low, first.high) != (second.low, second.high)


def test_single_block_per_stratum_gives_degenerate_interval() -> None:
    tiles = {
        "a1": tile_matrix(0),
        "a2": tile_matrix(1),
        "b1": tile_matrix(2),
    }
    blocks = {
        "a1": ("A", "A-only"),
        "a2": ("A", "A-only"),
        "b1": ("B", "B-only"),
    }

    result = block_bootstrap(tiles, blocks, metric=total_pixels, n=200, seed=0)

    assert result.low == result.estimate
    assert result.high == result.estimate


def test_resamples_never_mix_stratum_counts() -> None:
    # Stratum A: every block contributes 10 pixels of class 0; stratum B: 1 pixel.
    tiles = {}
    blocks = {}
    for i in range(3):
        matrix = np.zeros((4, 4), dtype=np.int64)
        matrix[0, 0] = 10
        tiles[f"a{i}"] = matrix
        blocks[f"a{i}"] = ("A", f"A-{i}")
    for i in range(2):
        matrix = np.zeros((4, 4), dtype=np.int64)
        matrix[0, 0] = 1
        tiles[f"b{i}"] = matrix
        blocks[f"b{i}"] = ("B", f"B-{i}")

    result = block_bootstrap(tiles, blocks, metric=total_pixels, n=300, seed=0)

    # Every resample draws exactly 3 A-blocks and 2 B-blocks, so the metric is constant.
    assert result.estimate == pytest.approx(10 * 3 + 1 * 2)
    assert result.low == pytest.approx(10 * 3 + 1 * 2)
    assert result.high == pytest.approx(10 * 3 + 1 * 2)


def test_missing_block_entry_raises_key_error() -> None:
    tiles = heterogeneous_tiles(4)
    blocks = heterogeneous_blocks(tiles)
    del blocks["A_00"]

    with pytest.raises(KeyError):
        block_bootstrap(tiles, blocks, n=10, seed=0)


def test_empty_tiles_raise_value_error() -> None:
    with pytest.raises(ValueError):
        block_bootstrap({}, {}, n=10, seed=0)


# ----------------------------------------------------------------- paired_block_bootstrap


def test_paired_identical_models_give_zero_difference() -> None:
    tiles = heterogeneous_tiles()
    blocks = heterogeneous_blocks(tiles)

    result = paired_block_bootstrap(tiles, dict(tiles), blocks, n=300, seed=0)

    assert result.estimate == pytest.approx(0.0)
    assert result.low == pytest.approx(0.0)
    assert result.high == pytest.approx(0.0)


def test_paired_better_model_has_positive_difference() -> None:
    tiles_a = heterogeneous_tiles()
    tiles_b = {name: matrix.copy() for name, matrix in tiles_a.items()}
    for matrix in tiles_b.values():
        matrix[0, 0] += 3
    blocks = heterogeneous_blocks(tiles_a)

    result = paired_block_bootstrap(tiles_a, tiles_b, blocks, metric=total_pixels, n=300, seed=0)

    expected = 3.0 * len(tiles_a)
    assert result.estimate == pytest.approx(expected)
    assert result.estimate > 0
    assert result.low == pytest.approx(expected)
    assert result.high == pytest.approx(expected)


def test_paired_mismatched_keys_raise_value_error() -> None:
    tiles_a = heterogeneous_tiles(4)
    tiles_b = dict(tiles_a)
    tiles_b.pop("A_00")
    blocks = heterogeneous_blocks(tiles_a)

    with pytest.raises(ValueError):
        paired_block_bootstrap(tiles_a, tiles_b, blocks, n=10, seed=0)


# ------------------------------------------------------------------------------ verdicts


def test_equal_verdict_small_difference_is_equal() -> None:
    assert equal_verdict(0.5, -0.2, 1.2) == "eşit"


def test_equal_verdict_clear_difference() -> None:
    assert equal_verdict(2.0, 1.0, 3.0) == "farklı"


def test_equal_verdict_interval_containing_zero_is_equal() -> None:
    assert equal_verdict(2.0, -0.5, 4.0) == "eşit"


def test_anchor_verdict_anchor_inside_interval_is_equal() -> None:
    assert anchor_verdict(41.5, 40.0, 43.0) == "eşit"


def test_anchor_verdict_above_anchor() -> None:
    assert anchor_verdict(45.0, 43.5, 46.5) == "üstünde"


def test_anchor_verdict_below_anchor() -> None:
    assert anchor_verdict(38.0, 36.5, 39.5) == "altında"


# -------------------------------------------------------------------------- turkey_blocks


def write_research_json(path: Path) -> Path:
    payload = {
        "tile_centers": [
            {"name": "tile_x1", "lat": 37.0, "lon": 37.0},
            {"name": "tile_x2", "lat": 37.001, "lon": 37.001},
            {"name": "tile_y1", "lat": 39.0, "lon": 39.0},
            {"name": "tile_y2", "lat": 39.001, "lon": 39.001},
        ],
        "clusters": [
            {"center_lat": 37.0, "center_lon": 37.0, "nearest_place": "X"},
            {"center_lat": 39.0, "center_lon": 39.0, "nearest_place": "Y"},
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_turkey_blocks_assigns_nearest_place(tmp_path: Path) -> None:
    path = write_research_json(tmp_path / "research.json")

    blocks = turkey_blocks(path)

    assert set(blocks) == {"tile_x1", "tile_x2", "tile_y1", "tile_y2"}
    assert blocks["tile_x1"][0] == "X"
    assert blocks["tile_x2"][0] == "X"
    assert blocks["tile_y1"][0] == "Y"
    assert blocks["tile_y2"][0] == "Y"


def test_turkey_blocks_groups_nearby_tiles(tmp_path: Path) -> None:
    path = write_research_json(tmp_path / "research.json")

    blocks = turkey_blocks(path)

    assert blocks["tile_x1"][1] == blocks["tile_x2"][1]
    assert blocks["tile_y1"][1] == blocks["tile_y2"][1]
    assert blocks["tile_x1"][1] != blocks["tile_y1"][1]


def test_turkey_blocks_uses_grid_ids(tmp_path: Path) -> None:
    path = write_research_json(tmp_path / "research.json")

    blocks = turkey_blocks(path)

    lat0 = (37.0 + 37.001 + 39.0 + 39.001) / 4
    for name, (_, block_id) in blocks.items():
        lat = {"tile_x1": 37.0, "tile_x2": 37.001, "tile_y1": 39.0, "tile_y2": 39.001}[name]
        lon = lat
        gy = math.floor(lat * 111 / 2.0)
        gx = math.floor(lon * 111 * math.cos(math.radians(lat0)) / 2.0)
        assert block_id == f"{gy}_{gx}"
