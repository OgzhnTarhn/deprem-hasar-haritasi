"""Tests for the qualitative gallery helpers (``hasar.analysis.gallery``).

Everything is tiny synthetic numpy data, so no image files and no network are needed. The panel
test checks the output geometry and a few pixel colours; the ranking and grouping tests use small
hand-written 4x4 confusion matrices.
"""

from __future__ import annotations

import numpy as np
import pytest

from hasar.analysis.gallery import PALETTE, colorize, group_summary, panel, rank_tiles
from hasar.eval.metrics import summary

PERFECT = np.diag([10, 10, 10, 10]).astype(np.int64)
# Every building pixel is predicted as background, so the three building IoUs are zero.
ALL_BACKGROUND = np.array(
    [
        [10, 0, 0, 0],
        [5, 0, 0, 0],
        [5, 0, 0, 0],
        [5, 0, 0, 0],
    ],
    dtype=np.int64,
)
# No building pixels at all: rows 1..3 are empty, the tile must be skipped by ``rank_tiles``.
NO_BUILDING = np.array(
    [
        [10, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ],
    dtype=np.int64,
)


def test_palette_has_every_class_with_expected_colours():
    assert set(PALETTE) == {0, 1, 2, 3, 255}
    assert tuple(PALETTE[0]) == (0, 0, 0)

    ignored = PALETTE[255]
    assert ignored[0] == ignored[1] == ignored[2]

    intact = PALETTE[1]
    damaged = PALETTE[2]
    destroyed = PALETTE[3]

    # Intact is a green with good contrast: green dominates both other channels.
    assert intact[1] > intact[0] and intact[1] > intact[2]
    # Damaged is orange: red > green > blue.
    assert damaged[0] > damaged[1] > damaged[2]
    # Destroyed is red: red dominates both other channels.
    assert destroyed[0] > destroyed[1] and destroyed[0] > destroyed[2]


def test_colorize_uses_palette_and_keeps_shape():
    label = np.array(
        [
            [0, 1, 2, 3],
            [255, 1, 0, 3],
        ],
        dtype=np.int64,
    )
    rgb = colorize(label)

    assert rgb.shape == (2, 4, 3)
    assert rgb.dtype == np.uint8
    for row in range(label.shape[0]):
        for col in range(label.shape[1]):
            assert tuple(rgb[row, col]) == tuple(PALETTE[int(label[row, col])])


def test_panel_geometry_and_pixel_colours():
    size = 16
    pre = np.zeros((8, 8, 3), dtype=np.uint8)
    pre[:, :] = (10, 20, 30)
    post = np.full((8, 8), 200, dtype=np.uint8)
    label = np.full((8, 8), 3, dtype=np.int64)
    pred = np.full((8, 8), 2, dtype=np.int64)

    image = panel(pre, post, label, pred, size=size)

    assert image.size == (4 * size, size)
    assert image.mode == "RGB"

    pixels = np.asarray(image)
    assert pixels.shape == (size, 4 * size, 3)
    y = size // 2

    def tile_centre(index: int) -> tuple[int, int]:
        return y, index * size + size // 2

    # Tile 0: the pre-disaster optical image, kept as is.
    assert tuple(pixels[tile_centre(0)]) == (10, 20, 30)
    # Tile 1: the post-disaster SAR image rendered as grey.
    post_pixel = pixels[tile_centre(1)]
    assert post_pixel[0] == post_pixel[1] == post_pixel[2] == 200
    # Tile 2: the label colourised with the palette.
    assert tuple(pixels[tile_centre(2)]) == tuple(PALETTE[3])
    # Tile 3: the prediction colourised with the palette.
    assert tuple(pixels[tile_centre(3)]) == tuple(PALETTE[2])


def test_panel_default_size_is_256():
    pre = np.zeros((4, 4, 3), dtype=np.uint8)
    post = np.zeros((4, 4), dtype=np.uint8)
    label = np.zeros((4, 4), dtype=np.int64)
    pred = np.zeros((4, 4), dtype=np.int64)

    image = panel(pre, post, label, pred)

    assert image.size == (4 * 256, 256)


def test_rank_tiles_worst_first_and_skips_tiles_without_buildings():
    per_tile = {
        "perfect": PERFECT,
        "bad": ALL_BACKGROUND,
        "empty": NO_BUILDING,
    }

    ranked = rank_tiles(per_tile, k=10)

    assert [name for name, _ in ranked] == ["bad", "perfect"]
    assert ranked[0][1] == pytest.approx(0.0)
    assert ranked[1][1] == pytest.approx(1.0)


def test_rank_tiles_best_first_and_respects_k():
    per_tile = {
        "perfect": PERFECT,
        "bad": ALL_BACKGROUND,
        "empty": NO_BUILDING,
    }

    ranked = rank_tiles(per_tile, worst=False, k=1)

    assert len(ranked) == 1
    assert ranked[0][0] == "perfect"
    assert ranked[0][1] == pytest.approx(1.0)


def test_rank_tiles_uses_the_requested_key():
    per_tile = {
        "perfect": PERFECT,
        "bad": ALL_BACKGROUND,
    }

    ranked = rank_tiles(per_tile, key="miou", worst=True, k=10)

    assert [name for name, _ in ranked] == ["bad", "perfect"]
    assert ranked[0][1] == pytest.approx(summary(ALL_BACKGROUND)["miou"])
    assert ranked[1][1] == pytest.approx(summary(PERFECT)["miou"])


def test_group_summary_sums_matrices_per_group_and_collects_unknown_tiles():
    per_tile = {
        "a": PERFECT,
        "b": ALL_BACKGROUND,
        "c": PERFECT,
        "d": NO_BUILDING,
    }
    groups = {"a": "capella", "b": "capella", "c": "umbra"}

    result = group_summary(per_tile, groups)

    assert set(result) == {"capella", "umbra", "bilinmiyor"}

    capella = result["capella"]
    assert capella["tiles"] == 2
    expected = summary(PERFECT + ALL_BACKGROUND)
    for key, value in expected.items():
        assert capella[key] == pytest.approx(value)

    umbra = result["umbra"]
    assert umbra["tiles"] == 1
    assert umbra["building_miou"] == pytest.approx(summary(PERFECT)["building_miou"])

    unknown = result["bilinmiyor"]
    assert unknown["tiles"] == 1
    assert unknown["pixels"] == int(NO_BUILDING.sum())
