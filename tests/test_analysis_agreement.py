"""Tests for the reference-agreement analysis (``hasar.analysis.agreement``).

Y1 asks how much the BRIGHT optical label, the HOT/OSM destroyed-building layer and the
learning-free ARIA damage proxy map (DPM) agree on the same buildings. Everything here is
synthetic: numpy arrays and a hand-written :class:`GeoMeta`, no network and no real tiles.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from hasar.analysis.agreement import (
    Agreement,
    agreement,
    crs_to_pixel,
    dpm_flags,
    pixel_to_crs,
    rasterize_polygons_crs,
    sample_points_mask,
)
from hasar.data.io import GeoMeta


def make_meta(
    *,
    width: int = 10,
    height: int = 10,
    origin_x: float = 0.0,
    origin_y: float = 10.0,
    pixel_x: float = 1.0,
    pixel_y: float = 1.0,
) -> GeoMeta:
    """North-up tile header: origin is the top-left corner, ``y`` grows downwards."""
    return GeoMeta(
        width=width,
        height=height,
        origin_x=origin_x,
        origin_y=origin_y,
        pixel_x=pixel_x,
        pixel_y=pixel_y,
        epsg=32636,
    )


def cohen_kappa(tt: int, tf: int, ft: int, ff: int) -> float:
    """Cohen's kappa of a 2x2 table, computed exactly from the counts."""
    n = tt + tf + ft + ff
    po = (tt + ff) / n
    pe = ((tt + tf) * (tt + ft) + (ft + ff) * (tf + ff)) / n**2
    return (po - pe) / (1 - pe)


# --------------------------------------------------------------------------- CRS <-> pixel


def test_crs_to_pixel_north_up() -> None:
    meta = make_meta(width=100, height=80, origin_x=500000.0, origin_y=4200000.0, pixel_x=0.5, pixel_y=0.5)

    col, row = crs_to_pixel(meta, 500010.0, 4199990.0)

    assert float(col) == pytest.approx(20.0)
    assert float(row) == pytest.approx(20.0)


def test_crs_to_pixel_pixel_centres_are_half_integers() -> None:
    meta = make_meta(width=4, height=4, origin_x=0.0, origin_y=4.0, pixel_x=1.0, pixel_y=1.0)

    col, row = crs_to_pixel(meta, 0.5, 3.5)

    assert float(col) == pytest.approx(0.5)
    assert float(row) == pytest.approx(0.5)


def test_crs_to_pixel_accepts_arrays() -> None:
    meta = make_meta(width=10, height=10, origin_x=0.0, origin_y=10.0, pixel_x=1.0, pixel_y=1.0)

    col, row = crs_to_pixel(meta, np.array([0.5, 2.5]), np.array([9.5, 7.5]))

    assert isinstance(col, np.ndarray)
    assert isinstance(row, np.ndarray)
    np.testing.assert_allclose(col, [0.5, 2.5])
    np.testing.assert_allclose(row, [0.5, 2.5])


def test_pixel_to_crs_round_trip() -> None:
    meta = make_meta(width=100, height=80, origin_x=500000.0, origin_y=4200000.0, pixel_x=0.5, pixel_y=0.5)

    x, y = pixel_to_crs(meta, 20.0, 20.0)
    assert float(x) == pytest.approx(500010.0)
    assert float(y) == pytest.approx(4199990.0)

    col, row = crs_to_pixel(meta, x, y)
    assert float(col) == pytest.approx(20.0)
    assert float(row) == pytest.approx(20.0)


def test_pixel_to_crs_round_trip_arrays() -> None:
    meta = make_meta(width=10, height=10, origin_x=1000.0, origin_y=2000.0, pixel_x=2.0, pixel_y=2.0)
    rows = np.array([0.0, 1.5, 9.0])
    cols = np.array([0.0, 2.5, 4.0])

    x, y = pixel_to_crs(meta, rows, cols)
    col_back, row_back = crs_to_pixel(meta, x, y)

    np.testing.assert_allclose(col_back, cols)
    np.testing.assert_allclose(row_back, rows)


# --------------------------------------------------------------------------- rasterize


def test_rasterize_square_covers_top_left_quarter() -> None:
    meta = make_meta(width=10, height=10, origin_x=1000.0, origin_y=2000.0, pixel_x=2.0, pixel_y=2.0)
    # x: 1000..1010 (5 columns), y: 2000..1990 (5 rows) -> the top-left quarter.
    square = np.array([[1000.0, 2000.0], [1010.0, 2000.0], [1010.0, 1990.0], [1000.0, 1990.0]])

    mask = rasterize_polygons_crs([square], meta)

    assert mask.shape == (10, 10)
    assert mask.dtype == np.bool_
    assert mask[:5, :5].all()
    assert not mask[5:, :].any()
    assert not mask[:, 5:].any()


def test_rasterize_empty_polygon_list_is_all_false() -> None:
    meta = make_meta(width=6, height=4)

    mask = rasterize_polygons_crs([], meta)

    assert mask.shape == (4, 6)
    assert not mask.any()


def test_rasterize_union_of_two_polygons() -> None:
    meta = make_meta(width=10, height=10, origin_x=0.0, origin_y=10.0, pixel_x=1.0, pixel_y=1.0)
    left = np.array([[0.0, 10.0], [2.0, 10.0], [2.0, 8.0], [0.0, 8.0]])
    right = np.array([[8.0, 10.0], [10.0, 10.0], [10.0, 8.0], [8.0, 8.0]])

    mask = rasterize_polygons_crs([left, right], meta)

    assert mask[:2, :2].all()
    assert mask[:2, 8:].all()
    assert int(mask.sum()) == 8


# --------------------------------------------------------------------------- sample points


def test_sample_points_mask_marks_containing_pixels() -> None:
    meta = make_meta(width=10, height=10, origin_x=0.0, origin_y=10.0, pixel_x=1.0, pixel_y=1.0)
    points = np.array([[0.5, 9.5], [2.5, 7.5]])

    mask = sample_points_mask(points, meta, (10, 10))

    assert mask.shape == (10, 10)
    assert mask.dtype == np.bool_
    assert mask[0, 0]
    assert mask[2, 2]
    assert int(mask.sum()) == 2


def test_sample_points_mask_ignores_out_of_tile_points() -> None:
    meta = make_meta(width=10, height=10, origin_x=0.0, origin_y=10.0, pixel_x=1.0, pixel_y=1.0)
    points = np.array([[0.5, 9.5], [-1.0, 5.0], [10.5, 5.0], [5.0, 11.0], [5.0, -1.0]])

    mask = sample_points_mask(points, meta, (10, 10))

    assert mask[0, 0]
    assert int(mask.sum()) == 1


# --------------------------------------------------------------------------- dpm_flags


def test_dpm_flags_threshold_and_nan() -> None:
    values = np.array([0.2, 0.6, 0.8, np.nan])

    flags = dpm_flags(values, 0.5)

    assert flags.dtype == np.bool_
    np.testing.assert_array_equal(flags, [False, True, True, False])


def test_dpm_flags_all_nan_is_all_false() -> None:
    values = np.array([np.nan, np.nan])

    flags = dpm_flags(values, 0.0)

    assert not flags.any()


# --------------------------------------------------------------------------- agreement


def test_agreement_dataclass_is_frozen() -> None:
    result = agreement(
        np.array([True, False]),
        np.array([True, False]),
        np.array([True, False]),
    )

    assert isinstance(result, Agreement)
    assert result.n == 2
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.n = 3  # type: ignore[misc]


def test_agreement_perfect_agreement() -> None:
    bright = np.array([True, True, False, False])
    osm = np.array([True, True, False, False])
    dpm = np.array([True, True, False, False])

    result = agreement(bright, osm, dpm)

    assert result.n == 4
    assert result.table["bright-osm"] == {"tt": 2, "tf": 0, "ft": 0, "ff": 2}
    assert result.table["bright-dpm"] == {"tt": 2, "tf": 0, "ft": 0, "ff": 2}
    assert result.table["osm-dpm"] == {"tt": 2, "tf": 0, "ft": 0, "ff": 2}
    assert result.kappa["bright-osm"] == pytest.approx(1.0)
    assert result.kappa["bright-dpm"] == pytest.approx(1.0)
    assert result.kappa["osm-dpm"] == pytest.approx(1.0)
    assert result.rates["all_agree"] == pytest.approx(1.0)


def test_agreement_hand_checked_kappa() -> None:
    # bright-osm table: tt=20, tf=5, ft=10, ff=65.
    bright = np.array([True] * 25 + [False] * 75)
    osm = np.array([True] * 20 + [False] * 5 + [True] * 10 + [False] * 65)
    dpm = np.zeros(100, dtype=bool)

    result = agreement(bright, osm, dpm)

    assert result.n == 100
    assert result.table["bright-osm"] == {"tt": 20, "tf": 5, "ft": 10, "ff": 65}

    expected = cohen_kappa(20, 5, 10, 65)
    # Standard Cohen's kappa for these counts: po = 0.85, pe = 0.60 -> 0.625.
    assert expected == pytest.approx(0.625)
    assert result.kappa["bright-osm"] == pytest.approx(expected)

    # dpm is all False: bright-dpm has pe == po, so kappa is undefined and reported as 0.0.
    assert result.table["bright-dpm"] == {"tt": 0, "tf": 25, "ft": 0, "ff": 75}
    assert result.kappa["bright-dpm"] == pytest.approx(0.0)

    # Only the 65 buildings where bright, osm and dpm are all False agree.
    assert result.rates["all_agree"] == pytest.approx(0.65)


def test_agreement_table_counts_sum_to_n() -> None:
    rng = np.random.default_rng(0)
    bright = rng.random(50) < 0.4
    osm = rng.random(50) < 0.5
    dpm = rng.random(50) < 0.3

    result = agreement(bright, osm, dpm)

    for pair in ("bright-osm", "bright-dpm", "osm-dpm"):
        counts = result.table[pair]
        assert set(counts) == {"tt", "tf", "ft", "ff"}
        assert sum(counts.values()) == result.n == 50


def test_agreement_without_dpm_has_only_two_reference_pairs() -> None:
    bright = np.array([True, True, False, False])
    osm = np.array([True, False, True, False])

    result = agreement(bright, osm, None)

    assert set(result.table) == {"bright-osm"}
    assert set(result.kappa) == {"bright-osm"}
    assert result.table["bright-osm"] == {"tt": 1, "tf": 1, "ft": 1, "ff": 1}
    assert result.kappa["bright-osm"] == pytest.approx(0.0)
    assert "model_vs_majority" not in result.rates
    # All three references are not available; only bright and osm must agree.
    assert result.rates["all_agree"] == pytest.approx(0.5)


def test_agreement_undefined_kappa_is_zero() -> None:
    bright = np.array([False, False, False])
    osm = np.array([False, False, False])

    result = agreement(bright, osm, None)

    assert result.kappa["bright-osm"] == pytest.approx(0.0)


def test_agreement_with_model_adds_pairs_and_majority_rate() -> None:
    bright = np.array([True, True, True, False])
    osm = np.array([True, True, False, False])
    dpm = np.array([True, False, False, False])
    model = np.array([True, False, False, True])

    result = agreement(bright, osm, dpm, model)

    assert set(result.table) == {
        "bright-osm",
        "bright-dpm",
        "osm-dpm",
        "model-bright",
        "model-osm",
        "model-dpm",
    }
    assert set(result.kappa) == set(result.table)
    assert result.table["model-bright"] == {"tt": 1, "tf": 1, "ft": 2, "ff": 0}
    # Majority of the three references: T, T, F, F; the model matches on buildings 0 and 2.
    assert result.rates["model_vs_majority"] == pytest.approx(0.5)
    # all_agree only looks at the references, not at the model: buildings 0 and 3.
    assert result.rates["all_agree"] == pytest.approx(0.5)


def test_agreement_majority_tie_counts_as_not_destroyed() -> None:
    # Two references disagree -> tie -> majority is "not destroyed".
    bright = np.array([True, True])
    osm = np.array([False, False])

    agrees = agreement(bright, osm, None, np.array([False, False]))
    disagrees = agreement(bright, osm, None, np.array([True, True]))

    assert agrees.rates["model_vs_majority"] == pytest.approx(1.0)
    assert disagrees.rates["model_vs_majority"] == pytest.approx(0.0)
