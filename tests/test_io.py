"""Tests for the tile reader (``hasar.data.io``).

Every test writes tiny synthetic GeoTIFFs into ``tmp_path``; nothing touches the network or the
real BRIGHT data set. The Turkey gate is exercised through ``hasar.guard``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import tifffile

from hasar.data.io import (
    GeoMeta,
    TilePaths,
    nodata_mask,
    read_geo_meta,
    read_label,
    read_tile,
    tile_paths,
)
from hasar.guard import TurkeyAccess, TurkeyLockError

TURKEY_NAME = "turkey-earthquake_00000001"


def write_tile_set(
    root: Path,
    name: str,
    *,
    pre: np.ndarray | None = None,
    post: np.ndarray | None = None,
    target: np.ndarray | None = None,
) -> TilePaths:
    """Write a synthetic BRIGHT tile set under ``root`` and return its paths."""
    if pre is None:
        pre = np.zeros((16, 16, 3), dtype=np.uint8)
    if post is None:
        post = np.zeros((16, 16), dtype=np.uint8)
    if target is None:
        target = np.zeros((16, 16), dtype=np.uint8)

    paths = tile_paths(root, name)
    for path, array in ((paths.pre, pre), (paths.post, post), (paths.target, target)):
        path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(path, array)
    return paths


def test_tile_paths_layout(tmp_path: Path) -> None:
    paths = tile_paths(tmp_path, "la_palma-volcano_00000001")

    assert isinstance(paths, TilePaths)
    assert paths.pre == tmp_path / "pre-event" / "la_palma-volcano_00000001_pre_disaster.tif"
    assert paths.post == tmp_path / "post-event" / "la_palma-volcano_00000001_post_disaster.tif"
    assert paths.target == tmp_path / "target" / "la_palma-volcano_00000001_building_damage.tif"


def test_tile_paths_is_frozen_dataclass() -> None:
    assert dataclasses.is_dataclass(TilePaths)
    assert [field.name for field in dataclasses.fields(TilePaths)] == ["pre", "post", "target"]

    paths = TilePaths(pre=Path("a"), post=Path("b"), target=Path("c"))
    with pytest.raises(dataclasses.FrozenInstanceError):
        paths.pre = Path("d")  # type: ignore[misc]


def test_read_tile_returns_arrays_unchanged(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    pre = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
    post = rng.integers(0, 256, size=(16, 16), dtype=np.uint8)
    target = rng.integers(0, 4, size=(16, 16), dtype=np.uint8)
    write_tile_set(tmp_path, "noto-earthquake_00000003", pre=pre, post=post, target=target)

    got_pre, got_post, got_label = read_tile(tmp_path, "noto-earthquake_00000003")

    assert got_pre.dtype == np.uint8
    assert got_post.dtype == np.uint8
    assert got_label.dtype == np.uint8
    assert got_pre.shape == (16, 16, 3)
    assert got_post.shape == (16, 16)
    assert got_label.shape == (16, 16)
    np.testing.assert_array_equal(got_pre, pre)
    np.testing.assert_array_equal(got_post, post)
    np.testing.assert_array_equal(got_label, target)


def test_read_tile_squeezes_single_channel_post(tmp_path: Path) -> None:
    post = np.arange(16 * 16, dtype=np.uint8).reshape(16, 16, 1)
    write_tile_set(tmp_path, "noto-earthquake_00000003", post=post)

    _, got_post, _ = read_tile(tmp_path, "noto-earthquake_00000003")

    assert got_post.shape == (16, 16)
    np.testing.assert_array_equal(got_post, post[:, :, 0])


def test_read_tile_rejects_non_three_channel_pre(tmp_path: Path) -> None:
    write_tile_set(tmp_path, "noto-earthquake_00000003", pre=np.zeros((16, 16, 4), dtype=np.uint8))

    with pytest.raises(ValueError):
        read_tile(tmp_path, "noto-earthquake_00000003")


def test_read_tile_rejects_mismatched_spatial_shapes(tmp_path: Path) -> None:
    write_tile_set(
        tmp_path,
        "noto-earthquake_00000003",
        pre=np.zeros((16, 16, 3), dtype=np.uint8),
        post=np.zeros((16, 16), dtype=np.uint8),
        target=np.zeros((8, 8), dtype=np.uint8),
    )

    with pytest.raises(ValueError):
        read_tile(tmp_path, "noto-earthquake_00000003")


def test_read_tile_missing_file(tmp_path: Path) -> None:
    write_tile_set(tmp_path, "noto-earthquake_00000003")
    tile_paths(tmp_path, "noto-earthquake_00000003").target.unlink()

    with pytest.raises(FileNotFoundError):
        read_tile(tmp_path, "noto-earthquake_00000003")


def test_read_tile_turkey_gate_before_disk(tmp_path: Path) -> None:
    # No files are written: the gate must fire before any disk access.
    with pytest.raises(TurkeyLockError):
        read_tile(tmp_path, TURKEY_NAME)


def test_read_tile_turkey_with_access(tmp_path: Path) -> None:
    pre = np.full((16, 16, 3), 5, dtype=np.uint8)
    post = np.full((16, 16), 7, dtype=np.uint8)
    target = np.full((16, 16), 2, dtype=np.uint8)
    write_tile_set(tmp_path, TURKEY_NAME, pre=pre, post=post, target=target)

    access = TurkeyAccess(tag="protokol-kilidi", commit="abc")
    got_pre, got_post, got_label = read_tile(tmp_path, TURKEY_NAME, access=access)

    np.testing.assert_array_equal(got_pre, pre)
    np.testing.assert_array_equal(got_post, post)
    np.testing.assert_array_equal(got_label, target)


def test_read_label_returns_label_only(tmp_path: Path) -> None:
    target = np.arange(16 * 16, dtype=np.uint8).reshape(16, 16) % 4
    write_tile_set(tmp_path, "noto-earthquake_00000003", target=target)

    got = read_label(tmp_path, "noto-earthquake_00000003")

    assert isinstance(got, np.ndarray)
    assert got.dtype == np.uint8
    assert got.shape == (16, 16)
    np.testing.assert_array_equal(got, target)


def test_read_label_turkey_gate_before_disk(tmp_path: Path) -> None:
    with pytest.raises(TurkeyLockError):
        read_label(tmp_path, TURKEY_NAME)


def test_read_label_turkey_with_access(tmp_path: Path) -> None:
    target = np.full((16, 16), 3, dtype=np.uint8)
    write_tile_set(tmp_path, TURKEY_NAME, target=target)

    access = TurkeyAccess(tag="protokol-kilidi", commit="abc")
    got = read_label(tmp_path, TURKEY_NAME, access=access)

    np.testing.assert_array_equal(got, target)


def test_nodata_mask_requires_all_optical_and_sar_zero() -> None:
    pre = np.zeros((2, 2, 3), dtype=np.uint8)
    post = np.zeros((2, 2), dtype=np.uint8)

    # (0, 0): all optical zero and SAR zero -> nodata.
    # (0, 1): optical zero but SAR 7 -> valid.
    pre[0, 1] = (0, 0, 0)
    post[0, 1] = 7
    # (1, 0): optical (0, 0, 1) and SAR 0 -> valid.
    pre[1, 0] = (0, 0, 1)
    # (1, 1): optical zero and SAR zero -> nodata.
    mask = nodata_mask(pre, post)

    assert mask.dtype == np.bool_
    assert mask.shape == (2, 2)
    np.testing.assert_array_equal(mask, np.array([[True, False], [False, True]]))


def test_geo_meta_bounds() -> None:
    meta = GeoMeta(
        width=10,
        height=20,
        origin_x=100.0,
        origin_y=200.0,
        pixel_x=0.5,
        pixel_y=0.25,
        epsg=32629,
    )

    assert meta.bounds() == (100.0, 200.0 - 20 * 0.25, 100.0 + 10 * 0.5, 200.0)


def test_geo_meta_is_frozen_dataclass() -> None:
    assert dataclasses.is_dataclass(GeoMeta)
    assert [field.name for field in dataclasses.fields(GeoMeta)] == [
        "width",
        "height",
        "origin_x",
        "origin_y",
        "pixel_x",
        "pixel_y",
        "epsg",
    ]

    meta = GeoMeta(width=1, height=1, origin_x=0.0, origin_y=0.0, pixel_x=1.0, pixel_y=1.0, epsg=None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        meta.width = 2  # type: ignore[misc]


def _write_geo_tiff(path: Path, geokeys: tuple[int, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.zeros((16, 16), dtype=np.uint8)
    tifffile.imwrite(
        path,
        array,
        extratags=[
            (33550, "d", 3, (0.5, 0.25, 0.0), True),
            (33922, "d", 6, (0.0, 0.0, 0.0, 100.0, 200.0, 0.0), True),
            (34735, "H", len(geokeys), geokeys, True),
        ],
    )


def test_read_geo_meta_projected(tmp_path: Path) -> None:
    path = tmp_path / "projected.tif"
    _write_geo_tiff(path, (1, 1, 0, 2, 1024, 0, 1, 1, 3072, 0, 1, 32629))

    meta = read_geo_meta(path)

    assert isinstance(meta, GeoMeta)
    assert meta.width == 16
    assert meta.height == 16
    assert meta.pixel_x == pytest.approx(0.5)
    assert meta.pixel_y == pytest.approx(0.25)
    assert meta.origin_x == pytest.approx(100.0)
    assert meta.origin_y == pytest.approx(200.0)
    assert meta.epsg == 32629


def test_read_geo_meta_geographic(tmp_path: Path) -> None:
    path = tmp_path / "geographic.tif"
    _write_geo_tiff(path, (1, 1, 0, 2, 1024, 0, 1, 2, 2048, 0, 1, 4326))

    meta = read_geo_meta(path)

    assert meta.epsg == 4326


def test_read_geo_meta_without_tags(tmp_path: Path) -> None:
    path = tmp_path / "plain.tif"
    tifffile.imwrite(path, np.zeros((16, 16), dtype=np.uint8))

    with pytest.raises(ValueError):
        read_geo_meta(path)
