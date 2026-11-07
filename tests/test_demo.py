"""Tests for the static demo site builder (``hasar.demo``).

The demo is built from the single-inference cache only: no model is run and nothing touches the
network. Every test writes two tiny synthetic Turkey tiles as EPSG 4326 GeoTIFFs plus their cache
``.npz`` files into ``tmp_path``, then checks the PNGs, the GeoJSON and the Leaflet page.

The module under test does not exist yet: the import at the top is the expected red.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile
from PIL import Image

from hasar.data.io import GeoMeta, tile_paths
from hasar.demo import build_demo, tile_footprint_lonlat
from hasar.guard import TurkeyAccess

SIZE = 32
NUM_CLF = 4
NUM_LOC = 2
NAMES = ["turkey-earthquake_00000001", "turkey-earthquake_00000002"]
ACCESS = TurkeyAccess(tag="protokol-kilidi", commit="abc")
THUMB = 256
DESTROYED_PX = 16
DAMAGED_PX = 16

#: GeoTIFF tags of a 16x16 EPSG 4326 tile: pixel scale, tiepoint and two geokeys
#: (GTModelTypeGeoKey = geographic, GeographicTypeGeoKey = 4326).
GEO_TAGS = [
    (33550, "d", 3, (0.5, 0.25, 0.0), True),
    (33922, "d", 6, (0.0, 0.0, 0.0, 29.0, 41.0, 0.0), True),
    (34735, "H", 12, (1, 1, 0, 2, 1024, 0, 1, 2, 2048, 0, 1, 4326), True),
]


def make_tile(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a synthetic ``(pre, post, label)`` tile with a nodata corner."""
    rng = np.random.default_rng(seed)
    label = rng.integers(0, 4, size=(SIZE, SIZE), dtype=np.uint8)

    pre = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    pre[..., 0] = label * 10
    pre[..., 1] = 1
    pre[..., 2] = 2
    post = (label * 10 + 100).astype(np.uint8)

    pre[:4, :4] = 0
    post[:4, :4] = 0
    label[:4, :4] = 2
    return pre, post, label


def write_tile(root: Path, name: str, seed: int = 0) -> None:
    """Write a synthetic EPSG 4326 GeoTIFF tile under ``root`` in the ``tile_paths`` layout."""
    pre, post, label = make_tile(seed)
    paths = tile_paths(root, name)
    for path, array in ((paths.pre, pre), (paths.post, post), (paths.target, label)):
        path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(path, array, extratags=GEO_TAGS)


def make_cache(seed: int = 0) -> dict[str, np.ndarray]:
    """Return a cache entry whose argmax has a fixed destroyed and damaged block."""
    rng = np.random.default_rng(seed)
    clf = np.zeros((NUM_CLF, SIZE, SIZE), dtype=np.float16)
    # A 4x4 destroyed block and a 4x4 damaged block, so the counts are known.
    clf[3, :4, :4] = 1.0
    clf[2, 4:8, :4] = 1.0
    clf_tta = rng.random((NUM_CLF, SIZE, SIZE)).astype(np.float16)
    loc = rng.random((NUM_LOC, SIZE, SIZE)).astype(np.float16)
    loc_tta = rng.random((NUM_LOC, SIZE, SIZE)).astype(np.float16)
    valid = np.ones((SIZE, SIZE), dtype=np.bool_)
    return {"clf": clf, "loc": loc, "clf_tta": clf_tta, "loc_tta": loc_tta, "valid": valid}


def write_cache(cache_dir: Path, name: str, seed: int = 0) -> None:
    """Write the single-inference cache entry of one tile."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cache_dir / f"{name}.npz", **make_cache(seed))


def run(tmp_path: Path, **kwargs) -> Path:
    """Write the synthetic tiles and cache, then call :func:`build_demo`."""
    data_root = tmp_path / "data"
    cache_dir = tmp_path / "cache"
    for seed, name in enumerate(NAMES):
        write_tile(data_root, name, seed=seed)
        write_cache(cache_dir, name, seed=seed)
    params: dict = {"access": ACCESS}
    params.update(kwargs)
    return build_demo(cache_dir, data_root, tmp_path / "site", **params)


# --------------------------------------------------------------------------- footprint


def test_tile_footprint_uses_bounds_for_epsg_4326() -> None:
    meta = GeoMeta(
        width=16,
        height=16,
        origin_x=29.0,
        origin_y=41.0,
        pixel_x=0.5,
        pixel_y=0.25,
        epsg=4326,
    )

    ring = tile_footprint_lonlat(meta)

    assert len(ring) == 5
    assert ring[0] == ring[-1]
    corners = {(29.0, 37.0), (37.0, 37.0), (37.0, 41.0), (29.0, 41.0)}
    assert {tuple(point) for point in ring} == corners


def test_tile_footprint_requires_to_lonlat_outside_4326() -> None:
    meta = GeoMeta(
        width=16,
        height=16,
        origin_x=100.0,
        origin_y=200.0,
        pixel_x=0.5,
        pixel_y=0.25,
        epsg=32629,
    )

    with pytest.raises(ValueError):
        tile_footprint_lonlat(meta)


def test_tile_footprint_transforms_with_to_lonlat() -> None:
    meta = GeoMeta(
        width=10,
        height=20,
        origin_x=100.0,
        origin_y=200.0,
        pixel_x=0.5,
        pixel_y=0.25,
        epsg=32629,
    )

    ring = tile_footprint_lonlat(meta, to_lonlat=lambda x, y: (x / 10.0, y / 10.0))

    assert len(ring) == 5
    assert ring[0] == ring[-1]
    corners = {(10.0, 19.5), (10.5, 19.5), (10.5, 20.0), (10.0, 20.0)}
    assert {tuple(point) for point in ring} == corners


# --------------------------------------------------------------------------- files


def test_build_demo_returns_index_html(tmp_path: Path) -> None:
    index = run(tmp_path)

    assert index == tmp_path / "site" / "index.html"
    assert index.is_file()


def test_build_demo_writes_four_pngs_per_tile(tmp_path: Path) -> None:
    run(tmp_path)

    tiles = tmp_path / "site" / "tiles"
    for name in NAMES:
        for suffix in ("pre", "post", "tahmin", "belirsizlik"):
            path = tiles / f"{name}_{suffix}.png"
            assert path.is_file(), path
            with Image.open(path) as image:
                assert image.size == (THUMB, THUMB), path


def test_thumb_parameter_sets_png_size(tmp_path: Path) -> None:
    run(tmp_path, thumb=32)

    path = tmp_path / "site" / "tiles" / f"{NAMES[0]}_pre.png"
    with Image.open(path) as image:
        assert image.size == (32, 32)


def test_uncertainty_png_is_grey(tmp_path: Path) -> None:
    run(tmp_path)

    path = tmp_path / "site" / "tiles" / f"{NAMES[0]}_belirsizlik.png"
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))

    assert rgb.shape == (THUMB, THUMB, 3)
    assert np.array_equal(rgb[..., 0], rgb[..., 1])
    assert np.array_equal(rgb[..., 1], rgb[..., 2])


# --------------------------------------------------------------------------- geojson


def test_geojson_has_one_polygon_per_tile(tmp_path: Path) -> None:
    run(tmp_path)

    payload = json.loads((tmp_path / "site" / "tiles.geojson").read_text(encoding="utf-8"))

    assert payload["type"] == "FeatureCollection"
    features = payload["features"]
    assert len(features) == len(NAMES)

    by_name = {feature["properties"]["name"]: feature for feature in features}
    assert set(by_name) == set(NAMES)
    for name, feature in by_name.items():
        assert feature["geometry"]["type"] == "Polygon", name
        ring = feature["geometry"]["coordinates"][0]
        assert len(ring) == 5, name
        assert ring[0] == ring[-1], name
        assert feature["properties"]["n_destroyed_px"] == DESTROYED_PX, name
        assert feature["properties"]["n_damaged_px"] == DAMAGED_PX, name


def test_names_limits_the_built_tiles(tmp_path: Path) -> None:
    run(tmp_path, names=[NAMES[0]])

    tiles = tmp_path / "site" / "tiles"
    assert (tiles / f"{NAMES[0]}_pre.png").is_file()
    assert not (tiles / f"{NAMES[1]}_pre.png").exists()

    payload = json.loads((tmp_path / "site" / "tiles.geojson").read_text(encoding="utf-8"))
    assert len(payload["features"]) == 1
    assert payload["features"][0]["properties"]["name"] == NAMES[0]


# --------------------------------------------------------------------------- index.html


def test_index_references_leaflet_and_geojson(tmp_path: Path) -> None:
    run(tmp_path)

    html = (tmp_path / "site" / "index.html").read_text(encoding="utf-8")

    assert "tiles.geojson" in html
    assert "leaflet@1.9.4" in html


def test_index_footer_credits_cache_and_imagery(tmp_path: Path) -> None:
    run(tmp_path)

    html = (tmp_path / "site" / "index.html").read_text(encoding="utf-8")

    assert "Maxar" in html
    assert "BRIGHT" in html
    assert "CC BY-NC 4.0" in html
