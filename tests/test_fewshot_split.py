"""Synthetic tests for the coordinate-only few-shot block split (``hasar.fewshot.split``).

All fixtures are generated in ``tmp_path``; no real BRIGHT data or research JSON is read.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from hasar.eval.bootstrap import _haversine_km, turkey_blocks
from hasar.fewshot.split import load_split, make_fewshot_split, write_split


def _make_synthetic_blocks_json(path: Path) -> Path:
    """Write a synthetic blocks JSON with 3 cities and 5 two-km blocks per city.

    Adjacent blocks have tiles placed near their shared boundary (~0.44 km apart) so that
    selecting a block into ``adapt`` produces both ``buffer`` tiles (< 1.0 km) and distant
    ``test`` tiles (>= 1.0 km).
    """
    cities = [
        ("sehir_a", 37.0, 36.0),
        ("sehir_b", 38.0, 37.5),
        ("sehir_c", 39.0, 39.0),
    ]
    clusters = [{"nearest_place": name, "center_lat": lat, "center_lon": lon} for name, lat, lon in cities]

    step_deg = 2.0 / 111.0  # 2 km latitude block step
    tile_centers: list[dict[str, object]] = []
    tile_idx = 1

    for _city_name, base_lat, base_lon in cities:
        for block_i in range(5):
            block_base = base_lat + block_i * step_deg
            # Alternating 2 and 3 tiles per block so block sizes vary.
            offsets = (0.002, 0.016) if block_i % 2 == 0 else (0.002, 0.009, 0.016)
            for lat_off in offsets:
                tile_centers.append(
                    {
                        "name": f"turkey-earthquake_{tile_idx:08d}",
                        "lat": block_base + lat_off,
                        "lon": base_lon + 0.002,
                    }
                )
                tile_idx += 1

    payload = {"clusters": clusters, "tile_centers": tile_centers}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_determinism_and_lf_line_endings(tmp_path: Path) -> None:
    blocks_path = _make_synthetic_blocks_json(tmp_path / "blocks.json")

    first = make_fewshot_split(blocks_path, fraction=0.25, seed=2023, buffer_km=1.0)
    second = make_fewshot_split(blocks_path, fraction=0.25, seed=2023, buffer_km=1.0)

    out_a = tmp_path / "nested" / "a.json"
    out_b = tmp_path / "nested" / "b.json"
    write_split(out_a, first)
    write_split(out_b, second)

    bytes_a = out_a.read_bytes()
    bytes_b = out_b.read_bytes()
    assert bytes_a == bytes_b
    assert b"\r\n" not in bytes_a
    assert bytes_a.endswith(b"\n")
    assert first["kind"] == "fewshot-split"
    assert first["params"] == {
        "fraction": 0.25,
        "seed": 2023,
        "buffer_km": 1.0,
        "source": blocks_path.as_posix(),
    }


def test_whole_blocks_and_per_city_fraction_rule(tmp_path: Path) -> None:
    blocks_path = _make_synthetic_blocks_json(tmp_path / "blocks.json")
    fraction = 0.35
    seed = 2023
    split = make_fewshot_split(blocks_path, fraction=fraction, seed=seed, buffer_km=1.0)

    block_map = turkey_blocks(blocks_path)
    adapt_set = set(split["adapt"])

    # Group tiles by (city, block_id).
    by_city_block: dict[str, dict[str, list[str]]] = {}
    for name, (city, block_id) in block_map.items():
        by_city_block.setdefault(city, {}).setdefault(block_id, []).append(name)

    # Every block must be taken whole or not at all.
    for city, blocks in by_city_block.items():
        for block_id, tiles in blocks.items():
            in_adapt = [t in adapt_set for t in tiles]
            assert all(in_adapt) or not any(in_adapt), f"{city}/{block_id} partially split"

    # Per-city fraction rule: adapt >= ceil(fraction * n), and dropping the last-taken block falls below.
    rng = np.random.default_rng(seed)
    for city in sorted(by_city_block):
        blocks = by_city_block[city]
        city_total = sum(len(tiles) for tiles in blocks.values())
        target = math.ceil(fraction * city_total)

        sorted_blocks = sorted(blocks)
        order = [sorted_blocks[int(i)] for i in rng.permutation(len(sorted_blocks))]
        taken_blocks = [b for b in order if blocks[b][0] in adapt_set]
        assert taken_blocks
        # Taken blocks must be a prefix of the permuted order.
        assert taken_blocks == order[: len(taken_blocks)]

        taken_count = sum(len(blocks[b]) for b in taken_blocks)
        assert taken_count >= target
        without_last = taken_count - len(blocks[taken_blocks[-1]])
        assert without_last < target


def test_disjointness_coverage_and_buffer_distance(tmp_path: Path) -> None:
    blocks_path = _make_synthetic_blocks_json(tmp_path / "blocks.json")
    buffer_km = 1.0
    split = make_fewshot_split(blocks_path, fraction=0.25, seed=2023, buffer_km=buffer_km)

    payload = json.loads(blocks_path.read_text(encoding="utf-8"))
    coords = {str(t["name"]): (float(t["lat"]), float(t["lon"])) for t in payload["tile_centers"]}

    adapt = set(split["adapt"])
    buf = set(split["buffer"])
    test = set(split["test"])

    assert split["adapt"] == sorted(adapt)
    assert split["buffer"] == sorted(buf)
    assert split["test"] == sorted(test)

    # Pairwise disjoint and complete coverage.
    assert adapt.isdisjoint(buf)
    assert adapt.isdisjoint(test)
    assert buf.isdisjoint(test)
    assert adapt | buf | test == set(coords)

    # Synthetic layout should exercise both buffer and test sets.
    assert len(buf) > 0
    assert len(test) > 0

    adapt_coords = [coords[name] for name in adapt]
    for name in buf:
        lat, lon = coords[name]
        min_dist = min(_haversine_km(lat, lon, alat, alon) for alat, alon in adapt_coords)
        assert min_dist < buffer_km

    for name in test:
        lat, lon = coords[name]
        min_dist = min(_haversine_km(lat, lon, alat, alon) for alat, alon in adapt_coords)
        assert min_dist >= buffer_km


def test_counts_and_per_city_consistency(tmp_path: Path) -> None:
    blocks_path = _make_synthetic_blocks_json(tmp_path / "blocks.json")
    split = make_fewshot_split(blocks_path, fraction=0.2, seed=2023, buffer_km=1.0)
    block_map = turkey_blocks(blocks_path)

    counts = split["counts"]
    assert counts["adapt"] == len(split["adapt"])
    assert counts["buffer"] == len(split["buffer"])
    assert counts["test"] == len(split["test"])
    assert counts["total"] == len(block_map)
    assert counts["adapt"] + counts["buffer"] + counts["test"] == counts["total"]

    per_city = split["per_city"]
    assert set(per_city) == {city for city, _ in block_map.values()}
    assert sum(c["adapt"] for c in per_city.values()) == counts["adapt"]
    assert sum(c["buffer"] for c in per_city.values()) == counts["buffer"]
    assert sum(c["test"] for c in per_city.values()) == counts["test"]


def test_load_split_roundtrip_and_tamper_detection(tmp_path: Path) -> None:
    blocks_path = _make_synthetic_blocks_json(tmp_path / "blocks.json")
    split = make_fewshot_split(blocks_path, fraction=0.2, seed=2023, buffer_km=1.0)
    split_path = tmp_path / "bolme.json"
    write_split(split_path, split)

    loaded = load_split(split_path)
    assert loaded == split

    # Tamper with test list without updating sha256.
    tampered = json.loads(split_path.read_text(encoding="utf-8"))
    tampered["test"].pop()
    tampered_path = tmp_path / "tampered.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        load_split(tampered_path)

    # Missing key.
    missing = json.loads(split_path.read_text(encoding="utf-8"))
    del missing["buffer"]
    missing_path = tmp_path / "missing.json"
    missing_path.write_text(json.dumps(missing), encoding="utf-8")
    with pytest.raises(ValueError):
        load_split(missing_path)

    # Overlapping lists even with recomputed sha256.
    overlap = json.loads(split_path.read_text(encoding="utf-8"))
    overlap["test"].append(overlap["adapt"][0])
    import hashlib

    overlap["sha256"]["test"] = hashlib.sha256("\n".join(overlap["test"]).encode("utf-8")).hexdigest()
    overlap_path = tmp_path / "overlap.json"
    overlap_path.write_text(json.dumps(overlap), encoding="utf-8")
    with pytest.raises(ValueError, match="kesişiyor"):
        load_split(overlap_path)


def test_input_validation(tmp_path: Path) -> None:
    blocks_path = _make_synthetic_blocks_json(tmp_path / "blocks.json")

    with pytest.raises(ValueError):
        make_fewshot_split(blocks_path, fraction=0.0)
    with pytest.raises(ValueError):
        make_fewshot_split(blocks_path, fraction=-0.1)
    with pytest.raises(ValueError):
        make_fewshot_split(blocks_path, fraction=1.1)
    with pytest.raises(ValueError):
        make_fewshot_split(blocks_path, buffer_km=-0.5)

    # Duplicate tile names.
    dup_payload = json.loads(blocks_path.read_text(encoding="utf-8"))
    dup_payload["tile_centers"].append(dict(dup_payload["tile_centers"][0]))
    dup_path = tmp_path / "dup.json"
    dup_path.write_text(json.dumps(dup_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Yinelenen"):
        make_fewshot_split(dup_path)
