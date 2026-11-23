"""Coordinate-only block split for the Türkiye few-shot adaptation experiment.

Design section 5.1
------------------
The split uses only the tile centres and city clusters from ``results/arastirma/turkiye_bloklar.json``
and never reads imagery or damage labels. Tile-to-block assignments come directly from
:func:`hasar.eval.bootstrap.turkey_blocks`, so the split and the city-stratified block bootstrap
share the exact same ``(city, block_id)`` identities.

Within each city (in sorted order), blocks are shuffled with a single ``np.random.default_rng(seed)``
and taken whole into ``adapt`` until the city's adapted tile count reaches
``ceil(fraction * city_tile_count)``. Non-adapted tiles whose centre is strictly closer than
``buffer_km`` to any adapted tile centre become ``buffer``; all remaining tiles form ``test``.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from hasar.eval.bootstrap import _haversine_km, turkey_blocks

_SPLIT_KEYS = ("adapt", "buffer", "test")


def _list_sha256(names: list[str]) -> str:
    """Return the hex SHA-256 digest of a newline-joined tile list."""
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def make_fewshot_split(
    blocks_path: Path | str,
    fraction: float = 0.2,
    seed: int = 2023,
    buffer_km: float = 0.5,
) -> dict[str, Any]:
    """Build a deterministic ``adapt`` / ``buffer`` / ``test`` split from tile coordinates only."""
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction 0 ile 1 arasında olmalı (0 < fraction <= 1), alınan: {fraction!r}")
    if not math.isfinite(buffer_km) or buffer_km < 0.0:
        raise ValueError(f"buffer_km sonlu ve negatif olmayan bir sayı olmalı, alınan: {buffer_km!r}")

    source_path = Path(blocks_path)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    tile_centers = payload["tile_centers"]

    coords: dict[str, tuple[float, float]] = {}
    for tile in tile_centers:
        name = str(tile["name"])
        if name in coords:
            raise ValueError(f"Yinelenen kare adı bulundu: '{name}'")
        coords[name] = (float(tile["lat"]), float(tile["lon"]))

    blocks = turkey_blocks(source_path)

    city_blocks: dict[str, dict[str, list[str]]] = {}
    city_counts: dict[str, int] = {}
    for name, (city, block_id) in blocks.items():
        city_blocks.setdefault(city, {}).setdefault(block_id, []).append(name)
        city_counts[city] = city_counts.get(city, 0) + 1

    rng = np.random.default_rng(seed)
    adapt_set: set[str] = set()
    for city in sorted(city_blocks):
        sorted_blocks = sorted(city_blocks[city])
        order = [sorted_blocks[int(idx)] for idx in rng.permutation(len(sorted_blocks))]
        target = math.ceil(fraction * city_counts[city])
        taken = 0
        for block_id in order:
            tiles_in_block = city_blocks[city][block_id]
            adapt_set.update(tiles_in_block)
            taken += len(tiles_in_block)
            if taken >= target:
                break

    adapt = sorted(adapt_set)
    adapt_coords = [coords[name] for name in adapt]

    buffer_set: set[str] = set()
    test_set: set[str] = set()
    for name in coords:
        if name in adapt_set:
            continue
        lat, lon = coords[name]
        if any(_haversine_km(lat, lon, alat, alon) < buffer_km for alat, alon in adapt_coords):
            buffer_set.add(name)
        else:
            test_set.add(name)

    buffer = sorted(buffer_set)
    test = sorted(test_set)

    per_city: dict[str, dict[str, int]] = {}
    for city in sorted(city_blocks):
        city_tiles = [name for names in city_blocks[city].values() for name in names]
        per_city[city] = {
            "adapt": sum(1 for name in city_tiles if name in adapt_set),
            "buffer": sum(1 for name in city_tiles if name in buffer_set),
            "test": sum(1 for name in city_tiles if name in test_set),
        }

    return {
        "kind": "fewshot-split",
        "params": {
            "fraction": fraction,
            "seed": seed,
            "buffer_km": buffer_km,
            "source": source_path.as_posix(),
        },
        "counts": {
            "adapt": len(adapt),
            "buffer": len(buffer),
            "test": len(test),
            "total": len(coords),
        },
        "per_city": per_city,
        "adapt": adapt,
        "buffer": buffer,
        "test": test,
        "sha256": {
            "adapt": _list_sha256(adapt),
            "buffer": _list_sha256(buffer),
            "test": _list_sha256(test),
        },
    }


def write_split(path: Path | str, split: dict[str, Any]) -> None:
    """Write ``split`` to ``path`` as canonical UTF-8 JSON with LF line endings."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(split, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    target.write_bytes(text.encode("utf-8"))


def load_split(path: Path | str) -> dict[str, Any]:
    """Load a few-shot split JSON and verify its SHA-256 digests and disjointness."""
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Bölme dosyası geçersiz ({source}): JSON nesnesi bekleniyordu.")

    digests = payload.get("sha256")
    if not isinstance(digests, dict):
        raise ValueError(f"Bölme dosyası geçersiz ({source}): 'sha256' alanı eksik veya geçersiz.")

    sets: dict[str, set[str]] = {}
    for key in _SPLIT_KEYS:
        if key not in payload:
            raise ValueError(f"Bölme dosyası geçersiz ({source}): '{key}' listesi eksik.")
        items = payload[key]
        if not isinstance(items, list) or not all(isinstance(item, str) for item in items):
            raise ValueError(f"Bölme dosyası geçersiz ({source}): '{key}' bir metin listesi olmalı.")
        expected = digests.get(key)
        if not isinstance(expected, str):
            raise ValueError(f"Bölme dosyası geçersiz ({source}): 'sha256.{key}' alanı eksik veya geçersiz.")
        actual = _list_sha256(items)
        if actual != expected:
            raise ValueError(f"Bölme dosyası bütünlük hatası ({source}): '{key}' listesinin sha256 özeti uyuşmuyor.")
        sets[key] = set(items)

    for left, right in (("adapt", "buffer"), ("adapt", "test"), ("buffer", "test")):
        overlap = sets[left] & sets[right]
        if overlap:
            raise ValueError(
                f"Bölme dosyası geçersiz ({source}): '{left}' ve '{right}' listeleri kesişiyor "
                f"({len(overlap)} ortak kare)."
            )

    return payload
