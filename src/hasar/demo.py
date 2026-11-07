"""Static demo site of the Faz 9 result, built from the real 6 February data.

The demo shows the Turkey tiles of the single ``final`` evaluation: the pre-disaster optical
image, the post-disaster SAR image, the predicted damage map and the building uncertainty, one
Leaflet polygon per tile on a satellite base map. Every prediction comes from the Faz 6 cache
(``<cache>/<name>.npz``); the model is never imported and never rerun here, so building the site
costs no GPU time and cannot change a single number.

The page is a single ``index.html`` with a module-level template string, the four PNGs per tile
under ``tiles/`` and one ``tiles.geojson``. The only external resource is the Leaflet stylesheet
and script from unpkg, pinned to the same version.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from string import Template

import numpy as np
from PIL import Image

from hasar.analysis.gallery import PALETTE
from hasar.data.io import GeoMeta, read_geo_meta, read_tile, tile_paths
from hasar.eval.calibration import entropy
from hasar.guard import TurkeyAccess

#: Default side length of one demo thumbnail in pixels.
DEFAULT_THUMB = 256

#: EPSG code of the geographic CRS the Leaflet page expects.
WGS84 = 4326

#: Suffixes of the four PNGs written per tile.
IMAGE_SUFFIXES = ("pre", "post", "tahmin", "belirsizlik")

#: Damage class indices of the ``clf`` head.
DESTROYED = 3
DAMAGED = 2

#: Leaflet version pinned in the page (stylesheet and script share it).
LEAFLET_VERSION = "1.9.4"

#: The single page of the demo; ``$`` placeholders are filled by :func:`build_demo`.
INDEX_TEMPLATE = Template(
    """<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Deprem hasar haritası — 6 Şubat 2023</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@$leaflet/dist/leaflet.css">
<style>
  html, body { margin: 0; height: 100%; font-family: system-ui, sans-serif; }
  #map { height: 100%; }
  .popup { width: 540px; }
  .popup .row { display: flex; gap: 4px; }
  .popup figure { margin: 0; flex: 1; text-align: center; font-size: 11px; }
  .popup img { width: 128px; height: 128px; image-rendering: pixelated; }
  footer {
    position: absolute; bottom: 0; left: 0; right: 0; z-index: 1000;
    background: rgba(255, 255, 255, 0.9); padding: 4px 8px; font-size: 12px;
  }
</style>
</head>
<body>
<div id="map"></div>
<footer>
  Görüntüler: Maxar (afet öncesi optik) ve BRIGHT veri seti (afet sonrası SAR), CC BY-NC 4.0.
  Tahminler Faz 6 önbelleğinden okunur; model bu sayfa için yeniden çalıştırılmadı.
  $count kare.
</footer>
<script src="https://unpkg.com/leaflet@$leaflet/dist/leaflet.js"></script>
<script>
const map = L.map('map');
L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
  maxZoom: 19,
  attribution: 'Maxar'
}).addTo(map);

fetch('tiles.geojson')
  .then((response) => response.json())
  .then((data) => {
    const layer = L.geoJSON(data, {
      style: { color: '#ff2d2d', weight: 1, fillOpacity: 0.15 },
      onEachFeature: (feature, target) => {
        const p = feature.properties;
        const figure = (suffix, caption) =>
          '<figure><img src="tiles/' + p.name + '_' + suffix + '.png" alt="' + caption + '">' +
          '<figcaption>' + caption + '</figcaption></figure>';
        target.bindPopup(
          '<div class="popup"><div class="row">' +
          figure('pre', 'Afet öncesi') +
          figure('post', 'Afet sonrası') +
          figure('tahmin', 'Tahmin') +
          figure('belirsizlik', 'Belirsizlik') +
          '</div><p>' + p.name + ' — yıkık ' + p.n_destroyed_px + ' px, hasarlı ' +
          p.n_damaged_px + ' px</p></div>'
        );
      }
    }).addTo(map);
    if (layer.getBounds().isValid()) {
      map.fitBounds(layer.getBounds());
    }
  });
</script>
</body>
</html>
"""
)


def _corners(meta: GeoMeta) -> list[tuple[float, float]]:
    """Return the four corners of ``meta`` in its own CRS, counter-clockwise from bottom-left."""
    left, bottom, right, top = meta.bounds()
    return [(left, bottom), (right, bottom), (right, top), (left, top)]


def tile_footprint_lonlat(
    meta: GeoMeta,
    to_lonlat: Callable[[float, float], tuple[float, float]] | None = None,
) -> list[tuple[float, float]]:
    """Return the closed ``(lon, lat)`` ring of one tile's footprint.

    An EPSG 4326 tile needs no transform and its bounds are used directly. Any other CRS needs
    ``to_lonlat`` (a pyproj transformer's ``transform``); without it a :class:`ValueError` is
    raised so a wrong footprint never reaches the page. The ring repeats its first point at the
    end, as GeoJSON polygons require.
    """
    corners = _corners(meta)
    if meta.epsg == WGS84:
        ring = corners
    elif to_lonlat is None:
        raise ValueError(f"Kare EPSG {meta.epsg} konumunda: enlem/boylam çokgeni için to_lonlat dönüşümü gerekli.")
    else:
        ring = [tuple(float(value) for value in to_lonlat(x, y)) for x, y in corners]
    return [*ring, ring[0]]


def _resize(array: np.ndarray, size: int) -> np.ndarray:
    """Resize an ``(H, W)`` or ``(H, W, 3)`` uint8 array to ``size x size`` (nearest neighbour)."""
    image = Image.fromarray(np.asarray(array, dtype=np.uint8))
    return np.asarray(image.resize((size, size), Image.NEAREST))


def _colorize(label: np.ndarray) -> np.ndarray:
    """Return the ``(H, W, 3)`` RGB image of a damage map using the gallery palette."""
    label = np.asarray(label)
    rgb = np.zeros(label.shape + (3,), dtype=np.uint8)
    for value, colour in PALETTE.items():
        rgb[label == value] = colour
    return rgb


def _uncertainty_grey(clf_tta: np.ndarray) -> np.ndarray:
    """Return the ``(H, W)`` uint8 grey image of the per-pixel entropy of the TTA probabilities."""
    probs = np.asarray(clf_tta, dtype=np.float64)
    classes, height, width = probs.shape
    values = entropy(probs.reshape(classes, -1)).reshape(height, width)
    scale = np.log(classes) if classes > 1 else 1.0
    return np.clip(values / scale * 255.0, 0, 255).astype(np.uint8)


def _save_png(array: np.ndarray, path: Path, size: int) -> None:
    """Resize ``array`` to ``size x size`` and write it as a PNG."""
    Image.fromarray(_resize(array, size)).save(path)


def _cache_names(cache_dir: Path) -> list[str]:
    """Return the sorted tile names that have a cache entry under ``cache_dir``."""
    return sorted(path.stem for path in cache_dir.glob("*.npz"))


def _load_cache(path: Path) -> dict[str, np.ndarray]:
    """Return the cached arrays of one tile."""
    with np.load(path) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _write_tile_images(
    name: str,
    pre: np.ndarray,
    post: np.ndarray,
    cache: dict[str, np.ndarray],
    tiles_dir: Path,
    thumb: int,
) -> None:
    """Write the four PNGs of one tile under ``tiles_dir``."""
    pred = np.asarray(cache["clf"]).argmax(axis=0)
    post_grey = np.repeat(np.asarray(post, dtype=np.uint8)[:, :, None], 3, axis=2)

    _save_png(pre, tiles_dir / f"{name}_pre.png", thumb)
    _save_png(post_grey, tiles_dir / f"{name}_post.png", thumb)
    _save_png(_colorize(pred), tiles_dir / f"{name}_tahmin.png", thumb)
    _save_png(_uncertainty_grey(cache["clf_tta"]), tiles_dir / f"{name}_belirsizlik.png", thumb)


def _feature(
    name: str,
    meta: GeoMeta,
    cache: dict[str, np.ndarray],
    to_lonlat: Callable[[float, float], tuple[float, float]] | None,
) -> dict:
    """Return the GeoJSON feature of one tile with its damage pixel counts."""
    pred = np.asarray(cache["clf"]).argmax(axis=0)
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [tile_footprint_lonlat(meta, to_lonlat)]},
        "properties": {
            "name": name,
            "n_destroyed_px": int((pred == DESTROYED).sum()),
            "n_damaged_px": int((pred == DAMAGED).sum()),
        },
    }


def build_demo(
    cache_dir: Path | str,
    data_root: Path | str,
    out_dir: Path | str,
    *,
    access: TurkeyAccess,
    names: Sequence[str] | None = None,
    thumb: int = DEFAULT_THUMB,
    to_lonlat: Callable[[float, float], tuple[float, float]] | None = None,
) -> Path:
    """Build the static demo site and return the path of ``index.html``.

    ``cache_dir`` holds the Faz 6 ``.npz`` entries, ``data_root`` the BRIGHT tiles and ``out_dir``
    the site. ``access`` is the Turkey token the tile reader requires; the model is never run.
    ``names`` limits the built tiles (default: every cache entry). ``to_lonlat`` reprojects the
    footprints of tiles that are not EPSG 4326.
    """
    cache_dir = Path(cache_dir)
    data_root = Path(data_root)
    out_dir = Path(out_dir)
    tiles_dir = out_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    selected = list(names) if names is not None else _cache_names(cache_dir)
    features: list[dict] = []
    for name in selected:
        cache = _load_cache(cache_dir / f"{name}.npz")
        pre, post, _label = read_tile(data_root, name, access=access)
        meta = read_geo_meta(tile_paths(data_root, name).pre)

        _write_tile_images(name, pre, post, cache, tiles_dir, thumb)
        features.append(_feature(name, meta, cache, to_lonlat))

    payload = {"type": "FeatureCollection", "features": features}
    (out_dir / "tiles.geojson").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    index = out_dir / "index.html"
    index.write_text(
        INDEX_TEMPLATE.substitute(leaflet=LEAFLET_VERSION, count=len(features)),
        encoding="utf-8",
    )
    return index


def first_epsg(cache_dir: Path | str, data_root: Path | str) -> int | None:
    """Return the EPSG code of the first cached tile, or ``None`` when the cache is empty."""
    names = _cache_names(Path(cache_dir))
    if not names:
        return None
    return read_geo_meta(tile_paths(Path(data_root), names[0]).pre).epsg
