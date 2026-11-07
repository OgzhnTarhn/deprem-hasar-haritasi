"""Reference agreement for the Y1 analysis: two optical interpretations plus one radar signal.

Y1 frames the question "against whom is the model wrong?" with three references on the same
buildings: the BRIGHT optical damage label, the HOT/OSM destroyed-building layer (a second optical
interpretation) and the learning-free ARIA damage proxy map (DPM, the radar signal). This module
measures how much those references agree with each other and with the model, using pairwise 2x2
tables, Cohen's kappa and a few simple rates.

External layers (HOT/OSM polygons, DPM rasters) must already be reprojected to the tile CRS before
they reach these helpers; reprojection with pyproj lives in ``scripts/analiz/`` and is deliberately
not part of the package. Everything here works on numpy arrays and a :class:`hasar.data.io.GeoMeta`
header, so it stays testable without data or network.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hasar.data.io import GeoMeta


@dataclass(frozen=True)
class Agreement:
    """Pairwise agreement of the references (and optionally the model).

    ``table`` maps a pair name to its 2x2 counts ``{"tt", "tf", "ft", "ff"}`` where the first
    element of the pair is the row and the second is the column. ``kappa`` holds Cohen's kappa per
    pair (0.0 when undefined) and ``rates`` the summary fractions.
    """

    n: int
    table: dict[str, dict[str, int]]
    kappa: dict[str, float]
    rates: dict[str, float]


def crs_to_pixel(
    meta: GeoMeta, x: np.ndarray | float, y: np.ndarray | float
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Map CRS coordinates to fractional ``(col, row)`` pixel coordinates.

    The tile is north-up: ``origin_x``/``origin_y`` is the top-left corner and ``row`` grows
    downwards. Pixel centres therefore land on half-integers.
    """
    xs = np.asarray(x, dtype=np.float64)
    ys = np.asarray(y, dtype=np.float64)
    col = (xs - meta.origin_x) / meta.pixel_x
    row = (meta.origin_y - ys) / meta.pixel_y
    if col.ndim == 0 and row.ndim == 0:
        return float(col), float(row)
    return col, row


def pixel_to_crs(
    meta: GeoMeta, row: np.ndarray | float, col: np.ndarray | float
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Map fractional ``(row, col)`` pixel coordinates back to CRS coordinates."""
    rows = np.asarray(row, dtype=np.float64)
    cols = np.asarray(col, dtype=np.float64)
    x = meta.origin_x + cols * meta.pixel_x
    y = meta.origin_y - rows * meta.pixel_y
    if x.ndim == 0 and y.ndim == 0:
        return float(x), float(y)
    return x, y


def _points_in_polygon(
    xs: np.ndarray,
    ys: np.ndarray,
    poly_x: np.ndarray,
    poly_y: np.ndarray,
) -> np.ndarray:
    """Even-odd ray-casting test of every ``(xs, ys)`` point against one polygon."""
    inside = np.zeros(xs.shape, dtype=np.bool_)
    count = poly_x.size
    for i in range(count):
        x1 = poly_x[i]
        y1 = poly_y[i]
        x2 = poly_x[(i + 1) % count]
        y2 = poly_y[(i + 1) % count]
        if y1 == y2:
            continue
        crosses = (y1 > ys) != (y2 > ys)
        x_cross = (x2 - x1) * (ys - y1) / (y2 - y1) + x1
        inside ^= crosses & (xs < x_cross)
    return inside


def rasterize_polygons_crs(polygons: list[np.ndarray], meta: GeoMeta) -> np.ndarray:
    """Rasterize CRS polygons onto the tile grid, returning an ``H x W`` boolean mask.

    A pixel is set when its centre falls inside at least one polygon. Polygons are ``N x 2``
    arrays of ``(x, y)`` vertices already expressed in the tile CRS.
    """
    mask = np.zeros((meta.height, meta.width), dtype=np.bool_)
    if not polygons:
        return mask

    cols = np.arange(meta.width, dtype=np.float64) + 0.5
    rows = np.arange(meta.height, dtype=np.float64) + 0.5
    grid_x, grid_y = np.meshgrid(cols, rows)

    for polygon in polygons:
        vertices = np.asarray(polygon, dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[0] < 3 or vertices.shape[1] != 2:
            raise ValueError(f"Poligon N x 2 ve en az 3 köşeli olmalı, bulunan: {vertices.shape}")
        poly_col, poly_row = crs_to_pixel(meta, vertices[:, 0], vertices[:, 1])
        mask |= _points_in_polygon(grid_x, grid_y, poly_col, poly_row)
    return mask


def sample_points_mask(points: np.ndarray, meta: GeoMeta, shape: tuple[int, int]) -> np.ndarray:
    """Mark the pixels of an ``H x W`` mask that contain one of the CRS ``points``.

    ``points`` is an ``N x 2`` array of ``(x, y)`` coordinates; points outside the tile are
    ignored. ``shape`` is ``(height, width)``.
    """
    mask = np.zeros(shape, dtype=np.bool_)
    array = np.asarray(points, dtype=np.float64)
    if array.size == 0:
        return mask
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"Noktalar N x 2 olmalı, bulunan: {array.shape}")

    col, row = crs_to_pixel(meta, array[:, 0], array[:, 1])
    cols = np.floor(col).astype(np.int64)
    rows = np.floor(row).astype(np.int64)
    height, width = shape
    valid = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
    mask[rows[valid], cols[valid]] = True
    return mask


def dpm_flags(values: np.ndarray, threshold: float) -> np.ndarray:
    """Threshold a DPM raster into a boolean damage mask; NaN pixels are never flagged."""
    array = np.asarray(values, dtype=np.float64)
    return np.asarray(np.isfinite(array) & (array > threshold), dtype=np.bool_)


def _pair_counts(a: np.ndarray, b: np.ndarray) -> dict[str, int]:
    """Return the 2x2 counts of two boolean arrays (``a`` is the row, ``b`` the column)."""
    return {
        "tt": int(np.count_nonzero(a & b)),
        "tf": int(np.count_nonzero(a & ~b)),
        "ft": int(np.count_nonzero(~a & b)),
        "ff": int(np.count_nonzero(~a & ~b)),
    }


def _cohen_kappa(counts: dict[str, int]) -> float:
    """Cohen's kappa of a 2x2 table; 0.0 when the expected agreement is 1 (undefined)."""
    tt, tf, ft, ff = counts["tt"], counts["tf"], counts["ft"], counts["ff"]
    n = tt + tf + ft + ff
    if n == 0:
        return 0.0
    po = (tt + ff) / n
    pe = ((tt + tf) * (tt + ft) + (ft + ff) * (tf + ff)) / n**2
    if pe == 1.0:
        return 0.0
    return (po - pe) / (1.0 - pe)


def agreement(
    bright: np.ndarray,
    osm: np.ndarray,
    dpm: np.ndarray | None,
    model: np.ndarray | None = None,
) -> Agreement:
    """Compare the references (and optionally the model) building by building.

    ``bright``, ``osm``, ``dpm`` and ``model`` are boolean arrays over the same buildings; ``dpm``
    may be ``None`` when the radar layer is unavailable. ``all_agree`` only looks at the available
    references, while ``model_vs_majority`` compares the model with the strict majority of those
    references (a tie counts as "not destroyed") and is reported only when a model is given.
    """
    bright = np.asarray(bright, dtype=np.bool_)
    osm = np.asarray(osm, dtype=np.bool_)
    n = int(bright.size)
    if osm.size != n:
        raise ValueError(f"Referans boyutları uyuşmuyor: bright {n}, osm {osm.size}")

    references = [bright, osm]
    pairs = [("bright-osm", bright, osm)]
    if dpm is not None:
        dpm = np.asarray(dpm, dtype=np.bool_)
        if dpm.size != n:
            raise ValueError(f"Referans boyutları uyuşmuyor: bright {n}, dpm {dpm.size}")
        references.append(dpm)
        pairs.append(("bright-dpm", bright, dpm))
        pairs.append(("osm-dpm", osm, dpm))

    if model is not None:
        model = np.asarray(model, dtype=np.bool_)
        if model.size != n:
            raise ValueError(f"Model boyutu uyuşmuyor: bright {n}, model {model.size}")
        pairs.append(("model-bright", model, bright))
        pairs.append(("model-osm", model, osm))
        if dpm is not None:
            pairs.append(("model-dpm", model, dpm))

    table = {name: _pair_counts(a, b) for name, a, b in pairs}
    kappa = {name: _cohen_kappa(counts) for name, counts in table.items()}

    all_agree = np.ones(n, dtype=np.bool_)
    for reference in references:
        all_agree &= reference == references[0]
    rates = {"all_agree": float(np.count_nonzero(all_agree) / n) if n else 0.0}

    if model is not None:
        votes = np.zeros(n, dtype=np.int64)
        for reference in references:
            votes += reference.astype(np.int64)
        majority = votes * 2 > len(references)
        rates["model_vs_majority"] = float(np.count_nonzero(model == majority) / n) if n else 0.0

    return Agreement(n=n, table=table, kappa=kappa, rates=rates)
