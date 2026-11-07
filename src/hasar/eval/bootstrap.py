"""City-stratified 2 km block bootstrap for the Turkey evaluation (numpy only).

Protocol section 8
------------------
The confidence interval of the headline metric is built with a *city-stratified 2 km block
bootstrap*: 10.000 resamples, a 95% percentile interval and a fixed seed that is written to the
results file. Two models are compared with a *paired* bootstrap that draws the same blocks for
both models, so the difference is measured on identical resamples.

Protocol section 9 ("eşit" rule)
--------------------------------
A difference is reported as "eşit" (equal) when it is smaller than 1.0 point *or* when the paired
95% interval contains 0. Against the 41.28 ``[ZS-EVENT mIoU]`` anchor only our own interval is
available, so the anchor is "eşit" when it falls inside that interval.

Why blocks?
-----------
Neighbouring tiles overlap and share the same city, sensor geometry and damage pattern, so their
confusion matrices are correlated. Resampling single tiles would treat them as independent and
make the interval artificially narrow. Resampling whole 2 km blocks (and stratifying by city)
keeps the correlation inside a resample and gives an honest interval.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hasar.eval.metrics import EPS, miou

Metric = Callable[[np.ndarray], float]

BLOCK_KM = 2.0
KM_PER_DEGREE = 111.0
DEFAULT_N = 10_000
DEFAULT_SEED = 0
ANCHOR_MIOU = 41.28
EQUAL_MARGIN = 1.0


@dataclass(frozen=True)
class BootstrapResult:
    """Point estimate and percentile interval of one bootstrap run."""

    estimate: float
    low: float
    high: float
    n: int
    seed: int


def _miou_rows(matrices: np.ndarray) -> np.ndarray:
    """Vectorised :func:`hasar.eval.metrics.miou` over a stack of ``(n, 4, 4)`` matrices."""
    diag = np.diagonal(matrices, axis1=1, axis2=2)
    denominator = matrices.sum(axis=2) + matrices.sum(axis=1) - diag + EPS
    return (diag / denominator).mean(axis=1)


def _metric_rows(matrices: np.ndarray, metric: Metric) -> np.ndarray:
    """Apply ``metric`` to every matrix; the default mIoU takes a vectorised shortcut."""
    if metric is miou:
        return _miou_rows(matrices)
    return np.array([metric(matrix) for matrix in matrices], dtype=np.float64)


def _block_layout(
    tiles: dict[str, np.ndarray], blocks: dict[str, tuple[str, str]]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return the ``(B, 4, 4)`` block matrices and the block indices of every stratum.

    Tiles that share a ``(stratum, block_id)`` pair are summed into one block matrix. A tile
    without a block entry raises ``KeyError``.
    """
    grouped: dict[tuple[str, str], np.ndarray] = {}
    for name, matrix in tiles.items():
        stratum, block_id = blocks[name]
        key = (stratum, block_id)
        matrix = np.asarray(matrix, dtype=np.int64)
        grouped[key] = grouped[key] + matrix if key in grouped else matrix.copy()

    keys = list(grouped)
    block_matrices = np.stack([grouped[key] for key in keys])

    strata: dict[str, list[int]] = {}
    for index, (stratum, _block_id) in enumerate(keys):
        strata.setdefault(stratum, []).append(index)
    stratum_indices = {stratum: np.array(idx, dtype=np.int64) for stratum, idx in strata.items()}
    return block_matrices, stratum_indices


def _draw_weights(stratum_indices: dict[str, np.ndarray], n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``n`` bootstrap weight vectors over the blocks, stratified by city.

    Each stratum keeps its own block count: a resample draws as many blocks from a stratum as the
    original data has, with replacement. The result is an ``(n, B)`` integer count matrix.
    """
    n_blocks = 1 + max(int(idx.max()) for idx in stratum_indices.values())
    weights = np.zeros((n, n_blocks), dtype=np.int64)
    rows = np.arange(n)[:, None]
    for indices in stratum_indices.values():
        k = indices.size
        draws = rng.integers(0, k, size=(n, k))
        flat = (draws + rows * k).ravel()
        counts = np.bincount(flat, minlength=n * k).reshape(n, k)
        weights[:, indices] = counts
    return weights


def _resample_matrices(weights: np.ndarray, block_matrices: np.ndarray) -> np.ndarray:
    """Sum the weighted block matrices into ``(n, 4, 4)`` resampled confusion matrices."""
    flat = block_matrices.reshape(block_matrices.shape[0], -1)
    return (weights @ flat).reshape(weights.shape[0], *block_matrices.shape[1:])


def _interval(values: np.ndarray, estimate: float, n: int, seed: int) -> BootstrapResult:
    """Build the 95% percentile interval of ``values`` around ``estimate``."""
    low, high = np.percentile(values, [2.5, 97.5])
    return BootstrapResult(estimate=float(estimate), low=float(low), high=float(high), n=n, seed=seed)


def block_bootstrap(
    tiles: dict[str, np.ndarray],
    blocks: dict[str, tuple[str, str]],
    metric: Metric = miou,
    n: int = DEFAULT_N,
    seed: int = DEFAULT_SEED,
) -> BootstrapResult:
    """Bootstrap one model's metric with a city-stratified block resampling.

    ``tiles`` maps a tile name to its 4x4 confusion matrix, ``blocks`` maps the same name to a
    ``(stratum, block_id)`` pair. The estimate is ``metric`` of the summed tile matrices; the
    interval comes from ``n`` resamples drawn with ``np.random.default_rng(seed)``.
    """
    if not tiles:
        raise ValueError("block_bootstrap needs at least one tile")

    block_matrices, stratum_indices = _block_layout(tiles, blocks)
    estimate = float(metric(np.sum(block_matrices, axis=0)))

    rng = np.random.default_rng(seed)
    weights = _draw_weights(stratum_indices, n, rng)
    values = _metric_rows(_resample_matrices(weights, block_matrices), metric)
    return _interval(values, estimate, n, seed)


def paired_block_bootstrap(
    tiles_a: dict[str, np.ndarray],
    tiles_b: dict[str, np.ndarray],
    blocks: dict[str, tuple[str, str]],
    metric: Metric = miou,
    n: int = DEFAULT_N,
    seed: int = DEFAULT_SEED,
) -> BootstrapResult:
    """Bootstrap the metric difference ``metric(b) - metric(a)`` on the same blocks.

    ``b`` is the candidate model and ``a`` the reference, so a positive difference means ``b`` is
    better. Both models must cover exactly the same tiles; every resample draws one set of blocks
    and applies it to both models, which keeps the comparison paired.
    """
    if set(tiles_a) != set(tiles_b):
        raise ValueError("paired_block_bootstrap needs identical tile keys for both models")
    if not tiles_a:
        raise ValueError("paired_block_bootstrap needs at least one tile")

    blocks_a, stratum_indices = _block_layout(tiles_a, blocks)
    blocks_b, _ = _block_layout(tiles_b, blocks)
    estimate = float(metric(np.sum(blocks_b, axis=0)) - metric(np.sum(blocks_a, axis=0)))

    rng = np.random.default_rng(seed)
    weights = _draw_weights(stratum_indices, n, rng)
    values = _metric_rows(_resample_matrices(weights, blocks_b), metric) - _metric_rows(
        _resample_matrices(weights, blocks_a), metric
    )
    return _interval(values, estimate, n, seed)


def equal_verdict(difference: float, low: float, high: float) -> str:
    """Return "eşit" or "farklı" for a paired difference (protocol section 9)."""
    if abs(difference) < EQUAL_MARGIN or low <= 0.0 <= high:
        return "eşit"
    return "farklı"


def anchor_verdict(estimate: float, low: float, high: float, anchor: float = ANCHOR_MIOU) -> str:
    """Compare our own interval with the 41.28 anchor: "eşit", "üstünde" or "altında"."""
    if low <= anchor <= high:
        return "eşit"
    return "üstünde" if estimate > anchor else "altında"


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres between two points."""
    radius = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * radius * math.asin(math.sqrt(a))


def turkey_blocks(path: str | Path, block_km: float = BLOCK_KM) -> dict[str, tuple[str, str]]:
    """Build the ``(stratum, block_id)`` map of the Turkey tiles from the research JSON.

    The stratum is the nearest city cluster of the tile centre; the block id is the ``gy_gx``
    grid cell of a ``block_km`` grid (latitude step ``111 km``, longitude step scaled by the mean
    latitude). The file is ``results/arastirma/turkiye_bloklar.json``.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    centers = payload["tile_centers"]
    clusters = payload["clusters"]
    if not centers:
        raise ValueError(f"no tile_centers in {path}")

    lat0 = sum(float(tile["lat"]) for tile in centers) / len(centers)
    lon_scale = KM_PER_DEGREE * math.cos(math.radians(lat0))

    out: dict[str, tuple[str, str]] = {}
    for tile in centers:
        lat, lon = float(tile["lat"]), float(tile["lon"])
        nearest = min(
            clusters,
            key=lambda c: _haversine_km(lat, lon, float(c["center_lat"]), float(c["center_lon"])),
        )
        gy = math.floor(lat * KM_PER_DEGREE / block_km)
        gx = math.floor(lon * lon_scale / block_km)
        out[str(tile["name"])] = (str(nearest["nearest_place"]), f"{gy}_{gx}")
    return out
