"""Paired comparison of the two models on the Turkey evaluation (Faz 8).

Both models are reported side by side: the main model (ChangeMamba as published) and the single
idea variant. The comparison never *selects* a model on the Turkey result -- the design of the
variant was locked with ``varyant-kilidi`` before the main model's Turkey number was seen, and the
headline table simply carries both rows.

The difference is written with the protocol section 9 "eşit" rule: a difference smaller than 1.0
point, or a paired 95% interval that contains 0, is reported as "eşit"; otherwise the difference
and its interval are written. The interval comes from the paired block bootstrap of protocol
section 8 (:func:`hasar.eval.bootstrap.paired_block_bootstrap`), which draws the same blocks for
both models so the difference is measured on identical resamples.

The ensemble row is reported only if KN1 locked it. Its weight is chosen on the 13 validation
events, never on Turkey, and the row averages the two single-inference caches
(:func:`ensemble_tile`) instead of running either model again.
"""

from __future__ import annotations

import numpy as np

from hasar.eval.bootstrap import DEFAULT_N, DEFAULT_SEED, equal_verdict, paired_block_bootstrap
from hasar.eval.metrics import miou
from hasar.results import ResultRecord

#: The metric tag of the paired difference (protocol section 9).
DIFF_TAG = "[ZS-EVENT mIoU, eslestirilmis-fark]"

#: The split every Turkey record carries.
SPLIT = "zero_shot/test"

#: The probability maps stored in the single-inference cache (protocol section 11).
CACHE_KEYS = ("clf", "loc", "clf_tta", "loc_tta")


def compare_models(
    per_tile_a: dict[str, np.ndarray],
    per_tile_b: dict[str, np.ndarray],
    blocks: dict[str, tuple[str, str]] | None = None,
    n: int = DEFAULT_N,
    seed: int = DEFAULT_SEED,
) -> dict[str, float | str]:
    """Compare two models on the same tiles with a paired block bootstrap.

    ``per_tile_a`` and ``per_tile_b`` map a tile name to its 4x4 confusion matrix; both must cover
    exactly the same tiles. ``blocks`` maps a tile name to a ``(stratum, block_id)`` pair; when it
    is ``None`` every tile is its own block in a single stratum. The point estimates are the mIoU
    of the summed matrices in percent, ``diff_pct`` is ``b - a`` and the interval is the paired
    95% percentile interval of that difference. The verdict follows protocol section 9.
    """
    if set(per_tile_a) != set(per_tile_b):
        raise ValueError("compare_models needs identical tile keys for both models")
    if not per_tile_a:
        raise ValueError("compare_models needs at least one tile")

    if blocks is None:
        blocks = {name: ("all", name) for name in per_tile_a}

    a_miou_pct = miou(sum(per_tile_a.values())) * 100.0
    b_miou_pct = miou(sum(per_tile_b.values())) * 100.0

    paired = paired_block_bootstrap(per_tile_a, per_tile_b, blocks, metric=miou, n=n, seed=seed)
    diff_pct = paired.estimate * 100.0
    low_pct = paired.low * 100.0
    high_pct = paired.high * 100.0

    return {
        "a_miou_pct": a_miou_pct,
        "b_miou_pct": b_miou_pct,
        "diff_pct": diff_pct,
        "low_pct": low_pct,
        "high_pct": high_pct,
        "verdict": equal_verdict(diff_pct, low_pct, high_pct),
    }


def ensemble_tile(
    cache_a: dict[str, np.ndarray],
    cache_b: dict[str, np.ndarray],
    weight: float = 0.5,
) -> dict[str, np.ndarray]:
    """Average the two single-inference caches with ``weight`` on model ``b``.

    ``weight`` is chosen on the 13 validation events (KN1), never on Turkey. Every probability map
    of the cache (``clf``, ``loc``, ``clf_tta``, ``loc_tta``) is combined as
    ``(1 - weight) * a + weight * b`` and returned as ``float32``.
    """
    out: dict[str, np.ndarray] = {}
    for name in CACHE_KEYS:
        a = np.asarray(cache_a[name], dtype=np.float32)
        b = np.asarray(cache_b[name], dtype=np.float32)
        out[name] = ((1.0 - weight) * a + weight * b).astype(np.float32)
    return out


def compare_record(
    result: dict[str, float | str],
    model_a: str,
    model_b: str,
    commit: str,
    config_hash: str,
    seed: int | None = None,
) -> ResultRecord:
    """Wrap a :func:`compare_models` result in a :class:`ResultRecord`.

    The numeric entries become ``values``; the verdict and both model names go to ``extra`` so the
    "eşit" rule is visible in the JSON file.
    """
    values = {key: float(value) for key, value in result.items() if isinstance(value, (int, float))}
    return ResultRecord(
        metric_tag=DIFF_TAG,
        values=values,
        model=model_a,
        split=SPLIT,
        commit=commit,
        config_hash=config_hash,
        seed=seed,
        extra={"verdict": result["verdict"], "model_a": model_a, "model_b": model_b},
    )
