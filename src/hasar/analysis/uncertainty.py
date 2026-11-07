"""Y2 analyst-budget prioritisation and building uncertainty (protocol section 10).

Protocol section 10 defines the building uncertainty as the entropy of the flip-TTA mean
probabilities, averaged inside each instance. Y2 asks how far a limited analyst budget goes when
the buildings are reviewed in order of that uncertainty: the risk-coverage curve and its area
(AURC) summarise the ranking, and the analyst-budget table compares uncertainty with the damage
probability, the DPM and a random order.

Every helper here works on cached single-inference outputs (label, instance map, flip-TTA mean
probabilities and optionally the DPM raster); nothing reruns the model. The building majority
helpers and ``building_f1`` come from :mod:`hasar.eval.building`, the entropy from
:mod:`hasar.eval.calibration`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from hasar.eval.building import building_classes, building_f1
from hasar.eval.calibration import entropy

#: Damage classes of the BRIGHT label (1 intact, 2 damaged, 3 destroyed).
DAMAGE_CLASSES = (2, 3)

#: Default analyst budgets as fractions of the building count.
DEFAULT_BUDGETS = (0.05, 0.10, 0.20)

#: Default ranking names; ``dpm`` is skipped when no record carries a DPM value.
DEFAULT_RANKINGS = ("belirsizlik", "hasar-olasiligi", "dpm", "rastgele")

#: Number of random draws averaged for the ``rastgele`` baseline.
DEFAULT_N_RANDOM = 100


@dataclass(frozen=True)
class BuildingRecord:
    """One building with its prediction, uncertainty and optional external signals.

    ``uncertainty`` is the mean flip-TTA entropy inside the instance, ``p_damage`` the mean
    probability of the damaged or destroyed class and ``dpm`` the mean damage proxy map value
    (``None`` when the radar layer is unavailable).
    """

    tile: str
    instance: int
    true_cls: int
    pred_cls: int
    uncertainty: float
    p_damage: float
    dpm: float | None = None


def building_records(
    tile: str,
    label: np.ndarray,
    instance_map: np.ndarray,
    clf_tta: np.ndarray,
    dpm: np.ndarray | None = None,
) -> list[BuildingRecord]:
    """Return one :class:`BuildingRecord` per labelled instance of a tile.

    ``clf_tta`` is the ``(C, H, W)`` flip-TTA mean probability map. The true and predicted classes
    come from :func:`hasar.eval.building.building_classes` (majority vote, instances without a
    label pixel skipped); the uncertainty, damage probability and DPM are averaged over every
    pixel of the instance.
    """
    label = np.asarray(label)
    instance_map = np.asarray(instance_map)
    clf_tta = np.asarray(clf_tta, dtype=np.float64)

    num_instances = int(instance_map.max()) if instance_map.size else 0
    if num_instances == 0:
        return []

    true_cls, pred_cls = building_classes(label, instance_map, clf_tta.argmax(axis=0))

    valid = label > 0
    present = np.bincount(instance_map[valid].ravel(), minlength=num_instances + 1) > 0
    present[0] = False
    ids = np.flatnonzero(present)

    flat_instances = instance_map.ravel()
    counts = np.bincount(flat_instances, minlength=num_instances + 1)

    pixel_entropy = entropy(clf_tta.reshape(clf_tta.shape[0], -1)).reshape(instance_map.shape)
    entropy_sums = np.bincount(flat_instances, weights=pixel_entropy.ravel(), minlength=num_instances + 1)

    p_damage_map = clf_tta[2:4].sum(axis=0)
    damage_sums = np.bincount(flat_instances, weights=p_damage_map.ravel(), minlength=num_instances + 1)

    dpm_sums = None
    if dpm is not None:
        dpm = np.asarray(dpm, dtype=np.float64)
        dpm_sums = np.bincount(flat_instances, weights=dpm.ravel(), minlength=num_instances + 1)

    records = []
    for index, instance in enumerate(ids):
        count = counts[instance]
        records.append(
            BuildingRecord(
                tile=tile,
                instance=int(instance),
                true_cls=int(true_cls[index]),
                pred_cls=int(pred_cls[index]),
                uncertainty=float(entropy_sums[instance] / count),
                p_damage=float(damage_sums[instance] / count),
                dpm=float(dpm_sums[instance] / count) if dpm_sums is not None else None,
            )
        )
    return records


def risk_coverage(records: list[BuildingRecord]) -> tuple[np.ndarray, np.ndarray, float]:
    """Return the risk-coverage curve and its area (AURC) for an uncertainty ranking.

    Buildings are accepted from the least to the most uncertain (a stable sort, so ties keep their
    input order). ``coverage`` is the accepted fraction and ``risk`` the error rate among the
    accepted buildings; ``aurc`` is the mean of ``risk`` over the curve.
    """
    n = len(records)
    if n == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64), 0.0

    uncertainties = np.array([item.uncertainty for item in records], dtype=np.float64)
    errors = np.array([item.true_cls != item.pred_cls for item in records], dtype=np.float64)
    order = np.argsort(uncertainties, kind="stable")

    cumulative = np.cumsum(errors[order])
    reviewed = np.arange(1, n + 1, dtype=np.float64)
    coverage = reviewed / n
    risk = cumulative / reviewed
    return coverage, risk, float(risk.mean())


def _budget_metrics(records: list[BuildingRecord], reviewed: np.ndarray) -> dict[str, float]:
    """Metrics after the analyst corrects the predictions of the ``reviewed`` buildings."""
    true_cls = np.array([item.true_cls for item in records], dtype=np.int32)
    initial = np.array([item.pred_cls for item in records], dtype=np.int32)
    pred_cls = initial.copy()
    pred_cls[reviewed] = true_cls[reviewed]

    errors = initial != true_cls
    total_errors = int(errors.sum())
    fixed = int(errors[reviewed].sum())
    errors_fixed = fixed / total_errors if total_errors else 1.0

    damaged = np.isin(true_cls, DAMAGE_CLASSES)
    found = int(np.sum(damaged & np.isin(pred_cls, DAMAGE_CLASSES)))
    damaged_recall = found / int(damaged.sum()) if damaged.any() else 0.0

    metrics = building_f1(true_cls, pred_cls)
    return {
        "macro_f1": float(metrics["macro_f1"]),
        "errors_fixed": float(errors_fixed),
        "damaged_recall": float(damaged_recall),
    }


def _ranking_score(item: BuildingRecord, ranking: str) -> tuple[bool, float]:
    """Return a sort key: missing values last, then the highest score first."""
    if ranking == "belirsizlik":
        return False, -item.uncertainty
    if ranking == "hasar-olasiligi":
        return False, -item.p_damage
    if ranking == "dpm":
        if item.dpm is None:
            return True, 0.0
        return False, -item.dpm
    raise ValueError(f"bilinmeyen sıralama: {ranking!r}")


def _ranking_order(records: list[BuildingRecord], ranking: str) -> list[int]:
    """Return the building indices from the highest to the lowest ranking score."""
    return sorted(range(len(records)), key=lambda index: _ranking_score(records[index], ranking))


def _random_metrics(
    records: list[BuildingRecord], reviewed: int, n_random: int, rng: np.random.Generator
) -> dict[str, float]:
    """Average the budget metrics over ``n_random`` random review sets of size ``reviewed``."""
    totals = {"macro_f1": 0.0, "errors_fixed": 0.0, "damaged_recall": 0.0}
    for _ in range(n_random):
        indices = rng.choice(len(records), size=reviewed, replace=False)
        metrics = _budget_metrics(records, indices)
        for name in totals:
            totals[name] += metrics[name]
    return {name: value / n_random for name, value in totals.items()}


def analyst_budget(
    records: list[BuildingRecord],
    budgets: tuple[float, ...] = DEFAULT_BUDGETS,
    rankings: tuple[str, ...] = DEFAULT_RANKINGS,
    n_random: int = DEFAULT_N_RANDOM,
    seed: int = 0,
) -> dict[str, dict[str, dict[str, float]]]:
    """Return the analyst-budget table: ``ranking -> budget -> metrics``.

    A budget is the fraction of buildings the analyst reviews; the reviewed buildings are the
    top of the ranking and their prediction is replaced by the true class. ``rastgele`` averages
    ``n_random`` random review sets drawn with ``seed``. The ``dpm`` ranking is skipped when no
    record carries a DPM value.
    """
    records = list(records)
    if not records:
        return {}
    for budget in budgets:
        if not 0.0 < budget <= 1.0:
            raise ValueError(f"bütçe (0, 1] aralığında olmalı, alınan: {budget!r}")
    if n_random < 1:
        raise ValueError(f"n_random pozitif olmalı, alınan: {n_random!r}")

    n = len(records)
    has_dpm = any(item.dpm is not None for item in records)
    rng = np.random.default_rng(seed)

    result: dict[str, dict[str, dict[str, float]]] = {}
    for ranking in rankings:
        if ranking == "dpm" and not has_dpm:
            continue
        entry: dict[str, dict[str, float]] = {}
        for budget in budgets:
            reviewed = min(max(1, math.ceil(budget * n)), n)
            if ranking == "rastgele":
                metrics = _random_metrics(records, reviewed, n_random, rng)
            else:
                order = _ranking_order(records, ranking)
                metrics = _budget_metrics(records, np.array(order[:reviewed], dtype=np.int64))
            entry[f"{budget:g}"] = metrics
        result[ranking] = entry
    return result
