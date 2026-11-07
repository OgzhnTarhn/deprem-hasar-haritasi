"""Tests for the uncertainty / analyst-budget analysis (``hasar.analysis.uncertainty``).

Phase 7 Y2: building-level uncertainty from the flip-TTA entropy, the risk-coverage curve and the
analyst-budget table. Everything here is tiny synthetic numpy data; no torch, no dataset, no
network, no Turkey imagery or labels.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from hasar.analysis.uncertainty import (
    BuildingRecord,
    analyst_budget,
    building_records,
    risk_coverage,
)


def record(
    uncertainty: float,
    true_cls: int,
    pred_cls: int,
    *,
    p_damage: float = 0.0,
    dpm: float | None = None,
    instance: int = 0,
    tile: str = "tile",
) -> BuildingRecord:
    """Return a ``BuildingRecord`` with the fields the tests care about."""
    return BuildingRecord(
        tile=tile,
        instance=instance,
        true_cls=true_cls,
        pred_cls=pred_cls,
        uncertainty=uncertainty,
        p_damage=p_damage,
        dpm=dpm,
    )


def budget_metrics(result: dict, ranking: str, budget: float) -> dict[str, float]:
    """Look up one ``(ranking, budget)`` cell, tolerating ``"1.0"`` vs ``"1"`` keys."""
    entry = result[ranking]
    for key in (str(budget), f"{budget:g}"):
        if key in entry:
            return entry[key]
    raise AssertionError(f"budget {budget} missing from {sorted(entry)}")


# --------------------------------------------------------------------------- BuildingRecord


def test_building_record_is_frozen_dataclass() -> None:
    assert dataclasses.is_dataclass(BuildingRecord)
    item = record(0.5, 2, 1, p_damage=0.75, dpm=0.3, instance=4, tile="t")
    assert item.tile == "t"
    assert item.instance == 4
    assert item.true_cls == 2
    assert item.pred_cls == 1
    assert item.uncertainty == 0.5
    assert item.p_damage == 0.75
    assert item.dpm == 0.3
    with pytest.raises(dataclasses.FrozenInstanceError):
        item.pred_cls = 2  # type: ignore[misc]


def test_building_record_dpm_defaults_to_none() -> None:
    assert record(0.0, 1, 1).dpm is None


# ------------------------------------------------------------------------- building_records


def _six_by_six() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Two labelled instances plus one instance without any label pixel.

    Instance 1 (rows 0-1, cols 0-2): five pixels labelled 2 and one background pixel. The five
    labelled pixels are one-hot class 2; the background pixel is uniform, so its argmax is 0.
    Instance 2 (rows 3-4, cols 0-2): six pixels labelled 3, all one-hot class 1 (a miss).
    Instance 3 (pixel 5, 5): no label pixel, so it must be skipped.
    """
    instance_map = np.zeros((6, 6), dtype=np.int32)
    instance_map[0:2, 0:3] = 1
    instance_map[3:5, 0:3] = 2
    instance_map[5, 5] = 3

    label = np.zeros((6, 6), dtype=np.int32)
    label[0:2, 0:3] = 2
    label[0, 0] = 0
    label[3:5, 0:3] = 3

    clf_tta = np.full((4, 6, 6), 0.25, dtype=np.float64)
    clf_tta[:, 0:2, 0:3] = 0.0
    clf_tta[2, 0:2, 0:3] = 1.0
    clf_tta[:, 0, 0] = 0.25  # uniform -> argmax 0, entropy ln 4
    clf_tta[:, 3:5, 0:3] = 0.0
    clf_tta[1, 3:5, 0:3] = 1.0

    dpm = np.zeros((6, 6), dtype=np.float64)
    dpm[0:2, 0:3] = 0.2
    dpm[3:5, 0:3] = 0.8
    return label, instance_map, clf_tta, dpm


def test_building_records_two_instances() -> None:
    label, instance_map, clf_tta, dpm = _six_by_six()

    records = building_records("tile-a", label, instance_map, clf_tta, dpm=dpm)

    assert len(records) == 2
    assert all(isinstance(item, BuildingRecord) for item in records)
    assert [item.instance for item in records] == [1, 2]
    assert all(item.tile == "tile-a" for item in records)

    first, second = records
    assert first.true_cls == 2
    assert first.pred_cls == 2
    assert first.uncertainty == pytest.approx(math.log(4.0) / 6.0)
    assert first.p_damage == pytest.approx(5.5 / 6.0)
    assert first.dpm == pytest.approx(0.2)

    assert second.true_cls == 3
    assert second.pred_cls == 1
    assert second.uncertainty == pytest.approx(0.0)
    assert second.p_damage == pytest.approx(0.0)
    assert second.dpm == pytest.approx(0.8)


def test_building_records_without_dpm() -> None:
    label, instance_map, clf_tta, _ = _six_by_six()

    records = building_records("tile-a", label, instance_map, clf_tta)

    assert len(records) == 2
    assert all(item.dpm is None for item in records)


def test_building_records_skips_instances_without_label() -> None:
    label, instance_map, clf_tta, _ = _six_by_six()

    records = building_records("tile-a", label, instance_map, clf_tta)

    assert 3 not in [item.instance for item in records]


# ---------------------------------------------------------------------------- risk_coverage


def test_risk_coverage_known_values() -> None:
    records = [
        record(0.1, 1, 1),
        record(0.2, 2, 2),
        record(0.9, 3, 1),
        record(1.0, 1, 2),
    ]

    coverage, risk, aurc = risk_coverage(records)

    np.testing.assert_allclose(coverage, np.array([0.25, 0.5, 0.75, 1.0]))
    np.testing.assert_allclose(risk, np.array([0.0, 0.0, 1.0 / 3.0, 0.5]))
    assert aurc == pytest.approx((0.0 + 0.0 + 1.0 / 3.0 + 0.5) / 4.0)


def test_risk_coverage_perfect_ranking_beats_reversed() -> None:
    perfect = [
        record(0.1, 1, 1),
        record(0.2, 2, 2),
        record(0.9, 3, 1),
        record(1.0, 1, 2),
    ]
    reversed_ranking = [
        record(0.1, 3, 1),
        record(0.2, 1, 2),
        record(0.9, 1, 1),
        record(1.0, 2, 2),
    ]

    _, _, aurc_perfect = risk_coverage(perfect)
    _, _, aurc_reversed = risk_coverage(reversed_ranking)

    assert aurc_perfect < aurc_reversed
    assert aurc_perfect == pytest.approx(5.0 / 24.0)
    assert aurc_reversed == pytest.approx(19.0 / 24.0)


def test_risk_coverage_sort_is_stable() -> None:
    correct_first = [record(0.5, 1, 1), record(0.5, 2, 1)]
    error_first = [record(0.5, 2, 1), record(0.5, 1, 1)]

    _, risk_correct_first, _ = risk_coverage(correct_first)
    _, risk_error_first, _ = risk_coverage(error_first)

    assert risk_correct_first[0] == pytest.approx(0.0)
    assert risk_error_first[0] == pytest.approx(1.0)


# --------------------------------------------------------------------------- analyst_budget


def _budget_records() -> list[BuildingRecord]:
    """Ten buildings, two errors, and the errors are exactly the most uncertain ones."""
    return [
        record(0.90, 2, 1, p_damage=0.9, dpm=0.9, instance=0),
        record(1.00, 3, 1, p_damage=0.8, dpm=0.8, instance=1),
        record(0.10, 1, 1, p_damage=0.1, dpm=0.1, instance=2),
        record(0.20, 2, 2, p_damage=0.2, dpm=0.2, instance=3),
        record(0.30, 3, 3, p_damage=0.3, dpm=0.3, instance=4),
        record(0.40, 1, 1, p_damage=0.4, dpm=0.4, instance=5),
        record(0.50, 2, 2, p_damage=0.5, dpm=0.5, instance=6),
        record(0.60, 3, 3, p_damage=0.6, dpm=0.6, instance=7),
        record(0.70, 1, 1, p_damage=0.7, dpm=0.7, instance=8),
        record(0.80, 2, 2, p_damage=0.8, dpm=0.8, instance=9),
    ]


def test_analyst_budget_full_budget_reaches_perfect_f1() -> None:
    records = _budget_records()

    result = analyst_budget(
        records,
        budgets=(1.0,),
        rankings=("belirsizlik", "hasar-olasiligi", "dpm", "rastgele"),
    )

    for ranking in ("belirsizlik", "hasar-olasiligi", "dpm", "rastgele"):
        metrics = budget_metrics(result, ranking, 1.0)
        assert metrics["macro_f1"] == pytest.approx(1.0)
        assert metrics["errors_fixed"] == pytest.approx(1.0)
        assert metrics["damaged_recall"] == pytest.approx(1.0)


def test_analyst_budget_uncertainty_beats_random() -> None:
    records = _budget_records()

    result = analyst_budget(
        records,
        budgets=(0.2,),
        rankings=("belirsizlik", "rastgele"),
        n_random=100,
        seed=0,
    )

    uncertainty = budget_metrics(result, "belirsizlik", 0.2)
    random = budget_metrics(result, "rastgele", 0.2)

    assert uncertainty["errors_fixed"] == pytest.approx(1.0)
    assert uncertainty["macro_f1"] == pytest.approx(1.0)
    assert random["errors_fixed"] < uncertainty["errors_fixed"]


def test_analyst_budget_random_is_deterministic() -> None:
    records = _budget_records()

    first = analyst_budget(records, budgets=(0.2,), rankings=("rastgele",), n_random=50, seed=7)
    second = analyst_budget(records, budgets=(0.2,), rankings=("rastgele",), n_random=50, seed=7)

    assert first == second


def test_analyst_budget_rejects_zero_budget() -> None:
    with pytest.raises(ValueError):
        analyst_budget(_budget_records(), budgets=(0.0,))


def test_analyst_budget_skips_dpm_when_absent() -> None:
    records = [record(0.1, 1, 1), record(0.2, 2, 2), record(0.3, 3, 3)]

    result = analyst_budget(records)

    assert "dpm" not in result
    assert "belirsizlik" in result
    assert "hasar-olasiligi" in result
    assert "rastgele" in result


def test_analyst_budget_rankings_pick_different_buildings() -> None:
    """One review each: dpm picks a correct building, the other rankings pick an error."""
    records = [
        record(0.1, 1, 1, p_damage=0.1, dpm=0.9, instance=0),  # highest dpm, already correct
        record(0.2, 3, 1, p_damage=0.9, dpm=0.5, instance=1),  # highest p_damage, error
        record(0.3, 2, 2, p_damage=0.2, dpm=0.1, instance=2),  # correct
        record(0.9, 2, 1, p_damage=0.3, dpm=None, instance=3),  # most uncertain, error
    ]

    result = analyst_budget(
        records,
        budgets=(0.25,),
        rankings=("belirsizlik", "hasar-olasiligi", "dpm"),
    )

    dpm = budget_metrics(result, "dpm", 0.25)
    damage = budget_metrics(result, "hasar-olasiligi", 0.25)
    uncertainty = budget_metrics(result, "belirsizlik", 0.25)

    # Two initial errors out of four buildings.
    assert dpm["errors_fixed"] == pytest.approx(0.0)
    assert damage["errors_fixed"] == pytest.approx(0.5)
    assert uncertainty["errors_fixed"] == pytest.approx(0.5)

    # Damaged recall counts true classes {2, 3} found when the prediction is in {2, 3}.
    assert dpm["damaged_recall"] == pytest.approx(1.0 / 3.0)
    assert damage["damaged_recall"] == pytest.approx(2.0 / 3.0)
    assert uncertainty["damaged_recall"] == pytest.approx(2.0 / 3.0)

    assert dpm["macro_f1"] == pytest.approx(7.0 / 18.0)
    assert damage["macro_f1"] == pytest.approx(7.0 / 9.0)
    assert uncertainty["macro_f1"] == pytest.approx(5.0 / 9.0)


def test_analyst_budget_keys_are_strings() -> None:
    result = analyst_budget(_budget_records(), budgets=(0.1, 0.2))

    for ranking in result:
        assert all(isinstance(key, str) for key in result[ranking])
