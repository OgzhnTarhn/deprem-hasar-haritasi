"""Tests for the segmentation metrics (``hasar.eval.metrics``).

All arrays are tiny synthetic label/pred maps, so no data files and no network are needed. The
cross-check test loads the official BRIGHT reference evaluator by path and compares the two
confusion matrices and IoU vectors on seeded random inputs.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from hasar.eval.metrics import (
    CLASS_NAMES,
    EPS,
    NUM_CLASSES,
    ConfusionAccumulator,
    always_background_miou,
    building_miou,
    confusion_matrix,
    iou_per_class,
    miou,
    pixel_accuracy,
    summary,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = REPO_ROOT / "third_party" / "bright" / "models" / "metrics_reference.py"

HAND_CM = np.array(
    [
        [80, 3, 1, 0],
        [4, 6, 1, 0],
        [1, 1, 1, 0],
        [0, 0, 1, 1],
    ],
    dtype=np.int64,
)


def load_reference():
    """Load the official BRIGHT ``Evaluator`` from its file path."""
    spec = importlib.util.spec_from_file_location("bright_metrics_reference", REFERENCE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Evaluator


def random_pair(rng: np.random.Generator, shape: tuple[int, ...] = (64, 64)) -> tuple[np.ndarray, np.ndarray]:
    """Return a random label map (with 255 nodata) and a random prediction map."""
    label = rng.integers(0, 4, size=shape, dtype=np.int64)
    label[rng.random(shape) < 0.1] = 255
    pred = rng.integers(0, 4, size=shape, dtype=np.int64)
    return label, pred


# --------------------------------------------------------------------------- constants


def test_constants() -> None:
    assert NUM_CLASSES == 4
    assert EPS == 1e-7
    assert CLASS_NAMES == ("background", "intact", "damaged", "destroyed")


# ---------------------------------------------------------------------- hand example


def test_hand_example_iou_and_miou() -> None:
    ious = iou_per_class(HAND_CM)

    assert ious.dtype == np.float64
    assert ious.shape == (4,)
    np.testing.assert_allclose(ious, [80 / 89, 6 / 15, 1 / 6, 1 / 2], atol=1e-6)
    assert miou(HAND_CM) == pytest.approx(np.mean([80 / 89, 6 / 15, 1 / 6, 1 / 2]), abs=1e-6)
    assert miou(HAND_CM) == pytest.approx(0.4914, abs=1e-3)


def test_hand_example_pixel_accuracy_and_building_miou() -> None:
    assert pixel_accuracy(HAND_CM) == pytest.approx(0.88, abs=1e-6)
    assert building_miou(HAND_CM) == pytest.approx(np.mean([6 / 15, 1 / 6, 1 / 2]), abs=1e-6)


def test_hand_example_summary() -> None:
    out = summary(HAND_CM)

    assert set(out) == {
        "miou",
        "building_miou",
        "pixel_accuracy",
        "iou_background",
        "iou_intact",
        "iou_damaged",
        "iou_destroyed",
        "pixels",
    }
    assert isinstance(out["pixels"], int)
    assert out["pixels"] == 100
    float_keys = (
        "miou",
        "building_miou",
        "pixel_accuracy",
        "iou_background",
        "iou_intact",
        "iou_damaged",
        "iou_destroyed",
    )
    for key in float_keys:
        assert isinstance(out[key], float)
        assert 0.0 <= out[key] <= 1.0
    assert out["miou"] == pytest.approx(miou(HAND_CM), abs=1e-12)
    assert out["building_miou"] == pytest.approx(building_miou(HAND_CM), abs=1e-12)
    assert out["pixel_accuracy"] == pytest.approx(0.88, abs=1e-6)
    assert out["iou_background"] == pytest.approx(80 / 89, abs=1e-6)
    assert out["iou_intact"] == pytest.approx(6 / 15, abs=1e-6)
    assert out["iou_damaged"] == pytest.approx(1 / 6, abs=1e-6)
    assert out["iou_destroyed"] == pytest.approx(1 / 2, abs=1e-6)


# ------------------------------------------------------------------- perfect prediction


def test_perfect_prediction_gives_miou_one() -> None:
    rng = np.random.default_rng(0)
    label = rng.integers(0, 4, size=(32, 32), dtype=np.int64)
    # Every class is present, so no class is degenerate.
    assert set(np.unique(label).tolist()) == {0, 1, 2, 3}

    cm = confusion_matrix(label, label)

    assert cm.dtype == np.int64
    assert cm.shape == (4, 4)
    assert miou(cm) == pytest.approx(1.0, abs=1e-6)
    assert pixel_accuracy(cm) == pytest.approx(1.0, abs=1e-6)
    assert building_miou(cm) == pytest.approx(1.0, abs=1e-6)


# ----------------------------------------------------------------- all-background prediction


def test_all_background_prediction_matches_baseline() -> None:
    rng = np.random.default_rng(1)
    label = rng.integers(0, 4, size=(40, 40), dtype=np.int64)
    pred = np.zeros_like(label)

    cm = confusion_matrix(label, pred)
    background_fraction = float(np.count_nonzero(label == 0)) / label.size

    assert miou(cm) == pytest.approx(always_background_miou(background_fraction), abs=1e-6)
    assert always_background_miou(background_fraction) == pytest.approx(background_fraction / 4, abs=1e-12)


# ------------------------------------------------------------------------- nodata labels


def test_label_255_is_ignored() -> None:
    rng = np.random.default_rng(2)
    label, pred = random_pair(rng, shape=(16, 16))

    cm = confusion_matrix(label, pred)

    valid = (label >= 0) & (label < NUM_CLASSES)
    assert int(cm.sum()) == int(np.count_nonzero(valid))
    assert int(cm.sum()) < label.size


def test_batched_3d_input_matches_flat() -> None:
    rng = np.random.default_rng(3)
    label, pred = random_pair(rng, shape=(3, 8, 8))

    batched = confusion_matrix(label, pred)
    flat = confusion_matrix(label.reshape(-1), pred.reshape(-1))

    np.testing.assert_array_equal(batched, flat)


# ------------------------------------------------------------- cross-check with reference


def test_confusion_matrix_matches_reference() -> None:
    Evaluator = load_reference()
    rng = np.random.default_rng(4)

    for _ in range(3):
        label, pred = random_pair(rng)

        reference = Evaluator(4)
        reference.add_batch(label, pred)

        ours = confusion_matrix(label, pred)
        np.testing.assert_array_equal(ours, reference.confusion_matrix)
        np.testing.assert_allclose(iou_per_class(ours), reference.Intersection_over_Union(), atol=1e-9)


# --------------------------------------------------------------- event-level accumulation


def test_accumulator_sums_tiles_like_concatenation() -> None:
    rng = np.random.default_rng(5)
    label_a, pred_a = random_pair(rng, shape=(8, 8))
    label_b, pred_b = random_pair(rng, shape=(8, 8))

    acc = ConfusionAccumulator()
    acc.add(label_a, pred_a, key="tile_a")
    acc.add(label_b, pred_b, key="tile_b")

    assert acc.total.dtype == np.int64
    assert acc.total.shape == (4, 4)
    np.testing.assert_array_equal(acc.total, confusion_matrix(np.stack([label_a, label_b]), np.stack([pred_a, pred_b])))
    assert set(acc.per_tile) == {"tile_a", "tile_b"}
    np.testing.assert_array_equal(acc.per_tile["tile_a"], confusion_matrix(label_a, pred_a))
    np.testing.assert_array_equal(acc.per_tile["tile_b"], confusion_matrix(label_b, pred_b))


def test_accumulator_sums_repeated_key() -> None:
    rng = np.random.default_rng(6)
    label_a, pred_a = random_pair(rng, shape=(8, 8))
    label_b, pred_b = random_pair(rng, shape=(8, 8))

    acc = ConfusionAccumulator()
    acc.add(label_a, pred_a, key="tile")
    acc.add(label_b, pred_b, key="tile")

    expected = confusion_matrix(np.stack([label_a, label_b]), np.stack([pred_a, pred_b]))
    np.testing.assert_array_equal(acc.per_tile["tile"], expected)


def test_accumulator_save_load_roundtrip(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    label_a, pred_a = random_pair(rng, shape=(8, 8))
    label_b, pred_b = random_pair(rng, shape=(8, 8))

    acc = ConfusionAccumulator()
    acc.add(label_a, pred_a, key="tile_a")
    acc.add(label_b, pred_b, key="tile_b")

    path = tmp_path / "confusion.npz"
    acc.save_npz(path)
    assert path.exists()

    restored = ConfusionAccumulator.load_npz(path)

    np.testing.assert_array_equal(restored.total, acc.total)
    assert set(restored.per_tile) == set(acc.per_tile)
    for key in acc.per_tile:
        np.testing.assert_array_equal(restored.per_tile[key], acc.per_tile[key])


# ------------------------------------------------------------------------- error paths


def test_shape_mismatch_raises() -> None:
    label = np.zeros((4, 4), dtype=np.int64)
    pred = np.zeros((4, 5), dtype=np.int64)

    with pytest.raises(ValueError):
        confusion_matrix(label, pred)


def test_out_of_range_prediction_raises() -> None:
    label = np.zeros((4, 4), dtype=np.int64)
    pred = np.zeros((4, 4), dtype=np.int64)
    pred[0, 0] = 4

    with pytest.raises(ValueError):
        confusion_matrix(label, pred)
