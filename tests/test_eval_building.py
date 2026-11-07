"""Tests for the instance-level building evaluation (``hasar.eval.building``).

Every test builds tiny synthetic arrays or a tiny COCO-like JSON file in ``tmp_path``; nothing
touches the network or the real BRIGHT data set.
"""

from __future__ import annotations

import codecs
import json

import numpy as np
import pytest

from hasar.eval.building import (
    building_classes,
    building_f1,
    load_instances,
    masked_prediction,
    parse_segmentation,
    rasterize_instances,
    vote_by_components,
)

SQUARE = [[2, 2, 6, 2, 6, 6, 2, 6]]


def annotation(seg, *, image_id: int = 1, category_id: int = 1, ann_id: int = 1) -> dict:
    """Return a minimal COCO-like annotation dict."""
    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": category_id,
        "segmentation": seg,
        "area": 16.0,
        "bbox": [2.0, 2.0, 4.0, 4.0],
        "iscrowd": 0,
    }


# --------------------------------------------------------------------------- load_instances


def test_load_instances_reads_bom_utf8_file(tmp_path) -> None:
    annotations = [annotation(SQUARE, ann_id=1), annotation(SQUARE, ann_id=2, category_id=2)]
    payload = {"images": [{"id": 1, "file_name": "tile.png"}], "annotations": annotations, "categories": []}
    path = tmp_path / "instances.json"
    path.write_bytes(codecs.BOM_UTF8 + json.dumps(payload).encode("utf-8"))

    loaded = load_instances(path)

    assert loaded == annotations
    assert len(loaded) == 2
    assert loaded[1]["category_id"] == 2


def test_load_instances_empty_annotations(tmp_path) -> None:
    path = tmp_path / "empty.json"
    path.write_bytes(codecs.BOM_UTF8 + json.dumps({"images": [], "annotations": [], "categories": []}).encode("utf-8"))

    assert load_instances(path) == []


# ---------------------------------------------------------------------- parse_segmentation


def test_parse_segmentation_from_json_string() -> None:
    polygons = parse_segmentation(json.dumps(SQUARE))

    assert len(polygons) == 1
    poly = polygons[0]
    assert isinstance(poly, np.ndarray)
    assert poly.shape == (4, 2)
    assert np.issubdtype(poly.dtype, np.floating)
    np.testing.assert_array_equal(poly, np.array([[2, 2], [6, 2], [6, 6], [2, 6]], dtype=np.float64))


def test_parse_segmentation_from_list() -> None:
    polygons = parse_segmentation(SQUARE)

    assert len(polygons) == 1
    np.testing.assert_array_equal(polygons[0], np.array([[2, 2], [6, 2], [6, 6], [2, 6]], dtype=np.float64))


def test_parse_segmentation_multiple_polygons() -> None:
    polygons = parse_segmentation(json.dumps([SQUARE[0], [0, 0, 1, 0, 1, 1]]))

    assert len(polygons) == 2
    assert polygons[0].shape == (4, 2)
    assert polygons[1].shape == (3, 2)


# --------------------------------------------------------------------- rasterize_instances


def test_rasterize_square_is_filled() -> None:
    instance_map = rasterize_instances([annotation(json.dumps(SQUARE))], shape=(10, 10))

    assert instance_map.shape == (10, 10)
    assert instance_map.dtype == np.int32
    # Interior of the square is painted, not just its outline.
    assert instance_map[4, 4] == 1
    assert instance_map[3, 3] == 1
    # Outside stays background.
    assert instance_map[0, 0] == 0
    assert instance_map[9, 9] == 0


def test_rasterize_later_instance_overwrites_earlier() -> None:
    first = annotation(json.dumps(SQUARE), ann_id=1)
    second = annotation(json.dumps([[4, 4, 8, 4, 8, 8, 4, 8]]), ann_id=2, category_id=2)

    instance_map = rasterize_instances([first, second], shape=(10, 10))

    # Instance index is 0-based, painted as i + 1.
    assert instance_map[2, 2] == 1
    assert instance_map[7, 7] == 2
    # Overlap region belongs to the later instance.
    assert instance_map[5, 5] == 2


def test_rasterize_accepts_list_segmentation() -> None:
    instance_map = rasterize_instances([annotation(SQUARE)], shape=(10, 10))

    assert instance_map[4, 4] == 1


# ----------------------------------------------------------------------- building_classes


def test_building_classes_majority_and_missed() -> None:
    instance_map = np.zeros((6, 6), dtype=np.int32)
    instance_map[0:2, 0:2] = 1  # building 1: correctly predicted as damaged
    instance_map[0:2, 3:5] = 2  # building 2: missed by the localisation head
    instance_map[4:6, 0:2] = 3  # building 3: no valid label pixels -> skipped

    label = np.zeros((6, 6), dtype=np.int32)
    label[0:2, 0:2] = 2
    label[0:2, 3:5] = 3
    label[4:6, 0:2] = 0

    pred = np.zeros((6, 6), dtype=np.int32)
    pred[0:2, 0:2] = 2
    pred[0:2, 3:5] = 0

    true_cls, pred_cls = building_classes(label, instance_map, pred)

    assert np.issubdtype(true_cls.dtype, np.integer)
    assert np.issubdtype(pred_cls.dtype, np.integer)
    assert len(true_cls) == len(pred_cls) == 2
    np.testing.assert_array_equal(true_cls, np.array([2, 3]))
    np.testing.assert_array_equal(pred_cls, np.array([2, 0]))


def test_building_classes_ties_break_to_smaller_class() -> None:
    instance_map = np.zeros((4, 4), dtype=np.int32)
    instance_map[0:2, 0:2] = 1

    label = np.zeros((4, 4), dtype=np.int32)
    label[0:2, 0:2] = 1
    label[0, 0] = 2
    label[0, 1] = 2  # two intact, two damaged pixels -> tie, smaller class wins

    pred = np.zeros((4, 4), dtype=np.int32)
    pred[0:2, 0:2] = 1
    pred[0, 0] = 3
    pred[0, 1] = 3  # two intact, two destroyed pixels -> tie, smaller class wins

    true_cls, pred_cls = building_classes(label, instance_map, pred)

    np.testing.assert_array_equal(true_cls, np.array([1]))
    np.testing.assert_array_equal(pred_cls, np.array([1]))


# --------------------------------------------------------------------------- building_f1


def test_building_f1_all_correct() -> None:
    true_cls = np.array([1, 2, 3], dtype=np.int32)
    pred_cls = np.array([1, 2, 3], dtype=np.int32)

    metrics = building_f1(true_cls, pred_cls)

    assert metrics["f1_intact"] == pytest.approx(1.0)
    assert metrics["f1_damaged"] == pytest.approx(1.0)
    assert metrics["f1_destroyed"] == pytest.approx(1.0)
    assert metrics["macro_f1"] == pytest.approx(1.0)
    assert metrics["buildings"] == 3
    assert metrics["missed"] == 0


def test_building_f1_with_one_missed_building() -> None:
    true_cls = np.array([2, 3], dtype=np.int32)
    pred_cls = np.array([2, 0], dtype=np.int32)

    metrics = building_f1(true_cls, pred_cls)

    assert metrics["f1_damaged"] == pytest.approx(1.0)
    assert metrics["f1_destroyed"] == pytest.approx(0.0)
    assert metrics["f1_intact"] == pytest.approx(0.0)
    assert metrics["macro_f1"] == pytest.approx(1.0 / 3.0)
    assert metrics["buildings"] == 2
    assert metrics["missed"] == 1


# --------------------------------------------------------------------- masked_prediction


def test_masked_prediction_zeroes_non_building_pixels() -> None:
    loc_pred = np.zeros((3, 3), dtype=np.int32)
    loc_pred[0, 0] = 1
    loc_pred[1, 1] = 1

    clf_scores = np.zeros((4, 3, 3), dtype=np.float32)
    clf_scores[2, 0, 0] = 5.0  # class 2 at (0, 0)
    clf_scores[3, 1, 1] = 5.0  # class 3 at (1, 1)
    # A strong class-1 score outside the localisation mask must be ignored.
    clf_scores[1, 2, 2] = 9.0

    out = masked_prediction(loc_pred, clf_scores)

    assert out.shape == (3, 3)
    assert out[0, 0] == 2
    assert out[1, 1] == 3
    assert out[2, 2] == 0
    assert out[0, 1] == 0
    assert out[1, 0] == 0


# -------------------------------------------------------------------- vote_by_components


def test_vote_by_components_diagonal_is_one_component() -> None:
    loc_pred = np.zeros((4, 4), dtype=np.int32)
    loc_pred[0, 0] = 1
    loc_pred[1, 1] = 1
    loc_pred[2, 2] = 1

    clf_scores = np.zeros((4, 4, 4), dtype=np.float32)
    clf_scores[2, 0, 0] = 5.0  # class 2
    clf_scores[2, 1, 1] = 5.0  # class 2
    clf_scores[3, 2, 2] = 5.0  # class 3

    out = vote_by_components(loc_pred, clf_scores)

    assert out.shape == (4, 4)
    # The diagonal chain is 8-connected, so it is a single component with majority class 2.
    assert out[0, 0] == 2
    assert out[1, 1] == 2
    assert out[2, 2] == 2
    assert set(np.unique(out).tolist()) == {0, 2}


def test_vote_by_components_separate_components_vote_independently() -> None:
    loc_pred = np.zeros((5, 5), dtype=np.int32)
    loc_pred[0, 0] = 1
    loc_pred[0, 1] = 1
    loc_pred[4, 4] = 1

    clf_scores = np.zeros((4, 5, 5), dtype=np.float32)
    clf_scores[1, 0, 0] = 5.0  # class 1
    clf_scores[3, 0, 1] = 5.0  # class 3 -> tie with class 1, smaller class wins
    clf_scores[2, 4, 4] = 5.0  # class 2

    out = vote_by_components(loc_pred, clf_scores)

    assert out[0, 0] == 1
    assert out[0, 1] == 1
    assert out[4, 4] == 2
    assert out[2, 2] == 0


def test_building_rows_ignore_background_score_inside_mask() -> None:
    """Protocol section 6: inside the mask the class is argmax(clf[1:]) + 1, never background."""
    loc_pred = np.ones((2, 2), dtype=np.int32)
    clf_scores = np.zeros((4, 2, 2), dtype=np.float32)
    clf_scores[0] = 9.0  # background dominates everywhere
    clf_scores[2] = 1.0  # damaged is the best building class

    assert (masked_prediction(loc_pred, clf_scores) == 2).all()
    assert (vote_by_components(loc_pred, clf_scores) == 2).all()
