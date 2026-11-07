"""Tests for the pre-declared evaluation rows (``hasar.eval.rows``).

Protocol section 6 declares every secondary row before the single Turkey run. The module turns
one tile's cached probabilities into the prediction of each row, so these tests build a tiny
hand-checked 4x4 tile and compare every row against a prediction written out by hand. Nothing
touches the network or the real BRIGHT data set.
"""

from __future__ import annotations

import numpy as np

from hasar.eval.metrics import confusion_matrix
from hasar.eval.rows import ROWS, row_confusions, row_predictions

# --------------------------------------------------------------------------- hand-built tile

# argmax(clf) of the plain pass. (0, 0) is a pixel where the classifier says background but the
# localisation head says building; (0, 2) is the opposite: loc background, clf destroyed.
CLF_CLASSES = np.array(
    [
        [0, 2, 3, 0],
        [2, 3, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 3, 3],
    ],
    dtype=np.int64,
)

# argmax(loc): two separate building components, A at the top left (2x2) and B at the bottom
# right (an L of three pixels). They are not 8-connected to each other.
LOC_CLASSES = np.array(
    [
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 1],
    ],
    dtype=np.int64,
)

# argmax(clf_tta): the flip-TTA mean disagrees with the plain pass everywhere.
TTA_CLASSES = np.ones((4, 4), dtype=np.int64)

LABEL = np.array(
    [
        [0, 1, 2, 3],
        [1, 2, 3, 0],
        [2, 3, 0, 1],
        [3, 0, 1, 2],
    ],
    dtype=np.int64,
)

# (3, 3) is nodata. Its label stays a valid class, so only the 'nodata-yok' row drops it.
VALID = np.ones((4, 4), dtype=bool)
VALID[3, 3] = False

HEADLINE = CLF_CLASSES
FLIP_TTA = TTA_CLASSES
BINA_MASKELI = np.array(
    [
        [1, 2, 0, 0],
        [2, 3, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 3, 3],
    ],
    dtype=np.int64,
)
# Component A has building classes 1, 2, 2, 3 -> majority 2; component B has 1, 3, 3 -> 3.
BINA_OYAMA = np.array(
    [
        [2, 2, 0, 0],
        [2, 2, 0, 0],
        [0, 0, 0, 3],
        [0, 0, 3, 3],
    ],
    dtype=np.int64,
)

EXPECTED = {
    "headline": HEADLINE,
    "flip-tta": FLIP_TTA,
    "bina-maskeli": BINA_MASKELI,
    "bina-oylama": BINA_OYAMA,
}


def probabilities(class_map: np.ndarray, *, background_pixel: tuple[int, int] | None = None) -> np.ndarray:
    """Return a (4, H, W) probability map whose argmax is ``class_map``.

    ``background_pixel`` overrides one pixel with scores where background wins but the best
    building class is intact (1), so ``argmax(clf[1:]) + 1`` is visible.
    """
    clf = np.full((4, *class_map.shape), 0.05, dtype=np.float32)
    for class_id in range(4):
        clf[class_id][class_map == class_id] = 0.9
    if background_pixel is not None:
        row, col = background_pixel
        clf[:, row, col] = [0.5, 0.4, 0.3, 0.2]
    return clf


def tile() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the (clf, loc, clf_tta) probabilities of the hand-built tile."""
    clf = probabilities(CLF_CLASSES, background_pixel=(0, 0))
    loc = probabilities(LOC_CLASSES)
    clf_tta = probabilities(TTA_CLASSES)
    return clf, loc, clf_tta


# ------------------------------------------------------------------------------- ROWS


def test_rows_are_the_predeclared_protocol_rows() -> None:
    assert ROWS == ("headline", "flip-tta", "bina-maskeli", "bina-oylama", "nodata-yok")


# ------------------------------------------------------------------------ row_predictions


def test_row_predictions_match_hand_built_maps() -> None:
    clf, loc, clf_tta = tile()

    preds = row_predictions(clf, loc, clf_tta, VALID)

    assert set(preds) == set(EXPECTED)
    for name, expected in EXPECTED.items():
        np.testing.assert_array_equal(preds[name], expected, err_msg=name)


def test_row_predictions_are_int64_tiles() -> None:
    clf, loc, clf_tta = tile()

    preds = row_predictions(clf, loc, clf_tta, VALID)

    for name, pred in preds.items():
        assert pred.shape == (4, 4), name
        assert pred.dtype == np.int64, name


def test_headline_ignores_localisation_and_tta() -> None:
    clf, loc, clf_tta = tile()

    preds = row_predictions(clf, loc, clf_tta, VALID)

    # (0, 2): loc says background, clf says destroyed -> headline 3, bina-maskeli 0.
    assert preds["headline"][0, 2] == 3
    assert preds["bina-maskeli"][0, 2] == 0
    # (0, 0): loc says building, clf says background -> headline 0, bina-maskeli intact.
    assert preds["headline"][0, 0] == 0
    assert preds["bina-maskeli"][0, 0] == 1


def test_bina_oylama_component_majority() -> None:
    clf, loc, clf_tta = tile()

    preds = row_predictions(clf, loc, clf_tta, VALID)

    # Component A (top-left 2x2) has clf majority 'damaged' -> every pixel 2.
    np.testing.assert_array_equal(preds["bina-oylama"][0:2, 0:2], np.full((2, 2), 2))
    # Component B (the bottom-right L) has clf majority 'destroyed' -> every pixel 3.
    assert preds["bina-oylama"][2, 3] == 3
    assert preds["bina-oylama"][3, 2] == 3
    assert preds["bina-oylama"][3, 3] == 3
    # Background stays 0.
    assert preds["bina-oylama"][0, 2] == 0
    assert preds["bina-oylama"][3, 0] == 0


def test_float16_probabilities_are_accepted() -> None:
    clf, loc, clf_tta = tile()

    preds32 = row_predictions(clf, loc, clf_tta, VALID)
    preds16 = row_predictions(clf.astype(np.float16), loc.astype(np.float16), clf_tta.astype(np.float16), VALID)

    assert set(preds16) == set(preds32)
    for name in preds32:
        np.testing.assert_array_equal(preds16[name], preds32[name], err_msg=name)


# ------------------------------------------------------------------------ row_confusions


def test_row_confusions_match_hand_built_predictions() -> None:
    clf, loc, clf_tta = tile()

    confusions = row_confusions(LABEL, clf, loc, clf_tta, VALID)

    assert set(confusions) == set(ROWS)
    for name, expected_pred in EXPECTED.items():
        expected = confusion_matrix(LABEL, expected_pred)
        np.testing.assert_array_equal(confusions[name], expected, err_msg=name)


def test_nodata_pixel_is_dropped_only_in_nodata_yok() -> None:
    clf, loc, clf_tta = tile()

    confusions = row_confusions(LABEL, clf, loc, clf_tta, VALID)

    # Every other row counts all 16 labelled pixels, nodata included.
    for name in ("headline", "flip-tta", "bina-maskeli", "bina-oylama"):
        assert int(confusions[name].sum()) == 16, name
    # 'nodata-yok' uses the headline prediction but drops the one nodata pixel.
    assert int(confusions["nodata-yok"].sum()) == 15
    np.testing.assert_array_equal(confusions["nodata-yok"], confusion_matrix(LABEL, HEADLINE) - _nodata_cell())


def _nodata_cell() -> np.ndarray:
    """Return the 4x4 matrix holding only the (3, 3) nodata pixel of the headline row."""
    cell = np.zeros((4, 4), dtype=np.int64)
    cell[LABEL[3, 3], HEADLINE[3, 3]] += 1
    return cell


def test_confusions_are_int64_4x4() -> None:
    clf, loc, clf_tta = tile()

    confusions = row_confusions(LABEL, clf, loc, clf_tta, VALID)

    for name, cm in confusions.items():
        assert cm.shape == (4, 4), name
        assert cm.dtype == np.int64, name
