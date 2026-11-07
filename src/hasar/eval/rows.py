"""Pre-declared secondary evaluation rows (protocol section 6).

Every row of the results table is declared *before* the ``protokol-kilidi`` tag is set, so no row
can be chosen after the single Turkey run. Nothing here is tuned on Turkey: the rows have no
settings at all, and each one is computed from the single-inference cache of protocol section 11
(``clf``, ``loc`` and the flip-TTA probabilities stored as ``float16``), never by running the model
again.

The rows are:

* ``headline`` -- ``argmax(clf)``, the locked headline path of protocol section 3;
* ``flip-tta`` -- ``argmax`` of the averaged four-view softmax probabilities;
* ``bina-maskeli`` -- background where ``argmax(loc)`` is background, ``argmax(clf[1:]) + 1``
  elsewhere;
* ``bina-oylama`` -- majority of ``argmax(clf[1:]) + 1`` over the 8-connected components of the
  predicted building mask (``argmax(loc)``), so the unit comes from the prediction and never from
  the Turkey label;
* ``nodata-yok`` -- the headline prediction with the nodata pixels dropped from the matrix.

The prediction logic lives in :mod:`hasar.eval.building` and the matrix in
:mod:`hasar.eval.metrics`; this module only names the rows and wires them together.
"""

from __future__ import annotations

import numpy as np

from hasar.eval.building import masked_prediction, vote_by_components
from hasar.eval.metrics import confusion_matrix

#: The pre-declared rows of protocol section 6, in table order.
ROWS = ("headline", "flip-tta", "bina-maskeli", "bina-oylama", "nodata-yok")


def row_predictions(
    clf: np.ndarray,
    loc: np.ndarray,
    clf_tta: np.ndarray,
    valid: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return the ``int64`` prediction map of every row that has one.

    ``clf``, ``loc`` and ``clf_tta`` are ``(4, H, W)`` probability maps of one tile from the
    single-inference cache; ``valid`` is the ``(H, W)`` nodata mask. The mask does not change any
    prediction -- only the ``nodata-yok`` row drops those pixels -- so it is accepted here to keep
    the call signature of :func:`row_confusions`.
    """
    loc_pred = np.asarray(loc).argmax(axis=0)
    return {
        "headline": np.asarray(clf).argmax(axis=0).astype(np.int64),
        "flip-tta": np.asarray(clf_tta).argmax(axis=0).astype(np.int64),
        "bina-maskeli": masked_prediction(loc_pred, clf).astype(np.int64),
        "bina-oylama": vote_by_components(loc_pred, clf).astype(np.int64),
    }


def row_confusions(
    label: np.ndarray,
    clf: np.ndarray,
    loc: np.ndarray,
    clf_tta: np.ndarray,
    valid: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return the ``int64`` 4x4 confusion matrix of every pre-declared row.

    ``label`` is the ``(H, W)`` damage label and ``valid`` the ``(H, W)`` nodata mask. Every row
    counts all labelled pixels, nodata included, exactly like the headline path; only
    ``nodata-yok`` drops the pixels where ``valid`` is false.
    """
    predictions = row_predictions(clf, loc, clf_tta, valid)
    confusions = {name: confusion_matrix(label, pred) for name, pred in predictions.items()}
    confusions["nodata-yok"] = confusion_matrix(np.asarray(label)[valid], predictions["headline"][valid])
    return confusions
