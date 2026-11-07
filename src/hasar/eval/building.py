"""Instance-level building evaluation (protocol sections 6-7).

The building-level macro F1 ``[ZS-EVENT bina-makro-F1]`` is computed on the CVPRW26 instance
polygons; the 1.001 Turkey tiles carry them. Connected components of the damage label are *not*
used as the building unit: they merge roughly 39% of adjacent buildings into one blob.

The secondary voting row uses the connected components of the model's *own* predicted building
mask, never the Turkey ground-truth polygons -- using the ground-truth instances there would leak
test labels into the prediction.

Majority voting is done with ``np.bincount`` per instance, vectorised as a single histogram over
``instance_id * 4 + class``. Ties break to the smaller class, because ``argmax`` returns the first
maximum.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

# Damage classes of the BRIGHT label: 0 background, 1 intact, 2 damaged, 3 destroyed.
NUM_CLASSES = 4
CLASS_NAMES = {1: "intact", 2: "damaged", 3: "destroyed"}

# All-ones 3x3 structure: diagonal neighbours belong to the same component (8-connectivity).
EIGHT_CONNECTED = np.ones((3, 3), dtype=bool)


def load_instances(path: str | Path) -> list[dict]:
    """Read the ``annotations`` list of a COCO-like instance file.

    The CVPRW26 files start with a UTF-8 BOM, hence ``utf-8-sig``.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return payload.get("annotations", [])


def parse_segmentation(segmentation: str | list) -> list[np.ndarray]:
    """Return the polygons of one annotation as ``(k, 2)`` float arrays.

    ``segmentation`` is either a JSON string (as stored in the COCO file) or an already decoded
    list of flat ``[x0, y0, x1, y1, ...]`` polygons.
    """
    polygons = json.loads(segmentation) if isinstance(segmentation, str) else segmentation
    if not polygons:
        return []
    if not isinstance(polygons[0], (list, tuple)):
        polygons = [polygons]
    return [np.asarray(poly, dtype=np.float64).reshape(-1, 2) for poly in polygons]


def rasterize_instances(annotations: list[dict], shape: tuple[int, int]) -> np.ndarray:
    """Paint the instance polygons into an ``int32`` map of shape ``shape``.

    Instance ``i`` (0-based) is painted as ``i + 1``; later instances overwrite earlier ones on
    overlap, and ``0`` is background.
    """
    height, width = shape
    image = Image.new("I", (width, height), 0)
    draw = ImageDraw.Draw(image)
    for i, annotation in enumerate(annotations):
        for polygon in parse_segmentation(annotation["segmentation"]):
            draw.polygon([tuple(xy) for xy in polygon], fill=i + 1, outline=i + 1)
    return np.asarray(image, dtype=np.int32)


def _majority(instance_map: np.ndarray, classes: np.ndarray, num_instances: int) -> np.ndarray:
    """Majority class per instance id (0..num_instances); ties break to the smaller class."""
    index = (instance_map.astype(np.int64) * NUM_CLASSES + classes.astype(np.int64)).ravel()
    counts = np.bincount(index, minlength=(num_instances + 1) * NUM_CLASSES)
    return counts.reshape(num_instances + 1, NUM_CLASSES).argmax(axis=1)


def building_classes(label: np.ndarray, instance_map: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the true and predicted damage class of every instance.

    Instances without a single valid label pixel (``label == 0``) are skipped. The predicted class
    is the majority of ``pred`` over the instance, so a building the localisation head missed gets
    class ``0``.
    """
    num_instances = int(instance_map.max()) if instance_map.size else 0
    if num_instances == 0:
        empty = np.zeros(0, dtype=np.int32)
        return empty, empty

    valid = label > 0
    true_majority = _majority(instance_map[valid], label[valid], num_instances)
    pred_majority = _majority(instance_map, pred, num_instances)

    present = np.bincount(instance_map[valid].ravel(), minlength=num_instances + 1) > 0
    present[0] = False
    ids = np.flatnonzero(present)
    return true_majority[ids].astype(np.int32), pred_majority[ids].astype(np.int32)


def building_f1(true_cls: np.ndarray, pred_cls: np.ndarray) -> dict[str, float]:
    """Per-class and macro F1 over buildings, plus the building and missed counts."""
    true_cls = np.asarray(true_cls)
    pred_cls = np.asarray(pred_cls)
    metrics: dict[str, float] = {}
    f1_scores = []
    for class_id, name in CLASS_NAMES.items():
        true_positive = int(np.sum((true_cls == class_id) & (pred_cls == class_id)))
        false_positive = int(np.sum((true_cls != class_id) & (pred_cls == class_id)))
        false_negative = int(np.sum((true_cls == class_id) & (pred_cls != class_id)))
        denominator = 2 * true_positive + false_positive + false_negative
        f1 = 2.0 * true_positive / denominator if denominator else 0.0
        metrics[f"f1_{name}"] = f1
        f1_scores.append(f1)
    metrics["macro_f1"] = float(np.mean(f1_scores))
    metrics["buildings"] = int(len(true_cls))
    metrics["missed"] = int(np.sum((true_cls > 0) & (pred_cls == 0)))
    return metrics


def building_class_map(clf_scores: np.ndarray) -> np.ndarray:
    """Return ``argmax(clf[1:]) + 1``: the best building class, background excluded.

    Protocol section 6: inside the predicted building mask a pixel is always one of the three
    building classes, even where the classifier's background score is the highest.
    """
    return (np.asarray(clf_scores)[1:].argmax(axis=0) + 1).astype(np.int32)


def masked_prediction(loc_pred: np.ndarray, clf_scores: np.ndarray) -> np.ndarray:
    """Best building class inside the predicted building mask, background outside it."""
    classes = building_class_map(clf_scores)
    return np.where(loc_pred > 0, classes, 0).astype(np.int32)


def vote_by_components(loc_pred: np.ndarray, clf_scores: np.ndarray) -> np.ndarray:
    """Majority-vote ``argmax(clf[1:]) + 1`` over the 8-connected components of the predicted mask.

    The components come from the model's own localisation output, never from ground-truth
    polygons. Every pixel of a component receives the component's majority class; background stays
    ``0``.
    """
    components, num_components = ndimage.label(loc_pred > 0, structure=EIGHT_CONNECTED)
    classes = building_class_map(clf_scores)
    majority = _majority(components, classes, int(num_components))
    return np.where(components > 0, majority[components], 0).astype(np.int32)
