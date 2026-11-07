"""Model sözleşmesi ve kayıt defteri testleri (torch gerektirmez)."""

import uuid
from collections.abc import Mapping

import numpy as np
import pytest

from hasar.models.base import (
    NUM_CLF_CLASSES,
    NUM_LOC_CLASSES,
    OUTPUT_KEYS,
    DamageModel,
    validate_output,
)
from hasar.models.registry import available, build, register


def _unique_name(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def _valid_output(batch: int = 2, height: int = 5, width: int = 7) -> dict[str, np.ndarray]:
    return {
        "loc": np.zeros((batch, NUM_LOC_CLASSES, height, width), dtype=np.float32),
        "clf": np.zeros((batch, NUM_CLF_CLASSES, height, width), dtype=np.float32),
    }


# --- sabitler -------------------------------------------------------------


def test_output_keys() -> None:
    assert OUTPUT_KEYS == ("loc", "clf")


def test_class_counts() -> None:
    assert NUM_LOC_CLASSES == 2
    assert NUM_CLF_CLASSES == 4


# --- DamageModel protokolü ------------------------------------------------


def test_protocol_is_runtime_checkable() -> None:
    class PlainModel:
        def forward(self, pre, post) -> Mapping:
            return _valid_output()

    assert isinstance(PlainModel(), DamageModel)


def test_object_without_forward_is_not_a_model() -> None:
    class NotAModel:
        def predict(self, pre, post) -> Mapping:
            return _valid_output()

    assert not isinstance(NotAModel(), DamageModel)


# --- validate_output ------------------------------------------------------


def test_validate_output_accepts_valid_shapes() -> None:
    assert validate_output(_valid_output(batch=3, height=4, width=6), 3, 4, 6) is None


def test_validate_output_rejects_missing_key() -> None:
    out = _valid_output()
    del out["clf"]
    with pytest.raises(ValueError, match="clf"):
        validate_output(out, 2, 5, 7)


def test_validate_output_rejects_extra_key() -> None:
    out = _valid_output()
    out["extra"] = np.zeros((2, 5, 7), dtype=np.float32)
    with pytest.raises(ValueError, match="extra"):
        validate_output(out, 2, 5, 7)


def test_validate_output_rejects_wrong_loc_shape() -> None:
    out = _valid_output()
    out["loc"] = np.zeros((2, 3, 5, 7), dtype=np.float32)
    with pytest.raises(ValueError, match="loc"):
        validate_output(out, 2, 5, 7)


def test_validate_output_rejects_wrong_clf_shape() -> None:
    out = _valid_output()
    out["clf"] = np.zeros((2, 4, 5, 6), dtype=np.float32)
    with pytest.raises(ValueError, match="clf"):
        validate_output(out, 2, 5, 7)


def test_validate_output_rejects_wrong_batch() -> None:
    out = _valid_output(batch=1)
    with pytest.raises(ValueError, match="loc|clf"):
        validate_output(out, 2, 5, 7)


# --- kayıt defteri --------------------------------------------------------


def test_register_returns_factory_unchanged() -> None:
    name = _unique_name("model")

    class Factory:
        pass

    assert register(name)(Factory) is Factory


def test_register_duplicate_name_raises() -> None:
    name = _unique_name("model")

    def factory() -> object:
        return object()

    register(name)(factory)
    with pytest.raises(ValueError, match=name):
        register(name)(factory)


def test_build_calls_factory_with_kwargs() -> None:
    name = _unique_name("model")

    class Factory:
        def __init__(self, depth: int, width: int) -> None:
            self.depth = depth
            self.width = width

    register(name)(Factory)
    model = build(name, depth=3, width=16)
    assert isinstance(model, Factory)
    assert (model.depth, model.width) == (3, 16)


def test_build_unknown_name_raises_key_error_listing_names() -> None:
    known = _unique_name("model")

    def factory() -> object:
        return object()

    register(known)(factory)
    missing = _unique_name("missing")
    with pytest.raises(KeyError) as excinfo:
        build(missing)
    message = str(excinfo.value)
    assert missing in message
    assert known in message


def test_available_is_sorted_and_contains_registered() -> None:
    first = _unique_name("a-model")
    second = _unique_name("b-model")

    def factory() -> object:
        return object()

    register(first)(factory)
    register(second)(factory)
    names = available()
    assert names == sorted(names)
    assert first in names
    assert second in names
