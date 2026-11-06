"""Tests for the result schema and JSON persistence (hasar.results)."""

from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from pathlib import Path

import pytest
from hasar.results import (
    BASE_TAGS,
    ResultRecord,
    ResultSchemaError,
    read_result,
    validate_metric_tag,
    write_result,
)

VALID_TAGS = [
    "[ZS-EVENT mIoU]",
    "[ZS-EVENT bina-makro-F1]",
    "[STD-EVENT mIoU]",
    "[STD-SET mIoU]",
    "[DFC-TEST mIoU]",
    "[CVPR-TEST segm_AP]",
    "[VAL13 mIoU]",
    "[ZS-EVENT mIoU, flip-tta]",
    "[VAL13 mIoU, seed-0]",
]

INVALID_TAGS = [
    "ZS-EVENT mIoU",
    "[mIoU]",
    "[ZS-EVENT mIoU, Flip TTA]",
    "[ZS-EVENT mIoU, flip-TTA]",
    "",
    "[ZS-EVENT mIoU,]",
    "[ZS-EVENT mIoU, ]",
    "[ZS-EVENT mIoU, -flip]",
    "[ZS-EVENT mIoU, flip_tta]",
    "[ZS-EVENT mIoU, flip-tta, extra]",
    "[]",
]


def make_record(**overrides: object) -> ResultRecord:
    """Build a valid record, letting each test override single fields."""
    fields: dict[str, object] = {
        "metric_tag": "[ZS-EVENT mIoU]",
        "values": {"miou": 41.28},
        "model": "changemamba",
        "split": "turkey-earthquake",
        "commit": "0123456789abcdef",
        "config_hash": "deadbeef",
    }
    fields.update(overrides)
    return ResultRecord(**fields)  # type: ignore[arg-type]


# --- BASE_TAGS -------------------------------------------------------------


def test_base_tags_is_tuple_of_allowed_tags() -> None:
    assert isinstance(BASE_TAGS, tuple)
    for tag in (
        "ZS-EVENT mIoU",
        "ZS-EVENT bina-makro-F1",
        "STD-EVENT mIoU",
        "STD-SET mIoU",
        "DFC-TEST mIoU",
        "CVPR-TEST segm_AP",
        "VAL13 mIoU",
    ):
        assert tag in BASE_TAGS


# --- validate_metric_tag ---------------------------------------------------


@pytest.mark.parametrize("tag", VALID_TAGS)
def test_validate_metric_tag_accepts_valid_forms(tag: str) -> None:
    assert validate_metric_tag(tag) is None


@pytest.mark.parametrize("tag", INVALID_TAGS)
def test_validate_metric_tag_rejects_invalid_forms(tag: str) -> None:
    with pytest.raises(ResultSchemaError):
        validate_metric_tag(tag)


def test_validate_metric_tag_rejects_unknown_base() -> None:
    with pytest.raises(ResultSchemaError):
        validate_metric_tag("[NOT-A-TAG mIoU]")


# --- ResultRecord ----------------------------------------------------------


def test_result_record_is_frozen() -> None:
    record = make_record()
    with pytest.raises(FrozenInstanceError):
        record.model = "other"  # type: ignore[misc]


def test_result_record_defaults() -> None:
    record = make_record()
    assert record.seed is None
    assert record.extra == {}
    assert isinstance(record.created_at, str)
    assert record.created_at != ""


def test_result_record_created_at_is_iso_utc() -> None:
    record = make_record()
    parsed = datetime.fromisoformat(record.created_at)
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == timezone.utc.utcoffset(None)


def test_result_record_keeps_given_created_at() -> None:
    record = make_record(created_at="2026-11-06T12:00:00+00:00")
    assert record.created_at == "2026-11-06T12:00:00+00:00"


def test_result_record_accepts_seed_and_extra() -> None:
    record = make_record(seed=0, extra={"note": "deneme"})
    assert record.seed == 0
    assert record.extra == {"note": "deneme"}


def test_result_record_rejects_invalid_metric_tag() -> None:
    with pytest.raises(ResultSchemaError):
        make_record(metric_tag="[mIoU]")


def test_result_record_rejects_empty_values() -> None:
    with pytest.raises(ResultSchemaError):
        make_record(values={})


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_result_record_rejects_non_finite_values(bad: float) -> None:
    with pytest.raises(ResultSchemaError):
        make_record(values={"miou": bad})


def test_result_record_rejects_non_numeric_values() -> None:
    with pytest.raises(ResultSchemaError):
        make_record(values={"miou": "41.28"})


@pytest.mark.parametrize("field", ["model", "split", "commit", "config_hash"])
def test_result_record_rejects_empty_required_strings(field: str) -> None:
    with pytest.raises(ResultSchemaError):
        make_record(**{field: ""})


# --- write_result / read_result -------------------------------------------


def test_write_result_creates_parent_dirs_and_json(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dir" / "result.json"
    record = make_record()

    written = write_result(path, record)

    assert written == path
    assert path.is_file()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["metric_tag"] == record.metric_tag
    assert payload["values"] == record.values
    assert payload["model"] == record.model
    assert payload["split"] == record.split
    assert payload["commit"] == record.commit
    assert payload["config_hash"] == record.config_hash
    assert payload["seed"] is None
    assert payload["extra"] == {}
    assert payload["created_at"] == record.created_at


def test_write_result_is_utf8_indented_and_sorted(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    record = make_record(extra={"not": "Türkçe karakter: ğüşiöç"})

    write_result(path, record)

    text = path.read_text(encoding="utf-8")
    assert "ğüşiöç" in text
    assert "\n  " in text
    payload = json.loads(text)
    assert list(payload) == sorted(payload)


def test_write_result_leaves_no_temp_file(tmp_path: Path) -> None:
    path = tmp_path / "result.json"

    write_result(path, make_record())

    assert [p.name for p in tmp_path.iterdir()] == ["result.json"]


def test_write_result_overwrites_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    write_result(path, make_record(values={"miou": 1.0}))
    write_result(path, make_record(values={"miou": 2.0}))

    assert read_result(path).values == {"miou": 2.0}
    assert [p.name for p in tmp_path.iterdir()] == ["result.json"]


def test_read_result_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    record = make_record(seed=7, extra={"kare": 1114, "not": "deneme"})

    write_result(path, record)

    assert read_result(path) == record


def test_read_result_missing_metric_tag_raises(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    payload = {
        "values": {"miou": 41.28},
        "model": "changemamba",
        "split": "turkey-earthquake",
        "commit": "0123456789abcdef",
        "config_hash": "deadbeef",
        "seed": None,
        "extra": {},
        "created_at": "2026-11-06T12:00:00+00:00",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ResultSchemaError):
        read_result(path)


def test_read_result_invalid_tag_raises(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    payload = {
        "metric_tag": "[mIoU]",
        "values": {"miou": 41.28},
        "model": "changemamba",
        "split": "turkey-earthquake",
        "commit": "0123456789abcdef",
        "config_hash": "deadbeef",
        "seed": None,
        "extra": {},
        "created_at": "2026-11-06T12:00:00+00:00",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ResultSchemaError):
        read_result(path)
