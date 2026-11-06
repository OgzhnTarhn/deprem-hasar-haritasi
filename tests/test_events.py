"""Tests for the event/split helpers (``hasar.data.events``).

Everything here is data-free: the synthetic lists are built in the test and the only real files
read are the split lists that live in git under ``third_party/bright/splitname/standard_ML``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from hasar.data.events import (
    TEST_EVENT,
    TILE_SUFFIXES,
    Split,
    SplitLeakError,
    check_no_leak,
    event_of,
    list_digest,
    load_standard_lists,
    make_zero_shot_split,
    read_list,
    split_summary,
    tile_stem,
)

STANDARD_EVENTS = {
    "bata-explosion",
    "beirut-explosion",
    "congo-volcano",
    "haiti-earthquake",
    "hawaii-wildfire",
    "la_palma-volcano",
    "libya-flood",
    "marshall-wildfire",
    "mexico-hurricane",
    "morocco-earthquake",
    "myanmar-hurricane",
    "noto-earthquake",
    "ukraine-conflict",
}


def test_constants() -> None:
    assert TILE_SUFFIXES == ("_pre_disaster", "_post_disaster", "_building_damage")
    assert TEST_EVENT == "turkey-earthquake"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("la_palma-volcano_00000001_post_disaster.tif", "la_palma-volcano_00000001"),
        ("la_palma-volcano_00000001_pre_disaster.TIF", "la_palma-volcano_00000001"),
        ("la_palma-volcano_00000001_post_disaster.tiff", "la_palma-volcano_00000001"),
        ("noto-earthquake_00000032_building_damage", "noto-earthquake_00000032"),
        ("noto-earthquake_00000032_building_damage.tif", "noto-earthquake_00000032"),
        ("la_palma-volcano_00000001", "la_palma-volcano_00000001"),
        ("turkey-earthquake_00000262", "turkey-earthquake_00000262"),
    ],
)
def test_tile_stem(name: str, expected: str) -> None:
    assert tile_stem(name) == expected


def test_tile_stem_drops_only_one_suffix() -> None:
    assert tile_stem("a-x_00000001_pre_disaster_post_disaster") == "a-x_00000001_pre_disaster"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("la_palma-volcano_00000001", "la_palma-volcano"),
        ("la_palma-volcano_00000001_post_disaster.tif", "la_palma-volcano"),
        ("turkey-earthquake_00000262_building_damage.tif", "turkey-earthquake"),
        ("noto-earthquake_00000032", "noto-earthquake"),
    ],
)
def test_event_of(name: str, expected: str) -> None:
    assert event_of(name) == expected


def test_event_of_keeps_underscored_event_name() -> None:
    # ``split('_')[0]`` would return ``la`` here; the event name must survive intact.
    assert event_of("la_palma-volcano_00000001") == "la_palma-volcano"


@pytest.mark.parametrize("name", ["la_palma-volcano", "turkey-earthquake", ""])
def test_event_of_without_underscore_raises(name: str) -> None:
    with pytest.raises(ValueError):
        event_of(name)


def test_read_list(tmp_path: Path) -> None:
    path = tmp_path / "names.txt"
    path.write_bytes(b"a-x_00000001\r\n  a-x_00000002  \n\nturkey-earthquake_00000001\n\r\n")

    assert read_list(path) == [
        "a-x_00000001",
        "a-x_00000002",
        "turkey-earthquake_00000001",
    ]


def test_read_list_empty(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("\n\n", encoding="utf-8")

    assert read_list(path) == []


def test_split_leak_error_is_runtime_error() -> None:
    assert issubclass(SplitLeakError, RuntimeError)


def test_split_is_frozen_dataclass() -> None:
    assert dataclasses.is_dataclass(Split)
    assert [field.name for field in dataclasses.fields(Split)] == ["train", "val", "test"]

    split = Split(train=("a-x_00000001",), val=("b-y_00000001",), test=("turkey-earthquake_00000001",))
    assert split.train == ("a-x_00000001",)
    assert split.val == ("b-y_00000001",)
    assert split.test == ("turkey-earthquake_00000001",)

    with pytest.raises(dataclasses.FrozenInstanceError):
        split.train = ()  # type: ignore[misc]


def test_split_counts() -> None:
    split = Split(
        train=("a-x_00000001", "a-x_00000002"),
        val=("b-y_00000001",),
        test=("turkey-earthquake_00000001", "turkey-earthquake_00000002", "turkey-earthquake_00000003"),
    )

    assert split.counts() == {"train": 2, "val": 1, "test": 3}


def test_make_zero_shot_split() -> None:
    train_list = ["a-x_00000001", "turkey-earthquake_00000001", "a-x_00000002"]
    val_list = ["b-y_00000001", "turkey-earthquake_00000002"]
    test_list = ["c-z_00000001", "turkey-earthquake_00000003"]

    split = make_zero_shot_split(train_list, val_list, test_list)

    assert split.train == ("a-x_00000001", "a-x_00000002", "c-z_00000001")
    assert split.val == ("b-y_00000001",)
    assert split.test == (
        "turkey-earthquake_00000001",
        "turkey-earthquake_00000002",
        "turkey-earthquake_00000003",
    )


def test_make_zero_shot_split_custom_test_event() -> None:
    split = make_zero_shot_split(
        ["a-x_00000001", "b-y_00000001"],
        ["b-y_00000002"],
        ["b-y_00000003"],
        test_event="b-y",
    )

    assert split.train == ("a-x_00000001",)
    assert split.val == ()
    assert split.test == ("b-y_00000001", "b-y_00000002", "b-y_00000003")


def test_make_zero_shot_split_rejects_duplicates() -> None:
    with pytest.raises(SplitLeakError):
        make_zero_shot_split(
            ["a-x_00000001", "a-x_00000001"],
            ["b-y_00000001"],
            ["turkey-earthquake_00000001"],
        )


def test_check_no_leak_accepts_good_split() -> None:
    split = make_zero_shot_split(
        ["a-x_00000001"],
        ["b-y_00000001"],
        ["turkey-earthquake_00000001"],
    )

    check_no_leak(split)


def test_check_no_leak_rejects_test_event_in_train() -> None:
    split = Split(
        train=("turkey-earthquake_00000001",),
        val=("b-y_00000001",),
        test=("turkey-earthquake_00000002",),
    )

    with pytest.raises(SplitLeakError):
        check_no_leak(split)


def test_check_no_leak_rejects_test_event_in_val() -> None:
    split = Split(
        train=("a-x_00000001",),
        val=("turkey-earthquake_00000001",),
        test=("turkey-earthquake_00000002",),
    )

    with pytest.raises(SplitLeakError):
        check_no_leak(split)


def test_check_no_leak_rejects_foreign_name_in_test() -> None:
    split = Split(
        train=("a-x_00000001",),
        val=("b-y_00000001",),
        test=("turkey-earthquake_00000001", "a-x_00000002"),
    )

    with pytest.raises(SplitLeakError):
        check_no_leak(split)


def test_check_no_leak_rejects_name_in_two_parts() -> None:
    split = Split(
        train=("a-x_00000001",),
        val=("a-x_00000001",),
        test=("turkey-earthquake_00000001",),
    )

    with pytest.raises(SplitLeakError):
        check_no_leak(split)


def test_check_no_leak_rejects_duplicate_inside_part() -> None:
    split = Split(
        train=("a-x_00000001", "a-x_00000001"),
        val=("b-y_00000001",),
        test=("turkey-earthquake_00000001",),
    )

    with pytest.raises(SplitLeakError):
        check_no_leak(split)


def test_load_standard_lists_counts() -> None:
    train_list, val_list, test_list = load_standard_lists()

    split = make_zero_shot_split(train_list, val_list, test_list)

    assert split.counts() == {"train": 2795, "val": 337, "test": 1114}


def test_load_standard_lists_events() -> None:
    train_list, val_list, test_list = load_standard_lists()

    split = make_zero_shot_split(train_list, val_list, test_list)

    assert {event_of(name) for name in split.train} == STANDARD_EVENTS
    assert {event_of(name) for name in split.val} == STANDARD_EVENTS
    assert {event_of(name) for name in split.test} == {TEST_EVENT}


def test_load_standard_lists_explicit_dir() -> None:
    list_dir = Path(__file__).resolve().parents[1] / "third_party" / "bright" / "splitname" / "standard_ML"

    assert load_standard_lists(list_dir) == load_standard_lists()


def test_split_summary() -> None:
    split = make_zero_shot_split(
        ["a-x_00000001", "a-x_00000002", "turkey-earthquake_00000001"],
        ["b-y_00000001", "turkey-earthquake_00000002"],
        ["turkey-earthquake_00000003"],
    )

    summary = split_summary(split)

    assert summary == {
        "counts": {"train": 2, "val": 1, "test": 3},
        "events": {
            "train": {"a-x": 2},
            "val": {"b-y": 1},
        },
        "test_event": TEST_EVENT,
        "test_count": 3,
    }


def test_split_summary_events_sorted() -> None:
    split = make_zero_shot_split(
        ["b-y_00000001", "a-x_00000001", "a-x_00000002"],
        ["c-z_00000001", "a-x_00000003"],
        ["turkey-earthquake_00000001"],
    )

    summary = split_summary(split)

    assert list(summary["events"]["train"]) == ["a-x", "b-y"]
    assert list(summary["events"]["val"]) == ["a-x", "c-z"]


def test_split_summary_hides_test_tiles() -> None:
    split = make_zero_shot_split(
        ["a-x_00000001"],
        ["b-y_00000001"],
        ["turkey-earthquake_00000001", "turkey-earthquake_00000002"],
    )

    summary = split_summary(split)

    assert set(summary) == {"counts", "events", "test_event", "test_count"}
    assert set(summary["events"]) == {"train", "val"}
    assert summary["test_count"] == 2
    assert "turkey-earthquake_00000001" not in repr(summary)


def test_list_digest_ignores_line_endings(tmp_path: Path) -> None:
    lf = tmp_path / "lf.txt"
    crlf = tmp_path / "crlf.txt"
    lf.write_bytes(b"a_00000001\nb_00000002\n")
    crlf.write_bytes(b"a_00000001\r\nb_00000002\r\n\r\n")
    assert list_digest(lf) == list_digest(crlf)
    assert len(list_digest(lf)) == 64
