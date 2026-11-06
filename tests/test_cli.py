"""Tests for the data summary helpers (``hasar.data.summary``) and the CLI (``hasar.cli``).

Everything is data-free: tiny synthetic label tiles are written into ``tmp_path`` with
:mod:`tifffile` and the split lists are small files built by the test. The only real files read
are the ``standard_ML`` lists that live in git, used to pin the default ``veri-ozet`` counts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile
from hasar.cli import main
from hasar.data.summary import build_split_report, label_histogram

from hasar.data.events import Split, load_standard_lists, make_zero_shot_split, split_summary
from hasar.data.io import tile_paths
from hasar.guard import TurkeyAccess, TurkeyLockError

TURKEY_NAME = "turkey-earthquake_00000001"


# --- helpers ---------------------------------------------------------------


def write_label(root: Path, name: str, label: np.ndarray) -> Path:
    """Write a synthetic damage label under ``root`` and return its path."""
    path = tile_paths(root, name).target
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, label)
    return path


def write_lists(list_dir: Path, train: list[str], val: list[str], test: list[str]) -> Path:
    """Write the three ``standard_ML`` list files and return the directory."""
    list_dir.mkdir(parents=True, exist_ok=True)
    for filename, names in (
        ("train_set.txt", train),
        ("val_set.txt", val),
        ("test_set.txt", test),
    ):
        (list_dir / filename).write_text("\n".join(names) + "\n", encoding="utf-8")
    return list_dir


# --- label_histogram -------------------------------------------------------


def test_label_histogram_counts_classes(tmp_path: Path) -> None:
    write_label(tmp_path, "la_palma-volcano_00000001", np.array([[0, 1], [2, 3]], dtype=np.uint8))
    write_label(tmp_path, "la_palma-volcano_00000002", np.array([[0, 0], [1, 1]], dtype=np.uint8))

    histogram = label_histogram(
        tmp_path,
        ["la_palma-volcano_00000001", "la_palma-volcano_00000002"],
    )

    assert histogram == {
        "pixels": {"0": 3, "1": 3, "2": 1, "3": 1},
        "other": 0,
        "tiles": 2,
    }


def test_label_histogram_string_keys_and_json_friendly(tmp_path: Path) -> None:
    write_label(tmp_path, "noto-earthquake_00000001", np.zeros((2, 2), dtype=np.uint8))

    histogram = label_histogram(tmp_path, ["noto-earthquake_00000001"])

    assert set(histogram["pixels"]) == {"0", "1", "2", "3"}
    assert all(isinstance(key, str) for key in histogram["pixels"])
    assert all(isinstance(value, int) for value in histogram["pixels"].values())
    assert json.loads(json.dumps(histogram)) == histogram


def test_label_histogram_counts_out_of_range_as_other(tmp_path: Path) -> None:
    write_label(tmp_path, "haiti-earthquake_00000001", np.array([[0, 4], [7, 255]], dtype=np.uint8))

    histogram = label_histogram(tmp_path, ["haiti-earthquake_00000001"])

    assert histogram["pixels"] == {"0": 1, "1": 0, "2": 0, "3": 0}
    assert histogram["other"] == 3
    assert histogram["tiles"] == 1


def test_label_histogram_respects_num_classes(tmp_path: Path) -> None:
    write_label(tmp_path, "libya-flood_00000001", np.array([[0, 1, 2, 3]], dtype=np.uint8))

    histogram = label_histogram(tmp_path, ["libya-flood_00000001"], num_classes=3)

    assert histogram["pixels"] == {"0": 1, "1": 1, "2": 1}
    assert histogram["other"] == 1


def test_label_histogram_empty_names(tmp_path: Path) -> None:
    histogram = label_histogram(tmp_path, [])

    assert histogram == {"pixels": {"0": 0, "1": 0, "2": 0, "3": 0}, "other": 0, "tiles": 0}


def test_label_histogram_turkey_locked(tmp_path: Path) -> None:
    # The guard runs before any disk access, so no file has to exist.
    with pytest.raises(TurkeyLockError):
        label_histogram(tmp_path, [TURKEY_NAME])


def test_label_histogram_turkey_with_access(tmp_path: Path) -> None:
    write_label(tmp_path, TURKEY_NAME, np.array([[0, 1]], dtype=np.uint8))
    access = TurkeyAccess(tag="protokol-kilidi", commit="deadbeef")

    histogram = label_histogram(tmp_path, [TURKEY_NAME], access=access)

    assert histogram["pixels"] == {"0": 1, "1": 1, "2": 0, "3": 0}
    assert histogram["tiles"] == 1


# --- build_split_report ----------------------------------------------------


def tiny_split() -> Split:
    return Split(
        train=("la_palma-volcano_00000001", "haiti-earthquake_00000001"),
        val=("noto-earthquake_00000001",),
        test=(TURKEY_NAME,),
    )


def test_build_split_report_without_root_is_split_summary() -> None:
    split = tiny_split()

    report = build_split_report(split)

    assert report == split_summary(split)
    assert "class_pixels" not in report


def test_build_split_report_with_root_adds_class_pixels(tmp_path: Path) -> None:
    split = tiny_split()
    write_label(tmp_path, "la_palma-volcano_00000001", np.array([[0, 1]], dtype=np.uint8))
    write_label(tmp_path, "haiti-earthquake_00000001", np.array([[2, 2]], dtype=np.uint8))
    write_label(tmp_path, "noto-earthquake_00000001", np.array([[3]], dtype=np.uint8))

    report = build_split_report(split, root=tmp_path)

    assert report["counts"] == {"train": 2, "val": 1, "test": 1}
    assert report["class_pixels"]["train"] == {
        "pixels": {"0": 1, "1": 1, "2": 2, "3": 0},
        "other": 0,
        "tiles": 2,
    }
    assert report["class_pixels"]["val"] == {
        "pixels": {"0": 0, "1": 0, "2": 0, "3": 1},
        "other": 0,
        "tiles": 1,
    }


def test_build_split_report_never_reads_test_labels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    split = tiny_split()
    calls: list[list[str]] = []

    def fake_histogram(root: Path, names, num_classes: int = 4, access=None) -> dict:
        calls.append(list(names))
        return {"pixels": {"0": 0, "1": 0, "2": 0, "3": 0}, "other": 0, "tiles": len(names)}

    monkeypatch.setattr("hasar.data.summary.label_histogram", fake_histogram)

    report = build_split_report(split, root=tmp_path)

    assert calls == [list(split.train), list(split.val)]
    assert all(TURKEY_NAME not in names for names in calls)
    assert report["class_pixels"]["train"]["tiles"] == len(split.train)
    assert report["class_pixels"]["val"]["tiles"] == len(split.val)


# --- cli: veri-ozet --------------------------------------------------------


def test_veri_ozet_writes_json(tmp_path: Path) -> None:
    list_dir = write_lists(
        tmp_path / "lists",
        train=["la_palma-volcano_00000001", "la_palma-volcano_00000002"],
        val=["ısparta-deprem_00000001"],
        test=["haiti-earthquake_00000001", TURKEY_NAME],
    )
    root = tmp_path / "data"
    write_label(root, "la_palma-volcano_00000001", np.array([[0, 1]], dtype=np.uint8))
    write_label(root, "la_palma-volcano_00000002", np.array([[2, 3]], dtype=np.uint8))
    write_label(root, "haiti-earthquake_00000001", np.array([[0, 0]], dtype=np.uint8))
    write_label(root, "ısparta-deprem_00000001", np.array([[1]], dtype=np.uint8))
    out = tmp_path / "nested" / "dir" / "ozet.json"

    code = main(["veri-ozet", "--out", str(out), "--root", str(root), "--list-dir", str(list_dir)])

    assert code == 0
    text = out.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert "\\u0131" not in text  # ensure_ascii=False
    report = json.loads(text)
    assert report["kind"] == "data-summary"
    assert report["counts"] == {"train": 3, "val": 1, "test": 1}
    assert report["test_event"] == "turkey-earthquake"
    assert report["test_count"] == 1
    assert report["class_pixels"]["train"]["tiles"] == 3
    assert report["class_pixels"]["val"]["tiles"] == 1
    assert report["class_pixels"]["train"]["pixels"] == {"0": 3, "1": 1, "2": 1, "3": 1}
    assert "ısparta-deprem" in report["events"]["val"]


def test_veri_ozet_without_root_has_no_class_pixels(tmp_path: Path) -> None:
    list_dir = write_lists(
        tmp_path / "lists",
        train=["la_palma-volcano_00000001"],
        val=["noto-earthquake_00000001"],
        test=[TURKEY_NAME],
    )
    out = tmp_path / "ozet.json"

    code = main(["veri-ozet", "--out", str(out), "--list-dir", str(list_dir)])

    assert code == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["kind"] == "data-summary"
    assert report["counts"] == {"train": 1, "val": 1, "test": 1}
    assert "class_pixels" not in report


def test_veri_ozet_default_lists_counts(tmp_path: Path) -> None:
    out = tmp_path / "ozet.json"

    code = main(["veri-ozet", "--out", str(out)])

    assert code == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["counts"] == {"train": 2795, "val": 337, "test": 1114}
    assert report["counts"] == make_zero_shot_split(*load_standard_lists()).counts()


# --- cli: indir ------------------------------------------------------------


def test_indir_calls_prepare_bright(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[Path, Path]] = []

    def fake_prepare_bright(root, cache_dir, **kwargs) -> Path:
        calls.append((Path(root), Path(cache_dir)))
        return Path(root)

    # Patch both the source module and the CLI module, so the test holds whether the CLI
    # calls ``hasar.data.download.prepare_bright`` or imported the name directly.
    monkeypatch.setattr("hasar.data.download.prepare_bright", fake_prepare_bright)
    monkeypatch.setattr("hasar.cli.prepare_bright", fake_prepare_bright, raising=False)

    root = tmp_path / "data"
    cache = tmp_path / "cache"

    code = main(["indir", "--root", str(root), "--cache", str(cache)])

    assert code == 0
    assert calls == [(root, cache)]


# --- cli: argument errors --------------------------------------------------


def test_unknown_subcommand_exits_nonzero() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["bilinmeyen-komut"])

    assert excinfo.value.code != 0
