"""Tests for the data summary helpers (``hasar.data.summary``) and the CLI (``hasar.cli``).

Everything is data-free: tiny synthetic label tiles are written into ``tmp_path`` with
:mod:`tifffile` and the split lists are small files built by the test. The only real files read
are the ``standard_ML`` lists that live in git, used to pin the default ``veri-ozet`` counts.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import tifffile

from hasar.cli import main
from hasar.data.events import Split, load_standard_lists, make_zero_shot_split, split_summary
from hasar.data.io import tile_paths
from hasar.data.summary import build_split_report, label_histogram
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


def test_cli_agirlik_with_config_downloads_damageformer(tmp_path, monkeypatch, capsys) -> None:
    from hasar.models import bright_baselines, weights

    config = tmp_path / "bright.yaml"
    config.write_text("model:\n  init: bright_zero_shot\n", encoding="utf-8")
    out = tmp_path / "checkpoints"
    checkpoint = "ckpt_DamageFormer_cross_event_zeroshot_turkey-earthquake.pth"
    path = out / checkpoint

    def download(name, dest):
        print("İndirme tamamlandı.")
        assert name == checkpoint
        assert dest == out
        return path

    legacy = Mock()
    monkeypatch.setattr(bright_baselines, "download_checkpoint", download)
    monkeypatch.setattr(weights, "download_pretrained", legacy)

    assert main(["agirlik", "--config", str(config), "--out", str(out)]) == 0
    assert capsys.readouterr().out.splitlines()[-1] == str(path)
    legacy.assert_not_called()


def test_cli_agirlik_with_config_downloads_vssm(tmp_path, monkeypatch, capsys) -> None:
    from hasar.models import bright_baselines, weights

    config = tmp_path / "vssm.yaml"
    config.write_text("model:\n  name: changemamba_mmbda_tiny\n  init: vssm_tiny_imagenet\n", encoding="utf-8")
    out = tmp_path / "checkpoints"
    path = out / "vssm_tiny.pth"
    download = Mock(return_value=path)
    bright = Mock()
    monkeypatch.setattr(weights, "download_pretrained", download)
    monkeypatch.setattr(bright_baselines, "download_checkpoint", bright)

    assert main(["agirlik", "--config", str(config), "--out", str(out)]) == 0
    assert capsys.readouterr().out.splitlines()[-1] == str(path)
    download.assert_called_once_with(out)
    bright.assert_not_called()


def test_cli_kiyas_writes_comparison(tmp_path, monkeypatch, capsys) -> None:
    from hasar.eval.metrics import ConfusionAccumulator

    monkeypatch.setattr("hasar.train.run._git_commit", lambda: "test-commit")
    a = ConfusionAccumulator()
    b = ConfusionAccumulator()
    label = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    for index in range(4):
        name = f"synthetic_{index}"
        a.add(label, np.zeros_like(label), name)
        b.add(label, label, name)
    path_a = tmp_path / "a.npz"
    path_b = tmp_path / "b.npz"
    a.save_npz(path_a)
    b.save_npz(path_b)
    out = tmp_path / "kiyas" / "varyant_vs_ana.json"

    code = main(
        ["kiyas", "--a", str(path_a), "--b", str(path_b), "--name-a", "ana", "--name-b", "varyant", "--out", str(out)]
    )

    assert code == 0
    record = json.loads(out.read_text(encoding="utf-8"))
    assert record["extra"]["verdict"]
    assert record["values"]["diff_pct"] > 0
    assert record["extra"]["model_a"] == "ana"
    assert record["extra"]["model_b"] == "varyant"
    assert record["commit"] == "test-commit"
    assert record["config_hash"] == "none"
    summary = capsys.readouterr()
    assert not summary.err
    assert f"{record['values']['diff_pct']:.2f}" in summary.out
    assert f"{record['values']['low_pct']:.2f}" in summary.out
    assert f"{record['values']['high_pct']:.2f}" in summary.out
    assert "%95" in summary.out
    assert record["extra"]["verdict"] in summary.out


def test_cli_kiyas_reports_unmatched_tile_counts(tmp_path, monkeypatch, capsys) -> None:
    from hasar.eval.metrics import ConfusionAccumulator

    commit = Mock()
    monkeypatch.setattr("hasar.train.run._git_commit", commit)
    label = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    path_a = tmp_path / "a.npz"
    path_b = tmp_path / "b.npz"
    for path, names in ((path_a, ("shared", "only_a")), (path_b, ("shared", "only_b1", "only_b2"))):
        accumulator = ConfusionAccumulator()
        for name in names:
            accumulator.add(label, label, name)
        accumulator.save_npz(path)
    out = tmp_path / "kiyas" / "comparison.json"

    code = main(
        ["kiyas", "--a", str(path_a), "--b", str(path_b), "--name-a", "ana", "--name-b", "varyant", "--out", str(out)]
    )

    assert code == 1
    assert not out.parent.exists()
    captured = capsys.readouterr()
    assert not captured.out
    assert "yalnız A'da 1" in captured.err
    assert "yalnız B'de 2" in captured.err
    commit.assert_not_called()


def test_cli_kiyas_writes_paired_comparison(tmp_path, monkeypatch, capsys) -> None:
    from hasar.eval.metrics import ConfusionAccumulator

    monkeypatch.setattr("hasar.train.run._git_commit", lambda: "deadbeef")
    a, b = ConfusionAccumulator(), ConfusionAccumulator()
    label = np.array([[0, 1, 2, 3]], dtype=np.uint8)
    for index in range(4):
        a.add(label, np.array([[1, 2, 3, 0]], dtype=np.uint8), f"synthetic_{index}")
    for index in reversed(range(4)):
        b.add(label, label, f"synthetic_{index}")
    path_a, path_b = tmp_path / "a.npz", tmp_path / "b.npz"
    a.save_npz(path_a)
    b.save_npz(path_b)
    out = tmp_path / "kiyas" / "varyant_vs_ana.json"

    code = main(
        ["kiyas", "--a", str(path_a), "--b", str(path_b), "--name-a", "ana", "--name-b", "varyant", "--out", str(out)]
    )

    assert code == 0
    record = json.loads(out.read_text(encoding="utf-8"))
    assert record["extra"]["verdict"] == "farklı"
    assert record["values"]["diff_pct"] == pytest.approx(100.0)
    assert record["values"]["low_pct"] == pytest.approx(100.0)
    assert record["values"]["high_pct"] == pytest.approx(100.0)
    assert record["extra"]["model_a"] == "ana"
    assert record["extra"]["model_b"] == "varyant"
    assert record["commit"] == "deadbeef"
    assert record["config_hash"] == "none"
    captured = capsys.readouterr()
    assert "100.00" in captured.out
    assert "95" in captured.out
    assert "farklı" in captured.out
    assert captured.err == ""


def test_cli_kiyas_summary_supports_windows_encoding(tmp_path, monkeypatch) -> None:
    from hasar.eval.metrics import ConfusionAccumulator

    monkeypatch.setattr("hasar.train.run._git_commit", lambda: "deadbeef")
    accumulator = ConfusionAccumulator()
    label = np.array([[0, 1, 2, 3]], dtype=np.uint8)
    for index in range(4):
        accumulator.add(label, label, f"synthetic_{index}")
    matrices = tmp_path / "matrices.npz"
    accumulator.save_npz(matrices)
    out = tmp_path / "comparison.json"
    buffer = io.BytesIO()
    stream = io.TextIOWrapper(buffer, encoding="cp1254")
    monkeypatch.setattr("sys.stdout", stream)

    code = main(
        [
            "kiyas",
            "--a",
            str(matrices),
            "--b",
            str(matrices),
            "--name-a",
            "ana",
            "--name-b",
            "varyant",
            "--out",
            str(out),
        ]
    )

    stream.flush()
    assert code == 0
    assert "varyant - ana" in buffer.getvalue().decode("cp1254")
    assert out.is_file()


@pytest.fixture
def downloaded_results(tmp_path, monkeypatch):
    """Build four synthetic tiles per model and a minimal README, without real data."""
    from hasar.eval.metrics import ConfusionAccumulator
    from hasar.results import ResultRecord, write_result

    monkeypatch.setattr("hasar.train.run._git_commit", lambda: "test-commit")
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "sonuclar"
    results = tmp_path / "results"
    readme = tmp_path / "README.md"
    readme.write_text("Başlık\n<!-- sonuclar:basla -->\neski\n<!-- sonuclar:bitir -->\n", encoding="utf-8")
    label = np.array([[0, 1, 2, 3]], dtype=np.uint8)
    predictions = {
        "ana": np.zeros_like(label),
        "varyant": label,
        "referans": np.array([[1, 2, 3, 0]], dtype=np.uint8),
    }
    for name, prediction in predictions.items():
        accumulator = ConfusionAccumulator()
        for index in range(4):
            accumulator.add(label, prediction, f"synthetic_{index}")
        if name == "referans":
            directory = results / "yeniden-uretim"
            directory.mkdir(parents=True)
            accumulator.save_npz(directory / "damageformer_test_matrices.npz")
            continue
        directory = source / name
        directory.mkdir(parents=True)
        accumulator.save_npz(directory / "turkiye_kare_matrisleri_headline.npz")
        accumulator.save_npz(directory / "turkiye_kare_matrisleri_tta.npz")
        record = ResultRecord(
            metric_tag="[ZS-EVENT mIoU]",
            values={"miou_pct": 6.25 if name == "ana" else 100.0},
            model=name,
            split="zero_shot/test",
            commit="synthetic",
            config_hash="none",
        )
        write_result(directory / "turkiye.json", record)
        for filename in ("turkiye_headline.json", "turkiye_bina_f1.json", "turkiye_kalibrasyon.json", "sicaklik.json"):
            (directory / filename).write_text("{}\n", encoding="utf-8")
        (directory / "ignore.txt").write_text("Kopyalanmasın", encoding="utf-8")
    return source, results, readme


def sonuc_al_args(downloaded_results):
    source, results, readme = downloaded_results
    return ["sonuc-al", "--kaynak", str(source), "--results", str(results), "--readme", str(readme)]


@pytest.mark.parametrize("with_blocks", [False, True])
def test_cli_sonuc_al_copies_compares_and_updates_readme(downloaded_results, with_blocks, monkeypatch, capsys):
    from hasar.eval import bootstrap

    source, results, readme = downloaded_results
    blocks = results / "arastirma" / "turkiye_bloklar.json"
    if with_blocks:
        blocks.parent.mkdir(parents=True)
        blocks.write_text(
            json.dumps({
                "tile_centers": [
                    {"name": f"synthetic_{i}", "lat": 37.0 + (i // 2) * 0.1, "lon": 36.0}
                    for i in range(4)
                ],
                "clusters": [{"nearest_place": "sentetik", "center_lat": 37.0, "center_lon": 36.0}],
            }),
            encoding="utf-8",
        )
    load_blocks = Mock(wraps=bootstrap.turkey_blocks)
    monkeypatch.setattr(bootstrap, "turkey_blocks", load_blocks)

    assert main([*sonuc_al_args(downloaded_results), "--blocks", str(blocks)]) == 0

    for name in ("ana", "varyant"):
        expected = {path.name for path in (source / name).iterdir() if path.suffix in (".json", ".npz")}
        assert {path.name for path in (results / name).iterdir()} == expected
        for filename in expected:
            assert (results / name / filename).read_bytes() == (source / name / filename).read_bytes()
    output = capsys.readouterr()
    assert output.err == ""
    for filename, a, b in (
        ("varyant_vs_ana.json", "ana", "varyant"),
        ("ana_vs_referans.json", "referans", "ana"),
        ("varyant_vs_referans.json", "referans", "varyant"),
    ):
        record = json.loads((results / "kiyas" / filename).read_text(encoding="utf-8"))
        assert record["extra"]["model_a"] == a
        assert record["extra"]["model_b"] == b
        assert record["values"]["diff_pct"] > 0
        line = next(line for line in output.out.splitlines() if line.startswith(f"{b} - {a}:"))
        for key in ("diff_pct", "low_pct", "high_pct"):
            assert f"{record['values'][key]:.2f}" in line
        assert record["extra"]["verdict"] in line
        assert "%95" in line
        assert f"kiyas/{filename}" in readme.read_text(encoding="utf-8")
    assert "ana: 6.25 [ZS-EVENT mIoU]" in output.out
    assert "varyant: 100.00 [ZS-EVENT mIoU]" in output.out
    assert "ana/turkiye.json" in readme.read_text(encoding="utf-8")
    assert "varyant/turkiye.json" in readme.read_text(encoding="utf-8")
    assert load_blocks.call_count == (3 if with_blocks else 0)


@pytest.mark.parametrize("name", ["ana", "varyant"])
def test_cli_sonuc_al_missing_headline_copies_nothing(downloaded_results, name, capsys):
    source, results, readme = downloaded_results
    original_readme = readme.read_bytes()
    (source / name / "turkiye.json").unlink()

    assert main(sonuc_al_args(downloaded_results)) == 1

    assert not (results / "ana").exists()
    assert not (results / "varyant").exists()
    assert not (results / "kiyas").exists()
    assert readme.read_bytes() == original_readme
    error = capsys.readouterr().err
    assert "bulunamadı" in error
    assert str(source / name / "turkiye.json") in error


@pytest.mark.parametrize("name", ["ana", "varyant"])
def test_cli_sonuc_al_conflicting_headline_is_preserved(downloaded_results, name, capsys):
    _, results, readme = downloaded_results
    original_readme = readme.read_bytes()
    headline = results / name / "turkiye.json"
    headline.parent.mkdir()
    headline.write_text('{"existing": true}\n', encoding="utf-8")
    original = headline.read_bytes()

    assert main(sonuc_al_args(downloaded_results)) == 1

    assert headline.read_bytes() == original
    assert list(headline.parent.iterdir()) == [headline]
    assert not (results / ("varyant" if name == "ana" else "ana")).exists()
    assert not (results / "kiyas").exists()
    assert readme.read_bytes() == original_readme
    error = capsys.readouterr().err
    assert "farklı" in error
    assert str(headline) in error


def test_cli_sonuc_al_identical_rerun_succeeds(downloaded_results):
    _, results, readme = downloaded_results
    assert main(sonuc_al_args(downloaded_results)) == 0
    headlines = {name: (results / name / "turkiye.json").read_bytes() for name in ("ana", "varyant")}
    table = readme.read_bytes()

    assert main(sonuc_al_args(downloaded_results)) == 0

    assert readme.read_bytes() == table
    for name, content in headlines.items():
        assert (results / name / "turkiye.json").read_bytes() == content
