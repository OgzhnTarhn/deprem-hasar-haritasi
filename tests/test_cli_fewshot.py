"""Synthetic tests for the few-shot CLI commands."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from hasar.cli import main
from hasar.fewshot import split as split_module
from hasar.results import ResultRecord


def test_sha256_streams_file(tmp_path, capsys):
    path = tmp_path / "model.pt"
    content = b"synthetic checkpoint" * 100_000
    path.write_bytes(content)

    assert main(["sha256", str(path)]) == 0
    assert capsys.readouterr().out == hashlib.sha256(content).hexdigest() + "\n"


def test_sha256_missing_file(tmp_path, capsys):
    assert main(["sha256", str(tmp_path / "missing.pt")]) == 1
    assert "missing.pt" in capsys.readouterr().err


def test_fewshot_split_writes_and_checks_exact_bytes(tmp_path, monkeypatch, capsys):
    tiles = {"adapt": ["tile-a"], "buffer": [], "test": ["tile-b", "tile-c"]}
    split = {
        **tiles,
        "counts": {"adapt": 1, "buffer": 0, "test": 2, "total": 3},
        "sha256": {key: hashlib.sha256("\n".join(names).encode()).hexdigest() for key, names in tiles.items()},
    }
    blocks = tmp_path / "blocks.json"
    out = tmp_path / "split.json"
    build = Mock(return_value=split)
    monkeypatch.setattr(split_module, "make_fewshot_split", build)
    args = ["fewshot-bolme", "--blocks", str(blocks), "--out", str(out)]

    assert main(args) == 0
    build.assert_called_once_with(blocks)
    assert split_module.load_split(out) == split
    output = capsys.readouterr().out
    assert str(out) in output
    assert "adapt=1, buffer=0, test=2" in output

    write = Mock(wraps=split_module.write_split)
    monkeypatch.setattr(split_module, "write_split", write)
    original = out.read_bytes()
    assert main([*args, "--check"]) == 0
    assert "güncel" in capsys.readouterr().out
    assert out.read_bytes() == original

    different = original.replace(b"\n", b"\r\n")
    out.write_bytes(different)
    assert main([*args, "--check"]) == 1
    assert str(out) in capsys.readouterr().err
    assert out.read_bytes() == different

    out.unlink()
    assert main([*args, "--check"]) == 1
    assert str(out) in capsys.readouterr().err
    assert not out.exists()
    assert all(item.args[0] != out for item in write.call_args_list)


def make_record(tag, values, **extra):
    return ResultRecord(
        metric_tag=tag,
        values=values,
        model="test-model",
        split="fewshot/test",
        commit="test-commit",
        config_hash="test-hash",
        extra=extra,
    )


@pytest.fixture
def fewshot_test_args(tmp_path):
    return [
        "fewshot-test",
        "--config",
        "a.yaml",
        "--arms",
        str(tmp_path / "a.yaml"),
        "b.yaml",
        "c.yaml",
        "--data",
        "synthetic-data",
        "--runs",
        "runs",
        "--out",
        "out",
        "--instances",
        "instances",
    ]


@pytest.fixture
def runner(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    runner = Mock()
    runner.load_split.return_value = {"sha256": {"test": "test-digest"}}
    runner.run_id.return_value = "a-run"
    runner.config_hash.return_value = "cfg-hash"
    runner.turkey_blocks.return_value = {"tile-b": ("city", "block")}
    runner.run_fewshot_test.return_value = {
        "test": make_record("[FS-EVENT mIoU]", {"miou_pct": 42.125}),
        "bina_f1": make_record("[FS-EVENT bina-makro-F1]", {"macro_f1_pct": 31.0}),
    }
    monkeypatch.setattr("hasar.cli._load_fewshot", Mock(return_value=runner), raising=False)
    return runner


@pytest.mark.parametrize(
    ("options", "blocks", "building_tiles", "device"),
    [
        ([], "results/arastirma/turkiye_bloklar.json", "results/arastirma/turkiye_bina_kareleri.json", "cuda"),
        (
            ["--blocks", "blocks.json", "--building-tiles", "tiles.json", "--device", "cpu"],
            "blocks.json",
            "tiles.json",
            "cpu",
        ),
    ],
)
def test_fewshot_test_checks_all_arms_then_runs(
    fewshot_test_args,
    runner,
    tmp_path,
    capsys,
    options,
    blocks,
    building_tiles,
    device,
):
    assert main([*fewshot_test_args, *options]) == 0
    assert runner.mock_calls[:4] == [
        call.require_runs_done(Path("runs"), {"a": tmp_path / "a.yaml", "b": Path("b.yaml"), "c": Path("c.yaml")}),
        call.require_fewshot_access(tmp_path, "test"),
        call.load_split(tmp_path / "results/fewshot/bolme.json"),
        call.load_config(Path("a.yaml")),
    ]
    runner.load_model.assert_called_once_with(runner.load_config.return_value, Path("runs"), device)
    runner.turkey_blocks.assert_called_once_with(Path(blocks))
    runner.run_fewshot_test.assert_called_once_with(
        runner.load_model.return_value,
        Path("synthetic-data"),
        Path("out"),
        run_id="a-run",
        split_sha256="test-digest",
        model_name="a",
        config_hash="cfg-hash",
        access=runner.require_fewshot_access.return_value,
        blocks=runner.turkey_blocks.return_value,
        instances_dir=Path("instances"),
        building_tiles=Path(building_tiles),
        device=device,
    )
    output = capsys.readouterr().out
    assert "Türkiye few-shot 42.12 [FS-EVENT mIoU]" in output
    assert "bina F1 31.00 [FS-EVENT bina-makro-F1]" in output


def test_fewshot_test_rejects_config_outside_arms(fewshot_test_args, monkeypatch, capsys):
    fewshot_test_args[2] = "other/a.yaml"
    loader = Mock()
    monkeypatch.setattr("hasar.cli._load_fewshot", loader, raising=False)
    assert main(fewshot_test_args) == 1
    assert "--arms" in capsys.readouterr().err
    loader.assert_not_called()


@pytest.mark.parametrize("error", [RuntimeError, ValueError])
def test_fewshot_test_stops_when_runs_are_not_done(fewshot_test_args, runner, capsys, error):
    runner.require_runs_done.side_effect = error("Kollar tamamlanmadı")
    assert main(fewshot_test_args) == 1
    assert capsys.readouterr().err == "Kollar tamamlanmadı\n"
    assert [item[0] for item in runner.mock_calls] == ["require_runs_done"]


def test_fewshot_compare_uses_test_names_and_prints_all_records(monkeypatch, capsys):
    diff_tag = "[FS-EVENT mIoU, eslestirilmis-fark]"
    class_tag = "[FS-EVENT IoU, eslestirilmis-fark]"
    zs_tag = "[FS-EVENT mIoU, zs-ana]"
    values = {"diff_pct": 0.5, "low_pct": -0.1, "high_pct": 1.2}
    records = {
        "b_vs_a": make_record(diff_tag, values, direction="B - A", verdict="eşit"),
        "a_vs_zs-ana": make_record(
            diff_tag,
            values,
            direction="A - zs-ana",
            verdict="eşit",
            note="Türkiye uyarlama etiketleri + ek 4000 adım eğitim",
        ),
        "b_vs_zs-ana": make_record(diff_tag, values, direction="B - zs-ana", verdict="eşit"),
        "b_vs_a_siniflar": make_record(
            class_tag,
            {
                "damaged_diff_pct": 2.1,
                "damaged_low_pct": 1.1,
                "damaged_high_pct": 3.1,
                "destroyed_diff_pct": -0.2,
                "destroyed_low_pct": -1.2,
                "destroyed_high_pct": 0.8,
            },
            verdict_damaged="B daha iyi",
            verdict_destroyed="eşit",
        ),
        "zs_ana": make_record(zs_tag, {"miou_pct": 38.12}),
    }
    load_split = Mock(return_value={"test": ["tile-b"]})
    compare = Mock(return_value=records)
    monkeypatch.setattr(
        "hasar.cli._load_fewshot",
        lambda: SimpleNamespace(load_split=load_split, compare_fewshot=compare),
        raising=False,
    )
    monkeypatch.setattr("hasar.cli._git_head", lambda: "current-head", raising=False)

    assert main(["fewshot-kiyas"]) == 0
    load_split.assert_called_once_with(Path("results/fewshot/bolme.json"))
    compare.assert_called_once_with(
        Path("results/fewshot/a/turkiye_fewshot_kare_matrisleri.npz"),
        Path("results/fewshot/b/turkiye_fewshot_kare_matrisleri.npz"),
        Path("results/ana/turkiye_kare_matrisleri_headline.npz"),
        test_names=["tile-b"],
        blocks=Path("results/arastirma/turkiye_bloklar.json"),
        out_dir=Path("results/fewshot/kiyas"),
        commit="current-head",
    )
    lines = capsys.readouterr().out.splitlines()
    suffix = f"0.50 puan {diff_tag}; %95 aralık [-0.10, 1.20]; eşit"
    assert lines[:3] == [
        f"B - A: {suffix}",
        f"A - zs-ana: {suffix} (Türkiye uyarlama etiketleri + ek 4000 adım eğitim)",
        f"B - zs-ana: {suffix}",
    ]
    assert f"hasarlı: 2.10 puan {class_tag}; %95 aralık [1.10, 3.10]; B daha iyi" in lines[3]
    assert f"yıkık: -0.20 puan {class_tag}; %95 aralık [-1.20, 0.80]; eşit" in lines[4]
    assert f"38.12 {zs_tag}" in lines[5]


@pytest.mark.parametrize("error", [FileNotFoundError, ValueError, KeyError])
def test_fewshot_compare_reports_input_errors(monkeypatch, capsys, error):
    runner = SimpleNamespace(
        load_split=Mock(return_value={"test": ["tile-b"]}),
        compare_fewshot=Mock(side_effect=error("Geçersiz kıyas girdisi")),
    )
    monkeypatch.setattr("hasar.cli._load_fewshot", lambda: runner, raising=False)
    monkeypatch.setattr("hasar.cli._git_head", lambda: "current-head", raising=False)

    assert main(["fewshot-kiyas"]) == 1
    assert "Geçersiz kıyas girdisi" in capsys.readouterr().err
