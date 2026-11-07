"""Tests for the ``hasar egit`` and ``hasar on-kontrol`` subcommands of :func:`hasar.cli.main`.

Both subcommands reach the training and preflight runners through the lazy loader
:func:`hasar.cli._load_training`, which imports torch only when a subcommand actually runs. The
tests replace that loader with a fake namespace holding recording ``run_training`` and
``run_preflight`` callables, so everything here is data-free and torch-free.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hasar.cli import main


class FakeRunner:
    """Record every call's keyword arguments and return a fixed exit code."""

    def __init__(self, code: int = 0) -> None:
        self.calls: list[dict[str, object]] = []
        self.code = code

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.code


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Patch ``hasar.cli._load_training`` with a fake namespace and return its parts."""
    training = FakeRunner()
    preflight = FakeRunner()
    namespace = SimpleNamespace(run_training=training, run_preflight=preflight)
    monkeypatch.setattr("hasar.cli._load_training", lambda: namespace, raising=False)
    return SimpleNamespace(training=training, preflight=preflight, namespace=namespace)


# --- cli: egit -------------------------------------------------------------


def test_egit_passes_arguments_through(tmp_path: Path, fake: SimpleNamespace) -> None:
    config = tmp_path / "configs" / "ana.yaml"
    data = tmp_path / "data"
    out = tmp_path / "out"
    pretrained = tmp_path / "vssm_tiny.pth"

    code = main(
        [
            "egit",
            "--config",
            str(config),
            "--data",
            str(data),
            "--out",
            str(out),
            "--device",
            "cuda",
            "--pretrained",
            str(pretrained),
            "--commit",
            "abc1234",
            "--steps",
            "12",
        ]
    )

    assert code == 0
    assert len(fake.training.calls) == 1
    call = fake.training.calls[0]
    assert call["config"] == config
    assert call["data"] == data
    assert call["out"] == out
    assert call["device"] == "cuda"
    assert call["pretrained"] == pretrained
    assert call["commit"] == "abc1234"
    assert call["steps"] == 12
    assert fake.preflight.calls == []


def test_egit_defaults(tmp_path: Path, fake: SimpleNamespace) -> None:
    config = tmp_path / "ana.yaml"
    data = tmp_path / "data"
    out = tmp_path / "out"

    code = main(["egit", "--config", str(config), "--data", str(data), "--out", str(out)])

    assert code == 0
    call = fake.training.calls[0]
    assert call["config"] == config
    assert call["data"] == data
    assert call["out"] == out
    assert call["device"] in ("cuda", "cpu")
    assert call["pretrained"] is None
    assert call["commit"] is None
    assert call["steps"] is None


def test_egit_returns_runner_value(tmp_path: Path, fake: SimpleNamespace) -> None:
    fake.training.code = 3

    code = main(
        [
            "egit",
            "--config",
            str(tmp_path / "ana.yaml"),
            "--data",
            str(tmp_path / "data"),
            "--out",
            str(tmp_path / "out"),
        ]
    )

    assert code == 3


def test_egit_missing_config_exits(tmp_path: Path, fake: SimpleNamespace) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["egit", "--data", str(tmp_path / "data"), "--out", str(tmp_path / "out")])

    assert excinfo.value.code != 0
    assert fake.training.calls == []


# --- cli: on-kontrol -------------------------------------------------------


def test_on_kontrol_passes_arguments_through(tmp_path: Path, fake: SimpleNamespace) -> None:
    config = tmp_path / "ana.yaml"
    data = tmp_path / "data"
    out = tmp_path / "out"
    pretrained = tmp_path / "vssm_tiny.pth"
    repro = tmp_path / "repro"

    code = main(
        [
            "on-kontrol",
            "--config",
            str(config),
            "--data",
            str(data),
            "--out",
            str(out),
            "--device",
            "cpu",
            "--pretrained",
            str(pretrained),
            "--gates",
            "2,3,4",
            "--repro-dir",
            str(repro),
        ]
    )

    assert code == 0
    assert len(fake.preflight.calls) == 1
    call = fake.preflight.calls[0]
    assert call["config"] == config
    assert call["data"] == data
    assert call["out"] == out
    assert call["device"] == "cpu"
    assert call["pretrained"] == pretrained
    assert call["gates"] == [2, 3, 4]
    assert call["repro_dir"] == repro
    assert fake.training.calls == []


def test_on_kontrol_defaults(tmp_path: Path, fake: SimpleNamespace) -> None:
    config = tmp_path / "ana.yaml"
    data = tmp_path / "data"
    out = tmp_path / "out"

    code = main(["on-kontrol", "--config", str(config), "--data", str(data), "--out", str(out)])

    assert code == 0
    call = fake.preflight.calls[0]
    assert call["config"] == config
    assert call["data"] == data
    assert call["out"] == out
    assert call["device"] in ("cuda", "cpu")
    assert call["pretrained"] is None
    assert call["gates"] is None
    assert call["repro_dir"] is None


@pytest.mark.parametrize(("raw", "expected"), [("2,3,4", [2, 3, 4]), ("5", [5]), ("11", [11])])
def test_on_kontrol_parses_gates(tmp_path: Path, fake: SimpleNamespace, raw: str, expected: list[int]) -> None:
    code = main(
        [
            "on-kontrol",
            "--config",
            str(tmp_path / "ana.yaml"),
            "--data",
            str(tmp_path / "data"),
            "--out",
            str(tmp_path / "out"),
            "--gates",
            raw,
        ]
    )

    assert code == 0
    assert fake.preflight.calls[0]["gates"] == expected


def test_on_kontrol_returns_runner_value(tmp_path: Path, fake: SimpleNamespace) -> None:
    fake.preflight.code = 1

    code = main(
        [
            "on-kontrol",
            "--config",
            str(tmp_path / "ana.yaml"),
            "--data",
            str(tmp_path / "data"),
            "--out",
            str(tmp_path / "out"),
        ]
    )

    assert code == 1


def test_on_kontrol_missing_config_exits(tmp_path: Path, fake: SimpleNamespace) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["on-kontrol", "--data", str(tmp_path / "data"), "--out", str(tmp_path / "out")])

    assert excinfo.value.code != 0
    assert fake.preflight.calls == []


# --- cli: existing subcommands keep working --------------------------------


def test_veri_ozet_still_works(tmp_path: Path, fake: SimpleNamespace) -> None:
    out = tmp_path / "ozet.json"

    code = main(["veri-ozet", "--out", str(out)])

    assert code == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["kind"] == "data-summary"
    assert fake.training.calls == []
    assert fake.preflight.calls == []
