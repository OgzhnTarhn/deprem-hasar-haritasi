"""Tests for the ``hasar final`` and ``hasar sicaklik`` subcommands of :func:`hasar.cli.main`.

Both subcommands reach the Turkey evaluation runner through the lazy loader
:func:`hasar.cli._load_final`, which imports torch only when a subcommand actually runs. The
tests replace that loader with a fake namespace holding recording ``run_final_cli`` and
``fit_temperature_cli`` callables, so everything here is data-free and torch-free.
"""

from __future__ import annotations

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
    """Patch ``hasar.cli._load_final`` with a fake namespace and return its parts."""
    final = FakeRunner()
    sicaklik = FakeRunner()
    namespace = SimpleNamespace(run_final_cli=final, fit_temperature_cli=sicaklik)
    monkeypatch.setattr("hasar.cli._load_final", lambda: namespace, raising=False)
    return SimpleNamespace(final=final, sicaklik=sicaklik, namespace=namespace)


# --- cli: final ------------------------------------------------------------


def test_final_passes_arguments_through(tmp_path: Path, fake: SimpleNamespace) -> None:
    config = tmp_path / "configs" / "ana.yaml"
    data = tmp_path / "data"
    runs = tmp_path / "runs"
    out = tmp_path / "results" / "ana"
    cache = tmp_path / "cache"
    instances = tmp_path / "instances"
    blocks = tmp_path / "bloklar.json"

    code = main(
        [
            "final",
            "--config",
            str(config),
            "--data",
            str(data),
            "--runs",
            str(runs),
            "--device",
            "cuda",
            "--out",
            str(out),
            "--cache",
            str(cache),
            "--instances",
            str(instances),
            "--blocks",
            str(blocks),
        ]
    )

    assert code == 0
    assert len(fake.final.calls) == 1
    call = fake.final.calls[0]
    assert call["config"] == config
    assert call["data"] == data
    assert call["runs"] == runs
    assert call["device"] == "cuda"
    assert call["out"] == out
    assert call["cache"] == cache
    assert call["instances"] == instances
    assert call["blocks"] == blocks
    assert fake.sicaklik.calls == []


def test_final_defaults(tmp_path: Path, fake: SimpleNamespace) -> None:
    config = tmp_path / "ana.yaml"
    data = tmp_path / "data"
    runs = tmp_path / "runs"

    code = main(["final", "--config", str(config), "--data", str(data), "--runs", str(runs)])

    assert code == 0
    call = fake.final.calls[0]
    assert call["config"] == config
    assert call["data"] == data
    assert call["runs"] == runs
    assert call["device"] == "cuda"
    assert call["out"] == Path("results/ana")
    assert call["cache"] is None
    assert call["instances"] is None
    assert call["blocks"] == Path("results/arastirma/turkiye_bloklar.json")


def test_final_returns_runner_value(tmp_path: Path, fake: SimpleNamespace) -> None:
    fake.final.code = 2

    code = main(
        [
            "final",
            "--config",
            str(tmp_path / "ana.yaml"),
            "--data",
            str(tmp_path / "data"),
            "--runs",
            str(tmp_path / "runs"),
        ]
    )

    assert code == 2


def test_final_missing_runs_exits(tmp_path: Path, fake: SimpleNamespace) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["final", "--config", str(tmp_path / "ana.yaml"), "--data", str(tmp_path / "data")])

    assert excinfo.value.code != 0
    assert fake.final.calls == []


def test_final_missing_config_exits(tmp_path: Path, fake: SimpleNamespace) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["final", "--data", str(tmp_path / "data"), "--runs", str(tmp_path / "runs")])

    assert excinfo.value.code != 0
    assert fake.final.calls == []


# --- cli: sicaklik ---------------------------------------------------------


def test_sicaklik_passes_arguments_through(tmp_path: Path, fake: SimpleNamespace) -> None:
    config = tmp_path / "ana.yaml"
    data = tmp_path / "data"
    runs = tmp_path / "runs"
    out = tmp_path / "results" / "ana"

    code = main(
        [
            "sicaklik",
            "--config",
            str(config),
            "--data",
            str(data),
            "--runs",
            str(runs),
            "--device",
            "cpu",
            "--out",
            str(out),
        ]
    )

    assert code == 0
    assert len(fake.sicaklik.calls) == 1
    call = fake.sicaklik.calls[0]
    assert call["config"] == config
    assert call["data"] == data
    assert call["runs"] == runs
    assert call["device"] == "cpu"
    assert call["out"] == out
    assert fake.final.calls == []


def test_sicaklik_defaults(tmp_path: Path, fake: SimpleNamespace) -> None:
    config = tmp_path / "ana.yaml"
    data = tmp_path / "data"
    runs = tmp_path / "runs"

    code = main(["sicaklik", "--config", str(config), "--data", str(data), "--runs", str(runs)])

    assert code == 0
    call = fake.sicaklik.calls[0]
    assert call["config"] == config
    assert call["data"] == data
    assert call["runs"] == runs
    assert call["device"] == "cuda"
    assert call["out"] == Path("results/ana")


def test_sicaklik_returns_runner_value(tmp_path: Path, fake: SimpleNamespace) -> None:
    fake.sicaklik.code = 4

    code = main(
        [
            "sicaklik",
            "--config",
            str(tmp_path / "ana.yaml"),
            "--data",
            str(tmp_path / "data"),
            "--runs",
            str(tmp_path / "runs"),
        ]
    )

    assert code == 4


def test_sicaklik_missing_runs_exits(tmp_path: Path, fake: SimpleNamespace) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["sicaklik", "--config", str(tmp_path / "ana.yaml"), "--data", str(tmp_path / "data")])

    assert excinfo.value.code != 0
    assert fake.sicaklik.calls == []


def test_model_label_names_the_fallback_when_it_took_over() -> None:
    from types import SimpleNamespace

    torch = pytest.importorskip("torch")
    from hasar.eval.final_run import model_label

    cfg = SimpleNamespace(model=SimpleNamespace(name="changemamba_mmbda_tiny", fallback="damageformer_swin_t"))

    class ChangeMambaWrapper(torch.nn.Module):
        pass

    assert model_label(cfg, ChangeMambaWrapper()) == "changemamba_mmbda_tiny"
    assert model_label(cfg, torch.nn.Linear(1, 1)) == "damageformer_swin_t"
    other = SimpleNamespace(model=SimpleNamespace(name="damageformer_swin_t", fallback="damageformer_swin_t"))
    assert model_label(other, torch.nn.Linear(1, 1)) == "damageformer_swin_t"
