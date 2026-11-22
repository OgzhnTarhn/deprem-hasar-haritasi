"""The gradient-clip threshold comes from gate 9 of a current preflight.json.

``clip_from_preflight`` reads gate 9's ``value["clip"]`` only from a complete, all-passed file whose
config hash matches; ``run_training`` uses it when the YAML leaves ``train.clip`` null, and the CLI
forwards ``--preflight`` to the runner.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from hasar.config import config_hash, load_config, run_id  # noqa: E402
from hasar.preflight.gates import GateResult, write_preflight  # noqa: E402
from hasar.train.loop import TrainState  # noqa: E402

run = pytest.importorskip("hasar.train.run")

CONFIG = "configs/ana.yaml"


@pytest.fixture
def legacy_config(tmp_path: Path) -> Path:
    """Write the original initialization explicitly for gradient-clip tests."""
    config = tmp_path / "legacy.yaml"
    config.write_text(
        "model:\n  name: changemamba_mmbda_tiny\n  init: vssm_tiny_imagenet\ntrain:\n  clip: null\n",
        encoding="utf-8",
    )
    return config


def _clip_from_preflight(path: Path, cfg_hash: str) -> float:
    """Call the gate-9 clip reader (imported lazily so a missing feature fails one test, not collection)."""
    from hasar.preflight.gates import clip_from_preflight

    return clip_from_preflight(path, cfg_hash)


def _gate(gate: int, passed: bool = True, value: dict | None = None) -> GateResult:
    """Return a GateResult for ``gate`` with the given verdict and value."""
    return GateResult(gate=gate, name=f"kapı {gate}", passed=passed, value=value or {}, detail="test")


def _write(path: Path, results: list[GateResult], cfg_hash: str, complete: bool = True) -> Path:
    """Write a preflight.json with the given gate results."""
    return write_preflight(path, results, cfg_hash, "abc", complete=complete)


def _clip_gates(clip: float | None = 2.5, passed: bool = True) -> list[GateResult]:
    """Return a single gate 9 result carrying ``clip``."""
    return [_gate(9, passed=passed, value={"clip": clip})]


# --- clip_from_preflight ----------------------------------------------------


def test_clip_from_preflight_returns_gate_nine_clip(tmp_path: Path) -> None:
    """A current, complete, all-passed file returns gate 9's clip value."""
    cfg_hash = config_hash(load_config(CONFIG))
    path = _write(tmp_path / "preflight.json", _clip_gates(2.5), cfg_hash)

    assert _clip_from_preflight(path, cfg_hash) == 2.5


def test_clip_from_preflight_missing_file_raises(tmp_path: Path) -> None:
    """A missing preflight.json raises RuntimeError."""
    cfg_hash = config_hash(load_config(CONFIG))

    with pytest.raises(RuntimeError):
        _clip_from_preflight(tmp_path / "preflight.json", cfg_hash)


def test_clip_from_preflight_hash_mismatch_raises(tmp_path: Path) -> None:
    """A file written for another config hash raises RuntimeError."""
    cfg_hash = config_hash(load_config(CONFIG))
    path = _write(tmp_path / "preflight.json", _clip_gates(2.5), "deadbeef0000")

    with pytest.raises(RuntimeError):
        _clip_from_preflight(path, cfg_hash)


def test_clip_from_preflight_failed_gate_raises(tmp_path: Path) -> None:
    """A file with a failed gate (all_passed false) raises RuntimeError."""
    cfg_hash = config_hash(load_config(CONFIG))
    results = [_gate(8, passed=True, value={"ok": True}), _gate(9, passed=False, value={"clip": 2.5})]
    path = _write(tmp_path / "preflight.json", results, cfg_hash)

    with pytest.raises(RuntimeError):
        _clip_from_preflight(path, cfg_hash)


def test_clip_from_preflight_incomplete_raises(tmp_path: Path) -> None:
    """A file written with complete=False raises RuntimeError."""
    cfg_hash = config_hash(load_config(CONFIG))
    path = _write(tmp_path / "preflight.json", _clip_gates(2.5), cfg_hash, complete=False)

    with pytest.raises(RuntimeError):
        _clip_from_preflight(path, cfg_hash)


def test_clip_from_preflight_without_gate_nine_raises(tmp_path: Path) -> None:
    """A file that never ran gate 9 raises RuntimeError."""
    cfg_hash = config_hash(load_config(CONFIG))
    path = _write(tmp_path / "preflight.json", [_gate(8, value={"ok": True})], cfg_hash)

    with pytest.raises(RuntimeError):
        _clip_from_preflight(path, cfg_hash)


@pytest.mark.parametrize("value", [{}, {"clip": None}, {"clip": 0.0}, {"clip": -1.0}])
def test_clip_from_preflight_bad_clip_raises(tmp_path: Path, value: dict) -> None:
    """A missing, None or non-positive clip raises RuntimeError."""
    cfg_hash = config_hash(load_config(CONFIG))
    path = _write(tmp_path / "preflight.json", [_gate(9, value=value)], cfg_hash)

    with pytest.raises(RuntimeError):
        _clip_from_preflight(path, cfg_hash)


# --- run_training -----------------------------------------------------------


def _fake_run(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Replace the dataset builder, the model builder and the training loop with recorders."""
    calls: list[tuple[object, Path]] = []
    built = {"datasets": 0}

    def fake_datasets(cfg, data):
        built["datasets"] += 1
        return object(), [0] * 10, [0] * 4

    def fake_train(cfg, model, train_set, val_set, out_dir, **kwargs):
        calls.append((cfg, Path(out_dir)))
        return TrainState(step=1, done=True, best_step=1, best_miou=0.5)

    monkeypatch.setattr(run, "build_datasets", fake_datasets)
    monkeypatch.setattr(run, "build_model", lambda *args, **kwargs: torch.nn.Linear(1, 1))
    monkeypatch.setattr(run, "train", fake_train)
    return SimpleNamespace(calls=calls, built=built)


def test_run_training_takes_clip_from_out_preflight(monkeypatch, tmp_path, legacy_config) -> None:
    """A valid preflight.json in the output folder supplies the clip; the run folder keeps the YAML hash."""
    fake = _fake_run(monkeypatch)
    cfg = load_config(legacy_config)
    _write(tmp_path / "preflight.json", _clip_gates(2.5), config_hash(cfg))

    code = run.run_training(config=legacy_config, data="x", out=tmp_path, device="cpu", commit="abc")

    assert code == 0
    assert len(fake.calls) == 1
    cfg_seen, out_dir = fake.calls[0]
    assert cfg_seen.train.clip == 2.5
    assert out_dir == Path(tmp_path) / run_id(cfg)


def test_run_training_takes_clip_from_given_preflight(monkeypatch, tmp_path, legacy_config) -> None:
    """An explicit preflight path is used when the output folder holds no preflight.json."""
    fake = _fake_run(monkeypatch)
    cfg = load_config(legacy_config)
    other = tmp_path / "baska" / "preflight.json"
    _write(other, _clip_gates(2.5), config_hash(cfg))

    code = run.run_training(config=legacy_config, data="x", out=tmp_path, device="cpu", commit="abc", preflight=other)

    assert code == 0
    assert len(fake.calls) == 1
    cfg_seen, out_dir = fake.calls[0]
    assert cfg_seen.train.clip == 2.5
    assert out_dir == Path(tmp_path) / run_id(cfg)


def test_run_training_without_preflight_raises(monkeypatch, tmp_path, legacy_config) -> None:
    """A null clip with no preflight.json raises before any dataset or training work."""
    fake = _fake_run(monkeypatch)

    with pytest.raises(RuntimeError):
        run.run_training(config=legacy_config, data="x", out=tmp_path, device="cpu", commit="abc")

    assert fake.built["datasets"] == 0
    assert fake.calls == []


def test_run_training_with_stale_preflight_raises(monkeypatch, tmp_path, legacy_config) -> None:
    """A preflight.json written for another config hash raises RuntimeError."""
    fake = _fake_run(monkeypatch)
    _write(tmp_path / "preflight.json", _clip_gates(2.5), "deadbeef0000")

    with pytest.raises(RuntimeError):
        run.run_training(config=legacy_config, data="x", out=tmp_path, device="cpu", commit="abc")

    assert fake.built["datasets"] == 0
    assert fake.calls == []


def test_run_training_steps_override_keeps_yaml_hash(monkeypatch, tmp_path, legacy_config) -> None:
    """A short run with steps=7 uses the YAML-hash preflight and names its folder after the steps config."""
    fake = _fake_run(monkeypatch)
    cfg = load_config(legacy_config)
    _write(tmp_path / "preflight.json", _clip_gates(2.5), config_hash(cfg))

    code = run.run_training(config=legacy_config, data="x", out=tmp_path, device="cpu", commit="abc", steps=7)

    assert code == 0
    assert len(fake.calls) == 1
    cfg_seen, out_dir = fake.calls[0]
    assert cfg_seen.train.steps == 7
    assert cfg_seen.train.clip == 2.5
    expected = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, steps=7))
    assert out_dir.name == run_id(expected)


def test_run_training_explicit_clip_needs_no_preflight(monkeypatch, tmp_path, legacy_config) -> None:
    """A YAML with an explicit clip trains without any preflight.json."""
    fake = _fake_run(monkeypatch)
    text = legacy_config.read_text(encoding="utf-8")
    assert "clip: null" in text
    config = tmp_path / "ana.yaml"
    config.write_text(text.replace("clip: null", "clip: 1.5"), encoding="utf-8")

    code = run.run_training(config=config, data="x", out=tmp_path, device="cpu", commit="abc")

    assert code == 0
    assert len(fake.calls) == 1
    cfg_seen, _ = fake.calls[0]
    assert cfg_seen.train.clip == 1.5


# --- cli --------------------------------------------------------------------


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


def test_egit_forwards_preflight(tmp_path: Path, fake: SimpleNamespace) -> None:
    """``hasar egit --preflight`` passes the path through to run_training."""
    from hasar.cli import main

    p = tmp_path / "preflight.json"

    code = main(["egit", "--config", CONFIG, "--data", "x", "--out", str(tmp_path), "--preflight", str(p)])

    assert code == 0
    assert len(fake.training.calls) == 1
    assert fake.training.calls[0]["preflight"] == p
