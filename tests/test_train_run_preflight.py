"""run_preflight records every gate, survives a crashing gate and writes preflight.json as it goes."""

import json
from unittest.mock import Mock

import pytest
import torch

from hasar.config import Config
from hasar.models import bright_baselines
from hasar.preflight.gates import GateResult

run = pytest.importorskip("hasar.train.run")


@pytest.fixture
def legacy_config(tmp_path):
    """Write the original model initialization explicitly for gate-only tests."""
    config = tmp_path / "legacy.yaml"
    config.write_text("model:\n  name: changemamba_mmbda_tiny\n  init: vssm_tiny_imagenet\n", encoding="utf-8")
    return config


def test_build_model_loads_zenodo_checkpoint_when_md5_matches(monkeypatch, tmp_path) -> None:
    model = torch.nn.Linear(1, 1)
    build = Mock(return_value=model)
    checkpoint = tmp_path / bright_baselines.BASELINES["damageformer"].checkpoint
    digest = Mock(return_value=bright_baselines.BASELINES["damageformer"].md5)
    load = Mock()
    monkeypatch.setattr(run.registry, "build", build)
    monkeypatch.setattr(bright_baselines, "file_md5", digest)
    monkeypatch.setattr(bright_baselines, "load_checkpoint", load)

    assert run.build_model(Config(), checkpoint, "cpu") is model
    build.assert_called_once_with("bright_damageformer")
    digest.assert_called_once_with(checkpoint)
    load.assert_called_once_with(model, checkpoint)
    assert model.weight.device.type == "cpu"


@pytest.mark.parametrize("filename", ["vssm_tiny.pth", "ckpt_DamageFormer_standard_ML_split.pth"])
def test_build_model_rejects_wrong_checkpoint_md5(monkeypatch, tmp_path, filename) -> None:
    monkeypatch.setattr(run.registry, "build", lambda name: torch.nn.Linear(1, 1))
    monkeypatch.setattr(bright_baselines, "file_md5", lambda path: "wrong-md5")
    load = Mock()
    monkeypatch.setattr(bright_baselines, "load_checkpoint", load)

    with pytest.raises(ValueError, match="md5") as error:
        run.build_model(Config(), tmp_path / filename, "cpu")

    assert filename in str(error.value)
    assert bright_baselines.BASELINES["damageformer"].checkpoint in str(error.value)
    load.assert_not_called()


def test_build_model_without_pretrained_returns_raw_damageformer(monkeypatch) -> None:
    model = torch.nn.Linear(1, 1)
    build = Mock(return_value=model)
    digest, load = Mock(), Mock()
    monkeypatch.setattr(run.registry, "build", build)
    monkeypatch.setattr(bright_baselines, "file_md5", digest)
    monkeypatch.setattr(bright_baselines, "load_checkpoint", load)

    assert run.build_model(Config(), None, "cpu") is model
    build.assert_called_once_with("bright_damageformer")
    assert model.weight.device.type == "cpu"
    digest.assert_not_called()
    load.assert_not_called()


def _assert_pretrained_required(runner, monkeypatch, tmp_path) -> None:
    config = tmp_path / "bright.yaml"
    config.write_text("model:\n  name: bright_damageformer\n  init: bright_zero_shot\n", encoding="utf-8")
    datasets = Mock(return_value=(None, [], []))
    clip, current, commit = Mock(return_value=1.0), Mock(return_value=True), Mock(return_value="test")
    monkeypatch.setattr(run, "build_model", Mock(return_value=torch.nn.Linear(1, 1)))
    monkeypatch.setattr(run, "train", Mock(return_value=Mock(step=1, best_miou=0.5)))
    monkeypatch.setattr(run, "build_datasets", datasets)
    monkeypatch.setattr(run, "clip_from_preflight", clip)
    monkeypatch.setattr(run, "preflight_is_current", current)
    monkeypatch.setattr(run, "_git_commit", commit)
    out = tmp_path / "out"

    with pytest.raises(ValueError, match="--pretrained") as error:
        runner(config=config, data=tmp_path / "missing", out=out, device="cpu")

    assert "DamageFormer" in str(error.value)
    assert "zero-shot" in str(error.value)
    assert "hasar agirlik --config" in str(error.value)
    for dependency in (datasets, clip, current, commit):
        dependency.assert_not_called()
    assert not out.exists()


def test_run_training_requires_pretrained_for_bright_zero_shot(monkeypatch, tmp_path) -> None:
    _assert_pretrained_required(run.run_training, monkeypatch, tmp_path)


def test_run_preflight_requires_pretrained_for_bright_zero_shot(monkeypatch, tmp_path) -> None:
    _assert_pretrained_required(run.run_preflight, monkeypatch, tmp_path)


def _passing(gate: int) -> GateResult:
    """Return a passing GateResult for ``gate``."""
    return GateResult(gate=gate, name=f"kapı {gate}", passed=True, value={"ok": True}, detail="geçti")


def _fake_datasets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the dataset builder so no data is read."""
    monkeypatch.setattr(run, "build_datasets", lambda cfg, data: (object(), [0] * 10, [0] * 4))


def _read(path) -> dict:
    """Read a preflight.json file."""
    return json.loads(path.read_text(encoding="utf-8"))


def test_crashing_gate_is_recorded_and_run_continues(monkeypatch: pytest.MonkeyPatch, tmp_path, legacy_config) -> None:
    """A gate that raises does not stop the run: it is recorded as failed and the next gate runs."""
    _fake_datasets(monkeypatch)

    def boom(split):
        raise RuntimeError("boom")

    monkeypatch.setattr(run, "gate_split", boom)
    monkeypatch.setattr(run, "gate_sampler_length", lambda *args, **kwargs: _passing(3))

    code = run.run_preflight(config=legacy_config, data="x", out=tmp_path, device="cpu", gates=[2, 3])

    assert code == 1
    payload = _read(tmp_path / "preflight.json")
    assert payload["complete"] is True
    assert len(payload["gates"]) == 2
    failed = payload["gates"][0]
    assert failed["gate"] == 2
    assert failed["passed"] is False
    assert failed["value"]["error"] == "RuntimeError"
    assert payload["gates"][1]["gate"] == 3


def test_preflight_is_written_after_every_gate(monkeypatch: pytest.MonkeyPatch, tmp_path, legacy_config) -> None:
    """When gate 3 starts, preflight.json already holds gate 2 with complete False."""
    _fake_datasets(monkeypatch)
    monkeypatch.setattr(run, "gate_split", lambda split: _passing(2))

    seen: list[dict] = []

    def gate_three(*args, **kwargs):
        seen.append(_read(tmp_path / "preflight.json"))
        return _passing(3)

    monkeypatch.setattr(run, "gate_sampler_length", gate_three)

    run.run_preflight(config=legacy_config, data="x", out=tmp_path, device="cpu", gates=[2, 3])

    assert len(seen) == 1
    assert seen[0]["complete"] is False
    assert [entry["gate"] for entry in seen[0]["gates"]] == [2]


def test_all_gates_passing_returns_zero(monkeypatch: pytest.MonkeyPatch, tmp_path, legacy_config) -> None:
    """Two passing gates give exit code 0 and a complete, all-passed file."""
    _fake_datasets(monkeypatch)
    monkeypatch.setattr(run, "gate_split", lambda split: _passing(2))
    monkeypatch.setattr(run, "gate_sampler_length", lambda *args, **kwargs: _passing(3))

    code = run.run_preflight(config=legacy_config, data="x", out=tmp_path, device="cpu", gates=[2, 3])

    assert code == 0
    payload = _read(tmp_path / "preflight.json")
    assert payload["complete"] is True
    assert payload["all_passed"] is True
    assert [entry["gate"] for entry in payload["gates"]] == [2, 3]


@pytest.mark.parametrize("record", [None, {}, {"values": {}}])
def test_preflight_gate4_fails_cleanly_without_repro_record(monkeypatch, tmp_path, record) -> None:
    _fake_datasets(monkeypatch)
    config = tmp_path / "bright.yaml"
    config.write_text("{}", encoding="utf-8")
    repro = tmp_path / "repro"
    repro.mkdir()
    record_path = repro / "damageformer_val.json"
    if record is not None:
        record_path.write_text(json.dumps(record), encoding="utf-8")
    build = Mock(side_effect=AssertionError("Model kurulmamalı."))
    monkeypatch.setattr(run, "build_model", build)
    monkeypatch.setattr(run, "_train_batch", lambda *args: {})

    code = run.run_preflight(
        config, "x", tmp_path, device="cpu", pretrained="checkpoint.pth", gates=[4], repro_dir=repro
    )

    assert code == 1
    result = _read(tmp_path / "preflight.json")["gates"][0]
    assert result["gate"] == 4
    assert result["passed"] is False
    assert str(record_path) in result["detail"]
    assert "Kapı çöktü" not in result["detail"]
    build.assert_not_called()


def test_preflight_gate4_reads_expected_miou_from_repro_record(monkeypatch, tmp_path) -> None:
    split, train_set, val_set = object(), [0] * 10, [0] * 4
    monkeypatch.setattr(run, "build_datasets", lambda cfg, data: (split, train_set, val_set))
    config = tmp_path / "bright.yaml"
    config.write_text("{}", encoding="utf-8")
    repro = tmp_path / "repro"
    repro.mkdir()
    (repro / "damageformer_val.json").write_text(json.dumps({"values": {"miou_pct": 74.48}}), encoding="utf-8")
    model = torch.nn.Linear(1, 1)
    build = Mock(return_value=model)
    loaded = Mock(return_value=_passing(4))
    monkeypatch.setattr(run, "build_model", build)
    monkeypatch.setattr(run, "gate_loaded_checkpoint", loaded, raising=False)
    monkeypatch.setattr(run, "_train_batch", lambda *args: {})
    monkeypatch.setattr(run, "gate_initial_loss", lambda *args: _passing(4))

    code = run.run_preflight(
        config, "x", tmp_path, device="cpu", pretrained="checkpoint.pth", gates=[4], repro_dir=repro
    )

    assert code == 0
    loaded.assert_called_once_with(model, val_set, 74.48, "cpu")
    build.assert_called_once_with(Config(), "checkpoint.pth", "cpu")
    assert _read(tmp_path / "preflight.json")["gates"][0]["passed"] is True


@pytest.mark.parametrize("budget", [15.0, 9.0])
def test_preflight_gate8_uses_config_budget(monkeypatch, tmp_path, legacy_config, budget) -> None:
    with legacy_config.open("a", encoding="utf-8") as handle:
        handle.write(f"train:\n  budget_cu: {budget}\n")
    _fake_datasets(monkeypatch)
    monkeypatch.setattr(run, "build_model", lambda *args: torch.nn.Linear(1, 1))
    monkeypatch.setattr(run, "_short_train", lambda *args: [{"wall": 0.5, "seconds": 0.4}] * 12)
    speed = Mock(return_value=_passing(8))
    monkeypatch.setattr(run, "gate_speed", speed)

    code = run.run_preflight(legacy_config, "x", tmp_path, device="cpu", gates=[8])

    assert code == 0
    speed.assert_called_once_with([0.5, 0.5], 4000, 13.0, budget)
