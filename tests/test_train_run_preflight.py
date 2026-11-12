"""run_preflight records every gate, survives a crashing gate and writes preflight.json as it goes."""

import json

import pytest

from hasar.preflight.gates import GateResult

run = pytest.importorskip("hasar.train.run")

CONFIG = "configs/ana.yaml"


def _passing(gate: int) -> GateResult:
    """Return a passing GateResult for ``gate``."""
    return GateResult(gate=gate, name=f"kapı {gate}", passed=True, value={"ok": True}, detail="geçti")


def _fake_datasets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the dataset builder so no data is read."""
    monkeypatch.setattr(run, "build_datasets", lambda cfg, data: (object(), [0] * 10, [0] * 4))


def _read(path) -> dict:
    """Read a preflight.json file."""
    return json.loads(path.read_text(encoding="utf-8"))


def test_crashing_gate_is_recorded_and_run_continues(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """A gate that raises does not stop the run: it is recorded as failed and the next gate runs."""
    _fake_datasets(monkeypatch)

    def boom(split):
        raise RuntimeError("boom")

    monkeypatch.setattr(run, "gate_split", boom)
    monkeypatch.setattr(run, "gate_sampler_length", lambda *args, **kwargs: _passing(3))

    code = run.run_preflight(config=CONFIG, data="x", out=tmp_path, device="cpu", gates=[2, 3])

    assert code == 1
    payload = _read(tmp_path / "preflight.json")
    assert payload["complete"] is True
    assert len(payload["gates"]) == 2
    failed = payload["gates"][0]
    assert failed["gate"] == 2
    assert failed["passed"] is False
    assert failed["value"]["error"] == "RuntimeError"
    assert payload["gates"][1]["gate"] == 3


def test_preflight_is_written_after_every_gate(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """When gate 3 starts, preflight.json already holds gate 2 with complete False."""
    _fake_datasets(monkeypatch)
    monkeypatch.setattr(run, "gate_split", lambda split: _passing(2))

    seen: list[dict] = []

    def gate_three(*args, **kwargs):
        seen.append(_read(tmp_path / "preflight.json"))
        return _passing(3)

    monkeypatch.setattr(run, "gate_sampler_length", gate_three)

    run.run_preflight(config=CONFIG, data="x", out=tmp_path, device="cpu", gates=[2, 3])

    assert len(seen) == 1
    assert seen[0]["complete"] is False
    assert [entry["gate"] for entry in seen[0]["gates"]] == [2]


def test_all_gates_passing_returns_zero(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Two passing gates give exit code 0 and a complete, all-passed file."""
    _fake_datasets(monkeypatch)
    monkeypatch.setattr(run, "gate_split", lambda split: _passing(2))
    monkeypatch.setattr(run, "gate_sampler_length", lambda *args, **kwargs: _passing(3))

    code = run.run_preflight(config=CONFIG, data="x", out=tmp_path, device="cpu", gates=[2, 3])

    assert code == 0
    payload = _read(tmp_path / "preflight.json")
    assert payload["complete"] is True
    assert payload["all_passed"] is True
    assert [entry["gate"] for entry in payload["gates"]] == [2, 3]
