"""Synthetic coverage for the few-shot training and preflight entry points."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch")

from hasar.config import Config, load_config  # noqa: E402
from hasar.data.sampler import MixedStepSampler  # noqa: E402
from hasar.guard import FEWSHOT_SPLIT_FILE, FewshotAccess  # noqa: E402
from hasar.preflight import gates  # noqa: E402
from hasar.train import run  # noqa: E402

FEWSHOT_A = Path(__file__).resolve().parents[1] / "configs/fewshot_a.yaml"
PRIMARY = [f"noto-earthquake_{i:08}" for i in range(2795, 0, -1)]
VAL = [f"beirut-explosion_{i:08}" for i in range(337)]
TEST = [f"turkey-earthquake_{i:08}" for i in range(1114)]
ADAPT = [TEST[9], TEST[3]]


@pytest.fixture
def ana_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(run.registry, "build", lambda name: torch.nn.Linear(1, 1))

    def save(step, *, matching_sha=True):
        path = tmp_path / "best.pt"
        state = {"weight": torch.tensor([[2.0]]), "bias": torch.tensor([3.0])}
        torch.save({"model": state, "step": step}, path)
        sha = hashlib.sha256(path.read_bytes()).hexdigest() if matching_sha else "0" * 64
        config = tmp_path / "ana_init.yaml"
        config.write_text(f'model:\n  init: ana_run\n  init_sha256: "{sha}"\n', encoding="utf-8")
        return load_config(config), path

    return save


def test_build_model_checks_sha_before_torch_load(ana_checkpoint, monkeypatch):
    cfg, path = ana_checkpoint(4000, matching_sha=False)
    load = Mock(side_effect=AssertionError("Checkpoint deserialized before verification"))
    monkeypatch.setattr(torch, "load", load)
    with pytest.raises(ValueError, match="sha256.*uyuşmazlığı"):
        run.build_model(cfg, path, "cpu")
    load.assert_not_called()


def test_build_model_rejects_non_final_step(ana_checkpoint):
    cfg, path = ana_checkpoint(3999)
    with pytest.raises(ValueError, match="adım.*4000"):
        run.build_model(cfg, path, "cpu")


def test_build_model_loads_ana_ema_weights(ana_checkpoint):
    cfg, path = ana_checkpoint(4000)
    model = run.build_model(cfg, path, "cpu")
    torch.testing.assert_close(model(torch.tensor([[4.0]])), torch.tensor([[11.0]]))


@pytest.fixture
def locked_split(monkeypatch, tmp_path):
    monkeypatch.setattr(run, "load_standard_lists", lambda: (PRIMARY, VAL, TEST))
    access = FewshotAccess("fewshot-kilidi", "test-commit", "adapt", frozenset(ADAPT))
    require = Mock(return_value=access)
    monkeypatch.setattr(run, "require_fewshot_access", require, raising=False)
    split_path = tmp_path / FEWSHOT_SPLIT_FILE
    split_path.parent.mkdir(parents=True)
    parts = {"adapt": ADAPT, "buffer": [TEST[4]], "test": [TEST[5]]}
    digests = {
        key: hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()
        for key, names in parts.items()
    }
    split_path.write_text(json.dumps({**parts, "sha256": digests}), encoding="utf-8")
    return access, require


def test_fewshot_datasets_append_only_locked_adapt_in_sorted_order(tmp_path, locked_split):
    access, require = locked_split
    split, train, val = run.build_datasets(load_config(FEWSHOT_A), tmp_path / "tiles", repo_root=tmp_path)
    require.assert_called_once_with(tmp_path, "adapt")
    assert list(train.names) == PRIMARY + sorted(ADAPT)
    assert tuple(train.names) == split.train
    assert train.access is access
    assert (train.n_primary, train.n_secondary) == (2795, 2)
    assert list(val.names) == VAL
    assert val.access is None
    assert list(split.test) == TEST


def test_zero_shot_datasets_do_not_request_fewshot_access(tmp_path, locked_split):
    _, require = locked_split
    _, train, val = run.build_datasets(Config(), tmp_path)
    require.assert_not_called()
    assert list(train.names) == PRIMARY
    assert list(val.names) == VAL
    assert train.access is None


@pytest.mark.parametrize("runner", [run.run_training, run.run_preflight])
def test_fewshot_entrypoints_require_ana_checkpoint(tmp_path, runner):
    with pytest.raises(ValueError, match=r"best\.pt.*--pretrained"):
        runner(FEWSHOT_A, tmp_path, tmp_path / "out", device="cpu")


def test_fewshot_training_uses_fixed_clip_without_preflight(monkeypatch, tmp_path, locked_split):
    monkeypatch.setattr(run, "build_model", lambda *args: torch.nn.Linear(1, 1))
    monkeypatch.setattr(run, "_git_root", lambda: tmp_path, raising=False)
    train = Mock(return_value=Mock(step=1, best_miou=0.5))
    monkeypatch.setattr(run, "train", train)

    assert run.run_training(FEWSHOT_A, tmp_path, tmp_path / "out", device="cpu", pretrained="best.pt") == 0
    assert train.call_args.args[0].train.clip == 3.1627


def test_fewshot_preflight_checks_split_sampler_and_ana_val13(monkeypatch, tmp_path, locked_split):
    record = tmp_path / "results/ana/val13.json"
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps({"values": {"best_miou_pct": 74.5525}}), encoding="utf-8")
    monkeypatch.setattr(run, "build_model", lambda *args: torch.nn.Linear(1, 1))
    sampler = Mock(wraps=MixedStepSampler)
    monkeypatch.setattr(gates, "MixedStepSampler", sampler, raising=False)

    def validate(model, val, expected, device):
        assert list(val.names) == VAL
        assert expected == 74.5525
        return gates._result(4, True, {"expected_miou_pct": expected}, "geçti")

    monkeypatch.setattr(run, "gate_loaded_checkpoint", validate)
    out = tmp_path / "out"
    code = run.run_preflight(FEWSHOT_A, tmp_path, out, "cpu", "best.pt", gates=[2, 3, 4], repo_root=tmp_path)
    assert code == 0
    split, length, val = json.loads((out / "preflight.json").read_text(encoding="utf-8"))["gates"]
    assert split["value"]["expected"] == [2797, 337, 1114]
    sampler.assert_called_once_with(2795, 2, 8, 42, total_steps=4000)
    assert length["value"]["length"] == 32000
    assert val["value"]["expected_miou_pct"] == 74.5525


@pytest.mark.parametrize("leak", ["validation", "unlisted", "duplicate"])
def test_fewshot_gate2_still_rejects_invalid_splits(tmp_path, locked_split, leak):
    access, _ = locked_split
    split, _, _ = run.build_datasets(load_config(FEWSHOT_A), tmp_path, repo_root=tmp_path)
    if leak == "validation":
        split = replace(split, val=(*split.val[:-1], ADAPT[0]))
    else:
        extra = TEST[5] if leak == "unlisted" else split.train[-2]
        split = replace(split, train=(*split.train[:-1], extra))
    result = gates.gate_split(split, expected=(2797, 337, 1114), allowed_adapt=access.allowed)
    assert result.passed is False


def test_fewshot_gate_batches_retain_access_and_mix_training_pools(monkeypatch, tmp_path, locked_split):
    cfg = load_config(FEWSHOT_A)
    _, train, _ = run.build_datasets(cfg, tmp_path, repo_root=tmp_path)

    def sample(dataset, key):
        assert dataset.access is train.access
        return {"name": dataset.names[key[0]]}

    monkeypatch.setattr(run.BrightDataset, "__getitem__", sample)
    assert run._fixed_batch(train, cfg)["name"] == PRIMARY[:8]
    names = run._train_batch(train, cfg)["name"]
    assert all(name in ADAPT for name in names[::2])
    assert all(name in PRIMARY for name in names[1::2])


def test_cached_ana_preflight_still_checks_checkpoint_sha(monkeypatch, tmp_path, ana_checkpoint):
    cfg, checkpoint = ana_checkpoint(4000, matching_sha=False)
    monkeypatch.setattr(run, "preflight_is_current", lambda *args: True)
    monkeypatch.setattr(run, "load_config", lambda path: cfg)
    with pytest.raises(ValueError, match="sha256.*uyuşmazlığı"):
        run.run_preflight("ana_init.yaml", tmp_path, tmp_path, "cpu", checkpoint)


@pytest.mark.parametrize("record", [None, {"values": {"best_miou_pct": "bozuk"}}])
def test_fewshot_gate4_reports_missing_or_invalid_ana_record(monkeypatch, tmp_path, locked_split, record):
    build = Mock(side_effect=AssertionError("Gate must report the record error before building the model"))
    monkeypatch.setattr(run, "build_model", build)
    record_path = tmp_path / "results/ana/val13.json"
    if record is not None:
        record_path.parent.mkdir(parents=True)
        record_path.write_text(json.dumps(record), encoding="utf-8")
    code = run.run_preflight(FEWSHOT_A, tmp_path, tmp_path, "cpu", "best.pt", gates=[4], repo_root=tmp_path)
    assert code == 1
    result = json.loads((tmp_path / "preflight.json").read_text(encoding="utf-8"))["gates"][0]
    assert result["passed"] is False
    assert "values.best_miou_pct" in result["detail"]
    assert result["value"]["path"] == str(record_path)
    build.assert_not_called()


def test_fewshot_gate10_rebuilds_datasets_with_repo_root_and_valid_micro_batch(monkeypatch, tmp_path, locked_split):
    _, require = locked_split

    def resume(cfg, model_factory, train, val, out, *, stop, device):
        assert cfg.train.micro_batch == cfg.train.batch == 2
        assert cfg.train.oom_halving is False
        assert train.n_secondary == len(ADAPT)
        assert list(val.names) == VAL[:8]
        return gates._result(10, True, {}, "geçti")

    monkeypatch.setattr(run, "gate_resume", resume)
    code = run.run_preflight(FEWSHOT_A, tmp_path, tmp_path, "cpu", "best.pt", gates=[10], repo_root=tmp_path)
    assert code == 0
    assert [call.args for call in require.call_args_list] == [(tmp_path, "adapt")] * 2
