"""Tests for the resumable training loop (``hasar.train.loop``).

Everything runs on the CPU with a tiny convolutional model and fake in-memory datasets, so no
data files and no network are needed. The tests pin the observable contract of :func:`train`:
the artefacts it writes, that a resumed run reproduces the uninterrupted one bit for bit, that a
finished run is a no-op, that a changed configuration is refused, that Turkey names are blocked
before the first step, and that an all-void batch is skipped without losing an optimizer step.
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hasar.config import Config, config_hash  # noqa: E402
from hasar.guard import TurkeyLockError  # noqa: E402
from hasar.results import read_result  # noqa: E402
from hasar.train.loop import TrainState, train  # noqa: E402

CROP = 16
TRAIN_NAMES = [f"noto-earthquake_0000000{i}" for i in range(4)]
VAL_NAMES = ["la_palma-volcano_00000001", "beirut-explosion_00000002"]
TURKEY_NAME = "turkey-earthquake_00000001"


# --- fixtures ---------------------------------------------------------------


class TinyModel(torch.nn.Module):
    """Two-head model with a dropout before the convolutions, so RNG state matters."""

    def __init__(self) -> None:
        super().__init__()
        self.drop = torch.nn.Dropout(0.1)
        self.loc = torch.nn.Conv2d(6, 2, 3, padding=1)
        self.clf = torch.nn.Conv2d(6, 4, 3, padding=1)

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.drop(torch.cat([pre, post], dim=1))
        return {"loc": self.loc(features), "clf": self.clf(features)}


def make_model(seed: int) -> TinyModel:
    """Return a freshly initialised :class:`TinyModel` from ``seed``."""
    torch.manual_seed(seed)
    return TinyModel()


def _sample(rng: np.random.Generator, name: str) -> dict[str, object]:
    """Return one deterministic sample drawn from ``rng``."""
    pre = rng.random((3, CROP, CROP), dtype=np.float32)
    post = rng.random((3, CROP, CROP), dtype=np.float32)
    clf = rng.integers(0, 4, size=(CROP, CROP), dtype=np.int64)
    loc = np.where((clf > 0) & (clf < 4), 1, clf).astype(np.int64)
    valid = np.ones((CROP, CROP), dtype=np.bool_)
    return {"name": name, "pre": pre, "post": post, "clf": clf, "loc": loc, "valid": valid}


class FakeTrainSet:
    """Map-style train set keyed by ``(index, sample_no)``, deterministic per key."""

    def __init__(self, names: list[str] | None = None) -> None:
        self.names = list(TRAIN_NAMES if names is None else names)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, key: tuple[int, int]) -> dict[str, object]:
        index, sample_no = key
        rng = np.random.default_rng([index, sample_no])
        return _sample(rng, self.names[index])


class FakeValSet:
    """Map-style eval set keyed by an integer index, deterministic per index."""

    def __init__(self, names: list[str] | None = None) -> None:
        self.names = list(VAL_NAMES if names is None else names)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> dict[str, object]:
        rng = np.random.default_rng([1000 + index])
        return _sample(rng, self.names[index])


class AllVoidTrainSet(FakeTrainSet):
    """Train set whose first batch (sample numbers 0 and 1) is entirely void (255)."""

    def __getitem__(self, key: tuple[int, int]) -> dict[str, object]:
        item = super().__getitem__(key)
        _, sample_no = key
        if sample_no in (0, 1):
            item["clf"] = np.full((CROP, CROP), 255, dtype=np.int64)
            item["loc"] = np.full((CROP, CROP), 255, dtype=np.int64)
        return item


def make_cfg(lr: float | None = None) -> Config:
    """Return the tiny test configuration, optionally with a different learning rate."""
    cfg = Config()
    cfg = dataclasses.replace(
        cfg,
        train=dataclasses.replace(cfg.train, batch=2, steps=6, amp="fp32", ema=0.9, seed=0),
        sched=dataclasses.replace(cfg.sched, warmup_steps=2),
        eval=dataclasses.replace(cfg.eval, every=3),
        ckpt=dataclasses.replace(cfg.ckpt, every=3, keep_last=2),
        data=dataclasses.replace(cfg.data, workers=0, crop=CROP),
    )
    if lr is not None:
        cfg = dataclasses.replace(cfg, optim=dataclasses.replace(cfg.optim, lr=lr))
    return cfg


def read_jsonl(path: Path) -> list[dict[str, object]]:
    """Return the JSON objects of a JSON-lines file, skipping blank lines."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


# --- tests ------------------------------------------------------------------


def test_full_run_writes_expected_artifacts(tmp_path: Path) -> None:
    cfg = make_cfg()
    state = train(cfg, make_model(0), FakeTrainSet(), FakeValSet(), tmp_path, device="cpu", commit="test")

    assert isinstance(state, TrainState)
    assert state.step == 6
    assert state.done is True
    assert 0.0 <= state.best_miou <= 1.0

    train_lines = read_jsonl(tmp_path / "train_log.jsonl")
    assert len(train_lines) == 6
    for step, entry in enumerate(train_lines, start=1):
        assert entry["step"] == step
        for key in ("lr", "loss", "ce_clf", "grad_norm"):
            assert key in entry

    val_lines = read_jsonl(tmp_path / "val_log.jsonl")
    assert [entry["step"] for entry in val_lines] == [3, 6]
    for entry in val_lines:
        assert "miou" in entry

    assert (tmp_path / "last.pt").exists()
    assert (tmp_path / "best.pt").exists()
    assert (tmp_path / "step_000006.pt").exists()
    assert len(list(tmp_path.glob("step_*.pt"))) <= 2

    done = json.loads((tmp_path / "DONE").read_text(encoding="utf-8"))
    assert done["steps"] == 6
    assert done["best_step"] == state.best_step
    assert done["best_miou"] == pytest.approx(state.best_miou)
    assert done["config_hash"] == config_hash(cfg)

    record = read_result(tmp_path / "val13.json")
    assert record.metric_tag == "[VAL13 mIoU, ema]"
    assert "best_miou_pct" in record.values
    assert "last_raw_miou_pct" in record.values


def test_resume_is_deterministic(tmp_path: Path) -> None:
    cfg = make_cfg()
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"

    model_a = make_model(0)
    state_a = train(cfg, model_a, FakeTrainSet(), FakeValSet(), out_a, device="cpu", commit="test")
    assert state_a.done is True
    assert state_a.step == 6

    state_b1 = train(cfg, make_model(0), FakeTrainSet(), FakeValSet(), out_b, device="cpu", commit="test", stop_at=3)
    assert state_b1.done is False
    assert state_b1.step == 3
    assert not (out_b / "DONE").exists()
    last = torch.load(out_b / "last.pt", map_location="cpu", weights_only=False)
    assert last["step"] == 3

    model_b = make_model(999)
    state_b2 = train(cfg, model_b, FakeTrainSet(), FakeValSet(), out_b, device="cpu", commit="test")
    assert state_b2.done is True
    assert state_b2.step == 6

    for (name, param_a), (other_name, param_b) in zip(
        model_a.state_dict().items(), model_b.state_dict().items(), strict=True
    ):
        assert name == other_name
        assert torch.equal(param_a, param_b), name

    losses_a = [entry["loss"] for entry in read_jsonl(out_a / "train_log.jsonl")]
    losses_b = [entry["loss"] for entry in read_jsonl(out_b / "train_log.jsonl")]
    assert len(losses_a) == 6
    assert len(losses_b) >= 3
    for loss_a, loss_b in zip(losses_a[-3:], losses_b[-3:], strict=True):
        assert loss_a == pytest.approx(loss_b, rel=1e-6, abs=1e-6)


def test_done_run_returns_immediately(tmp_path: Path) -> None:
    cfg = make_cfg()
    train(cfg, make_model(0), FakeTrainSet(), FakeValSet(), tmp_path, device="cpu", commit="test")

    before = {path.name: path.stat().st_mtime_ns for path in tmp_path.iterdir()}
    time.sleep(0.05)

    state = train(cfg, make_model(1), FakeTrainSet(), FakeValSet(), tmp_path, device="cpu", commit="test")

    assert state.done is True
    assert state.step == 6
    after = {path.name: path.stat().st_mtime_ns for path in tmp_path.iterdir()}
    assert after == before


def test_resume_with_different_config_raises(tmp_path: Path) -> None:
    cfg = make_cfg()
    train(cfg, make_model(0), FakeTrainSet(), FakeValSet(), tmp_path, device="cpu", commit="test", stop_at=3)

    other = make_cfg(lr=1e-3)
    with pytest.raises(RuntimeError):
        train(other, make_model(0), FakeTrainSet(), FakeValSet(), tmp_path, device="cpu", commit="test")


@pytest.mark.parametrize("bad_split", ["train", "val"])
def test_turkey_names_are_rejected_before_any_step(tmp_path: Path, bad_split: str) -> None:
    cfg = make_cfg()
    train_set = FakeTrainSet()
    val_set = FakeValSet()
    if bad_split == "train":
        train_set = FakeTrainSet([*TRAIN_NAMES, TURKEY_NAME])
    else:
        val_set = FakeValSet([*VAL_NAMES, TURKEY_NAME])

    with pytest.raises(TurkeyLockError):
        train(cfg, make_model(0), train_set, val_set, tmp_path, device="cpu", commit="test")

    assert not (tmp_path / "train_log.jsonl").exists()


def test_all_void_batch_is_skipped(tmp_path: Path) -> None:
    cfg = make_cfg()
    state = train(cfg, make_model(0), AllVoidTrainSet(), FakeValSet(), tmp_path, device="cpu", commit="test")

    assert state.done is True
    assert state.step == 6

    done = json.loads((tmp_path / "DONE").read_text(encoding="utf-8"))
    assert done["steps"] == 6
    assert done["skipped"] >= 1
