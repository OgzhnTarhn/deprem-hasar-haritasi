"""Tests for the training building blocks (``hasar.train.schedule``, ``ema``, ``checkpoint``).

Everything runs on the CPU with tiny tensors and ``tmp_path``; no data files and no network are
needed. The schedule is checked against the closed-form learning rate, the EMA against the
hand-computed first update, and the checkpoint helpers against a real save/load round-trip.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hasar.config import Sched  # noqa: E402
from hasar.train.checkpoint import (  # noqa: E402
    capture_rng,
    load_checkpoint,
    prune_checkpoints,
    restore_rng,
    save_checkpoint,
)
from hasar.train.ema import ModelEMA  # noqa: E402
from hasar.train.schedule import lr_at, make_scheduler  # noqa: E402

BASE_LR = 1e-4
MIN_LR = 1e-6
WARMUP = 10
TOTAL = 100


# --- schedule ---------------------------------------------------------------


def test_lr_at_warmup_start_is_base_over_warmup():
    assert lr_at(0, BASE_LR, WARMUP, TOTAL, MIN_LR) == pytest.approx(BASE_LR / WARMUP)


def test_lr_at_last_warmup_step_is_base():
    assert lr_at(WARMUP - 1, BASE_LR, WARMUP, TOTAL, MIN_LR) == pytest.approx(BASE_LR)


def test_lr_at_first_cosine_step_is_base():
    assert lr_at(WARMUP, BASE_LR, WARMUP, TOTAL, MIN_LR) == pytest.approx(BASE_LR)


def test_lr_at_total_is_min_lr():
    assert lr_at(TOTAL, BASE_LR, WARMUP, TOTAL, MIN_LR) == pytest.approx(MIN_LR)


def test_lr_at_matches_cosine_formula():
    step = 55
    t = (step - WARMUP) / (TOTAL - WARMUP)
    expected = MIN_LR + 0.5 * (BASE_LR - MIN_LR) * (1 + math.cos(math.pi * t))
    assert lr_at(step, BASE_LR, WARMUP, TOTAL, MIN_LR) == pytest.approx(expected)


def test_lr_at_is_monotone_non_increasing_after_warmup():
    values = [lr_at(step, BASE_LR, WARMUP, TOTAL, MIN_LR) for step in range(WARMUP, TOTAL + 1)]
    for previous, current in zip(values, values[1:], strict=False):
        assert current <= previous + 1e-12


def test_lr_at_without_warmup_starts_at_base():
    assert lr_at(0, BASE_LR, 0, TOTAL, MIN_LR) == pytest.approx(BASE_LR)
    assert lr_at(TOTAL, BASE_LR, 0, TOTAL, MIN_LR) == pytest.approx(MIN_LR)


def test_lr_at_clamps_past_total():
    assert lr_at(TOTAL + 50, BASE_LR, WARMUP, TOTAL, MIN_LR) == pytest.approx(MIN_LR)


def make_optimizer(lr: float = BASE_LR) -> torch.optim.Optimizer:
    """Return a one-parameter SGD optimizer at ``lr``."""
    param = torch.nn.Parameter(torch.zeros(1))
    return torch.optim.SGD([param], lr=lr)


@pytest.mark.parametrize("k", [0, 5, 50])
def test_make_scheduler_follows_lr_at(k):
    optimizer = make_optimizer()
    sched_cfg = Sched(warmup_steps=WARMUP, min_lr=MIN_LR)
    scheduler = make_scheduler(optimizer, sched_cfg, BASE_LR, TOTAL)

    assert isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR)
    for _ in range(k):
        scheduler.step()

    expected = lr_at(k, BASE_LR, WARMUP, TOTAL, MIN_LR)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(expected)


# --- EMA --------------------------------------------------------------------


def make_linear() -> torch.nn.Linear:
    """Return a deterministic 2x3 Linear layer."""
    torch.manual_seed(0)
    return torch.nn.Linear(3, 2)


def test_model_ema_module_is_deep_copy_in_eval_mode():
    model = make_linear()
    ema = ModelEMA(model, decay=0.999)

    assert ema.module is not model
    assert not ema.module.training
    assert all(not param.requires_grad for param in ema.module.parameters())
    for ema_param, param in zip(ema.module.parameters(), model.parameters(), strict=False):
        assert torch.equal(ema_param, param)


def test_model_ema_first_update_uses_step_zero_decay():
    model = make_linear()
    ema = ModelEMA(model, decay=0.999)
    old_weight = ema.module.weight.detach().clone()

    with torch.no_grad():
        model.weight.add_(1.0)
    new_weight = model.weight.detach().clone()

    ema.update(model, step=0)

    d = 0.1  # min(0.999, (1 + 0) / (10 + 0))
    expected = d * old_weight + (1 - d) * new_weight
    assert torch.allclose(ema.module.weight, expected)


def test_model_ema_copies_non_floating_buffers():
    model = torch.nn.BatchNorm1d(3)
    model.register_buffer("counter", torch.tensor(7, dtype=torch.long))
    ema = ModelEMA(model, decay=0.999)

    model.counter.fill_(42)
    ema.update(model, step=0)

    assert torch.equal(ema.module.counter, torch.tensor(42, dtype=torch.long))


def test_model_ema_state_dict_round_trip():
    model = make_linear()
    ema = ModelEMA(model, decay=0.999)
    with torch.no_grad():
        model.weight.add_(0.5)
    ema.update(model, step=0)

    state = ema.state_dict()
    other = ModelEMA(make_linear(), decay=0.999)
    other.load_state_dict(state)

    for key, value in ema.module.state_dict().items():
        assert torch.equal(other.module.state_dict()[key], value)


# --- checkpoint -------------------------------------------------------------


def test_capture_and_restore_rng_reproduces_draws():
    state = capture_rng()

    random.random()
    np.random.rand()
    torch.rand(4)

    restore_rng(state)

    py_value = random.random()
    np_value = np.random.rand()
    torch_value = torch.rand(4)

    restore_rng(state)

    assert random.random() == py_value
    assert np.random.rand() == np_value
    assert torch.equal(torch.rand(4), torch_value)


def test_save_checkpoint_is_atomic_and_round_trips(tmp_path):
    path = tmp_path / "step_000001.pt"
    state = {
        "tensor": torch.arange(3),
        "step": 7,
        "nested": {"lr": 1e-4, "name": "ana"},
    }

    save_checkpoint(path, state)

    assert path.exists()
    assert not (tmp_path / "step_000001.pt.tmp").exists()

    loaded = load_checkpoint(path)
    assert torch.equal(loaded["tensor"], state["tensor"])
    assert loaded["step"] == 7
    assert loaded["nested"] == {"lr": 1e-4, "name": "ana"}


def test_prune_checkpoints_keeps_highest_steps(tmp_path):
    for step in range(1, 6):
        (tmp_path / f"step_{step:06d}.pt").write_bytes(b"x")
    unrelated = tmp_path / "best.pt"
    unrelated.write_bytes(b"x")

    prune_checkpoints(tmp_path, keep=2)

    remaining = sorted(path.name for path in tmp_path.glob("step_*.pt"))
    assert remaining == ["step_000004.pt", "step_000005.pt"]
    assert unrelated.exists()


def test_prune_checkpoints_keep_zero_removes_all(tmp_path):
    for step in range(1, 4):
        (tmp_path / f"step_{step:06d}.pt").write_bytes(b"x")

    prune_checkpoints(tmp_path, keep=0)

    assert list(tmp_path.glob("step_*.pt")) == []


def test_prune_checkpoints_leaves_unrelated_files(tmp_path):
    (tmp_path / "step_000001.pt").write_bytes(b"x")
    (tmp_path / "step_000002.pt").write_bytes(b"x")
    other = tmp_path / "notes.txt"
    other.write_bytes(b"x")

    prune_checkpoints(tmp_path, keep=1)

    assert [path.name for path in tmp_path.glob("step_*.pt")] == ["step_000002.pt"]
    assert other.exists()
