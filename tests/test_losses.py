"""Tests for the training losses (``hasar.train.losses``).

Everything runs on the CPU with tiny tensors, so no data files and no network are needed. The
Lovász cross-check loads the vendored ChangeMamba implementation by file path and compares the two
on seeded random inputs, including ignored (255) pixels.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import torch.nn.functional as F  # noqa: E402

from hasar.config import Loss  # noqa: E402
from hasar.train.losses import damage_loss, lovasz_softmax  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDORED_PATH = REPO_ROOT / "third_party" / "changemamba" / "changedetection" / "utils_func" / "lovasz_loss.py"


def load_vendored():
    """Load the vendored ChangeMamba ``lovasz_loss`` module from its file path."""
    spec = importlib.util.spec_from_file_location("changemamba_lovasz_loss", VENDORED_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def random_case(seed: int, batch: int = 2, classes: int = 4, size: int = 8):
    """Return seeded random probabilities, labels and logits with some ignored pixels."""
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(batch, classes, size, size, generator=generator)
    probs = torch.softmax(logits, dim=1)
    labels = torch.randint(0, classes, (batch, size, size), generator=generator, dtype=torch.int64)
    # Ignore a few pixels so the ignore path is exercised.
    labels[0, 0, 0] = 255
    labels[1, size - 1, size - 1] = 255
    return probs, labels, logits


# --- lovasz_softmax ---------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_lovasz_softmax_matches_vendored(seed: int) -> None:
    vendored = load_vendored()
    probs, labels, _ = random_case(seed)

    expected = vendored.lovasz_softmax(probs, labels, ignore=255)
    actual = lovasz_softmax(probs, labels, ignore=255)

    assert torch.allclose(actual, expected, atol=1e-6)


def test_lovasz_softmax_perfect_prediction_is_zero() -> None:
    labels = torch.tensor([[[0, 1], [2, 3]]], dtype=torch.int64)
    probs = F.one_hot(labels, num_classes=4).permute(0, 3, 1, 2).float()

    loss = lovasz_softmax(probs, labels, ignore=255)

    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_lovasz_softmax_all_ignored_is_zero_without_nan() -> None:
    probs = torch.softmax(torch.randn(1, 4, 4, 4), dim=1)
    labels = torch.full((1, 4, 4), 255, dtype=torch.int64)

    loss = lovasz_softmax(probs, labels, ignore=255)

    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


# --- damage_loss ------------------------------------------------------------


def manual_total(outputs, loc_target, clf_target, cfg: Loss) -> float:
    """Recompute the total loss by hand from the vendored Lovász and ``F.cross_entropy``."""
    vendored = load_vendored()
    weight = torch.tensor(cfg.class_weights, dtype=torch.float32)

    ce_loc = F.cross_entropy(outputs["loc"], loc_target, ignore_index=cfg.ignore_index)
    ce_clf = F.cross_entropy(outputs["clf"], clf_target, weight=weight, ignore_index=cfg.ignore_index)
    lovasz_loc = vendored.lovasz_softmax(torch.softmax(outputs["loc"], dim=1), loc_target, ignore=cfg.ignore_index)
    lovasz_clf = vendored.lovasz_softmax(torch.softmax(outputs["clf"], dim=1), clf_target, ignore=cfg.ignore_index)
    return (
        cfg.ce_loc * ce_loc + cfg.ce_clf * ce_clf + cfg.lovasz_loc * lovasz_loc + cfg.lovasz_clf * lovasz_clf
    ).item()


def test_damage_loss_total_matches_manual_computation() -> None:
    cfg = Loss()
    generator = torch.Generator().manual_seed(7)
    outputs = {
        "loc": torch.randn(2, 2, 6, 6, generator=generator),
        "clf": torch.randn(2, 4, 6, 6, generator=generator),
    }
    loc_target = torch.randint(0, 2, (2, 6, 6), generator=generator, dtype=torch.int64)
    clf_target = torch.randint(0, 4, (2, 6, 6), generator=generator, dtype=torch.int64)
    loc_target[0, 0, 0] = 255
    clf_target[1, 5, 5] = 255

    total, parts = damage_loss(outputs, loc_target, clf_target, cfg)

    assert total.item() == pytest.approx(manual_total(outputs, loc_target, clf_target, cfg), abs=1e-5)
    assert set(parts) == {"ce_loc", "ce_clf", "lovasz_loc", "lovasz_clf", "total"}
    for name, value in parts.items():
        assert isinstance(value, float), name
    assert parts["total"] == pytest.approx(total.item(), abs=1e-6)


def test_damage_loss_uniform_logits_give_log_class_counts() -> None:
    cfg = Loss()
    outputs = {
        "loc": torch.zeros(1, 2, 4, 4),
        "clf": torch.zeros(1, 4, 4, 4),
    }
    loc_target = torch.zeros(1, 4, 4, dtype=torch.int64)
    clf_target = torch.zeros(1, 4, 4, dtype=torch.int64)

    _, parts = damage_loss(outputs, loc_target, clf_target, cfg)

    assert parts["ce_clf"] == pytest.approx(math.log(4), abs=1e-5)
    assert parts["ce_loc"] == pytest.approx(math.log(2), abs=1e-5)


def test_damage_loss_all_ignored_is_zero_and_differentiable() -> None:
    cfg = Loss()
    outputs = {
        "loc": torch.randn(1, 2, 4, 4, requires_grad=True),
        "clf": torch.randn(1, 4, 4, 4, requires_grad=True),
    }
    loc_target = torch.full((1, 4, 4), 255, dtype=torch.int64)
    clf_target = torch.full((1, 4, 4), 255, dtype=torch.int64)

    total, _ = damage_loss(outputs, loc_target, clf_target, cfg)

    assert torch.isfinite(total)
    assert total.item() == pytest.approx(0.0, abs=1e-6)
    total.backward()
    assert outputs["loc"].grad is not None
    assert outputs["clf"].grad is not None


def test_damage_loss_autocast_bf16_keeps_lovasz_in_float32() -> None:
    cfg = Loss(lovasz_fp32=True)
    generator = torch.Generator().manual_seed(11)
    loc_target = torch.randint(0, 2, (1, 4, 4), generator=generator, dtype=torch.int64)
    clf_target = torch.randint(0, 4, (1, 4, 4), generator=generator, dtype=torch.int64)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        outputs = {
            "loc": torch.randn(1, 2, 4, 4, generator=generator, dtype=torch.bfloat16),
            "clf": torch.randn(1, 4, 4, 4, generator=generator, dtype=torch.bfloat16),
        }
        total, parts = damage_loss(outputs, loc_target, clf_target, cfg)

    assert total.dtype == torch.float32
    assert torch.isfinite(total)
    assert all(math.isfinite(value) for value in parts.values())


def test_damage_loss_backward_gives_finite_grads() -> None:
    cfg = Loss()
    generator = torch.Generator().manual_seed(13)
    outputs = {
        "loc": torch.randn(2, 2, 5, 5, generator=generator, requires_grad=True),
        "clf": torch.randn(2, 4, 5, 5, generator=generator, requires_grad=True),
    }
    loc_target = torch.randint(0, 2, (2, 5, 5), generator=generator, dtype=torch.int64)
    clf_target = torch.randint(0, 4, (2, 5, 5), generator=generator, dtype=torch.int64)

    total, _ = damage_loss(outputs, loc_target, clf_target, cfg)
    total.backward()

    for name in ("loc", "clf"):
        grad = outputs[name].grad
        assert grad is not None, name
        assert torch.isfinite(grad).all(), name
