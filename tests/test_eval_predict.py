"""Tests for the tile-by-tile prediction and evaluation loop (``hasar.eval.predict``).

Everything runs on the CPU with tiny 16x16 tensors and a fake model defined here; no real BRIGHT
data, no checkpoint and no network. The fake model follows the model contract: ``forward(pre,
post)`` returns ``{"clf": ..., "loc": ...}`` logits.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hasar.eval.metrics import ConfusionAccumulator, confusion_matrix  # noqa: E402
from hasar.eval.predict import evaluate, predict  # noqa: E402

SIZE = 16
NUM_CLF = 4
NUM_LOC = 2


class FakeModel(torch.nn.Module):
    """1x1 convolution on the concatenated pre/post stack; flip-equivariant by construction."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.clf = torch.nn.Conv2d(6, NUM_CLF, kernel_size=1)
        self.loc = torch.nn.Conv2d(6, NUM_LOC, kernel_size=1)
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator))

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> dict[str, torch.Tensor]:
        features = torch.cat([pre, post], dim=1)
        return {"clf": self.clf(features), "loc": self.loc(features)}


class AsymmetricModel(FakeModel):
    """Adds a horizontal ramp to the input, so a horizontal flip is not a symmetry."""

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> dict[str, torch.Tensor]:
        ramp = torch.linspace(0.0, 1.0, pre.shape[-1]).view(1, 1, 1, -1)
        return super().forward(pre + ramp, post)


def _inputs(seed: int = 1, batch: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    pre = torch.randn(batch, 3, SIZE, SIZE, generator=generator)
    post = torch.randn(batch, 3, SIZE, SIZE, generator=generator)
    return pre, post


def _item(name: str, seed: int) -> dict[str, object]:
    """Return one eval item shaped like a ``BrightDataset`` sample."""
    rng = np.random.default_rng(seed)
    clf = rng.integers(0, NUM_CLF, size=(SIZE, SIZE)).astype(np.int64)
    clf[0, :4] = 255  # nodata pixels must be dropped by the confusion matrix
    loc = np.where((clf > 0) & (clf < NUM_CLF), 1, clf).astype(np.int64)
    return {
        "name": name,
        "pre": rng.standard_normal((3, SIZE, SIZE)).astype(np.float32),
        "post": rng.standard_normal((3, SIZE, SIZE)).astype(np.float32),
        "clf": clf,
        "loc": loc,
        "valid": rng.random((SIZE, SIZE)) > 0.1,
    }


def _dataset() -> list[dict[str, object]]:
    return [_item("tile_a", 10), _item("tile_b", 11), _item("tile_c", 12)]


# --------------------------------------------------------------------------- predict


def test_predict_returns_probabilities() -> None:
    model = FakeModel()
    pre, post = _inputs()

    clf, loc = predict(model, pre, post)

    assert clf.shape == (2, NUM_CLF, SIZE, SIZE)
    assert loc.shape == (2, NUM_LOC, SIZE, SIZE)
    assert torch.all(clf >= 0) and torch.all(loc >= 0)
    assert torch.allclose(clf.sum(dim=1), torch.ones(2, SIZE, SIZE), atol=1e-6)
    assert torch.allclose(loc.sum(dim=1), torch.ones(2, SIZE, SIZE), atol=1e-6)


def test_predict_matches_softmax_of_logits() -> None:
    model = FakeModel()
    pre, post = _inputs()

    clf, loc = predict(model, pre, post)

    with torch.no_grad():
        logits = model(pre, post)
    assert torch.allclose(clf, torch.softmax(logits["clf"], dim=1), atol=1e-6)
    assert torch.allclose(loc, torch.softmax(logits["loc"], dim=1), atol=1e-6)


def test_predict_tta_is_identity_for_equivariant_model() -> None:
    model = FakeModel()
    pre, post = _inputs()

    plain_clf, plain_loc = predict(model, pre, post, tta=False)
    tta_clf, tta_loc = predict(model, pre, post, tta=True)

    assert torch.allclose(plain_clf, tta_clf, atol=1e-6)
    assert torch.allclose(plain_loc, tta_loc, atol=1e-6)


def test_predict_tta_changes_asymmetric_model() -> None:
    model = AsymmetricModel()
    pre, post = _inputs()

    plain_clf, _ = predict(model, pre, post, tta=False)
    tta_clf, _ = predict(model, pre, post, tta=True)

    assert not torch.allclose(plain_clf, tta_clf, atol=1e-4)
    assert torch.allclose(tta_clf.sum(dim=1), torch.ones(2, SIZE, SIZE), atol=1e-6)


def test_predict_tta_averages_the_four_flip_views() -> None:
    model = AsymmetricModel()
    pre, post = _inputs()

    tta_clf, tta_loc = predict(model, pre, post, tta=True)

    expected_clf = torch.zeros_like(tta_clf)
    expected_loc = torch.zeros_like(tta_loc)
    with torch.no_grad():
        for dims in ([], [-1], [-2], [-2, -1]):
            flip = (lambda t, d=dims: torch.flip(t, dims=d)) if dims else (lambda t: t)
            logits = model(flip(pre), flip(post))
            expected_clf += flip(torch.softmax(logits["clf"], dim=1)) / 4
            expected_loc += flip(torch.softmax(logits["loc"], dim=1)) / 4
    assert torch.allclose(tta_clf, expected_clf, atol=1e-6)
    assert torch.allclose(tta_loc, expected_loc, atol=1e-6)


# --------------------------------------------------------------------------- evaluate


def test_evaluate_total_equals_sum_of_tiles() -> None:
    model = FakeModel()
    dataset = _dataset()

    acc = evaluate(model, dataset)

    assert isinstance(acc, ConfusionAccumulator)
    assert set(acc.per_tile) == {item["name"] for item in dataset}
    assert np.array_equal(acc.total, sum(acc.per_tile.values()))

    manual = np.zeros((NUM_CLF, NUM_CLF), dtype=np.int64)
    with torch.no_grad():
        for item in dataset:
            pre = torch.from_numpy(item["pre"]).unsqueeze(0)
            post = torch.from_numpy(item["post"]).unsqueeze(0)
            clf_prob, _ = predict(model, pre, post)
            pred = clf_prob.argmax(dim=1)[0].numpy()
            tile_cm = confusion_matrix(item["clf"], pred)
            assert np.array_equal(acc.per_tile[item["name"]], tile_cm)
            manual += tile_cm

    assert np.array_equal(acc.total, manual)
    assert acc.total.sum() == sum(int((np.asarray(item["clf"]) != 255).sum()) for item in dataset)


def test_evaluate_tta_uses_averaged_probabilities() -> None:
    model = AsymmetricModel()
    dataset = _dataset()

    acc = evaluate(model, dataset, tta=True)

    manual = np.zeros((NUM_CLF, NUM_CLF), dtype=np.int64)
    with torch.no_grad():
        for item in dataset:
            pre = torch.from_numpy(item["pre"]).unsqueeze(0)
            post = torch.from_numpy(item["post"]).unsqueeze(0)
            clf_prob, _ = predict(model, pre, post, tta=True)
            manual += confusion_matrix(item["clf"], clf_prob.argmax(dim=1)[0].numpy())

    assert np.array_equal(acc.total, manual)


def test_evaluate_writes_float16_cache(tmp_path) -> None:
    model = FakeModel()
    dataset = _dataset()

    evaluate(model, dataset, cache_dir=tmp_path)

    for item in dataset:
        path = tmp_path / f"{item['name']}.npz"
        assert path.exists()
        with np.load(path) as data:
            assert set(data.files) == {"clf", "loc"}
            clf = data["clf"]
            loc = data["loc"]
        assert clf.dtype == np.float16
        assert loc.dtype == np.float16
        assert clf.shape == (NUM_CLF, SIZE, SIZE)
        assert loc.shape == (NUM_LOC, SIZE, SIZE)

        pre = torch.from_numpy(item["pre"]).unsqueeze(0)
        post = torch.from_numpy(item["post"]).unsqueeze(0)
        with torch.no_grad():
            clf_prob, loc_prob = predict(model, pre, post)
        assert np.array_equal(clf.argmax(axis=0), clf_prob.argmax(dim=1)[0].numpy())
        assert np.array_equal(loc.argmax(axis=0), loc_prob.argmax(dim=1)[0].numpy())
        assert np.allclose(clf.astype(np.float32), clf_prob[0].numpy(), atol=1e-3)
        assert np.allclose(loc.astype(np.float32), loc_prob[0].numpy(), atol=1e-3)


def test_evaluate_leaves_model_in_eval_mode_without_grads() -> None:
    model = FakeModel()
    model.train()

    evaluate(model, _dataset())

    assert model.training is False
    for parameter in model.parameters():
        assert parameter.grad is None
