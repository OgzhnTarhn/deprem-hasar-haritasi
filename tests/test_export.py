"""Tests for the deployment export helpers (``hasar.export``).

Everything runs on the CPU with tiny 16x16 tensors and fake models defined here; no real BRIGHT
data, no checkpoint and no network. The fake model follows the model contract: ``forward(pre,
post)`` returns ``{"clf": ..., "loc": ...}`` logits.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from hasar.export import _ExportWrapper, latency_ms, try_export  # noqa: E402

SIZE = 16
NUM_CLF = 4
NUM_LOC = 2


class TinyModel(torch.nn.Module):
    """1x1 convolution on the concatenated pre/post stack, as in ``tests/test_eval_predict.py``."""

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


class RaisingModel(torch.nn.Module):
    """A model whose forward always raises, so every export attempt must fail."""

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> dict[str, torch.Tensor]:
        raise RuntimeError("kasten bozuk model")


def _inputs(seed: int = 1, batch: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    pre = torch.randn(batch, 3, SIZE, SIZE, generator=generator)
    post = torch.randn(batch, 3, SIZE, SIZE, generator=generator)
    return pre, post


# --------------------------------------------------------------------------- wrapper


def test_export_wrapper_returns_loc_clf_tuple() -> None:
    model = TinyModel()
    pre, post = _inputs()

    with torch.no_grad():
        output = _ExportWrapper(model)(pre, post)

    assert isinstance(output, tuple)
    assert len(output) == 2
    loc, clf = output
    with torch.no_grad():
        expected = model(pre, post)
    assert torch.allclose(loc, expected["loc"])
    assert torch.allclose(clf, expected["clf"])


# --------------------------------------------------------------------------- try_export


def test_torch_export_succeeds_and_writes_file(tmp_path) -> None:
    model = TinyModel()

    result = try_export(model, tmp_path, size=SIZE, formats=("torch-export",))

    assert set(result) == {"torch-export"}
    entry = result["torch-export"]
    assert entry["ok"] is True
    assert entry["error"] is None
    assert isinstance(entry["seconds"], float)
    assert entry["seconds"] >= 0.0
    assert entry["path"] is not None
    path = tmp_path / "model.pt2"
    assert path.exists()
    assert path.stat().st_size > 0
    assert entry["path"] == str(path)


def test_export_failure_is_recorded_not_raised(tmp_path) -> None:
    model = RaisingModel()

    result = try_export(model, tmp_path, size=SIZE, formats=("torch-export",))

    entry = result["torch-export"]
    assert entry["ok"] is False
    assert isinstance(entry["error"], str)
    assert entry["error"]
    assert entry["path"] is None
    assert isinstance(entry["seconds"], float)


def test_export_result_is_json_serialisable(tmp_path) -> None:
    model = TinyModel()

    result = try_export(model, tmp_path, size=SIZE, formats=("torch-export", "onnx"))

    assert set(result) == {"torch-export", "onnx"}
    for entry in result.values():
        assert set(entry) == {"ok", "error", "seconds", "path"}
        assert isinstance(entry["ok"], bool)
        assert entry["error"] is None or isinstance(entry["error"], str)
        assert isinstance(entry["seconds"], float)
        assert entry["path"] is None or isinstance(entry["path"], str)
    # Must survive a round trip through JSON, as the report writer stores it.
    assert json.loads(json.dumps(result)) == result


def test_onnx_export_reports_missing_package_or_writes_file(tmp_path) -> None:
    model = TinyModel()

    result = try_export(model, tmp_path, size=SIZE, formats=("onnx",))

    entry = result["onnx"]
    if entry["ok"]:
        path = tmp_path / "model.onnx"
        assert path.exists()
        assert path.stat().st_size > 0
        assert entry["path"] == str(path)
        assert entry["error"] is None
    else:
        # Without the ``onnx`` package the export must fail softly and name the cause.
        assert isinstance(entry["error"], str)
        assert "onnx" in entry["error"].lower()
        assert entry["path"] is None


# --------------------------------------------------------------------------- latency


def test_latency_ms_returns_median_forward_time() -> None:
    model = TinyModel()

    value = latency_ms(model, size=SIZE, runs=3)

    assert isinstance(value, float)
    assert value > 0.0
    assert value == pytest.approx(value)  # finite, not NaN
