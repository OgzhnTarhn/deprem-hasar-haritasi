"""Deployment export helpers: torch-export, ONNX and a latency probe.

The model contract returns a ``{"clf": ..., "loc": ...}`` dictionary, which neither the
``torch.export`` graph nor the ONNX exporter handles as cleanly as a plain tuple. :class:`_ExportWrapper`
therefore turns the dictionary into a ``(loc, clf)`` tuple, and every exporter runs on the wrapper.

:func:`try_export` never raises for a failed format: it records ``ok=False`` and the error text in the
per-format entry, so one broken backend (for example a missing ``onnx`` package) does not stop the
others. The result is plain JSON-serialisable data, ready for the research report writer.

This module is the Faz 4 ONNX trial and feeds the Faz 9 export decision; it is not part of the
training or evaluation path.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path

import torch
import torch.nn as nn

#: Formats :func:`try_export` knows how to write, mapped to the file name it produces.
FORMAT_FILES: dict[str, str] = {
    "torch-export": "model.pt2",
    "onnx": "model.onnx",
}


class _ExportWrapper(nn.Module):
    """Adapt the ``{"clf", "loc"}`` model contract to a ``(loc, clf)`` tuple for the exporters."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(loc, clf)`` logits for a batch of pre/post tiles."""
        output = self.model(pre, post)
        return output["loc"], output["clf"]


def _example_inputs(size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a deterministic ``1 x 3 x size x size`` pre/post pair on ``device``."""
    generator = torch.Generator().manual_seed(0)
    pre = torch.randn(1, 3, size, size, generator=generator).to(device)
    post = torch.randn(1, 3, size, size, generator=generator).to(device)
    return pre, post


def _export_torch_export(wrapper: nn.Module, path: Path, pre: torch.Tensor, post: torch.Tensor) -> None:
    """Export ``wrapper`` to an ``ExportedProgram`` and save it as ``path`` (``.pt2``).

    The program is written through an open file handle: ``torch.export.save`` mangles non-ASCII
    characters in a ``str`` path (a Windows user name such as ``Oğuzhan`` breaks it), while a handle
    is written as-is.
    """
    program = torch.export.export(wrapper, (pre, post))
    with open(path, "wb") as handle:
        torch.export.save(program, handle)


def _export_onnx(wrapper: nn.Module, path: Path, pre: torch.Tensor, post: torch.Tensor) -> None:
    """Export ``wrapper`` to ONNX at ``path``; raises when the ``onnx`` package is missing."""
    try:
        import onnx  # noqa: F401
    except ImportError as exc:
        raise ImportError("onnx paketi kurulu değil; ONNX dışa aktarımı yapılamaz") from exc
    with open(path, "wb") as handle:
        torch.onnx.export(
            wrapper,
            (pre, post),
            handle,
            input_names=["pre", "post"],
            output_names=["loc", "clf"],
            opset_version=17,
            dynamo=False,
        )


_EXPORTERS = {
    "torch-export": _export_torch_export,
    "onnx": _export_onnx,
}


def try_export(
    model: nn.Module,
    out_dir: Path | str,
    size: int = 256,
    formats: Iterable[str] = ("torch-export", "onnx"),
) -> dict[str, dict[str, object]]:
    """Try every format in ``formats`` and return one result entry per format.

    Each entry has ``ok`` (bool), ``error`` (``None`` or the error text), ``seconds`` (float) and
    ``path`` (``None`` or the written file). A failing format is recorded, never raised, so the
    caller can report all backends in one pass.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = next(model.parameters()).device if any(True for _ in model.parameters()) else torch.device("cpu")
    wrapper = _ExportWrapper(model).to(device).eval()
    pre, post = _example_inputs(size, device)

    results: dict[str, dict[str, object]] = {}
    for name in formats:
        path = out_dir / FORMAT_FILES[name]
        started = time.perf_counter()
        try:
            with torch.no_grad():
                _EXPORTERS[name](wrapper, path, pre, post)
        except Exception as exc:  # noqa: BLE001 - every backend failure is data, not a crash
            results[name] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "seconds": float(time.perf_counter() - started),
                "path": None,
            }
        else:
            results[name] = {
                "ok": True,
                "error": None,
                "seconds": float(time.perf_counter() - started),
                "path": str(path),
            }
    return results


def latency_ms(model: nn.Module, size: int = 256, runs: int = 10, warmup: int = 2) -> float:
    """Return the median forward time of ``model`` in milliseconds on its own device.

    ``warmup`` untimed forwards run first so lazy initialisation does not pollute the measurement;
    the median of ``runs`` timed forwards is returned.
    """
    device = next(model.parameters()).device if any(True for _ in model.parameters()) else torch.device("cpu")
    model.eval()
    pre, post = _example_inputs(size, device)

    with torch.inference_mode():
        for _ in range(max(warmup, 0)):
            model(pre, post)
        timings = []
        for _ in range(max(runs, 1)):
            started = time.perf_counter()
            model(pre, post)
            timings.append((time.perf_counter() - started) * 1000.0)

    timings.sort()
    middle = len(timings) // 2
    if len(timings) % 2:
        return float(timings[middle])
    return float((timings[middle - 1] + timings[middle]) / 2.0)
