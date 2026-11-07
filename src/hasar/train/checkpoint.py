"""Atomic full-state checkpoints and RNG capture for resumable training.

Design section 7: a checkpoint holds the model, EMA, optimizer, scheduler, step and every RNG, so
a resumed run continues exactly where it stopped. Saving writes a temporary file and renames it
into place, so a Colab disconnect never leaves a half-written checkpoint behind.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

RNGState = dict[str, Any]


def capture_rng() -> RNGState:
    """Return the state of the Python, numpy and torch (CPU/CUDA) random generators."""
    state: RNGState = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: RNGState) -> None:
    """Restore the generators captured by :func:`capture_rng`."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path: Path | str, state: dict[str, Any]) -> None:
    """Write ``state`` to ``path`` atomically (temporary file, then rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        torch.save(state, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def load_checkpoint(path: Path | str) -> dict[str, Any]:
    """Load a checkpoint written by :func:`save_checkpoint`."""
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def prune_checkpoints(directory: Path | str, keep: int) -> None:
    """Delete all but the ``keep`` highest-step ``step_*.pt`` files in ``directory``."""
    directory = Path(directory)
    checkpoints = sorted(
        directory.glob("step_*.pt"),
        key=lambda path: int(path.stem.rsplit("_", 1)[-1]),
    )
    for path in checkpoints[: max(len(checkpoints) - keep, 0)]:
        path.unlink()
