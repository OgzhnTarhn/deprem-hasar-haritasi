"""Tile-by-tile inference and evaluation loop (protocol section 3).

One inference, many analyses
----------------------------
The model is run over the Turkey tiles exactly once. Every secondary row of the results table
(flip TTA, localisation-masked prediction, component voting, bootstrap intervals, the demo) reads
the stored probabilities instead of running the model again. The cache therefore holds the full
``clf`` and ``loc`` probability maps as ``float16``: half precision halves the disk footprint and
is far below the rounding that could change an ``argmax``, so every downstream analysis sees the
same prediction as the headline path.

The headline path is ``tta=False`` with ``pred = argmax(clf)``; flip TTA is the pre-declared
secondary row and averages the probabilities of the image and its horizontal mirror.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

import numpy as np
import torch

from hasar.eval.metrics import ConfusionAccumulator

Progress = Callable[[int, int, str], None]


def predict(
    model: torch.nn.Module,
    pre: torch.Tensor,
    post: torch.Tensor,
    tta: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the ``clf`` and ``loc`` probabilities of one batch of pre/post tiles.

    ``pre`` and ``post`` are ``B x C x H x W`` tensors on the model's device. With ``tta=True``
    the probabilities of the image and its horizontal mirror are averaged; the mirror is flipped
    back before averaging, so the output keeps the input orientation.
    """
    with torch.inference_mode():
        logits = model(pre, post)
        clf = torch.softmax(logits["clf"], dim=1)
        loc = torch.softmax(logits["loc"], dim=1)
        if tta:
            flipped = model(torch.flip(pre, dims=[-1]), torch.flip(post, dims=[-1]))
            clf = (clf + torch.flip(torch.softmax(flipped["clf"], dim=1), dims=[-1])) / 2
            loc = (loc + torch.flip(torch.softmax(flipped["loc"], dim=1), dims=[-1])) / 2
    return clf, loc


def _save_cache(path: Path, clf: np.ndarray, loc: np.ndarray) -> None:
    """Write ``clf`` and ``loc`` as float16 to ``path`` atomically.

    The arrays go to a ``.tmp`` file first and are moved into place with :func:`os.replace`, so a
    crash never leaves a half-written cache file behind.
    """
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        np.savez(handle, clf=clf.astype(np.float16), loc=loc.astype(np.float16))
    os.replace(tmp, path)


def evaluate(
    model: torch.nn.Module,
    dataset: Iterable[Mapping[str, object]],
    device: str = "cpu",
    tta: bool = False,
    cache_dir: str | Path | None = None,
    progress: Progress | None = None,
) -> ConfusionAccumulator:
    """Run ``model`` over ``dataset`` and accumulate the per-tile confusion matrices.

    Each item is a ``BrightDataset`` sample with ``name``, ``pre``, ``post`` and the ``clf`` label.
    The prediction is ``argmax`` of the ``clf`` probabilities (averaged over the flip when
    ``tta=True``). When ``cache_dir`` is given, every tile's probabilities are stored as
    ``<name>.npz`` so later analyses never rerun the model. ``progress`` is called as
    ``progress(i, n, name)`` before each tile (``n`` is ``-1`` when the dataset has no length).
    """
    model.to(device)
    model.eval()

    # Tiles are read one at a time: a 1024 px tile is ~25 MB in float32, so materialising the
    # whole Turkey split (1,114 tiles) would need tens of gigabytes.
    total = len(dataset) if hasattr(dataset, "__len__") else -1
    cache_path = Path(cache_dir) if cache_dir is not None else None
    if cache_path is not None:
        cache_path.mkdir(parents=True, exist_ok=True)

    accumulator = ConfusionAccumulator()
    for i, item in enumerate(dataset):
        name = str(item["name"])
        if progress is not None:
            progress(i, total, name)

        pre = torch.from_numpy(np.asarray(item["pre"])).unsqueeze(0).to(device)
        post = torch.from_numpy(np.asarray(item["post"])).unsqueeze(0).to(device)
        clf_prob, loc_prob = predict(model, pre, post, tta=tta)

        clf = clf_prob[0].float().cpu().numpy()
        loc = loc_prob[0].float().cpu().numpy()
        pred = clf.argmax(axis=0)
        accumulator.add(np.asarray(item["clf"]), pred, name)

        if cache_path is not None:
            _save_cache(cache_path / f"{name}.npz", clf, loc)

    return accumulator
