"""Model contract shared by every damage-mapping model.

Every model in this project (the main ChangeMamba, the fallback DamageFormer, the single-idea
variant and the Zenodo baselines) implements this contract: a ``forward(pre, post)`` call that
returns a mapping with the ``loc`` and ``clf`` logits. ``post`` is the SAR tensor as produced by the
data layer, ``B x C x H x W`` with ``C = 3`` in the official mode (band replicated, ImageNet
statistics) or ``C = 1`` in the ``train_stats`` mode; a wrapper whose network needs three channels
replicates a single band itself (ADR-0004). Outputs are logits, not probabilities, so losses and
metrics apply their own activation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import torch

OUTPUT_KEYS = ("loc", "clf")
NUM_LOC_CLASSES = 2
NUM_CLF_CLASSES = 4


@runtime_checkable
class DamageModel(Protocol):
    """Structural contract for a damage-mapping model."""

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> Mapping[str, torch.Tensor]:
        """Return ``{"loc": ..., "clf": ...}`` logits for a batch of pre/post tiles."""
        ...


def validate_output(
    output: Mapping[str, torch.Tensor],
    batch: int,
    height: int,
    width: int,
) -> None:
    """Check that ``output`` holds ``loc`` and ``clf`` logits of the expected shape.

    ``loc`` must be ``batch x NUM_LOC_CLASSES x height x width`` and ``clf`` must be
    ``batch x NUM_CLF_CLASSES x height x width``. Raises :class:`ValueError` naming the offending
    key on any mismatch.
    """
    keys = set(output)
    missing = [key for key in OUTPUT_KEYS if key not in keys]
    if missing:
        raise ValueError(f"Model çıktısında eksik anahtar: {', '.join(missing)}")
    extra = sorted(keys - set(OUTPUT_KEYS))
    if extra:
        raise ValueError(f"Model çıktısında beklenmeyen anahtar: {', '.join(extra)}")

    expected = {
        "loc": (batch, NUM_LOC_CLASSES, height, width),
        "clf": (batch, NUM_CLF_CLASSES, height, width),
    }
    for key in OUTPUT_KEYS:
        shape = tuple(output[key].shape)
        if shape != expected[key]:
            raise ValueError(f"Model çıktısı '{key}' şekli {shape}, beklenen {expected[key]}")
