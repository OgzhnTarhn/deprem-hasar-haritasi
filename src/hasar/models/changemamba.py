"""Main ChangeMamba model and the DamageFormer fallback, both on the project contract.

The main model is the official ChangeMambaMMBDA (Chen et al., "ChangeMamba: Remote Sensing Change
Detection with Spatio-Temporal State Space Models") built with the official BRIGHT configuration
(``get_vssm_kwargs`` + ``vssm_tiny_224_0229flex.yaml``, see ``third_party/changemamba/SOURCE.md``):
a VSSM-Tiny backbone (``depths=[2, 2, 4, 2]``, ``dims=96``, ``ssm_d_state=1``, ``forward_type="v3noz"``)
with a semantic building decoder and a change damage decoder, ``output_building=2`` and
``output_damage=4``.

SAR replication rule (ADR-0004): the data layer produces ``post`` as ``B x C x H x W`` with ``C = 3``
in the official mode (single band replicated, ImageNet statistics) or ``C = 1`` in the ``train_stats``
mode. The official SAR encoder expects three channels, so this wrapper replicates a single band to
three itself; any other channel count is an error.

Scan backend (``PATCHES.md`` patch 3): the vendored ``vmamba.py`` uses the compiled
``selective_scan_cuda_oflex`` kernel when it is importable and the input is on CUDA, and otherwise
falls back to the pure-PyTorch ``selective_scan_torch`` (same mathematics, much slower). Setting
``HASAR_SCAN_BACKEND=torch`` forces the PyTorch path even on a GPU, which is how the kernel
equivalence gate is measured. :func:`scan_backend` reports which path is active.

Checkpoints (project rule 2): the Zenodo / ``standard_ML`` ChangeMamba checkpoints saw Turkey during
training and are never loaded here. The only weights this module accepts are the ImageNet
``vssm_tiny_0230_ckpt_epoch_262.pth`` encoder initialisation (``pretrained=``) and, for the
DamageFormer fallback, the torchvision ImageNet Swin-T weights.

torch is imported at module top: this module is only imported where torch exists.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import ModuleType

import torch
import torch.nn as nn

from hasar.models import bright_baselines
from hasar.models.registry import register

#: Root of the vendored ChangeMamba source tree (kept on ``sys.path`` for its absolute imports).
VENDOR = Path(__file__).resolve().parents[3] / "third_party" / "changemamba"

#: Official BRIGHT VSSM-Tiny configuration (``get_vssm_kwargs`` + ``vssm_tiny_224_0229flex.yaml``).
OFFICIAL_VSSM_KWARGS: dict[str, object] = {
    "patch_size": 4,
    "in_chans": 3,
    "num_classes": 1000,
    "depths": [2, 2, 4, 2],
    "dims": 96,
    "ssm_d_state": 1,
    "ssm_ratio": 2.0,
    "ssm_rank_ratio": 2.0,
    "ssm_dt_rank": "auto",
    "ssm_act_layer": "silu",
    "ssm_conv": 3,
    "ssm_conv_bias": False,
    "ssm_drop_rate": 0.0,
    "ssm_init": "v0",
    "forward_type": "v3noz",
    "mlp_ratio": 4.0,
    "mlp_act_layer": "gelu",
    "mlp_drop_rate": 0.0,
    "drop_path_rate": 0.2,
    "patch_norm": True,
    "norm_layer": "ln",
    "downsample_version": "v3",
    "patchembed_version": "v2",
    "gmlp": False,
    "use_checkpoint": False,
}


def _import_vendor() -> tuple[ModuleType, ModuleType]:
    """Put the vendored tree on ``sys.path`` and return the model and ``vmamba`` modules.

    The vendored files use absolute ``changedetection.`` imports, so the folder must stay on
    ``sys.path``. Imported lazily at build time so importing this module stays cheap.
    """
    vendor = str(VENDOR)
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    model = importlib.import_module("changedetection.models.ChangeMambaMMBDA")
    vmamba = importlib.import_module("changedetection.models.vmamba")
    return model, vmamba


def selective_scan_torch(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    delta_bias: torch.Tensor | None = None,
    delta_softplus: bool = False,
    oflex: bool = True,
) -> torch.Tensor:
    """Pure-PyTorch selective scan (thin wrapper around the vendored implementation)."""
    _, vmamba = _import_vendor()
    return vmamba.selective_scan_torch(u, delta, A, B, C, D, delta_bias, delta_softplus, oflex)


def scan_backend() -> str:
    """Return ``"cuda"`` when the compiled selective-scan kernel is used, else ``"torch"``."""
    if os.environ.get("HASAR_SCAN_BACKEND", "").lower() == "torch":
        return "torch"
    if not torch.cuda.is_available():
        return "torch"
    _, vmamba = _import_vendor()
    return "cuda" if hasattr(vmamba, "selective_scan_cuda_oflex") else "torch"


class ChangeMambaWrapper(nn.Module):
    """Adapt the official ChangeMambaMMBDA to the project's ``{"loc", "clf"}`` contract."""

    def __init__(self, pretrained: str | None = None, **overrides: object) -> None:
        super().__init__()
        model, _ = _import_vendor()
        kwargs = {**OFFICIAL_VSSM_KWARGS, **overrides}
        self.net = model.ChangeMambaMMBDA(
            output_building=2,
            output_damage=4,
            pretrained=pretrained,
            **kwargs,
        )

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return ``{"loc": ..., "clf": ...}`` logits for a batch of pre/post tiles."""
        if post.shape[1] == 1:
            post = post.expand(-1, 3, -1, -1)
        elif post.shape[1] != 3:
            raise ValueError(f"SAR girdisi 1 veya 3 kanallı olmalı, gelen: {post.shape[1]}")
        building, damage = self.net(pre, post)
        return {"loc": building, "clf": damage}


@register("changemamba_mmbda_tiny")
def _build_changemamba(pretrained: str | None = None) -> ChangeMambaWrapper:
    return ChangeMambaWrapper(pretrained=pretrained)


@register("damageformer_swin_t")
def _build_damageformer_swin_t(pretrained: bool = False) -> bright_baselines.BrightBaseline:
    """DamageFormer (Swin-T) fallback, optionally initialised with ImageNet Swin-T weights."""
    net = bright_baselines._vendor("DamageFormer").DamageFormer()
    if pretrained:
        from torchvision.models import Swin_T_Weights, swin_t

        state = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1).state_dict()
        net.encoder_1.swin_transformer.load_state_dict(state)
        net.encoder_2.swin_transformer.load_state_dict(state)
    return bright_baselines.BrightBaseline(net, two_stream=True)
