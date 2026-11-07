"""Exponential moving average of model weights for checkpoint selection.

Design section 7: EMA with decay 0.999; the EMA weights, not the raw ones, are evaluated and used
to pick the best checkpoint. The first update uses a step-dependent decay so the average starts
close to the live model instead of lagging behind it.
"""

from __future__ import annotations

import copy

import torch


class ModelEMA:
    """Keep a frozen, evaluation-mode copy of ``model`` updated as an exponential average."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.module = copy.deepcopy(model)
        self.module.eval()
        for param in self.module.parameters():
            param.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: torch.nn.Module, step: int) -> None:
        """Blend ``model`` into the average; ``step`` is the zero-based optimizer step."""
        decay = min(self.decay, (1.0 + step) / (10.0 + step))
        ema_params = dict(self.module.named_parameters())
        for name, param in model.named_parameters():
            ema_param = ema_params[name]
            if ema_param.dtype.is_floating_point:
                ema_param.mul_(decay).add_(param.detach(), alpha=1.0 - decay)
            else:
                ema_param.copy_(param.detach())
        ema_buffers = dict(self.module.named_buffers())
        for name, buffer in model.named_buffers():
            ema_buffers[name].copy_(buffer.detach())

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return the EMA weights and buffers for checkpointing."""
        return self.module.state_dict()

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Restore the EMA weights and buffers from ``state_dict``."""
        self.module.load_state_dict(state_dict)
