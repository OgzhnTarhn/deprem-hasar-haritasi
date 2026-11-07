"""Learning-rate schedule: linear warmup then cosine decay to ``min_lr``.

Design section 7: 1500 warmup steps, cosine decay, minimum learning rate 1e-6. The closed-form
:func:`lr_at` is the single source of truth; :func:`make_scheduler` wraps it in a ``LambdaLR`` so
the optimizer follows exactly the same curve.
"""

from __future__ import annotations

import math

import torch

from hasar.config import Sched


def lr_at(step: int, base_lr: float, warmup_steps: int, total_steps: int, min_lr: float) -> float:
    """Return the learning rate at optimizer ``step``.

    Warmup rises linearly from ``base_lr / warmup_steps`` at step 0 to ``base_lr`` at the last
    warmup step; the cosine then decays from ``base_lr`` at ``warmup_steps`` to ``min_lr`` at
    ``total_steps`` and stays there afterwards.
    """
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps

    if total_steps <= warmup_steps:
        return min_lr

    progress = min(max(step - warmup_steps, 0), total_steps - warmup_steps)
    t = progress / (total_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * t))


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    sched_cfg: Sched,
    base_lr: float,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Return a ``LambdaLR`` that scales ``base_lr`` by :func:`lr_at` at every step."""

    def scale(step: int) -> float:
        return lr_at(step, base_lr, sched_cfg.warmup_steps, total_steps, sched_cfg.min_lr) / base_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scale)
