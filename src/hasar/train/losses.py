"""Training losses for the damage-mapping model.

The objective is the official ChangeMamba BRIGHT loss (``tasks/bright.py``, ``train_step``)::

    CE_loc + CE_clf + 0.5 * Lovasz_loc + 0.75 * Lovasz_clf

with no class weights and ``ignore = 255``. Our only addition is that nodata pixels arrive as
255 from the data layer, so both the cross-entropy and the Lovász terms drop them.

The Lovász-softmax implementation is our own (Berman 2018, MIT) and mirrors the vendored
ChangeMamba one; it is not imported from ``third_party``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from hasar.config import Loss


def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Gradient of the Lovász extension w.r.t. the sorted errors (Alg. 1 in the paper)."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:  # cover the 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax_flat(probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Multi-class Lovász-softmax over the classes present in ``labels``.

    ``probs`` is ``[P, C]`` and ``labels`` is ``[P]``; ignored pixels must already be removed.
    """
    if probs.numel() == 0:
        # Only void pixels: keep the graph but contribute nothing.
        return probs.sum() * 0.0

    classes = probs.size(1)
    losses = []
    for c in range(classes):
        fg = (labels == c).float()  # foreground for class c
        if fg.sum() == 0:
            continue
        class_pred = probs[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))

    if not losses:
        return probs.sum() * 0.0
    return torch.stack(losses).mean()


def lovasz_softmax(probs: torch.Tensor, labels: torch.Tensor, ignore: int = 255) -> torch.Tensor:
    """Multi-class Lovász-softmax for ``probs`` ``[B, C, H, W]`` and ``labels`` ``[B, H, W]``."""
    if probs.dim() == 3:
        # Sigmoid output: [B, H, W] -> [B, 1, H, W].
        probs = probs.unsqueeze(1)

    batch, classes, height, width = probs.shape
    flat = probs.permute(0, 2, 3, 1).contiguous().view(-1, classes)
    flat_labels = labels.view(-1)

    valid = flat_labels != ignore
    flat = flat[valid]
    flat_labels = flat_labels[valid]

    if flat.numel() == 0:
        # Only void pixels: keep the graph, no NaN.
        return probs.sum() * 0.0
    return lovasz_softmax_flat(flat, flat_labels)


def damage_loss(
    outputs: dict[str, torch.Tensor],
    loc_target: torch.Tensor,
    clf_target: torch.Tensor,
    cfg: Loss | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return the weighted total loss and its float parts.

    ``outputs`` holds the ``loc`` (2-class) and ``clf`` (4-class) logits. Pixels labelled with
    ``cfg.ignore_index`` are dropped from every term. When a target has no valid pixel at all,
    cross-entropy would be NaN, so that term becomes ``logits.sum() * 0.0`` instead.
    """
    if cfg is None:
        cfg = Loss()

    ignore = cfg.ignore_index
    loc_logits = outputs["loc"]
    clf_logits = outputs["clf"]

    # Cross-entropy on float32 logits for a stable value (also under bf16 autocast).
    if (loc_target != ignore).any():
        ce_loc = F.cross_entropy(loc_logits.float(), loc_target, ignore_index=ignore)
    else:
        ce_loc = loc_logits.sum() * 0.0

    if (clf_target != ignore).any():
        weight = torch.tensor(cfg.class_weights, dtype=torch.float32, device=clf_logits.device)
        ce_clf = F.cross_entropy(clf_logits.float(), clf_target, weight=weight, ignore_index=ignore)
    else:
        ce_clf = clf_logits.sum() * 0.0

    def _lovasz(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if cfg.lovasz_fp32:
            with torch.autocast(device_type=logits.device.type, enabled=False):
                return lovasz_softmax(torch.softmax(logits.float(), dim=1), target, ignore=ignore)
        return lovasz_softmax(torch.softmax(logits.float(), dim=1), target, ignore=ignore)

    lovasz_loc = _lovasz(loc_logits, loc_target)
    lovasz_clf = _lovasz(clf_logits, clf_target)

    total = cfg.ce_loc * ce_loc + cfg.ce_clf * ce_clf + cfg.lovasz_loc * lovasz_loc + cfg.lovasz_clf * lovasz_clf

    parts = {
        "ce_loc": float(ce_loc.detach()),
        "ce_clf": float(ce_clf.detach()),
        "lovasz_loc": float(lovasz_loc.detach()),
        "lovasz_clf": float(lovasz_clf.detach()),
        "total": float(total.detach()),
    }
    return total, parts
