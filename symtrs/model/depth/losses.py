"""Depth losses.

Two families, matching the two depth protocols:

*Relative* models are trained with `silog`, the scale-invariant log loss from MiDaS.
It is invariant to the global scale of the prediction, which is exactly the freedom a
relative model has.

*Metric* models are trained with `berhu`, the reverse-Huber loss ZoeDepth uses for its
metric head. It behaves like L1 for small errors and L2 for large ones, which stops
the handful of very distant pixels in an aerial scene from dominating.

Both mask invalid pixels. Both clamp to positive values before any log, because a
logged negative depth is NaN and a NaN in the loss silently poisons every weight it
touches.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

#: Losses available to `train.loss`.
DEPTH_LOSSES = ("silog", "berhu", "l1", "log_l1")

EPS = 1e-7


def _masked_reduce(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over the masked elements only."""
    mask = mask.bool()
    selected = values[mask]
    if selected.numel() == 0:
        # Returning a zero that still carries the graph keeps DDP from deadlocking on
        # a rank that saw no valid pixels.
        return values.sum() * 0.0
    return selected.mean()


def silog_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    variance_focus: float = 0.5,
    eps: float = EPS,
) -> torch.Tensor:
    """Scale-invariant log loss.

    `sqrt(mean(g^2) - variance_focus * mean(g)^2)` where `g = log(pred) - log(target)`.
    `variance_focus=0.5` is the value used by ZoeDepth; 0.85 is the "scale-invariant"
    variant in the MiDaS paper. Only the *variance* of `g` drives the loss when the
    focus term is large, so a global scale error is free.
    """
    if mask is None:
        mask = torch.isfinite(target) & (target > 0)

    prediction = prediction.clamp_min(eps)
    target = target.clamp_min(eps)
    difference = torch.log(prediction) - torch.log(target)

    flat = difference.reshape(difference.shape[0], -1)
    flat_mask = mask.reshape(mask.shape[0], -1).bool()

    # Per-image then averaged: a single image with half the valid pixels should not
    # count the same as a fully valid one.
    per_image = []
    for index in range(flat.shape[0]):
        values = flat[index][flat_mask[index]]
        if values.numel() < 2:
            per_image.append(flat[index].sum() * 0.0)
            continue
        variance = (values ** 2).mean() - variance_focus * (values.mean() ** 2)
        per_image.append(torch.sqrt(variance.clamp_min(0.0) + EPS))
    return torch.stack(per_image).mean()


def berhu_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    threshold: float = 0.2,
    eps: float = EPS,
) -> torch.Tensor:
    """Reverse Huber: L1 below `threshold * max_error`, L2 above it.

    The threshold is rescaled per batch against the largest error present, which is
    what makes the transition point adaptive rather than a fixed metric distance.
    """
    if mask is None:
        mask = torch.isfinite(target) & (target > 0)

    difference = (prediction - target).abs()
    if mask.any():
        cutoff = threshold * difference[mask.bool()].max().detach()
    else:
        cutoff = torch.tensor(threshold, device=prediction.device)
    cutoff = cutoff.clamp_min(eps)

    quadratic = (difference ** 2 + cutoff ** 2) / (2.0 * cutoff)
    loss = torch.where(difference <= cutoff, difference, quadratic)
    return _masked_reduce(loss, mask)


def l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if mask is None:
        mask = torch.isfinite(target) & (target > 0)
    return _masked_reduce((prediction - target).abs(), mask)


def log_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = EPS,
) -> torch.Tensor:
    """L1 in log space: penalises relative rather than absolute error, but is not
    scale-invariant like `silog`."""
    if mask is None:
        mask = torch.isfinite(target) & (target > 0)
    difference = (torch.log(prediction.clamp_min(eps)) - torch.log(target.clamp_min(eps))).abs()
    return _masked_reduce(difference, mask)


LOSS_FUNCTIONS: Dict[str, callable] = {
    "silog": silog_loss,
    "berhu": berhu_loss,
    "l1": l1_loss,
    "log_l1": log_l1_loss,
}


def build_depth_loss(name: str, **kwargs):
    """Look up a loss by name, validating that the kwargs belong to it."""
    if not name:
        raise ValueError("a depth loss name is required")
    if name not in LOSS_FUNCTIONS:
        raise ValueError(f"unknown depth loss {name!r}; expected one of {DEPTH_LOSSES}")

    function = LOSS_FUNCTIONS[name]
    if not kwargs:
        return function

    allowed = set(function.__code__.co_varnames[: function.__code__.co_argcount])
    unknown = set(kwargs) - allowed
    if unknown:
        raise ValueError(
            f"unknown option(s) {sorted(unknown)} for depth loss {name!r}; "
            f"it accepts {sorted(allowed)}"
        )
    return lambda prediction, target, mask=None: function(
        prediction, target, mask, **kwargs
    )


def default_loss_for_protocol(protocol: str) -> str:
    """The loss a depth protocol uses when the config does not name one."""
    return "silog" if protocol == "relative" else "berhu"


def gradient_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """First-order edge consistency, optionally added to a depth loss.

    Not in `DEPTH_LOSSES` because it is a regulariser rather than a standalone
    objective; enable it with `train.loss_kwargs.gradient_weight`.
    """
    mask = mask.bool()
    prediction_dx = prediction[..., :, 1:] - prediction[..., :, :-1]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    prediction_dy = prediction[..., 1:, :] - prediction[..., :-1, :]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    mask_x = mask[..., :, 1:] & mask[..., :, :-1]
    mask_y = mask[..., 1:, :] & mask[..., :-1, :]

    loss = _masked_reduce((prediction_dx - target_dx).abs(), mask_x)
    loss = loss + _masked_reduce((prediction_dy - target_dy).abs(), mask_y)
    return loss * 0.5


def smoothness_loss(prediction: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Total-variation style smoothness on the prediction. Self-supervised fallback
    for unlabelled data; off by default."""
    prediction_dx = (prediction[..., :, 1:] - prediction[..., :, :-1]).abs()
    prediction_dy = (prediction[..., 1:, :] - prediction[..., :-1, :]).abs()
    if mask is not None:
        mask = mask.bool()
        prediction_dx = prediction_dx * mask[..., :, 1:] * mask[..., :, :-1]
        prediction_dy = prediction_dy * mask[..., 1:, :] * mask[..., :-1, :]
    return (prediction_dx.mean() + prediction_dy.mean()) * 0.5
