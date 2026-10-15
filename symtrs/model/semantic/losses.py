"""Semantic segmentation losses.

SyMTRS land-cover scenes are dominated by a handful of large classes (vegetation,
ground) while the classes the benchmark actually cares about (buildings, roads,
vehicles) are a small fraction of the pixels. Every loss here therefore has to be
usable as a standalone objective on that distribution, which is why the cross-entropy
family carries `class_weights`, `focal_loss` exists at all, and `dice_loss` averages
only over classes that are actually present.

Two rules hold for every function in this module:

*Ignore-index safety.* A prediction is never asked to be right about a pixel whose
label is `ignore_index`. The target is sanitised once - anything outside
`[0, num_classes)` or equal to `ignore_index` becomes `ignore_index` - and the loss is
then averaged over the remaining pixels only. Sanitising first, rather than passing
raw labels to `F.cross_entropy`, also means an out-of-range label (a 255 in a 6-class
run) is ignored instead of raising deep inside the reduction.

*Zero, not NaN, on an empty batch.* A batch whose every pixel is ignored has no valid
pixels to average over, and a plain `mean` of an empty selection is `NaN`. A `NaN`
loss poisons every weight it touches in the backward pass, and it is invisible until
training silently diverges, so an empty batch returns `logits.sum() * 0.0` instead -
zero, but still carrying the autograd graph, so distributed ranks stay in lockstep.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import torch
import torch.nn.functional as F

#: Losses available to `train.loss` for the segmentation task.
#:
#: `ce_aux` is cross-entropy used with an auxiliary head: it is the name DeepLabV3
#: declares as its default. It resolves to the same function as `ce`; what makes it
#: different is that the builder was asked for an auxiliary term (see `aux_weight`).
SEGMENTATION_LOSSES: tuple = ("ce", "ce_aux", "dice", "ce_dice", "focal")

#: Default weight of the auxiliary-head term when `aux_weight` is not given.
DEFAULT_AUX_WEIGHT = 0.4


# --------------------------------------------------------------------------- helpers


def _sanitise_target(
    mask: torch.Tensor,
    num_classes: int,
    ignore_index: int,
) -> torch.Tensor:
    """Map anything that is not a usable label onto `ignore_index`.

    Doing this in one place keeps the ignore convention consistent: after this call
    the only special value in the target is `ignore_index`, and every loss below can
    rely on that instead of re-deriving validity.
    """
    target = mask.long()
    valid = (target >= 0) & (target < num_classes) & (target != ignore_index)
    return torch.where(valid, target, torch.full_like(target, int(ignore_index)))


def _valid_pixels(
    mask: torch.Tensor,
    num_classes: int,
    ignore_index: int,
) -> torch.Tensor:
    """Boolean mask of the pixels that carry a real label."""
    target = mask.long()
    return (target >= 0) & (target < num_classes) & (target != ignore_index)


def _mean_over_valid(
    per_pixel: torch.Tensor,
    valid: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Mean of `per_pixel` over `valid`, or a graph-carrying zero if none is valid."""
    selected = per_pixel[valid.bool()]
    if selected.numel() == 0:
        # See the module docstring: an empty selection would be 0/0 = NaN, and a NaN in
        # the loss silently poisons every weight it touches on the backward pass.
        return reference.sum() * 0.0
    return selected.mean()


def _class_weight_tensor(
    class_weights: Optional[Sequence[float] | torch.Tensor],
    logits: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Bring per-class weights onto the logits' device and dtype."""
    if class_weights is None:
        return None
    if isinstance(class_weights, torch.Tensor):
        weight = class_weights.to(device=logits.device, dtype=logits.dtype)
    else:
        weight = torch.tensor(list(class_weights), device=logits.device, dtype=logits.dtype)
    if weight.numel() != logits.shape[1]:
        raise ValueError(
            f"class_weights has {weight.numel()} entries but the logits have "
            f"{logits.shape[1]} classes"
        )
    return weight


def _check_spatial(logits: torch.Tensor, mask: torch.Tensor) -> None:
    """Fail loudly on a resolution mismatch instead of inside a reduction."""
    if tuple(logits.shape[-2:]) != tuple(mask.shape[-2:]):
        raise ValueError(
            f"logits are {tuple(logits.shape[-2:])} but the mask is {tuple(mask.shape[-2:])}; "
            "run the logits through forward_at_size() before computing the loss"
        )


# --------------------------------------------------------------------------- losses


def cross_entropy_loss(
    logits: torch.Tensor,
    mask: torch.Tensor,
    class_weights: Optional[Sequence[float] | torch.Tensor] = None,
    ignore_index: int = 255,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Per-pixel cross-entropy, averaged over labelled pixels only.

    `class_weights` rescales each pixel's term by its true class, which is the blunt
    instrument for the imbalance problem; prefer `focal_loss` when the imbalance is
    extreme, since it down-weights the easy majority pixels as well.

    The reduction is left to `F.cross_entropy` so that a weighted loss keeps its
    convention of dividing by the weight of the valid pixels rather than by their
    count; the only thing added here is the all-ignored guard, which a bare reduction
    would answer with a NaN.
    """
    _check_spatial(logits, mask)
    target = _sanitise_target(mask, logits.shape[1], ignore_index)
    if not bool((target != ignore_index).any()):
        # See the module docstring: 0/0 is NaN, and a NaN loss poisons every weight it
        # touches on the backward pass.
        return logits.sum() * 0.0
    return F.cross_entropy(
        logits,
        target,
        weight=_class_weight_tensor(class_weights, logits),
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
    )


def dice_loss(
    logits: torch.Tensor,
    mask: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Soft Dice over the softmax probabilities, `1 - mean(dice over present classes)`.

    Only classes that appear in the target contribute. A class absent from the batch
    has an empty intersection and an empty union; scoring it as 0 would punish the
    model for classes the batch simply did not contain, and scoring it as 1 (an empty
    set divides to a perfect match) would let a batch of one single class look like a
    flawless prediction. Both make the loss depend on which classes happened to be
    sampled rather than on the model, so absent classes are dropped from the mean.

    Dice is scale-free, which is exactly why it is paired with cross-entropy: CE
    provides a per-pixel gradient everywhere, Dice stops the rare-class IoU from being
    washed out by the majority classes.
    """
    _check_spatial(logits, mask)
    probabilities = F.softmax(logits.float(), dim=1)
    target = mask.long()
    valid = _valid_pixels(mask, num_classes, ignore_index)

    # Ignored pixels are zeroed in both the probabilities and the one-hot target, so
    # they contribute to neither the intersection nor the union of any class.
    safe = torch.where(valid, target, torch.zeros_like(target))
    one_hot = F.one_hot(safe, num_classes).permute(0, 3, 1, 2).to(probabilities.dtype)
    keep = valid.unsqueeze(1).to(probabilities.dtype)
    probabilities = probabilities * keep
    one_hot = one_hot * keep

    reduce_dims = (0, 2, 3)
    intersection = (probabilities * one_hot).sum(dim=reduce_dims)
    denominator = probabilities.sum(dim=reduce_dims) + one_hot.sum(dim=reduce_dims)
    present = one_hot.sum(dim=reduce_dims) > 0

    if not bool(present.any()):
        # Every pixel ignored in every image of the batch.
        return logits.sum() * 0.0
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice[present].mean()


def ce_dice_loss(
    logits: torch.Tensor,
    mask: torch.Tensor,
    num_classes: int,
    class_weights: Optional[Sequence[float] | torch.Tensor] = None,
    ignore_index: int = 255,
    ce_weight: float = 1.0,
    dice_weight: float = 1.0,
    label_smoothing: float = 0.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Cross-entropy plus Dice, the standard pairing for imbalanced segmentation.

    `class_weights` applies to the cross-entropy term only; Dice is already balanced
    by construction, and weighting it as well would double-count the rebalancing.
    """
    ce = cross_entropy_loss(
        logits,
        mask,
        class_weights=class_weights,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
    )
    dice = dice_loss(logits, mask, num_classes=num_classes, ignore_index=ignore_index, eps=eps)
    return ce_weight * ce + dice_weight * dice


def focal_loss(
    logits: torch.Tensor,
    mask: torch.Tensor,
    gamma: float = 2.0,
    class_weights: Optional[Sequence[float] | torch.Tensor] = None,
    ignore_index: int = 255,
) -> torch.Tensor:
    """Focal loss: `-(1 - p_t)^gamma * log(p_t)`, averaged over labelled pixels.

    The `(1 - p_t)^gamma` factor shrinks the contribution of pixels the model already
    classifies confidently, so the gradient budget goes to the hard minority pixels -
    which, in an aerial scene, is where the roads and buildings are. `class_weights`
    plays the role of the per-class `alpha` in the paper's formulation.

    `gamma=2.0` is the value from the original paper.
    """
    _check_spatial(logits, mask)
    num_classes = logits.shape[1]
    valid = _valid_pixels(mask, num_classes, ignore_index)
    # gather() has no ignore semantics, so the ignored positions need an in-range index
    # before the valid mask removes them from the mean.
    index = mask.long().clamp(0, num_classes - 1)

    log_probabilities = F.log_softmax(logits.float(), dim=1)
    log_pt = log_probabilities.gather(1, index.unsqueeze(1)).squeeze(1)
    pt = log_pt.exp()
    focal = -((1.0 - pt) ** gamma) * log_pt
    weight = _class_weight_tensor(class_weights, logits)
    if weight is not None:
        focal = focal * weight[index]
    return _mean_over_valid(focal, valid, logits)


def aux_loss(
    aux_logits: torch.Tensor,
    mask: torch.Tensor,
    class_weights: Optional[Sequence[float] | torch.Tensor] = None,
    ignore_index: int = 255,
) -> torch.Tensor:
    """Plain cross-entropy on an auxiliary head's logits.

    DeepLabV3's auxiliary classifier is trained alongside the main head; the paper's
    recipe is to weight its loss by 0.4 and add it. This is that term on its own, so an
    auxiliary head can be evaluated and ablated separately from the main objective.

    Auxiliary logits are resized to the mask when the head produces a different
    resolution - the head is a secondary signal, so bilinear resampling of its logits
    is an acceptable approximation here in a way it never is for labels.
    """
    if tuple(aux_logits.shape[-2:]) != tuple(mask.shape[-2:]):
        aux_logits = F.interpolate(
            aux_logits.float(), size=tuple(mask.shape[-2:]), mode="bilinear", align_corners=False
        )
    return cross_entropy_loss(
        aux_logits, mask, class_weights=class_weights, ignore_index=ignore_index
    )


def combine_with_aux(
    main_loss: torch.Tensor,
    aux_logits: torch.Tensor,
    mask: torch.Tensor,
    aux_weight: float = DEFAULT_AUX_WEIGHT,
    class_weights: Optional[Sequence[float] | torch.Tensor] = None,
    ignore_index: int = 255,
) -> torch.Tensor:
    """`main_loss + aux_weight * CE(aux_logits, mask)`.

    Kept out of the individual loss signatures so that adding an auxiliary head does
    not change what "the dice loss" means: any main loss can be paired with an
    auxiliary term through this one function.
    """
    aux_weight = float(aux_weight)
    if aux_weight == 0.0:
        return main_loss
    return main_loss + aux_weight * aux_loss(
        aux_logits, mask, class_weights=class_weights, ignore_index=ignore_index
    )


LOSS_FUNCTIONS: Dict[str, Callable] = {
    "ce": cross_entropy_loss,
    # Same function as "ce"; the name records that this run expects an auxiliary head.
    "ce_aux": cross_entropy_loss,
    "dice": dice_loss,
    "ce_dice": ce_dice_loss,
    "focal": focal_loss,
}


def build_segmentation_loss(
    name: str,
    num_classes: int,
    class_weights: Optional[Sequence[float] | torch.Tensor] = None,
    ignore_index: int = 255,
    aux_weight: float = DEFAULT_AUX_WEIGHT,
    **kwargs,
) -> Callable[..., torch.Tensor]:
    """Look up a segmentation loss by name and bind its options.

    Returns a callable ``criterion(logits, mask, aux_logits=None)``. Passing auxiliary
    logits adds ``aux_weight`` times their cross-entropy to the main loss; leaving them
    out gives the main loss alone, which is what evaluation and the auxiliary-free
    models use.

    Every option in `**kwargs` is checked against the target function's own signature,
    so a typo such as ``loss_kwargs={"gama": 3.0}`` fails here, naming the options that
    would have worked, rather than being silently swallowed by a `**kwargs` somewhere
    inside the loss.
    """
    if not name:
        raise ValueError("a segmentation loss name is required")
    if name not in LOSS_FUNCTIONS:
        raise ValueError(
            f"unknown segmentation loss {name!r}; expected one of {SEGMENTATION_LOSSES}"
        )
    if float(aux_weight) < 0.0:
        raise ValueError(f"aux_weight must be non-negative, got {aux_weight}")

    function = LOSS_FUNCTIONS[name]
    allowed = set(function.__code__.co_varnames[: function.__code__.co_argcount])
    unknown = set(kwargs) - allowed
    if unknown:
        raise ValueError(
            f"unknown option(s) {sorted(unknown)} for segmentation loss {name!r}; "
            f"it accepts {sorted(allowed)}"
        )

    # Bind the options this builder owns, but only where the target loss understands
    # them: `num_classes` belongs to dice, `class_weights` to the cross-entropy family.
    call_kwargs: Dict[str, object] = dict(kwargs)
    if "num_classes" in allowed:
        call_kwargs["num_classes"] = int(num_classes)
    if "ignore_index" in allowed:
        call_kwargs["ignore_index"] = int(ignore_index)
    if class_weights is not None:
        if "class_weights" not in allowed:
            raise ValueError(
                f"segmentation loss {name!r} does not accept class_weights; "
                f"it accepts {sorted(allowed)}"
            )
        call_kwargs["class_weights"] = class_weights

    def criterion(
        logits: torch.Tensor,
        mask: torch.Tensor,
        aux_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss = function(logits, mask, **call_kwargs)
        if aux_logits is None:
            return loss
        return combine_with_aux(
            loss,
            aux_logits,
            mask,
            aux_weight=aux_weight,
            class_weights=class_weights,
            ignore_index=ignore_index,
        )

    return criterion
