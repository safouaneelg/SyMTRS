"""Shared model interface.

Every task model is a `torch.nn.Module` with three responsibilities the training loop
and the SDK rely on:

  * `preprocess_image` - raw `[0, 1]` tensor to whatever the network wants.
  * `forward`          - the network itself, returning raw output in the model's own
                         space.
  * `predict`          - end-to-end: raw image in, task-space answer out, including
                         any postprocessing such as converting to metres.

Splitting `forward` from `predict` is what lets training optimise in the model's
native space (disparity, logits) while evaluation and the SDK always see the same
thing: metres, or class indices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, Optional

import torch
import torch.nn as nn


@dataclass
class DepthTarget:
    """What the depth loss consumes, plus what is needed to get back to metres.

    `unit` is "metres" for metric models and "normalized_disparity" for relative
    ones; it is recorded in the checkpoint so a reloaded model knows how to interpret
    its own output.
    """

    values: torch.Tensor
    mask: torch.Tensor
    stats: Dict[str, torch.Tensor] = field(default_factory=dict)
    unit: str = "metres"

    def to(self, device) -> "DepthTarget":
        return DepthTarget(
            values=self.values.to(device),
            mask=self.mask.to(device),
            stats={k: v.to(device) for k, v in self.stats.items()},
            unit=self.unit,
        )


class TaskModel(nn.Module):
    """Base class for all SyMTRS task models."""

    #: Registry key, e.g. "midas_dpt_hybrid".
    key: str = ""
    #: "depth" | "segmentation".
    task: str = ""
    #: Default loss name used when the run config does not set one.
    default_loss: str = ""
    #: Importable modules this adapter needs; used by `symtrs doctor` to explain a
    #: missing optional dependency instead of failing with an ImportError trace.
    requires: tuple = ()

    def preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        """`[B, 3, H, W]` in `[0, 1]` to model input. Overridden by most adapters."""
        return image

    def forward_at_size(self, image: torch.Tensor) -> torch.Tensor:
        """`forward` with its output resized to the input's spatial size.

        `image` has **already** been preprocessed and padded by
        `symtrs.engine.loop.prepare_input`, which is the single owner of that step. An
        override must not normalise again: doing so double-applies the ImageNet
        transform and costs accuracy with no failure to notice. Adapters override this
        only when the network changes resolution on its own.
        """
        import torch.nn.functional as F

        prediction = self.forward(image)
        if prediction.dim() == 3:
            prediction = prediction.unsqueeze(1)
        if tuple(prediction.shape[-2:]) == tuple(image.shape[-2:]):
            return prediction
        return F.interpolate(
            prediction.float(), size=image.shape[-2:], mode="bilinear", align_corners=False
        )

    def predict(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """`[B, 3, H, W]` in `[0, 1]` to the task's answer space."""
        raise NotImplementedError

    def trainable_parameters(self, freeze: str = "") -> Iterator[nn.Parameter]:
        """Parameters to optimise under a freeze policy.

        `freeze` is "", "encoder", "head" or "last_n:<k>". The default is everything,
        which is only affordable for the small models.
        """
        return (p for p in self.parameters() if p.requires_grad)

    def describe(self) -> str:
        """One-line summary for logs and `symtrs doctor`."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"{self.key} ({self.task}, {type(self).__name__}): "
            f"{total / 1e6:.1f}M params, {trainable / 1e6:.1f}M trainable"
        )


#: What `freeze="encoder"` matches when the model does not say otherwise.
DEFAULT_ENCODER_PREFIXES = ("encoder", "backbone", "pretrained")


def apply_freeze_policy(model: nn.Module, freeze: str, encoder_prefixes=None) -> None:
    """Set `requires_grad` according to a freeze policy.

    Recognising the encoder by name is a heuristic; `freeze="head"` is the safe option
    on an unfamiliar architecture, and the parameter counts printed by `symtrs doctor`
    tell you immediately whether the heuristic did what you meant.

    A model whose own layout defeats the heuristic names its encoder with a
    `freeze_encoder_prefixes` class attribute rather than being special-cased here: MiDaS
    keeps its *entire* network under `backbone.`, so the default prefix froze the decoder
    along with the ViT and left nothing to train.
    """
    if encoder_prefixes is None:
        encoder_prefixes = getattr(model, "freeze_encoder_prefixes", None) or DEFAULT_ENCODER_PREFIXES
    if not freeze:
        return

    if freeze == "encoder":
        for name, parameter in model.named_parameters():
            if any(name.startswith(prefix) or f".{prefix}" in name for prefix in encoder_prefixes):
                parameter.requires_grad = False
        return

    if freeze == "head":
        # Freeze everything, then unfreeze the classifier/head modules.
        for parameter in model.parameters():
            parameter.requires_grad = False
        for name, parameter in model.named_parameters():
            lowered = name.lower()
            if any(token in lowered for token in ("head", "classifier", "decode", "last_layer", "output_conv")):
                parameter.requires_grad = True
        return

    if freeze.startswith("last_n:"):
        try:
            count = int(freeze.split(":", 1)[1])
        except ValueError:
            raise ValueError(f"malformed freeze policy {freeze!r}; expected last_n:<k>") from None
        blocks = list(model.children())
        for block in blocks[:-count] if count > 0 else []:
            for parameter in block.parameters():
                parameter.requires_grad = False
        return

    raise ValueError(
        f"unknown freeze policy {freeze!r}; expected '', 'encoder', 'head' or 'last_n:<k>'"
    )
