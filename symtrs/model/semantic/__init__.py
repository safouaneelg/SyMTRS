"""Semantic segmentation models behind the common `TaskModel` contract.

One module per architecture:

  * `deeplab.py`   - torchvision DeepLabV3 (ResNet-50 / ResNet-101)
  * `segformer.py` - the SegFormer MLP decoder on a timm MiT encoder
  * `unet.py`      - a small in-repo U-Net, the from-scratch baseline

They share the parts that must not drift apart, which live here:

  * `SegmentationModel` - the TaskModel subclass all three extend. It owns the
    preprocessing, the logits-to-class-indices step, the loss wiring and the freeze
    handling, so a change to the ignore-index convention or to the way auxiliary heads
    are combined happens once rather than three times.

There is deliberately no key table here. Which keys exist, and which `backbone` or
`variant` each one implies, is written down once in `symtrs.registry`; a second copy in
this file would be a second thing to keep in sync, and the drift would be silent -
which is the failure `symtrs.registry` exists to prevent. `build_model("segformer_b2")`
is the way in.

Importing this package is not free - it pulls in torch through `symtrs.model.base`,
like every other model module in the package - but it never touches torchvision or
timm until a model that needs them is constructed.
"""

from __future__ import annotations

import importlib
from typing import Dict, Iterator, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from symtrs.model.base import TaskModel, apply_freeze_policy
from symtrs.model.semantic.losses import build_segmentation_loss

#: Modules exposed lazily as attributes of this package.
_LAZY_EXPORTS = {
    "DeepLabV3Segmentation": "symtrs.model.semantic.deeplab",
    "SegFormerSegmentation": "symtrs.model.semantic.segformer",
    "UNetSegmentation": "symtrs.model.semantic.unet",
}


def _resolve(target: str):
    """Import a "module:ClassName" reference, reporting a bad reference clearly."""
    module_name, _, class_name = target.partition(":")
    if not module_name or not class_name:
        raise ValueError(f"malformed model reference {target!r}; expected 'module:ClassName'")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise ImportError(f"{module_name} has no {class_name}") from exc


class SegmentationModel(TaskModel):
    """Shared behaviour for the SyMTRS semantic segmentation models.

    Subclasses supply the network and its registry key; everything that has to agree
    across architectures - ImageNet input normalisation, resizing logits back to the
    input resolution, turning logits into class indices, and the loss/auxiliary-head
    wiring - is implemented once, here.
    """

    task = "segmentation"

    #: Loss used when the run config does not name one.
    default_loss = "ce"

    #: torchvision is listed because the flagship model needs it; a torch-only model
    #: such as the U-Net narrows this to ("torch",).
    requires = ("torch", "torchvision")

    def __init__(
        self,
        num_classes: int = 6,
        ignore_index: int = 255,
        loss: str = "",
        loss_kwargs: Optional[Dict] = None,
        freeze: str = "",
        pretrained: bool = True,
        **model_kwargs,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError(f"num_classes must be at least 2, got {num_classes}")
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.pretrained = bool(pretrained)
        # Recorded so the checkpoint carries enough to rebuild the architecture even
        # though these did not change it.
        self.model_kwargs = dict(model_kwargs)

        options = dict(loss_kwargs or {})
        # class_weights is kept out of the loss options: it is length-num_classes and
        # must be moved to the logits' device, which the loss functions do for it.
        class_weights = options.pop("class_weights", None)
        self.class_weights = class_weights
        self.loss_name = loss or self.default_loss
        self.criterion = build_segmentation_loss(
            self.loss_name,
            self.num_classes,
            class_weights=class_weights,
            ignore_index=self.ignore_index,
            **options,
        )
        self.freeze = freeze

        # Auxiliary-head logits from the most recent forward pass, or None. See
        # DeepLabV3Segmentation.forward for why a stashed tensor is necessary.
        self._last_aux: Optional[torch.Tensor] = None

    # ------------------------------------------------------------- preprocessing

    def preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        """ImageNet normalisation, which is what every backbone here expects.

        The models this module wraps are all initialised from ImageNet- or
        COCO-pretrained weights, so the input statistics are not a free choice: feeding
        raw `[0, 1]` values would leave the pretrained features seeing a differently
        scaled image than they were trained on.
        """
        from symtrs.data.transforms import normalize_image

        return normalize_image(image)

    # ------------------------------------------------------------------- forward

    def forward_at_size(self, image: torch.Tensor) -> torch.Tensor:
        """`forward`, with logits resized to the input's spatial size.

        Takes model-ready input, exactly as `DepthAdapter.forward_at_size` does, and as
        `forward` does; `predict` is the entry point that turns a raw `[0, 1]` image
        into that input. Doing the normalisation here instead would leave no way to call
        it on already-prepared input without applying it twice.

        Bilinear is the right resampling for logits, which are a continuous score per
        class: unlike a label map, a logit has a meaningful interpolation. Nearest
        neighbour would make the decision boundary jump by whole input pixels.
        """
        logits = self.forward(image)
        if tuple(logits.shape[-2:]) != tuple(image.shape[-2:]):
            logits = F.interpolate(
                logits.float(), size=tuple(image.shape[-2:]), mode="bilinear", align_corners=False
            )
        return logits

    # ------------------------------------------------------------------- predict

    @torch.no_grad()
    def predict(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """`[B, 3, H, W]` in `[0, 1]` to int64 class indices at the input resolution.

        `argmax` over the logits is the only postprocessing a segmentation model needs;
        the ignore value never appears in the output because `num_classes` excludes it,
        and the metrics treat the ground truth's ignore pixels as absent.
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        was_training = self.training
        self.eval()
        try:
            prepared = self.preprocess_image(image)
            logits = self.forward_at_size(prepared)
            prediction = logits.argmax(dim=1).to(torch.int64)
        finally:
            # Batch-norm statistics must not be updated by an inference call, and a
            # caller that predicts mid-training should get its model back as it was.
            if was_training:
                self.train()
        return prediction

    # ---------------------------------------------------------------------- loss

    def compute_loss(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """`(total_loss, components)` for the training loop.

        The auxiliary-head term, when the network produced one, is folded in here and
        reported as part of the same component, because it is not an independently
        meaningful number - it is the same objective on a coarser head.
        """
        aux_logits = self._last_aux
        loss = self.criterion(logits, mask, aux_logits=aux_logits)
        components = {f"loss_{self.loss_name}": float(loss.detach())}
        return loss, components

    # ------------------------------------------------------------- optimisation

    def trainable_parameters(self, freeze: str = "") -> Iterator[nn.Parameter]:
        """Apply the freeze policy, then yield the parameters left trainable.

        The describe() line is printed because a freeze policy that matched nothing
        (the encoder-recognition heuristic names modules by prefix) is otherwise
        invisible: the run simply trains ten times more parameters than intended.
        """
        policy = freeze or self.freeze
        apply_freeze_policy(self, policy)
        print(f"[symtrs] freeze={policy or 'none'} -> {self.describe()}")
        return (p for p in self.parameters() if p.requires_grad)


def __getattr__(name: str):
    """Expose the model classes as package attributes without importing them eagerly."""
    if name in _LAZY_EXPORTS:
        return _resolve(f"{_LAZY_EXPORTS[name]}:{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SegmentationModel",
    "DeepLabV3Segmentation",
    "SegFormerSegmentation",
    "UNetSegmentation",
]
