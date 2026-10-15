"""DeepLabV3 semantic segmentation, from torchvision.

DeepLabV3 over a ResNet backbone is the reference architecture for this benchmark:
atrous convolutions at the last two stages keep the output stride at 8, so the
prediction is dense compared with the stride-32 feature maps an encoder-only model
would give, and the ASPP head aggregates context at several field sizes at once. On
aerial imagery, where a roof and the road beside it are separate classes a few pixels
apart, that density is the property being benchmarked.

    from torchvision.models.segmentation import deeplabv3_resnet50
    model = deeplabv3_resnet50(weights="COCO_WITH_VOC_LABELS_V1")

The COCO+VOC checkpoint predicts 21 VOC classes. Its *encoder* is the valuable part;
the 21-class head is replaced here with a `num_classes` head, which is why
`pretrained=True` means "pretrained backbone with a fresh head" rather than a
ready-to-use SyMTRS model.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from symtrs.model.semantic import SegmentationModel

#: Registry backbone name -> the torchvision builder for it.
DEEPLAB_BUILDERS: Dict[str, str] = {
    "r50": "deeplabv3_resnet50",
    "r101": "deeplabv3_resnet101",
}


class DeepLabV3Segmentation(SegmentationModel):
    """DeepLabV3 with a ResNet-50 or ResNet-101 backbone.

    The auxiliary head is trained with the main one, weighted by
    `loss_kwargs["aux_weight"]` (0.4 by default, the value the paper uses). It is what
    makes `default_loss` `"ce_aux"`: the default run is the architecture used as
    published, not the main head alone.
    """

    task = "segmentation"
    default_loss = "ce_aux"
    requires = ("torch", "torchvision")

    def __init__(
        self,
        backbone: str = "r50",
        num_classes: int = 6,
        ignore_index: int = 255,
        loss: str = "",
        loss_kwargs: Optional[Dict] = None,
        freeze: str = "",
        pretrained: bool = True,
        **model_kwargs,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            ignore_index=ignore_index,
            loss=loss,
            loss_kwargs=loss_kwargs,
            freeze=freeze,
            pretrained=pretrained,
            **model_kwargs,
        )
        if backbone not in DEEPLAB_BUILDERS:
            raise ValueError(
                f"unknown DeepLabV3 backbone {backbone!r}; expected one of "
                f"{sorted(DEEPLAB_BUILDERS)}"
            )
        self.backbone = backbone
        # The registry key names the variant, because "deeplabv3" alone does not say
        # which ResNet depth a checkpoint was trained with.
        self.key = f"deeplabv3_{backbone}"

        self.model = self._build(backbone, pretrained)
        self._replace_heads(self.num_classes)

    # ------------------------------------------------------------------- building

    @staticmethod
    def _build(backbone: str, pretrained: bool) -> nn.Module:
        """The torchvision model, with COCO+VOC weights when asked for.

        torchvision renamed the `pretrained=True` flag to a `weights=` enum in 0.13.
        Both spellings are tried so the adapter does not pin the package version; only
        the API-mismatch exceptions are caught, so a failed weight download still
        surfaces as itself instead of being retried as an untrained model.
        """
        try:
            from torchvision.models import segmentation
        except ImportError as exc:  # pragma: no cover - torchvision is a declared extra
            raise ImportError(
                "DeepLabV3 requires torchvision; install it with `pip install torchvision`"
            ) from exc

        builder = getattr(segmentation, DEEPLAB_BUILDERS[backbone])
        if not pretrained:
            try:
                return builder(weights=None)
            except TypeError:
                return builder(pretrained=False)

        for weights in ("COCO_WITH_VOC_LABELS_V1", "DEFAULT"):
            try:
                return builder(weights=weights)
            except (TypeError, ValueError):
                continue
        return builder(pretrained=True)

    def _replace_heads(self, num_classes: int) -> None:
        """Swap the 21-class heads for `num_classes`, keeping the pretrained encoder.

        `classifier` is an ASPP followed by a small head whose `[4]` is the final 1x1
        convolution; `aux_classifier` is an FCN head with the same layout. Only that
        final convolution changes - everything before it is the part the pretrained
        weights are worth. The input width is read off the convolution being replaced
        rather than assumed, so a backbone change that alters it fails at construction
        instead of at the first forward pass.
        """
        head = self.model.classifier[4]
        if not isinstance(head, nn.Conv2d):
            raise RuntimeError(
                f"expected the DeepLabV3 classifier's last layer to be a Conv2d, got "
                f"{type(head).__name__}; torchvision's layout has changed"
            )
        self.model.classifier[4] = nn.Conv2d(head.in_channels, num_classes, kernel_size=1)
        in_channels = head.in_channels

        if self.model.aux_classifier is not None:
            aux_head = self.model.aux_classifier[4]
            if not isinstance(aux_head, nn.Conv2d):
                raise RuntimeError(
                    f"expected the DeepLabV3 auxiliary classifier's last layer to be a "
                    f"Conv2d, got {type(aux_head).__name__}"
                )
            self.model.aux_classifier[4] = nn.Conv2d(
                aux_head.in_channels, num_classes, kernel_size=1
            )
        print(f"[symtrs] {self.key}: replaced heads ({in_channels} -> {num_classes} classes)")

    # -------------------------------------------------------------------- forward

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Main-head logits `[B, num_classes, H, W]`.

        torchvision returns an `OrderedDict`: `"out"` always, and `"aux"` only while the
        module is in training mode and an auxiliary head exists. The training loop calls
        `compute_loss(logits, mask)` with only the returned tensor, so the auxiliary
        logits have nowhere else to go - they are stashed on the instance for
        `compute_loss` to pick up. That is the entire reason `self._last_aux` exists; it
        is set on every forward pass, including to None in eval, so a stale value from an
        earlier batch can never be combined with fresh logits.
        """
        output = self.model(image)
        if isinstance(output, dict):
            logits = output.get("out")
            if logits is None:
                raise RuntimeError(
                    f"DeepLabV3 forward returned a dict without 'out' (keys: {sorted(output)})"
                )
            self._last_aux = output.get("aux")
            return logits

        # Older torchvision returned the bare tensor when no auxiliary head was present.
        self._last_aux = None
        return output
