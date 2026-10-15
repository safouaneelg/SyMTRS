"""A small U-Net, trained from scratch.

The benchmark needs one model whose numbers do not depend on an external pretrained
checkpoint: it is the floor that says how much of a SyMTRS score comes from the
architecture or the head and how much comes from ImageNet/COCO features. This U-Net
is that model. It is deliberately small (`base_channels=32`, `depth=4`) so a
from-scratch run finishes in a reasonable time at the benchmark's lower resolutions,
and shallow enough to be trained on the full dataset rather than a curated subset.

There are no pretrained weights for it anywhere; `pretrained` is accepted and ignored,
so that a run config can be switched between `unet_small` and, say, `deeplabv3_r50`
without the model factory having to special-case one key.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from symtrs.model.semantic import SegmentationModel


class _DoubleConv(nn.Module):
    """Conv3x3 - BatchNorm - ReLU, twice, the U-Net block.

    Biased convolutions are omitted because the following BatchNorm subtracts its own
    mean; the bias would be a parameter with no effect.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetSegmentation(SegmentationModel):
    """A four-level U-Net encoder/decoder with skip connections.

    The decoder upsamples bilinearly and then convolves, rather than using a strided
    transposed convolution. Transposed convolutions with stride 2 lay down a periodic
    pattern that the following convolutions amplify - the checkerboard artefact - and
    in segmentation it lands exactly on the class boundaries, which is the one place
    the output is looked at. Interpolation is a fixed, artefact-free operation and
    leaves the network a parameter to spend elsewhere.
    """

    task = "segmentation"
    default_loss = "ce"
    requires = ("torch",)

    def __init__(
        self,
        num_classes: int = 6,
        ignore_index: int = 255,
        loss: str = "",
        loss_kwargs: Optional[Dict] = None,
        freeze: str = "",
        pretrained: bool = True,
        base_channels: int = 32,
        depth: int = 4,
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
        if depth < 1:
            raise ValueError(f"depth must be at least 1, got {depth}")
        if base_channels < 1:
            raise ValueError(f"base_channels must be at least 1, got {base_channels}")

        self.key = "unet_small"
        self.base_channels = int(base_channels)
        self.depth = int(depth)

        if pretrained:
            print(
                "[symtrs] unet_small has no pretrained weights; "
                "pretrained=True is ignored and it trains from scratch"
            )

        channels = [self.base_channels * (2 ** level) for level in range(self.depth)]

        self.encoders = nn.ModuleList()
        in_channels = 3
        for out_channels in channels:
            self.encoders.append(_DoubleConv(in_channels, out_channels))
            in_channels = out_channels
        self.pool = nn.MaxPool2d(kernel_size=2)

        bottleneck_channels = channels[-1] * 2
        self.bottleneck = _DoubleConv(channels[-1], bottleneck_channels)

        # The decoder mirrors the encoder: each block sees its skip connection
        # concatenated onto the upsampled feature map, so its input width is the
        # previous level's output plus that level's encoder width.
        self.decoders = nn.ModuleList()
        upsampled_channels = bottleneck_channels
        for level in reversed(range(self.depth)):
            self.decoders.append(_DoubleConv(upsampled_channels + channels[level], channels[level]))
            upsampled_channels = channels[level]

        self.classifier = nn.Conv2d(channels[0], self.num_classes, kernel_size=1)

    # -------------------------------------------------------------------- forward

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Logits `[B, num_classes, H, W]` at the input resolution for even sizes."""
        skips: List[torch.Tensor] = []
        features = image
        for encoder in self.encoders:
            features = encoder(features)
            # Checked out before pooling, so the decoder can concat the un-pooled map.
            skips.append(features)
            features = self.pool(features)

        features = self.bottleneck(features)

        for decoder, skip in zip(self.decoders, reversed(skips)):
            # Interpolate to the skip's exact size rather than by a factor of two: an odd
            # input dimension makes MaxPool2d floor, and the two paths would then differ.
            features = F.interpolate(
                features, size=tuple(skip.shape[-2:]), mode="bilinear", align_corners=False
            )
            features = decoder(torch.cat([skip, features], dim=1))

        return self.classifier(features)
