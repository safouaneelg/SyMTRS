"""SegFormer: the official MLP decoder, reimplemented on a timm MiT encoder.

SegFormer is a plain transformer encoder with a hierarchical, convolution-like feature
pyramid - strides 4/8/16/32 - and a head that does nothing but mix those four scales.
It is in the benchmark because it has no positional-encoding interpolation to fight
with: the encoder is fully convolutional in its spatial behaviour, so a model trained
at one resolution can be evaluated at another, which matters when the same checkpoint
is run at 256 and at 2048.

This is not a port of the official `SegFormer` repository. The encoder is timm's MiT
(`mit_b0`/`mit_b1`/`mit_b2`), and the head below is a reimplementation of the
published MLP-decoder structure on top of it: linear projections of the four stages to
a common width, a 1x1 fusion convolution, dropout, a 1x1 classifier, and a x4
bilinear upsample. It is faithful to the structure, not to the original weights, so
it is a from-pretrained-encoder fine-tune, not a reproduction of a published number.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from symtrs.model.semantic import SegmentationModel

#: Registry variant -> the timm encoder name. `variant` is the short form used in the
#: registry key; the encoder is `mit_<variant>`.
SEGFORMER_ENCODERS: Dict[str, str] = {
    "b0": "mit_b0",
    "b1": "mit_b1",
    "b2": "mit_b2",
}

#: The MiT encoder emits four stages, at output strides 4, 8, 16 and 32. The head is
#: built around exactly four inputs, so the count is checked rather than assumed.
NUM_STAGES = 4

#: The spatial size of the dummy image used to discover the encoder's channel counts.
_PROBE_SIZE = 224


class _SegFormerHead(nn.Module):
    """The MLP decoder: project every stage, fuse, classify.

    All four stages are upsampled to the stride-4 resolution before concatenation
    rather than fused at their native scales. That keeps the head a sequence of 1x1
    convolutions and linears - which is the point of SegFormer's decoder, and what
    makes it cheap - instead of the stack of 3x3 convolutions a U-Net-style decoder
    would need to merge differently-sized features.
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        embed_dim: int = 256,
        num_classes: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.projections = nn.ModuleList(
            [nn.Linear(channels, embed_dim) for channels in in_channels]
        )
        self.fuse = nn.Conv2d(len(in_channels) * embed_dim, embed_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, stages: Sequence[torch.Tensor]) -> torch.Tensor:
        """Four encoder feature maps -> logits at the stride-4 resolution."""
        target_size = tuple(stages[0].shape[-2:])
        projected: List[torch.Tensor] = []
        for stage, projection in zip(stages, self.projections):
            batch, channels, height, width = stage.shape
            tokens = stage.flatten(2).transpose(1, 2)  # [B, H*W, C]
            tokens = projection(tokens)  # [B, H*W, embed_dim]
            feature = tokens.transpose(1, 2).reshape(batch, -1, height, width)
            if tuple(feature.shape[-2:]) != target_size:
                feature = F.interpolate(
                    feature, size=target_size, mode="bilinear", align_corners=False
                )
            projected.append(feature)

        fused = self.fuse(torch.cat(projected, dim=1))
        fused = self.dropout(fused)
        logits = self.classifier(fused)
        return F.interpolate(logits, scale_factor=4.0, mode="bilinear", align_corners=False)


class SegFormerSegmentation(SegmentationModel):
    """A SegFormer MiT encoder with the MLP decoder from the paper.

    The decoder works at stride 4, then upsamples by 4; `SegmentationModel.forward_at_size`
    then guarantees the logits match the input exactly if the input's height or width
    was not a multiple of 4, where a fixed x4 upsample can land a pixel or two short.
    """

    task = "segmentation"
    default_loss = "ce"
    requires = ("torch", "timm")

    def __init__(
        self,
        variant: str = "b0",
        embed_dim: int = 256,
        num_classes: int = 6,
        ignore_index: int = 255,
        loss: str = "",
        loss_kwargs: Optional[Dict] = None,
        freeze: str = "",
        pretrained: bool = True,
        dropout: float = 0.1,
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
        if variant not in SEGFORMER_ENCODERS:
            raise ValueError(
                f"unknown SegFormer variant {variant!r}; expected one of "
                f"{sorted(SEGFORMER_ENCODERS)}"
            )
        self.variant = variant
        self.embed_dim = int(embed_dim)
        self.key = f"segformer_{variant}"

        self.encoder = self._build_encoder(variant, pretrained)
        in_channels = self._discover_channels()
        self.head = _SegFormerHead(
            in_channels, embed_dim=self.embed_dim, num_classes=self.num_classes, dropout=dropout
        )
        print(
            f"[symtrs] {self.key}: encoder channels {list(in_channels)}, "
            f"embed_dim={self.embed_dim}"
        )

    # ------------------------------------------------------------------- building

    @staticmethod
    def _build_encoder(variant: str, pretrained: bool) -> nn.Module:
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "SegFormer needs timm for its MiT encoder; install it with `pip install timm`"
            ) from exc

        return timm.create_model(
            SEGFORMER_ENCODERS[variant], pretrained=pretrained, features_only=True
        )

    @staticmethod
    def _stage_tensors(output) -> List[torch.Tensor]:
        """Normalise a timm feature-extraction result to a list of tensors.

        timm has returned feature maps as either a plain list of tensors or, in newer
        versions, a list of dicts carrying a `"feat"` key alongside metadata. Both
        shapes appear across the timm versions this benchmark runs on, and the
        difference is an import-time surprise rather than a modelling choice, so it is
        absorbed here once instead of at each of the two call sites.
        """
        stages: List[torch.Tensor] = []
        for item in output:
            if isinstance(item, dict):
                if "feat" not in item:
                    raise RuntimeError(
                        f"timm feature map dict has no 'feat' key (keys: {sorted(item)})"
                    )
                item = item["feat"]
            if not torch.is_tensor(item):
                raise RuntimeError(
                    f"unexpected timm feature map of type {type(item).__name__}; "
                    "expected a tensor or a dict with a 'feat' key"
                )
            stages.append(item)
        if len(stages) != NUM_STAGES:
            raise RuntimeError(
                f"the MiT encoder produced {len(stages)} stages; the SegFormer head is "
                f"built for {NUM_STAGES}"
            )
        return stages

    def _discover_channels(self) -> List[int]:
        """Read the four stage widths from the encoder with a probe forward pass.

        The channel counts differ between b0 (32/64/160/256) and b1/b2, and timm's
        configuration can change them, so they are measured rather than hardcoded:
        guessing them is a shape mismatch in the decoder that only shows up at the
        first training step, with a stack trace that points at the fusion convolution
        rather than at the wrong number. The probe runs under `no_grad` and on a
        throwaway 224x224 input, so it costs one forward pass of the smallest batch.
        """
        was_training = self.encoder.training
        self.encoder.eval()
        try:
            with torch.no_grad():
                probe = self.encoder(torch.zeros(1, 3, _PROBE_SIZE, _PROBE_SIZE))
                stages = self._stage_tensors(probe)
                channels = [int(stage.shape[1]) for stage in stages]
        finally:
            if was_training:
                self.encoder.train()
        return channels

    # -------------------------------------------------------------------- forward

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Logits `[B, num_classes, H, W]`, at the input resolution for even sizes."""
        stages = self._stage_tensors(self.encoder(image))
        return self.head(stages)
