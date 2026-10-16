"""SegFormer: the official MLP decoder, reimplemented on a transformers MiT encoder.

SegFormer is a plain transformer encoder with a hierarchical, convolution-like feature
pyramid - strides 4/8/16/32 - and a head that does nothing but mix those four scales.
It is in the benchmark because it has no positional-encoding interpolation to fight
with: the encoder is fully convolutional in its spatial behaviour, so a model trained
at one resolution can be evaluated at another, which matters when the same checkpoint
is run at 256 and at 2048.

This is not a port of the official `SegFormer` repository. The encoder is the MiT
backbone of `transformers.SegformerModel` (`nvidia/mit-b0`/`-b1`/`-b2`, ImageNet-1k
pretrained), and the head below is a reimplementation of the
published MLP-decoder structure on top of it: linear projections of the four stages to
a common width, a 1x1 fusion convolution, dropout, a 1x1 classifier, and a x4
bilinear upsample. It is faithful to the structure, not to the original weights, so
it is a from-pretrained-encoder fine-tune, not a reproduction of a published number.

Why `transformers` and not timm: timm has never shipped MiT (`timm.create_model("mit_b0")`
is "Unknown model" in every release, 0.6.12 and 1.0.x alike), so an earlier version of this
adapter could not build at all. `nvidia/mit-*` are the official MiT weights, and they load
in `SegformerModel` without any key remapping. `transformers` is imported lazily, here,
so the registry and `symtrs doctor` work without it. This also means SegFormer no longer
constrains the timm version, which is what lets ZoeDepth keep its own timm==0.6.12
environment (AGENTS.md ground rule 10).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from symtrs.model.semantic import SegmentationModel

#: Registry variant -> the Hub id of the pretrained MiT encoder.
SEGFORMER_ENCODERS: Dict[str, str] = {
    "b0": "nvidia/mit-b0",
    "b1": "nvidia/mit-b1",
    "b2": "nvidia/mit-b2",
}

#: The MiT architecture per variant (Xie et al. 2021, table 6), used when
#: `pretrained=False` so that a from-scratch build needs neither network nor cache.
#: Everything else in `SegformerConfig` is the library default, which is the paper's.
_MIT_ARCHITECTURE: Dict[str, Dict[str, list]] = {
    "b0": {"hidden_sizes": [32, 64, 160, 256], "depths": [2, 2, 2, 2]},
    "b1": {"hidden_sizes": [64, 128, 320, 512], "depths": [2, 2, 2, 2]},
    "b2": {"hidden_sizes": [64, 128, 320, 512], "depths": [3, 4, 6, 3]},
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
    requires = ("torch", "transformers")

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
            from transformers import SegformerConfig, SegformerModel
        except ImportError as exc:
            raise ImportError(
                "SegFormer needs `transformers` for its MiT encoder; install it with "
                "`pip install -e '.[segformer]'`"
            ) from exc

        if pretrained:
            return SegformerModel.from_pretrained(SEGFORMER_ENCODERS[variant])
        return SegformerModel(SegformerConfig(**_MIT_ARCHITECTURE[variant]))

    @staticmethod
    def _stage_tensors(output) -> List[torch.Tensor]:
        """The four `[B, C, H, W]` stage maps of a `SegformerModel` output.

        `output_hidden_states=True` returns one tensor per encoder stage, already
        reshaped to `[B, C, H, W]` (`config.reshape_last_stage` defaults to True, and the
        count is checked so a config that disabled it is a clear error rather than a
        shape mismatch in the decoder).
        """
        stages = list(output.hidden_states)
        if len(stages) != NUM_STAGES:
            raise RuntimeError(
                f"the MiT encoder produced {len(stages)} stages; the SegFormer head is "
                f"built for {NUM_STAGES}"
            )
        for stage in stages:
            if stage.dim() != 4:
                raise RuntimeError(
                    "the MiT encoder returned a non-spatial stage; keep "
                    "`reshape_last_stage=True` in the SegformerConfig"
                )
        return stages

    def _encode(self, image: torch.Tensor) -> List[torch.Tensor]:
        return self._stage_tensors(self.encoder(pixel_values=image, output_hidden_states=True))

    def _discover_channels(self) -> List[int]:
        """Read the four stage widths from the encoder with a probe forward pass.

        The channel counts differ between b0 (32/64/160/256) and b1/b2, and the
        encoder configuration can change them, so they are measured rather than hardcoded:
        guessing them is a shape mismatch in the decoder that only shows up at the
        first training step, with a stack trace that points at the fusion convolution
        rather than at the wrong number. The probe runs under `no_grad` and on a
        throwaway 224x224 input, so it costs one forward pass of the smallest batch.
        """
        was_training = self.encoder.training
        self.encoder.eval()
        try:
            with torch.no_grad():
                stages = self._encode(torch.zeros(1, 3, _PROBE_SIZE, _PROBE_SIZE))
                channels = [int(stage.shape[1]) for stage in stages]
        finally:
            if was_training:
                self.encoder.train()
        return channels

    # -------------------------------------------------------------------- forward

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Logits `[B, num_classes, H, W]`, at the input resolution for even sizes."""
        return self.head(self._encode(image))
