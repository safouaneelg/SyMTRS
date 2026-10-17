"""The depth adapter contract.

One adapter per model family. It is the only place model-specific behaviour lives:
how to feed the network, what its raw output means, how to turn a metric ground truth
into a target that loss matches, and how to get back to metres.

The contract is deliberately narrow, because this is the seam that lets MiDaS,
ZoeDepth, Depth Anything 3 and Depth Pro share one training loop despite disagreeing
about almost everything else.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from symtrs.metrics.depth import DISPARITY_ALIGNMENTS
from symtrs.model.base import DepthTarget, TaskModel, apply_freeze_policy
from symtrs.model.depth.losses import build_depth_loss, default_loss_for_protocol, gradient_loss

EPS = 1e-7

#: Conventions for aligning a relative model's output to metric ground truth. Kept
#: here as documentation of what each registry default means.
ALIGNMENT_NOTES = {
    "none": "no alignment; only meaningful for metric models",
    "median": "single scale from the ratio of medians",
    "scale": "least-squares scale in depth space; the Depth Anything / MiDaS comparison convention",
    "scale_shift": "least-squares scale and shift; most permissive",
    "scale_shift_disparity": (
        "least-squares scale and shift in disparity space against 1/GT, then inverted; "
        "the MiDaS protocol, for models whose output is s/z + t"
    ),
}


class DepthAdapter(TaskModel):
    """Base class for monocular depth models."""

    task = "depth"

    #: "metric" (outputs metres) or "relative" (affine-invariant, usually disparity).
    protocol = "metric"

    #: Alignment the metrics use by default for this model. Relative models override.
    default_align = "none"

    #: What the raw output is: "depth" (metres, or depth up to a scale) or "disparity" (MiDaS's
    #: `s / z + t`). With `protocol` it fixes the loss and the alignment
    #: (`symtrs.registry.depth_pairing`); `validate_alignment` refuses a mismatch.
    output_space = "depth"

    #: Preferred input size, 0 for "whatever you give it".
    native_input = 0

    #: Input dimensions must be a multiple of this. DPT-style models need 32.
    size_multiple = 32

    def __init__(
        self,
        loss: str = "",
        loss_kwargs: Optional[Dict] = None,
        align: str = "",
        freeze: str = "",
    ) -> None:
        super().__init__()
        loss_kwargs = dict(loss_kwargs or {})
        self.gradient_weight = float(loss_kwargs.pop("gradient_weight", 0.0))
        self.loss_name = loss or default_loss_for_protocol(self.protocol, self.output_space)
        self.criterion = build_depth_loss(self.loss_name, **loss_kwargs)
        self.align = align or self.default_align
        self.validate_alignment(self.align)
        self.freeze = freeze

    def validate_alignment(self, align: str) -> None:
        """Refuse an alignment that does not fit this model's output space.

        A MiDaS model scored with a depth-space alignment (`scale`, `median`, `scale_shift`) is
        scored on `1 / (s / z + t)`, which is not an affine function of depth, so the numbers
        look like a bad model (the 2026-10-17 audit found silog 41-111 against ~27 for a constant
        plane). The reverse - a disparity fit applied to something that already is depth - would
        invert metres. Both fail here, at build time and again at the start of an evaluation,
        rather than as a table that nobody can interpret.
        """
        disparity_alignment = align in DISPARITY_ALIGNMENTS
        if self.output_space == "disparity" and not disparity_alignment:
            raise ValueError(
                f"{type(self).__name__} outputs disparity (s/z + t), which only a disparity-space "
                f"fit can align; align={align!r} would score 1/(s/z + t) as if it were depth. "
                f"Use align='scale_shift_disparity' (the registry default) - see AGENTS.md rule 3."
            )
        if self.output_space != "disparity" and disparity_alignment:
            raise ValueError(
                f"{type(self).__name__} outputs depth, not disparity; align={align!r} would invert "
                f"it. Use 'none' (metric) or 'scale' (relative) - see AGENTS.md rule 3."
            )

    def scoring_input(self, prediction: torch.Tensor, align: str) -> torch.Tensor:
        """What `compute_depth_metrics` should be given for this alignment.

        Metres for every alignment except the disparity ones, which fit the raw disparity
        themselves: converting to metres first would throw away the sign and the shift they need
        (`to_metres` zeroes non-positive disparities).
        """
        if align in DISPARITY_ALIGNMENTS:
            return prediction.float()
        return self.to_metres(prediction)

    # ------------------------------------------------------------- preprocessing

    def preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        """ImageNet normalisation, which is what every one of these backbones expects."""
        from symtrs.data.transforms import normalize_image

        return normalize_image(image)

    def _pad_to_multiple(self, image: torch.Tensor) -> torch.Tensor:
        """Pad bottom/right so H and W are multiples of `size_multiple`.

        Reflection padding is used when the image is big enough for it, because a
        hard black border is a strong edge the network will read as geometry.
        """
        if self.size_multiple <= 1:
            return image
        height, width = image.shape[-2:]
        pad_h = (-height) % self.size_multiple
        pad_w = (-width) % self.size_multiple
        if pad_h == 0 and pad_w == 0:
            return image
        mode = "reflect" if (height > pad_h and width > pad_w) else "replicate"
        return F.pad(image, (0, pad_w, 0, pad_h), mode=mode)

    def _resize_like(self, prediction: torch.Tensor, size) -> torch.Tensor:
        """Bring a raw prediction back to the input's spatial size.

        Bilinear is right for a continuous depth field. If a model returns a 1-channel
        map at 1/4 resolution, this is the step that makes it comparable to the ground
        truth without changing what it means.
        """
        if prediction.dim() == 3:
            prediction = prediction.unsqueeze(1)
        if tuple(prediction.shape[-2:]) == tuple(size):
            return prediction
        return F.interpolate(prediction.float(), size=size, mode="bilinear", align_corners=False)

    # -------------------------------------------------------------------- target

    def prepare_target(self, depth_metres: torch.Tensor, valid: torch.Tensor) -> DepthTarget:
        """Metric ground truth to whatever this model's loss wants.

        Metric models consume metres directly. Relative models consume disparity
        (`1 / depth`), which is the space they were trained in and the space their
        scale-invariant loss assumes.
        """
        if depth_metres.dim() == 3:
            depth_metres = depth_metres.unsqueeze(1)
        if valid.dim() == 3:
            valid = valid.unsqueeze(1)

        depth_metres = depth_metres.float()
        valid = valid.bool() & torch.isfinite(depth_metres) & (depth_metres > 0)

        if self.protocol == "metric":
            return DepthTarget(values=depth_metres, mask=valid, unit="metres")
        if self.output_space == "depth":
            return DepthTarget(values=depth_metres, mask=valid, unit="relative_depth")

        disparity = torch.where(valid, 1.0 / depth_metres.clamp_min(EPS), torch.zeros_like(depth_metres))
        return DepthTarget(values=disparity, mask=valid, unit="normalized_disparity")

    def to_metres(self, prediction: torch.Tensor, stats: Optional[Dict] = None) -> torch.Tensor:
        """Raw model output to metres. Non-positive predictions become 0 and are
        filtered out by the metrics, since `1/x` of a negative disparity is a
        meaningless negative distance rather than a wrong one.

        For a disparity model this is `1 / disparity` with the unknown scale and shift still in
        it: it is NOT metres and is not what the evaluation scores (see `scoring_input`); it is
        what `predict` returns for qualitative use.
        """
        if self.protocol == "metric" or self.output_space == "depth":
            return prediction.clamp_min(0.0)

        disparity = prediction.float()
        depth = 1.0 / disparity.clamp_min(EPS)
        return torch.where(disparity > EPS, depth, torch.zeros_like(depth))

    # ---------------------------------------------------------------------- loss

    def compute_loss(self, prediction: torch.Tensor, target: DepthTarget):
        """Return `(total_loss, components)`. Components are logged separately so a
        diverging regulariser is visible rather than folded into one number."""
        loss = self.criterion(prediction, target.values, target.mask)
        components = {f"loss_{self.loss_name}": float(loss.detach())}

        if self.gradient_weight > 0:
            regulariser = gradient_loss(prediction, target.values, target.mask)
            components["loss_gradient"] = float(regulariser.detach())
            loss = loss + self.gradient_weight * regulariser

        return loss, components

    # ------------------------------------------------------------------- forward

    def forward_at_size(self, image: torch.Tensor) -> torch.Tensor:
        """Forward pass whose output is resized to match the input's spatial size.

        `image` is already preprocessed and padded: `symtrs.engine.loop.prepare_input`
        owns both steps and is the only caller. An adapter must not normalise here as
        well, and must not assume a raw `[0, 1]` batch - doing so silently double-applies
        the ImageNet transform and costs accuracy with no visible failure.
        """
        prediction = self.forward(image)
        return self._resize_like(prediction, image.shape[-2:])

    @torch.no_grad()
    def predict(self, image: torch.Tensor, tile_size: int = 0, tile_overlap: int = 0, **kwargs) -> torch.Tensor:
        """Raw `[B, 3, H, W]` in `[0, 1]` to depth in metres.

        Delegates to `symtrs.engine.loop.forward_native` so that the SDK, the training
        loop and the evaluation pass all reach the network the same way - including how
        padding and tiling are applied. `tile_size` enables sliding-window inference for
        images too large for one forward pass.
        """
        from symtrs.engine.loop import forward_native

        if image.dim() == 3:
            image = image.unsqueeze(0)
        was_training = self.training
        self.eval()
        prediction = forward_native(self, image, tile_size, tile_overlap)
        metres = self.to_metres(prediction)
        if was_training:
            self.train()
        return metres

    # ------------------------------------------------------------- optimisation

    def trainable_parameters(self, freeze: str = ""):
        self.apply_freeze(freeze or self.freeze)
        return (p for p in self.parameters() if p.requires_grad)

    def apply_freeze(self, freeze: str = "") -> None:
        freeze = freeze or self.freeze
        if not freeze:
            return
        apply_freeze_policy(self, freeze)
        print(f"[symtrs] freeze={freeze!r} -> {self.describe()}")
