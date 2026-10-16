"""MiDaS adapters, loaded from `torch.hub`.

MiDaS is the canonical relative-depth family: it predicts an affine-invariant
inverse depth (disparity), so it is normalised rather than metric, which is exactly
the split the benchmark asks for.

    midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

The hub returns the model and a companion module of input transforms. Both are
cached by torch.hub, so repeated construction is cheap.

MiDaS's own evaluation aligns predictions per image before measuring, so
`default_align` is `"scale"` - least-squares scale in depth space, which is the
convention used when MiDaS-family models are compared against each other.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

from symtrs.model.depth._hub import load_dropping_position_index
from symtrs.model.depth.base import DepthAdapter

#: Hub model name -> input size the checkpoint expects. The transform module also
#: encodes this, but recording it lets `symtrs doctor` report it before a download.
MIDAS_MODELS: Dict[str, int] = {
    "MiDaS_small": 256,
    "DPT_Hybrid": 384,
    "DPT_Large": 384,
    "DPT_SwinV2_T_256": 256,
    "DPT_SwinV2_B_384": 384,
    "DPT_SwinV2_L_384": 384,
    "DPT_SwinV2_B_512": 512,
    "DPT_BEiT_L_384": 384,
    "DPT_BEiT_L_512": 512,
}

#: Hub model name -> the transform entry point that checkpoint expects. MiDaS pairs one
#: transform with each backbone size, and loading the wrong one silently changes the
#: input statistics.
_TRANSFORM_NAMES = {
    "MiDaS_small": "small_transform",
    "DPT_Hybrid": "dpt_transform",
    "DPT_Large": "dpt_transform",
    # Upstream `hubconf.transforms()` has one entry per backbone input size. Swin-V2-T used
    # to be mapped to `dpt_transform` (384), which fed a 256-pixel checkpoint 384-pixel
    # images and failed in the first forward with "Input image height (384) doesn't match
    # model (256)" - the build succeeded, so only a forward pass shows it.
    "DPT_SwinV2_T_256": "swin256_transform",
    "DPT_SwinV2_B_384": "swin384_transform",
    "DPT_SwinV2_L_384": "swin384_transform",
    "DPT_SwinV2_B_512": "beit512_transform",
    "DPT_BEiT_L_384": "dpt_transform",
    "DPT_BEiT_L_512": "beit512_transform",
}


class MiDaSAdapter(DepthAdapter):
    """A MiDaS checkpoint behind the common depth interface."""

    protocol = "relative"
    default_align = "scale"
    size_multiple = 32
    # timm and cv2 are load-time requirements of the published checkpoints, not of this
    # adapter: the hub repo's `hubconf.py` reaches `import timm` through `midas.blocks` ->
    # `backbones.beit`, and its `transforms()` entry point - which we call, so that the
    # normalisation matches the checkpoint - imports cv2. The upstream `hubconf.py` declares
    # only `dependencies = ["torch"]`, so nothing warns you until the load fails.
    requires = ("torch", "timm", "cv2")

    def __init__(
        self,
        hub_model: str = "DPT_Hybrid",
        pretrained: bool = True,
        loss: str = "",
        loss_kwargs: Optional[Dict] = None,
        align: str = "",
        freeze: str = "",
    ) -> None:
        super().__init__(loss=loss, loss_kwargs=loss_kwargs, align=align, freeze=freeze)

        if hub_model not in MIDAS_MODELS:
            raise ValueError(
                f"unknown MiDaS model {hub_model!r}; expected one of {sorted(MIDAS_MODELS)}"
            )
        self.hub_model = hub_model
        self.native_input = MIDAS_MODELS[hub_model]
        self.key = f"midas_{hub_model.lower()}"

        self.backbone = self._load_backbone(hub_model, pretrained)
        self.transform = self._load_transform(hub_model)

    @staticmethod
    def _load_backbone(hub_model: str, pretrained: bool):
        try:
            import torch as _torch
        except ImportError as exc:  # pragma: no cover - torch is a hard dependency
            raise ImportError("MiDaS requires torch") from exc

        # The BEiT-based variants (DPT_BEiT_L_384, DPT_BEiT_L_512) store
        # `relative_position_index` in their released weights, and the hub entry point runs
        # its own `load_state_dict`, so a newer timm that no longer persists that buffer makes
        # the load fail from inside `torch.hub`. See `symtrs.model.depth._hub`.
        try:
            with load_dropping_position_index():
                return _torch.hub.load(
                    "intel-isl/MiDaS", hub_model, pretrained=pretrained, trust_repo=True
                )
        except RuntimeError as exc:
            # The Swin-V2 checkpoints predate timm's move of the patch-merging
            # (`downsample.reduction`) between stages, so on timm >= 0.9 the load reports
            # it missing. Renaming keys by hand would be a guess about which stage owns
            # the weights; the environment MiDaS v3.1 was released against is the fix.
            if "SwinV2" in hub_model and "downsample.reduction" in str(exc):
                raise RuntimeError(
                    f"{hub_model} does not load on this timm: its checkpoint was written "
                    "for timm==0.6.12 (Swin-V2 patch-merging layout). Run it from the "
                    "second environment (`.venv-zoe`, timm==0.6.12); `symtrs doctor` "
                    f"reports the installed timm. Original error: {exc}"
                ) from exc
            raise

    @staticmethod
    def _load_transform(hub_model: str):
        import torch as _torch

        transforms = _torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        name = _TRANSFORM_NAMES.get(hub_model, "dpt_transform")
        if not hasattr(transforms, name):
            available = [a for a in dir(transforms) if a.endswith("transform")]
            raise AttributeError(
                f"the MiDaS transforms module has no {name!r}; available: {available}"
            )
        return getattr(transforms, name)

    def preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        """The hub transform operates on a uint8 HWC numpy image and does its own
        resizing and normalisation, so the tensor has to make a round trip. Doing it
        this way rather than reimplementing the normalisation keeps the adapter
        faithful to the checkpoint."""
        if image.dim() == 3:
            image = image.unsqueeze(0)

        arrays = []
        for index in range(image.shape[0]):
            array = (
                image[index].detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0
            )
            arrays.append(array.round().astype(np.uint8))

        # The transform is per-image, so the batch is reassembled afterwards.
        transformed = [self._as_batched(self.transform(array)) for array in arrays]
        return torch.cat(transformed, dim=0).to(image.device)

    @staticmethod
    def _as_batched(output) -> torch.Tensor:
        """A hub transform's output as `[1, 3, H, W]`.

        Every MiDaS transform ends in `PrepareForNet`, which adds the batch axis itself, so
        one call on one image already returns `[1, 3, H, W]`. Reading that as unbatched and
        stacking a list of them gives `[B, 1, 3, H, W]` - a five-dimensional input that
        `conv2d` rejects several frames later, where nothing points back at this line. The
        3-D shape is still accepted so that a transform which does not batch works too.
        """
        if not isinstance(output, torch.Tensor):
            output = torch.as_tensor(output)
        return output.unsqueeze(0) if output.dim() == 3 else output

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Raw disparity prediction at the network's own output resolution."""
        prediction = self.backbone(image)
        if isinstance(prediction, dict):
            for key in ("predicted_depth", "depth", "out"):
                if key in prediction:
                    prediction = prediction[key]
                    break
            else:
                raise RuntimeError(
                    f"MiDaS forward returned a dict with unknown keys {sorted(prediction)}"
                )
        if prediction.dim() == 3:
            prediction = prediction.unsqueeze(1)
        return prediction
