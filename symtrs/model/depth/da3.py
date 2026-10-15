"""Depth Anything 3 through `transformers`.

Depth Anything 3 is distributed as a Hugging Face checkpoint and, unlike the other
depth adapters, ships its own preprocessing: the model's `AutoImageProcessor` owns the
resize, the normalisation and the size divisibility, and the model's own weights decide
what the output means. So this adapter delegates all of that to the hub objects rather
than reimplementing them.

Two conventions have to be decided per checkpoint, and the model id is the only
information available before the checkpoint is downloaded:

    protocol        "metric" if the checkpoint was trained with metric supervision
                    (millimetres/metres), otherwise "relative".
    default_align   "none" for metric; "scale" for relative, the convention the Depth
                    Anything evaluations use.

`metric` infers that from the model id - an id containing "metric" (case-insensitive) is
taken to be metric. That inference is a *guess* keyed on the naming convention of the
released checkpoints (`depth-anything/DA3METRIC-LARGE` is metric,
`depth-anything/DA3-LARGE` and `depth-anything/DA3MONO-LARGE` are not), and it is a guess
about the checkpoint, not about the architecture. Pass `metric=True` or `metric=False`
explicitly to overrule it. Getting it wrong is not silently absorbed: a metric model
marked relative is trained against `1 / depth` instead of `depth`, so the loss will be
visibly wrong rather than merely suboptimal.

Depth, not disparity
--------------------
The base adapter assumes a relative model emits disparity, so it converts the target to
`1 / depth` and inverts the prediction in `to_metres`. Depth Anything emits *depth*:
its `predicted_depth` grows with distance, and the relative checkpoints are
scale-invariant in depth space, not in disparity space. This adapter therefore keeps a
relative model's target and prediction in depth space (see `prepare_target` and
`to_metres`), the same override the ZoeDepth-N adapter makes and for the same reason.

Sizing
------
The processor resizes and pads to whatever divisibility the backbone needs, so
`size_multiple` is left at 1: the inherited `_pad_to_multiple` must not re-pad an
already-prepared tensor, because an extra reflect border would change the image the
network sees. `native_input` is 0 for the same reason - the processor's own default
resolution is the model's preferred resolution, and it is not knowable without the
checkpoint.

Unverified: the `transformers` route
------------------------------------
Loading through `AutoModelForDepthEstimation` / `AutoImageProcessor` is the API the task
brief specifies, but it could not be exercised without `transformers` installed and it
does not match what is published on the Hub: the `depth-anything/DA3-*` repositories are
tagged `library_name: depth-anything-3`, and at least one of them lists only
`config.json` and `model.safetensors`, i.e. no image-processor config. If that is true of
the checkpoint you ask for, construction fails with a `RuntimeError` that says so,
rather than falling back to an invented normalisation. Verify with
`symtrs doctor --model da3_large`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Dict, Optional

import numpy as np
import torch

from symtrs.model.base import DepthTarget
from symtrs.model.depth.base import DepthAdapter

#: Keys a `transformers` depth output may carry the tensor under, in priority order.
_DEPTH_KEYS = ("predicted_depth", "depth", "out")

#: Fallback checkpoint when the caller does not name one.
DEFAULT_DA3_MODEL = "depth-anything/DA3-LARGE"


def _slug_model_id(model_id: str) -> str:
    """A registry-safe key fragment for a Hugging Face model id.

    The family token is folded into the "da3_" prefix, so `depth-anything/DA3-LARGE`
    becomes `da3_large` and `depth-anything/DA3METRIC-LARGE` becomes `da3_metric_large`,
    matching the keys in `symtrs.registry` instead of doubling the family name.
    """
    name = model_id.rstrip("/").rsplit("/", 1)[-1].lower()
    if name.startswith("da3"):
        name = name[3:]
    slug = re.sub(r"[^0-9a-z]+", "_", name).strip("_")
    return slug or "model"


def _infer_metric(model_id: str) -> bool:
    """Guess the metric/relative split from the checkpoint name."""
    return "metric" in model_id.lower()


def _as_depth_tensor(result: Any, context: str) -> torch.Tensor:
    """Pull a `[B, 1, h, w]` tensor out of a `transformers` model output.

    `ModelOutput` is an `OrderedDict` and `BatchFeature` is a `UserDict`, so the mapping
    branch has to test `Mapping` rather than `dict`: only the first of those two is a
    `dict` subclass, and keying on `dict` would silently miss every processor output.
    The attribute branch covers the `outputs.predicted_depth` access style that the
    model docstrings use.
    """
    if isinstance(result, Mapping):
        for key in _DEPTH_KEYS:
            if key in result:
                result = result[key]
                break
        else:
            raise ValueError(
                f"{context} returned a mapping with keys {sorted(result)}, none of "
                f"{list(_DEPTH_KEYS)}"
            )
    elif not torch.is_tensor(result):
        for key in _DEPTH_KEYS:
            value = getattr(result, key, None)
            if torch.is_tensor(value):
                result = value
                break
    if not torch.is_tensor(result):
        raise TypeError(
            f"{context} returned {type(result).__name__}; expected a tensor or a mapping "
            f"with one of the keys {list(_DEPTH_KEYS)}"
        )
    if result.dim() == 3:
        result = result.unsqueeze(1)
    elif result.dim() == 2:
        result = result.unsqueeze(0).unsqueeze(0)
    return result


def _depth_target(depth_metres: torch.Tensor, valid: torch.Tensor) -> DepthTarget:
    """A depth-space target for a relative model whose output is depth, not disparity."""
    if depth_metres.dim() == 3:
        depth_metres = depth_metres.unsqueeze(1)
    if valid.dim() == 3:
        valid = valid.unsqueeze(1)
    depth_metres = depth_metres.float()
    valid = valid.bool() & torch.isfinite(depth_metres) & (depth_metres > 0)
    return DepthTarget(values=depth_metres, mask=valid, unit="relative_depth")


class DepthAnything3Adapter(DepthAdapter):
    """A Depth Anything 3 checkpoint behind the common depth interface."""

    requires = ("torch", "transformers")
    size_multiple = 1
    native_input = 0

    # Defaults describing the release's non-metric checkpoints; `__init__` replaces them
    # when the model id (or an explicit `metric`) says otherwise.
    protocol = "relative"
    default_align = "scale"

    def __init__(
        self,
        model_id: str = DEFAULT_DA3_MODEL,
        metric: Optional[bool] = None,
        pretrained: bool = True,
        loss: str = "",
        loss_kwargs: Optional[Dict] = None,
        align: str = "",
        freeze: str = "",
    ) -> None:
        """Args:
            model_id: any `AutoModelForDepthEstimation`-compatible Depth Anything 3 id.
            metric: True for a metric checkpoint, False for an affine-invariant one.
                None (default) infers it from `model_id`; see the module docstring for
                why that inference is a guess and how to overrule it.
            pretrained: Depth Anything 3 has no random-init mode, so False is rejected
                rather than silently ignored.
        """
        if not pretrained:
            raise ValueError(
                "Depth Anything 3 only ships trained weights; pretrained=False has no "
                "meaning here. Omit it or pass True."
            )
        try:
            import transformers
        except ImportError as exc:
            raise ImportError(
                "DepthAnything3Adapter needs the `transformers` package: "
                "pip install transformers"
            ) from exc

        self.model_id = model_id
        self.metric = _infer_metric(model_id) if metric is None else bool(metric)

        # DepthAdapter.__init__ reads both of these, so they must be set first.
        self.protocol = "metric" if self.metric else "relative"
        self.default_align = "none" if self.metric else "scale"
        self.key = "da3_" + _slug_model_id(model_id)
        super().__init__(loss=loss, loss_kwargs=loss_kwargs, align=align, freeze=freeze)

        self.processor = self._load_processor(transformers, model_id)
        self.model = self._load_model(transformers, model_id)

    # -------------------------------------------------------------------- loading

    def _load_processor(self, transformers: Any, model_id: str):
        """The checkpoint's `AutoImageProcessor`.

        Wrapped because this is the step that is least likely to be present: the
        Depth Anything 3 repositories on the Hub are tagged `library_name:
        depth-anything-3` and some publish only `config.json` and `model.safetensors`,
        with no image-processor config at all. Guessing a substitute normalisation would
        be silently wrong, so the failure is made explicit instead.
        """
        try:
            return transformers.AutoImageProcessor.from_pretrained(model_id)
        except Exception as exc:
            raise RuntimeError(
                f"no image processor could be loaded for Depth Anything 3 checkpoint "
                f"{model_id!r} ({type(exc).__name__}: {exc}). This adapter needs both an "
                f"`AutoModelForDepthEstimation` config and an `AutoImageProcessor` config "
                f"in the repository; a checkpoints-only repository cannot be used through "
                f"`transformers`. Run `symtrs doctor --model {self.key}` and report its output."
            ) from exc

    def _load_model(self, transformers: Any, model_id: str):
        try:
            return transformers.AutoModelForDepthEstimation.from_pretrained(model_id)
        except Exception as exc:
            raise RuntimeError(
                f"could not load Depth Anything 3 checkpoint {model_id!r} with "
                f"AutoModelForDepthEstimation ({type(exc).__name__}: {exc}). "
                f"Run `symtrs doctor --model {self.key}` and report its output."
            ) from exc

    # ------------------------------------------------------------- preprocessing

    def preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        """The checkpoint's own `AutoImageProcessor`, which is the model's input contract.

        Reimplementing it would be a guess about the normalisation the weights were
        trained with, so the tensor makes a round trip through a uint8 HWC array the way
        the processor expects - the same discipline as the MiDaS adapter.
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)

        arrays = []
        for index in range(image.shape[0]):
            array = (
                image[index].detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0
            )
            arrays.append(array.round().astype(np.uint8))

        processed = self.processor(images=arrays, return_tensors="pt")
        if isinstance(processed, Mapping):
            pixel_values = processed.get("pixel_values")
        else:
            pixel_values = processed if torch.is_tensor(processed) else None
        if pixel_values is None:
            raise RuntimeError(
                f"{type(self.processor).__name__} returned {type(processed).__name__} "
                f"without a 'pixel_values' entry; Depth Anything 3 needs it. "
                f"Run `symtrs doctor --model {self.key}` and report its output."
            )
        return pixel_values.to(image.device)

    # -------------------------------------------------------------------- target

    def prepare_target(self, depth_metres: torch.Tensor, valid: torch.Tensor) -> DepthTarget:
        """Metric checkpoints pass through; relative ones stay in depth space."""
        if self.protocol == "metric":
            return super().prepare_target(depth_metres, valid)
        return _depth_target(depth_metres, valid)

    def to_metres(self, prediction: torch.Tensor, stats: Optional[Dict] = None) -> torch.Tensor:
        """Metric checkpoints clamp to metres; relative ones return depth unchanged.

        The relative result is still defined only up to a scale, so `default_align =
        "scale"` is what makes it comparable to the ground truth.
        """
        if self.protocol == "metric":
            return super().to_metres(prediction, stats)
        return prediction.float().clamp_min(0.0)

    # ------------------------------------------------------------------- forward

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Raw depth prediction at the processor's output resolution."""
        try:
            try:
                outputs = self.model(pixel_values=image)
            except TypeError:
                # A few depth heads take the tensor positionally only.
                outputs = self.model(image)
        except Exception as exc:
            raise RuntimeError(
                f"calling {type(self.model).__module__}.{type(self.model).__name__} "
                f"({self.model_id!r}) failed: {type(exc).__name__}: {exc}. "
                f"Run `symtrs doctor --model {self.key}` and report its output."
            ) from exc
        return _as_depth_tensor(outputs, f"{type(self.model).__name__}.forward")
