"""Apple Depth Pro.

Depth Pro is the benchmark's metric anchor: a ViT-L encoder and a multi-scale
convolutional decoder that predict metric depth with absolute scale, without camera
intrinsics, at 1536x1536. It is also by far the largest model here - roughly 1.9GB of
weights - so it is meant to be used with a `freeze` policy ("head" or "last_n:<k>")
rather than finetuned end to end. `symtrs doctor` prints the trainable parameter count
via `describe()`, which is how you check that a freeze policy actually did something
rather than silently matching everything or nothing.

Metric, in metres
-----------------
`protocol = "metric"` and `default_align = "none"`: the released `infer` rescales the
canonical inverse depth by the predicted focal length and inverts it, so the number it
returns is in metres and needs no alignment against a metric ground truth.

Loading
-------
The released repository has no `hubconf.py`, so the `torch.hub.load("apple/ml-depth-pro",
"depth_pro")` call in the task brief cannot resolve against it as published. This
adapter attempts the hub entrypoint anyway (a fork that adds one will work), falls back
to the package's own `depth_pro.create_model_and_transforms()`, and prints a `[symtrs]`
note when it does. The transform always comes from `create_model_and_transforms`, which
is the only public source of it: `ToTensor`, normalise with mean = scale = 0.5, then cast
to the requested precision - it does not resize, because `DepthPro.infer` resizes to the
network's own `img_size` internally and maps the result back to the input resolution.
That function builds a model as well as a transform, so a fork that did provide the hub
entrypoint would end up with two copies of the 1.9GB weights; the released repository
always takes the fallback path, which loads them once.

Memory, and why it cannot be shrunk from the outside
----------------------------------------------------
`DepthPro.forward` asserts a 1536x1536 input and `infer` resizes everything to it, so
the work per image is fixed by the architecture: the patch encoder's 384px position
embedding is tiled 4x4 into 35 overlapping ViT-L passes. A smaller crop or a lower
`resolution` therefore does not make a step cheaper (the configs run batch 1 and
uncropped at every resolution); only the batch and the freeze policy do. Measured on a
12GB RTX 5070 Ti Laptop, amp on, batch 1: one training step peaks at ~7.9GiB with
`freeze: encoder` (3.6GiB of fp32 weights, the rest decoder activations; the frozen ViTs
build no autograd graph because nothing upstream of them requires grad), and runs out of
memory above 11GiB without it. Scoring the native 2048 frame in one pass peaks at
~5.7GiB. `symtrs.engine.trainer.apply_run_freeze` is what makes the policy real; it was
once only stored and never applied.

Gradients
---------
`DepthPro.infer` is decorated with `@torch.no_grad()`, which would make the model
untrainable, so `forward` calls the undecorated function whenever gradients are needed.
The decorator is a thin grad-mode wrapper around the same arithmetic
(`functools.wraps` leaves the raw function on `__wrapped__`), so this changes only
whether the graph is recorded, never the numbers. If `__wrapped__` is absent the
decorated method is used and a missing gradient surfaces as a hard autograd error at
`backward()` rather than as a model that quietly never learns.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from symtrs.model.depth.base import DepthAdapter

#: Keys the depth tensor may be carried under. The released `infer` uses "depth"; note
#: that its focal-length entry is spelled "focallength_px", not "focal_length", and is
#: not consumed here.
_DEPTH_KEYS = ("depth", "predicted_depth", "metric_depth", "out")

#: The network's trained-against input size. `infer` upsamples to this internally, so it
#: is metadata about the checkpoint rather than a requirement on the caller.
_NATIVE_SIZE = 1536


def _as_depth_tensor(result: Any, context: str) -> torch.Tensor:
    """Pull a `[1, 1, h, w]` tensor out of whatever an inference call returned.

    `Mapping` and not `dict` for the same reason as elsewhere in this package: a
    `UserDict`-style return value is a mapping without being a `dict`.
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
    elif isinstance(result, (tuple, list)):
        tensors = [item for item in result if torch.is_tensor(item)]
        if not tensors:
            raise ValueError(
                f"{context} returned a {len(result)}-element sequence "
                f"{[type(item).__name__ for item in result]} containing no tensor"
            )
        result = tensors[0]
    if not torch.is_tensor(result):
        raise TypeError(f"{context} returned {type(result).__name__}, not a tensor")
    if result.dim() == 2:
        result = result.unsqueeze(0).unsqueeze(0)
    elif result.dim() == 3:
        result = result.unsqueeze(1)
    return result


class DepthProAdapter(DepthAdapter):
    """Apple Depth Pro behind the common depth interface."""

    requires = ("torch", "depth_pro")
    protocol = "metric"
    default_align = "none"
    native_input = _NATIVE_SIZE
    size_multiple = 32
    key = "depth_pro"

    def __init__(
        self,
        f_px: Optional[float] = None,
        pretrained: bool = True,
        loss: str = "",
        loss_kwargs: Optional[Dict] = None,
        align: str = "",
        freeze: str = "",
    ) -> None:
        """Args:
            f_px: focal length in pixels. None (default) lets the model's own field-of-view
                head estimate it per image, which is the intended zero-shot behaviour and
                the only option when camera metadata is unavailable, as it is for aerial
                imagery. Supply it only if the intrinsics are known, because the predicted
                metric scale is directly proportional to it.
            pretrained: Depth Pro only ships trained weights, so False is rejected rather
                than silently ignored.
        """
        if not pretrained:
            raise ValueError(
                "Depth Pro only ships trained weights; pretrained=False has no meaning "
                "here. Omit it or pass True."
            )
        try:
            import depth_pro
        except ImportError as exc:
            raise ImportError(
                "DepthProAdapter needs Apple's depth_pro package: "
                "pip install git+https://github.com/apple/ml-depth-pro.git"
            ) from exc

        self.f_px = None if f_px is None else float(f_px)
        super().__init__(loss=loss, loss_kwargs=loss_kwargs, align=align, freeze=freeze)

        self.model, self.transform = self._load(depth_pro)

    # -------------------------------------------------------------------- loading

    def _load(self, depth_pro: Any) -> Tuple[Any, Any]:
        """The model and its input transform, from whichever source is available."""
        model = None
        transform = None
        try:
            loaded = torch.hub.load("apple/ml-depth-pro", "depth_pro", trust_repo=True)
        except Exception as exc:
            print(
                f"[symtrs] note: torch.hub.load('apple/ml-depth-pro', 'depth_pro') is not "
                f"available ({type(exc).__name__}: {exc}); using "
                f"depth_pro.create_model_and_transforms() instead."
            )
        else:
            # A hub entrypoint could reasonably return either the model or the (model,
            # transform) pair, so accept both rather than assuming.
            if isinstance(loaded, (tuple, list)) and len(loaded) == 2:
                model, transform = loaded
            else:
                model = loaded

        if model is None or transform is None:
            package_model, package_transform = depth_pro.create_model_and_transforms()
            model = model if model is not None else package_model
            transform = transform if transform is not None else package_transform
        return model, transform

    # ------------------------------------------------------------- preprocessing

    def preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        """The transform from `create_model_and_transforms`, applied per image.

        It takes a uint8 HWC array, so the tensor makes the same round trip the MiDaS
        adapter uses. Writing the normalisation out by hand instead would be a guess,
        and the model is sensitive to it: it is a plain `(x - 0.5) / 0.5`.
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)

        tensors = []
        for index in range(image.shape[0]):
            array = (
                image[index].detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0
            )
            tensors.append(self.transform(array.round().astype(np.uint8)))
        return torch.stack(tensors, dim=0).to(image.device)

    # ------------------------------------------------------------------- forward

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Depth in metres at the input's spatial size.

        `model(x)` cannot be used here: `DepthPro.forward` returns the *canonical inverse*
        depth, and the focal-length scaling that turns it into metres lives only in
        `infer`. Reading `model(x)` would hand the metrics an un-scaled inverse depth that
        still looks like a plausible positive map, so the adapter calls `infer` and does
        not fall back to the raw forward.
        """
        infer = self._infer_callable()
        if infer is None:
            raise self._diagnosis(
                AttributeError("the loaded module has no infer() method, so metric depth "
                               "cannot be obtained")
            )

        # `infer` ends with `f_px.squeeze()`, so a supplied focal length has to be a
        # tensor already, and it divides the prediction by it, so it has to live on the
        # image's device.
        f_px = (
            None
            if self.f_px is None
            else torch.as_tensor(self.f_px, dtype=torch.float32, device=image.device)
        )

        # `infer` ends in `depth.squeeze()`, which collapses a batch dimension, so it is
        # called one image at a time and the batch is reassembled here.
        depths = []
        for index in range(image.shape[0]):
            single = image[index : index + 1]
            try:
                result = infer(single) if f_px is None else infer(single, f_px=f_px)
            except Exception as exc:
                raise self._diagnosis(exc) from exc
            depths.append(_as_depth_tensor(result, "DepthPro.infer"))
        return torch.cat(depths, dim=0)

    def _infer_callable(self):
        """`model.infer`, with `@torch.no_grad()` stripped when gradients are needed.

        See "Gradients" in the module docstring: the arithmetic is identical, only the
        grad mode differs. Returns None if the module has no `infer` at all.
        """
        bound = getattr(self.model, "infer", None)
        if bound is None:
            return None
        needs_grad = torch.is_grad_enabled() and any(
            parameter.requires_grad for parameter in self.parameters()
        )
        if not needs_grad:
            return bound
        raw = getattr(bound, "__wrapped__", None)
        if raw is None:
            return bound
        return raw.__get__(self.model, type(self.model))

    def _diagnosis(self, exc: Exception) -> RuntimeError:
        """A RuntimeError that names the module's public API and one command to run."""
        attributes = [name for name in dir(self.model) if not name.startswith("_")]
        return RuntimeError(
            f"cannot read a metric depth tensor from {type(self.model).__module__}."
            f"{type(self.model).__name__} (key {self.key!r}). "
            f"Underlying error: {type(exc).__name__}: {exc}. "
            f"Expected infer() to return a mapping with one of the keys {list(_DEPTH_KEYS)}. "
            f"Public attributes available on the model: {attributes}. "
            f"Run `symtrs doctor --model {self.key}` and report its output."
        )
