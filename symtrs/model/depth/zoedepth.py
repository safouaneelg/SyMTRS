"""ZoeDepth adapters, loaded from `torch.hub`.

ZoeDepth bolts a metric bin head onto a frozen MiDaS DPT-BEiT-Large core:

    model = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True)

Three checkpoints ship in one release and they are not interchangeable:

    ZoeD_N   trained on 12 datasets with no metric supervision -> affine-invariant
             depth, so `protocol="relative"` and `default_align="scale"`.
    ZoeD_K   trained with KITTI metric supervision -> depth in metres, so
             `protocol="metric"` and `default_align="none"`.
    ZoeD_NK  NYU and KITTI metric heads behind a learned domain router; metric like
             ZoeD_K, with one bin configuration per domain.

Why ZoeD_N is relative but still emits depth, not disparity
-----------------------------------------------------------
The MiDaS convention - and the base adapter's default for relative models - is that a
relative model emits *disparity*, so the target becomes `1 / depth` and `to_metres`
inverts the prediction. ZoeDepth does not follow that convention. Its head is a softmax
over `n_bins` bin *centres* that live in depth space, and the prediction is their
expectation, `out = sum(prob * bin_center)`, computed in
`zoedepth/models/zoedepth/zoedepth_v1.py`. ZoeD_N simply never received metric
supervision, so what comes out is depth up to a positive affine transform: it is
scale-invariant in *depth* space, not in disparity space. This adapter therefore
overrides `prepare_target` and `to_metres` to keep both sides in depth space. Feeding
the head `1 / depth` would ask it to represent a quantity its bin centres cannot
express, and the loss would be small only by accident.

Preprocessing is a no-op here
-----------------------------
The hub entrypoints return the model only; the release ships no companion transform
module. It does not need one, because the model preprocesses inside itself:
`MidasCore.forward` applies `PrepForMidas`, which resizes (min side 384, aspect kept,
rounded up to a multiple of 32) and normalises with `mean = std = [0.5, 0.5, 0.5]`
(`zoedepth/models/base_models/midas.py`). `DepthModel.infer_pil` hands it a plain
`ToTensor()` image, i.e. `[0, 1]`, with no ImageNet normalisation. So
`preprocess_image` is the identity, and inheriting the base class default (ImageNet
normalisation) would double-normalise and be silently wrong.

The hub object's `forward` returns a dict whose key set differs between the N/K and NK
classes, so `forward` accepts a plain tensor, a tuple or a dict and, when none of them
fits, raises a `RuntimeError` naming the module type and its public attributes. That is
deliberate: it turns an unknown third-party API into one `symtrs doctor` command
instead of a stack trace.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Dict, Optional

import torch

from symtrs.model.base import DepthTarget
from symtrs.model.depth._hub import load_dropping_position_index
from symtrs.model.depth.base import DepthAdapter

#: variant -> (hub entrypoint, protocol, default alignment).
_VARIANTS = {
    "N": ("ZoeD_N", "relative", "scale"),
    "K": ("ZoeD_K", "metric", "none"),
    "NK": ("ZoeD_NK", "metric", "none"),
}

#: Keys a ZoeDepth forward result may carry the depth tensor under, in priority order.
#: `metric_depth` is what both shipped classes actually use; the others are accepted so
#: that a renamed key fails loudly only if it is genuinely unrecognisable.
_DEPTH_KEYS = ("metric_depth", "out", "predicted_depth", "depth")

#: Internal resolution of the shared MiDaS DPT-BEiT-Large core: min side, aspect kept.
_CORE_SIZE = 384


def _as_depth_tensor(result, context: str) -> torch.Tensor:
    """Pull a `[B, 1, h, w]` tensor out of whatever a forward call returned.

    `Mapping` rather than `dict` so that a `UserDict`-style result is not silently
    mistaken for a bare object.
    """
    if isinstance(result, Mapping):
        for key in _DEPTH_KEYS:
            if key in result:
                result = result[key]
                break
        else:
            raise ValueError(
                f"{context} returned a dict with keys {sorted(result)}, none of "
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
    if result.dim() == 3:
        result = result.unsqueeze(1)
    return result


def _depth_target(depth_metres: torch.Tensor, valid: torch.Tensor) -> DepthTarget:
    """A depth-space target for a relative model whose output is depth, not disparity.

    `unit` is recorded in the checkpoint; "relative_depth" is deliberately distinct
    from both "metres" and "normalized_disparity" so a reloaded model cannot be confused
    about which space its weights were trained in.
    """
    if depth_metres.dim() == 3:
        depth_metres = depth_metres.unsqueeze(1)
    if valid.dim() == 3:
        valid = valid.unsqueeze(1)
    depth_metres = depth_metres.float()
    valid = valid.bool() & torch.isfinite(depth_metres) & (depth_metres > 0)
    return DepthTarget(values=depth_metres, mask=valid, unit="relative_depth")


class ZoeDepthAdapter(DepthAdapter):
    """ZoeDepth-N, ZoeDepth-K or ZoeDepth-NK behind the common depth interface."""

    # timm, not cv2. The ZoeDepth hub entrypoint builds a MiDaS DPT-BEiT-Large core, and
    # reaching that core imports `midas.blocks` -> `backbones.beit`, which does `import timm`
    # at module level - so timm is needed to construct the model at all, not only to load our
    # own encoders. cv2 is deliberately *not* here: it is reached only through MiDaS's
    # `transforms()` entrypoint, which this adapter never calls (the model normalises its own
    # input, see `preprocess_image`). Measured: a pretrained ZoeD_N forward pass runs with cv2
    # blocked in `sys.meta_path`.
    requires = ("torch", "timm")
    size_multiple = 32

    # Family-level fallback (the relative variant); `__init__` overwrites `protocol` and
    # `default_align` from `variant`. Declared so the class can be described without
    # downloading a checkpoint.
    protocol = "relative"
    default_align = "scale"
    native_input = _CORE_SIZE

    def __init__(
        self,
        variant: str = "NK",
        pretrained: bool = True,
        domain_id: int = 0,
        loss: str = "",
        loss_kwargs: Optional[Dict] = None,
        align: str = "",
        freeze: str = "",
    ) -> None:
        """Args:
            variant: "N" (relative), "K" (metric, KITTI) or "NK" (metric, NYU + KITTI).
            pretrained: load the released weights; False is a random-init debugging aid.
            domain_id: only meaningful for ZoeD_NK. The released NK model routes between
                its NYU and KITTI heads with a learned classifier and its `forward` has
                no domain parameter, so this value is forwarded only when the loaded
                model's signature actually names a matching argument (see
                `_domain_kwarg`); otherwise it is inert and the router decides.
        """
        variant = variant.upper()
        if variant not in _VARIANTS:
            raise ValueError(
                f"unknown ZoeDepth variant {variant!r}; expected one of {sorted(_VARIANTS)}"
            )
        entrypoint, protocol, default_align = _VARIANTS[variant]

        # DepthAdapter.__init__ reads `protocol` and `default_align`, so they have to be
        # in place before super() runs.
        self.variant = variant
        self.protocol = protocol
        self.default_align = default_align
        self.native_input = _CORE_SIZE
        self.key = f"zoedepth_{variant.lower()}"
        self.domain_id = int(domain_id)
        super().__init__(loss=loss, loss_kwargs=loss_kwargs, align=align, freeze=freeze)

        self.model = self._load(entrypoint, pretrained)
        self._domain_kwarg = self._find_domain_kwarg()

    # -------------------------------------------------------------------- loading

    @staticmethod
    def _load(entrypoint: str, pretrained: bool):
        try:
            # The hub entry point does its own `load_state_dict`, so a released checkpoint
            # whose `relative_position_index` buffers a newer timm no longer persists can only
            # be relaxed from outside it. See `symtrs.model.depth._hub`.
            with load_dropping_position_index():
                return torch.hub.load(
                    "isl-org/ZoeDepth", entrypoint, pretrained=pretrained, trust_repo=True
                )
        except Exception as exc:
            raise RuntimeError(
                f"torch.hub.load('isl-org/ZoeDepth', {entrypoint!r}, pretrained={pretrained}) "
                f"failed: {type(exc).__name__}: {exc}. The entrypoints used by this adapter "
                f"are ZoeD_N, ZoeD_K and ZoeD_NK (see the repo's hubconf.py). If the release "
                f"renamed them or the download is blocked, run `symtrs doctor --model "
                f"zoedepth_{entrypoint.rsplit('_', 1)[-1].lower()}` and report its output."
            ) from exc

    def _find_domain_kwarg(self) -> Optional[str]:
        """The forward parameter that takes a domain id, if the loaded model has one.

        ZoeD_NK's released `forward` has neither a `dataset` nor a `domain` parameter -
        it learns the domain internally - but it does take `**kwargs`, which would
        swallow a stray keyword without complaining. Matching on parameter *names* is
        therefore what keeps a typo or a stale argument from being silently ignored.
        """
        try:
            parameters = inspect.signature(self.model.forward).parameters
        except (TypeError, ValueError):  # pragma: no cover - builtins and C++ modules
            return None
        for name in ("dataset", "domain", "domain_id"):
            if name in parameters:
                return name
        return None

    # ------------------------------------------------------------- preprocessing

    def preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        """Identity: `MidasCore` applies its own resize and 0.5 normalisation.

        See the module docstring. Overriding the base class here is not cosmetic - the
        inherited ImageNet normalisation would be applied on top of the model's own
        normalisation and quietly halve the effective contrast of every input.
        """
        return image

    # -------------------------------------------------------------------- target

    def prepare_target(self, depth_metres: torch.Tensor, valid: torch.Tensor) -> DepthTarget:
        """Metric variants pass through; ZoeD_N keeps the target in depth space.

        See "Why ZoeD_N is relative but still emits depth" in the module docstring.
        """
        if self.protocol == "metric":
            return super().prepare_target(depth_metres, valid)
        return _depth_target(depth_metres, valid)

    def to_metres(self, prediction: torch.Tensor, stats: Optional[Dict] = None) -> torch.Tensor:
        """Metric variants clamp; ZoeD_N's depth output is returned as-is.

        The result for a relative model is still only defined up to a scale, so the
        engine's per-image alignment is what makes it comparable to the ground truth -
        exactly as for the MiDaS adapters, just in depth space rather than disparity
        space.
        """
        if self.protocol == "metric":
            return super().to_metres(prediction, stats)
        return prediction.float().clamp_min(0.0)

    # ------------------------------------------------------------------- forward

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Raw prediction at the network's own output resolution."""
        kwargs = {}
        if self._domain_kwarg is not None:
            kwargs[self._domain_kwarg] = self.domain_id
        try:
            result = self.model(image, **kwargs)
            return _as_depth_tensor(result, f"{type(self.model).__name__}.forward")
        except Exception as exc:
            raise self._diagnosis(exc) from exc

    def _diagnosis(self, exc: Exception) -> RuntimeError:
        """A RuntimeError that turns an unfamiliar third-party API into one command.

        The attribute dump is the payload: it is what a maintainer needs to write the
        one line of extraction this adapter is missing, and it is the reason the
        message points at `symtrs doctor` rather than at a stack trace.
        """
        attributes = [name for name in dir(self.model) if not name.startswith("_")]
        return RuntimeError(
            f"cannot read a depth tensor from {type(self.model).__module__}."
            f"{type(self.model).__name__} (variant {self.variant!r}, key {self.key!r}). "
            f"Underlying error: {type(exc).__name__}: {exc}. "
            f"Expected forward() to return a tensor, a sequence containing one, or a dict "
            f"with one of the keys {list(_DEPTH_KEYS)}. Public attributes available on the "
            f"model: {attributes}. "
            f"Run `symtrs doctor --model {self.key}` and report its output."
        )
