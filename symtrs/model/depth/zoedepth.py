"""ZoeDepth adapters, loaded from `torch.hub`.

ZoeDepth bolts a metric bin head onto a frozen MiDaS DPT-BEiT-Large core:

    model = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True)

Three checkpoints ship in one release and they are not interchangeable:

    ZoeD_N   ZoeD_M12_N: MiDaS-pretrained on 12 datasets, then fine-tuned with metric
             supervision on NYU-Depth-v2 (indoor, 0-10 m). It IS a metric model - but its metres
             are indoor metres, which do not transfer to a camera 70-270 m above the ground, so
             it is scored here as `protocol="relative"` (depth up to a scale,
             `default_align="scale"`). That is a protocol choice for this benchmark, not a
             property of the checkpoint.
    ZoeD_K   ZoeD_M12_K: fine-tuned on KITTI (outdoor driving, 0-80 m) -> depth in metres, so
             `protocol="metric"` and `default_align="none"`.
    ZoeD_NK  NYU and KITTI metric heads behind a learned domain router; metric like
             ZoeD_K, with one bin configuration per domain.

Which heads have a ceiling (read from the pinned hub code, `localbins_layers.py` /
`attractor.py`)
---------------------------------------------------------------------------------------
Only the *normed* bin head bounds its output. ZoeD_K is built with `bin_centers_type="normed"`:
the bin centres live in `[min_depth, max_depth]`, so no prediction can exceed `max_depth`.
ZoeD_N and ZoeD_NK are built with `"softplus"`: `SeedBinRegressorUnnormed` documents
`min_depth`/`max_depth` as "not used", so their output is unbounded (the NK `bin_conf` values of
10 m and 80 m are inert). Their 1-4 m on aerial scenes is the model misjudging the scene, not
a ceiling.

ZoeD_K deserves care: the hub entry point `ZoeD_K` calls `get_config(..., config_version=
"kitti")` without `dataset="kitti"`, so the `max_depth: 80` of the KITTI dataset config is never
merged and `ZoeDepth.__init__` falls back to its own default of **10 m**. The released weights
were trained with 80. This adapter therefore always passes `max_depth` to the hub (default
`ZOED_K_TRAINED_MAX_DEPTH` = 80, the faithful zero-shot setting). For fine-tuning on this
dataset, whose scenes reach ~280 m, pass a larger value (the generated configs use 300): the
parameters keep their shapes, so the pretrained head loads, and its normalised bin layout is
simply stretched over a longer range - a starting point, not a calibration; the fine-tune has
to relearn the distribution. The key is recorded in the checkpoint's `model_kwargs`, so a
reloaded model keeps its range.

Why ZoeD_N is relative but still emits depth, not disparity
-----------------------------------------------------------
The MiDaS convention - a relative model emits *disparity*, so the target becomes
`1 / depth` and the alignment is a disparity-space fit - does not apply here. ZoeDepth's
head is a softmax over `n_bins` bin *centres* that live in depth space, and the prediction is
their expectation, `out = sum(prob * bin_center)`, computed in
`zoedepth/models/zoedepth/zoedepth_v1.py`. So what comes out is depth - in NYU metres, which
are the wrong metres for this dataset - and it is treated as depth known up to a scale, aligned
by `scale` (`output_space = "depth"`). This adapter therefore
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

#: The depth ceiling ZoeD_M12_K was trained with (KITTI). The hub's `ZoeD_K` entry point builds
#: the head with 10 instead; see the module docstring. `registry.py` repeats this number in the
#: `zoedepth_k` default kwargs and `tests/test_depth_conventions.py` keeps the two equal.
ZOED_K_TRAINED_MAX_DEPTH = 80.0

#: Variants whose head overflows in float16. Measured on the RTX 5070 Ti (one real test image,
#: forward only, 512 and 1024 px): ZoeD_N under fp16 autocast returned NaN for ~10% of the
#: pixels (finite = 0.903 / 0.906); the same forward in fp32 and in bfloat16 was entirely finite
#: and agreed to 3 digits (mean 1.674 vs 1.674 m). ZoeD_K and ZoeD_NK were finite in all three
#: modes on that image. The NaN made the training loss non-finite from the first step and every
#: evaluated prediction non-positive.
_FP16_UNSAFE_VARIANTS = frozenset({"N"})


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
        max_depth: Optional[float] = None,
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
            max_depth: ZoeD_K only - the ceiling of its bounded bin head, in metres. None means
                `ZOED_K_TRAINED_MAX_DEPTH` (80, what the released weights were trained with; the
                hub would otherwise build 10). Raise it to fine-tune on scenes deeper than that.
                ZoeD_N and ZoeD_NK have unbounded heads, so setting it there is refused rather
                than silently ignored.
        """
        variant = variant.upper()
        if variant not in _VARIANTS:
            raise ValueError(
                f"unknown ZoeDepth variant {variant!r}; expected one of {sorted(_VARIANTS)}"
            )
        entrypoint, protocol, default_align = _VARIANTS[variant]
        if variant == "K":
            max_depth = ZOED_K_TRAINED_MAX_DEPTH if max_depth is None else float(max_depth)
            if max_depth <= 0:
                raise ValueError(f"max_depth must be positive (metres); got {max_depth}")
        elif max_depth is not None:
            raise ValueError(
                f"max_depth has no effect on ZoeDepth-{variant}: its bin centres are softplus "
                "(unbounded), so there is no ceiling to move. Only ZoeDepth-K has a bounded "
                "head; see the module docstring."
            )

        # DepthAdapter.__init__ reads `protocol` and `default_align`, so they have to be
        # in place before super() runs.
        self.variant = variant
        self.protocol = protocol
        self.default_align = default_align
        self.native_input = _CORE_SIZE
        self.key = f"zoedepth_{variant.lower()}"
        self.domain_id = int(domain_id)
        super().__init__(loss=loss, loss_kwargs=loss_kwargs, align=align, freeze=freeze)

        self.max_depth = max_depth
        self.model = self._load(entrypoint, pretrained, max_depth)
        self._domain_kwarg = self._find_domain_kwarg()

    # -------------------------------------------------------------------- loading

    @staticmethod
    def _load(entrypoint: str, pretrained: bool, max_depth: Optional[float] = None):
        # Only ZoeD_K takes it (`config_zoedepth_kitti.json` is "normed", and `max_depth` is the
        # upper bound of its bin centres); the others are built exactly as the hub would.
        hub_kwargs = {} if max_depth is None else {"max_depth": max_depth}
        try:
            # The hub entry point does its own `load_state_dict`, so a released checkpoint
            # whose `relative_position_index` buffers a newer timm no longer persists can only
            # be relaxed from outside it. See `symtrs.model.depth._hub`.
            with load_dropping_position_index():
                return torch.hub.load(
                    "isl-org/ZoeDepth", entrypoint, pretrained=pretrained, trust_repo=True,
                    **hub_kwargs,
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
            with self._safe_precision(image):
                result = self.model(image, **kwargs)
            depth = _as_depth_tensor(result, f"{type(self.model).__name__}.forward")
            return depth.float() if self.variant in _FP16_UNSAFE_VARIANTS else depth
        except Exception as exc:
            raise self._diagnosis(exc) from exc

    def _safe_precision(self, image: torch.Tensor):
        """Keep a fp16-unsafe head out of float16, whatever the caller's autocast says.

        `train.amp` / `eval.amp` switch on float16 autocast for the whole step, and a config
        cannot ask for bfloat16, so the fix lives with the model that needs it. Inside an active
        CUDA autocast, ZoeD_N runs in bfloat16 (same range as fp32, so no overflow; supported on
        Ampere and later) or, on a GPU without it, in plain float32. Other variants and CPU
        runs are untouched.
        """
        import contextlib

        if (
            self.variant not in _FP16_UNSAFE_VARIANTS
            or not image.is_cuda
            or not torch.is_autocast_enabled()
        ):
            return contextlib.nullcontext()
        if torch.cuda.is_bf16_supported():
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return torch.autocast("cuda", enabled=False)

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
