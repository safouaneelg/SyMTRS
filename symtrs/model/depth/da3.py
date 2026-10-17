"""Depth Anything 3 through the `depth_anything_3` package.

The `depth-anything/DA3-*` checkpoints on the Hub are not `transformers` models. Their
`config.json` is a ByteDance object-graph (`{"__object__": {"path":
"depth_anything_3.model.da3", "name": "DepthAnything3Net"}, ...}`), the repositories carry
no image-processor config, and `model.safetensors` keys are prefixed `model.backbone.*`,
`model.head.*`. `AutoImageProcessor` / `AutoModelForDepthEstimation` therefore cannot load
them (the first version of this adapter failed with "no image processor could be loaded").
This adapter builds the network the way the package's own `DepthAnything3.from_pretrained`
does - `create_object(config)` on the checkpoint's own `config.json` - and loads the
safetensors into it.

Only the network is taken from the package. The package's high-level `api` module imports
the whole export stack (open3d, moviepy, pycolmap, gradio...) and runs multi-view
inference with a bf16 autocast, none of which this benchmark wants, so `depth_anything_3.api`
is never imported. Install the package without its dependency list:

    pip install -e '.[da3]'                   # omegaconf, addict, safetensors, huggingface_hub
    pip install --no-deps depth-anything-3==0.1.1

`--no-deps` is deliberate: the wheel pins `numpy<2` and lists xformers, open3d, evo, e3nn,
pycolmap... for features (3D export, Gaussian splats) that `model.da3` never imports.

Preprocessing is explicit, not delegated: the package's `InputProcessor` is ImageNet
mean/std normalisation (`T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])`) plus an
aspect-preserving resize to a multiple of the 14-pixel patch. The first is exactly
`DepthAdapter.preprocess_image`; the second is `size_multiple = 14` here. Nothing resizes
the image to the package's 504-pixel default: this benchmark's `resolution` names the pixel
size of the data (AGENTS.md ground rule 1), so the network sees the data at that size.

Two conventions have to be decided per checkpoint, and the model id is the only
information available before the checkpoint is downloaded:

    protocol        "metric" if the checkpoint was trained with metric supervision
                    (millimetres/metres), otherwise "relative".
    default_align   "none" for metric; "scale" for relative, the convention the Depth
                    Anything evaluations use.

`metric` infers that from the model id - an id containing "metric" (case-insensitive) is
taken to be metric. That inference is a *guess* keyed on the naming convention of the
released checkpoints (`depth-anything/DA3METRIC-LARGE` is metric,
`depth-anything/DA3-LARGE` is not), and it is a guess about the checkpoint, not about the
architecture. Pass `metric=True` or `metric=False` explicitly to overrule it.

Canonical metric depth
----------------------
The DA3METRIC model card says "canonical metric depth; multiplying by focal length gives
metric depth", and the package does `depth * focal / 300` (`apply_metric_scaling`). The
dataset's camera intrinsics are not part of the depth interface, so `focal_px` is an
explicit argument: when given, the metric output is scaled by `focal_px / 300`; when
`None`, the output is the canonical depth, unscaled, and the metric numbers of this model
are only meaningful up to that factor. It is not guessed here - but it is not left unset in
a run either: `symtrs.model.wrapper.build_from_config` derives it from the data (registry
`focal_kwarg`; `focal_px_for_resolution(resolution, data.camera_hfov_deg)`), because the
capture camera's field of view is a property of the dataset and the focal length in pixels
follows the resolution axis. With it unset the real-data smoke runs scored abs_rel 0.91-0.96
and d1 = 0 (predictions ~4% of the ground truth).

Depth, not disparity
--------------------
The base adapter assumes a relative model emits disparity, so it converts the target to
`1 / depth` and inverts the prediction in `to_metres`. Depth Anything 3 emits *depth*, so
this adapter keeps a relative model's target and prediction in depth space (see
`prepare_target` and `to_metres`), the same override the ZoeDepth-N adapter makes.

Aux head weights
----------------
The released DA3-LARGE checkpoint omits the `scratch.output_conv2_aux.{1,2,3}.2` layers of
the dual-DPT head. The package loads with `strict=False` for that reason; those layers
belong to auxiliary levels whose output is discarded (only `output_conv2_aux[-1]` feeds the
ray/pose branch). This adapter tolerates exactly those keys and raises on any other
missing or unexpected key, so a genuinely wrong checkpoint is still an error.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Dict, Optional

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


#: The canonical focal length the metric head is normalised to (`apply_metric_scaling`'s
#: default `scale_factor` in the package): metres = canonical_depth * focal_px / 300.
CANONICAL_FOCAL = 300.0

#: Keys the released DA3-LARGE checkpoint legitimately lacks; see the module docstring.
_TOLERATED_MISSING = re.compile(r"^head\.scratch\.output_conv2_aux\.[123]\.2\.(weight|bias)$")


class DepthAnything3Adapter(DepthAdapter):
    """A Depth Anything 3 checkpoint behind the common depth interface."""

    requires = ("torch", "depth_anything_3", "omegaconf", "addict", "safetensors")
    # The backbone patch size; any other size is a shape error inside the ViT.
    size_multiple = 14
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
        focal_px: Optional[float] = None,
    ) -> None:
        """Args:
            model_id: a `depth-anything/DA3*` Hub id (or a local directory holding the
                checkpoint's `config.json` and `model.safetensors`).
            metric: True for a metric checkpoint, False for an affine-invariant one.
                None (default) infers it from `model_id`; see the module docstring for
                why that inference is a guess and how to overrule it.
            pretrained: Depth Anything 3 has no random-init mode, so False is rejected
                rather than silently ignored.
            focal_px: focal length in pixels of the image given to the network. Metric
                output is scaled by `focal_px / 300`; None leaves it canonical.
        """
        if not pretrained:
            raise ValueError(
                "Depth Anything 3 only ships trained weights; pretrained=False has no "
                "meaning here. Omit it or pass True."
            )
        self.model_id = model_id
        self.metric = _infer_metric(model_id) if metric is None else bool(metric)
        self.focal_px = None if focal_px is None else float(focal_px)
        if self.metric and self.focal_px is None:
            print(
                f"[symtrs] warning: {model_id} is metric but focal_px is unset, so it returns "
                "canonical depth (a 300 px camera), not metres. A run built from a config gets "
                "it from data.camera_hfov_deg; for SDK use pass focal_px=..."
            )

        # DepthAdapter.__init__ reads both of these, so they must be set first.
        self.protocol = "metric" if self.metric else "relative"
        self.default_align = "none" if self.metric else "scale"
        self.key = "da3_" + _slug_model_id(model_id)
        super().__init__(loss=loss, loss_kwargs=loss_kwargs, align=align, freeze=freeze)

        self.model = self._load_model(model_id)

    # -------------------------------------------------------------------- loading

    def _load_model(self, model_id: str):
        """Build `DepthAnything3Net` from the checkpoint's own `config.json` + weights."""
        import json
        import os

        try:
            from omegaconf import OmegaConf
            from safetensors.torch import load_file

            from depth_anything_3.cfg import create_object
        except ImportError as exc:
            raise ImportError(
                f"{self.key} needs the `depth_anything_3` package and omegaconf/addict/"
                f"safetensors ({exc}). Install with `pip install -e '.[da3]'` and then "
                "`pip install --no-deps depth-anything-3==0.1.1` (--no-deps: the wheel pins "
                "numpy<2 and lists 3D-export packages this adapter never imports)."
            ) from exc

        try:
            if os.path.isdir(model_id):
                config_path = os.path.join(model_id, "config.json")
                weights_path = os.path.join(model_id, "model.safetensors")
            else:
                from huggingface_hub import hf_hub_download

                config_path = hf_hub_download(model_id, "config.json")
                weights_path = hf_hub_download(model_id, "model.safetensors")
            with open(config_path, encoding="utf-8") as handle:
                config = json.load(handle)
            network = create_object(OmegaConf.create(config["config"]))
            state = load_file(weights_path)
        except Exception as exc:
            raise RuntimeError(
                f"could not load Depth Anything 3 checkpoint {model_id!r} "
                f"({type(exc).__name__}: {exc}). Offline runs need the files in the Hugging "
                f"Face cache (HF_HOME); run `python scripts/prefetch_weights.py --keys "
                f"{self.key}` once with network access."
            ) from exc

        # The Hub files prefix every key with the wrapper attribute (`model.`) that the
        # package's `DepthAnything3` class holds the network under.
        state = {
            (key[len("model."):] if key.startswith("model.") else key): value
            for key, value in state.items()
        }
        result = network.load_state_dict(state, strict=False)
        missing = [key for key in result.missing_keys if not _TOLERATED_MISSING.match(key)]
        if missing or result.unexpected_keys:
            raise RuntimeError(
                f"checkpoint {model_id!r} does not match the installed depth_anything_3 "
                f"network: missing {missing[:5]}, unexpected {list(result.unexpected_keys)[:5]}. "
                "Install the package version the checkpoint was released with "
                "(`pip install --no-deps depth-anything-3==0.1.1`)."
            )
        return network

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
        """Depth `[B, 1, H, W]` for a preprocessed `[B, 3, H, W]` batch.

        The network is multi-view (`[B, N, 3, H, W]`); a monocular image is one view, so
        `N = 1` and nothing about cross-view attention is exercised.
        """
        try:
            outputs = self.model(image.unsqueeze(1))
        except Exception as exc:
            raise RuntimeError(
                f"calling {type(self.model).__module__}.{type(self.model).__name__} "
                f"({self.model_id!r}) failed: {type(exc).__name__}: {exc}. "
                f"Run `symtrs doctor --model {self.key}` and report its output."
            ) from exc
        depth = _as_depth_tensor(outputs, f"{type(self.model).__name__}.forward")
        if self.protocol == "metric" and self.focal_px is not None:
            depth = depth * (self.focal_px / CANONICAL_FOCAL)
        return depth
