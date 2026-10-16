"""The model registry.

One table mapping a short key to a task model class and the default arguments that
class needs. It is the only place that knows which models exist, so the CLI, the YAML
loader, `autoloader` and `symtrs tables` all agree on what `--model midas_dpt_hybrid`
means.

Entries hold an *import path* rather than the class itself. That matters: importing
this module must not pull in `transformers`, `depth_pro` or `timm`, because the whole
point of the registry is to let `symtrs doctor` tell you that one of them is missing
instead of dying at import time.

The `protocol` and `default_align` fields are copies of the adapter's own class
attributes, kept here so the run matrix can be printed without instantiating anything
(and therefore without downloading weights).
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

from symtrs.config import TASKS


class RegistryError(KeyError):
    """Raised for an unknown model key or a registry entry that cannot be built."""


@dataclass(frozen=True)
class ModelSpec:
    """Everything needed to build and describe one model."""

    key: str
    task: str
    #: "package.module:ClassName".
    target: str
    #: Arguments merged into whatever the caller passes; caller wins.
    default_kwargs: Mapping[str, Any] = field(default_factory=dict)
    #: Alternate names accepted on the command line.
    aliases: Tuple[str, ...] = ()
    description: str = ""
    #: Importable modules the adapter needs. Empty means "only the base deps".
    requires: Tuple[str, ...] = ()
    #: depth only: "metric" or "relative".
    protocol: str = ""
    #: depth only: the alignment convention the model is compared under.
    default_align: str = ""

    def load_class(self):
        """Import the adapter class this entry points at."""
        module_name, _, attribute = self.target.partition(":")
        if not attribute:
            raise RegistryError(f"registry entry {self.key!r} has a malformed target {self.target!r}")
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            missing = ", ".join(self.requires) or "a dependency of " + module_name
            raise RegistryError(
                f"model {self.key!r} needs {missing}, which is not installed ({exc}). "
                f"Install it with the matching extra, e.g. `pip install -e '.[{self.key.split('_')[0]}]'`."
            ) from exc
        try:
            return getattr(module, attribute)
        except AttributeError as exc:
            raise RegistryError(
                f"registry entry {self.key!r} points at {self.target!r}, but "
                f"{module_name} has no attribute {attribute!r}"
            ) from exc


def _spec(key: str, task: str, target: str, **kwargs: Any) -> ModelSpec:
    return ModelSpec(key=key, task=task, target=target, **kwargs)


_DEPTH = "symtrs.model.depth"
_SEMANTIC = "symtrs.model.semantic"

#: The registry. Order is the order `symtrs models` prints, grouped by task and then
#: by size within a family, so the small ones (the ones you can actually finetune on
#: one GPU) come first.
REGISTRY: Dict[str, ModelSpec] = {
    # ------------------------------------------------------------------- depth
    "midas_small": _spec(
        "midas_small",
        "depth",
        f"{_DEPTH}.midas:MiDaSAdapter",
        requires=("torch", "timm", "cv2"),
        default_kwargs={"hub_model": "MiDaS_small"},
        aliases=("midas",),
        description="MiDaS small, 256px, relative/DPT-lite. The cheapest depth baseline.",
        protocol="relative",
        default_align="scale",
    ),
    "midas_dpt_hybrid": _spec(
        "midas_dpt_hybrid",
        "depth",
        f"{_DEPTH}.midas:MiDaSAdapter",
        requires=("torch", "timm", "cv2"),
        default_kwargs={"hub_model": "DPT_Hybrid"},
        description="MiDaS DPT-Hybrid, 384px, relative. The standard published baseline.",
        protocol="relative",
        default_align="scale",
    ),
    "midas_dpt_large": _spec(
        "midas_dpt_large",
        "depth",
        f"{_DEPTH}.midas:MiDaSAdapter",
        requires=("torch", "timm", "cv2"),
        default_kwargs={"hub_model": "DPT_Large"},
        description="MiDaS DPT-Large, 384px, relative.",
        protocol="relative",
        default_align="scale",
    ),
    "midas_dpt_swin2_tiny": _spec(
        "midas_dpt_swin2_tiny",
        "depth",
        f"{_DEPTH}.midas:MiDaSAdapter",
        requires=("torch", "timm", "cv2"),
        default_kwargs={"hub_model": "DPT_SwinV2_T_256"},
        description="MiDaS DPT-SwinV2-Tiny, 256px, relative.",
        protocol="relative",
        default_align="scale",
    ),
    "midas_dpt_beit_large": _spec(
        "midas_dpt_beit_large",
        "depth",
        f"{_DEPTH}.midas:MiDaSAdapter",
        requires=("torch", "timm", "cv2"),
        default_kwargs={"hub_model": "DPT_BEiT_L_384"},
        description="MiDaS DPT-BEiT-Large, 384px, relative. Largest MiDaS variant.",
        protocol="relative",
        default_align="scale",
    ),
    # The ZoeDepth entries need timm but not cv2, which is worth a word because the MiDaS
    # entries need both. timm arrives with the MiDaS DPT-BEiT core that ZoeDepth builds on:
    # the hub entrypoint imports `midas.blocks` -> `backbones.beit`, which does `import timm`
    # at module level. cv2 is only reached through MiDaS's `transforms()` entrypoint, and the
    # ZoeDepth adapter never calls it - the model normalises its own input - so cv2 is absent
    # from these three lines. Measured: ZoeD_N builds, loads its released weights and runs a
    # forward pass with cv2 blocked from `sys.meta_path`.
    "zoedepth_n": _spec(
        "zoedepth_n",
        "depth",
        f"{_DEPTH}.zoedepth:ZoeDepthAdapter",
        default_kwargs={"variant": "N"},
        description="ZoeDepth-N, relative. A MiDaS backbone with a relative depth head.",
        requires=("torch", "timm"),
        protocol="relative",
        default_align="scale",
    ),
    "zoedepth_k": _spec(
        "zoedepth_k",
        "depth",
        f"{_DEPTH}.zoedepth:ZoeDepthAdapter",
        default_kwargs={"variant": "K"},
        aliases=("zoedepth",),
        description="ZoeDepth-K, metric (metres). The metric baseline to beat.",
        requires=("torch", "timm"),
        protocol="metric",
        default_align="none",
    ),
    "zoedepth_nk": _spec(
        "zoedepth_nk",
        "depth",
        f"{_DEPTH}.zoedepth:ZoeDepthAdapter",
        default_kwargs={"variant": "NK"},
        description="ZoeDepth-NK, metric with a domain classifier; pass model_kwargs.domain_id.",
        requires=("torch", "timm"),
        protocol="metric",
        default_align="none",
    ),
    "da3_large": _spec(
        "da3_large",
        "depth",
        f"{_DEPTH}.da3:DepthAnything3Adapter",
        default_kwargs={"model_id": "depth-anything/DA3-LARGE"},
        aliases=("da3",),
        description="Depth Anything 3 Large, relative. The un-metriced DA3 checkpoint.",
        requires=("torch", "depth_anything_3", "omegaconf", "addict", "safetensors"),
        protocol="relative",
        default_align="scale",
    ),
    "da3_metric_large": _spec(
        "da3_metric_large",
        "depth",
        f"{_DEPTH}.da3:DepthAnything3Adapter",
        default_kwargs={"model_id": "depth-anything/DA3METRIC-LARGE", "metric": True},
        description="Depth Anything 3 Large, metric head (metres).",
        requires=("torch", "depth_anything_3", "omegaconf", "addict", "safetensors"),
        protocol="metric",
        default_align="none",
    ),
    "depth_pro": _spec(
        "depth_pro",
        "depth",
        f"{_DEPTH}.depth_pro:DepthProAdapter",
        aliases=("depthpro",),
        description="Apple Depth Pro, metric (metres), 1536px. ViT-L, ~1.9GB of weights.",
        requires=("depth_pro",),
        protocol="metric",
        default_align="none",
    ),
    # ------------------------------------------------------------ segmentation
    "unet_small": _spec(
        "unet_small",
        "segmentation",
        f"{_SEMANTIC}.unet:UNetSegmentation",
        aliases=("unet",),
        description="Small from-scratch U-Net. The cheapest segmentation baseline.",
        requires=("torch",),
    ),
    "deeplabv3_r50": _spec(
        "deeplabv3_r50",
        "segmentation",
        f"{_SEMANTIC}.deeplab:DeepLabV3Segmentation",
        default_kwargs={"backbone": "r50"},
        aliases=("deeplab",),
        description="torchvision DeepLabV3 with a ResNet-50 backbone, COCO+VOC pretrained.",
        requires=("torch", "torchvision"),
    ),
    "deeplabv3_r101": _spec(
        "deeplabv3_r101",
        "segmentation",
        f"{_SEMANTIC}.deeplab:DeepLabV3Segmentation",
        default_kwargs={"backbone": "r101"},
        description="torchvision DeepLabV3 with a ResNet-101 backbone.",
        requires=("torch", "torchvision"),
    ),
    "segformer_b0": _spec(
        "segformer_b0",
        "segmentation",
        f"{_SEMANTIC}.segformer:SegFormerSegmentation",
        default_kwargs={"variant": "b0"},
        description="SegFormer with a MiT-B0 encoder (transformers) and the SegFormer decode head.",
        requires=("torch", "transformers"),
    ),
    "segformer_b1": _spec(
        "segformer_b1",
        "segmentation",
        f"{_SEMANTIC}.segformer:SegFormerSegmentation",
        default_kwargs={"variant": "b1"},
        description="SegFormer with a MiT-B1 encoder (transformers) and the SegFormer decode head.",
        requires=("torch", "transformers"),
    ),
    "segformer_b2": _spec(
        "segformer_b2",
        "segmentation",
        f"{_SEMANTIC}.segformer:SegFormerSegmentation",
        default_kwargs={"variant": "b2"},
        aliases=("segformer",),
        description="SegFormer with a MiT-B2 encoder (transformers) and the SegFormer decode head.",
        requires=("torch", "transformers"),
    ),
    # ----------------------------------------------------------- domain adaptation
    "cyclegan": _spec(
        "cyclegan",
        "domain_adapt",
        "symtrs.model.domain:DomainAdaptation",
        default_kwargs={"direction": "day2night"},
        aliases=("cycle_gan",),
        description=(
            "CycleGAN day<->night translation, unpaired. Direction is model_kwargs.direction."
        ),
        requires=("torch",),
    ),
    "pix2pix": _spec(
        "pix2pix",
        "domain_adapt",
        "symtrs.model.domain:DomainAdaptation",
        default_kwargs={"direction": "day2night"},
        description=(
            "pix2pix day<->night translation, paired by filename. Direction is "
            "model_kwargs.direction."
        ),
        requires=("torch",),
    ),
    # ------------------------------------------------------------- shadow removal
    # Like domain adaptation, the networks are upstream's (third_party/, pinned in
    # third_party/PINS.md); `symtrs.model.shadow` stages the data, launches the worker that
    # imports the class, and scores the result. Architecture arguments live in
    # `symtrs/model/shadow/archs.py`, so this table only says that the models exist.
    "shadowformer": _spec(
        "shadowformer",
        "shadow_removal",
        "symtrs.model.shadow:ShadowRemoval",
        description=(
            "ShadowFormer (guolanqing/ShadowFormer), shadow-interaction transformer; takes the "
            "derived shadow mask as a second input. 2048 only."
        ),
        requires=("torch", "timm", "einops"),
    ),
    "restormer": _spec(
        "restormer",
        "shadow_removal",
        "symtrs.model.shadow:ShadowRemoval",
        description="Restormer (swz30/Restormer), channel-attention transformer. 2048 only.",
        requires=("torch", "einops"),
    ),
    "uformer": _spec(
        "uformer",
        "shadow_removal",
        "symtrs.model.shadow:ShadowRemoval",
        description="Uformer-B (ZhendongWang6/Uformer), windowed U-shaped transformer. 2048 only.",
        requires=("torch", "timm", "einops"),
    ),
    # ------------------------------------------------------------ super-resolution
    # These four are not built by this package. `symtrs/legacy/superres/models/` owns the
    # architectures and `symtrs/legacy/superres/train.py` owns the training loop; the
    # entries exist so that `symtrs models`, `symtrs sweep --task superresolution` and the
    # YAML loader agree with the trainer about which models exist. The class they point at
    # is a command-line translator, not an `nn.Module`.
    "srcnn": _spec(
        "srcnn",
        "superresolution",
        "symtrs.model.superres:SuperResolution",
        description="SRCNN: three conv layers on a bicubic-upsampled input. The cheap baseline.",
        requires=("torch",),
    ),
    "autoencoder": _spec(
        "autoencoder",
        "superresolution",
        "symtrs.model.superres:SuperResolution",
        description="Encoder-decoder refiner on a bicubic-upsampled input.",
        requires=("torch",),
    ),
    "srgan": _spec(
        "srgan",
        "superresolution",
        "symtrs.model.superres:SuperResolution",
        description="SRGAN: adversarial training; the only model here with a discriminator.",
        requires=("torch",),
    ),
    "swinir": _spec(
        "swinir",
        "superresolution",
        "symtrs.model.superres:SuperResolution",
        description="SwinIR: window-attention transformer. The heaviest SR baseline.",
        requires=("torch",),
    ),
}

#: alias -> canonical key, built once so lookups are exact.
_ALIASES: Dict[str, str] = {}
for _key, _entry in REGISTRY.items():
    _ALIASES[_key] = _key
    for _alias in _entry.aliases:
        _ALIASES.setdefault(_alias, _key)
del _key, _entry, _alias


def get_spec(model: str) -> ModelSpec:
    """Look up a spec by key or alias."""
    if not model:
        raise RegistryError("a model key is required")
    canonical = _ALIASES.get(model)
    if canonical is None:
        close = _suggest(model)
        hint = f" Did you mean {close!r}?" if close else ""
        raise RegistryError(
            f"unknown model {model!r}.{hint} Run `symtrs models` for the full list."
        )
    return REGISTRY[canonical]


def list_models(task: Optional[str] = None) -> List[ModelSpec]:
    """Every spec, optionally restricted to one task."""
    if task is not None and task not in TASKS:
        raise RegistryError(f"unknown task {task!r}; expected one of {TASKS}")
    return [spec for spec in REGISTRY.values() if task is None or spec.task == task]


def model_keys(task: Optional[str] = None) -> List[str]:
    return [spec.key for spec in list_models(task)]


def build_model(model: str, **overrides: Any):
    """Instantiate a model by key, merging registry defaults with the caller's kwargs.

    The returned module's `.key` is forced to the registry key. Adapters sometimes
    derive their own key from a model id, and a derived key containing a slash would
    turn `runs/<task>/<model>/<resolution>` into a nested path, so the registry key
    wins.
    """
    spec = get_spec(model)
    kwargs: Dict[str, Any] = dict(spec.default_kwargs)
    kwargs.update({k: v for k, v in overrides.items() if v is not None})

    unknown = set(kwargs) - set(spec.default_kwargs) - _COMMON_MODEL_KWARGS
    if unknown:
        # Not fatal: a model-specific option such as Domain id belongs here. Warn so a
        # typo in the YAML is visible rather than silently ignored by **kwargs.
        print(
            f"[symtrs] note: {sorted(unknown)} is not a documented option of "
            f"{spec.key!r}; passing it through to {spec.target.rsplit(':', 1)[-1]}."
        )

    cls = spec.load_class()
    instance = cls(**kwargs)
    instance.key = spec.key
    if not getattr(instance, "task", ""):
        instance.task = spec.task
    return instance


#: Constructor arguments every adapter in the registry accepts.
_COMMON_MODEL_KWARGS = frozenset(
    {"pretrained", "loss", "loss_kwargs", "align", "freeze", "num_classes", "ignore_index", "device"}
)


def describe_registry(task: Optional[str] = None, specs: Optional[List[ModelSpec]] = None) -> str:
    """A printable table of the registry, for `symtrs models`."""
    lines = []
    current = None
    for spec in list_models(task) if specs is None else specs:
        if spec.task != current:
            current = spec.task
            lines.append(f"\n{current}")
            lines.append("-" * 78)
            lines.append(f"{'key':<26} {'protocol':<10} {'align':<12} description")
        protocol = spec.protocol or "-"
        align = spec.default_align or "-"
        lines.append(f"{spec.key:<26} {protocol:<10} {align:<12} {spec.description}")
        if spec.aliases:
            lines.append(f"{'':<26} {'':<10} {'':<12} aliases: {', '.join(spec.aliases)}")
    return "\n".join(lines).lstrip("\n")


def _suggest(model: str, cutoff: float = 0.6) -> Optional[str]:
    """Closest registry key by a cheap ratio, for a typo hint."""
    import difflib

    matches = difflib.get_close_matches(model, list(_ALIASES), n=1, cutoff=cutoff)
    return matches[0] if matches else None


__all__ = [
    "ModelSpec",
    "RegistryError",
    "REGISTRY",
    "build_model",
    "describe_registry",
    "get_spec",
    "list_models",
    "model_keys",
]
