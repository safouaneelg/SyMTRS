"""Models: the task interface, the registry, and the SDK loader.

The common case is two lines:

    from symtrs.model import autoloader

    model = autoloader("weights/best.pt")     # or autoloader("midas_dpt_hybrid")
    depth = model.predict("scene.png")

Adapter classes are re-exported lazily through `__getattr__`. Importing them eagerly
would mean that a missing optional dependency - `timm` for SegFormer, `transformers`
for Depth Anything 3, `depth_pro` for Depth Pro - breaks `import symtrs.model` for
everyone, including the people who only want MiDaS.
"""

from __future__ import annotations

from typing import Any

from symtrs.model.base import DepthTarget, TaskModel, apply_freeze_policy
from symtrs.model.wrapper import SymTRSModel, autoloader, build_from_config, to_tensor

#: Attribute name -> module that defines it. Resolved on first access.
_LAZY = {
    "DepthAdapter": "symtrs.model.depth",
    "ALIGNMENT_NOTES": "symtrs.model.depth",
    "MiDaSAdapter": "symtrs.model.depth",
    "MIDAS_MODELS": "symtrs.model.depth",
    "ZoeDepthAdapter": "symtrs.model.depth",
    "DepthAnything3Adapter": "symtrs.model.depth",
    "DepthProAdapter": "symtrs.model.depth",
    "DeepLabV3Segmentation": "symtrs.model.semantic",
    "SegFormerSegmentation": "symtrs.model.semantic",
    "UNetSegmentation": "symtrs.model.semantic",
    "build_segmentation_loss": "symtrs.model.semantic",
    "build_depth_loss": "symtrs.model.depth.losses",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(module_name)
    try:
        value = getattr(module, name)
    except AttributeError as exc:
        raise AttributeError(
            f"{module_name} does not export {name!r}; it may not have been installed "
            "with the extra that provides it"
        ) from exc
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    "ALIGNMENT_NOTES",
    "DepthAdapter",
    "DepthProAdapter",
    "DepthTarget",
    "DepthAnything3Adapter",
    "DeepLabV3Segmentation",
    "MIDAS_MODELS",
    "MiDaSAdapter",
    "SegFormerSegmentation",
    "SymTRSModel",
    "TaskModel",
    "UNetSegmentation",
    "ZoeDepthAdapter",
    "apply_freeze_policy",
    "autoloader",
    "build_from_config",
    "build_depth_loss",
    "build_segmentation_loss",
    "to_tensor",
]
