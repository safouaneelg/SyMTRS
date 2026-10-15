"""Data layer: the resolution axis, the depth reader, and the task datasets.

    from symtrs.data import dataloader

    loader = dataloader(task="segmentation", root="/data/SyMTRS", split="train",
                        resolution=512, batch_size=8)

`resolve_split_dirs`, `describe_layout` and everything else in `splits` is importable
*without* torch:

    from symtrs.data.splits import describe_layout

That is worth the small amount of machinery below. Those functions only touch paths, and
the two places that most need them - `symtrs inspect` and a dry run that checks the
dataset layout before downloading any weights - are exactly the places you run on a login
node, where torch may not be installed at all. Everything that needs torch is resolved on
first attribute access instead, so importing this package stays cheap and a missing torch
breaks only the parts that genuinely need it.
"""

from __future__ import annotations

import importlib
from typing import Any

# The path-only half of the package. Imported eagerly, and deliberately so: it has no
# dependencies beyond the standard library, so there is nothing to defer.
from symtrs.data.splits import (
    RESOLUTIONS,
    RESOLUTION_TO_SCALE,
    RESOLUTION_TO_SUFFIX,
    SPLITS,
    MissingDirectoryError,
    SplitDirs,
    available_resolutions,
    describe_layout,
    resolve_split_dirs,
    scale_for_resolution,
    subdir_for,
)

#: Attribute name -> module that defines it. Resolved on first access, so that a missing
#: torch - or a missing PIL, or OpenEXR - only breaks the reader that needs it.
_LAZY = {
    "DEPTH_EXTENSIONS": "symtrs.data.exr",
    "DepthReadError": "symtrs.data.exr",
    "depth_summary": "symtrs.data.exr",
    "depth_to_tensor": "symtrs.data.exr",
    "guess_scale_from_summary": "symtrs.data.exr",
    "list_depth_files": "symtrs.data.exr",
    "read_depth_file": "symtrs.data.exr",
    "ImageReadError": "symtrs.data.io",
    "coverage_report": "symtrs.data.io",
    "list_images": "symtrs.data.io",
    "match_by_stem": "symtrs.data.io",
    "normalize_stem": "symtrs.data.io",
    "read_image": "symtrs.data.io",
    "read_label_map": "symtrs.data.io",
    "build_dataset": "symtrs.data.factory",
    "dataloader": "symtrs.data.factory",
    "from_config": "symtrs.data.factory",
    "DepthTaskDataset": "symtrs.data.depth_dataset",
    "depth_coverage": "symtrs.data.depth_dataset",
    "DEFAULT_NUM_CLASSES": "symtrs.data.semantic_dataset",
    "SemanticDataset": "symtrs.data.semantic_dataset",
    "check_num_classes": "symtrs.data.semantic_dataset",
    "detect_class_values": "symtrs.data.semantic_dataset",
    "semantic_coverage": "symtrs.data.semantic_dataset",
    "EvalTransform": "symtrs.data.transforms",
    "PairedTransform": "symtrs.data.transforms",
    "IMAGENET_MEAN": "symtrs.data.transforms",
    "IMAGENET_STD": "symtrs.data.transforms",
    "apply_crop": "symtrs.data.transforms",
    "center_crop_pair": "symtrs.data.transforms",
    "flip_tensors": "symtrs.data.transforms",
    "match_spatial": "symtrs.data.transforms",
    "normalize_image": "symtrs.data.transforms",
    "random_crop_pair": "symtrs.data.transforms",
    "resize_labels": "symtrs.data.transforms",
    "sample_crop_params": "symtrs.data.transforms",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name)
    try:
        value = getattr(module, name)
    except AttributeError as exc:
        raise AttributeError(
            f"{module_name} does not export {name!r}; its dependency may not be installed"
        ) from exc
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    "DEFAULT_NUM_CLASSES",
    "DEPTH_EXTENSIONS",
    "DepthReadError",
    "DepthTaskDataset",
    "EvalTransform",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "ImageReadError",
    "MissingDirectoryError",
    "PairedTransform",
    "RESOLUTIONS",
    "RESOLUTION_TO_SCALE",
    "RESOLUTION_TO_SUFFIX",
    "SPLITS",
    "SemanticDataset",
    "SplitDirs",
    "apply_crop",
    "available_resolutions",
    "build_dataset",
    "center_crop_pair",
    "check_num_classes",
    "coverage_report",
    "dataloader",
    "depth_coverage",
    "depth_summary",
    "depth_to_tensor",
    "describe_layout",
    "detect_class_values",
    "flip_tensors",
    "from_config",
    "guess_scale_from_summary",
    "list_depth_files",
    "list_images",
    "match_by_stem",
    "match_spatial",
    "normalize_image",
    "normalize_stem",
    "random_crop_pair",
    "read_depth_file",
    "read_image",
    "read_label_map",
    "resize_labels",
    "resolve_split_dirs",
    "sample_crop_params",
    "scale_for_resolution",
    "semantic_coverage",
    "subdir_for",
]
