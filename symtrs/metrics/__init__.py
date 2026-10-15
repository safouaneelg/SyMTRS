"""Task metrics.

    depth.py         RMSE/MAE/abs_rel/sq_rel/delta/SILog, plus per-image alignment
    semantic.py      confusion-matrix-accumulated mIoU, pixel accuracy, Dice, FWIOU
    domain.py        FID and KID via Inception-V3 features

`metrics.depth` and `metrics.semantic` are import-light (torch and numpy only).
`metrics.domain` pulls `torchvision`, so it is imported on demand rather than here.
"""

from __future__ import annotations

from typing import Any

from symtrs.metrics.depth import (
    ALIGNMENTS,
    DEPTH_METRIC_NAMES,
    AllInvalidError,
    AlignmentFit,
    align_prediction,
    compute_depth_metrics,
    summarise_alignment,
)
from symtrs.metrics.semantic import SEGMENTATION_METRIC_NAMES, ConfusionMatrix


def __getattr__(name: str) -> Any:
    """FID/KID helpers, loaded on first use so torchvision is not a hard dependency."""
    if name in {
        "InceptionFeatures",
        "compute_fid",
        "compute_kid",
        "distribution_metrics",
        "extract_features",
    }:
        import importlib

        module = importlib.import_module("symtrs.metrics.domain")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | {"InceptionFeatures", "compute_fid", "compute_kid",
                                    "distribution_metrics", "extract_features"})


__all__ = [
    "ALIGNMENTS",
    "AllInvalidError",
    "AlignmentFit",
    "ConfusionMatrix",
    "DEPTH_METRIC_NAMES",
    "SEGMENTATION_METRIC_NAMES",
    "align_prediction",
    "compute_depth_metrics",
    "distribution_metrics",
    "summarise_alignment",
]
