"""Shared, torch-only helpers used by every task.

    from symtrs.common import MetricTracker, save_checkpoint

Everything here needs torch, so unlike `symtrs.data` there is no path-only half to import
eagerly. The modules are still resolved lazily, because `symtrs.common.table` - a CSV
writer - is reachable from `symtrs tables` and from the run summary, and neither of those
should require a working torch install to print a table.
"""

from __future__ import annotations

import importlib
from typing import Any

_LAZY = {
    "CHECKPOINT_KEY": "symtrs.common.checkpoint",
    "load_checkpoint": "symtrs.common.checkpoint",
    "read_checkpoint_metadata": "symtrs.common.checkpoint",
    "save_checkpoint": "symtrs.common.checkpoint",
    "cleanup_ddp": "symtrs.common.ddp",
    "is_main_process": "symtrs.common.ddp",
    "setup_ddp": "symtrs.common.ddp",
    "world_size_from_env": "symtrs.common.ddp",
    "AverageMeter": "symtrs.common.meters",
    "MetricTracker": "symtrs.common.meters",
    "save_curves": "symtrs.common.plots",
    "save_depth_preview": "symtrs.common.plots",
    "save_segmentation_preview": "symtrs.common.plots",
    "resolve_device": "symtrs.common.seed",
    "set_seed": "symtrs.common.seed",
    "append_csv_row": "symtrs.common.table",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name)
    try:
        value = getattr(module, name)
    except AttributeError as exc:
        raise AttributeError(f"{module_name} does not export {name!r}") from exc
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    "CHECKPOINT_KEY",
    "AverageMeter",
    "MetricTracker",
    "append_csv_row",
    "cleanup_ddp",
    "is_main_process",
    "load_checkpoint",
    "read_checkpoint_metadata",
    "resolve_device",
    "save_checkpoint",
    "save_curves",
    "save_depth_preview",
    "save_segmentation_preview",
    "set_seed",
    "setup_ddp",
    "world_size_from_env",
]
