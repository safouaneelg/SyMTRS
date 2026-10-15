"""Training and evaluation engine.

    tiling.py   sliding-window inference for images too large for one forward pass
    loop.py     one training epoch, one evaluation pass, task-agnostic
    trainer.py  optimiser, schedule, AMP, checkpoints, CSV, plots, distributed launch
"""

from __future__ import annotations

from symtrs.engine.loop import (
    evaluate,
    evaluate_depth,
    evaluate_segmentation,
    format_metrics,
    forward_native,
    train_one_epoch,
)
from symtrs.engine.tiling import plan_tiling, predict_tiled, tile_grid
from symtrs.engine.trainer import (
    build_optimizer,
    build_scheduler,
    checkpoint_metadata,
    evaluate_run,
    resolve_monitor,
    run,
    train,
)

__all__ = [
    "build_optimizer",
    "build_scheduler",
    "checkpoint_metadata",
    "evaluate",
    "evaluate_depth",
    "evaluate_run",
    "evaluate_segmentation",
    "format_metrics",
    "forward_native",
    "plan_tiling",
    "predict_tiled",
    "resolve_monitor",
    "run",
    "tile_grid",
    "train",
    "train_one_epoch",
]
