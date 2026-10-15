"""Super-resolution: ×1, ×2, ×4 and ×8 from the `_lr` folders.

The models, the loading and the metrics predate this package and now live inside it, at
`symtrs/legacy/superres/` and `symtrs/legacy/dataloader/`. `runner.py` maps a `RunConfig`
onto the trainer's command line (`python -m symtrs.legacy.superres.train`), launches it as a
subprocess, and folds its test metrics into the `summary.json` every other task writes.

    symtrs train configs/superresolution/srcnn_x4.yaml --data-root /data/SyMTRS

Nothing here reimplements the models: a second copy of SwinIR in this package would be a
second thing to keep in step with the checkpoints, which is exactly the failure mode the
rest of the package avoids by keeping one registry.

`resolution` names the input scale, as it does for every other task - 2048 native,
1024 `_lr/x2`, 512 `_lr/x4`, 256 `_lr/x8` - and the upsampling ratio follows from it
rather than being configured separately.
"""

from __future__ import annotations

from symtrs.model.superres.runner import (
    SCALE_FOR_RESOLUTION,
    SR_MODELS,
    TRAIN_MODULE,
    SuperResolution,
    SuperResolutionError,
    expected_test_dirs,
    hr_dir,
    lr_dir,
    run_super_resolution,
    scale_for,
    summary_from_run,
    trainer_missing,
)

__all__ = [
    "SCALE_FOR_RESOLUTION",
    "SR_MODELS",
    "TRAIN_MODULE",
    "SuperResolution",
    "SuperResolutionError",
    "expected_test_dirs",
    "hr_dir",
    "lr_dir",
    "run_super_resolution",
    "scale_for",
    "summary_from_run",
    "trainer_missing",
]
