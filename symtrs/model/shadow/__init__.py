"""Shadow removal: ShadowFormer, Restormer, Uformer and ShadowMaskFormer on `rgb_day` -> `rgb_noshadow`.

    symtrs train configs/shadow_removal/restormer.yaml --data-root /data/SyMTRS \
                 --set shadow.iterations=500

Like domain adaptation and super-resolution this task does not own its networks: the
classes come from the upstream checkouts under `third_party/` (pinned in
`third_party/PINS.md`, recreated by `scripts/fetch_third_party.py`). `symtrs` owns the
folder mapping (`staging.py`), the launch (`runner.py`, `worker.py`) and the scoring
(`symtrs.metrics.shadow`), and folds the result into the usual `summary.json`.
"""

from __future__ import annotations

from symtrs.model.shadow.archs import ARCHS, Arch, arch_for
from symtrs.model.shadow.runner import (
    ShadowRemoval,
    run_shadow_removal,
    worker_command,
)
from symtrs.model.shadow.staging import ShadowStagingError, stage_split

__all__ = [
    "ARCHS",
    "Arch",
    "ShadowRemoval",
    "ShadowStagingError",
    "arch_for",
    "run_shadow_removal",
    "stage_split",
    "worker_command",
]
