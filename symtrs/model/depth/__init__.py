"""Depth adapters: one module per model family.

    from symtrs.model.depth import ZoeDepthAdapter

    model = ZoeDepthAdapter(variant="K")

Every adapter subclasses `DepthAdapter` (imported here for convenience) and adds the
five class attributes the rest of the benchmark reads without instantiating anything:
`protocol`, `default_align`, `native_input`, `size_multiple` and `requires`. `requires`
is what lets `symtrs doctor` explain a missing optional dependency instead of dying
with an ImportError trace.

Importing this package is cheap on purpose. `torch` and `numpy` are hard dependencies
and are imported at module level, but nothing here imports `transformers`, imports
`depth_pro`, or calls `torch.hub.load` until an adapter is constructed - so `import
symtrs`, the CLI and the registry all keep working in an environment where the heavy
extras are absent.
"""

from __future__ import annotations

from symtrs.model.depth.base import ALIGNMENT_NOTES, DepthAdapter
from symtrs.model.depth.da3 import DepthAnything3Adapter
from symtrs.model.depth.depth_pro import DepthProAdapter
from symtrs.model.depth.midas import MIDAS_MODELS, MiDaSAdapter
from symtrs.model.depth.zoedepth import ZoeDepthAdapter

__all__ = [
    "ALIGNMENT_NOTES",
    "DepthAdapter",
    "DepthAnything3Adapter",
    "DepthProAdapter",
    "MIDAS_MODELS",
    "MiDaSAdapter",
    "ZoeDepthAdapter",
]
