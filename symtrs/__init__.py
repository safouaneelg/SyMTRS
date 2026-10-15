"""SyMTRS multi-task benchmark toolkit.

The package is importable without torch installed: heavy imports live behind the
functions that need them, so `import symtrs` stays cheap and `symtrs --help`
works in a bare environment.

Typical use:

    from symtrs.data import dataloader
    from symtrs.model import autoloader

    loader = dataloader(task="depth", root="/data/SyMTRS", split="train", resolution=1024)
    model = autoloader("midas_dpt_hybrid")
    depth_metres = model.predict(image)
"""

from __future__ import annotations

__version__ = "0.1.0"

CONFIG_FORMAT_VERSION = 1

__all__ = ["__version__", "CONFIG_FORMAT_VERSION"]
