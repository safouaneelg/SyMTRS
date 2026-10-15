"""The super-resolution implementation, launched - never imported - by `symtrs`.

`symtrs.model.superres.runner` builds a command line from a `RunConfig` and runs
`python -m symtrs.legacy.superres.train`; it does not import a single class from here. That
keeps this tree's state - its RNG, its matplotlib backend, its `torch.multiprocessing`
start-up - inside its own process, and a crash in it out of the CLI's.

    models/   the four networks, as they were at `models/superresolution/`
    utils/    losses, metrics, image and training helpers, as they were at
              `utils/superresolution/`
    train.py  the training and scoring loop, as it was at `train_superresolution.py`

This module deliberately imports nothing, so `import symtrs.legacy.superres` costs nothing.
"""

from __future__ import annotations
