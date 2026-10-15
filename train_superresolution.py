"""Compatibility shim: the trainer moved into the package; this keeps the old command.

The models, the loading, the metrics and the loop now live at
`symtrs/legacy/superres/train.py` (`symtrs/legacy/README.md` has the full mapping), so an
installed `symtrs` contains its own super-resolution task instead of one that only runs
from a checkout. This file stays because it is the command in the README, in the configs and
in the author's habits:

    python train_superresolution.py --lr_root ... --hr_root ... --model srcnn --scale 4

It contains no logic of its own - everything is the imported `main` - and no `sys.path`
work, because a script's own directory is already on the path, so `symtrs` is importable
wherever this file is run from.
"""

from __future__ import annotations

import sys

from symtrs.legacy.superres.train import main

if __name__ == "__main__":
    sys.exit(main())
