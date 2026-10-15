"""Code that predates this package, moved inside it unchanged.

Two things live here:

  * `superres/` - the super-resolution task: four networks, their losses and metrics, the
    dataset loader and the training loop that `symtrs.model.superres` launches.
  * `dataloader/` - the small PyTorch datasets the trainer reads, plus two datasets (depth,
    domain adaptation) that nothing in this package uses.

The rule for this tree is that it was **moved, not ported**. Every file is byte-identical
to the commit that created it, because the claim a reader has to be able to check is that
the network behind a published checkpoint is the network in the paper - and a relocation
keeps that checkable while a tidy-up does not. `README.md` beside this file maps every old
path to its new one and gives the commands that verify the claim.

Nothing here imports anything at import time, and nothing outside this tree may import from
it: `symtrs.model.superres` launches the trainer as a subprocess rather than calling into
it, and the rest of the package works whether this tree exists or not.
"""

from __future__ import annotations
