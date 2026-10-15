"""Compatibility shim: the package moved into `symtrs`; this keeps the imports working.

`dataloader/` was this repository's own code, not a third-party tree, and it now lives at
`symtrs/legacy/dataloader/` so that it ships with the package. The imports the README and
`symtrs/legacy/dataloader/README.md` document keep resolving through this file, which
re-exports and adds nothing:

    from dataloader import SuperResDataset, read_image

New code should import `symtrs.legacy.dataloader` directly. The depth and domain-adaptation
datasets moved with the rest: nothing in `symtrs` uses them, and they are kept because
scripts outside this repository may.
"""

from __future__ import annotations

from symtrs.legacy.dataloader import (
    DepthDataset,
    DomainAdaptDataset,
    SuperResDataset,
    read_image,
    split_indices,
    split_list,
)

__all__ = [
    "read_image",
    "split_indices",
    "split_list",
    "DepthDataset",
    "SuperResDataset",
    "DomainAdaptDataset",
]
