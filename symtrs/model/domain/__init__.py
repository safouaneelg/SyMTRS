"""Domain adaptation: day-to-night and night-to-day translation.

This is the one task where SyMTRS does not own the network. `upstream.py` prepares and
drives the original `junyanz/pytorch-CycleGAN-and-pix2pix` code; `runner.py` wires it to
the SyMTRS dataset layout and scores the output with the shared FID/KID implementation.

    symtrs train --task domain_adapt --model cyclegan --data-root /data/SyMTRS \\
                 --set model_kwargs.direction=day2night

The complete upstream source tree is vendored in this repository under
`pytorch-CycleGAN-and-pix2pix/`. Nothing is cloned at run time, so a domain-adaptation
run works offline; `symtrs setup-adapt` verifies the tree and `--fetch` re-derives the
delta against the pinned revision.
"""

from __future__ import annotations

from symtrs.model.domain.runner import (
    DIRECTIONS,
    UPSTREAM_MODELS,
    DomainAdaptation,
    evaluate_adaptation,
    run_domain_adaptation,
)
from symtrs.model.domain.upstream import (
    UPSTREAM_REPO,
    UPSTREAM_REVISION,
    UpstreamError,
    build_staging_dir,
    default_upstream,
    is_complete,
    setup,
    verify,
)

__all__ = [
    "DIRECTIONS",
    "DomainAdaptation",
    "UPSTREAM_MODELS",
    "UPSTREAM_REPO",
    "UPSTREAM_REVISION",
    "UpstreamError",
    "build_staging_dir",
    "default_upstream",
    "evaluate_adaptation",
    "is_complete",
    "run_domain_adaptation",
    "setup",
    "verify",
]
