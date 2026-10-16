"""Reproducibility: seeding and device selection."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed python, numpy and torch.

    `deterministic` additionally forces cuDNN into deterministic mode. It is off by
    default because it costs throughput and some ops have no deterministic kernel;
    turn it on when reproducing a specific number.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(requested: str = "cuda") -> torch.device:
    """Return a usable device, falling back to CPU with a warning.

    Asking for cuda on a machine without it is a common way to lose an hour, so it
    warns loudly rather than raising.
    """
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(
            "[symtrs] WARNING: cuda was requested but is not available; "
            "falling back to CPU. Training will be very slow."
        )
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda":
        apply_memory_guard(device)
    return device


# Environment, not RunConfig: this is a property of the machine, it never changes a number a
# run produces, and it has to reach every process that picks a device (the trainer, the
# shadow-removal worker) without each config carrying it.
MEMORY_FRACTION_ENV = "SYMTRS_CUDA_MEM_FRACTION"


def apply_memory_guard(device: torch.device) -> None:
    """Cap this process's CUDA allocations at a fraction of the card, if asked to.

    On Windows (WDDM) the NVIDIA driver does not raise out-of-memory when a run outgrows
    the card: it silently spills into system RAM ("sysmem fallback"). A U-Net at batch 32
    on a 12 GB laptop GPU wanted 18.6 GB and ran 100-500x slower instead of failing, which
    reads as a broken data loader. With `SYMTRS_CUDA_MEM_FRACTION=0.92` the caching
    allocator raises a plain OOM before the driver ever spills, so the fix (a smaller
    `--batch-size`) is obvious from the log.
    """
    raw = os.environ.get(MEMORY_FRACTION_ENV, "").strip()
    if not raw:
        return
    fraction = float(raw)
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"{MEMORY_FRACTION_ENV}={raw!r} must be in (0, 1]")
    index = device.index if device.index is not None else torch.cuda.current_device()
    torch.cuda.set_per_process_memory_fraction(fraction, index)
    total = torch.cuda.get_device_properties(index).total_memory / 2**30
    print(f"[symtrs] CUDA memory capped at {fraction:.0%} of {total:.1f} GiB "
          f"({MEMORY_FRACTION_ENV}); exceeding it raises OOM instead of spilling to RAM")


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        return f"cuda:{index} ({torch.cuda.get_device_name(index)})"
    return str(device)
