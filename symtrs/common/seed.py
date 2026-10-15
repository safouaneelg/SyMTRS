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
    return torch.device(requested)


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        return f"cuda:{index} ({torch.cuda.get_device_name(index)})"
    return str(device)
