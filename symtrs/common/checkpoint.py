"""Checkpoint format.

Every SyMTRS checkpoint is self-describing. The `symtrs` block records the task,
model key, resolution, depth protocol, normalisation statistics and the fully
resolved run config, so `autoloader` can rebuild the architecture and reproduce the
postprocessing without the training YAML being present.

Layout:

    {
      "symtrs": {format_version, task, model_key, resolution, protocol, ...,
                 "config": {...resolved RunConfig...}},
      "model": state_dict,
      "optimizer": state_dict | None,
      "scheduler": state_dict | None,
      "epoch": int,
      "best": float,
      "metrics": {...},
    }
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch

from symtrs import CONFIG_FORMAT_VERSION

CHECKPOINT_KEY = "symtrs"


class CheckpointError(RuntimeError):
    """Raised when a file is not a readable SyMTRS checkpoint."""


def save_checkpoint(
    path: str,
    state: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> None:
    """Write a checkpoint, stamping the metadata block with the format version."""
    directory = os.path.dirname(str(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    payload = dict(state)
    block = dict(metadata)
    block.setdefault("format_version", CONFIG_FORMAT_VERSION)
    payload[CHECKPOINT_KEY] = block
    # Atomic-ish write: a crash mid-save must not leave a half-written "best.pt".
    tmp = f"{path}.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str, map_location: Any = "cpu") -> Dict[str, Any]:
    if not Path(path).is_file():
        raise CheckpointError(f"checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # torch < 2.0 has no weights_only
        payload = torch.load(path, map_location=map_location)
    except Exception as exc:  # pragma: no cover - depends on the corrupt file
        raise CheckpointError(f"could not read checkpoint {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CheckpointError(f"checkpoint {path} does not contain a dict")
    return payload


def read_checkpoint_metadata(path: str, map_location: Any = "cpu") -> Dict[str, Any]:
    """Read just the `symtrs` metadata block.

    Falls back to reading the whole file on older torch versions that cannot map a
    subset of tensors, so callers always get a dict or a clear error.
    """
    payload = load_checkpoint(path, map_location=map_location)
    block = payload.get(CHECKPOINT_KEY)
    if block is None:
        raise CheckpointError(
            f"{path} has no {CHECKPOINT_KEY!r} metadata block, so it was not written "
            "by symtrs. Upstream checkpoints (CycleGAN/pix2pix) are loaded through "
            "the domain_adapt task instead."
        )
    return block


def is_symtrs_checkpoint(path: str) -> bool:
    try:
        read_checkpoint_metadata(path)
        return True
    except Exception:
        return False


def resolve_checkpoint_path(path: str) -> Optional[str]:
    """Accept a file or a run directory; return the checkpoint file or None.

    A run directory is searched for `weights/best.pt` then `weights/last.pt`, which
    is where the trainer writes them.
    """
    candidate = Path(path)
    if candidate.is_file():
        return str(candidate)
    if candidate.is_dir():
        for relative in ("weights/best.pt", "weights/last.pt", "best.pt", "last.pt"):
            nested = candidate / relative
            if nested.is_file():
                return str(nested)
    return None
