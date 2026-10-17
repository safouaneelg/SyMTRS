"""Shared fixtures.

Two kinds of fixture live here, split by what they need:

  * A **layout fixture** that only creates directories. Everything about the resolution
    axis, the split names and the `_lr/x2` suffixes is pure pathlib, so those tests run
    anywhere, including on a machine with no torch - which is the situation you are in
    when you are checking that a freshly generated dataset has the right shape.

  * A **data fixture** that writes actual `.npy` depth maps and label arrays and builds a
    `RunConfig` over them. Those tests need torch and skip when it is absent.

`requires_torch` is a skip marker rather than a hard import at module scope on purpose: a
collection error in a file that cannot import torch would hide the torch-free tests in the
same run, and those are the ones that catch a mis-generated dataset.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: Resolutions and the directory suffix each one reads, mirroring the dataset layout.
RESOLUTION_SUFFIX = {2048: None, 1024: "x2", 512: "x4", 256: "x8"}


def _have(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


HAVE_TORCH = _have("torch")
HAVE_NUMPY = _have("numpy")
HAVE_PIL = _have("PIL")

requires_torch = pytest.mark.skipif(not HAVE_TORCH, reason="torch is not installed")
requires_numpy = pytest.mark.skipif(not HAVE_NUMPY, reason="numpy is not installed")
requires_pil = pytest.mark.skipif(not HAVE_PIL, reason="Pillow is not installed")

#: Anything that builds a model downloads weights. Opt in with `-m network` explicitly.
network = pytest.mark.network


@pytest.fixture
def layout(tmp_path: Path) -> Path:
    """An empty but correctly shaped SyMTRS dataset root.

    Directories only - no images, no depth, no labels. This is what `symtrs inspect`
    and a dry run look at, and it is what the resolution-mapping tests assert against.
    """
    root = tmp_path / "SyMTRS"
    for split in ("train", "val", "test"):
        split_root = root / split
        (split_root / "rgb_day").mkdir(parents=True)
        (split_root / "rgb_night").mkdir(parents=True)
        (split_root / "rgb_noshadow").mkdir(parents=True)
        (split_root / "depth").mkdir(parents=True)
        (split_root / "semantic").mkdir(parents=True)
        for suffix in ("x2", "x4", "x8"):
            (split_root / "rgb_day_lr" / suffix).mkdir(parents=True)
            (split_root / "depth_lr" / suffix).mkdir(parents=True)
            (split_root / "semantic_lr" / suffix).mkdir(parents=True)
    return root


@pytest.fixture
def partial_layout(tmp_path: Path) -> Path:
    """A dataset that is missing everything below native resolution.

    The shape a dataset has while it is still being generated, which is exactly when you
    want `symtrs inspect` to tell you what exists instead of raising.
    """
    root = tmp_path / "SyMTRS_partial"
    for split in ("train", "val", "test"):
        split_root = root / split
        (split_root / "rgb_day").mkdir(parents=True)
        (split_root / "rgb_night").mkdir(parents=True)
        (split_root / "depth").mkdir(parents=True)
        (split_root / "semantic").mkdir(parents=True)
        (split_root / "rgb_day_lr" / "x2").mkdir(parents=True)
    return root
