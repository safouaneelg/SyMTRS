# E:\dataloader\utils.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch
from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(root: str | Path) -> List[Path]:
    root = Path(root)
    files = [p for p in root.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    return sorted(files)


def list_npys(root: str | Path) -> List[Path]:
    root = Path(root)
    files = [p for p in root.iterdir() if p.suffix.lower() == ".npy"]
    return sorted(files)


def _to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    # Use owned tensor storage for DataLoader worker collation stability.
    tensor = torch.tensor(arr, dtype=torch.float32).permute(2, 0, 1).contiguous() / 255.0
    return tensor


def read_image(path: str | Path, mode: str = "rgb") -> torch.Tensor:
    """Read an image as float tensor in [0,1], shape [C,H,W]."""
    path = Path(path)
    img = Image.open(path)
    if mode.lower() == "rgb":
        img = img.convert("RGB")
    elif mode.lower() in {"l", "gray", "grayscale"}:
        img = img.convert("L")
    return _to_tensor(img)


def _finite_minmax(arr: np.ndarray) -> Tuple[float, float]:
    finite = np.isfinite(arr)
    if not np.any(finite):
        return 0.0, 1.0
    vmin = float(np.min(arr[finite]))
    vmax = float(np.max(arr[finite]))
    return vmin, vmax


def load_depth_npy(path: str | Path, normalize: str = "raw") -> torch.Tensor:
    """Load depth .npy as float tensor [1,H,W].

    normalize: "raw" or "per_image_minmax"
    """
    path = Path(path)
    arr = np.load(path)
    arr = arr.astype(np.float32)

    if normalize.lower() == "per_image_minmax":
        vmin, vmax = _finite_minmax(arr)
        denom = max(vmax - vmin, 1e-6)
        arr = (arr - vmin) / denom
    elif normalize.lower() != "raw":
        raise ValueError("normalize must be 'raw' or 'per_image_minmax'")

    if arr.ndim == 2:
        arr = arr[None, :, :]
    # Clone into owned, contiguous storage to avoid non-resizable NumPy-backed tensors.
    return torch.tensor(arr, dtype=torch.float32).contiguous()


def match_by_stem(
    a_files: Iterable[Path],
    b_files: Iterable[Path],
    a_stem_fn=None,
    b_stem_fn=None,
) -> List[Tuple[Path, Path]]:
    a_stem_fn = a_stem_fn or (lambda p: p.stem)
    b_stem_fn = b_stem_fn or (lambda p: p.stem)

    b_map = {b_stem_fn(p): p for p in b_files}
    pairs = []
    for a in a_files:
        key = a_stem_fn(a)
        b = b_map.get(key)
        if b is not None:
            pairs.append((a, b))
    return pairs
