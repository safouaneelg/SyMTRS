"""Image reading and file discovery for the SyMTRS layout."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

#: Infix tokens that some exports insert between the scene id and the index, e.g.
#: `RS.depth.100200.exr` for image `RS.100200.png`. Stripped before matching.
_DECORATION_TOKENS = (".depth.", ".semantic.", ".seg.", ".label.")


class ImageReadError(RuntimeError):
    """Raised when an image cannot be read."""


def list_images(root) -> List[Path]:
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def read_image(path, mode: str = "rgb") -> torch.Tensor:
    """Read an image as a float32 tensor `[C, H, W]` in `[0, 1]`."""
    path = Path(path)
    try:
        image = Image.open(path)
        if mode.lower() == "rgb":
            image = image.convert("RGB")
        elif mode.lower() in {"l", "gray", "grayscale"}:
            image = image.convert("L")
        array = np.array(image)
    except Exception as exc:
        raise ImageReadError(f"could not read {path}: {exc}") from exc

    if array.ndim == 2:
        array = array[:, :, None]
    tensor = torch.from_numpy(np.ascontiguousarray(array)).float()
    return tensor.permute(2, 0, 1).contiguous() / 255.0


def read_label_map(path) -> torch.Tensor:
    """Read a grayscale label PNG as an int64 tensor `[H, W]`.

    Values are taken verbatim from the pixel data; no rescaling happens, because
    silently turning label 3 into label 1 would be undetectable downstream. A
    palette-mode PNG is converted to its palette indices instead of its RGB values.
    """
    path = Path(path)
    try:
        image = Image.open(path)
        if image.mode == "P":
            array = np.array(image)
        elif image.mode in {"I;16", "I", "L"}:
            array = np.array(image)
        else:
            array = np.array(image.convert("L"))
    except Exception as exc:
        raise ImageReadError(f"could not read label map {path}: {exc}") from exc

    if array.ndim == 3:
        array = array[..., 0]
    return torch.from_numpy(np.ascontiguousarray(array)).long()


def normalize_stem(stem: str) -> str:
    """Strip export decoration so depth/label files match their RGB image."""
    for token in _DECORATION_TOKENS:
        stem = stem.replace(token, ".")
    return stem.strip(".")


def match_by_stem(
    primary: List[Path],
    secondary: List[Path],
    primary_key=None,
    secondary_key=None,
) -> List[Tuple[Path, Path]]:
    """Pair files whose normalised stems are equal, preserving `primary` order."""
    primary_key = primary_key or (lambda p: normalize_stem(p.stem))
    secondary_key = secondary_key or (lambda p: normalize_stem(p.stem))

    lookup: Dict[str, Path] = {}
    duplicates = 0
    for path in secondary:
        key = secondary_key(path)
        if key in lookup:
            duplicates += 1
            continue
        lookup[key] = path
    if duplicates:
        print(
            f"[symtrs.data] WARNING: {duplicates} file(s) in the secondary folder "
            "share a stem with another and were ignored"
        )

    pairs = []
    for path in primary:
        partner = lookup.get(primary_key(path))
        if partner is not None:
            pairs.append((path, partner))
    return pairs


def coverage_report(primary: List[Path], secondary: List[Path]) -> dict:
    """How well two folders line up, for `symtrs inspect`."""
    primary_keys = {normalize_stem(p.stem) for p in primary}
    secondary_keys = {normalize_stem(p.stem) for p in secondary}
    matched = primary_keys & secondary_keys
    return {
        "primary": len(primary_keys),
        "secondary": len(secondary_keys),
        "matched": len(matched),
        "primary_only": len(primary_keys - secondary_keys),
        "secondary_only": len(secondary_keys - primary_keys),
        "primary_only_examples": sorted(primary_keys - secondary_keys)[:5],
        "secondary_only_examples": sorted(secondary_keys - primary_keys)[:5],
    }
