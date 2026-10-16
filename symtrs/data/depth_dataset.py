"""Depth dataset over the revised SyMTRS layout.

Returns metric depth in metres plus a validity mask. The mask is not a convenience:
rendered depth marks sky and missing geometry with zero, and treating those pixels
as a distance of zero metres would dominate every loss and metric.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from symtrs.data.exr import read_depth_file, depth_to_tensor
from symtrs.data.io import list_images, check_pairing, match_by_stem, read_image, coverage_report
from symtrs.data.splits import SplitDirs, resolve_split_dirs
from symtrs.data.transforms import apply_crop, flip_tensors, match_spatial, sample_crop_params

DEPTH_EXTENSIONS = (".exr", ".npy")


class DepthTaskDataset(Dataset):
    """Image -> metric depth pairs at one resolution.

    Yields:
        image: float32 `[3, H, W]` in `[0, 1]`, colour-normalised by the model adapter,
            not here, so the dataset stays model-agnostic.
        depth: float32 `[1, H, W]` in metres. Invalid pixels hold 0.0.
        valid: bool `[1, H, W]`, False for non-finite or non-positive depth.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        resolution: int = 2048,
        depth_scale: float = 0.01,
        max_depth: Optional[float] = None,
        depth_channel: int = 0,
        depth_backend: str = "auto",
        max_samples: int = 0,
        patch_size: int = 0,
        augment: bool = False,
        horizontal_flip: bool = True,
        vertical_flip: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        self.dirs: SplitDirs = resolve_split_dirs(root, split, resolution)
        self.split = split
        self.resolution = resolution
        self.depth_scale = float(depth_scale)
        self.max_depth = None if max_depth is None else float(max_depth)
        self.depth_channel = int(depth_channel)
        self.depth_backend = depth_backend
        self.patch_size = int(patch_size)
        self.augment = bool(augment)
        self.horizontal_flip = bool(horizontal_flip)
        self.vertical_flip = bool(vertical_flip)
        self.generator = generator

        day_dir = self.dirs.path_for("day")
        depth_dir = self.dirs.path_for("depth")

        images = list_images(day_dir)
        depths = _list_depth(depth_dir)
        pairs = match_by_stem(images, depths)

        check_pairing(pairs, images, depths, "image/depth", split, resolution)
        if not pairs:
            raise RuntimeError(
                f"no image/depth pairs for split={split!r} resolution={resolution}.\n"
                f"  images: {day_dir} ({len(images)} files)\n"
                f"  depth:  {depth_dir} ({len(depths)} files)\n"
                "  run `symtrs doctor --data-root <dataset root>` for a coverage report"
            )

        if max_samples and max_samples > 0:
            pairs = pairs[:max_samples]
        self.pairs: List[Tuple[Path, Path]] = pairs

    # ------------------------------------------------------------------ dunder

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        # Advance past unreadable samples rather than killing a long run on one
        # corrupt file. Bounded so a genuinely broken split still fails loudly.
        attempts = 0
        while attempts < len(self.pairs):
            image_path, depth_path = self.pairs[index]
            try:
                image = read_image(image_path, mode="rgb")
                raw = read_depth_file(
                    depth_path, channel=self.depth_channel, backend=self.depth_backend
                )
                metres, valid = depth_to_tensor(
                    raw, scale=self.depth_scale, max_depth=self.max_depth
                )
                break
            except Exception as exc:
                print(
                    f"[symtrs.data] skipping unreadable pair image={image_path} "
                    f"depth={depth_path}: {type(exc).__name__}: {exc}"
                )
                attempts += 1
                index = (index + 1) % len(self.pairs)
        else:
            raise RuntimeError(
                f"all {len(self.pairs)} depth pairs for split={self.split!r} "
                f"resolution={self.resolution} failed to load"
            )

        depth = torch.from_numpy(metres).float().unsqueeze(0)
        valid_t = torch.from_numpy(valid).bool().unsqueeze(0)

        image, depth, valid_t = match_spatial([image, depth, valid_t])

        height, width = image.shape[-2:]
        box = sample_crop_params(height, width, self.patch_size, self.generator)
        if self.augment:
            horizontal = self.horizontal_flip and torch.rand(1, generator=self.generator).item() < 0.5
            vertical = self.vertical_flip and torch.rand(1, generator=self.generator).item() < 0.5
            image, depth, valid_t = flip_tensors([image, depth, valid_t], horizontal, vertical)
        image, depth, valid_t = apply_crop([image, depth, valid_t], box)

        return {
            "image": image,
            "depth": depth,
            "valid": valid_t,
            "image_path": str(image_path),
            "depth_path": str(depth_path),
        }


def _list_depth(root: Path) -> List[Path]:
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.suffix.lower() in DEPTH_EXTENSIONS)


def depth_coverage(root: str, split: str = "train", resolution: int = 2048) -> dict:
    """Stem-match report between images and depth, for `symtrs inspect`."""
    dirs = resolve_split_dirs(root, split, resolution)
    return coverage_report(list_images(dirs.day), _list_depth(dirs.depth))
