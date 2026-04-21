# E:\dataloader\depth_dataset.py
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from .utils import list_images, list_npys, load_depth_npy, match_by_stem, read_image


def _depth_to_image_stem(p: Path) -> str:
    # Example: RS.depth.100200.npy -> RS.100200
    stem = p.stem
    return stem.replace(".depth.", ".")


class DepthDataset(Dataset):
    def __init__(
        self,
        depth_root: str | Path,
        hr_root: str | Path,
        normalize_depth: str = "raw",
        image_transform: Optional[Callable] = None,
        depth_transform: Optional[Callable] = None,
    ) -> None:
        self.depth_root = Path(depth_root)
        self.hr_root = Path(hr_root)
        self.normalize_depth = normalize_depth
        self.image_transform = image_transform
        self.depth_transform = depth_transform

        depth_files = list_npys(self.depth_root)
        hr_files = list_images(self.hr_root)
        pairs = match_by_stem(depth_files, hr_files, a_stem_fn=_depth_to_image_stem)
        self.pairs = pairs

        if len(self.pairs) == 0:
            raise RuntimeError("No depth/image pairs found. Check filenames and roots.")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        depth_path, image_path = self.pairs[idx]

        image = read_image(image_path, mode="rgb")
        depth = load_depth_npy(depth_path, normalize=self.normalize_depth)

        if self.image_transform:
            image = self.image_transform(image)
        if self.depth_transform:
            depth = self.depth_transform(depth)

        return {
            "image": image,
            "depth": depth,
            "image_path": str(image_path),
            "depth_path": str(depth_path),
        }
