# E:\dataloader\superres_dataset.py
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional

import torch
from torch.utils.data import Dataset

from .utils import list_images, match_by_stem, read_image


class SuperResDataset(Dataset):
    def __init__(
        self,
        lr_root: str | Path,
        hr_root: str | Path,
        image_transform: Optional[Callable] = None,
        lr_transform: Optional[Callable] = None,
        hr_transform: Optional[Callable] = None,
    ) -> None:
        self.lr_root = Path(lr_root)
        self.hr_root = Path(hr_root)
        self.image_transform = image_transform
        self.lr_transform = lr_transform
        self.hr_transform = hr_transform

        lr_files = list_images(self.lr_root)
        hr_files = list_images(self.hr_root)
        pairs = match_by_stem(lr_files, hr_files)
        self.pairs = pairs

        if len(self.pairs) == 0:
            raise RuntimeError("No LR/HR pairs found. Check filenames and roots.")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Skip corrupt images by advancing to the next pair.
        attempts = 0
        start_idx = idx
        while attempts < len(self.pairs):
            lr_path, hr_path = self.pairs[idx]
            try:
                lr = read_image(lr_path, mode="rgb")
                hr = read_image(hr_path, mode="rgb")
                break
            except Exception as exc:
                print(f"[superres_dataset] Skipping corrupt pair: lr={lr_path} hr={hr_path} err={exc}")
                attempts += 1
                idx = (idx + 1) % len(self.pairs)
        else:
            raise RuntimeError("All image pairs appear corrupt; cannot load any samples.")

        if self.image_transform:
            lr = self.image_transform(lr)
            hr = self.image_transform(hr)
        if self.lr_transform:
            lr = self.lr_transform(lr)
        if self.hr_transform:
            hr = self.hr_transform(hr)

        return {
            "lr": lr,
            "hr": hr,
            "lr_path": str(lr_path),
            "hr_path": str(hr_path),
        }
