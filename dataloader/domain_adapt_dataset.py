# E:\dataloader\domain_adapt_dataset.py
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional

import torch
from torch.utils.data import Dataset

from .utils import list_images, match_by_stem, read_image


class DomainAdaptDataset(Dataset):
    def __init__(
        self,
        day_root: str | Path,
        night_root: str | Path,
        paired: bool = False,
        image_transform: Optional[Callable] = None,
    ) -> None:
        self.day_root = Path(day_root)
        self.night_root = Path(night_root)
        self.paired = paired
        self.image_transform = image_transform

        self.day_files = list_images(self.day_root)
        self.night_files = list_images(self.night_root)

        if paired:
            self.pairs = match_by_stem(self.day_files, self.night_files)
            if len(self.pairs) == 0:
                raise RuntimeError("No day/night pairs found. Check filenames and roots.")
        else:
            if len(self.day_files) == 0 or len(self.night_files) == 0:
                raise RuntimeError("Day or night folder is empty.")
            self.pairs = None

    def __len__(self) -> int:
        if self.paired:
            return len(self.pairs)
        return max(len(self.day_files), len(self.night_files))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.paired:
            day_path, night_path = self.pairs[idx]
        else:
            day_path = self.day_files[idx % len(self.day_files)]
            night_path = self.night_files[idx % len(self.night_files)]

        day = read_image(day_path, mode="rgb")
        night = read_image(night_path, mode="rgb")

        if self.image_transform:
            day = self.image_transform(day)
            night = self.image_transform(night)

        return {
            "day": day,
            "night": night,
            "day_path": str(day_path),
            "night_path": str(night_path),
        }
