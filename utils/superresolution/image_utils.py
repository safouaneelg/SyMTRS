# E:\utils\superresolution\image_utils.py
import os
from typing import Tuple

import numpy as np
import torch
from PIL import Image


def rgb_to_y(img: torch.Tensor) -> torch.Tensor:
    """Convert RGB tensor [B,3,H,W] or [3,H,W] to Y channel [B,1,H,W]."""
    if img.dim() == 3:
        img = img.unsqueeze(0)
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return y


def match_size(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Center-crop a and b to the same spatial size (min of each)."""
    h = min(a.shape[-2], b.shape[-2])
    w = min(a.shape[-1], b.shape[-1])

    def crop(x):
        top = (x.shape[-2] - h) // 2
        left = (x.shape[-1] - w) // 2
        return x[..., top : top + h, left : left + w]

    return crop(a), crop(b)


def _to_uint8(img: torch.Tensor) -> np.ndarray:
    img = img.detach().cpu().clamp(0, 1)
    if img.dim() == 4:
        img = img[0]
    img = img.permute(1, 2, 0).numpy()
    if img.shape[2] == 1:
        img = img[:, :, 0]
    img = (img * 255.0).round().astype(np.uint8)
    return img


def save_image_triplet(lr: torch.Tensor, sr: torch.Tensor, hr: torch.Tensor, path: str) -> None:
    """Save LR/SR/HR side-by-side for quick visual checks."""
    lr_img = _to_uint8(lr)
    sr_img = _to_uint8(sr)
    hr_img = _to_uint8(hr)

    h = max(lr_img.shape[0], sr_img.shape[0], hr_img.shape[0])
    imgs = [lr_img, sr_img, hr_img]
    padded = []
    for im in imgs:
        if im.shape[0] != h:
            pad_h = h - im.shape[0]
            if im.ndim == 2:
                im = np.pad(im, ((0, pad_h), (0, 0)), mode="edge")
            else:
                im = np.pad(im, ((0, pad_h), (0, 0), (0, 0)), mode="edge")
        padded.append(im)

    cat = np.concatenate(padded, axis=1)
    Image.fromarray(cat).save(path)
