import os

import torch
from PIL import Image


def _to_uint8(img: torch.Tensor) -> torch.Tensor:
    img = img.detach().cpu()
    if img.dim() == 4:
        img = img[0]
    img = img.clamp(0.0, 1.0)
    img = (img * 255.0).round().to(torch.uint8)
    return img


def save_image_grid(images: torch.Tensor, path: str, nrow: int = 8, value_range: str = "zero_one") -> None:
    """Save image batch [B,C,H,W] as a simple grid using PIL."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    imgs = images.detach().cpu()
    if value_range == "minus_one_one":
        imgs = (imgs + 1.0) * 0.5
    imgs = imgs.clamp(0.0, 1.0)

    b, c, h, w = imgs.shape
    nrow = max(1, min(nrow, b))
    ncol = (b + nrow - 1) // nrow

    canvas = torch.zeros((c, ncol * h, nrow * w), dtype=torch.float32)
    for i in range(b):
        r = i // nrow
        col = i % nrow
        canvas[:, r * h : (r + 1) * h, col * w : (col + 1) * w] = imgs[i]

    canvas = _to_uint8(canvas)
    if canvas.size(0) == 1:
        arr = canvas.squeeze(0).numpy()
        Image.fromarray(arr, mode="L").save(path)
    else:
        arr = canvas.permute(1, 2, 0).numpy()
        Image.fromarray(arr, mode="RGB").save(path)
