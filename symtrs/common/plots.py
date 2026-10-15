"""Plots and qualitative sample images.

Matplotlib is imported lazily so that importing `symtrs` in a headless environment
without a display backend configured does not fail at import time.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch


def save_curves(history: Sequence[Mapping], out_dir: str) -> None:
    """Write one PNG per numeric metric series, train and val on the same axes."""
    if not history:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    keys = [k for k in history[0] if k not in {"epoch"}]
    train_keys = [k for k in keys if k.startswith("train_")]
    metrics = [k[len("train_") :] for k in train_keys]
    epochs = [row.get("epoch", i + 1) for i, row in enumerate(history)]

    for name in metrics:
        train_key = f"train_{name}"
        val_key = f"val_{name}"
        has_val = val_key in history[0]
        if not has_val and not any(isinstance(row.get(train_key), (int, float)) for row in history):
            continue

        figure = plt.figure(figsize=(8, 4))
        if any(isinstance(row.get(train_key), (int, float)) for row in history):
            plt.plot(epochs, [row.get(train_key) for row in history], label=train_key)
        if has_val:
            plt.plot(epochs, [row.get(val_key) for row in history], label=val_key)
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel(name)
        plt.tight_layout()
        figure.savefig(out_dir / f"{name}_curve.png")
        plt.close(figure)


def _as_map(tensor: torch.Tensor) -> torch.Tensor:
    """A single `[H, W]` map, from however many leading batch/channel axes arrived.

    These three tensors come from three different places - a prediction out of the model, a
    ground truth and a validity mask out of the loader - and they do not agree on their rank:
    in the evaluation loop the prediction is `[1, H, W]` and the mask is `[H, W]`. Collapsing
    only some of them left `tgt` two-dimensional and `mask` three-dimensional, so
    `tgt[mask]` raised `IndexError: too many indices for tensor of dimension 2`, and every
    preview panel was lost. Normalising all three in one place is what keeps them agreeing.
    """
    while tensor.dim() > 2:
        tensor = tensor[0]
    return tensor


def _to_uint8(array: torch.Tensor) -> np.ndarray:
    array = array.detach().float().cpu().clamp(0, 1)
    if array.dim() == 3:
        array = array.permute(1, 2, 0)
    if array.dim() == 3 and array.shape[2] == 1:
        array = array[..., 0]
    if array.dim() == 3 and array.shape[2] == 3:
        return (array.numpy() * 255.0).round().astype(np.uint8)
    return (array.numpy() * 255.0).round().astype(np.uint8)


def colorize_depth(depth: np.ndarray, vmin: float, vmax: float, cmap: str = "turbo") -> np.ndarray:
    """A float depth map to an `(H, W, 3)` uint8 image on a fixed scale.

    The one place depth becomes a picture, so a preview panel and a `symtrs infer` colour
    map of the same field are comparable. `vmin`/`vmax` are passed in rather than derived
    here, because the two callers legitimately differ: a preview scales to the ground
    truth's valid quantile, an inference scales to the prediction's own.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    span = max(float(vmax) - float(vmin), 1e-6)
    normalized = np.clip((np.asarray(depth, dtype=np.float64) - vmin) / span, 0.0, 1.0)
    rgba = plt.get_cmap(cmap)(normalized)
    return (rgba[..., :3] * 255.0).round().astype(np.uint8)


def save_depth_preview(
    image: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    path: str,
) -> None:
    """RGB / predicted-depth / ground-truth-depth side by side, on a shared colour scale."""
    from PIL import Image

    pred = _as_map(prediction.detach().float().cpu())
    tgt = _as_map(target.detach().float().cpu())
    mask = _as_map(valid.detach().cpu().bool())

    finite = tgt[mask] if mask.any() else tgt
    finite = finite[torch.isfinite(finite)]
    if finite.numel() == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(finite.min())
        vmax = float(torch.quantile(finite, 0.99))
        if vmax <= vmin:
            vmax = vmin + 1e-6

    # Zero out invalid ground truth so holes are visually obvious rather than
    # pretending to be a real distance.
    tgt = torch.where(mask, tgt, torch.full_like(tgt, float("nan")))
    pred = torch.where(mask, pred, torch.full_like(pred, float("nan")))

    panels = [
        _to_uint8(image),
        colorize_depth(np.nan_to_num(pred.numpy(), nan=vmin), vmin, vmax),
        colorize_depth(np.nan_to_num(tgt.numpy(), nan=vmin), vmin, vmax),
    ]
    heights = max(panel.shape[0] for panel in panels)
    padded = []
    for panel in panels:
        if panel.ndim == 2:
            panel = np.stack([panel] * 3, axis=-1)
        if panel.shape[0] != heights:
            pad = heights - panel.shape[0]
            panel = np.pad(panel, ((0, pad), (0, 0), (0, 0)), mode="edge")
        padded.append(panel)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(padded, axis=1)).save(path)


def save_segmentation_preview(
    image: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    path: str,
    ignore_index: int = 255,
) -> None:
    """RGB / prediction / ground truth, with errors highlighted in red."""
    from PIL import Image

    pred = prediction.detach().cpu()
    tgt = target.detach().cpu()
    if pred.dim() == 3:
        pred = pred[0]
    if tgt.dim() == 3:
        tgt = tgt[0]

    palette = _class_palette(num_classes)
    valid = tgt != ignore_index
    wrong = (pred != tgt) & valid

    pred_rgb = palette[pred.clamp(0, num_classes - 1).numpy()]
    tgt_rgb = palette[tgt.clamp(0, num_classes - 1).numpy()]
    error_rgb = pred_rgb.copy()
    error_rgb[wrong.numpy()] = np.array([255, 0, 0], dtype=np.uint8)

    panels = [_to_uint8(image), pred_rgb, tgt_rgb, error_rgb]
    if panels[0].ndim == 2:
        panels[0] = np.stack([panels[0]] * 3, axis=-1)
    heights = max(panel.shape[0] for panel in panels)
    padded = []
    for panel in panels:
        if panel.shape[0] != heights:
            pad = heights - panel.shape[0]
            panel = np.pad(panel, ((0, pad), (0, 0), (0, 0)), mode="edge")
        padded.append(panel)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(padded, axis=1)).save(path)


def _class_palette(num_classes: int) -> np.ndarray:
    """A fixed, distinguishable palette that is stable across runs."""
    base = np.array(
        [
            [0, 0, 0],
            [128, 0, 0],
            [0, 128, 0],
            [128, 128, 0],
            [0, 0, 128],
            [128, 0, 128],
            [0, 128, 128],
            [128, 128, 128],
            [64, 0, 0],
            [192, 0, 0],
            [64, 128, 0],
            [192, 128, 0],
            [64, 0, 128],
            [192, 0, 128],
            [64, 128, 128],
            [192, 128, 128],
        ],
        dtype=np.uint8,
    )
    if num_classes <= len(base):
        return base[:num_classes]
    # Fall back to a deterministic spread for larger label sets.
    extra = np.linspace(0, 255, num_classes - len(base), dtype=np.uint8)
    more = np.stack([extra, np.roll(extra, 1), np.roll(extra, 2)], axis=-1)
    return np.concatenate([base, more.astype(np.uint8)], axis=0)
