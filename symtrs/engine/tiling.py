"""Sliding-window inference for images that do not fit in one forward pass.

At 2048x2048 a single forward pass through a ViT-L is not affordable on a 24GB card,
and several of these networks were never trained above ~512px, so feeding them the
full native image is both slow and off-distribution. Tiling gives a middle ground:
run the network on overlapping windows at roughly the resolution it expects, then
stitch the windows back into one field.

Two details make the stitching honest:

  * Predictions are blended in the *model's own output space* (disparity for relative
    depth models, logits for segmentation) rather than after conversion to metres or
    class indices. Blending is linear, and the conversion back to metres is not, so
    blending afterwards would not equal the model's own answer anywhere.
  * Each window is weighted by a 2D Hann window, which goes to zero at the tile
    border. A tile border is exactly where a network is least reliable because it has
    no context beyond it, so down-weighting it there removes the visible seams a flat
    average leaves behind.
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

import torch
import torch.nn.functional as F

#: (top, left, height, width) of one window.
Box = Tuple[int, int, int, int]


def tile_positions(length: int, tile: int, step: int) -> List[int]:
    """Start offsets along one axis, ending flush with `length`.

    The final window is clamped to `length - tile` rather than allowed to overhang, so
    every pixel is covered and the last window is not mostly padding.
    """
    if tile >= length:
        return [0]
    positions = list(range(0, length - tile + 1, step))
    if positions[-1] != length - tile:
        positions.append(length - tile)
    return positions


def tile_grid(height: int, width: int, tile: int, overlap: int) -> List[Box]:
    """Every window of a `tile`-sized grid over `height` x `width`."""
    if tile <= 0:
        raise ValueError("tile_size must be positive")
    if overlap < 0:
        raise ValueError("tile_overlap must not be negative")
    step = max(1, tile - overlap)
    boxes: List[Box] = []
    for top in tile_positions(height, min(tile, height), step):
        for left in tile_positions(width, min(tile, width), step):
            boxes.append((top, left, min(tile, height), min(tile, width)))
    return boxes


def hann_window(height: int, width: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """A 2D raised-cosine window, minimum at the border, never zero.

    `periodic=False` puts both endpoints exactly at zero, which would leave the outermost
    pixel of every tile weighted by nothing at all - so a pixel covered only by tile
    borders would be divided by a zero denominator. The `clamp_min` lifts them to 1e-3,
    which keeps the seam-suppressing shape while guaranteeing a real denominator; it is
    the floor that matters here, not the `periodic` flag.
    """
    row = torch.hann_window(height, periodic=False, device=device, dtype=dtype).clamp_min(1e-3)
    column = torch.hann_window(width, periodic=False, device=device, dtype=dtype).clamp_min(1e-3)
    return row[:, None] * column[None, :]


def _resize(prediction: torch.Tensor, size: Sequence[int]) -> torch.Tensor:
    if tuple(prediction.shape[-2:]) == tuple(size):
        return prediction
    return F.interpolate(prediction.float(), size=tuple(size), mode="bilinear", align_corners=False)


@torch.no_grad()
def predict_tiled(
    image: torch.Tensor,
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    tile_size: int,
    overlap: int = 128,
) -> torch.Tensor:
    """Run `forward_fn` over overlapping windows and stitch the results.

    Args:
        image: `[B, C, H, W]`. One batch element is handled at a time; tiling a batch
            would multiply peak memory by the batch size for no benefit, since the
            point of tiling is to make a single image fit.
        forward_fn: processed tile `[1, C, h, w]` -> prediction `[1, K, h', w']`. The
            output may be at a different resolution than the tile; it is resized back
            before blending.
        tile_size: window side, in input pixels.
        overlap: how much neighbouring windows share. Must be shorter than `tile_size`.

    Returns:
        `[B, K, H, W]` in whatever space `forward_fn` produces.
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)
    if tile_size <= 0:
        raise ValueError("tile_size must be positive; use forward_at_size for whole-image inference")

    height, width = image.shape[-2:]
    outputs = []
    for batch_index in range(image.shape[0]):
        single = image[batch_index : batch_index + 1]
        accumulator = None
        weights = None

        for top, left, tile_h, tile_w in tile_grid(height, width, tile_size, overlap):
            tile = single[..., top : top + tile_h, left : left + tile_w]
            prediction = forward_fn(tile)
            if prediction.dim() == 3:
                prediction = prediction.unsqueeze(1)
            prediction = _resize(prediction, (tile_h, tile_w)).float()

            if accumulator is None:
                channels = prediction.shape[1]
                accumulator = torch.zeros(
                    (1, channels, height, width), device=prediction.device, dtype=prediction.dtype
                )
                weights = torch.zeros(
                    (1, 1, height, width), device=prediction.device, dtype=prediction.dtype
                )

            window = hann_window(tile_h, tile_w, device=prediction.device, dtype=prediction.dtype)
            accumulator[..., top : top + tile_h, left : left + tile_w] += prediction * window
            weights[..., top : top + tile_h, left : left + tile_w] += window

        outputs.append((accumulator / weights.clamp_min(1e-6)).to(image.dtype))

    return torch.cat(outputs, dim=0)


def plan_tiling(height: int, width: int, tile_size: int, overlap: int) -> dict:
    """How many windows a configuration implies, for logging before a long run."""
    if tile_size <= 0:
        return {"tiles": 1, "tile_size": 0, "overlap": 0, "coverage": 1.0}
    boxes = tile_grid(height, width, tile_size, overlap)
    covered = sum(box[2] * box[3] for box in boxes)
    return {
        "tiles": len(boxes),
        "tile_size": tile_size,
        "overlap": overlap,
        "coverage": covered / float(height * width),
    }


__all__ = ["Box", "hann_window", "plan_tiling", "predict_tiled", "tile_grid", "tile_positions"]
