"""Sliding-window inference for images that do not fit in one forward pass.

At 2048x2048 a single forward pass through a ViT-L is not affordable on a 24GB card,
and several of these networks were never trained above ~512px, so feeding them the
full native image is both slow and off-distribution. Tiling gives a middle ground:
run the network on overlapping windows at roughly the resolution it expects, then
stitch the windows back into one field.

Three details make the stitching honest:

  * Predictions are blended in the *model's own output space* (disparity for relative
    depth models, logits for segmentation) rather than after conversion to metres or
    class indices. Blending is linear, and the conversion back to metres is not, so
    blending afterwards would not equal the model's own answer anywhere.
  * Each window is weighted by a 2D Hann window, which goes to zero at the tile
    border. A tile border is exactly where a network is least reliable because it has
    no context beyond it, so down-weighting it there removes the visible seams a flat
    average leaves behind.

A third matters for *relative* depth only. A relative model fixes its output only up to an
unknown map **per forward pass**, so two neighbouring windows disagree about the same pixels and a
plain blend of them is neither tile's answer. Which map depends on the output space
(`symtrs.registry.depth_pairing`), and the matching has to be the same family or it injects the very
error the final per-image alignment cannot remove:

  * disparity (MiDaS: `s / z + t`) is ambiguous in scale AND shift -> `match="affine"`;
  * depth up to a scale (ZoeD-N scored relative, DA3 relative) is ambiguous in scale only ->
    `match="scale"`. A fitted shift there would add an offset that the final `scale` alignment
    (no shift) cannot remove;
  * metric depth and logits are absolute -> `match="none"`; matching them would only add noise.

`match_mode(model)` derives the mode from the model's `protocol` and `output_space`. Each window is
fitted against the overlap with the stitched field so far, i.e. with the Hann-weighted blend of ALL
windows already placed (not only its left/top neighbour), so a single bad neighbour is averaged out
rather than inherited. The raster order still anchors the global frame on the first window; that
frame is one global affine map of the truth, which is exactly what the per-image alignment that
scores the model removes. The per-image alignment is still done once, on the finished full-image
field, by the caller; this only makes the field one consistent surface.

A fit is only as good as the texture in the overlap. On a flat overlap the slope is noise, so the
fit degrades in steps (see `fit_overlap`) and every degradation is counted in the `stats` dict, which
`evaluate_depth` writes into the metrics as `tile_match_fallbacks`.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

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


#: Fewest already-stitched pixels a window must share before a fit is trusted.
_MIN_MATCH_PIXELS = 256

#: Matching modes. "none" is the default and what absolute outputs (metric depth, logits) get.
MATCH_MODES = ("none", "affine", "scale")

#: A window may not be rescaled by more than this factor relative to the stitched field. Real
#: per-forward scale differences between overlapping windows of one scene are a few tens of percent;
#: a fitted slope outside this band is the fit failing, not the model disagreeing with itself.
SLOPE_RANGE = (0.25, 4.0)

#: The overlap counts as flat when its variance is below this fraction of its squared mean. Relative
#: so that it does not depend on the unit of the model (disparity in 1/m or in arbitrary units).
_FLAT_RELATIVE_VARIANCE = 1e-6

#: Below this correlation between the window and the stitched field the overlap does not show the
#: same structure (different content at the border, or noise), and a slope fitted to it is noise.
_MIN_CORRELATION = 0.5


def match_mode(model) -> str:
    """Which window matching a model's output needs: "affine", "scale" or "none".

    Keyed on the OUTPUT SPACE, not on `protocol == "relative"` alone: a relative model that emits
    depth up to a scale must not have a shift fitted (nothing downstream can remove it), while one
    that emits disparity needs both. Segmentation models and metric depth have no ambiguity.
    """
    if getattr(model, "protocol", "") != "relative":
        return "none"
    return "affine" if getattr(model, "output_space", "depth") == "disparity" else "scale"


def fit_overlap(
    source: torch.Tensor, reference: torch.Tensor, weight: torch.Tensor, mode: str = "affine"
) -> Optional[Tuple[float, float, bool]]:
    """Map `source` onto `reference` over their shared pixels: `(slope, shift, degraded)`.

    `None` means no trustworthy fit exists at all (too few pixels, non-finite sums, or no signal)
    and the window is left on its own scale. Otherwise the fit is returned, and `degraded` says
    whether it had to fall back from the full estimate:

      affine (disparity)
        full:      the symmetric (reduced-major-axis) slope `std(ref) / std(src)` and the shift
                   that matches the weighted means. OLS `cov / var` would be biased toward 0 when
                   the window is noisy, which flattens every later window onto a too-small scale;
                   the symmetric slope does not care which side carries the noise.
        fallback:  shift-only (slope 1) when the overlap is flat (relative variance threshold),
                   or uncorrelated with the field (`_MIN_CORRELATION`, which also covers a
                   non-positive slope: a relative model is monotone in the truth, so a flip is a
                   failed fit). A slope outside `SLOPE_RANGE` is clamped, the shift re-derived
                   from the means.
      scale (depth up to a scale)
        full:      the weighted least-squares slope through the origin, `sum(w s r) / sum(w s s)`.
                   On a flat overlap this is the ratio of means, i.e. the scale is kept even when
                   there is no texture; it is clamped to `SLOPE_RANGE`. There is never a shift.
    """
    if mode not in ("affine", "scale"):
        raise ValueError(f"unknown matching mode {mode!r}; expected 'affine' or 'scale'")
    if source.numel() < _MIN_MATCH_PIXELS:
        return None
    source, reference, weight = source.double(), reference.double(), weight.double()
    total = weight.sum()
    if not torch.isfinite(total) or total <= 0:
        return None

    low, high = SLOPE_RANGE
    if mode == "scale":
        energy = (weight * source * source).sum()
        if not torch.isfinite(energy) or energy <= 0:
            return None
        slope = float((weight * source * reference).sum() / energy)
        if not (slope == slope) or slope <= 0:
            return None
        clamped = min(max(slope, low), high)
        return clamped, 0.0, clamped != slope

    mean_source = float((weight * source).sum() / total)
    mean_reference = float((weight * reference).sum() / total)
    variance = float((weight * (source - mean_source) ** 2).sum() / total)
    variance_ref = float((weight * (reference - mean_reference) ** 2).sum() / total)
    if not (variance == variance and variance_ref == variance_ref) or variance == float("inf"):
        return None
    flat = variance <= _FLAT_RELATIVE_VARIANCE * mean_source ** 2 + 1e-24
    if flat or variance_ref <= 0:
        return 1.0, mean_reference - mean_source, True
    covariance = float((weight * (source - mean_source) * (reference - mean_reference)).sum() / total)
    correlation = covariance / (variance * variance_ref) ** 0.5
    if not correlation >= _MIN_CORRELATION:
        return 1.0, mean_reference - mean_source, True
    slope = (variance_ref / variance) ** 0.5
    clamped = min(max(slope, low), high)
    return clamped, mean_reference - clamped * mean_source, clamped != slope


@torch.no_grad()
def predict_tiled(
    image: torch.Tensor,
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    tile_size: int,
    overlap: int = 128,
    match_affine: bool = False,
    match: str = "none",
    stats: Optional[Dict[str, int]] = None,
    scale: int = 1,
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
        match_affine: legacy spelling of `match="affine"`.
        match: "none", "affine" or "scale": bring each window onto the field already stitched
            (see the module docstring and `match_mode`). Relative depth only.
        stats: if given, incremented in place with `windows` (windows placed), `matched` (fitted
            at all), `fallbacks` (fitted, but degraded: flat/uncorrelated overlap, clamped slope)
            and `unmatched` (overlap too small or degenerate: left on its own scale).
        scale: how much larger than its window the network's answer is. 1 for every dense-prediction
            task (depth, segmentation, shadow removal). Super-resolution passes its upscale factor:
            windows are laid out and overlapped in INPUT pixels, `forward_fn` returns
            `[1, K, scale * h, scale * w]`, and the stitched field is `scale * H` by `scale * W`.
            A window whose answer has any other size is refused rather than resized, because
            stretching it would blend a misregistered tile into its neighbours.

    Returns:
        `[B, K, scale * H, scale * W]` in whatever space `forward_fn` produces.
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)
    if tile_size <= 0:
        raise ValueError("tile_size must be positive; use forward_at_size for whole-image inference")
    if match_affine and match == "none":
        match = "affine"
    if match not in MATCH_MODES:
        raise ValueError(f"unknown match mode {match!r}; expected one of {MATCH_MODES}")
    if scale < 1:
        raise ValueError("scale must be a positive integer")
    if stats is None:
        stats = {}
    for key in ("windows", "matched", "fallbacks", "unmatched"):
        stats.setdefault(key, 0)

    height, width = image.shape[-2:]
    out_height, out_width = height * scale, width * scale
    outputs = []
    for batch_index in range(image.shape[0]):
        single = image[batch_index : batch_index + 1]
        accumulator = None
        weights = None

        for in_top, in_left, tile_h, tile_w in tile_grid(height, width, tile_size, overlap):
            tile = single[..., in_top : in_top + tile_h, in_left : in_left + tile_w]
            prediction = forward_fn(tile)
            if prediction.dim() == 3:
                prediction = prediction.unsqueeze(1)
            # Everything below is in OUTPUT pixels (identical to input pixels when scale == 1).
            top, left, tile_h, tile_w = in_top * scale, in_left * scale, tile_h * scale, tile_w * scale
            if scale > 1 and tuple(prediction.shape[-2:]) != (tile_h, tile_w):
                raise ValueError(
                    f"forward_fn returned {tuple(prediction.shape[-2:])} for a window that should "
                    f"come back as {(tile_h, tile_w)} (scale {scale}); an upscaling network must "
                    "return exactly scale x its input, so fix forward_fn instead of resizing"
                )
            prediction = _resize(prediction, (tile_h, tile_w)).float()

            if accumulator is None:
                channels = prediction.shape[1]
                accumulator = torch.zeros(
                    (1, channels, out_height, out_width),
                    device=prediction.device, dtype=prediction.dtype,
                )
                weights = torch.zeros(
                    (1, 1, out_height, out_width), device=prediction.device, dtype=prediction.dtype
                )

            window = hann_window(tile_h, tile_w, device=prediction.device, dtype=prediction.dtype)
            stats["windows"] += 1
            if match != "none" and prediction.shape[1] == 1:
                region = (..., slice(top, top + tile_h), slice(left, left + tile_w))
                seen = weights[region]
                shared = (seen > 0)[0, 0]
                if bool(shared.any()):
                    reference = (accumulator[region] / seen.clamp_min(1e-6))[0, 0][shared]
                    fit = fit_overlap(
                        prediction[0, 0][shared], reference, (seen[0, 0] * window)[shared], match
                    )
                    if fit is None:
                        stats["unmatched"] += 1
                    else:
                        stats["matched"] += 1
                        stats["fallbacks"] += int(fit[2])
                        prediction = prediction * fit[0] + fit[1]
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


__all__ = [
    "Box",
    "MATCH_MODES",
    "SLOPE_RANGE",
    "fit_overlap",
    "hann_window",
    "match_mode",
    "plan_tiling",
    "predict_tiled",
    "tile_grid",
    "tile_positions",
]
