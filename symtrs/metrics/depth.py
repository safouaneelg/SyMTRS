"""Depth metrics and prediction alignment.

Two protocols, matching the two kinds of depth model:

*Metric* models (ZoeDepth-K, Depth Anything 3 metric, Depth Pro) predict metres, so
their error is reported directly and no alignment is applied.

*Relative* models predict an affine-invariant quantity, and which affine family depends on
the output space (`symtrs.registry.depth_pairing`). ZoeDepth-N and relative Depth Anything 3
emit depth up to a scale, aligned by `scale`. MiDaS/DPT emits disparity `s / z + t`, aligned by
`scale_shift_disparity`: scale AND shift fitted against `1 / ground truth`, then inverted. Their
numbers are only meaningful after per-image alignment, and the convention differs between
papers, so `align` is recorded per model in the registry rather than hardcoded.

Everything is computed on the validity mask. Invalid ground-truth pixels (sky, no
geometry) are excluded from every metric rather than being treated as zero metres.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

#: Alignment strategies for relative-depth models.
ALIGNMENTS = ("none", "median", "scale", "scale_shift", "scale_shift_disparity")

#: Alignments whose input is the raw disparity of the model, not metres. Everything else takes
#: depth. `DepthAdapter.scoring_input` is what hands the right one over.
DISPARITY_ALIGNMENTS = ("scale_shift_disparity",)

#: Ceiling, in metres, applied to the depth recovered from an aligned disparity when the caller
#: does not pass `max_depth`. A fitted disparity can reach or cross zero (sky, far field), and
#: `1 / d` then explodes or flips sign; the MiDaS evaluation clamps the disparity to the
#: dataset's depth ceiling for the same reason. 1000 m is `data.max_depth` in every generated
#: config, i.e. the same cut-off that defines which ground-truth pixels are valid.
DEFAULT_MAX_DEPTH = 1000.0

#: Depth, in metres, that a prediction which is not a depth (zero, negative, NaN, inf after the
#: alignment) is scored as. Dropping such pixels from the mask - what this module used to do - made
#: RMSE and abs_rel optimistic for exactly the models that fail there; scoring them at a tiny
#: positive depth makes them count as the large errors they are (abs_rel ~ 1, d1 = 0) while keeping
#: the log-based metrics finite. 1 mm is below any depth the dataset can hold.
DEFAULT_MIN_DEPTH = 1e-3

#: Canonical metric order. Anything that logs or tabulates depth uses this order, so
#: CSV columns and result tables stay stable. `invalid_pred_fraction` is a diagnostic, not a
#: score: the share of valid ground-truth pixels whose prediction was not a positive finite depth
#: (and was therefore scored at `min_depth`). Near 0 for a healthy model.
DEPTH_METRIC_NAMES = (
    "rmse", "mae", "abs_rel", "sq_rel", "d1", "d2", "d3", "silog", "invalid_pred_fraction",
)

EPS = 1e-8


class AllInvalidError(RuntimeError):
    """Every pixel was masked out, so no metric can be computed."""


@dataclass
class AlignmentFit:
    """Parameters of the affine fit applied to one prediction, kept for reporting."""

    mode: str
    scale: float = 1.0
    shift: float = 0.0


def align_prediction(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    mode: str = "scale_shift",
    eps: float = EPS,
    max_depth: Optional[float] = None,
) -> tuple:
    """Fit `pred ~ scale * pred + shift` to `target` per image and apply it.

    All tensors are `[B, 1, H, W]` (or `[B, H, W]`); `mask` is boolean and True where
    the target is valid. The fit is a closed-form least squares over masked pixels,
    computed per image so one image's scale cannot leak into another's.

    Returns `(aligned_pred, [AlignmentFit, ...])`.

    Modes:
        none          - identity.
        median        - single scale, `median(target) / median(pred)`. This is the
                        convention used by several relative-depth papers.
        scale         - least-squares scale only.
        scale_shift   - least-squares scale and shift. The most permissive, and the
                        right default when comparing models whose output space is
                        unknown.
        scale_shift_disparity
                      - the MiDaS protocol. `pred` is the model's raw DISPARITY (not
                        metres); scale and shift are fitted by least squares against
                        `1 / target` over the valid pixels, the fitted disparity is clamped
                        to at least `1 / max_depth`, and the returned map is DEPTH
                        (`1 / disparity`). Fitting after inverting, as `scale` on
                        `1 / pred` would, cannot absorb the shift `t` in `s / z + t`.
                        The returned fit is in disparity units.
    """
    if mode not in ALIGNMENTS:
        raise ValueError(f"unknown alignment {mode!r}; expected one of {ALIGNMENTS}")

    if mode == "none":
        return pred, [AlignmentFit(mode="none") for _ in range(pred.shape[0])]

    squeeze = pred.dim() == 3
    if squeeze:
        pred = pred.unsqueeze(1)
        target = target.unsqueeze(1)
        mask = mask.unsqueeze(1)

    batch = pred.shape[0]
    aligned = pred.clone()
    fits = []
    for index in range(batch):
        p = pred[index].reshape(-1).double()
        t = target[index].reshape(-1).double()
        m = mask[index].reshape(-1).bool()
        if m.sum() < 2:
            raise AllInvalidError(
                f"image {index} has {int(m.sum())} valid pixels; alignment needs at least 2"
            )
        p_m, t_m = p[m], t[m]

        if mode == "scale_shift_disparity":
            disparity_target = 1.0 / t_m
            p_mean = p_m.mean()
            g_mean = disparity_target.mean()
            p_centered = p_m - p_mean
            denominator = (p_centered * p_centered).sum().clamp_min(eps)
            scale = float((p_centered * (disparity_target - g_mean)).sum() / denominator)
            shift = float(g_mean - scale * p_mean)
            floor = 1.0 / float(max_depth if max_depth else DEFAULT_MAX_DEPTH)
            fitted = (pred[index].double() * scale + shift).clamp_min(floor)
            aligned[index] = (1.0 / fitted).to(pred.dtype)
            fits.append(AlignmentFit(mode=mode, scale=scale, shift=shift))
            continue

        if mode == "median":
            p_med = torch.median(p_m)
            t_med = torch.median(t_m)
            scale = float(t_med / p_med.clamp_min(eps))
            shift = 0.0
        elif mode == "scale":
            # scale = <p, t> / <p, p>, with the mean removed from neither.
            scale = float((p_m * t_m).sum() / (p_m * p_m).sum().clamp_min(eps))
            shift = 0.0
        else:  # scale_shift
            p_mean = p_m.mean()
            t_mean = t_m.mean()
            p_centered = p_m - p_mean
            t_centered = t_m - t_mean
            denominator = (p_centered * p_centered).sum().clamp_min(eps)
            scale = float((p_centered * t_centered).sum() / denominator)
            shift = float(t_mean - scale * p_mean)

        aligned[index] = (pred[index].double() * scale + shift).to(pred.dtype)
        fits.append(AlignmentFit(mode=mode, scale=scale, shift=shift))

    if squeeze:
        aligned = aligned.squeeze(1)
    return aligned, fits


def compute_depth_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    align: str = "none",
    max_depth: Optional[float] = None,
    min_depth: Optional[float] = None,
) -> Dict[str, float]:
    """Depth metrics over the valid pixels of a batch.

    Args:
        pred: predicted depth in metres, `[B, 1, H, W]` or `[B, H, W]` - or, when `align`
            is in `DISPARITY_ALIGNMENTS`, the model's raw disparity, which the alignment turns
            into metres.
        target: ground-truth depth in metres, same shape.
        mask: boolean validity mask; defaults to "finite and positive in target".
        align: one of `ALIGNMENTS`, applied before the metrics.
        max_depth: ceiling for depth recovered from disparity (see `DEFAULT_MAX_DEPTH`);
            ignored by the other alignments.
        min_depth: metres a non-positive or non-finite prediction is scored as
            (`DEFAULT_MIN_DEPTH`). Such pixels are never dropped; see `invalid_pred_fraction`.

    Returns a dict with the keys in `DEPTH_METRIC_NAMES`:
        rmse, mae        - metres, the interpretable ones for a metric benchmark.
        abs_rel, sq_rel  - scale-free relative error.
        d1, d2, d3       - fraction within 1.25, 1.25^2, 1.25^3.
        silog            - scale-invariant log RMSE, in percent. Comparable to the
                           MiDaS/ZoeDepth tables.
        invalid_pred_fraction
                         - fraction of valid-GT pixels whose prediction (after alignment) was
                           <= 0 or non-finite. These are SCORED at `min_depth`, not dropped.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"prediction shape {tuple(pred.shape)} does not match target "
            f"{tuple(target.shape)}"
        )

    if mask is None:
        mask = torch.isfinite(target) & (target > 0)
    else:
        mask = mask.bool() & torch.isfinite(target) & (target > 0)

    if mask.sum() == 0:
        raise AllInvalidError("no valid pixels in this batch; nothing to measure")

    pred, target = pred.float(), target.float()

    if align != "none":
        # Align per image. An image with too few valid pixels is left untouched rather
        # than allowed to abort the whole run; the metrics below will simply reflect
        # that image being wrong, which is the honest outcome.
        aligned = pred.clone()
        for index in range(pred.shape[0]):
            single_mask = mask[index : index + 1]
            if int(single_mask.sum()) >= 2:
                fixed, _ = align_prediction(
                    pred[index : index + 1], target[index : index + 1], single_mask, mode=align,
                    max_depth=max_depth,
                )
                aligned[index : index + 1] = fixed
            elif align in DISPARITY_ALIGNMENTS:
                # Left untouched, a disparity map would be read as metres and score as an
                # absurd model rather than as an unscorable image.
                raise AllInvalidError(
                    f"image {index} has fewer than 2 valid pixels; a disparity alignment cannot "
                    "be fitted, and the raw disparity is not a depth"
                )
        pred = aligned

    # A model can emit non-positive or non-finite depth (a metric head's ReLU, an aligned
    # disparity crossing zero, an fp16 overflow). The log-based metrics cannot see such values, but
    # DROPPING them from the mask - as was done before - rewards the model for being wrong exactly
    # where it is wrong. They are scored at `min_depth` instead, so the error counts, and the share
    # is reported so a table can show that a number comes from a model that does this.
    floor = float(min_depth if min_depth else DEFAULT_MIN_DEPTH)
    invalid = ~(torch.isfinite(pred) & (pred > 0))
    invalid_fraction = float((invalid & mask).sum()) / float(mask.sum())
    pred = torch.where(invalid, torch.full_like(pred, floor), pred)

    p = pred[mask].double()
    t = target[mask].double()

    difference = p - t
    rmse = torch.sqrt((difference ** 2).mean())
    mae = difference.abs().mean()
    abs_rel = (difference.abs() / t).mean()
    sq_rel = ((difference ** 2) / t).mean()

    ratio = torch.maximum(p / t, t / p)
    d1 = (ratio < 1.25).double().mean()
    d2 = (ratio < 1.25 ** 2).double().mean()
    d3 = (ratio < 1.25 ** 3).double().mean()

    log_difference = torch.log(p) - torch.log(t)
    # E[x^2] - E[x]^2 is a variance and cannot be negative, but in floating point it can come out
    # as -1e-17 for a near-constant log error (a perfect or perfectly-scaled prediction), and the
    # sqrt of that is NaN, which makes the whole epoch non-finite. Clamp the variance at 0.
    silog = torch.sqrt(((log_difference ** 2).mean() - (log_difference.mean() ** 2)).clamp_min(0.0)) * 100.0

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "abs_rel": float(abs_rel),
        "sq_rel": float(sq_rel),
        "d1": float(d1),
        "d2": float(d2),
        "d3": float(d3),
        "silog": float(silog),
        "invalid_pred_fraction": invalid_fraction,
    }


def summarise_alignment(fits) -> Dict[str, float]:
    """Mean scale and shift across a batch, for logging. Useful for spotting drift."""
    if not fits:
        return {}
    scales = [f.scale for f in fits]
    shifts = [f.shift for f in fits]
    return {
        "align_scale_mean": sum(scales) / len(scales),
        "align_shift_mean": sum(shifts) / len(shifts),
    }
