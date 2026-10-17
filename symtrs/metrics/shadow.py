"""Shadow-removal scoring: the one place the shadow derivation and the metrics live.

Every shadow-removal model is scored here, from the 8-bit PNG it wrote, against the 8-bit
`rgb_noshadow` target - not from whatever tensor the model happened to hold. That keeps
the metric independent of the model's framework, precision and tiling, and means a number
in the table can be re-derived from files on disk.

What this module deliberately does not do: import torch, scipy, scikit-image or OpenCV.
The package's dependency rule (AGENTS.md section 2, rule 9) leaves only numpy/PIL, so the
colour conversion, the SSIM window and the morphology below are written against numpy.

Protocol (ISTD, as in the ShadowFormer / ST-CGAN papers), with two documented differences:

  * PSNR and SSIM on all pixels, plus the same restricted to the shadow region and to the
    non-shadow region of the derived mask.
  * "RMSE" in LAB. The ISTD literature's number is the *mean absolute* LAB error summed over
    the three channels (`mae_lab` here, directly comparable with published tables). The
    root-mean-square error is also reported (`rmse_lab`), because the name promises it.
  * Regional numbers are computed over the region's pixels only. ShadowFormer's `test.py`
    zeroes the other region and scores the whole image, which inflates regional PSNR/SSIM
    with a large block of identical black pixels; the numbers here are therefore lower and
    not comparable with a published regional PSNR/SSIM, only with each other.
  * SSIM is computed on the **luma channel only** (Rec. 709 Y of the 8-bit sRGB values), hence
    the key `ssim_y`. ShadowFormer's `test.py` scores SSIM per RGB channel and averages, which
    gives a different number; `ssim_y` must not be set beside a published SSIM. The name, not
    a footnote, carries the difference so a table cannot drop it. `SSIM_NOTE` is copied into
    every shadow summary for the same reason.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np

#: Rec. 709 luma weights, applied to the stored (non-linear) sRGB values. It is a
#: brightness proxy for the mask, not a photometric quantity, so linearising would only
#: change what the threshold means without making the mask better.
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

REGIONS = ("all", "shadow", "nonshadow")

#: Written into summary.json beside the metrics (see `symtrs.model.shadow.runner`).
SSIM_NOTE = (
    "ssim_y is SSIM on Rec.709 luma only (Gaussian 11x11, sigma 1.5, valid region); published "
    "ShadowFormer/ISTD tables average SSIM over the RGB channels, so the two are not comparable"
)
METRICS = ("psnr", "ssim_y", "mae_lab", "rmse_lab")


def luminance(image: np.ndarray) -> np.ndarray:
    """`[H, W]` luma of a `[H, W, 3]` image in [0, 1]."""
    return image.astype(np.float32) @ LUMA


def _shift_and(mask: np.ndarray, size: int, op) -> np.ndarray:
    """Separable box erosion (`op=np.logical_and`) or dilation (`np.logical_or`)."""
    radius = size // 2
    result = mask
    for axis in (0, 1):
        padded = np.pad(
            result,
            [(radius, radius) if a == axis else (0, 0) for a in (0, 1)],
            constant_values=(op is np.logical_and),
        )
        length = result.shape[axis]
        combined = None
        for offset in range(size):
            window = np.take(padded, np.arange(offset, offset + length), axis=axis)
            combined = window if combined is None else op(combined, window)
        result = combined
    return result


def opening(mask: np.ndarray, size: int) -> np.ndarray:
    """Binary opening with a `size` x `size` square: drops specks smaller than the square.

    An odd `size` is required so the element has a centre; an even one is rounded up
    rather than silently shifting the mask by half a pixel.
    """
    if size <= 1:
        return mask
    size += 1 - size % 2
    eroded = _shift_and(mask, size, np.logical_and)
    return _shift_and(eroded, size, np.logical_or)


def derive_mask(
    day: np.ndarray, noshadow: np.ndarray, threshold: float = 0.1, open_size: int = 3
) -> np.ndarray:
    """Boolean `[H, W]` shadow mask from a shadowed image and its shadow-free twin.

    Both inputs are `[H, W, 3]` in [0, 1]. A shadow can only darken, so the signed
    relative darkening `(Y_noshadow - Y_day) / max(Y_noshadow, eps)` is thresholded; a
    pixel that got brighter is never shadow. `eps` keeps black pixels (0/0) out of the mask.
    """
    y_day = luminance(day)
    y_clean = luminance(noshadow)
    darkening = (y_clean - y_day) / np.maximum(y_clean, 1e-3)
    return opening(darkening > threshold, open_size)


# --------------------------------------------------------------------- colour


def rgb_to_lab(image: np.ndarray) -> np.ndarray:
    """sRGB [0, 1] `[H, W, 3]` -> CIE L*a*b* (D65), L in 0..100: what OpenCV/skimage give."""
    rgb = image.astype(np.float32)
    linear = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    matrix = np.array(
        [[0.4124564, 0.3575761, 0.1804375],
         [0.2126729, 0.7151522, 0.0721750],
         [0.0193339, 0.1191920, 0.9503041]],
        dtype=np.float32,
    )
    xyz = linear @ matrix.T / np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack(
        [116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])],
        axis=-1,
    )


# ----------------------------------------------------------------------- SSIM


def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    axis = np.arange(size, dtype=np.float32) - size // 2
    kernel = np.exp(-(axis ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


def _filter(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Valid-mode separable correlation, rows then columns."""
    size = len(kernel)
    out = sum(kernel[i] * image[i : image.shape[0] - size + 1 + i] for i in range(size))
    return sum(kernel[i] * out[:, i : out.shape[1] - size + 1 + i] for i in range(size))


def ssim_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-pixel SSIM of two `[H, W]` images in [0, 1] (Gaussian 11x11, sigma 1.5).

    The map is `(H-10, W-10)`: valid positions only, as in the reference implementation.
    Use `crop_mask` to bring a full-size region mask to the same footprint.
    """
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    kernel = _gaussian_kernel()
    mu_a, mu_b = _filter(a, kernel), _filter(b, kernel)
    var_a = _filter(a * a, kernel) - mu_a ** 2
    var_b = _filter(b * b, kernel) - mu_b ** 2
    cov = _filter(a * b, kernel) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))


def crop_mask(mask: np.ndarray, size: int = 11) -> np.ndarray:
    """A full-size mask on the footprint of `ssim_map`."""
    margin = size // 2
    return mask[margin : mask.shape[0] - margin, margin : mask.shape[1] - margin]


# --------------------------------------------------------------------- scoring


def score_image(
    prediction: np.ndarray, target: np.ndarray, shadow: np.ndarray
) -> Dict[str, Optional[float]]:
    """Every metric for one image, as `{metric}` and `{metric}_shadow` / `_nonshadow`.

    `prediction` and `target` are `[H, W, 3]` in [0, 1]; `shadow` is boolean `[H, W]`. A
    region with no pixels yields `None` for its metrics (an image can have no shadow), and
    `aggregate` averages only over the images that have one.
    """
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction {prediction.shape} and target {target.shape} differ in size; the "
            "model must write an image of the input's size"
        )
    squared = ((prediction.astype(np.float32) - target.astype(np.float32)) ** 2).mean(axis=-1)
    delta = rgb_to_lab(prediction) - rgb_to_lab(target)
    absolute = np.abs(delta).sum(axis=-1)  # summed over L, a, b: the ISTD convention
    lab_squared = (delta ** 2).sum(axis=-1)
    ssim = ssim_map(luminance(prediction), luminance(target))
    ssim_shadow = crop_mask(shadow)

    result: Dict[str, Optional[float]] = {}
    for region, pixels, ssim_pixels in (
        ("all", None, None),
        ("shadow", shadow, ssim_shadow),
        ("nonshadow", ~shadow, ~ssim_shadow),
    ):
        suffix = "" if region == "all" else f"_{region}"
        if pixels is None:
            mse, mae, mse_lab, ssim_value = squared.mean(), absolute.mean(), lab_squared.mean(), ssim.mean()
        elif pixels.any() and ssim_pixels.any():
            mse, mae = squared[pixels].mean(), absolute[pixels].mean()
            mse_lab, ssim_value = lab_squared[pixels].mean(), ssim[ssim_pixels].mean()
        else:
            for name in METRICS:
                result[name + suffix] = None
            continue
        result["psnr" + suffix] = float(10 * np.log10(1.0 / max(float(mse), 1e-12)))
        result["ssim_y" + suffix] = float(ssim_value)
        result["mae_lab" + suffix] = float(mae)
        result["rmse_lab" + suffix] = float(np.sqrt(mse_lab))
    result["shadow_fraction"] = float(shadow.mean())
    return result


def aggregate(rows: Iterable[Dict[str, Optional[float]]]) -> Dict[str, float]:
    """Mean of each metric over the images that have it, plus `images` and `n_<region>`."""
    rows = list(rows)
    out: Dict[str, float] = {}
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    for key in keys:
        values = [row[key] for row in rows if row.get(key) is not None]
        if values:
            out[key] = float(np.mean(values))
    out["images"] = float(len(rows))
    out["images_with_shadow"] = float(sum(1 for r in rows if r.get("psnr_shadow") is not None))
    return out


__all__ = [
    "METRICS",
    "REGIONS",
    "SSIM_NOTE",
    "aggregate",
    "crop_mask",
    "derive_mask",
    "luminance",
    "opening",
    "rgb_to_lab",
    "score_image",
    "ssim_map",
]
