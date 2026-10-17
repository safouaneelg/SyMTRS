"""Distribution metrics for the domain-adaptation task.

FID and KID both compare the *distribution* of a set of generated images against a set
of real ones in the feature space of an ImageNet classifier, so they measure whether a
day-to-night translation looks like the target domain rather than merely preserving
content. That is exactly what a pixel-wise metric cannot tell you.

Two conventions are implemented deliberately rather than taken from
`torchvision.models.Inception_V3_Weights.transforms()`:

  * Images are resized to 299 with no centre crop. The original FID code resizes and
    nothing else; the classifier's own eval transform centre-crops, which would throw
    away the border of every image and change the number.
  * Features are the 2048-d pooled activations before the classifier, not the logits.
    Logits are already collapsed onto 1000 ImageNet classes and carry far less
    information about an aerial night scene.

Both matter for numbers being comparable with anything published.

Two further conventions exist because a SyMTRS comparison is *asymmetric* by construction:

  * The real side is native 2048 px, the generated side is whatever upstream wrote (256 px
    for CycleGAN/pix2pix `test.py`). Pushing both straight to 299 would hand the real set an
    aliased downscale (x7) and the generated set a smooth upscale (x1.2), and Inception's
    features are sensitive to exactly that, so FID would partly measure sharpness rather
    than translation quality. `match_resolution` therefore brings the real images to the
    generated size first (antialiased), and `inception_preprocess` then applies one
    identical antialiased resize to 299 to both. The generated size is read from the
    generated files by the caller, never hard-coded here.
  * **KID is the primary number, FID is kept for reference.** The validation/test splits
    hold tens of images (24 per split at the time of writing) against 2048 feature
    dimensions: the covariance matrix has rank at most N-1, so FID is badly biased and
    its sign of improvement is not trustworthy, whereas KID's MMD^2 estimator is unbiased
    at any N (it can be slightly negative, which is not a bug). `PRIMARY_METRIC` names
    it and `symtrs tables` orders `kid` before `fid`.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#: ImageNet statistics, which the Inception weights were trained with.
INCEPTION_MEAN = (0.485, 0.456, 0.406)
INCEPTION_STD = (0.229, 0.224, 0.225)

#: FID resolution. Fixed by the Inception network's trained input size.
FID_RESOLUTION = 299

#: The distribution metric to quote. FID is still computed and reported, but its
#: covariance is rank-deficient at the split sizes used here (see the module docstring).
PRIMARY_METRIC = "kid"


def match_resolution(images: torch.Tensor, size) -> torch.Tensor:
    """Resize `[B, 3, H, W]` to `size` (int or `(h, w)`) with antialiasing.

    Used on the real side so it reaches Inception through the same low-resolution path as
    the generated images. A no-op when the size already matches, so generated images are
    never resampled before the shared 299 resize.
    """
    target = (size, size) if isinstance(size, int) else (int(size[0]), int(size[1]))
    if tuple(images.shape[-2:]) == target:
        return images
    return F.interpolate(
        images.float(), size=target, mode="bilinear", align_corners=False, antialias=True
    )


def inception_preprocess(images: torch.Tensor, size: int = FID_RESOLUTION) -> torch.Tensor:
    """`[B, 3, H, W]` in `[0, 1]` to Inception input, resizing only.

    `antialias=True` so a large image is low-passed before it is subsampled; it changes
    nothing when upsampling (the generated 256 px side), which is why one call serves both.
    """
    if images.dim() == 3:
        images = images.unsqueeze(0)
    resized = F.interpolate(
        images.float(), size=(size, size), mode="bilinear", align_corners=False,
        antialias=True,
    )
    mean = torch.tensor(INCEPTION_MEAN, device=resized.device).view(1, 3, 1, 1)
    std = torch.tensor(INCEPTION_STD, device=resized.device).view(1, 3, 1, 1)
    return (resized - mean) / std


class InceptionFeatures(nn.Module):
    """2048-d pooled Inception-V3 features.

    The classifier is replaced by `Identity`, which leaves the pooled activation as the
    output - the last layer that still encodes scene content rather than a decision
    about it.
    """

    def __init__(self, device: Optional[torch.device] = None) -> None:
        super().__init__()
        from torchvision.models import Inception_V3_Weights, inception_v3

        self.model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        self.model.fc = nn.Identity()
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        if device is not None:
            self.model.to(device)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(inception_preprocess(images).to(self.device))


@torch.no_grad()
def extract_features(
    loader,
    extractor: Optional[InceptionFeatures] = None,
    max_samples: int = 0,
    image_key: str = "image",
    progress: bool = True,
    resize_to=None,
) -> np.ndarray:
    """Inception features for every image a dataloader yields.

    `resize_to` (int or `(h, w)`) first brings every image to that size with
    `match_resolution`; pass the generated images' size for the real side.
    """
    extractor = extractor or InceptionFeatures()
    features: List[np.ndarray] = []
    seen = 0

    for batch in loader:
        images = batch[image_key] if isinstance(batch, dict) else batch
        if images.dim() == 3:
            images = images.unsqueeze(0)
        if images.shape[1] != 3:
            # A grayscale preview or a single-channel depth map would otherwise fail
            # deep inside Inception with a shape error nobody can trace.
            images = images.repeat(1, 3, 1, 1)
        if resize_to is not None:
            images = match_resolution(images, resize_to)
        features.append(extractor(images).cpu().numpy())
        seen += images.shape[0]
        if progress and len(features) % 20 == 0:
            print(f"[symtrs] features: {seen} images")
        if max_samples and seen >= max_samples:
            break

    if not features:
        raise RuntimeError("no images were yielded, so no features could be extracted")
    stacked = np.concatenate(features, axis=0).astype(np.float64)
    return stacked[:max_samples] if max_samples else stacked


def _covariance(features: np.ndarray) -> np.ndarray:
    centred = features - features.mean(axis=0, keepdims=True)
    return (centred.T @ centred) / max(features.shape[0] - 1, 1)


def _sqrtm_symmetric(matrix: np.ndarray) -> np.ndarray:
    """Matrix square root of a symmetric positive-semidefinite matrix.

    Eigendecomposition rather than a Schur/Newton iteration: the covariance is
    symmetric by construction, and anything with a meaningfully negative eigenvalue is
    a sign that the feature set is degenerate (fewer images than dimensions) rather
    than something to be solved numerically.
    """
    values, vectors = np.linalg.eigh(matrix)
    values = np.clip(values, 0.0, None)
    return (vectors * np.sqrt(values)) @ vectors.T


def compute_fid(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    """Frechet Inception Distance between two feature sets."""
    if real_features.ndim != 2 or fake_features.ndim != 2:
        raise ValueError("feature arrays must be 2D (n_images, n_features)")
    if real_features.shape[0] < 2 or fake_features.shape[0] < 2:
        raise ValueError(
            f"FID needs at least 2 images per side; got {_n_images(real_features)} and "
            f"{_n_images(fake_features)}. A number computed from one image is noise, "
            "not a metric."
        )
    if real_features.shape[1] != fake_features.shape[1]:
        raise ValueError(
            f"feature dimensions differ: {real_features.shape[1]} vs {fake_features.shape[1]}"
        )

    mu_real = real_features.mean(axis=0)
    mu_fake = fake_features.mean(axis=0)
    cov_real = _covariance(real_features)
    cov_fake = _covariance(fake_features)

    difference = mu_real - mu_fake
    # The product of two PSD matrices has non-negative eigenvalues in exact arithmetic;
    # clipping removes the negative round-off that would otherwise make sqrtm NaN.
    product = _sqrtm_symmetric(cov_real) @ cov_fake @ _sqrtm_symmetric(cov_real)
    trace_root = np.trace(_sqrtm_symmetric(product).real)
    return float(difference @ difference + np.trace(cov_real) + np.trace(cov_fake) - 2.0 * trace_root)


def _n_images(features: np.ndarray) -> int:
    return int(features.shape[0]) if features.ndim >= 1 else 0


def polynomial_kernel(a: np.ndarray, b: np.ndarray, degree: int = 3, coef: float = 1.0) -> np.ndarray:
    return (a @ b.T / a.shape[1] + coef) ** degree


def compute_kid(
    real_features: np.ndarray,
    fake_features: np.ndarray,
    degree: int = 3,
    coef: float = 1.0,
    num_subsets: int = 0,
    subset_size: int = 100,
    seed: int = 0,
) -> Dict[str, Optional[float]]:
    """Kernel Inception Distance: an unbiased MMD^2 with a polynomial kernel.

    KID is reported instead of FID whenever the validation split is small, because FID's
    covariance estimate is badly biased below a few thousand images while KID is not.
    If `num_subsets > 0` the mean and standard deviation over random subsets are
    returned, which is how the original code reports a confidence interval.

    With `num_subsets=0` (the default, and what the domain runner uses) there is one
    full-set estimate and no spread was measured, so `kid_std` is `None` and lands in
    `summary.json` as `null`. It used to be `0.0`, which reads as a measured zero spread
    ("the estimate is exact") next to an N of 24; a quantity that was never estimated is
    not zero (AGENTS.md rule 11: what cannot be computed is `null`, never a number).
    """
    if real_features.shape[0] < 2 or fake_features.shape[0] < 2:
        raise ValueError("KID needs at least 2 images per side")

    def mmd2(a: np.ndarray, b: np.ndarray) -> float:
        k_aa = polynomial_kernel(a, a, degree, coef)
        k_bb = polynomial_kernel(b, b, degree, coef)
        k_ab = polynomial_kernel(a, b, degree, coef)
        m, n = a.shape[0], b.shape[0]
        # Unbiased estimators: the diagonal is excluded from the within-set terms.
        sum_aa = (k_aa.sum() - np.trace(k_aa)) / (m * (m - 1))
        sum_bb = (k_bb.sum() - np.trace(k_bb)) / (n * (n - 1))
        return float(sum_aa + sum_bb - 2.0 * k_ab.mean())

    if not num_subsets:
        return {"kid": mmd2(real_features, fake_features), "kid_std": None}

    generator = np.random.default_rng(seed)
    values = []
    for _ in range(num_subsets):
        size = min(subset_size, real_features.shape[0], fake_features.shape[0])
        a = real_features[generator.choice(real_features.shape[0], size, replace=False)]
        b = fake_features[generator.choice(fake_features.shape[0], size, replace=False)]
        values.append(mmd2(a, b))
    return {"kid": float(np.mean(values)), "kid_std": float(np.std(values))}


def distribution_metrics(
    real_features: np.ndarray,
    fake_features: np.ndarray,
    with_kid: bool = True,
    num_subsets: int = 0,
) -> Dict[str, Optional[float]]:
    """KID (primary) and FID in one call. KID is skipped below 2 images per side.

    The dict lists `kid` first on purpose: it is the number to quote when N is small.
    """
    results: Dict[str, Optional[float]] = {}
    if with_kid:
        try:
            results.update(compute_kid(real_features, fake_features, num_subsets=num_subsets))
        except ValueError as exc:
            results["kid"] = float("nan")
            print(f"[symtrs] KID not computable: {exc}")
    results["fid"] = compute_fid(real_features, fake_features)
    # With N <= 2048 the covariance is singular; say so next to the number.
    results["fid_rank_deficient"] = float(
        min(real_features.shape[0], fake_features.shape[0]) <= real_features.shape[1]
    )
    results["n_real"] = float(real_features.shape[0])
    results["n_fake"] = float(fake_features.shape[0])
    return results


__all__ = [
    "FID_RESOLUTION",
    "INCEPTION_MEAN",
    "INCEPTION_STD",
    "InceptionFeatures",
    "PRIMARY_METRIC",
    "compute_fid",
    "compute_kid",
    "distribution_metrics",
    "extract_features",
    "inception_preprocess",
    "match_resolution",
    "polynomial_kernel",
]
