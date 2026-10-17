"""Tensor transforms for paired image/target data.

Deliberately minimal and dependency-free: the inputs are already tensors from
`symtrs.data.io`, so a full augmentation library would be more surface than the
benchmark needs.

The one rule that matters: a geometric transform must be applied to the image and
its target with identical parameters. `RandomCropPair` and `RandomFlipPair` are
built for that; `flip_pair` and `crop_pair` are the functional forms.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

#: ImageNet statistics, used for backbones pretrained on ImageNet (most of them).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def normalize_image(
    image: torch.Tensor,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> torch.Tensor:
    """Per-channel normalisation. Expects a 3-channel `[C, H, W]` or `[B, C, H, W]`."""
    if image.shape[-3] != len(mean):
        raise ValueError(
            f"normalize_image expects {len(mean)} channels, got {image.shape[-3]}"
        )
    mean_t = torch.tensor(mean, dtype=image.dtype, device=image.device).view(-1, 1, 1)
    std_t = torch.tensor(std, dtype=image.dtype, device=image.device).view(-1, 1, 1)
    return (image - mean_t) / std_t


def denormalize_image(
    image: torch.Tensor,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> torch.Tensor:
    mean_t = torch.tensor(mean, dtype=image.dtype, device=image.device).view(-1, 1, 1)
    std_t = torch.tensor(std, dtype=image.dtype, device=image.device).view(-1, 1, 1)
    return image * std_t + mean_t


def crop_pair(
    image: torch.Tensor,
    target: torch.Tensor,
    top: int,
    left: int,
    size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Crop both tensors to the same window. Targets keep their dtype and are never
    interpolated, because resampling a depth map or a label map invents data."""
    height, width = image.shape[-2:]
    size_h = min(size, height - top)
    size_w = min(size, width - left)
    image_crop = image[..., top : top + size_h, left : left + size_w]
    target_crop = target[..., top : top + size_h, left : left + size_w] if target.dim() >= 3 else target[top : top + size_h, left : left + size_w]
    return image_crop, target_crop


def random_crop_pair(
    image: torch.Tensor,
    target: torch.Tensor,
    size: int,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Random crop of `size`, or the whole image when it is already that small.

    Returning the whole image rather than padding keeps the 256-resolution runs
    honest: every sample stays real pixels.
    """
    height, width = image.shape[-2:]
    if size <= 0 or (height <= size and width <= size):
        return image, target
    top = int(torch.randint(0, max(height - size, 0) + 1, (1,), generator=generator).item())
    left = int(torch.randint(0, max(width - size, 0) + 1, (1,), generator=generator).item())
    return crop_pair(image, target, top, left, size)


def center_crop_pair(image: torch.Tensor, target: torch.Tensor, size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Deterministic centre crop, used at evaluation so results are comparable."""
    height, width = image.shape[-2:]
    if size <= 0 or (height <= size and width <= size):
        return image, target
    top = max((height - size) // 2, 0)
    left = max((width - size) // 2, 0)
    return crop_pair(image, target, top, left, size)


def flip_pair(
    image: torch.Tensor,
    target: torch.Tensor,
    horizontal: bool,
    vertical: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if horizontal:
        image = torch.flip(image, dims=[-1])
        target = torch.flip(target, dims=[-1])
    if vertical:
        image = torch.flip(image, dims=[-2])
        target = torch.flip(target, dims=[-2])
    return image, target


def resize_target(target: torch.Tensor, size: Tuple[int, int], mode: str = "bilinear") -> torch.Tensor:
    """Resize a continuous target (depth). Never use this on label maps."""
    if target.dim() == 2:
        target = target.unsqueeze(0)
    resized = F.interpolate(
        target.unsqueeze(0).float(), size=size, mode=mode, align_corners=False
    )
    return resized.squeeze(0)


def resize_labels(labels: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """Nearest-neighbour resize for label maps. Interpolating labels is always wrong."""
    if labels.dim() == 2:
        labels = labels.unsqueeze(0)
    resized = F.interpolate(labels.unsqueeze(0).float(), size=size, mode="nearest")
    return resized.squeeze(0).long()


def match_spatial(tensors: Sequence[torch.Tensor]):
    """Centre-crop every tensor to the smallest common `(H, W)`.

    Image and target should already agree, but a mismatched export would otherwise
    produce an out-of-range crop that fails deep inside the dataloader worker. Doing
    it here turns that into a consistent, visible behaviour.
    """
    heights = [t.shape[-2] for t in tensors]
    widths = [t.shape[-1] for t in tensors]
    height, width = min(heights), min(widths)
    out = []
    for tensor in tensors:
        top = (tensor.shape[-2] - height) // 2
        left = (tensor.shape[-1] - width) // 2
        if tensor.dim() >= 3:
            out.append(tensor[..., top : top + height, left : left + width])
        else:
            out.append(tensor[top : top + height, left : left + width])
    return out


def sample_crop_params(
    height: int,
    width: int,
    size: int,
    generator: Optional[torch.Generator] = None,
) -> Optional[Tuple[int, int, int, int]]:
    """A single `(top, left, h, w)` crop box, or None to mean "whole image".

    Sampling the box once and applying it to every tensor of a sample is what keeps
    image, depth and validity mask aligned. `PairedTransform` covers the two-tensor
    case; the task datasets use this directly because they carry three.
    """
    if size <= 0 or (height <= size and width <= size):
        return None
    top = int(torch.randint(0, max(height - size, 0) + 1, (1,), generator=generator).item())
    left = int(torch.randint(0, max(width - size, 0) + 1, (1,), generator=generator).item())
    return (top, left, min(size, height - top), min(size, width - left))


def center_crop_params(
    height: int,
    width: int,
    size: int,
) -> Optional[Tuple[int, int, int, int]]:
    """The `(top, left, h, w)` box of a centred `size` crop, or None for "whole image".

    The deterministic counterpart of `sample_crop_params`. The task datasets use it for any
    sample that is not being augmented: a validation or test image must be the same pixels
    on every pass, or two evaluations of one checkpoint cannot be compared and "best" is
    picked on noise. A random evaluation crop is exactly what scored the 2048px rows on a
    different 512px window every time.
    """
    if size <= 0 or (height <= size and width <= size):
        return None
    top = max((height - size) // 2, 0)
    left = max((width - size) // 2, 0)
    return (top, left, min(size, height - top), min(size, width - left))


def apply_crop(tensors: Sequence[torch.Tensor], box: Optional[Tuple[int, int, int, int]]):
    """Apply one crop box to every tensor in `tensors`."""
    if box is None:
        return list(tensors)
    top, left, height, width = box
    cropped = []
    for tensor in tensors:
        if tensor.dim() >= 3:
            cropped.append(tensor[..., top : top + height, left : left + width])
        else:
            cropped.append(tensor[top : top + height, left : left + width])
    return cropped


def flip_tensors(
    tensors: Sequence[torch.Tensor],
    horizontal: bool,
    vertical: bool = False,
):
    """Flip every tensor in `tensors` along the same axes."""
    if not horizontal and not vertical:
        return list(tensors)
    dims = []
    if vertical:
        dims.append(-2)
    if horizontal:
        dims.append(-1)
    return [torch.flip(tensor, dims=dims) for tensor in tensors]


@dataclass
class PairedTransform:
    """Train-time geometry for one sample.

    Apply to the image and its target together; the same crop and flip is used for
    both, which is the whole point of the class.
    """

    patch_size: int = 0
    horizontal_flip: bool = True
    vertical_flip: bool = False
    augment: bool = True

    def __call__(self, image: torch.Tensor, target: torch.Tensor):
        if self.augment:
            if self.horizontal_flip and random.random() < 0.5:
                image, target = flip_pair(image, target, horizontal=True, vertical=False)
            if self.vertical_flip and random.random() < 0.5:
                image, target = flip_pair(image, target, horizontal=False, vertical=True)
        if self.patch_size > 0:
            image, target = random_crop_pair(image, target, self.patch_size)
        return image, target


@dataclass
class EvalTransform:
    """Deterministic evaluation geometry: an optional centre crop, nothing random."""

    patch_size: int = 0

    def __call__(self, image: torch.Tensor, target: torch.Tensor):
        if self.patch_size > 0:
            return center_crop_pair(image, target, self.patch_size)
        return image, target
