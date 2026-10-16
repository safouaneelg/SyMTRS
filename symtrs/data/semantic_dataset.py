"""Semantic segmentation dataset over the revised SyMTRS layout.

Labels are 8-bit grayscale PNGs holding `class_id * label_step` (the current export:
step 63, 5 classes - background, building, green, road, car). `decode_labels` turns the
grey level back into a class index; a grey level that is not near a multiple of the step
becomes `ignore_index` rather than being rounded into some class. The class count is a
configuration value that `detect_class_values` cross-checks rather than a value the code
infers silently.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from symtrs.data.io import list_images, check_pairing, match_by_stem, read_image, read_label_map, coverage_report
from symtrs.data.splits import SplitDirs, resolve_split_dirs
from symtrs.data.transforms import apply_crop, flip_tensors, match_spatial, sample_crop_params

#: Default class count and grey-level step for the revised SyMTRS semantic labels. The
#: defaults live in `symtrs/config.py`; these mirror them for direct dataset use.
DEFAULT_NUM_CLASSES = 5
DEFAULT_LABEL_STEP = 63
CLASS_NAMES = ("background", "building", "green", "road", "car")

#: Grey levels a pixel may sit away from a multiple of the step and still be that class
#: (resampling or PNG quantisation); beyond it the pixel is unlabelled, not rounded.
LABEL_TOLERANCE = 2


def decode_labels(raw: torch.Tensor, label_step: int, ignore_index: int = 255) -> torch.Tensor:
    """Grey levels -> class indices: `round(v / label_step)`, off-grid -> `ignore_index`.

    With `label_step == 1` the mask already holds class ids and is returned as read.
    Rounding silently would turn a stray 100 into class 2; marking it ignored keeps it out
    of the loss and the metrics, where the cost of being wrong is a missing pixel and not
    a wrong label.
    """
    if label_step <= 1:
        return raw
    step = int(label_step)
    classes = torch.div(raw + step // 2, step, rounding_mode="floor")
    off_grid = (raw - classes * step).abs() > LABEL_TOLERANCE
    return torch.where(off_grid, torch.full_like(classes, int(ignore_index)), classes)


class SemanticDataset(Dataset):
    """Image -> class-index label map at one resolution.

    Yields:
        image: float32 `[3, H, W]` in `[0, 1]`.
        mask: int64 `[H, W]`, class indices, `ignore_index` where unlabelled.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        resolution: int = 2048,
        num_classes: int = DEFAULT_NUM_CLASSES,
        label_step: int = DEFAULT_LABEL_STEP,
        ignore_index: int = 255,
        max_samples: int = 0,
        patch_size: int = 0,
        augment: bool = False,
        horizontal_flip: bool = True,
        vertical_flip: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        self.dirs: SplitDirs = resolve_split_dirs(root, split, resolution)
        self.split = split
        self.resolution = resolution
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.label_step = int(label_step)
        self.patch_size = int(patch_size)
        self.augment = bool(augment)
        self.horizontal_flip = bool(horizontal_flip)
        self.vertical_flip = bool(vertical_flip)
        self.generator = generator

        day_dir = self.dirs.path_for("day")
        semantic_dir = self.dirs.path_for("semantic")

        images = list_images(day_dir)
        labels = list_images(semantic_dir)
        pairs = match_by_stem(images, labels)

        check_pairing(pairs, images, labels, "image/semantic", split, resolution)
        if not pairs:
            raise RuntimeError(
                f"no image/semantic pairs for split={split!r} resolution={resolution}.\n"
                f"  images:   {day_dir} ({len(images)} files)\n"
                f"  semantic: {semantic_dir} ({len(labels)} files)\n"
                "  run `symtrs doctor --data-root <dataset root>` for a coverage report"
            )

        if max_samples and max_samples > 0:
            pairs = pairs[:max_samples]
        self.pairs: List[Tuple[Path, Path]] = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        attempts = 0
        while attempts < len(self.pairs):
            image_path, mask_path = self.pairs[index]
            try:
                image = read_image(image_path, mode="rgb")
                mask = decode_labels(read_label_map(mask_path), self.label_step, self.ignore_index)
                break
            except Exception as exc:
                print(
                    f"[symtrs.data] skipping unreadable pair image={image_path} "
                    f"mask={mask_path}: {type(exc).__name__}: {exc}"
                )
                attempts += 1
                index = (index + 1) % len(self.pairs)
        else:
            raise RuntimeError(
                f"all {len(self.pairs)} semantic pairs for split={self.split!r} "
                f"resolution={self.resolution} failed to load"
            )

        image, mask = match_spatial([image, mask])

        height, width = image.shape[-2:]
        box = sample_crop_params(height, width, self.patch_size, self.generator)
        if self.augment:
            horizontal = self.horizontal_flip and torch.rand(1, generator=self.generator).item() < 0.5
            vertical = self.vertical_flip and torch.rand(1, generator=self.generator).item() < 0.5
            image, mask = flip_tensors([image, mask], horizontal, vertical)
        image, mask = apply_crop([image, mask], box)

        return {
            "image": image,
            "mask": mask.long(),
            "image_path": str(image_path),
            "mask_path": str(mask_path),
        }


def detect_class_values(
    root: str,
    split: str = "train",
    resolution: int = 2048,
    sample: int = 16,
    label_step: int = 1,
    ignore_index: int = 255,
) -> Counter:
    """Count label pixel values over a sample of the split.

    Returns a `Counter` value -> pixel count. By default the grey levels are counted as
    stored (`label_step=1`), so `symtrs inspect` shows what is really on disk; pass the
    configured step to count decoded class indices instead. Used by `symtrs inspect` and to warn
    when the configured `num_classes` disagrees with the data.
    """
    dirs = resolve_split_dirs(root, split, resolution)
    labels = list_images(dirs.path_for("semantic"))
    if not labels:
        return Counter()

    step = max(len(labels) // max(sample, 1), 1)
    counter: Counter = Counter()
    for path in labels[::step][:sample]:
        try:
            values, counts = torch.unique(
                decode_labels(read_label_map(path), label_step, ignore_index), return_counts=True
            )
        except Exception as exc:
            print(f"[symtrs.data] could not inspect {path}: {type(exc).__name__}: {exc}")
            continue
        for value, count in zip(values.tolist(), counts.tolist()):
            counter[int(value)] += int(count)
    return counter


def check_num_classes(
    counter: Counter,
    num_classes: int = DEFAULT_NUM_CLASSES,
    ignore_index: int = 255,
) -> Optional[str]:
    """Return a warning string when the data disagrees with the config, else None.

    Advisory only. Two failure modes are worth catching before a long run: a class
    value outside `[0, num_classes)` would be silently clamped and trained as the
    wrong class, and an entirely unused class usually means a wrong `num_classes`.
    """
    if not counter:
        return None

    present = sorted(v for v in counter if v != ignore_index)
    if not present:
        return None

    problems = []
    out_of_range = [v for v in present if v < 0 or v >= num_classes]
    if out_of_range:
        problems.append(
            f"label values {out_of_range} fall outside [0, {num_classes}); they would "
            "be clamped to a valid class and trained as the wrong label. "
            "Raise data.num_classes."
        )

    missing = [c for c in range(num_classes) if c not in counter]
    if missing:
        problems.append(
            f"classes {missing} never appear in the sampled labels; check that "
            "data.num_classes is right and that the split is complete."
        )

    if counter.get(ignore_index, 0) > 0:
        problems.append(
            f"ignore_index={ignore_index} appears in the labels as a real value; "
            "those pixels will be excluded from the loss and the metrics."
        )

    return " ".join(problems) if problems else None


def semantic_coverage(root: str, split: str = "train", resolution: int = 2048) -> dict:
    dirs = resolve_split_dirs(root, split, resolution)
    return coverage_report(list_images(dirs.day), list_images(dirs.semantic))
