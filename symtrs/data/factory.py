"""The `dataloader(...)` factory.

Two ways in, both landing on the same dataset classes:

    from symtrs.data import dataloader

    # explicit
    loader = dataloader(task="depth", root="/data/SyMTRS", split="train",
                        resolution=1024, batch_size=8)

    # from a resolved run config
    loader = dataloader.from_config(cfg, split="val")

The dataset classes themselves are also exported directly for anyone who wants to
write their own loop:

    from symtrs.data import DepthTaskDataset, SemanticDataset
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from symtrs.config import RunConfig
from symtrs.data.depth_dataset import DepthTaskDataset, depth_coverage
from symtrs.data.semantic_dataset import (
    DEFAULT_LABEL_STEP,
    DEFAULT_NUM_CLASSES,
    SemanticDataset,
    check_num_classes,
    detect_class_values,
    semantic_coverage,
)
from symtrs.data.splits import (
    RESOLUTIONS,
    SPLITS,
    MissingDirectoryError,
    SplitDirs,
    available_resolutions,
    describe_layout,
    resolve_split_dirs,
)

#: Tasks that expose a tensor dataset. `domain_adapt` is path-based: it hands file
#: lists to the vendored upstream loaders rather than yielding tensors, and
#: `superresolution` is served by the pre-existing training script.
TENSOR_TASKS = ("depth", "segmentation")


def build_dataset(
    task: str,
    root: str,
    split: str = "train",
    resolution: int = 2048,
    train: bool = True,
    patch_size: int = 0,
    augment: Optional[bool] = None,
    depth_scale: float = 0.01,
    max_depth: Optional[float] = None,
    label_step: int = DEFAULT_LABEL_STEP,
    depth_channel: int = 0,
    depth_backend: str = "auto",
    num_classes: int = DEFAULT_NUM_CLASSES,
    ignore_index: int = 255,
    max_samples: int = 0,
    generator: Optional[torch.Generator] = None,
) -> Dataset:
    """Build the dataset for one task/split/resolution.

    `augment` defaults to whatever `train` implies, which is the behaviour you want
    almost always; it is explicit so a validation loader can be built from a config
    that was written for training.
    """
    if task not in TENSOR_TASKS:
        raise ValueError(
            f"task {task!r} has no tensor dataset; expected one of {TENSOR_TASKS}"
        )
    if augment is None:
        augment = bool(train)

    if task == "depth":
        return DepthTaskDataset(
            root=root,
            split=split,
            resolution=resolution,
            depth_scale=depth_scale,
            max_depth=max_depth,
            depth_channel=depth_channel,
            depth_backend=depth_backend,
            max_samples=max_samples,
            patch_size=patch_size,
            augment=augment,
            generator=generator,
        )

    return SemanticDataset(
        root=root,
        split=split,
        resolution=resolution,
        num_classes=num_classes,
        label_step=label_step,
        ignore_index=ignore_index,
        max_samples=max_samples,
        patch_size=patch_size,
        augment=augment,
        generator=generator,
    )


def dataloader(
    task: str,
    root: str,
    split: str = "train",
    resolution: int = 2048,
    batch_size: int = 1,
    shuffle: Optional[bool] = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
    train: bool = True,
    patch_size: int = 0,
    sampler: Any = None,
    generator: Optional[torch.Generator] = None,
    **dataset_kwargs: Any,
) -> DataLoader:
    """Build a `torch.utils.data.DataLoader` for a SyMTRS task.

    `shuffle` defaults to True for the training split and False otherwise, which is
    the split-and-shuffle pairing that silently leaks validation data when it is
    wrong.
    """
    dataset = build_dataset(
        task=task,
        root=root,
        split=split,
        resolution=resolution,
        train=train,
        patch_size=patch_size,
        generator=generator,
        **dataset_kwargs,
    )
    if shuffle is None:
        shuffle = bool(train) and sampler is None

    # Deterministic shuffling under a seeded generator only applies to DataLoader's
    # own RandomSampler.
    loader_kwargs: dict = {
        "batch_size": batch_size,
        "shuffle": shuffle if sampler is None else False,
        "num_workers": num_workers,
        "pin_memory": pin_memory and torch.cuda.is_available(),
        "drop_last": drop_last,
    }
    if sampler is not None:
        loader_kwargs["sampler"] = sampler
    elif generator is not None:
        loader_kwargs["generator"] = generator

    return DataLoader(dataset, **loader_kwargs)


def from_config(cfg: RunConfig, split: Optional[str] = None, train: bool = True, **overrides: Any) -> DataLoader:
    """Build the loader a run config asks for.

    `split` defaults to the config's training split, or to its val split when
    `train=False`, which is what the engine needs for its validation pass.
    """
    if split is None:
        split = cfg.data.split if train else cfg.data.val_split

    # Evaluation never inherits the training crop. `val_patch_size: 0` means the WHOLE image,
    # which is what the configs say; falling back to `train.patch_size` when it was 0 scored
    # every 2048px val/test pass (model selection included) on one random 512px window, so a
    # 2048 number was neither reproducible nor comparable with the whole-image 1024/512/256
    # rows. A positive `val_patch_size` is a deterministic centre crop (see the datasets);
    # images too large for one forward pass are handled by `eval.tile_size`, not by cropping.
    patch_size = overrides.pop(
        "patch_size",
        cfg.train.patch_size if train else cfg.train.val_patch_size,
    )
    limit = overrides.pop(
        "limit_batches",
        cfg.data.limit_train_batches if train else cfg.data.limit_val_batches,
    )

    loader = dataloader(
        task=cfg.task,
        root=cfg.data.root,
        split=split,
        resolution=cfg.resolution,
        batch_size=cfg.train.batch_size if train else cfg.eval.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        train=train,
        patch_size=patch_size,
        depth_scale=cfg.data.depth_scale,
        max_depth=cfg.data.max_depth,
        label_step=cfg.data.label_step,
        depth_channel=cfg.data.depth_channel,
        depth_backend=cfg.data.depth_backend,
        num_classes=cfg.data.num_classes,
        ignore_index=cfg.data.ignore_index,
        max_samples=cfg.data.max_samples,
        **overrides,
    )
    if limit and limit > 0:
        loader = _LimitedLoader(loader, limit)
    return loader


class _LimitedLoader:
    """Wrap a DataLoader to yield at most `limit` batches.

    Used by `limit_train_batches` / `limit_val_batches` so a smoke run over a real
    dataset finishes in seconds without needing a separate miniature copy.
    """

    def __init__(self, loader: DataLoader, limit: int) -> None:
        self.loader = loader
        self.limit = int(limit)

    def __iter__(self):
        for index, batch in enumerate(self.loader):
            if index >= self.limit:
                break
            yield batch

    def __len__(self) -> int:
        return min(len(self.loader), self.limit)

    @property
    def dataset(self):
        return self.loader.dataset


dataloader.from_config = from_config  # type: ignore[attr-defined]

__all__ = [
    "DEFAULT_NUM_CLASSES",
    "DepthTaskDataset",
    "MissingDirectoryError",
    "RESOLUTIONS",
    "SPLITS",
    "SemanticDataset",
    "SplitDirs",
    "available_resolutions",
    "build_dataset",
    "check_num_classes",
    "dataloader",
    "describe_layout",
    "detect_class_values",
    "depth_coverage",
    "resolve_split_dirs",
    "semantic_coverage",
]
