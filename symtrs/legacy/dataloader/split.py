# E:\dataloader\split.py
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import random


def _validate_ratios(ratios: Sequence[float]) -> List[float]:
    if len(ratios) < 2:
        raise ValueError("Provide at least two split ratios")
    total = sum(ratios)
    if total <= 0:
        raise ValueError("Ratios must sum to a positive value")
    return [r / total for r in ratios]


def split_indices(
    n: int,
    ratios: Sequence[float] = (0.8, 0.2),
    seed: int = 42,
    shuffle: bool = True,
) -> List[List[int]]:
    ratios = _validate_ratios(ratios)
    indices = list(range(n))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(indices)

    split_points = []
    acc = 0
    for r in ratios[:-1]:
        acc += int(round(r * n))
        split_points.append(acc)

    splits = []
    start = 0
    for end in split_points:
        splits.append(indices[start:end])
        start = end
    splits.append(indices[start:])
    return splits


def split_list(
    items: Sequence,
    ratios: Sequence[float] = (0.8, 0.2),
    seed: int = 42,
    shuffle: bool = True,
):
    indices_splits = split_indices(len(items), ratios=ratios, seed=seed, shuffle=shuffle)
    return [[items[i] for i in idxs] for idxs in indices_splits]
