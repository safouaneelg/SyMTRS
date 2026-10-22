"""The shadow worker's `Store`: a bounded cache instead of the whole split in RAM.

The preview dataset was 96 pairs (~3 GB), so preloading a split was invisible. The real
split is 14 000 pairs of 2048 px images (~400 GB), and a preload would be killed before the
first iteration - on a machine with more RAM than this one, it would swap instead, which is
worse because it looks like slow training rather than a broken run. The properties pinned
here are the ones the training loop relies on: a crop is the same pixels it would be with
the split in memory, the cache stays bounded, and the prefetch pool fills it.

Needs torch because it imports the worker module; nothing here runs a network.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_pil, requires_torch

pytest.importorskip("torch", reason="torch is not installed")
pytest.importorskip("PIL", reason="Pillow is not installed")

pytestmark = [requires_torch, requires_pil]


def _staging(root, count=6, size=32):
    """`input/`, `target/`, `mask/` with distinguishable per-key images."""
    from PIL import Image

    for role in ("input", "target", "mask"):
        (root / role).mkdir(parents=True, exist_ok=True)
    arrays = {}
    for index in range(count):
        key = f"rgb_day_2048_key{index:02d}"
        image = np.full((size, size, 3), index * 10, dtype=np.uint8)
        image[..., 1] = np.arange(size, dtype=np.uint8)[None, :]
        target = np.full((size, size, 3), 255 - index * 10, dtype=np.uint8)
        mask = np.zeros((size, size), dtype=np.uint8)
        mask[: size // 2] = 255
        Image.fromarray(image).save(root / "input" / f"{key}.png")
        Image.fromarray(target).save(root / "target" / f"{key}.png")
        Image.fromarray(mask).save(root / "mask" / f"{key}.png")
        arrays[key] = (image, target, mask > 127)
    return arrays


def test_a_crop_is_the_same_pixels_as_the_split_held_in_memory(tmp_path):
    from symtrs.model.shadow.worker import Store

    arrays = _staging(tmp_path)
    store = Store(tmp_path, cache_images=2)
    assert len(store) == 6
    for index, key in enumerate(store.keys):
        image, target, mask = store.crop(index, 4, 8, 16)
        expected = arrays[key]
        assert np.array_equal(image, expected[0][4:20, 8:24])
        assert np.array_equal(target, expected[1][4:20, 8:24])
        assert np.array_equal(mask, expected[2][4:20, 8:24].astype(np.uint8))


def test_the_cache_stays_bounded_however_many_images_are_read(tmp_path):
    """The point of the change: reading every image must not keep every image."""
    from symtrs.model.shadow.worker import Store

    _staging(tmp_path, count=12, size=16)
    store = Store(tmp_path, cache_images=3)
    for index in range(len(store)):
        store.crop(index, 0, 0, 8)
    assert len(store.cache) == 3
    assert store.shape(0) == (16, 16)


def test_prefetch_decodes_ahead_and_get_returns_the_same_image(tmp_path):
    from symtrs.model.shadow.worker import Store

    arrays = _staging(tmp_path)
    store = Store(tmp_path, cache_images=4)
    store.prefetch([0, 1, 2])
    image, _, _ = store.get(1)
    assert np.array_equal(image, arrays[store.keys[1]][0])
    # Every prefetched index is either resolved into the cache or still pending; a second
    # prefetch of the same indices must not queue a duplicate decode.
    pending_before = len(store.pending)
    store.prefetch([0, 1, 2])
    assert len(store.pending) == pending_before


def test_a_split_with_nothing_staged_is_an_error(tmp_path):
    from symtrs.model.shadow.worker import Store

    for role in ("input", "target", "mask"):
        (tmp_path / role).mkdir()
    with pytest.raises(RuntimeError, match="nothing staged"):
        Store(tmp_path)
