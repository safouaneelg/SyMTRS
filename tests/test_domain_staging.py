"""pix2pix pairs may be staged at the size upstream will read, and switch size safely.

Why this exists: `write_aligned_pairs` writes one lossless PNG per scene, and at 2048 that is
~200 GB and ~9 h of single-core encoding per direction, against ~5 GB and ~20 min at the 286 px
upstream resizes to anyway. The pairs have to stay honest though: a folder half native and half
resized would train on two different rescalings of the same scene, so the size is recorded in
the folder and a change rebuilds it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PIL")

from PIL import Image  # noqa: E402

from symtrs.model.domain.upstream import PAIR_SIZE_MARKER, write_aligned_pairs  # noqa: E402


def make_split(root: Path, name: str, count: int, side: int = 32, colour=(10, 20, 30)):
    """`<root>/<name>A` and `<root>/<name>B` with `count` equal-size images each."""
    for domain, fill in (("A", colour), ("B", (200, 100, 50))):
        folder = root / f"{name}{domain}"
        folder.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            Image.new("RGB", (side, side), fill).save(folder / f"scene{index}.png")
    return [f"scene{index}.png" for index in range(count)]


def test_native_pairs_keep_their_size(tmp_path):
    keys = make_split(tmp_path, "train", 2)
    staging = tmp_path
    write_aligned_pairs(staging, "train", keys)
    pair = Image.open(staging / "train" / "scene0.png")
    assert pair.size == (64, 32)
    assert (staging / "train" / PAIR_SIZE_MARKER).read_text() == "0"


def test_pairs_can_be_staged_at_the_size_upstream_reads(tmp_path):
    keys = make_split(tmp_path, "train", 2)
    written = write_all_pairs(tmp_path, "train", keys, size=8)
    assert written == 2
    pair = Image.open(tmp_path / "train" / "scene0.png")
    assert pair.size == (16, 8), "each half is `size`, the file holds A then B"
    assert (tmp_path / "train" / PAIR_SIZE_MARKER).read_text() == "8"


def test_changing_the_size_rebuilds_instead_of_mixing(tmp_path):
    keys = make_split(tmp_path, "train", 2)
    write_all_pairs(tmp_path, "train", keys)
    assert Image.open(tmp_path / "train" / "scene0.png").size == (64, 32)
    # Same keys, same mtimes: a size-agnostic reuse check would keep the native pairs.
    written = write_all_pairs(tmp_path, "train", keys, size=8)
    assert written == 2, "every pair is rewritten when the recorded size changes"
    assert Image.open(tmp_path / "train" / "scene0.png").size == (16, 8)


def test_marker_survives_the_stale_file_sweep(tmp_path):
    keys = make_split(tmp_path, "train", 2)
    write_all_pairs(tmp_path, "train", keys, size=8)
    write_all_pairs(tmp_path, "train", [keys[0]], size=8)  # one scene left the split
    folder = tmp_path / "train"
    assert (folder / PAIR_SIZE_MARKER).is_file()
    assert not (folder / "scene1.png").exists()


def write_all_pairs(root: Path, name: str, keys, size: int | None = None) -> int:
    """`write_aligned_pairs` with the size only passed when the caller set one."""
    return write_aligned_pairs(root, name, keys) if size is None else write_aligned_pairs(
        root, name, keys, size=size
    )
