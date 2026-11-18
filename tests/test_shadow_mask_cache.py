"""The shared shadow-mask cache: derived once per (pair, settings), valid only while the sources are.

Why it exists: a mask costs ~1 s per 2048 px pair and every shadow run (4 shadow-removal models
plus the shadow2noshadow translators' scoring) used to re-derive all of them into its own
`<run_dir>/staging/`; at 20,000+ views that is ~5.5 h of identical work per run. These tests
protect the three ways a cache can lie (stale after a source or setting changed, half-written
after a crash, shared between datasets) and the one way it can hurt (writing into the dataset).
Pairs are tiny PIL images, so nothing here needs torch or the real data.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from conftest import REPO_ROOT, requires_numpy, requires_pil

pytestmark = [requires_numpy, requires_pil]


def _make_root(tmp_path: Path, n: int = 3, splits=("train",)) -> Path:
    import numpy as np
    from PIL import Image

    root = tmp_path / "SyMTRS"
    for split in splits:
        (root / split / "rgb_day").mkdir(parents=True)
        (root / split / "rgb_noshadow").mkdir(parents=True)
        for i in range(n):
            clean = np.full((32, 32, 4), 200, np.uint8)
            dark = clean.copy()
            dark[4 + i : 20 + i, 6:22, :3] = 60 + 10 * i
            Image.fromarray(dark).save(root / split / "rgb_day" / f"rgb_day_2048_loc{i:03d}.png")
            Image.fromarray(clean).save(
                root / split / "rgb_noshadow" / f"rgb_noshadow_2048_loc{i:03d}.png"
            )
    return root


@pytest.fixture
def counted(monkeypatch):
    """Counts calls to `derive_mask` (serial fills only: the pool runs in other processes)."""
    import symtrs.metrics.shadow as shadow

    calls = []
    real = shadow.derive_mask

    def wrapper(*args, **kwargs):
        calls.append(args[2:])
        return real(*args, **kwargs)

    monkeypatch.setattr(shadow, "derive_mask", wrapper)
    return calls


def _stage(root, run, cache, split="train", threshold=0.1, open_size=3, workers=0):
    from symtrs.model.shadow.staging import stage_split

    return stage_split(
        str(root), str(run), split, threshold, open_size, cache=str(cache), workers=workers
    )


def _snapshot(folder: Path):
    return {
        str(p.relative_to(folder)): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in sorted(folder.rglob("*"))
    }


def test_second_staging_computes_nothing_even_in_another_run_dir(tmp_path, counted):
    root, cache = _make_root(tmp_path), tmp_path / "cache"
    _stage(root, tmp_path / "run_a", cache)
    assert len(counted) == 3
    _stage(root, tmp_path / "run_a", cache)
    staging_b = _stage(root, tmp_path / "run_b", cache)  # a different run shares the cache
    assert len(counted) == 3, "the cache must make later runs derive nothing"
    assert sorted(p.name for p in (staging_b / "mask").glob("*.png")) == [
        "loc000.png", "loc001.png", "loc002.png"
    ]


def test_staged_mask_is_bit_identical_to_a_direct_derivation(tmp_path):
    import io

    import numpy as np
    from PIL import Image

    from symtrs.metrics.shadow import derive_mask
    from symtrs.model.shadow.staging import pairs_for

    root = _make_root(tmp_path)
    staging = _stage(root, tmp_path / "run", tmp_path / "cache")
    for key, day, clean in pairs_for(str(root), "train"):
        a = np.asarray(Image.open(day).convert("RGB"), dtype=np.float32) / 255.0
        b = np.asarray(Image.open(clean).convert("RGB"), dtype=np.float32) / 255.0
        expected = Image.fromarray(derive_mask(a, b, 0.1, 3).astype(np.uint8) * 255, mode="L")
        buffer = io.BytesIO()
        expected.save(buffer, format="PNG")
        assert (staging / "mask" / f"{key}.png").read_bytes() == buffer.getvalue()


def test_other_settings_use_their_own_folder(tmp_path, counted):
    from symtrs.model.shadow.mask_cache import cache_dir_for

    root, cache = _make_root(tmp_path), tmp_path / "cache"
    _stage(root, tmp_path / "run", cache, threshold=0.1)
    _stage(root, tmp_path / "run2", cache, threshold=0.2)
    assert len(counted) == 6, "a different threshold must not reuse the 0.1 masks"
    a = cache_dir_for(str(cache), str(root), "train", 0.1, 3)
    b = cache_dir_for(str(cache), str(root), "train", 0.2, 3)
    c = cache_dir_for(str(cache), str(root), "train", 0.1, 5)
    assert len({a, b, c}) == 3 and a.is_dir() and b.is_dir() and not c.exists()
    _stage(root, tmp_path / "run3", cache, threshold=0.1)  # the first settings are still valid
    assert len(counted) == 6


def test_rewriting_a_source_invalidates_only_that_pair(tmp_path, counted):
    import numpy as np
    from PIL import Image

    root, cache = _make_root(tmp_path), tmp_path / "cache"
    _stage(root, tmp_path / "run", cache)
    counted.clear()
    path = root / "train" / "rgb_day" / "rgb_day_2048_loc001.png"
    brighter = np.full((32, 32, 4), 201, np.uint8)  # also changes the size on disk
    Image.fromarray(brighter).save(path)
    staging = _stage(root, tmp_path / "run", cache)
    assert len(counted) == 1
    assert np.asarray(Image.open(staging / "mask" / "loc001.png")).max() == 0  # no shadow left
    assert np.asarray(Image.open(staging / "mask" / "loc000.png")).max() == 255


def test_touching_a_source_invalidates_it_by_mtime(tmp_path, counted):
    root, cache = _make_root(tmp_path), tmp_path / "cache"
    _stage(root, tmp_path / "run", cache)
    counted.clear()
    path = root / "train" / "rgb_noshadow" / "rgb_noshadow_2048_loc002.png"
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))
    _stage(root, tmp_path / "run", cache)
    assert len(counted) == 1


def test_counts_report_cached_computed_and_invalidated(tmp_path):
    from symtrs.model.shadow.mask_cache import ensure_masks
    from symtrs.model.shadow.staging import pairs_for

    root, cache = _make_root(tmp_path), tmp_path / "cache"
    pairs = pairs_for(str(root), "train")
    _, first = ensure_masks(pairs, str(root), "train", 0.1, 3, str(cache), 0)
    assert (first.cached, first.computed, first.invalidated) == (0, 3, 0)
    _, second = ensure_masks(pairs, str(root), "train", 0.1, 3, str(cache), 0)
    assert (second.cached, second.computed, second.invalidated) == (3, 0, 0)
    path = root / "train" / "rgb_day" / "rgb_day_2048_loc000.png"
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))
    _, third = ensure_masks(pairs, str(root), "train", 0.1, 3, str(cache), 0)
    assert (third.cached, third.computed, third.invalidated) == (2, 1, 1)


def test_nothing_is_written_under_the_data_root(tmp_path):
    root = _make_root(tmp_path, splits=("train", "val"))
    before = _snapshot(root)
    _stage(root, tmp_path / "run", tmp_path / "cache", "train")
    _stage(root, tmp_path / "run", tmp_path / "cache", "val", workers=2)
    assert _snapshot(root) == before


def test_a_cache_inside_the_data_root_is_refused(tmp_path):
    from symtrs.model.shadow.mask_cache import MaskCacheError

    root = _make_root(tmp_path)
    with pytest.raises(MaskCacheError, match="inside the data root.*shadow.mask_cache"):
        _stage(root, tmp_path / "run", root / ".cache")


def test_two_data_roots_do_not_share_entries(tmp_path, counted):
    from symtrs.model.shadow.mask_cache import root_id

    one = _make_root(tmp_path / "one")
    two = _make_root(tmp_path / "two")
    assert root_id(str(one)) != root_id(str(two))  # same folder name, different location
    cache = tmp_path / "cache"
    _stage(one, tmp_path / "run1", cache)
    _stage(two, tmp_path / "run2", cache)
    assert len(counted) == 6


def test_disabled_cache_keeps_the_per_run_behaviour(tmp_path, counted):
    import json

    root = _make_root(tmp_path)
    staging = _stage(root, tmp_path / "run", "")
    assert len(counted) == 3
    assert not (tmp_path / ".cache").exists()
    assert not any(p.is_symlink() for p in (staging / "mask").glob("*.png"))
    _stage(root, tmp_path / "run", "")  # reused through mask_params.json, as before
    assert len(counted) == 3
    assert json.loads((staging / "mask_params.json").read_text())["cache"] == ""
    _stage(root, tmp_path / "run", "", threshold=0.2)  # a changed setting re-derives
    assert len(counted) == 6


def test_turning_the_cache_off_never_writes_through_a_link_into_it(tmp_path):
    root, cache = _make_root(tmp_path), tmp_path / "cache"
    _stage(root, tmp_path / "run", cache)
    cached = sorted(cache.rglob("loc000.png"))[0]
    before = cached.read_bytes(), cached.stat().st_mtime_ns
    _stage(root, tmp_path / "run", "", threshold=0.1)  # same run dir, cache now off
    assert (cached.read_bytes(), cached.stat().st_mtime_ns) == before


def test_params_file_records_where_the_masks_came_from(tmp_path):
    import json

    from symtrs.model.shadow.mask_cache import cache_dir_for

    root, cache = _make_root(tmp_path), tmp_path / "cache"
    staging = _stage(root, tmp_path / "run", cache)
    params = json.loads((staging / "mask_params.json").read_text())
    assert params["threshold"] == 0.1 and params["open"] == 3
    assert params["cache"] == str(cache_dir_for(str(cache), str(root), "train", 0.1, 3))


def test_a_failed_write_leaves_no_partial_file_and_no_trusted_entry(tmp_path, monkeypatch):
    import symtrs.model.shadow.mask_cache as mask_cache
    from symtrs.model.shadow.staging import pairs_for

    root, cache = _make_root(tmp_path, n=1), tmp_path / "cache"
    pairs = pairs_for(str(root), "train")
    folder, _ = mask_cache.ensure_masks(pairs, str(root), "train", 0.1, 3, str(cache), 0)
    # Make the entry stale, then fail the rewrite between "temp file written" and "renamed".
    day = pairs[0][1]
    stat = day.stat()
    os.utime(day, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))
    monkeypatch.setattr(
        mask_cache.os, "replace", lambda *a: (_ for _ in ()).throw(OSError("disk full"))
    )
    with pytest.raises(OSError, match="disk full"):
        mask_cache.ensure_masks(pairs, str(root), "train", 0.1, 3, str(cache), 0)
    assert not list(folder.glob("*.tmp")) and not list(folder.glob(".*.tmp"))
    assert not (folder / "loc000.json").exists(), "a rewrite in flight must not stay trusted"
    monkeypatch.undo()
    _, counts = mask_cache.ensure_masks(pairs, str(root), "train", 0.1, 3, str(cache), 0)
    assert (counts.computed, counts.invalidated) == (1, 1)


def test_a_torn_png_without_its_sidecar_is_recomputed(tmp_path):
    from symtrs.model.shadow.mask_cache import ensure_masks
    from symtrs.model.shadow.staging import pairs_for

    root, cache = _make_root(tmp_path, n=1), tmp_path / "cache"
    pairs = pairs_for(str(root), "train")
    folder, _ = ensure_masks(pairs, str(root), "train", 0.1, 3, str(cache), 0)
    (folder / "loc000.json").unlink()
    (folder / "loc000.png").write_bytes(b"\x89PNG half")
    _, counts = ensure_masks(pairs, str(root), "train", 0.1, 3, str(cache), 0)
    assert counts.computed == 1
    from PIL import Image

    Image.open(folder / "loc000.png").load()


def test_parallel_and_serial_fills_are_identical(tmp_path):
    from symtrs.model.shadow.mask_cache import ensure_masks
    from symtrs.model.shadow.staging import pairs_for

    root = _make_root(tmp_path)
    pairs = pairs_for(str(root), "train")
    serial, _ = ensure_masks(pairs, str(root), "train", 0.1, 3, str(tmp_path / "serial"), 0)
    parallel, counts = ensure_masks(pairs, str(root), "train", 0.1, 3, str(tmp_path / "par"), 2)
    assert counts.computed == 3
    for key, _, _ in pairs:
        assert (serial / f"{key}.png").read_bytes() == (parallel / f"{key}.png").read_bytes()
        assert (parallel / f"{key}.json").is_file()


def test_misaligned_pair_names_the_key(tmp_path):
    import numpy as np
    from PIL import Image

    from symtrs.model.shadow.staging import ShadowStagingError

    root = _make_root(tmp_path, n=1)
    Image.fromarray(np.zeros((16, 16, 4), np.uint8)).save(
        root / "train" / "rgb_day" / "rgb_day_2048_loc000.png"
    )
    with pytest.raises(ShadowStagingError, match="loc000.*not aligned"):
        _stage(root, tmp_path / "run", tmp_path / "cache")


def test_config_defaults_and_anchoring(tmp_path, monkeypatch):
    from symtrs.config import ConfigError, RunConfig

    cfg = RunConfig(model="shadowformer", task="shadow_removal")
    assert cfg.shadow.mask_cache  # on by default
    assert 1 <= cfg.shadow.mask_workers <= 8
    monkeypatch.chdir(tmp_path)
    anchored = cfg.anchored()
    # The subprocess cwd is the upstream repo: a relative cache path would point into it.
    assert Path(anchored.shadow.mask_cache) == (tmp_path / cfg.shadow.mask_cache).resolve()
    cfg.shadow.mask_cache = ""
    assert cfg.anchored().shadow.mask_cache == ""
    cfg.shadow.mask_workers = -1
    with pytest.raises(ConfigError, match="mask_workers"):
        cfg.shadow.validate()


def test_default_cache_folder_is_gitignored():
    lines = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".cache" in lines
