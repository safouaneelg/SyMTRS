"""A shared, validated cache of the derived shadow masks.

The dataset has no shadow masks; `symtrs.metrics.shadow.derive_mask` derives one from each
`rgb_day` / `rgb_noshadow` pair. That costs about 1 s per 2048 px pair (0.7 s decoding the two
PNGs, 0.3 s the derivation), and the result is a pure function of the two files and of
`shadow.mask_threshold` / `shadow.mask_open`. Staging used to re-derive it into every run's own
`<run_dir>/staging/<split>/mask/`, so at the planned 20,000+ views each of the four
shadow-removal models, and each shadow2noshadow translator that is scored like one, paid
~5.5 h single-threaded for a mask that was already on disk in the run before. This module
derives each mask once and lets every run link to it.

Layout, for `shadow.mask_cache = <cache>`:

    <cache>/<root name>-<hash of the resolved data root>/thr0.1_open3/<split>/<key>.png
                                                                              <key>.json

* The settings are part of the folder name, so a sensitivity run at another threshold gets its
  own masks and never overwrites (or reads) these.
* The sidecar records the resolved path, size and `mtime_ns` of both source images. A mask is
  valid only while both still match, so a regenerated dataset (the SyMTRS capture is being
  re-rendered) invalidates exactly the pairs that changed, with no global fingerprint to keep
  in step. Size + mtime is the same trade-off `make` and rsync accept; hashing 40 GB of PNGs
  on every start would cost more than the derivation it saves.
* The data-root hash keeps two datasets that share pair keys (`loc000` exists in both) from
  sharing a mask when the sidecar happens to match. The stats alone would invalidate them, but
  two datasets would then evict each other's entries on every alternating run.
* The cache lives outside the dataset and refuses to be placed inside it: the dataset is
  read-only input.

Writes are atomic (temp file in the same folder, then `os.replace`), and a sidecar is removed
before its mask is rewritten and written last, so a crash or two concurrent runs can leave a
missing entry but never a half-written PNG that later reads as valid.

Missing masks are computed in a process pool of bounded size (`shadow.mask_workers`): the
laptop this was developed on blue-screens under combined CPU and GPU load, so the default is
moderate and 0/1 means serial.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Pairs are `(key, rgb_day, rgb_noshadow)` as returned by `staging.pairs_for`.
Pair = Tuple[str, Path, Path]


class MaskCacheError(RuntimeError):
    """The cache cannot be used as configured; the message names the setting to change."""


@dataclass
class MaskCounts:
    """What one fill did. `invalidated` is the part of `computed` that replaced a stale entry."""

    cached: int = 0
    computed: int = 0
    invalidated: int = 0

    def __iadd__(self, other: "MaskCounts") -> "MaskCounts":
        self.cached += other.cached
        self.computed += other.computed
        self.invalidated += other.invalidated
        return self


@dataclass(frozen=True)
class _Job:
    """One mask to derive. Top-level and made of plain values so it pickles under Windows spawn."""

    key: str
    day: str
    clean: str
    mask_path: str
    sidecar_path: Optional[str]  # None when writing straight into a run's staging
    sidecar: Optional[Dict[str, Any]]
    threshold: float
    open_size: int


# ------------------------------------------------------------------------------ paths


def settings_name(threshold: float, open_size: int) -> str:
    return f"thr{threshold!r}_open{open_size}"


def root_id(root: str) -> str:
    """`<folder name>-<8 hex>` of the resolved data root; readable, and unique per location."""
    resolved = Path(root).resolve()
    digest = hashlib.sha1(os.path.normcase(str(resolved)).encode("utf-8")).hexdigest()[:8]
    name = re.sub(r"[^A-Za-z0-9._-]", "_", resolved.name) or "root"
    return f"{name}-{digest}"


def cache_dir_for(cache: str, root: str, split: str, threshold: float, open_size: int) -> Path:
    return Path(cache) / root_id(root) / settings_name(threshold, open_size) / split


def _check_outside_dataset(cache: str, root: str) -> None:
    cache_path, root_path = Path(cache).resolve(), Path(root).resolve()
    if cache_path == root_path or root_path in cache_path.parents:
        raise MaskCacheError(
            f"shadow.mask_cache ({cache_path}) is inside the data root ({root_path}); the "
            "dataset is read-only input. Point it elsewhere with "
            "`--set shadow.mask_cache=<folder>` (the default is .cache/shadow_masks)."
        )


# ----------------------------------------------------------------------------- writes


def _atomic_write(dest: Path, write) -> None:
    """`write(tmp_path)`, then rename over `dest`. Never leaves a partial `dest`.

    `os.replace` swaps the directory entry, so it also replaces (rather than follows) a
    symlink or breaks a hardlink at `dest`: a run's staged mask can never be written through
    into the shared cache.
    """
    tmp = dest.with_name(f".{dest.name}.{os.getpid()}.tmp")
    try:
        write(tmp)
        os.replace(tmp, dest)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def _derive_job(job: _Job) -> str:
    """Decode a pair, derive its mask and write it. The one implementation of that step."""
    import numpy as np
    from PIL import Image

    from symtrs.metrics.shadow import derive_mask
    from symtrs.model.shadow.staging import ShadowStagingError

    with Image.open(job.day) as image:
        a = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    with Image.open(job.clean) as image:
        b = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    if a.shape != b.shape:
        raise ShadowStagingError(
            f"{job.key}: rgb_day is {a.shape[1]}x{a.shape[0]} but rgb_noshadow is "
            f"{b.shape[1]}x{b.shape[0]}; the pair is not aligned"
        )
    mask = derive_mask(a, b, job.threshold, job.open_size)
    picture = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")

    mask_path = Path(job.mask_path)
    sidecar_path = Path(job.sidecar_path) if job.sidecar_path else None
    if sidecar_path is not None:
        # Invalidate first, validate last: between the two there is a mask nobody trusts.
        try:
            sidecar_path.unlink()
        except FileNotFoundError:
            pass
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(mask_path, lambda tmp: picture.save(tmp, format="PNG"))
    if sidecar_path is not None:
        _atomic_write(
            sidecar_path, lambda tmp: tmp.write_text(json.dumps(job.sidecar), encoding="utf-8")
        )
    return job.key


def run_jobs(jobs: Sequence[_Job], workers: int, label: str = "") -> None:
    """Run `jobs` serially, or in a pool of at most `workers` processes."""
    if not jobs:
        return
    step = max(1, len(jobs) // 10)
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
            done = pool.map(_derive_job, jobs, chunksize=1)
            for index, _ in enumerate(done, 1):
                if len(jobs) >= 20 and index % step == 0:
                    print(f"[symtrs]   {label} {index}/{len(jobs)}", flush=True)
    else:
        for index, job in enumerate(jobs, 1):
            _derive_job(job)
            if len(jobs) >= 20 and index % step == 0:
                print(f"[symtrs]   {label} {index}/{len(jobs)}", flush=True)


# ------------------------------------------------------------------------- the cache


def _stat(path: Path) -> Dict[str, Any]:
    resolved = Path(path).resolve()
    info = resolved.stat()
    return {"path": str(resolved), "size": info.st_size, "mtime_ns": info.st_mtime_ns}


def _sidecar(day: Path, clean: Path, threshold: float, open_size: int) -> Dict[str, Any]:
    return {
        "day": _stat(day),
        "noshadow": _stat(clean),
        "threshold": threshold,
        "open": open_size,
    }


def _entry_state(mask_path: Path, sidecar_path: Path, expected: Dict[str, Any]) -> str:
    """"valid", "stale" (something is there but cannot be trusted) or "absent"."""
    if not mask_path.is_file() and not sidecar_path.is_file():
        return "absent"
    if not (mask_path.is_file() and sidecar_path.is_file()):
        return "stale"
    try:
        recorded = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "stale"
    return "valid" if recorded == expected else "stale"


def ensure_masks(
    pairs: Iterable[Pair],
    root: str,
    split: str,
    threshold: float,
    open_size: int,
    cache: str,
    workers: int,
) -> Tuple[Path, MaskCounts]:
    """Make sure every pair has a valid cached mask; return the folder and what was done."""
    _check_outside_dataset(cache, root)
    folder = cache_dir_for(cache, root, split, threshold, open_size)
    folder.mkdir(parents=True, exist_ok=True)

    counts = MaskCounts()
    jobs: List[_Job] = []
    for key, day, clean in pairs:
        mask_path, sidecar_path = folder / f"{key}.png", folder / f"{key}.json"
        # Stats are taken here, before the images are read, so a file rewritten while it is
        # being decoded fails the comparison next time instead of being blessed.
        expected = _sidecar(day, clean, threshold, open_size)
        state = _entry_state(mask_path, sidecar_path, expected)
        if state == "valid":
            counts.cached += 1
            continue
        counts.computed += 1
        counts.invalidated += state == "stale"
        jobs.append(
            _Job(
                key, str(day), str(clean), str(mask_path), str(sidecar_path), expected,
                threshold, open_size,
            )
        )
    run_jobs(jobs, workers, label=f"{split} masks")
    return folder, counts


def derive_into(
    pairs: Iterable[Pair], folder: Path, threshold: float, open_size: int, workers: int
) -> None:
    """The cache-less path: derive every mask straight into `folder` (a run's staging)."""
    folder.mkdir(parents=True, exist_ok=True)
    jobs = [
        _Job(key, str(day), str(clean), str(folder / f"{key}.png"), None, None, threshold, open_size)
        for key, day, clean in pairs
    ]
    run_jobs(jobs, workers, label="masks")


def precompute(
    root: str,
    splits: Sequence[str],
    threshold: float,
    open_size: int,
    cache: str,
    workers: int,
) -> Dict[str, Any]:
    """Fill the cache for whole splits ahead of any run; the same function staging calls."""
    from symtrs.model.shadow.staging import pairs_for

    started = time.perf_counter()
    total = MaskCounts()
    per_split: Dict[str, Dict[str, int]] = {}
    for split in splits:
        _folder, counts = ensure_masks(
            pairs_for(root, split), root, split, threshold, open_size, cache, workers
        )
        per_split[split] = vars(counts).copy()
        total += counts
    return {
        "per_split": per_split,
        "total": vars(total).copy(),
        "elapsed_s": time.perf_counter() - started,
    }


__all__ = [
    "MaskCacheError",
    "MaskCounts",
    "cache_dir_for",
    "derive_into",
    "ensure_masks",
    "precompute",
    "root_id",
    "run_jobs",
    "settings_name",
]
