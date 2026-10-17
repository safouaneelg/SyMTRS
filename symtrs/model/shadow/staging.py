"""Stage a split as `input/ target/ mask/` folders of per-pair files named by pairing key.

Shadow-removal upstreams disagree about everything on disk (ShadowFormer wants ISTD's
`train_A/train_B/train_C`, Uformer a `groundtruth/input` pair, Restormer a basicsr YAML),
and none of them knows the SyMTRS names `rgb_day_2048_<key>.png`. The worker that drives
all four (see `worker.py`) reads one layout, so this module is the single place that maps
the dataset onto it:

    <run_dir>/staging/<split>/input/<key>.png    link to rgb_day_2048_<key>.png
    <run_dir>/staging/<split>/target/<key>.png   link to rgb_noshadow_2048_<key>.png
    <run_dir>/staging/<split>/mask/<key>.png     derived, 8-bit, 255 = shadow

Links come from `symtrs.model.domain.upstream.link_file` (symlink, then hardlink, then
copy) - the same staging primitive domain adaptation uses. The dataset's PNGs are RGBA and
are converted to RGB by the reader, never rewritten here.

The mask does not exist in the dataset; `symtrs.metrics.shadow.derive_mask` derives it from
the pair. With `shadow.mask_cache` set it is derived once per (pair, settings) for every run
(`mask_cache.py` says why, and how a stale one is detected) and only linked here. Without it
the mask is written into the run and reused until `shadow.mask_threshold` / `shadow.mask_open`
change, which `mask_params.json` records, because a stale mask would silently keep the old
regional metrics.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from symtrs.model.domain.upstream import link_file

SUBDIRS = ("input", "target", "mask")


class ShadowStagingError(RuntimeError):
    """The dataset cannot be staged; the message names the folder and the command to check."""


def pairs_for(root: str, split: str) -> List[Tuple[str, Path, Path]]:
    """`(key, rgb_day, rgb_noshadow)` for every matched pair of a split, in key order."""
    from symtrs.data.io import list_images, match_by_stem, normalize_stem
    from symtrs.data.splits import resolve_split_dirs

    directories = resolve_split_dirs(root, split, 2048)
    for role in ("day", "noshadow"):
        folder = getattr(directories, role)
        if folder is None or not Path(folder).is_dir():
            raise ShadowStagingError(
                f"the {split} split has no {role} folder ({folder}). Shadow removal needs "
                f"rgb_day and rgb_noshadow at 2048; run `symtrs doctor --data-root {root}` "
                "for a coverage report."
            )
    day = list_images(directories.day)
    clean = list_images(directories.noshadow)
    matched = match_by_stem(day, clean)
    if not matched:
        raise ShadowStagingError(
            f"the {split} split has no rgb_day / rgb_noshadow pair with a matching name "
            f"({len(day)} day, {len(clean)} noshadow files); run "
            f"`symtrs doctor --data-root {root}` for a coverage report"
        )
    if len(matched) < max(len(day), len(clean)):
        print(
            f"[symtrs] WARNING: {split}: only {len(matched)} of {len(day)} day / "
            f"{len(clean)} noshadow images pair up; the others are not used"
        )
    return sorted((normalize_stem(a.stem), a, b) for a, b in matched)


def stage_split(
    root: str,
    run_dir: str,
    split: str,
    threshold: float,
    open_size: int,
    *,
    cache: str = "",
    workers: int = 0,
) -> Path:
    """Build `<run_dir>/staging/<split>` and return it. Idempotent.

    `cache` is `shadow.mask_cache`: with a folder, the masks come from the shared cache
    (`mask_cache.ensure_masks`) and are only linked here; with "" every run derives its own
    into `mask/` as it did before the cache existed. `workers` is `shadow.mask_workers`.
    The empty-string and zero here mean "feature off" for a caller that has no config; the
    defaults a run actually uses live in `symtrs.config.ShadowConfig`.
    """
    from symtrs.model.shadow import mask_cache

    pairs = pairs_for(root, split)
    staging = Path(run_dir) / "staging" / split
    for name in SUBDIRS:
        folder = staging / name
        if folder.is_symlink():
            folder.unlink()
        elif folder.is_dir() and name != "mask":
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)

    methods = set()
    for key, day, clean in pairs:
        methods.add(link_file(day.resolve(), staging / "input" / f"{key}.png"))
        methods.add(link_file(clean.resolve(), staging / "target" / f"{key}.png"))

    params = {
        "threshold": threshold,
        "open": open_size,
        "keys": [k for k, _, _ in pairs],
        # Where the masks came from, so the file stays truthful when a run dir is reused with
        # the cache turned on or off ("" = derived into this run).
        "cache": str(mask_cache.cache_dir_for(cache, root, split, threshold, open_size))
        if cache
        else "",
    }
    params_path = staging / "mask_params.json"
    mask_dir = staging / "mask"
    if cache:
        folder, counts = mask_cache.ensure_masks(
            pairs, root, split, threshold, open_size, cache, workers
        )
        print(
            f"[symtrs] shadow masks {split} (threshold={threshold}, open={open_size}): "
            f"{counts.cached} cached, {counts.computed} computed "
            f"({counts.invalidated} stale) <- {folder}"
        )
        for key, _, _ in pairs:
            methods.add(link_file((folder / f"{key}.png").resolve(), mask_dir / f"{key}.png"))
    else:
        cached = None
        if params_path.is_file():
            try:
                cached = json.loads(params_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                cached = None
        complete = all((mask_dir / f"{k}.png").is_file() for k, _, _ in pairs)
        if cached != params or not complete:
            print(
                f"[symtrs] deriving {len(pairs)} shadow masks for {split} "
                f"(threshold={threshold}, open={open_size})"
            )
            mask_cache.derive_into(pairs, mask_dir, threshold, open_size, workers)
    params_path.write_text(json.dumps(params), encoding="utf-8")

    print(
        f"[symtrs] staged {split}: {len(pairs)} pairs "
        f"({'/'.join(sorted(methods)) or 'empty'}) -> {staging}"
    )
    if methods - {"symlink"}:
        print(
            "[symtrs]   symlinks are unavailable here (on Windows: enable Developer Mode); "
            "fell back to hardlinks/copies, which behave the same for the reader"
        )
    return staging


def staged_keys(staging: Path) -> List[str]:
    """Keys present in all three folders."""
    names: Dict[str, set] = {
        name: {p.stem for p in (staging / name).glob("*.png")} for name in SUBDIRS
    }
    return sorted(set.intersection(*names.values()))


__all__ = ["SUBDIRS", "ShadowStagingError", "pairs_for", "stage_split", "staged_keys"]
