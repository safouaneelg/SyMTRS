#!/usr/bin/env python3
"""Stage real-night and night->day-translated views as a dataset root the stock eval path reads.

Reviewer R2.3 asks whether image translation actually *adapts*: does a target-domain task get
better because of it? The four arms of that answer (REVIEWERS.md, R2.3) all score the same depth
ground truth, which is geometric and therefore identical for the day and the night render of a
view:

    (a) day-trained depth model on the real night render      -> this root, `<out>/night`
    (b) the same model on a night->day translation             -> this root, `<out>/night2day_*`
    (d) a depth model trained on real night renders            -> train in `<out>/night`

There is no input-modality knob in `RunConfig`, and the released layout has no
`rgb_night_lr`, so instead of adding one to the loader (pinned by tests, and a dataset
description the authors are keeping as written) these arms are served by a *staging root* in
the ordinary layout: `rgb_day_lr/x8/` holds the night render -- or a translation of it -- and
`depth_lr/x8/` is a symlink to the real ground truth. Nothing in `symtrs` knows the difference,
which is the point: the numbers come out of the same `symtrs eval` that produced every other
depth row, with the same alignment and the same metrics.

**Everything is at 256 px** because that is the resolution the translation networks work in:
upstream's `test.py` sets `load_size = crop_size = 256` and its `Resize` is what the staged
input goes through, so a panel is exactly the tensor the generator saw. Scoring a 256 px
translation against a 2048 px target would measure an upsampling, not a translation.

    # night renders at 256 for training the oracle, and for arm (a)
    python scripts/stage_night_adaptation.py --data-root /mnt/ssd1/SyMTRS \
        --out /mnt/ssd1/symtrs_night256 --runs runs/campaign2/cyclegan_night2day \
        runs/campaign2/pix2pix_night2day --workers 16

Writes `<out>/night/{train,val,test}/` and `<out>/night2day_<model>/test/`, both in the layout
`symtrs` reads. `--verify` re-reads the staged test set against the runs' own `real_A` panels,
which is the proof that arm (a) and arm (b) differ only by the translation.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

#: `role` -> the folder the layout expects. `day` is what `symtrs` reads as the input of every
#: depth run; the staged root puts night or translated pixels there on purpose.
DAY_PREFIX = "rgb_day"
DEPTH_SUFFIX = "x8"  # 2048 / 8 = 256


def _resize_job(job: tuple[str, str]) -> None:
    """One night render at 256, through upstream's own resampler (BICUBIC on PIL).

    `upstream/test.py` resizes with `transforms.Resize([load_size, load_size], BICUBIC)`, so
    using anything else here would make arm (a) differ from arm (b) by a resampling as well as
    by the translation. `--verify` measures that it does not.
    """
    import torchvision.transforms as T
    from torchvision.transforms import InterpolationMode

    source, target = job
    with Image.open(source) as image:
        T.Resize([256, 256], InterpolationMode.BICUBIC)(image.convert("RGB")).save(target)


def key_of(stem: str) -> str:
    """The scene id after the `rgb_night_2048_`-style prefix."""
    parts = stem.split("_")
    return "_".join(parts[3:])


def list_split(root: Path, split: str, role: str) -> list[Path]:
    folder = root / split / role
    if not folder.is_dir():
        raise SystemExit(f"missing {folder}; run `symtrs doctor --data-root {root}`")
    return sorted(p for p in folder.iterdir() if p.suffix.lower() == ".png")


def link_depth(root: Path, out: Path, split: str) -> None:
    """`depth_lr/x8` is the untouched ground truth: depth is geometry, not a lighting state."""
    target = out / split / "depth_lr" / DEPTH_SUFFIX
    source = root / split / "depth_lr" / DEPTH_SUFFIX
    if not source.is_dir():
        raise SystemExit(f"missing {source}; the staged root scores at 256 only")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.exists():
        target.unlink()
    os.symlink(source, target)


def build_night(args, splits: list[str]) -> None:
    root, out = args.data_root, args.out / "night"
    jobs = []
    for split in splits:
        images = list_split(root, split, "rgb_night")
        target_dir = out / split / "rgb_day_lr" / DEPTH_SUFFIX
        target_dir.mkdir(parents=True, exist_ok=True)
        for source in images:
            jobs.append((str(source), str(target_dir / f"{DAY_PREFIX}_256_{key_of(source.stem)}.png")))
        link_depth(root, out, split)
        print(f"night/{split}: {len(images)} views -> {target_dir}", flush=True)
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as pool:
            list(pool.map(_resize_job, jobs, chunksize=32))
    else:
        for job in jobs:
            _resize_job(job)


def images_dir(run: Path) -> Path:
    return run / "results" / f"symtrs_{run.name}" / "test_latest" / "images"


def build_translated(args) -> None:
    root = args.data_root
    for run in args.runs:
        panels = sorted(images_dir(run).glob("*_fake_B.png"))
        if not panels:
            raise SystemExit(f"no fake_B panels in {images_dir(run)}; run `symtrs eval {run}` first")
        target_dir = args.out / f"night2day_{run.name.split('_')[0]}" / "test" / "rgb_day_lr" / DEPTH_SUFFIX
        target_dir.mkdir(parents=True, exist_ok=True)
        for panel in panels:
            key = panel.name[: -len("_fake_B.png")]
            target = target_dir / f"{DAY_PREFIX}_256_{key}.png"
            if target.is_symlink() or target.exists():
                target.unlink()
            os.symlink(panel.resolve(), target)
        link_depth(root, args.out / f"night2day_{run.name.split('_')[0]}", "test")
        print(f"{target_dir.parent.parent.parent.name}: {len(panels)} translated views -> {target_dir}",
              flush=True)


def verify(args) -> int:
    """Staged night must be the very tensor the generators saw: compare against `real_A`."""
    import numpy as np

    failures = 0
    for run in args.runs:
        panels = sorted(images_dir(run).glob("*_real_A.png"))[:8]
        for panel in panels:
            key = panel.name[: -len("_real_A.png")]
            staged = args.out / "night" / "test" / "rgb_day_lr" / DEPTH_SUFFIX / f"{DAY_PREFIX}_256_{key}.png"
            with Image.open(panel) as reference, Image.open(staged) as candidate:
                difference = np.abs(np.asarray(reference.convert("RGB"), np.int16)
                                    - np.asarray(candidate.convert("RGB"), np.int16)).mean()
            # 2.5/255 and not 0: the two front-ends resample slightly differently (CycleGAN's
            # unaligned pipeline lands at ~0.3, pix2pix's aligned one at ~1.6). The tolerance
            # is set by what a *framing* mistake costs: a 4 px crop of the same scene already
            # moves the number to 4.8, a flip to 50+, so anything at 2.5 is the same view.
            status = "ok" if difference <= 2.5 else "DIFFERS"
            failures += difference > 2.5
            print(f"  {key}: |staged - real_A| = {difference:.2f}/255  {status}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/ssd1/SyMTRS"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="*", default=[],
                        help="night2day run directories whose fake_B panels become arm (b)")
    parser.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--verify", action="store_true",
                        help="compare the staged test night against the runs' real_A panels")
    args = parser.parse_args()

    if args.verify:
        raise SystemExit(verify(args) or 0)
    if not args.runs:
        raise SystemExit("pass --runs <night2day run dir ...>; both arms need their panels")
    build_night(args, args.splits)
    build_translated(args)


if __name__ == "__main__":
    main()
