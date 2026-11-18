#!/usr/bin/env python3
"""Apply `eval.retain_predictions` to runs that finished before it existed.

The shadow and domain scorers write one lossless 2048 px PNG per image (a 3000-image split
is ~45 GB) and, until `eval.retain_predictions`, kept them after scoring. Every number they
exist for is in `per_image_<split>.json` by then, so this shrinks them to `RETAINED_SIZE`
copies, keeps the scenes `preview_keys` selects at full resolution (the ones figures zoom
into), and records what it did in the same `_retained.json` marker the scorer writes. It
also backfills `summary.json` with the distribution and that record, so a run scored before
the change is not poorer than one scored after it (additive keys only).

One implementation: the shrinking itself is `symtrs.model.shadow.runner.retain_predictions`,
the function scoring calls, so a run pruned here behaves exactly like a freshly scored one.

Deliberately conservative - it touches a split only when all of these hold, and otherwise
says why it skipped it:

  * `<run>/per_image_<split>.json` exists (the split was scored; deleting the images of an
    unscored split would throw away the only copy of the predictions),
  * every key in that JSON has its PNG (the directory is complete),
  * no `_retained.json` yet (nothing to do).

    # what it would do (default: dry run, changes nothing)
    python scripts/prune_predictions.py --runs runs/campaign2
    # do it
    python scripts/prune_predictions.py --runs runs/campaign2 --apply
    # remove the prediction images instead of shrinking them
    python scripts/prune_predictions.py --runs runs/campaign2 --mode none --apply

It does not touch `checkpoints/`, `weights/`, `staging/`, `results/` (upstream's own output
for the unpaired domain directions), `preview/` or anything a metric is read from.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_SCRIPTS = Path(__file__).resolve().parent
if str(REPO_SCRIPTS.parent) not in sys.path:
    sys.path.insert(0, str(REPO_SCRIPTS.parent))


def gib(size: int) -> str:
    return f"{size / 2**30:.2f} GB" if size >= 2**30 else f"{size / 2**20:.1f} MB"


def _run_dirs(root: Path):
    """Every `predictions/` under `root`, as the run directory that holds it."""
    seen = set()
    for predictions in sorted(root.rglob("predictions")):
        if not predictions.is_dir() or predictions.parent in seen:
            continue
        seen.add(predictions.parent)
        yield predictions.parent


def _splits(run_dir: Path):
    return sorted(p.name for p in (run_dir / "predictions").iterdir() if p.is_dir())


def _label(cfg, split: str) -> str:
    if split == cfg.data.test_split:
        return "test"
    if split == cfg.data.val_split:
        return "val"
    return split


def refresh_summary(cfg, run_dir: Path, split: str, rows) -> bool:
    """Add the distribution and the retention record to a `summary.json` written before
    `eval.retain_predictions` (and before `aggregate` reported std/median/percentiles).

    Only additive: the means already in it come from `aggregate` over these same rows, so
    they are unchanged, and the domain directions keep their FID/KID keys. The point is that
    the summary - not the images - is where a reader looks, so it should not be poorer than
    what a run scored today writes.
    """
    from symtrs.common.jsonio import dumps as json_dumps_safe
    from symtrs.metrics.shadow import aggregate
    from symtrs.model.shadow.runner import retention_record

    path = run_dir / "summary.json"
    if not path.is_file():
        return False
    summary = json.loads(path.read_text(encoding="utf-8"))
    label = _label(cfg, split)
    block = summary.get(label)
    if isinstance(block, dict):
        # `per_image_<split>.json` carries the scene `key` with each row; the scorer calls
        # `aggregate` on rows without it, so drop it here to mean exactly the same thing.
        block.update(aggregate([{k: v for k, v in row.items() if k != "key"} for row in rows]))
    summary["predictions"] = retention_record(cfg, run_dir)
    path.write_text(json_dumps_safe(summary, indent=2), encoding="utf-8")
    return True


def prune_split(cfg, run_dir: Path, split: str, mode: str, apply: bool) -> dict:
    from symtrs.model.shadow.runner import RETAIN_MARKER, preview_keys, retain_predictions

    predictions = run_dir / "predictions" / split
    rows_file = run_dir / f"per_image_{split}.json"
    if not rows_file.is_file():
        return {"split": split, "status": f"skipped: no {rows_file.name} (not scored)"}
    rows = json.loads(rows_file.read_text(encoding="utf-8"))
    if apply:
        cfg.eval.retain_predictions = mode
    if (predictions / RETAIN_MARKER).is_file():
        if apply:
            refresh_summary(cfg, run_dir, split, rows)
        return {"split": split, "status": "already retained" + (", summary refreshed" if apply else "")}
    keys = [row["key"] for row in rows]
    images = {key: predictions / f"{key}.png" for key in keys}
    missing = [key for key, path in images.items() if not path.is_file()]
    if missing:
        return {
            "split": split,
            "status": f"skipped: {len(missing)} of {len(keys)} predictions missing "
                      f"(first {missing[0]}) - not scoring-complete or mid-run",
        }

    before = sum(path.stat().st_size for path in images.values())
    scenes = preview_keys(rows, cfg.eval.num_visualizations)
    if apply:
        retain_predictions(cfg, run_dir, split, keys, scenes)
        # After the images moved, not before: the record names the scenes now at full
        # resolution under preview/full/.
        refresh_summary(cfg, run_dir, split, rows)
        after = sum(p.stat().st_size for p in predictions.glob("*.png"))
        kept = sum(p.stat().st_size for p in (run_dir / "preview" / "full").glob("*.png"))
        return {
            "split": split,
            "status": "done",
            "images": len(keys),
            "scenes_kept": len(scenes),
            "before": before,
            "after": after + kept,
        }
    # Dry run: the retained copy is RETAINED_SIZE^2 pixels instead of the full image, so the
    # estimate is by area. It is an estimate; --apply reports the real bytes.
    from symtrs.model.shadow.runner import RETAINED_SIZE

    scene_share = len(scenes) / max(1, len(keys))
    after = before * (1 - scene_share) * (RETAINED_SIZE / 2048) ** 2 if mode == "downsample" else 0
    return {
        "split": split,
        "status": "would prune",
        "images": len(keys),
        "scenes_kept": len(scenes),
        "before": before,
        "after": after,
    }


def main(argv=None) -> int:
    from symtrs.config import RETAIN_MODES, RunConfig

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", default="runs", help="a runs root to walk (default: runs)")
    parser.add_argument("--mode", choices=RETAIN_MODES, default="downsample")
    parser.add_argument("--apply", action="store_true", help="change files; default is a dry run")
    args = parser.parse_args(argv)

    root = Path(args.runs)
    if not root.is_dir():
        parser.error(f"{root} is not a directory")
    if args.mode == "all":
        parser.error("--mode all keeps everything; there is nothing to prune")

    total_before = total_after = 0
    changed = 0
    for run_dir in _run_dirs(root):
        config = run_dir / "config.yaml"
        if not config.is_file():
            print(f"{run_dir}: no config.yaml, skipping")
            continue
        cfg = RunConfig.from_yaml(str(config))
        for split in _splits(run_dir):
            result = prune_split(cfg, run_dir, split, args.mode, args.apply)
            line = f"{run_dir.name}/{split}: {result['status']}"
            if "before" in result:
                line += (
                    f" - {result['images']} images, {gib(result['before'])} -> ~{gib(int(result['after']))}"
                    f" ({result['scenes_kept']} scene(s) at full resolution)"
                )
                total_before += result["before"]
                total_after += result["after"]
                changed += 1
            print(line)

    verb = "freed" if args.apply else "would free"
    print(
        f"\n{changed} split(s) {'pruned' if args.apply else 'to prune'}: "
        f"{gib(total_before)} -> {gib(int(total_after))} ({verb} ~{gib(int(total_before - total_after))})"
    )
    if not args.apply:
        print("dry run: nothing changed. Re-run with --apply to do it.")
        print("re-scoring a pruned split re-runs prediction from the checkpoint "
              "(`symtrs eval <config> --eval-split <split>`).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
