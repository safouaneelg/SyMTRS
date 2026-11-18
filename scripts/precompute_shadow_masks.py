"""Build the derived shadow masks for whole splits once, before any run needs them.

Shadow-removal runs (and the shadow2noshadow scoring of CycleGAN / pix2pix) need a mask per
pair that the dataset does not ship. Deriving one costs ~1 s per 2048 px pair, so at 20,000+
views it is hours; this fills the shared cache (`shadow.mask_cache`) in a bounded process pool,
and every later run only links to it. It calls exactly the function staging calls
(`symtrs.model.shadow.mask_cache.precompute`), reads the dataset and never writes under it.
A second invocation computes nothing; a regenerated pair or another threshold is recomputed.

    .venv/Scripts/python.exe scripts/precompute_shadow_masks.py --data-root "<root>" \\
        --splits train,val,test [--workers 4] [--threshold 0.1] [--open 3] [--cache .cache/shadow_masks]

Threshold, opening, cache folder and workers default to `symtrs.config.ShadowConfig`.
"""

from __future__ import annotations

import argparse
import json
import sys

from symtrs.config import RunConfig
from symtrs.model.shadow.mask_cache import MaskCacheError, precompute


def main() -> int:
    shadow = RunConfig().shadow
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--splits", default="train,val,test", help="comma-separated (default: all)")
    ap.add_argument("--threshold", type=float, default=shadow.mask_threshold)
    ap.add_argument("--open", type=int, default=shadow.mask_open, dest="open_size")
    ap.add_argument("--cache", default=shadow.mask_cache)
    ap.add_argument("--workers", type=int, default=shadow.mask_workers,
                    help="processes filling the cache; 0 or 1 = serial")
    ap.add_argument("--json", default="", help="also write the counts to this file")
    args = ap.parse_args()

    if not args.cache:
        ap.error("--cache is empty: the point of this script is the shared cache "
                 "(default .cache/shadow_masks)")
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    try:
        result = precompute(
            args.data_root, splits, args.threshold, args.open_size, args.cache, args.workers
        )
    except MaskCacheError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for split, counts in result["per_split"].items():
        print(f"{split}: {counts['cached']} cached, {counts['computed']} computed "
              f"({counts['invalidated']} invalidated)")
    total = result["total"]
    print(f"total: {total['cached']} cached, {total['computed']} computed "
          f"({total['invalidated']} invalidated) in {result['elapsed_s']:.1f} s "
          f"[threshold={args.threshold}, open={args.open_size}, workers={args.workers}]")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
