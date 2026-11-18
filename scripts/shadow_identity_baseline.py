"""Score the "do nothing" shadow-removal baseline: the shadowed input as the prediction.

A shadow-removal PSNR means nothing on its own: on SyMTRS the shadows are large (about
40 % of the pixels under the derived mask) and deep, so the input itself already sits at a
fixed PSNR against `rgb_noshadow`. A model is only removing shadows if it beats this row.
It uses the same staging pairs, derived mask and metric code as the trained models
(`symtrs.model.shadow.staging.pairs_for`, `symtrs.metrics.shadow`), so the numbers are
directly comparable; it reads the dataset and writes one JSON file outside it.

    .venv/Scripts/python.exe scripts/shadow_identity_baseline.py --data-root "<root>" \
        [--out runs/validation/_baselines/shadow_identity.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from symtrs.config import RunConfig
from symtrs.metrics.shadow import aggregate, derive_mask, score_image
from symtrs.model.shadow.staging import pairs_for


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--out", default="runs/validation/_baselines/shadow_identity.json")
    args = ap.parse_args()

    # The mask parameters are the trained runs' defaults, read from the one place they live.
    shadow = RunConfig().shadow
    result = {"model": "identity (input as prediction)",
              "mask": {"threshold": shadow.mask_threshold, "open": shadow.mask_open}}
    for split in args.splits:
        rows = []
        for _key, day_path, clean_path in pairs_for(args.data_root, split):
            day = np.asarray(Image.open(day_path).convert("RGB"), dtype=np.float32) / 255.0
            clean = np.asarray(Image.open(clean_path).convert("RGB"), dtype=np.float32) / 255.0
            mask = derive_mask(day, clean, shadow.mask_threshold, shadow.mask_open)
            rows.append(score_image(day, clean, mask))
        result[split] = aggregate(rows)
        print(f"{split}: psnr {result[split]['psnr']:.3f}  psnr_shadow "
              f"{result[split]['psnr_shadow']:.3f}  psnr_nonshadow {result[split]['psnr_nonshadow']:.3f}"
              f"  ({int(result[split]['images'])} images)")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
