"""Preview the derived shadow mask of one pair beside its `rgb_day` and `rgb_noshadow`.

The mask the shadow-removal task trains and scores with does not exist in the dataset; it is
derived from the pair (`symtrs.metrics.shadow.derive_mask`: Rec. 709 luma of both renders,
relative darkening `(Y_noshadow - Y_day) / Y_noshadow`, threshold, 3x3 opening). This script
calls that same function with the same settings as a run (`RunConfig().shadow`), so what it
shows is what the models get. Five panels: the two renders, the darkening map with the
threshold marked on its colour bar, the mask, and the mask outlined in red on `rgb_day`.

It only reads the dataset; the figure is shown on screen, or written with `--save`:

    .venv/Scripts/python.exe scripts/preview_shadow_mask.py --data-root "<root>"
    .venv/Scripts/python.exe scripts/preview_shadow_mask.py --data-root "<root>" --split test \
        --key loc010_rot045deg_h00 --save shadow_preview.png
    .venv/Scripts/python.exe scripts/preview_shadow_mask.py --data-root "<root>" --list

`--key` matches a substring of the pairing key (the part of the file name after
`rgb_day_2048_`); `--index` picks by position instead. `--threshold` / `--open` override the
run defaults to see how sensitive the mask is.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from symtrs.config import RunConfig
from symtrs.metrics.shadow import derive_mask, luminance
from symtrs.model.shadow.staging import pairs_for


def read_rgb(path: Path) -> np.ndarray:
    # The dataset's PNGs are RGBA; the alpha channel carries nothing, as in the run's reader.
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def pick(pairs, key: str, index: int):
    if key:
        hits = [p for p in pairs if key in p[0]]
        if not hits:
            sys.exit(f"no pair key contains {key!r}; run with --list to see the keys")
        if len(hits) > 1:
            print(f"{len(hits)} keys contain {key!r}; showing the first ({hits[0][0]})")
        return hits[0]
    if not 0 <= index < len(pairs):
        sys.exit(f"--index must be in 0..{len(pairs) - 1}")
    return pairs[index]


def main() -> int:
    defaults = RunConfig().shadow
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split", default="train", choices=("train", "val", "test"))
    ap.add_argument("--key", default="", help="substring of the pairing key")
    ap.add_argument("--index", type=int, default=0, help="position in key order (default 0)")
    ap.add_argument("--threshold", type=float, default=defaults.mask_threshold)
    ap.add_argument("--open", type=int, default=defaults.mask_open, dest="open_size")
    ap.add_argument("--save", default="", help="write the figure here instead of showing it")
    ap.add_argument("--list", action="store_true", help="print the pairing keys and exit")
    args = ap.parse_args()

    pairs = pairs_for(args.data_root, args.split)
    if args.list:
        for i, (key, _, _) in enumerate(pairs):
            print(f"{i:4d}  {key}")
        return 0
    key, day_path, clean_path = pick(pairs, args.key, args.index)

    import matplotlib
    if args.save:
        matplotlib.use("Agg")  # headless: no window needed to write the file
    import matplotlib.pyplot as plt

    day, clean = read_rgb(day_path), read_rgb(clean_path)
    mask = derive_mask(day, clean, args.threshold, args.open_size)
    y_day, y_clean = luminance(day), luminance(clean)
    darkening = (y_clean - y_day) / np.maximum(y_clean, 1e-3)  # the quantity derive_mask thresholds

    overlay = day.copy()
    edge = mask ^ np.roll(mask, 1, 0) | mask ^ np.roll(mask, 1, 1)
    overlay[mask] = 0.6 * overlay[mask] + 0.4 * np.array([1.0, 0.0, 0.0])
    overlay[edge] = (1.0, 0.0, 0.0)

    fig, axes = plt.subplots(1, 5, figsize=(25, 5.6), constrained_layout=True)
    axes[0].imshow(day)
    axes[0].set_title("rgb_day (input, with shadows)")
    axes[1].imshow(clean)
    axes[1].set_title("rgb_noshadow (target)")
    im = axes[2].imshow(np.clip(darkening, -0.2, 1.0), cmap="magma", vmin=-0.2, vmax=1.0)
    axes[2].set_title("darkening (Y_noshadow - Y_day) / Y_noshadow")
    bar = fig.colorbar(im, ax=axes[2], fraction=0.046)
    bar.ax.axhline(args.threshold, color="cyan", linewidth=2)
    bar.ax.text(1.6, args.threshold, f" threshold {args.threshold:g}", color="cyan",
                va="center", transform=bar.ax.get_yaxis_transform())
    axes[3].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title(f"mask (> {args.threshold:g}, {args.open_size}x{args.open_size} opening): "
                      f"{mask.mean():.1%} shadow")
    axes[4].imshow(overlay)
    axes[4].set_title("mask on rgb_day")
    for ax in axes:
        ax.set_axis_off()
    band = ((darkening > 0.05) & (darkening < 0.3)).mean()
    fig.suptitle(f"{args.split} / {key}   ({band:.1%} of pixels in the ambiguous 5-30 % band)")

    if args.save:
        fig.savefig(args.save, dpi=110)
        print(f"wrote {args.save}")
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
