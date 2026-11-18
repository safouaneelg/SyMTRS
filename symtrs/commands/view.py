"""`symtrs view` - look at one transform of the dataset: RGB, no-shadow, night, mask, depth.

    symtrs view <any file of a transform> [--res 1024] [--save out.png]
    symtrs view --data-root /data/SyMTRS --split val --index 5

This is a port of the dataset generator's `view_dataset.py`, kept here so the people who
check the numbers can also check the pictures without a second repository. It reads the
files; it never trains, resizes or rewrites anything.

What it deliberately does not do: guess. A missing sibling is reported by name rather
than drawn as a blank panel, because an empty panel looks like a rendering fault in the
dataset. The one waiver, as in the original, is the resolution: `rgb_night` and
`rgb_noshadow` exist at 2048 only, so they are always taken from the native folder.

File formats: `rgb_*` PNG (RGBA on disk, converted to RGB), `seg_*` 8-bit grey with
pixel = class id x 63, `depth_*` single-channel EXR in centimetres (shown in metres).
Depth goes through `symtrs.data.exr`, so every backend that package supports works here.

Path logic is pure pathlib, and numpy / PIL / matplotlib are imported inside the
functions that draw, so the sibling resolution can be tested anywhere.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SPLITS = ("train", "val", "test")
RESOLUTIONS = (2048, 1024, 512, 256)

#: Gray level = class id * GRAY_STEP, so five classes fit 0..252 with room to spot a resize.
GRAY_STEP = 63
CLASS_NAMES = ("background", "building", "green", "road", "car")
CLASS_COLORS = ((0, 0, 0), (0, 0, 255), (0, 255, 0), (128, 128, 128), (255, 0, 0))

#: Depth EXRs are in centimetres; the picture and the printed statistics are in metres.
CM_TO_M = 0.01

_NAME = re.compile(
    r"^(?P<kind>rgb|seg|depth)_(?P<light>day|night|noshadow)_(?P<res>\d+)_(?P<key>.+)\.(?P<ext>png|exr)$"
)


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser("view", help="show RGB / night / mask / depth of one transform")
    parser.add_argument("file", nargs="?", default=None,
                        help="any rgb/seg/depth file of the transform (omit with --data-root)")
    parser.add_argument("--res", type=int, default=None, choices=RESOLUTIONS,
                        help="resolution to show (default: the given file's)")
    parser.add_argument("--data-root", default=None, help="dataset root, instead of a file path")
    parser.add_argument("--split", default="val", choices=SPLITS, help="split for --data-root (default val)")
    parser.add_argument("--index", type=int, default=0, help="which transform of the split (sorted order)")
    parser.add_argument("--save", default=None, metavar="PNG",
                        help="write the figure here and do not open a window")
    parser.set_defaults(handler=_run)


# ------------------------------------------------------------------ path logic


def parse_name(path) -> Dict[str, str]:
    """Split `rgb_day_2048_<key>.png` into kind / light / res / key / ext."""
    match = _NAME.match(Path(path).name)
    if not match:
        raise ValueError(
            f"{Path(path).name!r} does not look like a SyMTRS file "
            "(expected {rgb|seg|depth}_{day|night|noshadow}_<res>_<key>.{png|exr})"
        )
    return match.groupdict()


def split_root(path) -> Path:
    """The `<root>/<split>` folder a file lives under, walking up through `_lr/xN`."""
    path = Path(path).resolve()
    for parent in path.parents:
        if parent.name in SPLITS:
            return parent
    raise ValueError(
        f"{path} is not under a train/, val/ or test/ folder; point at a file inside the "
        "dataset layout, or use --data-root"
    )


def modality_dir(kind: str, light: str, res: int) -> str:
    """Folder (relative to the split) holding `kind` at `res`: `semantic_lr/x4` and so on."""
    base = {"seg": "semantic", "depth": "depth"}.get(kind, f"{kind}_{light}")
    return base if res >= 2048 else f"{base}_lr/x{2048 // res}"


def sibling(path, kind: str, light: str, res: int) -> Path:
    """Path of the `kind`/`light` file at `res` that shares `path`'s transform key."""
    info = parse_name(path)
    ext = "exr" if kind == "depth" else "png"
    name = f"{kind}_{light}_{res}_{info['key']}.{ext}"
    return split_root(path) / modality_dir(kind, light, res) / name


def list_transforms(root, split: str) -> List[Path]:
    """Native `rgb_day` files of a split, sorted - the index `--index` counts into."""
    folder = Path(root) / split / "rgb_day"
    if not folder.is_dir():
        raise FileNotFoundError(
            f"{folder} does not exist; check --data-root (run `symtrs doctor --data-root {root}`)"
        )
    files = sorted(p for p in folder.iterdir() if _NAME.match(p.name))
    if not files:
        raise FileNotFoundError(f"no rgb_day files in {folder}")
    return files


def plan(path, res: Optional[int] = None) -> List[Tuple[str, str, Path]]:
    """(panel title, kind, path) in display order. Existence is not checked here."""
    res = res or int(parse_name(path)["res"])
    return [
        ("RGB day", "rgb", sibling(path, "rgb", "day", res)),
        ("RGB day, no shadows (2048)", "rgb", sibling(path, "rgb", "noshadow", 2048)),
        ("RGB night (2048)", "rgb", sibling(path, "rgb", "night", 2048)),
        ("Semantic mask", "seg", sibling(path, "seg", "day", res)),
        ("Depth (m)", "depth", sibling(path, "depth", "day", res)),
    ]


# ---------------------------------------------------------------------- pixels


def seg_ids(gray):
    """Grey mask -> (class ids, grey values that are not a class; a resize leaves some)."""
    import numpy as np

    gray = np.asarray(gray)
    ids = np.rint(gray.astype(np.float32) / GRAY_STEP).astype(np.uint8)
    unexpected = np.setdiff1d(np.unique(gray), np.arange(len(CLASS_NAMES)) * GRAY_STEP)
    return ids, unexpected


def colorize(ids):
    """Class ids -> HxWx3 uint8; an id past the last class is clipped, not dropped."""
    import numpy as np

    table = np.array(CLASS_COLORS, dtype=np.uint8)
    return table[np.clip(ids, 0, len(CLASS_COLORS) - 1)]


def class_percentages(ids) -> List[float]:
    import numpy as np

    counts = np.bincount(ids.ravel(), minlength=len(CLASS_NAMES))[: len(CLASS_NAMES)]
    return list(counts / ids.size * 100)


def _read_rgb(path):
    import numpy as np
    from PIL import Image

    with Image.open(path) as im:
        return np.array(im.convert("RGB"))  # on-disk PNGs are RGBA


def _read_depth_m(path):
    """Depth in metres with invalid pixels as NaN, via the package's EXR reader."""
    import numpy as np

    from symtrs.data.exr import depth_to_tensor, read_depth_file

    metres, valid = depth_to_tensor(read_depth_file(path), scale=CM_TO_M)
    return np.where(valid, metres, np.nan)


# ------------------------------------------------------------------------ figure


def build_figure(path, res: Optional[int] = None, say=print):
    """Draw the transform of `path`. Raises FileNotFoundError listing every missing file."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch
    from PIL import Image

    steps = plan(path, res)
    shown_res = res or int(parse_name(path)["res"])
    missing = [str(p) for _, _, p in steps if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"missing files for this transform at {shown_res}: " + "; ".join(missing)
            + ". Run `symtrs doctor --data-root <root>` for per-folder coverage."
        )

    panels = []
    for title, kind, p in steps:
        if kind == "rgb":
            panels.append((title, _read_rgb(p), None))
        elif kind == "seg":
            with Image.open(p) as im:
                ids, odd = seg_ids(np.array(im))
            if odd.size:
                say(f"warning: {p.name} has non-class grey values {odd[:10].tolist()} (resized or converted?)")
            say("classes: " + ", ".join(f"{n} {c:.1f}%" for n, c in zip(CLASS_NAMES, class_percentages(ids))))
            panels.append((title, colorize(ids), "seg"))
        else:
            d = _read_depth_m(p)
            if np.isfinite(d).any():
                say(f"depth: min {np.nanmin(d):.2f} m, max {np.nanmax(d):.2f} m, median {np.nanmedian(d):.2f} m")
            else:
                say("depth: no valid pixels")
            panels.append((title, d, "depth"))

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5.5))
    for ax, (title, img, kind) in zip(np.atleast_1d(axes), panels):
        ax.set_title(title)
        ax.axis("off")
        if kind == "depth":
            # 1-99 percentile so sky and outliers do not flatten the colours; near = red.
            finite = img[np.isfinite(img)]
            lo, hi = np.percentile(finite, [1, 99]) if finite.size else (0.0, 1.0)
            im = ax.imshow(img, cmap="turbo_r", vmin=lo, vmax=hi)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="m")
        else:
            ax.imshow(img)
        if kind == "seg":
            ax.legend(handles=[Patch(color=np.array(c) / 255.0, label=n)
                               for n, c in zip(CLASS_NAMES, CLASS_COLORS)],
                      loc="lower left", fontsize=7, framealpha=0.8)
    fig.suptitle(f"{parse_name(path)['key']}  ({shown_res})", fontsize=9)
    fig.tight_layout()
    return fig


def _run(args: argparse.Namespace) -> int:
    if args.file:
        target = Path(args.file)
    elif args.data_root:
        files = list_transforms(args.data_root, args.split)
        if not 0 <= args.index < len(files):
            raise ValueError(f"--index {args.index} out of range: {args.split} has {len(files)} transforms")
        target = files[args.index]
    else:
        raise ValueError("give a file of a transform, or --data-root with --split/--index")
    print(f"[symtrs] view: {target}")

    if args.save:
        import matplotlib

        matplotlib.use("Agg")  # no window, so it works over ssh and in CI
    import matplotlib.pyplot as plt

    fig = build_figure(target, args.res)
    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=110)
        print(f"[symtrs] saved {out}")
    else:
        plt.show()
    plt.close(fig)
    return 0


__all__ = ["add_parser", "sibling", "plan", "list_transforms", "seg_ids", "colorize", "build_figure"]
