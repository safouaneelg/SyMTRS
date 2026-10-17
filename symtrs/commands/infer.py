"""`symtrs infer` - run a model on images and write the result out.

    symtrs infer --task depth --model midas_dpt_hybrid --image scene.png --out preview.png
    symtrs infer --task segmentation --checkpoint runs/seg/best.pt --image scene.png --out labels.png
    symtrs infer --task depth --checkpoint best.pt --images-dir data/test/rgb_day --out-dir out/

The checkpoint argument is the same one `autoloader` takes, so anything that works here
also works from Python:

    from symtrs.model import autoloader
    model = autoloader("runs/depth/midas_dpt_hybrid/1024/weights/best.pt")
    depth = model.predict("scene.png")          # HxW float array, metres
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

#: Extensions accepted when scanning a directory.
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser("infer", help="run a model on one image or a directory")
    parser.add_argument("--checkpoint", default="",
                        help="a symtrs .pt, a run directory, or a registry key")
    parser.add_argument("--model", default=None, help="registry key, when there is no checkpoint")
    parser.add_argument("--task", default=None, choices=["depth", "segmentation"])
    parser.add_argument("--image", default="", help="a single image")
    parser.add_argument("--images-dir", dest="images_dir", default="",
                        help="a directory of images; every one is written out")
    parser.add_argument("--out", default="", help="output file for --image")
    parser.add_argument("--out-dir", dest="out_dir", default="predictions")
    parser.add_argument("--resolution", type=int, default=None,
                        help="recorded in the output filename; inference is at the input's own size")
    parser.add_argument("--tile-size", dest="tile_size", type=int, default=0,
                        help="sliding-window inference for large images")
    parser.add_argument("--tile-overlap", dest="tile_overlap", type=int, default=128)
    parser.add_argument("--align", default="",
                        help="depth alignment before saving: none | median | scale | scale_shift | "
                             "scale_shift_disparity")
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-npy", dest="save_npy", action="store_true",
                        help="also write the raw array next to the picture")
    parser.add_argument("--limit", type=int, default=0, help="stop after this many images")
    parser.set_defaults(handler=_run)


def _gather(args: argparse.Namespace) -> List[Path]:
    if args.image:
        path = Path(args.image)
        if not path.is_file():
            raise FileNotFoundError(f"no such image: {path}")
        return [path]
    if args.images_dir:
        directory = Path(args.images_dir)
        if not directory.is_dir():
            raise NotADirectoryError(f"no such directory: {directory}")
        found = sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        return found[: args.limit] if args.limit else found
    raise ValueError("pass --image or --images-dir")


def _run(args: argparse.Namespace) -> int:
    import numpy as np

    if args.task == "segmentation":
        print(
            "[symtrs] note: segmentation output is written as an 8-bit label map. If your "
            "class ids do not fit in 8 bits, use --save-npy."
        )
    if args.task == "depth":
        print(
            "[symtrs] note: depth is written as a 16-bit PNG in millimetres with a colour "
            "preview beside it. Use --save-npy for the raw metre values."
        )

    source = args.checkpoint or args.model
    if not source:
        raise ValueError("pass --checkpoint or --model")

    from symtrs.config import RESOLUTIONS, RunConfig
    from symtrs.model.wrapper import autoloader

    model = autoloader(source, device=args.device)
    if args.task and model.task and args.task != model.task:
        raise ValueError(f"{source!r} is a {model.task} model; --task says {args.task}")
    if args.align:
        model.net.align = args.align

    resolution = args.resolution or model.resolution or 0
    if resolution and resolution not in RESOLUTIONS:
        raise ValueError(f"resolution must be one of {RESOLUTIONS}; got {resolution}")

    images = _gather(args)
    if not images:
        print("[symtrs] no images found")
        return 0
    print(f"[symtrs] {model.describe()}")
    print(f"[symtrs] {len(images)} image(s), tile_size={args.tile_size}, overlap={args.tile_overlap}")

    out_dir = Path(args.out_dir)
    # Only made when something will actually land in it. `--out file.png` for a single
    # image is the common case, and creating the default `predictions/` anyway leaves an
    # empty directory in whatever the caller's working directory happens to be.
    into_out_dir = not (args.out and len(images) == 1)
    if into_out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for index, path in enumerate(images):
        prediction = model.predict(
            str(path), tile_size=args.tile_size, tile_overlap=args.tile_overlap
        )
        target = out_dir / f"{path.stem}.png" if into_out_dir else Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)

        if model.task == "segmentation":
            _save_labels(prediction, target, args.save_npy)
        else:
            _save_depth(prediction, target, args.save_npy)
        written.append(target)

        if (index + 1) % 25 == 0:
            print(f"[symtrs] {index + 1}/{len(images)}")
        if args.limit and index + 1 >= args.limit:
            break

    if len(written) == 1:
        print(f"[symtrs] wrote {written[0]}")
    else:
        print(f"[symtrs] wrote {len(written)} predictions to {written[0].parent}")
    return 0


def _save_labels(prediction, target: Path, save_npy: bool) -> None:
    import numpy as np
    from PIL import Image

    labels = prediction.astype(np.int64)
    target.parent.mkdir(parents=True, exist_ok=True)
    if labels.max(initial=0) > 255:
        np.save(target.with_suffix(".npy"), labels)
        print(f"[symtrs] {target.stem}: class ids exceed 255; wrote a .npy instead of a PNG")
    else:
        Image.fromarray(labels.astype(np.uint8)).save(target)
    if save_npy:
        np.save(target.with_suffix(".npy"), labels)


def _save_depth(prediction, target: Path, save_npy: bool) -> None:
    """Write depth twice: a millimetre PNG for viewing, a colour preview for reading.

    A raw 16-bit PNG of metres looks like noise, and a colour map loses the values, so
    both are written rather than making the caller choose.
    """
    import numpy as np
    from PIL import Image

    metres = np.nan_to_num(prediction.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    millimetres = np.clip(metres * 1000.0, 0, 65535).astype(np.uint16)

    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(millimetres, mode="I;16").save(target)

    finite = metres[metres > 0]
    if finite.size:
        vmin = float(finite.min())
        vmax = float(np.quantile(finite, 0.99))
    else:
        vmin, vmax = 0.0, 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-6

    # The colour map needs matplotlib. It is a reading aid written beside the file that
    # actually holds the numbers, so it must not be able to fail the command - and it goes
    # next to the millimetre PNG, not in `out_dir`, which `--out` may have bypassed.
    try:
        from symtrs.common.plots import colorize_depth

        rgb = colorize_depth(metres, vmin, vmax)
        Image.fromarray(rgb).save(target.with_name(f"{target.stem}_color.png"))
    except Exception as exc:  # noqa: BLE001 - a preview is not the output
        print(f"[symtrs] {target.stem}: colour map skipped ({type(exc).__name__}: {exc}); "
              "install matplotlib for a preview beside the PNG")

    if save_npy:
        np.save(target.with_suffix(".npy"), metres)
