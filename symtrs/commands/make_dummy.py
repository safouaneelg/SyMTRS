"""`symtrs make-dummy` - a tiny synthetic dataset in the real layout.

    symtrs make-dummy --out /tmp/SyMTRS_dummy --size 256 --train 8 --val 2 --test 2
    symtrs train --task depth --model midas_small --resolution 256 \\
                 --data-root /tmp/SyMTRS_dummy --epochs 1 --batch-size 2

Produces exactly the folder structure the real dataset uses:

    <out>/<split>/rgb_day/  rgb_night/  depth/  semantic/
    <out>/<split>/rgb_day_lr/{x2,x4,x8}/  depth_lr/...  semantic_lr/...

so that every code path - resolution selection, EXR reading, class remapping, the
trainer, the metrics, `symtrs tables` - can be exercised and debugged before the real
dataset exists. `--size` controls the native resolution; the real dataset is 2048, but
256 is what makes a smoke run take seconds instead of minutes.

The scene is procedural but *consistent*: one ground plane, boxes standing on it, and a
class map that agrees with the geometry. Depth, semantics and both illuminations all
describe the same scene, so a bug that mixes up a pairing between them shows up as a
metric that moves for the wrong reason.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

#: The six SyMTRS classes, in label order.
CLASSES: Tuple[str, ...] = ("terrain", "road", "building", "vegetation", "water", "vehicle")

#: Native resolution -> the scale factor of its `_lr` folders.
LR_SCALES = {"x2": 2, "x4": 4, "x8": 8}

#: Per-class base colour, used for both illuminations.
CLASS_COLORS = {
    0: (96, 108, 72),
    1: (82, 82, 86),
    2: (176, 170, 158),
    3: (58, 92, 50),
    4: (52, 84, 116),
    5: (150, 60, 54),
}


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser("make-dummy", help="write a small synthetic dataset in the real layout")
    parser.add_argument("--out", required=True, help="output dataset root")
    parser.add_argument("--size", type=int, default=256, help="native resolution (real dataset: 2048)")
    parser.add_argument("--train", type=int, default=8)
    parser.add_argument("--val", type=int, default=2)
    parser.add_argument("--test", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--format", choices=["npy", "exr"], default="npy",
                        help="depth format; npy needs nothing installed, exr needs OpenEXR")
    parser.add_argument("--no-lr", dest="no_lr", action="store_true",
                        help="skip the downsampled folders")
    parser.set_defaults(handler=_run)


def _scene(size: int, generator: np.random.Generator) -> Dict[str, np.ndarray]:
    """One random aerial-ish scene: depth in metres and a matching class map."""
    rows = np.arange(size, dtype=np.float32)[:, None] * np.ones((1, size), dtype=np.float32)
    columns = np.ones((size, 1), dtype=np.float32) * np.arange(size, dtype=np.float32)[None, :]

    # A camera looking down at a ground plane that tilts toward the top of the frame.
    horizon = size * (0.30 + 0.15 * generator.random())
    ground = np.full((size, size), np.nan, dtype=np.float32)
    below = rows > horizon
    # Depth grows as 1/(v - horizon), the perspective of a plane. Farther than the
    # horizon is sky and stays invalid, which is what gives the depth mask a real edge
    # to exercise rather than a fully valid image.
    ground[below] = 4.0 / np.maximum((rows[below] - horizon) / size, 1e-3)

    classes = np.zeros((size, size), dtype=np.uint8)
    depth = ground.copy()

    # A road band across the middle, plus one that runs toward the horizon.
    road_rows = slice(int(size * 0.52), int(size * 0.66))
    classes[road_rows, :] = 1
    road_column = int(generator.integers(size * 0.25, size * 0.75))
    classes[int(horizon) : size, max(0, road_column - size // 14) : road_column + size // 14] = 1

    # Buildings: axis-aligned footprints that occlude the ground behind them.
    for _ in range(int(generator.integers(4, 9))):
        height = int(generator.integers(size // 16, size // 6))
        width = int(generator.integers(size // 16, size // 5))
        top = int(generator.integers(0, max(1, size - height)))
        left = int(generator.integers(0, max(1, size - width)))
        box = (slice(top, top + height), slice(left, left + width))
        if not np.isfinite(depth[box]).any():
            continue
        base = np.nanmedian(depth[box])
        # The roof sits closer than the ground it stands on, and gets a slightly noisy
        # surface so a smoothness or gradient regulariser has something to act on.
        roof = base * float(generator.uniform(0.35, 0.75))
        depth[box] = roof * (1.0 + 0.02 * generator.standard_normal((height, width)))
        classes[box] = 2

    # Vegetation blobs and one water patch.
    for _ in range(int(generator.integers(2, 5))):
        radius = int(generator.integers(size // 24, size // 10))
        centre_row = int(generator.integers(size // 4, size))
        centre_column = int(generator.integers(radius, size - radius))
        patch = (
            slice(max(0, centre_row - radius), min(size, centre_row + radius)),
            slice(max(0, centre_column - radius), min(size, centre_column + radius)),
        )
        valid = np.isfinite(depth[patch])
        classes[patch][valid] = 3

    water_rows = slice(int(size * 0.78), int(size * 0.90))
    water_columns = slice(int(size * 0.05), int(size * 0.30))
    valid = np.isfinite(depth[water_rows, water_columns])
    classes[water_rows, water_columns][valid] = 4

    # Vehicles: small rectangles on the road band.
    for _ in range(int(generator.integers(1, 5))):
        height, width = max(2, size // 48), max(3, size // 32)
        top = int(generator.integers(road_rows.start, max(road_rows.start + 1, road_rows.stop - height)))
        left = int(generator.integers(0, max(1, size - width)))
        box = (slice(top, top + height), slice(left, left + width))
        if not np.isfinite(depth[box]).any():
            continue
        classes[box] = 5
        depth[box] = np.nanmedian(depth[box]) * 0.97

    return {"depth": depth, "classes": classes, "columns": columns}


def _render(scene: Dict[str, np.ndarray], night: bool, generator: np.random.Generator) -> np.ndarray:
    """RGB for one illumination, from the same geometry."""
    size = scene["depth"].shape[0]
    classes = scene["classes"]
    depth = scene["depth"]

    rgb = np.zeros((size, size, 3), dtype=np.float32)
    for label, colour in CLASS_COLORS.items():
        mask = classes == label
        if mask.any():
            rgb[mask] = np.array(colour, dtype=np.float32)

    # Distance haze. The real data is aerial, so distant ground is dimmer in daylight.
    finite = np.isfinite(depth) & (depth > 0)
    haze = np.zeros_like(depth)
    haze[finite] = 1.0 / (1.0 + 0.02 * depth[finite])

    if night:
        rgb *= 0.18
        rgb[..., 2] *= 1.35  # a cool cast, as sodium lighting gives way to moon/sky
        rgb[..., 0] *= 0.85
        # Warm window lights on buildings and headlights on vehicles.
        lights = ((classes == 2) | (classes == 5)) & (
            generator.random((size, size)) < 0.02
        )
        rgb[lights] = np.array([240.0, 190.0, 110.0])
        rgb *= 0.6 + 0.4 * haze[..., None]
    else:
        rgb *= 0.55 + 0.45 * haze[..., None]

    rgb += generator.normal(0.0, 3.0, rgb.shape)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _resize(array: np.ndarray, size: int, nearest: bool) -> np.ndarray:
    """Downsample by an integer factor, averaging or picking.

    Averaging is right for an image and wrong for a label map, so the two call sites say
    which they want rather than sharing one function that is wrong for one of them.
    """
    factor = array.shape[0] // size
    if factor <= 1:
        return array
    if nearest:
        return array[::factor, ::factor]
    trimmed = array[: size * factor, : size * factor]
    if trimmed.ndim == 3:
        return trimmed.reshape(size, factor, size, factor, -1).mean(axis=(1, 3)).astype(array.dtype)
    return trimmed.reshape(size, factor, size, factor).mean(axis=(1, 3)).astype(array.dtype)


def _write_depth(array: np.ndarray, path: Path, fmt: str) -> None:
    if fmt == "npy":
        # NaN is kept rather than zeroed: the reader and the metrics both treat a
        # non-finite depth as invalid, and writing 0 would make "invalid" and "at the
        # camera" indistinguishable.
        np.save(path.with_suffix(".npy"), array.astype(np.float32))
        return

    try:
        import OpenEXR

        header = OpenEXR.Header(array.shape[1], array.shape[0])
        header["channels"] = {"R": OpenEXR.Imath.Channel(OpenEXR.Imath.UINT)}
        filled = np.nan_to_num(array, nan=0.0).astype(np.float32)
        file = OpenEXR.OutputFile(str(path.with_suffix(".exr")), header)
        file.writePixels({"R": filled.tobytes()})
        file.close()
    except ImportError as exc:
        raise ImportError(
            "writing EXR needs OpenEXR: pip install -e '.[exr]'. Use --format npy to "
            "skip EXR entirely, which the reader supports too."
        ) from exc


def _write_split(
    root: Path,
    split: str,
    count: int,
    size: int,
    fmt: str,
    with_lr: bool,
    generator: np.random.Generator,
) -> None:
    directories = {
        "rgb_day": root / split / "rgb_day",
        "rgb_night": root / split / "rgb_night",
        "depth": root / split / "depth",
        "semantic": root / split / "semantic",
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)

    if with_lr:
        for suffix in LR_SCALES:
            for name in directories:
                (root / split / f"{name}_lr" / suffix).mkdir(parents=True, exist_ok=True)

    from PIL import Image

    for index in range(count):
        scene = _scene(size, generator)
        stem = f"{index:06d}"

        day = _render(scene, night=False, generator=generator)
        night = _render(scene, night=True, generator=generator)
        labels = scene["classes"]
        depth = scene["depth"]

        Image.fromarray(day).save(directories["rgb_day"] / f"{stem}.png")
        Image.fromarray(night).save(directories["rgb_night"] / f"{stem}.png")
        Image.fromarray(labels).save(directories["semantic"] / f"{stem}.png")
        _write_depth(depth, directories["depth"] / f"{stem}.exr", fmt)

        if not with_lr:
            continue
        for suffix, factor in LR_SCALES.items():
            target = size // factor
            if target < 16:
                continue
            Image.fromarray(_resize(day, target, nearest=False)).save(
                root / split / "rgb_day_lr" / suffix / f"{stem}.png"
            )
            Image.fromarray(_resize(night, target, nearest=False)).save(
                root / split / "rgb_night_lr" / suffix / f"{stem}.png"
            )
            Image.fromarray(_resize(labels, target, nearest=True)).save(
                root / split / "semantic_lr" / suffix / f"{stem}.png"
            )
            _write_depth(
                _resize(depth, target, nearest=False),
                root / split / "depth_lr" / suffix / f"{stem}.exr",
                fmt,
            )


def _run(args: argparse.Namespace) -> int:
    root = Path(args.out)
    generator = np.random.default_rng(args.seed)

    splits = {"train": args.train, "val": args.val, "test": args.test}
    for split, count in splits.items():
        if count <= 0:
            continue
        print(f"[symtrs] {split}: {count} scenes at {args.size}px")
        _write_split(root, split, count, args.size, args.format, not args.no_lr, generator)

    manifest = {
        "synthetic": True,
        "note": "generated by `symtrs make-dummy`; not the real SyMTRS dataset",
        "native_resolution": args.size,
        "real_dataset_native_resolution": 2048,
        "lr_scales": LR_SCALES,
        "depth_format": args.format,
        "depth_units": "metres",
        "depth_invalid": "NaN, and any value <= 0",
        "num_classes": len(CLASSES),
        "class_names": list(CLASSES),
        "splits": {name: count for name, count in splits.items() if count > 0},
    }
    (root / "dataset.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[symtrs] wrote {root}")
    print(f"[symtrs] try: symtrs train --task depth --model midas_small --resolution "
          f"{min(args.size, 256)} --data-root {root} --epochs 1 --limit-train-batches 2")
    return 0


__all__ = ["CLASSES", "add_parser"]
