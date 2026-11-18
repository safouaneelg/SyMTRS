"""`symtrs make-dummy` - a tiny synthetic dataset in the real layout.

    symtrs make-dummy --out /tmp/SyMTRS_dummy --size 256 --train 8 --val 2 --test 2
    symtrs train --task depth --model midas_small --resolution 256 \\
                 --data-root /tmp/SyMTRS_dummy --epochs 1 --batch-size 2

Produces exactly the folder structure the real dataset uses:

    <out>/<split>/rgb_day/  rgb_night/  rgb_noshadow/  depth/  semantic/
    <out>/<split>/rgb_day_lr/{x2,x4,x8}/  depth_lr/...  semantic_lr/...

`rgb_night` and `rgb_noshadow` are native-only, as in the real data. File names are the
real ones, `{rgb_day|rgb_night|rgb_noshadow|seg_day|depth_day}_{pixels}_{key}`; RGB
files are RGBA PNGs; masks hold `class_id * 63` with 5 classes; depth is in CENTIMETRES
with sky as a huge value. Every default in `symtrs/config.py` therefore works on the dummy
and on the real dataset alike - that equivalence is what a smoke run on the dummy is for.

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

#: The five SyMTRS classes, in label order, and the grey-level step the masks use.
CLASSES: Tuple[str, ...] = ("background", "building", "green", "road", "car")
LABEL_STEP = 63

#: The scene is built with six internal ids (terrain, road, building, vegetation, water,
#: vehicle); this table maps them to the dataset's five. Water has no class of its own in
#: the real labels and is background there too.
_INTERNAL_TO_CLASS = np.array([0, 3, 1, 2, 0, 4], dtype=np.uint8)

#: Depth is written in centimetres, as the real EXRs are, so the default
#: `data.depth_scale: 0.01` is right for both. The sky is a huge finite value, as in the
#: real renders, so `data.max_depth` is exercised too (NaN would hide that).
CM_PER_METRE = 100.0
SKY_CM = 1.0e7

#: Real file-name key, varied by scene index only in the location counter.
_KEY = "loc{index:03d}_rot152deg_h00_x-049013_y+035369_z+009230"

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


def _render(
    scene: Dict[str, np.ndarray],
    night: bool,
    generator: np.random.Generator,
    shadows: bool = True,
    noise: np.ndarray = None,
) -> np.ndarray:
    """RGBA for one illumination, from the same geometry.

    `shadows=False` is the `rgb_noshadow` twin of the day image: identical (same `noise`)
    except for the darkened strip cast beside each building.
    """
    size = scene["depth"].shape[0]
    classes = scene["classes"]
    depth = scene["depth"]

    if noise is None:
        noise = generator.normal(0.0, 3.0, (size, size, 3))
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
        if shadows:
            shift = max(2, size // 32)
            cast = np.roll(classes == 2, shift, axis=1) & (classes != 2)
            rgb[cast] *= 0.6

    rgb += noise
    rgba = np.empty((size, size, 4), dtype=np.uint8)
    rgba[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    rgba[..., 3] = 255  # the real PNGs are RGBA; the loaders must drop this channel
    return rgba


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
        # Non-finite values are kept as given: the reader and the metrics both treat them
        # as invalid, and writing 0 would make "invalid" and "at the camera"
        # indistinguishable. `_write_split` has already replaced the sky by SKY_CM.
        np.save(path.with_suffix(".npy"), array.astype(np.float32))
        return

    try:
        import OpenEXR

        header = OpenEXR.Header(array.shape[1], array.shape[0])
        # One float32 channel "Y", like the real files (it was UINT, which stored the
        # float bytes under the wrong type).
        header["channels"] = {
            "Y": OpenEXR.Imath.Channel(OpenEXR.Imath.PixelType(OpenEXR.Imath.PixelType.FLOAT))
        }
        filled = np.nan_to_num(array, nan=0.0).astype(np.float32)
        file = OpenEXR.OutputFile(str(path.with_suffix(".exr")), header)
        file.writePixels({"Y": filled.tobytes()})
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
        "rgb_noshadow": root / split / "rgb_noshadow",
        "depth": root / split / "depth",
        "semantic": root / split / "semantic",
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)

    if with_lr:
        for suffix in LR_SCALES:
            # rgb_night and rgb_noshadow have no downsampled twin in the real data.
            for name in ("rgb_day", "depth", "semantic"):
                (root / split / f"{name}_lr" / suffix).mkdir(parents=True, exist_ok=True)

    from PIL import Image

    def native(prefix: str, key: str, ext: str) -> str:
        return f"{prefix}_{size}_{key}{ext}"

    for index in range(count):
        scene = _scene(size, generator)
        key = _KEY.format(index=index)

        grain = generator.normal(0.0, 3.0, (size, size, 3))
        day = _render(scene, night=False, generator=generator, noise=grain)
        noshadow = _render(scene, night=False, generator=generator, shadows=False, noise=grain)
        night = _render(scene, night=True, generator=generator)
        labels = _INTERNAL_TO_CLASS[scene["classes"]] * np.uint8(LABEL_STEP)
        # Sky (NaN above the horizon) -> a huge finite value; metres -> centimetres.
        depth = np.where(np.isfinite(scene["depth"]), scene["depth"] * CM_PER_METRE, SKY_CM)
        depth = depth.astype(np.float32)

        Image.fromarray(day).save(directories["rgb_day"] / native("rgb_day", key, ".png"))
        Image.fromarray(night).save(directories["rgb_night"] / native("rgb_night", key, ".png"))
        Image.fromarray(noshadow).save(
            directories["rgb_noshadow"] / native("rgb_noshadow", key, ".png")
        )
        Image.fromarray(labels).save(directories["semantic"] / native("seg_day", key, ".png"))
        _write_depth(depth, directories["depth"] / native("depth_day", key, ".exr"), fmt)

        if not with_lr:
            continue
        for suffix, factor in LR_SCALES.items():
            target = size // factor
            if target < 16:
                continue
            Image.fromarray(_resize(day, target, nearest=False)).save(
                root / split / "rgb_day_lr" / suffix / f"rgb_day_{target}_{key}.png"
            )
            Image.fromarray(_resize(labels, target, nearest=True)).save(
                root / split / "semantic_lr" / suffix / f"seg_day_{target}_{key}.png"
            )
            # Picked, not averaged: averaging across the sky edge would invent depths.
            _write_depth(
                _resize(depth, target, nearest=True),
                root / split / "depth_lr" / suffix / f"depth_day_{target}_{key}.exr",
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
        "depth_units": "centimetres",
        "depth_invalid": "non-finite, <= 0, or sky (a huge value; set data.max_depth)",
        "label_step": LABEL_STEP,
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
