#!/usr/bin/env python3
"""Write the per-view metadata and the split manifests the release is documented to carry.

The dataset's filenames already encode everything a user needs to reason about a view---
modality, lighting state, resolution, location, yaw, altitude index and the engine position--
but a filename is not a manifest: a reader who wants "the 3,000 test views" or "every view
taken from 160 m" should not have to parse names, and a split that exists only as three
directories cannot be quoted, versioned or checked. Reviewer R1.12 asks for exactly this.

    metadata/splits/{train,val,test}.txt   one view key per line, sorted
    metadata/views.csv                    one row per view: split, location, rotation,
                                          altitude, engine x/y/z, and the ground sample
                                          distance at each of the four resolutions
    metadata/camera.yaml                  the pinhole model and its focal length per resolution
    metadata/classes.yaml                 the five classes and the grey value each is stored as

`camera.yaml` and `classes.yaml` are read out of `symtrs.config` rather than retyped, so the
release cannot disagree with the code that reads it. The four resolutions are the dataset's own
levels, not a downsample factor: each covers the same ground at a quarter of the pixels.

Nothing is inferred from the images themselves: every field is parsed out of the name, so the
file is a restatement of the release rather than a measurement of it. The one computed column
is the ground sample distance, which follows from the camera: a 60 degree horizontal field of
view over a 2048 px render at height `z` covers `2 z tan(30)` metres, and each coarser level
covers the same ground at half the pixels.

The script also reports, per split, which views are missing one of the fourteen files, because a
manifest that lists a view whose night render failed to export is worse than no manifest.

    python scripts/build_metadata.py --data-root /mnt/ssd1/SyMTRS
    python scripts/build_metadata.py --data-root /mnt/ssd1/SyMTRS --dry-run
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path

import yaml

from symtrs.config import DataConfig
# The grammar and the completeness set live in the package because the hub launcher needs
# the same parse to build its flat compatibility view. Re-exported under the old names so
# this script's readers (and its test) keep one import surface.
from symtrs.data.naming import EXPECTED_VIEW_FILES as EXPECTED
from symtrs.data.naming import MASK_ALIASES
from symtrs.data.naming import parse_view_name as parse_name

RESOLUTIONS = (2048, 1024, 512, 256)
CLASS_NAMES = ("background", "building", "green", "road", "car")

SPLITS = ("train", "val", "test")

DATA = DataConfig()


def gsd_cm(height_m: float, resolution: int) -> float:
    """Ground sample distance of a nadir view: the footprint's width over the pixel count."""
    width_m = 2.0 * height_m * math.tan(math.radians(DATA.camera_hfov_deg / 2.0))
    return 100.0 * width_m / resolution


def focal_px(resolution: int) -> float:
    """The pinhole focal length this field of view implies, as the manuscript quotes it."""
    return (resolution / 2.0) / math.tan(math.radians(DATA.camera_hfov_deg / 2.0))


def scan_split(split_root: Path) -> tuple[dict[str, dict], Counter]:
    """Every view of a split, plus which of the fourteen files each one is missing."""
    views: dict[str, dict] = {}
    present: dict[str, set] = {}
    for folder in sorted(p for p in split_root.iterdir() if p.is_dir()):
        for path in folder.rglob("*"):
            if not path.is_file():
                continue
            fields = parse_name(path.stem)
            if fields is None:
                continue
            key = fields["key"]
            views.setdefault(key, fields)
            modality = fields["modality"]
            modality = "seg" if modality in MASK_ALIASES else modality
            present.setdefault(key, set()).add((modality, fields["light"], fields["resolution"]))
    missing = Counter()
    for key, have in present.items():
        for expected in EXPECTED - have:
            missing[expected] += 1
    return views, missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/ssd1/SyMTRS"))
    parser.add_argument("--out", type=Path, default=None,
                        help="metadata directory (default: <data-root>/metadata)")
    parser.add_argument("--dry-run", action="store_true", help="report, write nothing")
    args = parser.parse_args()

    out = args.out or args.data_root / "metadata"
    all_rows: list[dict] = []
    for split in SPLITS:
        split_root = args.data_root / split
        if not split_root.is_dir():
            raise SystemExit(f"missing split directory {split_root}")
        views, missing = scan_split(split_root)
        if not views:
            raise SystemExit(f"no SyMTRS filenames under {split_root}; check --data-root")
        keys = sorted(views)
        altitudes = sorted({views[k]["altitude"] for k in keys})
        print(f"{split}: {len(keys)} views, {len({views[k]['location'] for k in keys})} locations, "
              f"altitude codes {', '.join(altitudes)}", flush=True)
        if missing:
            for (modality, light, resolution), count in sorted(missing.items(), key=lambda z: -z[1]):
                print(f"  missing {resolution:>4} px {light:<8} {modality:<5} in {count} views",
                      flush=True)
        for key in keys:
            fields = views[key]
            row = {
                "view_key": key,
                "split": split,
                "location": fields["location"],
                "rotation_deg": fields["rotation"],
                "altitude_code": fields["altitude"],
                "x_m": f"{fields['x']:.2f}",
                "y_m": f"{fields['y']:.2f}",
                "z_m": f"{fields['z']:.2f}",
            }
            for resolution in RESOLUTIONS:
                row[f"gsd_{resolution}_cm"] = f"{gsd_cm(fields['z'], resolution):.2f}"
            all_rows.append(row)
        if args.dry_run:
            continue
        (out / "splits").mkdir(parents=True, exist_ok=True)
        (out / "splits" / f"{split}.txt").write_text("\n".join(keys) + "\n")

    if args.dry_run:
        print(f"dry run: would write {out}/splits/*.txt and {out}/views.csv "
              f"({len(all_rows)} views)")
        return

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "views.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    camera = {
        "model": "pinhole",
        "horizontal_fov_deg": DATA.camera_hfov_deg,
        "principal_point": "image centre",
        "projection": "f = R / (2 tan(hfov / 2)); x = (u - R/2) z / f",
        "focal_length_px": {str(r): round(focal_px(r), 1) for r in RESOLUTIONS},
        "note": "the same footprint at every resolution; each level is its own render, "
                "not a downsample of 2048",
    }
    (out / "camera.yaml").write_text(yaml.safe_dump(camera, sort_keys=False))

    classes = {
        "num_classes": DATA.num_classes,
        "ignore_index": DATA.ignore_index,
        "label_step": DATA.label_step,
        "stored_as": "8-bit grey, class_id * label_step; any other value decodes to "
                     "ignore_index",
        "classes": {name: index * DATA.label_step for index, name in enumerate(CLASS_NAMES)},
    }
    (out / "classes.yaml").write_text(yaml.safe_dump(classes, sort_keys=False))
    print(f"wrote {out / 'splits'}/*.txt, {out / 'views.csv'} ({len(all_rows)} views), "
          f"{out / 'camera.yaml'}, {out / 'classes.yaml'}")


if __name__ == "__main__":
    main()
