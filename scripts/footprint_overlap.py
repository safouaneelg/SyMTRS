#!/usr/bin/env python3
"""Ground overlap between views: the quantitative half of the split argument.

Reviewer R1.2 asks whether the retained views are diverse or redundant, and R1.3 whether
overlapping footprints cross the split. Both are answerable from the file names, because a
name carries the camera position, the altitude and the yaw, and those fix the square a nadir
view covers:

    half side w(h) = h * tan(hfov / 2),   hfov = 60 deg  ->  40.4, 57.7, 75.0, 92.4 m
                                                             at 70, 100, 130, 160 m

A view is a square of half side w(h), rotated about its centre by the yaw in its name, so the
footprint of a view is a convex quadrilateral and the intersection of two is another one.
The overlap is computed exactly (Sutherland--Hodgman clipping), never with bounding circles:
the difference matters, because the closest test and training locations are 247 m apart and
two squares of half side 92 m would still overlap at that distance if they were aligned, so a
distance argument alone would not settle R1.3. Nothing here opens an image.

Three numbers come out, and each answers a different question:

  * ``test vs train`` --- the leakage R1.3 asks about. Exact, per test view;
  * ``val vs train`` --- the same for the split that selects checkpoints;
  * ``<split> vs nearest other location`` --- the redundancy R1.2 asks about: how much of a
    view's ground is also covered by a view of a *different* location at the same altitude.
    The ten rotations and four altitudes of one location share that ground by construction,
    which the script reports separately rather than mixing it into the distribution.

    python scripts/footprint_overlap.py --data-root /mnt/ssd1/SyMTRS

Reads `<root>/<split>/rgb_day`; writes `runs/campaign2/dataset_overlap.json`.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

NAME = re.compile(
    r"^rgb_day_(?P<resolution>\d+)_loc(?P<location>\d+)_rot(?P<rotation>\d+)deg_"
    r"h(?P<altitude>\d+)_x(?P<x>[+-]\d+)_y(?P<y>[+-]\d+)_z(?P<z>[+-]\d+)$"
)
SPLITS = ("train", "val", "test")
HFOV_DEG = 60.0
HEIGHTS_M = (70.0, 100.0, 130.0, 160.0)
OUT = Path("runs/campaign2/dataset_overlap.json")


def footprint(view: dict) -> np.ndarray:
    """The four corners of a view's ground square, in metres, anticlockwise."""
    half = HEIGHTS_M[view["altitude"]] * math.tan(math.radians(HFOV_DEG / 2.0))
    angle = math.radians(view["rotation"])
    local = np.array([[half, half], [-half, half], [-half, -half], [half, -half]])
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    return local @ rotation.T + np.array([view["x"], view["y"]])


def _clip(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Sutherland--Hodgman clipping of a convex `subject` by a convex `clip`, both anticlockwise."""
    output = [tuple(point) for point in subject]
    for i in range(len(clip)):
        if not output:
            return np.empty((0, 2))
        a, b = clip[i], clip[(i + 1) % len(clip)]
        edge = b - a
        inside = [edge[0] * (p[1] - a[1]) - edge[1] * (p[0] - a[0]) >= 0.0 for p in output]
        clipped = []
        for j, p in enumerate(output):
            q = output[(j + 1) % len(output)]
            enters, leaves = inside[j], inside[(j + 1) % len(output)]
            if enters and leaves:
                clipped.append(q)
            elif enters != leaves:
                dp = edge[0] * (p[1] - a[1]) - edge[1] * (p[0] - a[0])
                dq = edge[0] * (q[1] - a[1]) - edge[1] * (q[0] - a[0])
                t = dp / (dp - dq)
                clipped.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
                if not enters:
                    clipped.append(q)
        output = clipped
    return np.array(output, dtype=float)


def area(polygon: np.ndarray) -> float:
    if len(polygon) < 3:
        return 0.0
    x, y = polygon[:, 0], polygon[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def iou(a: np.ndarray, b: np.ndarray, area_a: float, area_b: float) -> float:
    intersection = area(_clip(a, b))
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def views(root: Path, split: str) -> list[dict]:
    directory = root / split / "rgb_day"
    if not directory.is_dir():
        raise SystemExit(f"{directory} is not a directory; is {root} a SyMTRS root?")
    found = []
    for path in directory.iterdir():
        match = NAME.match(path.stem)
        if match is None:
            continue
        found.append(
            {
                "location": int(match["location"]),
                "rotation": int(match["rotation"]),
                "altitude": int(match["altitude"]),
                "x": int(match["x"]) / 100.0,
                "y": int(match["y"]) / 100.0,
            }
        )
    for view in found:
        view["corners"] = footprint(view)
        view["area"] = area(view["corners"])
    return found


def best_overlap(source: list[dict], target: list[dict], same_location: bool | None = None) -> np.ndarray:
    """For each view of `source`, the largest IoU with a view of `target`.

    `same_location=True` keeps only pairs of one location, `False` only pairs of two, and
    `None` keeps every pair (a view then meets itself and every overlap is 1.0 by definition,
    which is why the caller never asks for that).
    """
    from scipy.spatial import cKDTree

    coordinates = np.array([[view["x"], view["y"]] for view in target])
    tree = cKDTree(coordinates)
    # Two squares overlap only if their centres are closer than the sum of their circumradii,
    # w * sqrt(2) each; anything farther is skipped without clipping.
    radius = 2.0 * HEIGHTS_M[-1] * math.tan(math.radians(HFOV_DEG / 2.0)) * math.sqrt(2.0)
    best = np.zeros(len(source))
    for index, view in enumerate(source):
        candidates = tree.query_ball_point((view["x"], view["y"]), radius)
        for other in candidates:
            partner = target[other]
            if same_location is not None and (view["location"] == partner["location"]) != same_location:
                continue
            best[index] = max(best[index], iou(view["corners"], partner["corners"], view["area"], partner["area"]))
    return best


def summarise(name: str, values: np.ndarray) -> dict:
    return {
        "comparison": name,
        "views": int(len(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
        "fraction_above_0.01": float((values > 0.01).mean()),
        "fraction_above_0.25": float((values > 0.25).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/ssd1/SyMTRS"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    data = {split: views(args.data_root, split) for split in SPLITS}
    rows = []
    for split in ("val", "test"):
        rows.append(summarise(f"{split} vs train", best_overlap(data[split], data["train"])))
    for split in SPLITS:
        rows.append(
            summarise(f"{split}, nearest other location", best_overlap(data[split], data[split], same_location=False))
        )

    for row in rows:
        print(
            f"{row['comparison']:34s} n={row['views']:6d}  median {row['median']:.3f}  "
            f"p95 {row['p95']:.3f}  max {row['max']:.3f}  "
            f">0.01 {100 * row['fraction_above_0.01']:5.1f}%  >0.25 {100 * row['fraction_above_0.25']:5.1f}%"
        )
    for split in SPLITS:
        locations = {view["location"] for view in data[split]}
        print(f"{split:5s}: {len(data[split])} views, {len(locations)} locations, "
              f"{len({v['rotation'] for v in data[split]})} rotations")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
