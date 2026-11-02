#!/usr/bin/env python3
"""Zero-shot segmentation of *real* labelled aerial imagery with the campaign's checkpoints.

Reviewer R1.4 asks how the benchmark's models do on data they were not trained on. The only
real, labelled aerial imagery in hand is an ISPRS Potsdam tile (RGB + 2D labels, 5 cm/px, true
nadir), and its six label colours map onto four of SyMTRS's five classes. That mapping is the
only judgement in this file and it is stated at `POTSDAM` below. "Road" is Potsdam's
*impervious surfaces* (roads, sidewalks, paving, plazas), so its IoU is an upper bound on roads
proper; clutter is ignored rather than scored as background, because SyMTRS's class 0 is absent
from the capture and scoring Potsdam clutter against it would measure a class nothing saw.

**The crops are resized, not cropped smaller.** `resolution` in SyMTRS names a pixel size over
a fixed ground footprint, not the size of the scene: the `_lr/xN` levels are the same view
rendered at fewer pixels (AGENTS.md, ground rule 1). A zero-shot test therefore has to give
every checkpoint the *same ground area* at the pixel count it was trained on. Cropping 512 px
out of the source instead would hand the 512 model a 26 m view where the 2048 model sees 102 m,
and the comparison would measure scale, not the network.

Evaluation is the run's own protocol: `eval.tile_size` and `eval.tile_overlap` are read back
from each run's `config.yaml`, so a row is scored the way its checkpoint was scored, never with
a constant chosen here.

    python scripts/zeroshot_real_data.py --runs runs/campaign2 \
        --rgb top_potsdam_7_9_RGB.tif --label top_potsdam_7_9_label.tif \
        --out runs/campaign2/real_zeroshot --device cuda:0

Writes `<out>/zeroshot.json` (per model and resolution), `<out>/table.md` (the table the paper
prints) and `<out>/predictions/` for one crop per resolution, so a figure can be drawn without
re-running a model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # the tiles are 6000x6000, over PIL's decompression-bomb limit

#: SyMTRS's five classes, in the checkpoint's own index order.
NAMES = ["background", "building", "green", "road", "car"]

#: Potsdam's six 2D-label colours -> SyMTRS ids. `255` is `data.ignore_index`: clutter has no
#: counterpart in the capture, and a class the model never saw must not be scored as a mistake.
POTSDAM = {
    (255, 255, 255): 3,   # impervious surfaces -> road (an upper bound: includes paving, plazas)
    (0, 0, 255): 1,       # building
    (0, 255, 255): 2,     # low vegetation -> green
    (0, 255, 0): 2,       # tree -> green
    (255, 255, 0): 4,     # car
    (255, 0, 0): 255,     # clutter -> ignore
}

#: (model key, the run suffixes to look for). Every model the campaign trained, at every
#: resolution it supports, so the table says what transfers and not just what one cell does.
SEG_MODELS = ["unet_small", "deeplabv3_r50", "deeplabv3_r101", "segformer_b0", "segformer_b1",
              "segformer_b2"]
RESOLUTIONS = [256, 512, 1024, 2048]


def map_labels(rgb: np.ndarray) -> np.ndarray:
    """Potsdam RGB labels -> SyMTRS class ids, refusing a colour the mapping does not know."""
    out = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    known = np.zeros(rgb.shape[:2], dtype=bool)
    for colour, index in POTSDAM.items():
        hit = np.all(rgb == np.array(colour, dtype=np.uint8), axis=2)
        out[hit] = index
        known |= hit
    if not known.all():
        raise SystemExit(f"{int((~known).sum())} label pixels are not a known Potsdam colour; "
                         "add the colour to POTSDAM rather than letting it fall into 'ignore'")
    return out


def crop_origins(size: int, crop: int, grid: int) -> list[int]:
    """`grid` evenly spaced origins covering the tile with the fewest overlaps and no gap."""
    last = size - crop
    if grid == 1:
        return [0]
    return [round(i * last / (grid - 1)) for i in range(grid)]


def iou_rows(confusion: np.ndarray) -> list[tuple[str, float, int]]:
    """Per-class IoU over the classes with a non-empty union, as `symtrs` scores mIoU.

    Class 0 has no pixels on either side and drops out; counting it would divide a real
    number by five and make every row look worse by a factor the reader cannot see.
    """
    rows = []
    for k in range(len(NAMES)):
        tp = confusion[k, k]
        union = confusion[k, :].sum() + confusion[:, k].sum() - tp
        if union == 0:
            continue
        rows.append((NAMES[k], tp / union, int(confusion[k, :].sum())))
    return rows


def constant_baseline(truth: np.ndarray) -> dict[str, float]:
    """mIoU of "always predict class k", per class -- the floor a model has to beat.

    A model that scores 0.18 has to be read against this, not against zero: on Potsdam's class
    balance an all-green guess already collects most of one class's IoU for free.
    """
    out = {}
    for k in range(1, len(NAMES)):
        predicted = np.zeros_like(truth)
        predicted[...] = k
        valid = truth != 255
        rows = iou_rows_confusion(truth[valid], predicted[valid])
        out[NAMES[k]] = float(np.mean([iou for _, iou, _ in rows]))
    return out


def iou_rows_confusion(truth: np.ndarray, predicted: np.ndarray) -> list[tuple[str, float, int]]:
    confusion = np.zeros((len(NAMES), len(NAMES)), dtype=np.int64)
    np.add.at(confusion, (truth, predicted), 1)
    return iou_rows(confusion)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, default=Path("runs/campaign2"))
    parser.add_argument("--rgb", type=Path, default=Path("top_potsdam_7_9_RGB.tif"))
    parser.add_argument("--label", type=Path, default=Path("top_potsdam_7_9_label.tif"))
    parser.add_argument("--out", type=Path, default=Path("runs/campaign2/real_zeroshot"))
    parser.add_argument("--crop", type=int, default=2048,
                        help="native crop, in source pixels; the finest level's footprint")
    parser.add_argument("--grid", type=int, default=3, help="crops per side")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--models", nargs="*", default=SEG_MODELS)
    parser.add_argument("--resolutions", nargs="*", type=int, default=RESOLUTIONS)
    args = parser.parse_args()

    import yaml

    from symtrs.model.wrapper import autoloader

    rgb = np.asarray(Image.open(args.rgb).convert("RGB"))
    truth_all = map_labels(np.asarray(Image.open(args.label).convert("RGB")))
    origins = crop_origins(rgb.shape[0], args.crop, args.grid)
    print(f"tile {rgb.shape[1]}x{rgb.shape[0]}, {len(origins)**2} crops of {args.crop} px "
          f"({args.crop * 0.05:.0f} m at Potsdam's 5 cm/px)", flush=True)
    valid_all = truth_all != 255
    shares = {NAMES[k]: 100.0 * (truth_all == k).sum() / valid_all.sum() for k in range(5)
              if (truth_all == k).any()}
    print("label shares: " + ", ".join(f"{k} {v:.1f}%" for k, v in shares.items())
          + f"; clutter ignored {100.0 * (truth_all == 255).sum() / truth_all.size:.1f}%",
          flush=True)
    baseline = constant_baseline(truth_all)
    print("constant-prediction mIoU: "
          + ", ".join(f"all-{k} {v:.4f}" for k, v in sorted(baseline.items(), key=lambda z: -z[1])),
          flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "predictions").mkdir(exist_ok=True)
    results: dict[str, dict] = {}

    for model_key in args.models:
        for resolution in args.resolutions:
            run_dir = args.runs / f"{model_key}_{resolution}"
            if not (run_dir / "weights" / "best.pt").exists():
                continue
            cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
            tile = int((cfg.get("eval") or {}).get("tile_size", 0) or 0)
            overlap = int((cfg.get("eval") or {}).get("tile_overlap", 128) or 128)

            model = autoloader(str(run_dir / "weights" / "best.pt"), device=args.device)
            confusion = np.zeros((len(NAMES), len(NAMES)), dtype=np.int64)
            kept = 0
            for top in origins:
                for left in origins:
                    fine = Image.fromarray(rgb[top:top + args.crop, left:left + args.crop])
                    # Same ground footprint, the checkpoint's own pixel count (see the module
                    # docstring): the dataset's levels are resolutions of one view, not crops.
                    view = fine if resolution == args.crop else fine.resize(
                        (resolution, resolution), Image.LANCZOS)
                    prediction = np.asarray(model.predict(view, tile_size=tile,
                                                          tile_overlap=overlap))
                    if prediction.shape != (args.crop, args.crop):
                        prediction = np.asarray(
                            Image.fromarray(prediction.astype(np.uint8)).resize(
                                (args.crop, args.crop), Image.NEAREST))
                    truth = truth_all[top:top + args.crop, left:left + args.crop]
                    valid = truth != 255
                    np.add.at(confusion, (truth[valid], prediction[valid]), 1)
                    if kept < 4:  # one fine crop's worth of pictures, for the figure
                        view.save(args.out / "predictions" / f"{model_key}_{resolution}_"
                                                               f"x{left}_y{top}_input.png")
                        Image.fromarray((prediction.astype(np.uint8) * 51)).save(
                            args.out / "predictions" / f"{model_key}_{resolution}_"
                                                        f"x{left}_y{top}_pred.png")
                        kept += 1
            del model

            rows = iou_rows(confusion)
            miou = float(np.mean([iou for _, iou, _ in rows]))
            in_domain = json.loads((run_dir / "summary.json").read_text())
            in_domain = (in_domain.get("test") or {}).get("miou")
            record = {
                "model": model_key, "resolution": resolution,
                "miou": miou,
                "per_class": {name: iou for name, iou, _ in rows},
                "support": {name: support for name, _, support in rows},
                "pixel_acc": float(np.trace(confusion) / confusion.sum()),
                "miou_classes": len(rows),
                "in_domain_miou": in_domain,
                "confusion": confusion.tolist(),
                "tile_size": tile, "tile_overlap": overlap,
            }
            results[f"{model_key}_{resolution}"] = record
            print(f"  {model_key:14s} {resolution:5d}  zero-shot mIoU {miou:.4f}   "
                  f"in-domain {in_domain:.4f}" if in_domain else
                  f"  {model_key:14s} {resolution:5d}  zero-shot mIoU {miou:.4f}", flush=True)

    (args.out / "zeroshot.json").write_text(json.dumps(
        {"tile": str(args.rgb), "grid": args.grid, "crop": args.crop,
         "crop_count": len(origins) ** 2, "label_shares": shares,
         "constant_baseline": baseline, "results": results}, indent=1))
    write_table(args.out / "table.md", results, baseline)
    print(f"wrote {args.out / 'zeroshot.json'} and {args.out / 'table.md'}")


def write_table(path: Path, results: dict, baseline: dict) -> None:
    """The markdown table the paper prints: model rows, resolution columns, two blocks."""
    by_model: dict[str, dict[int, dict]] = {}
    for record in results.values():
        by_model.setdefault(record["model"], {})[record["resolution"]] = record
    columns = sorted({r["resolution"] for r in results.values()})

    lines = ["| model | " + " | ".join(f"{c}" for c in columns) + " | in-domain 2048 |",
             "| --- | " + " | ".join("---" for _ in columns) + " | --- |"]
    for model in by_model:
        cells = []
        for column in columns:
            record = by_model[model].get(column)
            cells.append(f"{record['miou']:.3f}" if record else "--")
        native = by_model[model].get(2048) or {}
        lines.append(f"| {model} | " + " | ".join(cells)
                     + f" | {native.get('in_domain_miou') or float('nan'):.3f} |")
    lines.append("")
    lines.append("Constant-prediction mIoU: "
                 + ", ".join(f"all-{k} {v:.3f}"
                             for k, v in sorted(baseline.items(), key=lambda z: -z[1])))
    lines.append("")
    lines.append("| model | resolution | " + " | ".join(NAMES[1:]) + " | mIoU |")
    lines.append("| --- | --- | " + " | ".join("---" for _ in NAMES[1:]) + " | --- |")
    for record in results.values():
        cells = [f"{record['per_class'].get(name, float('nan')):.3f}" for name in NAMES[1:]]
        lines.append(f"| {record['model']} | {record['resolution']} | " + " | ".join(cells)
                     + f" | {record['miou']:.3f} |")
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
