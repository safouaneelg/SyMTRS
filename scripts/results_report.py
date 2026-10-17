"""Write docs/claude/RESULTS.md: the TEST metrics of every finished validation run, per task.

Reads `<runs>/<stem>/summary.json` through `symtrs.commands.tables.collect` (the same loader
and `n/a` formatting as `symtrs tables`), so the numbers here are the ones the runs wrote and
nothing is recomputed. Summaries from older code (the smoke pass 1 format) lack some keys
(`test_weights`, `invalid_pred_fraction`, `miou_classes`, ...): those cells print `n/a`.

    .venv/Scripts/python.exe scripts/results_report.py [--runs runs/validation] [--out docs/claude/RESULTS.md]

Deliberately not done: ranking, bolding a "best" row, averaging across tasks. These are
validation runs; the header says so and the tables do not invite a comparison the protocol
does not support.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from symtrs.commands.tables import _format, collect  # noqa: E402

TASK_ORDER = ["segmentation", "depth", "superresolution", "shadow_removal", "domain_adapt"]

# (header, source key or callable name). Keys are read from summary["test"] unless noted.
COLUMNS: Dict[str, List[Tuple[str, str]]] = {
    "segmentation": [("mIoU", "miou"), ("mIoU classes", "miou_classes"), ("pixel acc", "pixel_acc"),
                     ("mean Dice", "mean_dice"), ("FWIoU", "fwiou"), ("test weights", "@test_weights")],
    "depth": [("abs_rel", "abs_rel"), ("RMSE (m)", "rmse"), ("SILog", "silog"), ("d1", "d1"),
              ("align", "@align"), ("invalid pred", "invalid_pred_fraction"),
              ("test weights", "@test_weights")],
    "superresolution": [("scale", "@scale"), ("PSNR (dB)", "psnr"), ("SSIM", "ssim")],
    "shadow_removal": [("init", "@init"), ("PSNR", "psnr"), ("SSIM_y", "ssim_y"),
                       ("MAE LAB all", "mae_lab"), ("MAE LAB shadow", "mae_lab_shadow"),
                       ("MAE LAB non-shadow", "mae_lab_nonshadow"),
                       ("RMSE LAB all", "rmse_lab"), ("RMSE LAB shadow", "rmse_lab_shadow"),
                       ("RMSE LAB non-shadow", "rmse_lab_nonshadow")],
    "domain_adapt": [("KID (primary)", "kid"), ("KID std", "kid_std"), ("FID", "fid"),
                     ("FID rank-def.", "fid_rank_deficient"), ("N gen.", "n_fake")],
}
# Paired columns, only meaningful for the shadow2noshadow rows (ground truth exists there).
PAIRED_COLUMNS = [("PSNR", "psnr"), ("SSIM_y", "ssim_y"), ("MAE LAB all", "mae_lab"),
                  ("MAE LAB shadow", "mae_lab_shadow"), ("MAE LAB non-shadow", "mae_lab_nonshadow"),
                  ("RMSE LAB all", "rmse_lab"), ("RMSE LAB shadow", "rmse_lab_shadow"),
                  ("RMSE LAB non-shadow", "rmse_lab_nonshadow")]

NOTES = {
    "segmentation": "Test split, best-val-mIoU weights. `mIoU classes` = how many of the 6 classes "
                    "the mean covers (a class absent from the split is excluded, not scored 0). "
                    "Class shares differ strongly between the geographic splits.",
    "depth": "Test split, metres, pixels beyond `max_depth` masked. `align` is the alignment the "
             "model is scored under (relative models: per-image fit; metric: none). "
             "`invalid pred` = fraction of pixels with a non-positive prediction (kept in the "
             "metrics, not dropped). `test weights` = which checkpoint was scored (`best`/`last`). "
             "DA3-metric uses an assumed 60 deg horizontal FOV to derive the focal length. "
             "Test depth is deeper than train (median ~208 m vs ~92 m).",
    "superresolution": "PSNR / SSIM on the legacy trainer's definition (Y channel when "
                       "`--use_y_channel`), LR at `2048 / scale` -> native `rgb_day`. x1 is the "
                       "refiner control, not super-resolution.",
    "shadow_removal": "`rgb_day` -> `rgb_noshadow` at 2048. `init` = what the run was "
                      "initialised from; the three models do NOT start from comparable weights "
                      "(ShadowFormer: ISTD+ shadow removal; Restormer/Uformer: denoising), so rows "
                      "must not be ranked. SSIM_y is luma-only SSIM (not comparable with "
                      "published per-channel SSIM). Shadow mask is derived from the pair.",
    "domain_adapt": "CycleGAN / pix2pix train and predict at 256 px; the generated images are "
                    "upsampled and scored against the real ones at 2048. KID is the primary "
                    "metric; FID (N=24 vs 2048-d features, rank-deficient covariance) is shown "
                    "for completeness only.",
}


def _dataset_fingerprint() -> str:
    md = REPO / "docs" / "claude" / "DATASET.md"
    if md.exists():
        hit = re.search(r"Dataset fingerprint: `(\w+)`", md.read_text(encoding="utf-8"))
        if hit:
            return hit.group(1)
    return "?"


def _run_config(row: Dict[str, Any]) -> Dict[str, Any]:
    path = Path(row["run"]) / "config.yaml"
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _summary(row: Dict[str, Any]) -> Dict[str, Any]:
    import json

    try:
        return json.loads((Path(row["run"]) / "summary.json").read_text(encoding="utf-8"))
    except Exception:
        return {}


def _depth_align(row: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    """The alignment the run was scored under: the config's override, else the model default."""
    override = (cfg.get("eval") or {}).get("align")
    if override:
        return str(override)
    try:
        from symtrs.registry import get_spec

        return get_spec(row["model"]).default_align or "none"
    except Exception:
        return "n/a"


def _special(name: str, row: Dict[str, Any], summ: Dict[str, Any], cfg: Dict[str, Any]) -> Any:
    if name == "test_weights":
        return summ.get("test_weights")
    if name == "align":
        return _depth_align(row, cfg)
    if name == "scale":
        return f"x{summ['scale']}" if summ.get("scale") else (
            f"x{2048 // int(row['resolution'])}" if row.get("resolution") else None)
    if name == "init":
        src = summ.get("init_source_task")
        w = summ.get("init_weights")
        if w:
            return f"{Path(str(w)).name} ({src})" if src else Path(str(w)).name
        return "scratch" if "init_weights" in summ else None
    return None


def _cell(spec: str, row: Dict[str, Any], summ: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    value = _special(spec[1:], row, summ, cfg) if spec.startswith("@") else row.get(spec)
    if value is None or value == "":
        return "n/a"
    # Counts are stored as floats by the metric aggregators; show them as integers.
    if spec in ("n_fake", "n_real", "miou_classes") and isinstance(value, (int, float)):
        return str(int(value))
    return _format(value)


def _table(rows: Sequence[Dict[str, Any]], head: Sequence[Tuple[str, str]], lead: Sequence[str]) -> List[str]:
    headers = list(lead) + [h for h, _ in head]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        summ, cfg = _summary(row), _run_config(row)
        lead_cells = []
        for name in lead:
            if name == "run":
                lead_cells.append(f"`{Path(row['run']).name}`")
            elif name == "direction":
                lead_cells.append(str(row.get("direction", "n/a")))
            else:
                lead_cells.append(str(row.get(name, "n/a")))
        out.append("| " + " | ".join(lead_cells + [_cell(s, row, summ, cfg) for _, s in head]) + " |")
    return out


def _sort_key(row: Dict[str, Any]) -> Tuple:
    return (str(row.get("model", "")), str(row.get("direction", "")), int(row.get("resolution") or 0))


def render(rows: List[Dict[str, Any]], runs_label: str) -> str:
    n_total = len(rows)
    L = [
        "# RESULTS",
        "",
        "> **These are validation runs, not benchmark results.** Per AGENTS.md (\"The dataset is "
        "under revision\"), no number below is a benchmark number: it shows that each model "
        "trains and is scored on the real data, nothing more.",
        "",
        f"Generated by `scripts/results_report.py` on {dt.datetime.now():%Y-%m-%d %H:%M} from "
        f"`{runs_label}/*/summary.json` ({n_total} finished runs). Do not edit by hand: re-run "
        "the script. How each run was configured and launched: [RUNS.md](RUNS.md).",
        "",
        "## Read this before reading any number",
        "",
        "- **Budget**: 2 epochs (shadow removal: 500 iterations) per model. Nothing is converged.",
        f"- **Data**: preview dataset, 144 transforms (96 train / 24 val / 24 test), dataset "
        f"fingerprint `{_dataset_fingerprint()}` ([DATASET.md](DATASET.md)).",
        "- **Geographic splits with distribution shift**: the test split is deeper (median depth "
        "~208 m vs ~92 m in train) and class shares differ strongly between splits (val is 63 % "
        "road). A generalisation gap here is partly a dataset property, not only a model one.",
        "- **FOV**: the capture camera's 60 deg horizontal FOV is an assumption (the Unreal "
        "actor default); it sets the focal length that DA3-metric (and Depth Pro, `f_px`) use. "
        "Zero-shot DA3-metric stays far too small on nadir views at 70-270 m (domain gap, not a "
        "unit bug).",
        "- **Shadow-removal inits are heterogeneous**: ShadowFormer starts from ISTD+ "
        "(shadow-removal weights), Restormer and Uformer from denoising weights (no shadow "
        "weights released). The rows are not a fair comparison.",
        "- **Shadow translation (CycleGAN / pix2pix)** trains and predicts at 256 px and is "
        "scored at 2048. **KID is the primary metric over FID** (N=24 samples against "
        "2048-d features: the FID covariance is rank-deficient).",
        "- `n/a` = metric absent from that summary (older code) or not computable; never 0.",
        "",
    ]
    by_task: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_task.setdefault(row.get("task", ""), []).append(row)
    for task in TASK_ORDER:
        items = sorted(by_task.get(task, []), key=_sort_key)
        if not items:
            continue
        L += [f"## {task} ({len(items)} runs)", "", NOTES[task], ""]
        if task == "domain_adapt":
            plain = [r for r in items if r.get("direction") != "shadow2noshadow"]
            paired = [r for r in items if r.get("direction") == "shadow2noshadow"]
            if plain:
                L += _table(plain, COLUMNS[task], ["model", "direction"]) + [""]
            if paired:
                L += ["### shadow -> no shadow (paired metrics vs `rgb_noshadow`)", "",
                      "Same metric definitions as shadow_removal, for a direct read against it.", ""]
                L += _table(paired, COLUMNS[task][:3] + PAIRED_COLUMNS, ["model", "direction"]) + [""]
        else:
            lead = ["model", "resolution"] if task in ("segmentation", "depth") else ["model"]
            if task == "superresolution":
                lead = ["model"]
            L += _table(items, COLUMNS[task], lead) + [""]
    seen = set(by_task) - set(TASK_ORDER)
    for task in sorted(seen):
        L += [f"## {task or '(unknown task)'} ({len(by_task[task])} runs)", "",
              "No dedicated table for this task; see the run summaries.", ""]
    if not rows:
        L += ["_No finished run with a `test` entry found yet._", ""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", default="runs/validation")
    ap.add_argument("--out", default="docs/claude/RESULTS.md")
    args = ap.parse_args()
    runs = Path(args.runs)
    runs = runs if runs.is_absolute() else REPO / runs
    rows = collect(str(runs))
    # Only direct children (<runs>/<stem>/summary.json): a run's own nested dirs are not runs.
    rows = [r for r in rows if Path(r["run"]).parent.resolve() == runs.resolve()]
    out = Path(args.out)
    out = out if out.is_absolute() else REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(rows, args.runs.replace("\\", "/")) + "\n", encoding="utf-8")
    print(f"{len(rows)} runs -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
