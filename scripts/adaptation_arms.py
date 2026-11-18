#!/usr/bin/env python3
"""The four arms reviewer R2.3 asks about, scored with the stock evaluation path.

The question is what a day-trained depth model gains from translating real night imagery into
daylight before it sees it. Four arms answer it, and only the fourth says what night imagery
can support at all:

    (a) in domain       day-trained model, real day           (the reference, not an arm)
    (a) no adaptation   day-trained model, real night
    (b) translation     day-trained model, night->day translated by CycleGAN and by pix2pix
    (c) fine-tune       day-trained model fine-tuned on real night, tested on real night
    (d) oracle          model trained from scratch on real night, tested on real night

Arms (a)-(c) differ only in what `scripts/stage_night_adaptation.py` put under
`<root>/<split>/rgb_day_lr/x8`; the depth ground truth is the same symlink in every one of them,
so every row is scored against the same targets with the same alignment. Nothing here builds a
model, an alignment or a metric: (a)-(b) are one `symtrs eval` of the same checkpoint and
(c)-(d) are that eval of a run this script does not own, which is why a row can be quoted
without qualification.

The two night-trained arms are trained elsewhere and only read here, on the same 20-epoch
night schedule as the campaign, with `train.val_subset=300`:

    # (d) from scratch
    symtrs train configs/depth/256/midas_small_256.yaml --data-root <staging>/night \\
        --out-dir runs/campaign2/adaptation/oracle_night_midas_small_256 --set train.val_subset=300
    # (c) the day-trained weights, half the campaign learning rate, no warmup, fresh schedule
    symtrs train configs/depth/256/midas_small_256.yaml --data-root <staging>/night \\
        --init-weights runs/campaign2/midas_small_256/weights/best.pt --lr 1e-5 \\
        --set train.warmup_epochs=0 --set train.val_subset=300 \\
        --out-dir runs/campaign2/adaptation/finetune_night_midas_small_256/night_finetune

    python scripts/adaptation_arms.py --device cuda:2
    python scripts/adaptation_arms.py --device cuda:2 --models midas_small_256 --dry-run

Writes `runs/campaign2/adaptation/arms.json` and `arms.md` (the table in the manuscript).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "campaign2"
STAGING = Path("/mnt/ssd1/symtrs_night256")
OUT = RUNS / "adaptation"

MODELS = ("midas_small_256", "midas_dpt_hybrid_256")
ARMS = ("night", "night2day_cyclegan", "night2day_pix2pix")
ARM_TITLE = {
    "night": "no adaptation (real night)",
    "night2day_cyclegan": "night→day, CycleGAN",
    "night2day_pix2pix": "night→day, pix2pix",
}
METRICS = ("abs_rel", "rmse", "d1", "silog")
LINE = re.compile(r"\[symtrs\] rmse=(?P<rmse>[-\d.eE+]+) mae=(?P<mae>[-\d.eE+]+) "
                  r"abs_rel=(?P<abs_rel>[-\d.eE+]+).*?d1=(?P<d1>[-\d.eE+]+).*?"
                  r"silog=(?P<silog>[-\d.eE+]+)")


def run_arm(checkpoint: Path, data_root: Path, device: str) -> dict:
    command = [sys.executable, "-m", "symtrs", "eval", str(checkpoint),
               "--data-root", str(data_root), "--eval-split", "test", "--device", device]
    result = subprocess.run(command, capture_output=True, text=True, cwd=ROOT)
    if result.returncode != 0:
        # The interpreter is in the message because a run started from the wrong environment
        # fails here rather than at import, and the stderr tail is then only loader warnings.
        tail = "\n".join(line for line in result.stderr.splitlines()
                         if "WARN:0" not in line)[-2000:]
        raise SystemExit(f"{sys.executable} failed on {data_root} (exit {result.returncode}) "
                         f"with: {' '.join(command)}\n{tail}")
    match = LINE.search(result.stdout + result.stderr)
    if match is None:
        raise SystemExit(f"no metric line from {' '.join(command)}:\n{result.stdout[-2000:]}")
    return {key: float(match.group(key)) for key in METRICS}


def in_domain(model: str) -> dict:
    return _test_metrics(RUNS / model / "summary.json")


def _test_metrics(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return {key: json.loads(path.read_text())["test"][key] for key in METRICS}


def trained_on_night(model: str, arm: str) -> dict | None:
    """The two arms that train on real night: `oracle` (from scratch) and `finetune`."""
    name = model.replace("_256", "")
    if arm == "oracle":
        return _test_metrics(OUT / f"oracle_night_{name}_256" / "summary.json")
    return _test_metrics(OUT / f"finetune_night_{name}_256" / "night_finetune" / "summary.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--staging-root", type=Path, default=STAGING)
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true", help="print the commands only")
    args = parser.parse_args()

    rows: dict[str, dict[str, dict]] = {}
    for model in args.models:
        checkpoint = RUNS / model / "weights" / "best.pt"
        if not checkpoint.exists():
            raise SystemExit(f"missing {checkpoint}; score the campaign run first")
        rows[model] = {"in_domain": in_domain(model)}
        for arm in ("oracle", "finetune"):
            found = trained_on_night(model, arm)
            if found is not None:
                rows[model][arm] = found
        for arm in ARMS:
            root = args.staging_root / arm
            if not (root / "test").is_dir():
                print(f"skip {model} / {arm}: {root} is not staged", file=sys.stderr)
                continue
            if args.dry_run:
                print(f"symtrs eval {checkpoint} --data-root {root} --eval-split test "
                      f"--device {args.device}")
                continue
            print(f"{model} / {arm} ...", flush=True)
            rows[model][arm] = run_arm(checkpoint, root, args.device)
            print(f"  abs_rel={rows[model][arm]['abs_rel']:.4f}", flush=True)

    if args.dry_run:
        return

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "arms.json").write_text(json.dumps(rows, indent=2) + "\n")
    lines = []
    for model, arms in rows.items():
        lines += [f"### {model}", "",
                  "| arm | abs_rel ↓ | RMSE ↓ | δ1 ↑ | silog ↓ |",
                  "| --- | --- | --- | --- | --- |"]
        for arm, label in _labels(arms):
            value = arms[arm]
            lines.append(f"| {label} | {value['abs_rel']:.4f} | {value['rmse']:.2f} | "
                         f"{value['d1']:.4f} | {value['silog']:.2f} |")
        lines.append("")
    (OUT / "arms.md").write_text("\n".join(lines))
    print(f"wrote {OUT / 'arms.json'} and {OUT / 'arms.md'}")


def _labels(arms: dict) -> list[tuple[str, str]]:
    order = [("in_domain", "in domain (real day)")]
    order += [(arm, ARM_TITLE[arm]) for arm in ARMS]
    order += [("finetune", "fine-tuned on real night (from the day weights)")]
    order += [("oracle", "oracle: trained on real night from scratch")]
    return [(key, label) for key, label in order if key in arms]


if __name__ == "__main__":
    main()
