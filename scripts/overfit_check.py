"""Can each model learn at all on SyMTRS? Overfit a tiny subset and keep the ones that do.

Why this exists: on the first real-data campaign the three ZoeDepth rows ran 1050 steps at
512 px with a flat loss (ZoeD-N 2.81 -> 2.78, ZoeD-K 84 -> 83, ZoeD-NK ~172 and 107 skipped
non-finite batches), while MiDaS and SegFormer dropped within 100 steps. A row that cannot
fit 32 images it sees every epoch will not fit 14,000, and finding that out after days of
GPU time is the expensive way. This is the cheap way, run before a campaign:

    python scripts/overfit_check.py subset --data-root /mnt/ssd1/SyMTRS --out runs/overfit/data
    python scripts/overfit_check.py queue  --out runs/queue_overfit.txt
    .venv/bin/python scripts/run_queue.py --queue runs/queue_overfit.txt \
        --out-root runs/overfit/runs --data-root runs/overfit/data --jobs 3 --devices 0,1,2 \
        --num-workers 4 --omp-threads 2 --timeout-min 120
    python scripts/overfit_check.py verdict runs/overfit/runs

The subset is symlinks only (the dataset is never written): the same `--train` samples in
every folder of `train/`, and the first `--eval` of them as `val/` and `test/`, so the
validation score measures fit on images the model trained on. That is the point here and
the reason no number from it may be reported as a result.

Each config keeps its own optimiser, learning rate, freezing and loss - the question is
whether the row *as configured* can learn - and only the length changes: depth and
segmentation get about `STEPS` optimiser steps whatever their effective batch, the
subprocess tasks a fixed budget. What this does not do: tell a good model from a bad one.
A row that passes has gradients that reach something able to fit; a row that fails has a
configuration (or a model) problem worth fixing before it is given the full dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: Optimiser steps given to every depth / segmentation row.
STEPS = 120
#: Train-loss ratio (last epoch / first epoch) under which a row "learns", and over which it
#: does not. In between is "weak": it moves, but not like a model fitting 32 images.
LEARNS, STUCK = 0.7, 0.9

SAMPLE = re.compile(r"(_loc\d+_.*)\.\w+$")

# (venv, config, extra flags). One resolution per depth/segmentation model is enough: the
# freeze, lr and loss come from one table row in configs/generate_grid.py for every size.
TIMM_0612 = {"midas_dpt_swin2_tiny", "midas_dpt_beit_large"}
SHADOW_INIT = {
    "shadowformer": "checkpoints/shadow/shadowformer_istdplus.pth",
    "restormer": "checkpoints/shadow/restormer_real_denoising.pth",
}


def cmd_subset(args) -> None:
    root, out = Path(args.data_root), Path(args.out)
    names = sorted(os.listdir(root / "train" / "rgb_day"))
    stride = max(1, len(names) // args.train)
    keys = [SAMPLE.search(n).group(1) for n in names[::stride][: args.train]]
    for split, chosen in (("train", keys), ("val", keys[: args.eval]), ("test", keys[: args.eval])):
        wanted = set(chosen)
        for folder, _, files in os.walk(root / "train"):
            rel = Path(folder).relative_to(root / "train")
            picked = [f for f in files if (m := SAMPLE.search(f)) and m.group(1) in wanted]
            if not picked:
                continue
            dest = out / split / rel
            dest.mkdir(parents=True, exist_ok=True)
            for f in picked:
                link = dest / f
                if not link.exists():
                    link.symlink_to(Path(folder) / f)
    print(f"{out}: {len(keys)} train samples, {args.eval} val/test (a subset of train)")


def _depth_seg_lines():
    from symtrs.config import RunConfig

    lines = []
    for task in ("depth", "segmentation"):
        for path in sorted((REPO / "configs" / task / "512").glob("*.yaml")):
            cfg = RunConfig.from_yaml(str(path))
            effective = cfg.train.batch_size * max(1, cfg.train.grad_accum)
            # Steps per epoch on 32 images; a window larger than the subset still steps once.
            per_epoch = max(1, math.ceil(math.ceil(32 / cfg.train.batch_size)
                                         / max(1, cfg.train.grad_accum)))
            epochs = max(4, math.ceil(STEPS / per_epoch))
            venv = ".venv-zoe" if cfg.model in TIMM_0612 else ".venv"
            lines.append(f"{venv}  {path.relative_to(REPO)}  --set train.epochs={epochs}  "
                         f"--set train.val_every={max(1, epochs // 4)}  "
                         f"--set train.save_samples_every=0    # effective batch {effective}")
    return lines


def cmd_queue(args) -> None:
    lines = ["# Overfit check (scripts/overfit_check.py). NOT results: val == a subset of train."]
    lines += _depth_seg_lines()
    for name in ("srcnn", "autoencoder", "srgan", "swinir"):
        lines.append(f".venv  configs/superresolution/{name}_x4.yaml  --set train.epochs=40")
    for name in ("shadowformer", "restormer", "uformer", "shadowmaskformer"):
        init = f"  --set shadow.init_weights={SHADOW_INIT[name]}" if name in SHADOW_INIT else ""
        lines.append(f".venv  configs/shadow_removal/{name}.yaml  --set train.epochs=50  "
                     f"--set shadow.val_every=100{init}")
    for name in ("cyclegan", "pix2pix"):
        for direction in ("day2night", "shadow2noshadow"):
            lines.append(f".venv  configs/domain_adapt/{name}_{direction}.yaml  "
                         f"--set train.epochs=20")
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{args.out}: {len(lines) - 1} jobs")


def _rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _num(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _domain_losses(run: Path):
    """Means of upstream's reconstruction loss (L1 or cycle) over the first and second half
    of the logged epochs.

    Upstream logs one batch-1 iteration every few epochs, and a CycleGAN's cycle terms swing
    2x between two such lines, so the first and last line alone called a CycleGAN that beat
    the identity baseline by 7 dB on shadow pixels "stuck". Halves average that out; the
    test score beside it (when the direction is paired) is the stronger evidence.
    """
    logs = list(run.glob("checkpoints/*/loss_log.txt"))
    if not logs:
        return None, None, "no loss_log.txt"
    per_epoch = {}
    for line in logs[0].read_text(encoding="utf-8", errors="replace").splitlines():
        epoch = re.search(r"epoch: (\d+)", line)
        terms = dict(re.findall(r"(\w+): (-?[\d.]+(?:e-?\d+)?)", line))
        keys = [k for k in ("G_L1", "cycle_A", "cycle_B") if k in terms]
        if epoch and keys:
            per_epoch.setdefault(int(epoch.group(1)), []).append(sum(float(terms[k]) for k in keys))
    if len(per_epoch) < 2:
        return None, None, "fewer than two epochs logged"
    ordered = [sum(v) / len(v) for _, v in sorted(per_epoch.items())]
    half = len(ordered) // 2
    first = sum(ordered[:half]) / half
    last = sum(ordered[half:]) / (len(ordered) - half)
    note = "reconstruction loss, first vs second half of logged epochs"
    summary = run / "summary.json"
    if summary.exists():
        test = json.loads(summary.read_text(encoding="utf-8")).get("test") or {}
        if "psnr_shadow" in test:
            note += (f"; test psnr {test['psnr']:.2f} / shadow {test['psnr_shadow']:.2f} "
                     "(compare scripts/shadow_identity_baseline.py)")
    return first, last, note


def _judge(run: Path):
    """(signal, first, last, val_first, val_best, note) for one run directory."""
    csv_path = run / "metrics.csv"
    if run.name.startswith(("cyclegan", "pix2pix")):
        first, last, note = _domain_losses(run)
        return "loss", first, last, None, None, note
    if not csv_path.exists():
        return "", None, None, None, None, "no metrics.csv"
    rows = _rows(csv_path)
    if "split" in rows[0]:  # shadow removal worker
        train = [_num(r.get("loss")) for r in rows if r["split"] == "train"]
        val = [_num(r.get("psnr")) for r in rows if r["split"] == "val_crops"]
        train, val = [v for v in train if v is not None], [v for v in val if v is not None]
        return ("loss", train[0] if train else None, train[-1] if train else None,
                val[0] if val else None, max(val) if val else None, "val-crop PSNR (higher)")
    if "mse_train" in rows[0]:  # legacy super-resolution: pixel MSE, not the GAN total
        train = [_num(r["mse_train"]) for r in rows]
        val = [_num(r["val_psnr"]) for r in rows]
        train, val = [v for v in train if v is not None], [v for v in val if v is not None]
        return ("mse", train[0] if train else None, train[-1] if train else None,
                val[0] if val else None, max(val) if val else None, "val PSNR (higher)")
    train = [_num(r.get("train_loss")) for r in rows]
    train = [v for v in train if v is not None]
    for key, better, label in (("val_miou", max, "val mIoU (higher)"),
                               # rmse, not abs_rel: it is what the depth configs select
                               # best.pt on, and abs_rel moved the wrong way for ZoeD-K
                               # while its rmse and d1 improved (near pixels dominate it).
                               ("val_rmse", min, "val rmse (lower)")):
        val = [_num(r.get(key)) for r in rows if _num(r.get(key)) is not None]
        if val:
            return ("loss", train[0] if train else None, train[-1] if train else None,
                    val[0], better(val), label)
    return ("loss", train[0] if train else None, train[-1] if train else None, None, None,
            "no validation row")


def cmd_verdict(args) -> None:
    root = Path(args.runs)
    out = []
    for run in sorted(p for p in root.iterdir() if p.is_dir()):
        signal, first, last, v0, vbest, note = _judge(run)
        failed = (run / "summary_failed.json").exists()
        if first is None or last is None or first <= 0:
            verdict, ratio = ("FAILED" if failed else "NO DATA"), None
        else:
            ratio = last / first
            verdict = "learns" if ratio <= LEARNS else "weak" if ratio <= STUCK else "STUCK"
            # The val images are a subset of the train images, so a falling train loss with a
            # flat val score is a model fitting the loss and not the task (ZoeD-N at 10x lr:
            # silog halved, scale-aligned rmse never beat its first value).
            if verdict == "learns" and v0 is not None and vbest is not None:
                if abs(vbest - v0) <= 0.02 * abs(v0):
                    verdict = "weak (val flat)"
        out.append({"run": run.name, "verdict": verdict, "signal": signal, "first": first,
                    "last": last, "ratio": ratio, "val_first": v0, "val_best": vbest,
                    "val": note})
    fmt = lambda v: "-" if v is None else f"{v:.4g}"  # noqa: E731
    print("| run | verdict | train first | last | last/first | val first | val best | val metric |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in out:
        print(f"| {r['run']} | {r['verdict']} | {fmt(r['first'])} | {fmt(r['last'])} | "
              f"{fmt(r['ratio'])} | {fmt(r['val_first'])} | {fmt(r['val_best'])} | {r['val']} |")
    (root / "overfit_verdict.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("subset")
    s.add_argument("--data-root", required=True)
    s.add_argument("--out", default="runs/overfit/data")
    s.add_argument("--train", type=int, default=32)
    s.add_argument("--eval", type=int, default=8)
    q = sub.add_parser("queue")
    q.add_argument("--out", default="runs/queue_overfit.txt")
    v = sub.add_parser("verdict")
    v.add_argument("runs", nargs="?", default="runs/overfit/runs")
    args = ap.parse_args()
    {"subset": cmd_subset, "queue": cmd_queue, "verdict": cmd_verdict}[args.cmd](args)


if __name__ == "__main__":
    main()
