"""Write docs/claude/RUNS.md: how every model was run, on which data, with which code.

Generated from what the runs themselves wrote - each run's resolved `config.yaml` (every
parameter actually used, after CLI overrides), `summary.json`, the exact command line that
`scripts/run_queue.py` logged, and the queue's status file - so the document cannot drift
from what was executed. Re-run it after the queue finishes (or at any point: it reports
whatever exists).

    .venv/Scripts/python.exe scripts/runs_report.py [--runs runs/validation] [--out docs/claude/RUNS.md]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_queue import REPO, run_env  # noqa: E402

TASK_ORDER = ["segmentation", "depth", "superresolution", "shadow_removal", "domain_adapt"]

# What each task reads and predicts, stated once (AGENTS.md section 1 and ground rules 1-5).
PROTOCOL = {
    "segmentation": "`rgb_day{_lr/xN}` -> `semantic{_lr/xN}`; input and label at the same "
                    "resolution (2048 native, 1024 = x2, 512 = x4, 256 = x8). Labels decoded "
                    "as `round(gray / label_step)`, off-grid gray -> `ignore_index` 255. "
                    "Monitor: val mIoU.",
    "depth": "`rgb_day{_lr/xN}` -> `depth{_lr/xN}` (float32 EXR, centimetres along the camera "
             "axis) x `depth_scale` -> metres; pixels beyond `max_depth` m (sky) invalid. "
             "Relative models: SILog + per-image scale alignment (MiDaS: SSI + scale-and-shift fit in disparity); metric models: "
             "BerHu loss, no alignment (registry `protocol`). Monitor: val RMSE.",
    "superresolution": "`rgb_day_lr/xN` (LR input, `resolution` = its size) -> native "
                       "`rgb_day` (2048); scale = 2048 / resolution, x1 = refiner control. "
                       "Legacy trainer `symtrs/legacy/superres/train.py` run as a subprocess.",
    "shadow_removal": "`rgb_day` (2048) -> `rgb_noshadow` (2048), native only. Shadow mask "
                      "(ShadowFormer input and the shadow / non-shadow metric split) derived "
                      "from the pair: `(Y_noshadow - Y_day) / Y_noshadow > mask_threshold`, "
                      "then a `mask_open` px opening. Upstream code in `third_party/` "
                      "(pins in `third_party/PINS.md`). Metrics: PSNR, SSIM, LAB RMSE over "
                      "all / shadow / non-shadow pixels.",
    "domain_adapt": "`rgb_day` <-> `rgb_night` (2048 only); vendored "
                    "`pytorch-CycleGAN-and-pix2pix` at `2a7afba` run as a subprocess; data "
                    "staged per split under `<run_dir>/staging/`. Metrics: FID / KID.",
}


def sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, cwd=REPO,
                              timeout=120).stdout.strip()
    except Exception as exc:  # the report must still be written
        return f"? ({exc})"


def venv_versions(venv: str) -> str:
    py = REPO / venv / "Scripts" / "python.exe"
    if not py.exists():
        return "absent"
    code = ("import sys,importlib\n"
            "out=['python '+sys.version.split()[0]]\n"
            "for m in ('torch','torchvision','timm','transformers','numpy'):\n"
            "  try: out.append(m+' '+importlib.import_module(m).__version__)\n"
            "  except Exception: pass\n"
            "import torch; out.append('cuda '+str(torch.version.cuda))\n"
            "print(', '.join(out))")
    return sh([str(py), "-c", code])


def logged_command(stem: str) -> str:
    """Last command line run_queue.py wrote for this config (`=== <time> <cmd>`)."""
    logs = sorted((REPO / "logs").glob(f"{stem}_*.log"), key=lambda p: p.stat().st_mtime)
    for log in reversed(logs):
        hits = re.findall(r"^=== \S+ \S+ (.+)$", log.read_text(encoding="utf-8", errors="replace"), re.M)
        if hits:
            return hits[-1].replace(str(REPO) + "\\", "").replace(str(REPO) + "/", "")
    return "(no log found)"


def status_rows(runs: Path) -> dict:
    rows = {}
    tsv = runs / "status.tsv"
    if tsv.exists():
        for line in tsv.read_text(encoding="utf-8").splitlines():
            p = line.split("\t")
            if len(p) >= 4:
                rows[p[1]] = (p[2], p[3], p[0])  # last row wins: the latest attempt
    return rows


def fmt_metrics(d: dict, limit: int = 6) -> str:
    if not isinstance(d, dict) or not d:
        return "-"
    keys = [k for k, v in d.items() if isinstance(v, (int, float)) and not re.match(r"(iou|dice)_\d", k)]
    return ", ".join(f"{k} {d[k]:.4g}" for k in keys[:limit])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", default="runs/validation")
    ap.add_argument("--out", default="docs/claude/RUNS.md")
    args = ap.parse_args()
    runs = REPO / args.runs
    status = status_rows(runs)

    dataset_md = REPO / "docs" / "claude" / "DATASET.md"
    fp = re.search(r"Dataset fingerprint: `(\w+)`", dataset_md.read_text(encoding="utf-8")) \
        if dataset_md.exists() else None
    env = run_env()
    L = [
        "# How every model was run",
        "",
        f"Generated by `scripts/runs_report.py` on {dt.datetime.now():%Y-%m-%d %H:%M} from "
        f"`{args.runs}/` (each run's resolved `config.yaml`, `summary.json`, the logged command "
        "and `status.tsv`). Do not edit by hand: re-run the script.",
        "",
        "**These are validation runs, not benchmark results**: preview dataset (144 transforms, "
        "96/24/24), 2 epochs (shadow removal: 500 iterations). They show that each model trains "
        "and is scored on the real data; the numbers are not comparable to a converged run "
        "(AGENTS.md: the dataset is under revision).",
        "",
        "## Data",
        "",
        f"- Root: `{next((yaml.safe_load((d / 'config.yaml').read_text(encoding='utf-8'))['data']['root'] for d in sorted(runs.glob('*')) if (d / 'config.yaml').exists()), '?')}`",
        f"- Dataset fingerprint: `{fp.group(1) if fp else '?'}` - per-folder counts, sizes and "
        "fingerprints in [DATASET.md](DATASET.md) (`scripts/dataset_manifest.py`).",
        "- Splits: train for training, val for model selection (best checkpoint), test scored "
        "once with the best checkpoint. No split is carved out of train in code.",
        "",
        "## Code and environment",
        "",
        f"- Code: `{sh(['git', 'rev-parse', '--short', 'HEAD'])}` on branch "
        f"`{sh(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])}`"
        f"{' + uncommitted changes' if sh(['git', 'status', '--porcelain']) else ''}.",
        f"- GPU: {sh(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'])}"
        " (driver CUDA sysmem fallback disabled).",
        f"- `.venv`: {venv_versions('.venv')}",
        f"- `.venv-zoe` (ZoeDepth, MiDaS Swin2-tiny: timm 0.6.12): {venv_versions('.venv-zoe')}",
        "- Environment set for every run (`scripts/run_queue.py` `run_env`): " + ", ".join(
            f"`{k}={v if 'cache' not in v else v.replace(str(REPO), '<repo>')}`"
            for k, v in env.items() if k in ("TORCH_HOME", "HF_HOME", "HF_HUB_OFFLINE",
                                              "SYMTRS_CUDA_MEM_FRACTION", "OMP_NUM_THREADS")) + ".",
        "- Batch sizes: the YAMLs' values were tried first and halved on out-of-memory by "
        "`scripts/smoke_all.py` (record: `runs/smoke/smoke.tsv`); the value used is in the "
        "table below and in each run's `config.yaml`.",
        "",
        "## Per task protocol",
        "",
    ]
    for task in TASK_ORDER:
        L.append(f"- **{task}**: {PROTOCOL[task]}")
    L.append("")

    by_task: dict = {}
    for d in sorted(p for p in runs.glob("*") if (p / "config.yaml").exists()):
        cfg = yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8"))
        summ = {}
        if (d / "summary.json").exists():
            summ = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        by_task.setdefault(cfg["task"], []).append((d.name, cfg, summ))

    for task in TASK_ORDER:
        items = by_task.get(task, [])
        if not items:
            continue
        L += [f"## {task} ({len(items)} runs)", "",
              "| run | model | res | budget | batch x accum | patch | lr | loss | eval | status | min | test |",
              "|---|---|---:|---|---|---:|---|---|---|---|---:|---|"]
        for name, cfg, summ in items:
            t, e, s = cfg["train"], cfg["eval"], cfg.get("shadow", {})
            budget = (f"{s.get('iterations')} it" if task == "shadow_removal" else f"{t['epochs']} ep")
            evals = (f"align={e.get('align') or 'none'}" if task == "depth" else "") + \
                    (f" tile={e['tile_size']}" if e.get("tile_size") else "")
            if task == "shadow_removal" and s.get("init_weights"):
                evals += f" init={Path(s['init_weights']).name}"
            st = status.get(name, ("-", "-", ""))
            L.append(f"| `{name}` | {cfg['model']} | {cfg['resolution']} | {budget} | "
                     f"{t['batch_size']} x {t.get('grad_accum', 1)} | {t.get('patch_size') or '-'} | "
                     f"{t['lr']:g} | {t.get('loss') or '-'} | {evals.strip() or '-'} | {st[0]} | {st[1]} | "
                     f"{fmt_metrics(summ.get('test', {}))} |")
        L += ["", "<details><summary>Exact commands</summary>", "", "```"]
        L += [logged_command(name) for name, _, _ in items]
        L += ["```", "", "</details>", ""]

    out = REPO / args.out
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"{sum(len(v) for v in by_task.values())} runs -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
