"""Smoke-test every generated config on the real dataset, find a batch size that fits the
card, and write the validation queue (`runs/queue.txt`) for `scripts/run_queue.py`.

Why the batch search: the generated configs carry batch sizes chosen for large cards
(U-Net 512 at batch 32 wants 18.6 GB). With `SYMTRS_CUDA_MEM_FRACTION` set (run_env does),
an oversized batch raises OOM within seconds instead of spilling into system RAM, so each
config is tried at its own batch size and halved on OOM until it fits. What was chosen is
recorded per config in `runs/smoke/smoke.tsv` and carried into the queue as `--batch-size`;
the YAMLs are not edited (they stay the reference grid for the full dataset).

Resumable like the queue: a config with an `ok` row in smoke.tsv is not re-run, so after a
crash just start it again. One job at a time, by design (this machine blue-screens under
combined load).

    .venv/Scripts/python.exe scripts/smoke_all.py --data-root "<root>" [--tasks segmentation,depth] [--only midas_small_512]
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_queue import REPO, now, run_env, run_job, train_cmd  # noqa: E402

TASK_ORDER = ["segmentation", "depth", "superresolution", "shadow_removal", "domain_adapt"]
# Models whose upstream code pins timm 0.6.12 (AGENTS.md ground rule 10).
ZOE_VENV_MODELS = ("zoedepth_n", "zoedepth_k", "zoedepth_nk", "midas_dpt_swin2_tiny")
OOM = re.compile(r"OutOfMemoryError|CUDA out of memory|CUBLAS_STATUS_ALLOC_FAILED")

# Smoke budget: enough to cross every code path (train step, val, checkpoint, test scoring),
# not to learn anything.
SMOKE_FLAGS = {
    "default": ["--epochs", "1", "--limit-train-batches", "3", "--limit-val-batches", "2"],
    "superresolution": ["--epochs", "1"],
    "domain_adapt": ["--epochs", "1"],
    "shadow_removal": ["--set", "shadow.iterations=20"],
}
# Validation budget written to the queue: "does it train and score on the real data".
VALIDATION_FLAGS = {
    "default": ["--epochs", "2"],
    "shadow_removal": ["--set", "shadow.iterations=500"],
}


def read_header(path: Path) -> dict:
    """task / model / batch_size from a generated YAML, without importing yaml or symtrs."""
    out = {}
    section = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" "):
            section = line.split(":", 1)[0].strip()
        key, _, value = line.strip().partition(":")
        value = value.strip()
        if section in ("task", "model") and key == section:
            out[key] = value
        if section == "train" and key == "batch_size":
            out["batch_size"] = int(value)
    return out


def configs_for(tasks: list[str]) -> list[Path]:
    found = []
    for task in tasks:
        found += sorted((REPO / "configs" / task).rglob("*.yaml"))
    return found


def load_done(tsv: Path) -> dict:
    done = {}
    if tsv.exists():
        for line in tsv.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) >= 5 and parts[2] == "ok":
                done[parts[1]] = int(parts[3])
    return done


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--tasks", default=",".join(TASK_ORDER))
    ap.add_argument("--only", default="", help="comma-separated config stems")
    ap.add_argument("--timeout-min", type=float, default=30.0)
    ap.add_argument("--queue", default="runs/queue.txt")
    ap.add_argument("--num-workers", default="2")
    args = ap.parse_args()

    smoke_root = REPO / "runs" / "smoke"
    smoke_root.mkdir(parents=True, exist_ok=True)
    (REPO / "logs").mkdir(exist_ok=True)
    tsv = smoke_root / "smoke.tsv"
    done = load_done(tsv)
    env = run_env()
    only = {s for s in args.only.split(",") if s}
    stop = REPO / "runs" / "STOP"

    for config in configs_for(args.tasks.split(",")):
        stem = config.stem
        if (only and stem not in only) or stem in done:
            continue
        if stop.exists():
            print(f"{now()}  runs/STOP found, stopping"); break
        head = read_header(config)
        task, model = head["task"], head["model"]
        venv = ".venv-zoe" if model in ZOE_VENV_MODELS else ".venv"
        batch = head.get("batch_size", 1)
        log = REPO / "logs" / f"smoke_{stem}_{time.strftime('%Y%m%d')}.log"
        while True:
            run_dir = smoke_root / stem
            shutil.rmtree(run_dir, ignore_errors=True)
            extra = SMOKE_FLAGS.get(task, SMOKE_FLAGS["default"]) + [
                "--batch-size", str(batch), "--num-workers", args.num_workers]
            print(f"{now()}  {stem} ({venv}, batch {batch}) ...", flush=True)
            t0 = time.time()
            rc = run_job(train_cmd(venv, str(config.relative_to(REPO)), args.data_root,
                                   run_dir, extra), log, env, args.timeout_min)
            mins = (time.time() - t0) / 60
            text = log.read_text(encoding="utf-8", errors="replace")
            tail = text[text.rfind("\n=== "):]
            if rc != 0 and OOM.search(tail) and batch > 1:
                print(f"{now()}  {stem} OOM at batch {batch}, halving", flush=True)
                batch //= 2
                continue
            ok = rc == 0 and (run_dir / "summary.json").exists()
            state = "ok" if ok else f"FAIL rc={rc}"
            reason = "" if ok else (tail.strip().splitlines() or [""])[-1][:200]
            print(f"{now()}  {stem} {state} batch={batch} {mins:.1f} min {reason}", flush=True)
            with tsv.open("a", encoding="utf-8") as fh:
                fh.write(f"{now()}\t{stem}\t{state}\t{batch}\t{mins:.1f}\t{venv}\t{task}\t{log.name}\t{reason}\n")
            break

    # The queue is rebuilt from every ok row, in task order, so re-running after more
    # configs pass simply extends it.
    done = load_done(tsv)
    lines = ["# generated by scripts/smoke_all.py: <venv> <config> <validation flags>"]
    for config in configs_for(TASK_ORDER):
        if config.stem not in done:
            continue
        head = read_header(config)
        venv = ".venv-zoe" if head["model"] in ZOE_VENV_MODELS else ".venv"
        flags = VALIDATION_FLAGS.get(head["task"], VALIDATION_FLAGS["default"])
        rel = config.relative_to(REPO).as_posix()
        lines.append(f"{venv} {rel} {' '.join(flags)} --batch-size {done[config.stem]} "
                     f"--num-workers {args.num_workers}")
    Path(REPO / args.queue).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{now()}  queue: {len(lines) - 1} jobs -> {args.queue}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
