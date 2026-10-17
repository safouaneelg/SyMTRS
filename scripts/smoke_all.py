"""Smoke-test every generated config on the real dataset, find a batch size that fits the
card, and write the validation queue (`runs/queue.txt`) for `scripts/run_queue.py`.

Why the batch search: the generated configs carry batch sizes chosen for large cards
(U-Net 512 at batch 32 wants 18.6 GB). With `SYMTRS_CUDA_MEM_FRACTION` set (run_env does),
an oversized batch raises OOM within seconds instead of spilling into system RAM, so each
config is tried at its own batch size and halved on OOM until it fits. What was chosen is
recorded per config in `runs/smoke/smoke.tsv` and carried into the queue as `--batch-size`;
the YAMLs are not edited (they stay the reference grid for the full dataset).

The *effective* batch (batch x grad_accum) is kept equal to the grid's: halving the batch
doubles `--grad-accum` (depth and segmentation, the tasks whose loop accumulates), so the
queue trains with the same optimisation step as the grid on a smaller card. Halving without
that would silently change the effective batch at an unchanged LR. For the tasks that do not
accumulate (super-resolution, shadow removal, domain adaptation) the effective batch does
shrink, and the row says so (`eff_batch` < `grid_eff_batch`). An OOM raised in whole-image
validation/test scoring (eval batch is 1) is not cured by a smaller train batch and is
recorded as `FAIL ... eval OOM` at once instead of being halved down to 1.

smoke.tsv columns: when, config, state, batch, minutes, venv, task, log, reason, then
(appended, so rows written by older versions still parse) accum, eff_batch, grid_eff_batch.

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
from run_queue import REPO, keep_awake, now, run_env, run_job, run_verdict, train_cmd  # noqa: E402

TASK_ORDER = ["segmentation", "depth", "superresolution", "shadow_removal", "domain_adapt"]
# Models whose upstream code pins timm 0.6.12 (AGENTS.md ground rule 10).
ZOE_VENV_MODELS = (
    "zoedepth_n",
    "zoedepth_k",
    "zoedepth_nk",
    "midas_dpt_swin2_tiny",
    "midas_dpt_beit_large",
)
OOM = re.compile(r"OutOfMemoryError|CUDA out of memory|CUBLAS_STATUS_ALLOC_FAILED")
# Tasks whose training loop honours train.grad_accum (symtrs/engine/loop.py). The others
# ignore the key, so doubling it there would change nothing and would only misreport.
ACCUM_TASKS = ("depth", "segmentation")
# Traceback frames ("File ..., line N, in NAME") that say where an OOM was raised. The
# innermost frame in either list decides; `forward_native` is deliberately in neither
# because both the training step and evaluation call it.
TRAIN_FRAMES = ("train_one_epoch", "_training_step")
# `eval_one_epoch` / `evaluate_on_test` are the legacy super-resolution trainer's: its traceback is
# in the same log (the front-end inherits stdout), and without them an OOM there read as
# "unknown" and was halved like a training OOM, which cannot cure it.
EVAL_FRAMES = ("evaluate", "evaluate_depth", "evaluate_segmentation", "validate_crops", "predict",
               "eval_one_epoch", "evaluate_on_test", "super_resolve")
FRAME = re.compile(r'File "[^"]*", line \d+, in (\w+)')

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
# Shadow removal is fine-tuned from the authors' released weights (decision 2026-10-16):
# 500 iterations from scratch leave ShadowFormer far from even the identity mapping (test PSNR
# ~11 on the dummy), so a from-scratch number would say nothing about the model. All three
# load fully (274/274, 406/406, 759/759 tensors, see docs/claude/PROGRESS.md).
# ShadowMaskFormer is deliberately absent: the authors release no weights, so it is always
# trained from scratch (runs/queue_shadowmaskformer.txt has its line, without init_weights).
SHADOW_INIT = {
    "shadowformer": ".cache/third_party/ShadowFormer/istd_plus.pth",
    "restormer": ".cache/third_party/Restormer/real_denoising.pth",
    "uformer": ".cache/third_party/Uformer/Uformer_B_SIDD.pth",
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
        if section == "train" and key == "grad_accum":
            out["grad_accum"] = int(value)
    return out


def classify_oom(tail: str) -> str:
    """Where the last OOM in a job's log was raised: `train`, `eval` or `unknown`.

    Reads the last traceback ending in an OOM message and takes its innermost training or
    evaluation frame (see TRAIN_FRAMES / EVAL_FRAMES). `unknown` (no traceback, e.g. a
    subprocess that only printed the message) is treated as a training OOM by the caller,
    which is the old behaviour: halve and retry.
    """
    matches = list(OOM.finditer(tail))
    if not matches:
        return "unknown"
    before = tail[: matches[-1].start()]
    start = before.rfind("Traceback (most recent call last)")
    block = before[start:] if start >= 0 else ""
    decisive = [m for m in FRAME.findall(block) if m in TRAIN_FRAMES or m in EVAL_FRAMES]
    if not decisive:
        return "unknown"
    return "train" if decisive[-1] in TRAIN_FRAMES else "eval"


def halve(batch: int, accum: int, effective: int, task: str) -> tuple[int, int]:
    """Next (batch, accum) after an OOM: half the batch, same effective batch where possible."""
    new_batch = max(1, batch // 2)
    if task in ACCUM_TASKS:
        return new_batch, max(accum, -(-effective // new_batch))  # ceil, never below current
    return new_batch, accum


def configs_for(tasks: list[str]) -> list[Path]:
    found = []
    for task in tasks:
        found += sorted((REPO / "configs" / task).rglob("*.yaml"))
    return found


def load_rows(tsv: Path) -> dict:
    """`{config stem: {"batch": int, "accum": int | None}}` for every `ok` row.

    Accepts both the 9-column rows older versions wrote (accum unknown: None, meaning "the
    config's own value was used") and the 12-column ones.
    """
    done = {}
    if tsv.exists():
        for line in tsv.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) >= 5 and parts[2] == "ok":
                accum = None
                if len(parts) >= 10 and parts[9].strip().isdigit():
                    accum = int(parts[9])
                done[parts[1]] = {"batch": int(parts[3]), "accum": accum}
    return done


def load_done(tsv: Path) -> dict:
    """`{config stem: batch}` for every `ok` row (kept for callers of the old shape)."""
    return {stem: row["batch"] for stem, row in load_rows(tsv).items()}


def verified(rows: dict, smoke_root: Path) -> dict:
    """Keep only the `ok` rows whose run directory still passes `run_verdict`.

    smoke.tsv is a ledger, not proof: rows written before the verdict existed recorded 13 failed
    super-resolution runs as `ok`, and `done` would skip them for ever. Re-checking the summary
    on disk makes such a row re-run without anyone editing the ledger by hand.
    """
    return {stem: row for stem, row in rows.items() if run_verdict(0, smoke_root / stem)[0]}


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
    done = {stem: row["batch"] for stem, row in verified(load_rows(tsv), smoke_root).items()}
    keep_awake()
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
        accum = head.get("grad_accum", 1)
        effective = batch * accum  # the grid's effective batch, preserved across halvings
        log = REPO / "logs" / f"smoke_{stem}_{time.strftime('%Y%m%d')}.log"
        while True:
            run_dir = smoke_root / stem
            shutil.rmtree(run_dir, ignore_errors=True)
            extra = SMOKE_FLAGS.get(task, SMOKE_FLAGS["default"]) + [
                "--batch-size", str(batch), "--num-workers", args.num_workers]
            if task in ACCUM_TASKS:
                extra += ["--grad-accum", str(accum)]
            print(f"{now()}  {stem} ({venv}, batch {batch} x accum {accum}) ...", flush=True)
            t0 = time.time()
            rc = run_job(train_cmd(venv, str(config.relative_to(REPO)), args.data_root,
                                   run_dir, extra), log, env, args.timeout_min)
            mins = (time.time() - t0) / 60
            text = log.read_text(encoding="utf-8", errors="replace")
            tail = text[text.rfind("\n=== "):]
            oom = rc != 0 and OOM.search(tail) is not None
            where = classify_oom(tail) if oom else ""
            if oom and where != "eval" and batch > 1:
                new_batch, new_accum = halve(batch, accum, effective, task)
                print(f"{now()}  {stem} OOM at batch {batch} x accum {accum}, "
                      f"halving to {new_batch} x {new_accum}", flush=True)
                batch, accum = new_batch, new_accum
                continue
            # Not just "exit 0 and a summary exists": a super-resolution run that crashed used to
            # leave a summary with returncode 1 behind and exit 0, and 13 such runs were recorded
            # `ok` (see run_queue.run_verdict, which also demands a non-empty `test`).
            ok, why = run_verdict(rc, run_dir)
            state = "ok" if ok else f"FAIL rc={rc}"
            if ok:
                reason = ""
            elif rc == 0:
                reason = why  # exit 0 but not a finished run: say which check failed
            elif where == "eval":
                # Evaluation runs at batch 1 on whole frames: a smaller train batch cannot
                # fix it, so do not burn halvings (each a full re-run) down to 1.
                reason = "eval OOM"
            else:
                reason = (tail.strip().splitlines() or [""])[-1][:200]
            reason = reason.replace("\t", " ")
            print(f"{now()}  {stem} {state} batch={batch} accum={accum} {mins:.1f} min {reason}", flush=True)
            with tsv.open("a", encoding="utf-8") as fh:
                fh.write(f"{now()}\t{stem}\t{state}\t{batch}\t{mins:.1f}\t{venv}\t{task}\t{log.name}\t{reason}"
                         f"\t{accum}\t{batch * (accum if task in ACCUM_TASKS else 1)}"
                         f"\t{effective if task in ACCUM_TASKS else head.get('batch_size', 1)}\n")
            break

    # The queue is rebuilt from every ok row, in task order, so re-running after more
    # configs pass simply extends it.
    done = verified(load_rows(tsv), smoke_root)
    lines = ["# generated by scripts/smoke_all.py: <venv> <config> <validation flags>"]
    for config in configs_for(TASK_ORDER):
        if config.stem not in done:
            continue
        head = read_header(config)
        venv = ".venv-zoe" if head["model"] in ZOE_VENV_MODELS else ".venv"
        flags = list(VALIDATION_FLAGS.get(head["task"], VALIDATION_FLAGS["default"]))
        if head["model"] in SHADOW_INIT:
            flags += ["--set", f"shadow.init_weights={SHADOW_INIT[head['model']]}"]
        rel = config.relative_to(REPO).as_posix()
        row = done[config.stem]
        batch, accum = row["batch"], row["accum"]
        accum_flag = ""
        if accum is not None and head["task"] in ACCUM_TASKS:
            accum_flag = f" --grad-accum {accum}"
        if head["task"] in ACCUM_TASKS:
            grid = head.get("batch_size", 1) * head.get("grad_accum", 1)
            effective = batch * (accum if accum is not None else head.get("grad_accum", 1))
        else:
            grid, effective = head.get("batch_size", 1), batch
        note = f"  # eff_batch={effective} grid_eff_batch={grid}" + (" (SHRUNK)" if effective < grid else "")
        lines.append(f"{venv} {rel} {' '.join(flags)} --batch-size {batch}{accum_flag} "
                     f"--num-workers {args.num_workers}{note}")
    Path(REPO / args.queue).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{now()}  queue: {len(lines) - 1} jobs -> {args.queue}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
