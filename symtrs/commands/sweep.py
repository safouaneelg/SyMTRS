"""`symtrs sweep` - the full run matrix.

    symtrs sweep --task depth --data-root /data/SyMTRS --out-dir runs --plan runs/depth_plan.csv
    symtrs sweep --task depth --data-root /data/SyMTRS --execute --jobs 1

The plan is written before anything runs, so the matrix can be reviewed (and pasted
into a paper's experiment section) independently of whether it has been executed. Each
entry is launched as a separate `python -m symtrs train` process rather than in-process:
CUDA memory from a failed 2048 run is not reliably released, and one OOM should not take
the remaining forty runs down with it.

Re-running is safe. An entry whose run directory already contains `summary.json` is
skipped unless `--force` is given.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from symtrs.config import RESOLUTIONS
from symtrs.registry import model_keys

#: Tasks that exist only at native resolution, because the domain labels (day/night)
#: are not present in the downsampled folders.
NATIVE_ONLY_TASKS = ("domain_adapt", "shadow_removal")

#: Tasks with no released weights, so a zero-shot entry has nothing to load. `symtrs eval`
#: re-scores the run's own checkpoint for these instead, which is the finetune half of the
#: row - a zero-shot entry would only fail with "there is no checkpoint at ...".
NO_ZERO_SHOT_TASKS = ("superresolution", "shadow_removal")


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser("sweep", help="plan and run the full model x resolution grid")
    parser.add_argument("--task", required=True, help="depth, segmentation, domain_adapt, superresolution or shadow_removal")
    parser.add_argument("--data-root", dest="data_root", required=True)
    parser.add_argument("--out-dir", dest="out_dir", default="runs")
    parser.add_argument("--models", default="all",
                        help="'all', or a comma-separated list of registry keys")
    parser.add_argument("--resolutions", default="2048,1024,512,256",
                        help="comma-separated, or 'all'")
    parser.add_argument("--modes", default="zero_shot,finetune",
                        help="zero_shot, finetune, or both")
    parser.add_argument("--plan", default="", help="where to write the plan CSV")
    parser.add_argument("--execute", action="store_true", help="actually run the plan")
    parser.add_argument("--jobs", type=int, default=1,
                        help="how many entries to run at once; 1 for a single GPU")
    parser.add_argument("--force", action="store_true", help="re-run entries that already finished")
    parser.add_argument("--extra", action="append", default=None, metavar="FLAG=VALUE",
                        help="passed to every run as --set FLAG=VALUE")
    parser.set_defaults(handler=_run)


def build_plan(
    task: str,
    models: List[str],
    resolutions: List[int],
    modes: List[str],
) -> List[Dict[str, str]]:
    """The run matrix as a list of dicts, one per run."""
    if task in NATIVE_ONLY_TASKS:
        if [r for r in resolutions if r != 2048]:
            print(
                f"[symtrs] {task} only exists at 2048: the rgb_night folders are not "
                "downsampled, so the lower resolutions are dropped from the plan"
            )
        resolutions = [2048]

    if task in NO_ZERO_SHOT_TASKS and "zero_shot" in modes:
        print(
            f"[symtrs] {task} has no released weights to evaluate zero-shot, so the "
            "zero_shot entries are dropped from the plan; use --modes finetune"
        )
        modes = [mode for mode in modes if mode != "zero_shot"]

    plan: List[Dict[str, str]] = []
    for model in models:
        for resolution in resolutions:
            for mode in modes:
                plan.append(
                    {"task": task, "model": model, "resolution": str(resolution), "mode": mode}
                )
    return plan


def _entry_command(entry: Dict[str, str], args: argparse.Namespace) -> List[str]:
    subcommand = "eval" if entry["mode"] == "zero_shot" else "train"
    command = [
        sys.executable, "-m", "symtrs", subcommand,
        "--task", entry["task"],
        "--model", entry["model"],
        "--resolution", entry["resolution"],
        "--data-root", args.data_root,
        "--out-dir", str(Path(args.out_dir) / entry["mode"] / entry["task"]
                         / entry["model"] / entry["resolution"]),
    ]
    for item in args.extra or []:
        command += ["--set", item]
    return command


def _already_done(entry: Dict[str, str], args: argparse.Namespace) -> bool:
    run_dir = Path(args.out_dir) / entry["mode"] / entry["task"] / entry["model"] / entry["resolution"]
    return (run_dir / "summary.json").is_file()


def _run(args: argparse.Namespace) -> int:
    models = model_keys(args.task) if args.models == "all" else [
        item.strip() for item in args.models.split(",") if item.strip()
    ]
    if not models:
        print(f"[symtrs] no models registered for task {args.task!r}")
        return 1

    resolutions = (
        list(RESOLUTIONS) if args.resolutions == "all"
        else [int(item) for item in args.resolutions.split(",") if item.strip()]
    )
    unknown = [r for r in resolutions if r not in RESOLUTIONS]
    if unknown:
        print(f"[symtrs] unknown resolutions {unknown}; expected a subset of {RESOLUTIONS}")
        return 1

    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    bad = [m for m in modes if m not in ("zero_shot", "finetune")]
    if bad:
        print(f"[symtrs] unknown modes {bad}; expected zero_shot and/or finetune")
        return 1

    plan = build_plan(args.task, models, resolutions, modes)
    pending = [entry for entry in plan if args.force or not _already_done(entry, args)]
    if not args.force:
        print(f"[symtrs] {len(plan) - len(pending)} of {len(plan)} entries already have a summary.json")

    plan_path = args.plan or str(Path(args.out_dir) / f"{args.task}_plan.csv")
    Path(plan_path).parent.mkdir(parents=True, exist_ok=True)
    with open(plan_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task", "model", "resolution", "mode"])
        writer.writeheader()
        writer.writerows(plan)
    print(f"[symtrs] wrote {len(plan)} entries to {plan_path}")

    for entry in plan:
        print(
            f"  {'done ' if _already_done(entry, args) else '     '}"
            f"{entry['mode']:<10} {entry['task']:<15} {entry['model']:<24} {entry['resolution']:>4}"
        )

    if not args.execute:
        print("[symtrs] this was a plan only; add --execute to run it")
        return 0

    failures: List[Dict[str, str]] = []
    jobs = max(1, args.jobs)
    if jobs == 1:
        for index, entry in enumerate(pending):
            if not _launch(index, len(pending), entry, args):
                failures.append(entry)
    else:
        # `jobs` above 1 means several runs share the GPUs. That is a throughput choice,
        # not a memory one: each run still gets its own process and its own CUDA context.
        # If a run OOMs, lower `jobs` before lowering the batch size.
        from concurrent.futures import ThreadPoolExecutor

        print(f"[symtrs] running {jobs} entries at a time")
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            outcomes = list(
                pool.map(
                    lambda pair: (pair[1], _launch(pair[0], len(pending), pair[1], args)),
                    enumerate(pending),
                )
            )
        failures = [entry for entry, ok in outcomes if not ok]

    print(f"\n[symtrs] {len(pending) - len(failures)}/{len(pending)} entries succeeded")
    for entry in failures:
        print(f"[symtrs] failed: {entry}")
    print(f"[symtrs] collect the results with: symtrs tables {args.out_dir} --task {args.task}")
    return 1 if failures else 0


def _launch(index: int, total: int, entry: Dict[str, str], args: argparse.Namespace) -> bool:
    command = _entry_command(entry, args)
    print(f"\n[symtrs] [{index + 1}/{total}] {' '.join(command)}")
    completed = subprocess.run(command)
    if completed.returncode != 0:
        print(f"[symtrs] entry failed with exit code {completed.returncode}")
        return False
    return True


__all__ = ["add_parser", "build_plan"]
