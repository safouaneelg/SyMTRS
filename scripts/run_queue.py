"""Run a queue of `symtrs train` jobs so that it can be killed and restarted at any time.

Why this exists: the validation runs take a night, and the machine they run on has
blue-screened under load, so the queue keeps no state in memory:

- a job whose `<run_dir>/summary.json` exists AND says it succeeded (`run_verdict`) is done and
  is skipped; a summary of a run that failed, or that scored no test split, is not;
- a job with `<run_dir>/weights/last.pt` but no summary was interrupted and is resumed from it;
- `runs/STOP` (an empty file) makes the queue exit after the current job, cleanly;
- every start / end is appended to `logs/queue_<date>.log` and `<out_root>/status.tsv`, so the
  next session reads where things stand instead of guessing.

Queue file format, one job per line (blank lines and `#` comments ignored):

    <venv>  <config.yaml>  [extra symtrs flags ...]

`<venv>` is `.venv` or `.venv-zoe` (ZoeDepth, midas_dpt_swin2_tiny and midas_dpt_beit_large need timm
0.6.12, see AGENTS.md ground rule 10). This script does not decide budgets: `scripts/smoke_all.py` measures
what fits on the card and writes them into the queue file. Launch it detached with
`scripts/run_queue.ps1` so it outlives the shell that started it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def parse_queue(path: Path) -> list[tuple[str, str, list[str]]]:
    jobs = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = shlex.split(line, posix=True)
        jobs.append((parts[0], parts[1], parts[2:]))
    return jobs


def run_dir_for(out_root: Path, config: str) -> Path:
    # One folder per config file: the stem is unique across configs/ (e.g. midas_small_512,
    # srcnn_x4, cyclegan_day2night), and keeping the name lets `symtrs tables` read them all.
    return out_root / Path(config).stem


def run_verdict(rc, run_dir: Path, eval_only: bool = False) -> tuple:
    """`(ok, why)`: did this job really finish, judged by what it left behind.

    A job is `ok` only if ALL of these hold:

    * its exit code is 0;
    * `<run_dir>/summary.json` exists and parses;
    * the summary has no `returncode`, or it is 0 (the super-resolution front-end used to write
      `returncode: 1` into summary.json and exit 0, so 13 failed smoke runs were recorded `ok`
      and the queue would have skipped them forever);
    * the summary has a non-empty `test` dict, unless `eval_only` (a run that trained and
      scored nothing is not a result; `symtrs tables` leaves such a row blank on purpose).

    `rc` may be "timeout" or any non-zero value; `why` names the first condition that failed so
    that smoke.tsv and status.tsv say it instead of a bare "FAIL".
    """
    if rc != 0:
        return False, f"rc={rc}"
    path = run_dir / "summary.json"
    if not path.is_file():
        return False, "no summary.json"
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False, "summary.json unreadable"
    if not isinstance(summary, dict):
        return False, "summary.json is not an object"
    if summary.get("returncode", 0) != 0:
        return False, f"summary returncode={summary['returncode']}"
    if not eval_only and not (isinstance(summary.get("test"), dict) and summary["test"]):
        return False, "summary has no test metrics"
    return True, ""


def keep_awake() -> None:
    """Stop Windows from idle-sleeping while this process runs (no setting is changed).

    The first smoke pass started at 19:42 and the laptop went into standby at ~20:00 until
    someone touched it at 03:39: 7.6 h lost, and the job in flight reported 460 min. This
    only blocks *idle* sleep; closing the lid still follows the lid policy.
    """
    if os.name == "nt":
        import ctypes

        es_continuous, es_system_required = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required)


def run_env() -> dict:
    """Environment for every training subprocess (queue and smoke tests alike)."""
    env = dict(os.environ)
    # Weights were prefetched into the repo-local cache; offline mode makes a model that was
    # missed fail in seconds instead of hanging on a slow connection.
    env.setdefault("TORCH_HOME", str(REPO / ".cache" / "torch"))
    env.setdefault("HF_HOME", str(REPO / ".cache" / "huggingface"))
    env.setdefault("XDG_CACHE_HOME", str(REPO / ".cache" / "xdg"))
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    # OOM instead of a silent spill into system RAM on Windows (see symtrs/common/seed.py).
    env.setdefault("SYMTRS_CUDA_MEM_FRACTION", "0.92")
    # This laptop blue-screened under combined CPU+GPU load; keep the CPU side moderate.
    env.setdefault("OMP_NUM_THREADS", "8")
    env.setdefault("MKL_NUM_THREADS", "8")
    return env


def train_cmd(venv: str, config: str, data_root: str, run_dir: Path, extra: list[str]) -> list[str]:
    return [str(REPO / venv / "Scripts" / "python.exe"), "-m", "symtrs.cli", "train", config,
            "--data-root", data_root, "--out-dir", str(run_dir), *extra]


def run_job(cmd: list[str], log: Path, env: dict, timeout_min: float):
    """Run one job with its output appended to `log`; return the exit code or "timeout".

    On timeout the whole process TREE is killed: the subprocess front-ends (domain
    adaptation, super-resolution, shadow removal) start their own python children, and on
    Windows killing only the direct child leaves those holding the GPU for the next job.
    """
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"\n=== {now()} {' '.join(cmd)}\n")
        fh.flush()
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=REPO)
        try:
            return proc.wait(timeout=timeout_min * 60)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                proc.kill()
            proc.wait()
            return "timeout"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--queue", default="runs/queue.txt")
    ap.add_argument("--out-root", default="runs/validation")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--timeout-min", type=float, default=180.0,
                    help="kill a job that runs longer than this (a hung job must not eat the night)")
    args = ap.parse_args()

    os.chdir(REPO)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (REPO / "logs").mkdir(exist_ok=True)
    day = dt.date.today().strftime("%Y%m%d")
    qlog = REPO / "logs" / f"queue_{day}.log"
    status = out_root / "status.tsv"
    stop = REPO / "runs" / "STOP"

    def note(msg: str) -> None:
        line = f"{now()}  {msg}"
        print(line, flush=True)
        with qlog.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    keep_awake()
    env = run_env()
    jobs = parse_queue(Path(args.queue))
    note(f"queue start: {len(jobs)} jobs from {args.queue}")
    for i, (venv, config, extra) in enumerate(jobs, 1):
        if stop.exists():
            note("runs/STOP found, exiting before the next job")
            return 0
        run_dir = run_dir_for(out_root, config)
        tag = f"[{i}/{len(jobs)}] {Path(config).stem}"
        if run_verdict(0, run_dir)[0]:
            note(f"{tag} skip (summary.json exists and records a scored run)")
            continue
        cmd = train_cmd(venv, config, args.data_root, run_dir, extra)
        last = run_dir / "weights" / "last.pt"
        if last.exists():
            cmd += ["--resume", str(last)]
        log = REPO / "logs" / f"{Path(config).stem}_{day}.log"
        note(f"{tag} start{' (resume)' if last.exists() else ''} -> {log.name}")
        t0 = time.time()
        rc = run_job(cmd, log, env, args.timeout_min)
        mins = (time.time() - t0) / 60
        ok, why = run_verdict(rc, run_dir)
        state = "ok" if ok else f"FAIL rc={rc}" + (f" ({why})" if why != f"rc={rc}" else "")
        note(f"{tag} {state} in {mins:.1f} min")
        with status.open("a", encoding="utf-8") as fh:
            fh.write(f"{now()}\t{Path(config).stem}\t{state}\t{mins:.1f}\t{log.name}\n")
    note("queue finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
