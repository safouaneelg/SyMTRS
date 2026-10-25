"""Run a queue of `symtrs train` jobs so that it can be killed and restarted at any time.

Why this exists: the validation runs take a night, and the machine they run on has
blue-screened under load, so the queue keeps no state in memory:

- a job whose `<run_dir>/summary.json` exists AND says it succeeded (`run_verdict`) is done and
  is skipped; a summary of a run that failed, or that scored no test split, is not;
- a job with `<run_dir>/weights/last.pt` but no summary was interrupted and is resumed from it;
- a job some other process is already writing into (another queue over an overlapping entry
  list, a run started by hand) is skipped, not resumed: two writers on one run directory
  interleave `metrics.csv` and race on `weights/last.pt` (see `inflight`);
- `runs/STOP` (an empty file) makes the queue exit after the current job, cleanly;
- every start / end is appended to `logs/queue_<date>.log` and `<out_root>/status.tsv`, so the
  next session reads where things stand instead of guessing.

Queue file format, one job per line (blank lines and `#` comments ignored):

    <venv>  <config.yaml>  [extra symtrs flags ...]

A line may instead start with `group`, to run several compatible configs as one process on
one shared batch (`symtrs train-group`, see AGENTS.md):

    group  <venv>  <config.yaml>  <config.yaml>  [...]  [extra symtrs flags ...]

"Compatible" is `symtrs.engine.trainer.group_conflicts`: same task, resolution, patch size,
data and seed, and an effective batch (`batch_size x grad_accum`) that divides the largest
member's. The members need not share a micro-batch split - the shared loader yields the
largest window and each member cuts it into its own windows and micro-batches - so a `4x2`,
an `8x1` and a `4x1` config can be on the same line (the `4x1` steps twice per batch).
The configs end at the first `-` token; everything after it is passed to `train-group`
unchanged. Group members write to their own `<out_root>/<config stem>` like every other job,
so `symtrs tables` reads them the same way, and a member whose summary already scores is
dropped from the group on a restart rather than being retrained.

`<venv>` is `.venv` or `.venv-zoe` (midas_dpt_swin2_tiny and midas_dpt_beit_large need timm
0.6.12, see AGENTS.md ground rule 10). This script does not decide budgets: `scripts/smoke_all.py` measures
what fits on the card and writes them into the queue file. Launch it detached with
`scripts/run_queue.ps1` so it outlives the shell that started it.

`--jobs N --devices 0,1,2` runs N jobs at once, one per visible GPU: each job gets its
own `CUDA_VISIBLE_DEVICES`, so the run still sees a single device as `cuda:0` and no
config changes. The validation queue was written for one 12 GB card on Windows; this is
how it uses a multi-GPU Linux box without changing what a job does. A timeout kills the
job's whole process tree on both platforms, because the three subprocess front-ends
start Python children of their own and a killed parent would leave them holding the GPU.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import queue
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def parse_queue(path: Path) -> list[dict]:
    """Queue lines -> jobs. `{"venv", "configs", "group", "extra"}`, one config per plain job."""
    jobs = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = shlex.split(line, posix=True)
        if parts[0] == "group":
            if len(parts) < 4:
                raise SystemExit(
                    f"{path}: a group line needs a venv and at least two configs: {line!r}"
                )
            venv, rest = parts[1], parts[2:]
            # A config path never starts with `-`; the first flag starts the extras, whose
            # values (after `--set`, say) are then part of the extras too.
            cut = next((i for i, item in enumerate(rest) if item.startswith("-")), len(rest))
            configs = rest[:cut]
            if len(configs) < 2:
                raise SystemExit(
                    f"{path}: a group line needs at least two configs before the flags: {line!r}"
                )
            jobs.append({"venv": venv, "configs": configs, "group": True, "extra": rest[cut:]})
        else:
            jobs.append({"venv": parts[0], "configs": [parts[1]], "group": False,
                         "extra": parts[2:]})
    return jobs


def job_stem(job: dict) -> str:
    """The queue's name for a job, in logs and status.tsv."""
    return "+".join(Path(config).stem for config in job["configs"])


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


def inflight(run_dir: Path, config: str | None = None,
             proc_root: Path = Path("/proc")) -> str | None:
    """The command of a live process already writing `run_dir`, or None.

    "No summary.json, has last.pt" cannot tell an interrupted job from one another queue is
    running right now, and on 2026-10-25 two overlapping queues both read it the same way: the
    rest queue started a second `srcnn_x2` into the directory the tail queue was training, and
    `midas_small_256` was trained by two processes at once (its metrics.csv carries epochs
    10-19 twice). A run that something is visibly working in is not this queue's to start.

    Matching is on the command line, because the subprocess front-ends (domain adaptation,
    shadow removal, super-resolution) run their own Python children and the parent's
    `<run_dir>/weights/last.pt` is the only shared state. Both the spelling the queue uses
    (`runs/campaign2/srcnn_x2`) and its absolute form are accepted: a queue passes `--out-dir`
    as given, so the child's argv usually holds the relative one.
    """
    spellings = {str(run_dir), str(run_dir.resolve())}
    patterns = [re.compile(re.escape(item) + r"(?![\w.-])") for item in spellings]
    if config:
        # `train-group` names its members by config path and writes to `<out_root>/<stem>`, so
        # a member a live group is training has no `--out-dir` of its own to match on.
        patterns.append(re.compile(re.escape(Path(config).name) + r"(?![\w.-])"))
    try:
        entries = list(proc_root.iterdir())
    except OSError:
        return None
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue  # raced with the process exiting
        cmdline = raw.replace(b"\x00", b" ").decode("utf-8", "replace").strip()
        # A zombie has an empty cmdline; an unrelated tool that merely mentions the path
        # (`symtrs tables runs`, an editor) must not block a queued run.
        if not cmdline or "symtrs" not in cmdline:
            continue
        if any(pattern.search(cmdline) for pattern in patterns):
            return f"pid {entry.name}: {cmdline[:140]}"
    return None


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


def run_env(omp_threads: int = 0) -> dict:
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
    # Logs are redirected to a file, so stdout is block-buffered: a job that prints its epoch
    # every 50 steps showed nothing for the first ten minutes of a multi-hour 2048 run, and a
    # crashed job looked identical to a slow one.
    env.setdefault("PYTHONUNBUFFERED", "1")
    # OOM instead of a silent spill into system RAM on Windows (see symtrs/common/seed.py).
    env.setdefault("SYMTRS_CUDA_MEM_FRACTION", "0.92")
    # This laptop blue-screened under combined CPU+GPU load; keep the CPU side moderate.
    env.setdefault("OMP_NUM_THREADS", "8")
    env.setdefault("MKL_NUM_THREADS", "8")
    if omp_threads > 0:
        env["OMP_NUM_THREADS"] = str(omp_threads)
        env["MKL_NUM_THREADS"] = str(omp_threads)
    return env


def venv_python(venv: str) -> Path:
    """The interpreter of a venv, on either layout (`Scripts/python.exe` / `bin/python`)."""
    return REPO / venv / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")


def train_cmd(venv: str, config: str, data_root: str, run_dir: Path, extra: list[str],
              num_workers: int = 0) -> list[str]:
    cmd = [str(venv_python(venv)), "-m", "symtrs.cli", "train", config,
           "--data-root", data_root, "--out-dir", str(run_dir)]
    if num_workers > 0:
        # `--num-workers` is the documented flag for data.num_workers and reaches the
        # three subprocess front-ends too (the SR runner passes it on as `--num_workers`).
        cmd += ["--num-workers", str(num_workers)]
    return cmd + list(extra)


def group_cmd(venv: str, configs: list[str], data_root: str, out_root: Path,
              extra: list[str], num_workers: int = 0) -> list[str]:
    """`train-group` for the members still to do, writing beside the flat queue's runs.

    `--out-dir <out_root>` gives every member `<out_root>/<config stem>`, which is exactly
    where a plain job for that config writes; the queue's skip/resume checks do not have to
    know a run was grouped. `--resume-existing` is per member: each has its own `last.pt`,
    which one shared `--set train.resume` cannot name.
    """
    cmd = [str(venv_python(venv)), "-m", "symtrs.cli", "train-group", *configs,
           "--data-root", data_root, "--out-dir", str(out_root), "--resume-existing"]
    if num_workers > 0:
        cmd += ["--num-workers", str(num_workers)]
    return cmd + list(extra)


def run_job(cmd: list[str], log: Path, env: dict, timeout_min: float):
    """Run one job with its output appended to `log`; return the exit code or "timeout".

    On timeout the whole process TREE is killed: the subprocess front-ends (domain
    adaptation, super-resolution, shadow removal) start their own python children, and
    killing only the direct child leaves those holding the GPU for the next job. On POSIX
    the job is started in its own session so `killpg` reaches those children; on Windows
    `taskkill /T` does the same.
    """
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"\n=== {now()} {' '.join(cmd)}\n")
        fh.flush()
        proc = subprocess.Popen(
            cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=REPO,
            start_new_session=(os.name != "nt"),
        )
        try:
            # <= 0 means no timeout: a real benchmark run (20+ epochs at 2048) is hours long,
            # and the timeout exists to catch a hang, not to bound the work.
            return proc.wait(timeout=timeout_min * 60 if timeout_min > 0 else None)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
            proc.wait()
            return "timeout"


def env_for(base_env: dict, device: str) -> dict:
    """Per-job environment: at most one visible GPU, so a `device: cuda` config sees one."""
    env = dict(base_env)
    if device:
        env["CUDA_VISIBLE_DEVICES"] = device
    return env


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--queue", default="runs/queue.txt")
    ap.add_argument("--out-root", default="runs/validation")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--timeout-min", type=float, default=180.0,
                    help="kill a job that runs longer than this (a hung job must not eat the "
                         "night); 0 disables the timeout for real benchmark runs")
    ap.add_argument("--jobs", type=int, default=1,
                    help="how many jobs to run at once (default 1); each gets its own GPU")
    ap.add_argument("--devices", default="",
                    help="comma-separated CUDA device ids to pin jobs to, e.g. 0,1,2 "
                         "(default: 0..jobs-1 when --jobs > 1)")
    ap.add_argument("--num-workers", type=int, default=0,
                    help="data workers per job, overriding data.num_workers (default: keep the "
                         "config's). The 2048px path is memory-bandwidth bound: measured on this "
                         "box it saturates at ~9 samples/s with 8 threads and DROPS to 6.6 at 24, "
                         "so three concurrent jobs at 8 workers each decode less than one job "
                         "alone. Size the total (jobs x workers) to the PHYSICAL cores, not "
                         "`nproc`: this box has 18 physical / 36 logical, and 24 CPU-bound "
                         "workers took a 2048 step from 0.76 s to 1.2 s.")
    ap.add_argument("--omp-threads", type=int, default=0,
                    help="OMP/MKL threads per job (default: keep run_env's). With several jobs "
                         "on one box, torch's default of one thread per core multiplies into "
                         "oversubscription even before the data workers are counted")
    args = ap.parse_args()

    os.chdir(REPO)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (REPO / "logs").mkdir(exist_ok=True)
    day = dt.date.today().strftime("%Y%m%d")
    qlog = REPO / "logs" / f"queue_{day}.log"
    status = out_root / "status.tsv"
    stop = REPO / "runs" / "STOP"

    lock = threading.Lock()

    def note(msg: str) -> None:
        line = f"{now()}  {msg}"
        with lock:
            print(line, flush=True)
            with qlog.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    keep_awake()
    base_env = run_env(args.omp_threads)
    jobs = parse_queue(Path(args.queue))
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    if args.jobs > 1 and not devices:
        devices = [str(index) for index in range(args.jobs)]
    if args.jobs > 1 and len(devices) < args.jobs:
        note(f"only {len(devices)} device(s) for {args.jobs} jobs; they will share")
    note(f"queue start: {len(jobs)} jobs from {args.queue}"
         + (f", {args.jobs} at a time on cuda:{','.join(devices)}" if args.jobs > 1 else ""))

    def run_one(index: int, job: dict, device: str) -> None:
        venv, extra = job["venv"], job["extra"]
        tag = f"[{index}/{len(jobs)}] {job_stem(job)}"
        configs = job["configs"]

        if job["group"]:
            # A member that already scored is done; re-running it would only re-score its
            # test split (hours at 2048) and, worse, the group would otherwise retrain a
            # finished member from scratch.
            pending = [c for c in configs if not run_verdict(0, run_dir_for(out_root, c))[0]]
            if not pending:
                note(f"{tag} skip (every member has a scored summary.json)")
                return
            # A member another process is training is left alone; resuming it here would put
            # two writers on one weights/last.pt and interleave metrics.csv (see `inflight`).
            busy = {c: inflight(run_dir_for(out_root, c), c) for c in pending}
            if any(busy.values()):
                for config, where in busy.items():
                    if where:
                        note(f"{tag} {Path(config).stem} skip (already running: {where})")
                pending = [c for c in pending if not busy[c]]
                if not pending:
                    return
            if len(pending) < len(configs):
                note(f"{tag} {len(configs) - len(pending)} member(s) already scored; "
                     f"running the remaining {len(pending)}")
            if len(pending) == 1:
                # One member left is a plain run: the shared loader would have nothing to share.
                cmd = train_cmd(venv, pending[0], args.data_root,
                                run_dir_for(out_root, pending[0]), extra, args.num_workers)
                if (run_dir_for(out_root, pending[0]) / "weights" / "last.pt").exists():
                    cmd += ["--resume", str(run_dir_for(out_root, pending[0]) / "weights" / "last.pt")]
            else:
                cmd = group_cmd(venv, pending, args.data_root, out_root, extra, args.num_workers)
            log = REPO / "logs" / f"{job_stem(job)}_{day}.log"
            note(f"{tag} start group of {len(pending)}"
                 f"{f' on cuda:{device}' if device else ''} -> {log.name}")
            t0 = time.time()
            rc = run_job(cmd, log, env_for(base_env, device), args.timeout_min)
            mins = (time.time() - t0) / 60
            for config in pending:
                run_dir = run_dir_for(out_root, config)
                ok, why = run_verdict(rc, run_dir)
                state = "ok" if ok else f"FAIL rc={rc}" + (f" ({why})" if why != f"rc={rc}" else "")
                note(f"{tag} {Path(config).stem} {state} in {mins:.1f} min")
                with lock:
                    with status.open("a", encoding="utf-8") as fh:
                        fh.write(f"{now()}\t{Path(config).stem}\t{state}\t{mins:.1f}\t{log.name}\n")
            return

        config = configs[0]
        run_dir = run_dir_for(out_root, config)
        if run_verdict(0, run_dir)[0]:
            note(f"{tag} skip (summary.json exists and records a scored run)")
            return
        busy = inflight(run_dir, config)
        if busy:
            note(f"{tag} skip (already running: {busy})")
            return
        cmd = train_cmd(venv, config, args.data_root, run_dir, extra, args.num_workers)
        last = run_dir / "weights" / "last.pt"
        if last.exists():
            cmd += ["--resume", str(last)]
        # One device per job, so an unmodified config (device: cuda, gpus: 1) still sees a
        # single GPU and the runs cannot fight over memory.
        log = REPO / "logs" / f"{job_stem(job)}_{day}.log"
        note(f"{tag} start{' (resume)' if last.exists() else ''}"
             f"{f' on cuda:{device}' if device else ''} -> {log.name}")
        t0 = time.time()
        rc = run_job(cmd, log, env_for(base_env, device), args.timeout_min)
        mins = (time.time() - t0) / 60
        ok, why = run_verdict(rc, run_dir)
        state = "ok" if ok else f"FAIL rc={rc}" + (f" ({why})" if why != f"rc={rc}" else "")
        note(f"{tag} {state} in {mins:.1f} min")
        with lock:
            with status.open("a", encoding="utf-8") as fh:
                fh.write(f"{now()}\t{Path(config).stem}\t{state}\t{mins:.1f}\t{log.name}\n")

    if args.jobs <= 1:
        for index, item in enumerate(jobs, 1):
            if stop.exists():
                note("runs/STOP found, exiting before the next job")
                return 0
            run_one(index, item, devices[0] if devices else "")
    else:
        # A free-device pool rather than a fixed assignment: jobs differ by hours, and a
        # round-robin slot would leave one GPU idle while another finishes two short runs.
        free = queue.Queue()
        for device in devices:
            free.put(device)

        def work(pair: tuple) -> None:
            index, item = pair
            if stop.exists():
                note(f"[{index}/{len(jobs)}] {job_stem(item)} not started (runs/STOP)")
                return
            device = free.get()
            try:
                run_one(index, item, device)
            finally:
                free.put(device)

        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            list(pool.map(work, enumerate(jobs, 1)))
    note("queue finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
