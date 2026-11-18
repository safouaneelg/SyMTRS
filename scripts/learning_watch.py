"""Is every run of a campaign still learning? A periodic check that flags the ones that are not.

    python scripts/learning_watch.py runs/campaign2                 # one report, flags first
    python scripts/learning_watch.py runs/campaign2 --every 30      # repeat every 30 min

Why this exists: a multi-day campaign is only checked by hand when someone looks, and the
failures that cost days are quiet - a model whose loss stops moving, a NaN that is skipped
batch after batch, a validation score that never beats its first value, a job that hangs.
`scripts/overfit_check.py` answers "can this config learn at all" before a campaign; this
answers "is it still learning" during one, from the same files (`metrics.csv`, the GAN
`loss_log.txt`, the shadow worker's rows) read by the same `_judge`.

Flags (a run can have several):
  NONFINITE  a non-finite loss was logged (the trainer skips the batch, so the run goes on)
  RISING     after >= 2 epochs, the last epoch's training loss is above the first's
  FLAT       after >= 3 epochs, the training loss fell by less than 5 %
  VAL-FLAT   after >= 2 scored validations, the best is within 1 % of the first
  STALE      no new epoch for 3x the run's mean epoch time (hung, or its job died)

Each report is appended to `<runs>/learning_watch.log`, flagged runs first. What this does
not do: judge whether a number is good. A run with no flag is learning; how well is the
results table's job.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from overfit_check import _judge  # noqa: E402  - one reader for every task's metrics

REPO = Path(__file__).resolve().parents[1]
FLAT_RATIO, VAL_FLAT, STALE_FACTOR = 0.95, 0.01, 3.0


def _epochs_and_times(run: Path):
    """(completed epochs, mean seconds per epoch, last update time) from metrics.csv."""
    path = run / "metrics.csv"
    if not path.exists():
        return 0, None, None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if rows and "split" in rows[0]:  # shadow worker: iteration rows, not epochs
        rows = [r for r in rows if r.get("split") == "train"]
    seconds = []
    for row in rows:
        try:
            seconds.append(float(row.get("seconds") or "nan"))
        except ValueError:
            pass
    seconds = [s for s in seconds if math.isfinite(s) and s > 0]
    return len(rows), (sum(seconds) / len(seconds) if seconds else None), path.stat().st_mtime


def _non_finite(run: Path) -> int:
    """Non-finite-loss warnings in the run's newest log, if the queue wrote one for it."""
    logs = [p for p in (REPO / "logs").glob(f"*{run.name}*.log") if p.is_file()]
    if not logs:
        return 0
    # Only the newest log file, not the sum over every file the name matches: a run keeps a
    # log per day it was launched on, so summing them re-counted the float16 warnings of
    # 2026-10-22 against the bfloat16 runs that replaced them and flagged two healthy runs
    # every four hours for days (midas_dpt_beit_large_512: 114 hits, all in _20261022.log;
    # the 2048 group: 12052, all in its 10-22 section). A log that stopped being written is
    # history, not evidence about the run as it stands.
    log = max(logs, key=lambda p: p.stat().st_mtime)
    text = log.read_text(encoding="utf-8", errors="replace")
    # Only the newest launch inside that file: the queue appends every restart to the same log
    # under a "=== <time> <command>" header, and a crashed earlier attempt's warnings (7074
    # from the float16 MiDaS run) must not flag the run that replaced it.
    text = text.rsplit("\n=== ", 1)[-1]
    # Only this run's section of a shared group log: the lines that name it.
    return sum(1 for line in text.splitlines()
               if "non-finite" in line and (f"[{run.name.rsplit('_', 1)[0]}]" in line
                                            or "+" not in log.name))


def check(root: Path) -> list:
    now = time.time()
    out = []
    for run in sorted(p for p in root.iterdir() if p.is_dir()):
        epochs, mean_s, updated = _epochs_and_times(run)
        if not epochs and not (run / "checkpoints").exists():
            continue
        done = (run / "summary.json").exists()
        signal, first, last, v0, vbest, note = _judge(run)
        flags = []
        if _non_finite(run):
            flags.append("NONFINITE")
        if first is not None and last is not None and epochs >= 2 and last > first:
            flags.append("RISING")
        if (first and last is not None and epochs >= 3 and first > 0
                and last / first > FLAT_RATIO):
            flags.append("FLAT")
        # With val_every=2, four epochs is two scored validations.
        if (v0 is not None and vbest is not None and epochs >= 4
                and abs(vbest - v0) <= VAL_FLAT * abs(v0)):
            flags.append("VAL-FLAT")
        if not done and mean_s and updated and now - updated > STALE_FACTOR * mean_s:
            flags.append("STALE")
        fmt = lambda v: "-" if v is None else f"{v:.4g}"  # noqa: E731
        out.append((flags, f"{run.name:<28} ep {epochs:>3}{' done' if done else ''}  "
                           f"train {fmt(first)} -> {fmt(last)}  val {fmt(v0)} -> best {fmt(vbest)}"
                           f"  ({note})"))
    return out


def report(root: Path) -> str:
    rows = check(root)
    flagged = [r for r in rows if r[0]]
    lines = [f"=== {datetime.now():%Y-%m-%d %H:%M}  {len(rows)} runs, {len(flagged)} flagged"]
    lines += [f"FLAG {','.join(f):<18} {text}" for f, text in flagged]
    lines += [f"ok   {'':<18} {text}" for f, text in rows if not f]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="?", default="runs/campaign2")
    ap.add_argument("--every", type=float, default=0, help="minutes between reports (0 = once)")
    args = ap.parse_args()
    root = Path(args.runs)
    while True:
        text = report(root)
        print(text, flush=True)
        with (root / "learning_watch.log").open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")
        if not args.every:
            return
        time.sleep(args.every * 60)


if __name__ == "__main__":
    main()
