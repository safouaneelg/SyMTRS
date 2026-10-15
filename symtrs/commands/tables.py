"""`symtrs tables` - collect finished runs into one table.

    symtrs tables runs/ --format markdown
    symtrs tables runs/ --metric depth --split test --csv results/depth.csv

Walks a runs directory for `summary.json` files and prints the test metrics of every
run it finds. This is the step that turns a directory full of run folders into the
table that goes in the paper, and doing it by hand from several dozen `metrics.csv`
files is how transcription errors get into a results section.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

#: Columns that identify a run rather than measure it.
IDENTITY = ("task", "model", "resolution", "mode", "best_epoch")

#: Metric order per task, so two tables of the same task line up.
COLUMN_ORDER = {
    "depth": ("rmse", "mae", "abs_rel", "sq_rel", "d1", "d2", "d3", "silog"),
    "segmentation": ("miou", "pixel_acc", "mean_acc", "mean_dice", "fwiou", "loss"),
    "domain_adapt": ("fid", "kid", "kid_std"),
}


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser("tables", help="collect run summaries into one table")
    parser.add_argument("root", nargs="?", default="runs", help="directory to search (default: runs)")
    parser.add_argument("--task", default=None, help="only runs of this task")
    parser.add_argument("--model", default=None, help="only runs whose model key contains this")
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--metric", default=None,
                        help="restrict the columns to one metric group: depth, segmentation, domain_adapt")
    parser.add_argument("--format", choices=["markdown", "csv", "json"], default="markdown")
    parser.add_argument("--out", default=None, help="write to a file instead of stdout")
    parser.add_argument("--sort-by", dest="sort_by", default="", help="column to sort on")
    parser.set_defaults(handler=_run)


def collect(root: str) -> List[Dict[str, Any]]:
    """Every run summary under `root`, one row each."""
    rows: List[Dict[str, Any]] = []
    for path in sorted(Path(root).rglob("summary.json")):
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[symtrs] skipping unreadable {path}: {exc}")
            continue
        if not isinstance(summary, dict) or "test" not in summary:
            continue
        row: Dict[str, Any] = {
            "run": str(path.parent),
            "task": summary.get("task", ""),
            "model": summary.get("model", ""),
            "resolution": summary.get("resolution", ""),
            "mode": summary.get("mode", ""),
            "best_epoch": summary.get("best_epoch", ""),
        }
        for key, value in (summary.get("test") or {}).items():
            row[key] = value
        rows.append(row)
    return rows


def _columns(rows: List[Dict[str, Any]], metric: Optional[str]) -> List[str]:
    identity = [name for name in ("run", "task", "model", "resolution", "mode") if any(name in r for r in rows)]
    if metric:
        measured = [name for name in COLUMN_ORDER.get(metric, ()) if any(name in r for r in rows)]
    else:
        seen: List[str] = []
        for row in rows:
            for name in row:
                if name not in identity and name not in seen:
                    seen.append(name)
        measured = sorted(seen, key=lambda n: (
            next((i for i, names in enumerate(COLUMN_ORDER.values()) if n in names), 99), n
        ))
    return identity + measured


def _format(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}" if abs(value) < 1e4 else f"{value:.3g}"
    return str(value)


def render(rows: List[Dict[str, Any]], columns: List[str], style: str) -> str:
    if style == "json":
        return json.dumps(rows, indent=2, default=str)

    if style == "csv":
        lines = [",".join(columns)]
        for row in rows:
            lines.append(",".join(
                '"%s"' % _format(row.get(column, "")).replace('"', "'") for column in columns
            ))
        return "\n".join(lines)

    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join("---" for _ in columns) + "|"
    body = [
        "| " + " | ".join(_format(row.get(column, "")) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, divider] + body)


def _run(args: argparse.Namespace) -> int:
    rows = collect(args.root)
    if args.task:
        rows = [r for r in rows if r.get("task") == args.task]
    if args.model:
        rows = [r for r in rows if args.model in str(r.get("model", ""))]
    if args.resolution:
        rows = [r for r in rows if r.get("resolution") == args.resolution]

    if not rows:
        print(f"[symtrs] no run summaries under {args.root}. Runs write summary.json when they finish.")
        return 1

    if args.sort_by:
        rows.sort(key=lambda r: (r.get(args.sort_by) is None, r.get(args.sort_by)))

    columns = _columns(rows, args.metric)
    text = render(rows, columns, args.format)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"[symtrs] wrote {len(rows)} rows to {args.out}")
    else:
        print(text)
    return 0


__all__ = ["add_parser", "collect", "render"]
