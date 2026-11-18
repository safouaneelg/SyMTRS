"""`symtrs train-group` - train several compatible models on one shared batch.

    symtrs train-group configs/depth/2048/midas_small_2048.yaml \\
                       configs/depth/2048/midas_dpt_hybrid_2048.yaml \\
                       --data-root /data/SyMTRS --device cuda:0

The point is the data path, not the models: one pass over the loader feeds every member
the same decoded images, so N models cost one set of PNG/EXR decodes instead of N. What
is shared is the largest member's optimiser window (`batch_size x grad_accum` samples);
each member cuts it into its own windows and micro-batches, so a 4x2 and an 8x1 member
share one window, and a 4x1 member beside them takes two optimiser steps out of it. The
models do not share weights, optimisers or checkpoints - each member writes exactly what
`symtrs train` on that config writes, in the same run directory, with `cogroup` in its
summary recording what it shared the pass with.

The members have to agree on everything that shapes the batch, and each one's effective
batch has to divide the window (`symtrs/engine/trainer.py`, `group_conflicts`); this
command refuses a group that does not, rather than quietly
training one member on another member's crops. Only depth and segmentation co-train: the
other three tasks are subprocesses and keep their own data loading.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List

from symtrs.config import ConfigError, RunConfig
from symtrs.commands._common import parse_set


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "train-group",
        help="train several compatible configs in one process on one shared batch",
    )
    parser.add_argument("configs", nargs="+", metavar="CONFIG.yaml",
                        help="two or more run configs that agree on task, resolution and the data")
    parser.add_argument("--data-root", dest="data_root", default=None,
                        help="dataset root, overriding every member's data.root")
    parser.add_argument("--out-dir", dest="out_dir", default=None,
                        help="parent directory; each member writes <out-dir>/<config stem>. "
                             "Without it every member keeps the run dir its own config names")
    parser.add_argument("--device", default=None, help="cuda, cuda:1, cpu (one device, one process)")
    parser.add_argument("--num-workers", dest="num_workers", type=int, default=None,
                        help="data workers for the one shared loader")
    parser.add_argument("--set", action="append", default=None, metavar="KEY=VALUE",
                        help="override applied to every member, e.g. --set train.epochs=5")
    parser.add_argument("--resume-existing", action="store_true",
                        help="for every member with no train.resume set, resume from its own "
                             "<run dir>/weights/last.pt when it exists (what the queue wants; "
                             "a solo `symtrs train` never resumes implicitly)")
    parser.add_argument("--print-config", action="store_true",
                        help="print every resolved config and exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="check that the members can share a pass, print the plan, exit")
    parser.set_defaults(handler=_run)


def build_group(args: argparse.Namespace) -> List[RunConfig]:
    """Read every config, apply the shared overrides, validate, and check the group."""
    from pathlib import Path

    overrides: Dict[str, Any] = parse_set(args.set)
    if args.data_root:
        overrides["data.root"] = args.data_root
    if args.num_workers is not None:
        overrides["data.num_workers"] = args.num_workers

    cfgs: List[RunConfig] = []
    for path in args.configs:
        cfg = RunConfig.from_yaml(path)
        if args.out_dir:
            overrides_per_member = dict(overrides, out_dir=str(Path(args.out_dir) / Path(path).stem))
        else:
            overrides_per_member = overrides
        cfgs.append((cfg.with_overrides(overrides_per_member) if overrides_per_member else cfg).validate())

    from symtrs.engine.trainer import group_conflicts

    conflicts = group_conflicts(cfgs)
    if conflicts:
        names = ", ".join(cfg.model for cfg in cfgs)
        raise ConfigError(
            f"cannot co-train {names}: " + "; ".join(conflicts) + ". One group reads one "
            "loader and hands every member the same optimiser window, so the members have to "
            "agree on the data and on `batch_size x grad_accum`. Run them separately "
            "(`symtrs train <config>`, one at a time) or make the configs agree."
        )
    return cfgs


def _run(args: argparse.Namespace) -> int:
    from pathlib import Path

    cfgs = build_group(args)

    if args.resume_existing:
        for cfg in cfgs:
            last = Path(cfg.resolved_out_dir) / "weights" / "last.pt"
            if not cfg.train.resume and last.is_file():
                # Each member has its own checkpoint, which a shared `--set train.resume`
                # cannot express: hence a flag rather than a config value.
                cfg.train.resume = str(last)
                print(f"[symtrs] [{cfg.model}] resuming from {last}")

    if args.print_config:
        import yaml

        for cfg in cfgs:
            print(f"# {cfg.model} -> {cfg.resolved_out_dir}")
            print(yaml.safe_dump(cfg.to_dict(), sort_keys=False, default_flow_style=False))
        return 0

    if args.dry_run:
        first = cfgs[0]
        from symtrs.engine.trainer import effective_batch, shared_window

        window = shared_window(cfgs)
        splits = ", ".join(
            f"{cfg.model} {cfg.train.batch_size}x{cfg.train.grad_accum}"
            f" x{window // effective_batch(cfg)} steps"
            for cfg in cfgs
        )
        print(
            f"[symtrs] dry run - {len(cfgs)} members can share one pass: "
            f"{first.task} / {first.resolution}px / "
            f"window={window} ({splits}) / "
            f"patch_size={first.train.patch_size}"
        )
        for cfg in cfgs:
            print(f"[symtrs]   {cfg.model:<24} -> {cfg.resolved_out_dir}")
        return 0

    from symtrs.engine.trainer import train_group

    summaries = train_group(cfgs, device=args.device)
    for summary in summaries:
        test = summary.get("test", {})
        if test:
            print(f"[symtrs] {summary['model']}: {summary['run_dir']}/summary.json")
    return 0
