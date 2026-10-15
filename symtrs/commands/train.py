"""`symtrs train` - supervised training on the SyMTRS dataset.

    symtrs train --task depth --model midas_dpt_hybrid --resolution 1024 \\
                 --data-root /data/SyMTRS --epochs 20

    symtrs train --task segmentation --model segformer_b2 --resolution 2048 \\
                 --data-root /data/SyMTRS --batch-size 2 --patch-size 512 --freeze encoder
"""

from __future__ import annotations

import argparse

from symtrs.commands._common import add_common_args, build_config, should_stop_early


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser("train", help="train a model on the train split")
    parser.add_argument("config_positional", nargs="?", default=None, metavar="CONFIG.yaml",
                        help="YAML run config; same as --config")
    add_common_args(parser, training=True)
    parser.set_defaults(handler=_run)


def _run(args: argparse.Namespace) -> int:
    if args.config_positional and not args.config:
        args.config = args.config_positional

    cfg = build_config(args)
    if should_stop_early(args, cfg):
        return 0

    from symtrs.engine.trainer import train

    summary = train(cfg, mode="finetune")
    test = summary.get("test", {})
    if test:
        print(f"[symtrs] done. test metrics are in {summary['run_dir']}/summary.json")
    return 0
