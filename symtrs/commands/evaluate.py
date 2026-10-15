"""`symtrs eval` - evaluate a model without training.

Three ways to say what to evaluate:

    symtrs eval --task depth --model midas_dpt_hybrid --resolution 1024 \\
                --data-root /data/SyMTRS                    # zero-shot, released weights

    symtrs eval --task depth --checkpoint runs/depth/midas_dpt_hybrid/1024/weights/best.pt

    symtrs eval --config configs/depth_zoedepth_k_2048.yaml --checkpoint <run dir>

Passing a run directory works too; `weights/best.pt` is found inside it.
"""

from __future__ import annotations

import argparse

from symtrs.commands._common import add_common_args, build_config, should_stop_early


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser("eval", help="evaluate a model on the val or test split")
    parser.add_argument("config_positional", nargs="?", default=None, metavar="CONFIG.yaml")
    parser.add_argument("--checkpoint", default="",
                        help="a symtrs .pt or a run directory; default is the released weights")
    # Deliberately not the same flag as `--split`: that one sets data.split, which must
    # differ from the evaluation splits, so reusing it here would make `--split test`
    # fail validation. This one selects the split to score, and nothing else.
    parser.add_argument("--eval-split", dest="eval_split", default=None,
                        help="split to score: val or test (default: data.test_split)")
    add_common_args(parser, training=False)
    parser.set_defaults(handler=_run)


def _run(args: argparse.Namespace) -> int:
    if args.config_positional and not args.config:
        args.config = args.config_positional

    cfg = build_config(args)
    if should_stop_early(args, cfg):
        return 0

    from symtrs.engine.loop import format_metrics
    from symtrs.engine.trainer import evaluate_run

    summary = evaluate_run(cfg, split=args.eval_split, checkpoint=args.checkpoint)

    # `superresolution` and `domain_adapt` are driven as subprocesses and report through
    # their own summary shape, which is the one `symtrs tables` reads. They have already
    # printed their numbers by this point; only the tensor tasks fold their metrics into
    # a `metrics` key for printing here.
    if "metrics" in summary:
        print(f"[symtrs] {format_metrics(summary['metrics'])}")
    elif "test" in summary:
        print(f"[symtrs] {format_metrics(summary['test'])}")
    return 0
