"""`symtrs models` - what can be trained, and what each model needs installed."""

from __future__ import annotations

import argparse
import importlib.util
import sys


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser("models", help="list the model registry")
    parser.add_argument("--task", default=None,
                        help="restrict to one task: depth, segmentation, domain_adapt")
    parser.add_argument("--available", action="store_true",
                        help="hide entries whose optional dependency is missing")
    parser.set_defaults(handler=_run)


def _run(args: argparse.Namespace) -> int:
    from symtrs.registry import describe_registry, list_models

    specs = list_models(args.task)
    if args.available:
        specs = [s for s in specs if all(_installed(m) for m in s.requires)]
        if not specs:
            print("[symtrs] no model's dependencies are installed in this environment")
            return 1
        print(f"[symtrs] {len(specs)} of {len(list_models(args.task))} entries are usable here")

    print(describe_registry(args.task, specs=specs))
    print()
    print("install one with:  pip install -e '.[timm]'   (or 'da3', 'depthpro', 'exr', 'cv2')")
    print("check an install:   symtrs doctor")
    return 0


def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


__all__ = ["add_parser"]
