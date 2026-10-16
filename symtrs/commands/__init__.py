"""Command implementations, one module per subcommand.

Each module exposes `add_parser(subparsers)` and a `handler` on the resulting parser.
`symtrs.cli` is the only caller, so this package has no side effects on import - and none
of these modules imports torch at module level, which keeps `symtrs --help` fast.
"""

from __future__ import annotations

import argparse
import importlib
from typing import Dict

#: Subcommand -> module. Insertion order is the order `symtrs --help` lists them, which
#: is roughly the order you would use them in.
COMMANDS: Dict[str, str] = {
    "train": "symtrs.commands.train",
    "eval": "symtrs.commands.evaluate",
    "infer": "symtrs.commands.infer",
    "sweep": "symtrs.commands.sweep",
    "tables": "symtrs.commands.tables",
    "models": "symtrs.commands.models",
    "inspect": "symtrs.commands.inspect",
    "doctor": "symtrs.commands.doctor",
    "view": "symtrs.commands.view",
    "setup-adapt": "symtrs.commands.setup_adapt",
    "make-dummy": "symtrs.commands.make_dummy",
}


def register_all(subparsers) -> None:
    for name, module_name in COMMANDS.items():
        module = importlib.import_module(module_name)
        module.add_parser(subparsers)


def dispatch(args: argparse.Namespace) -> int:
    handler = getattr(args, "handler", None)
    if handler is None:  # pragma: no cover - every parser sets one
        raise RuntimeError(f"the {getattr(args, 'command', '?')!r} command has no handler")
    return handler(args)


__all__ = ["COMMANDS", "dispatch", "register_all"]
