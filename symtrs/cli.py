"""The `symtrs` command line.

    symtrs train --task segmentation --model segformer_b2 --resolution 2048 \\
                 --data-root /data/SyMTRS
    symtrs train_sr_x2.yaml                 # a YAML file alone means "train with it"
    symtrs eval --task depth --model midas_dpt_hybrid --resolution 1024 --data-root ...
    symtrs infer --task segmentation --checkpoint runs/.../best.pt --image scene.png

Adding a subcommand means adding a module to `symtrs.commands` and one line to
`COMMANDS`. Nothing else in the package needs to know the command line exists, which is
why the same runs are reachable from YAML and from `symtrs.model.autoloader`.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from symtrs import __version__
from symtrs.commands import COMMANDS, dispatch, register_all

#: Errors that are the user's to fix: a bad config, a bad path, a missing dependency.
#: These get a one-line message instead of a traceback, because the traceback would be
#: inside this package and say nothing about which flag was wrong.
_REPORTED: Optional[tuple] = None


def reported_errors() -> tuple:
    global _REPORTED
    if _REPORTED is None:
        from symtrs.common.checkpoint import CheckpointError
        from symtrs.config import ConfigError
        from symtrs.data.exr import DepthReadError
        from symtrs.data.io import ImageReadError
        from symtrs.data.splits import MissingDirectoryError
        from symtrs.model.domain import UpstreamError
        from symtrs.model.superres import SuperResolutionError
        from symtrs.registry import RegistryError

        _REPORTED = (
            ConfigError,
            CheckpointError,
            RegistryError,
            UpstreamError,
            SuperResolutionError,
            ImageReadError,
            DepthReadError,
            MissingDirectoryError,
            FileNotFoundError,
            NotADirectoryError,
            # A missing extra is a setup problem, not a bug in this package. Every adapter
            # that needs timm, torchvision, transformers or depth_anything_3 raises an ImportError
            # naming the package and the pip command, and reporting that as an unexpected
            # internal error sends the reader to the wrong place - which is exactly the
            # wrong move on a fresh cloud GPU box, where several extras are missing at once.
            ImportError,
            ValueError,
        )
    return _REPORTED

EPILOG = """\
bare `symtrs` opens the interactive launcher (download only the groups and models you need).

examples:
  symtrs
  symtrs get --list
  symtrs get --preset depth-2048 --data-root /data/SyMTRS --yes
  symtrs models
  symtrs make-dummy --out /tmp/SyMTRS_dummy --size 256
  symtrs doctor --data-root /data/SyMTRS
  symtrs train --task depth --model da3_metric_large --resolution 2048 --data-root /data/SyMTRS
  symtrs train configs/depth_midas_dpt_hybrid_1024.yaml
  symtrs sweep --task segmentation --data-root /data/SyMTRS --execute
  symtrs tables runs/ --task depth --format markdown

python:
  from symtrs.model import autoloader
  model = autoloader("runs/depth/da3_metric_large/2048/weights/best.pt")
  depth = model.predict("scene.png")

run `symtrs <command> --help` for the flags of one command.
"""


def _interactive() -> bool:
    """Whether there is a terminal to draw a TUI in (and a person to press space)."""
    return sys.stdin.isatty() and sys.stdout.isatty()


def _launcher_installed() -> bool:
    """Whether the optional launcher stack is present, without importing either package."""
    import importlib.util

    for module in ("textual", "huggingface_hub"):
        try:
            if importlib.util.find_spec(module) is None:
                return False
        except (ImportError, ValueError):
            return False
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="symtrs",
        description=(
            "Download the SyMTRS groups you need, then train and evaluate depth, "
            "segmentation and domain-adaptation models on them."
        ),
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"symtrs {__version__}")
    parser.add_argument("--traceback", action="store_true",
                        help="let errors propagate instead of printing a one-line message")

    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    register_all(subparsers)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # `symtrs train_sr_x2.yaml` is a request people make constantly, and answering it
    # with "invalid choice" helps nobody. A bare config file means `train`.
    if argv and not argv[0].startswith("-") and argv[0].endswith((".yaml", ".yml")):
        argv = ["train", *argv]

    parser = build_parser()
    if not argv:
        # Bare `symtrs` is the launcher. Without a terminal (a CI log, a pipe) there is
        # nobody to press space, and without the optional extra the right answer is the
        # help plus the install line, not an error from a package the user never asked for.
        if _interactive() and _launcher_installed():
            argv = ["get"]
        else:
            parser.print_help()
            if _interactive():
                print(
                    "\n[symtrs] the interactive launcher is not installed: "
                    "pip install -e '.[launcher]'",
                    file=sys.stderr,
                )
            return 0

    args = parser.parse_args(argv)
    if getattr(args, "command", None) is None:
        parser.print_help()
        return 0

    try:
        return dispatch(args)
    except KeyboardInterrupt:
        print("\n[symtrs] interrupted", file=sys.stderr)
        return 130
    except reported_errors() as exc:
        if getattr(args, "traceback", False):
            raise
        name = type(exc).__name__
        print(f"[symtrs] {name}: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        # Anything else is a bug in this package or in a model adapter. Say so, and
        # point at the flag that shows where it happened.
        if getattr(args, "traceback", False):
            raise
        print(
            f"[symtrs] unexpected {type(exc).__name__}: {exc}\n"
            "[symtrs] re-run with --traceback for the full stack, and include it in a bug report",
            file=sys.stderr,
        )
        return 3


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
