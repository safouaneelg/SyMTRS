"""`symtrs setup-adapt` - check the vendored CycleGAN/pix2pix tree.

    symtrs setup-adapt                 # verify the vendored tree, print provenance
    symtrs setup-adapt --fetch         # compare it against the pinned upstream revision
    symtrs setup-adapt --check         # exit 1 if the tree is not usable

The complete `junyanz/pytorch-CycleGAN-and-pix2pix` source is committed to this
repository under `pytorch-CycleGAN-and-pix2pix/`, so there is no clone step and nothing
to install: `symtrs train --task domain_adapt` runs offline on a fresh checkout. This
command exists to answer two questions that a paper's reproduction instructions have to
answer, and to answer them without a network round trip:

  * is the tree that is committed here complete and usable? (`--check`)
  * is it the code we say it is? (`--fetch` re-derives the delta against
    `UPSTREAM_REVISION` and prints it; it never writes to the tree)

Use `--target` or `SYMTRS_UPSTREAM` to point at a separately-managed checkout instead,
which is what a shared cluster install wants.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "setup-adapt", help="check the vendored CycleGAN/pix2pix code"
    )
    parser.add_argument("--target", default="",
                        help="the tree to check (default: the vendored folder)")
    parser.add_argument("--revision", default="",
                        help="upstream revision to compare against (default: the pinned one)")
    parser.add_argument("--fetch", action="store_true",
                        help="clone the pinned revision and report which files differ")
    parser.add_argument("--check", action="store_true", help="only report whether the tree is ready")
    parser.set_defaults(handler=_run)


def _run(args: argparse.Namespace) -> int:
    from symtrs.model.domain import upstream

    target = Path(args.target).expanduser() if args.target else upstream.default_upstream()

    if args.check:
        if upstream.is_complete(target):
            print(f"[symtrs] {target} is ready")
            return 0
        missing = upstream.missing_files(target) if target.is_dir() else list(upstream.REQUIRED_FILES)
        print(f"[symtrs] {target} is not ready; missing {len(missing)} file(s):")
        for name in missing:
            print(f"  {name}")
        print("[symtrs] this tree is committed to the repository, so restore it with "
              "`git checkout -- pytorch-CycleGAN-and-pix2pix`")
        return 1

    upstream.setup(
        target=str(target),
        revision=args.revision or upstream.UPSTREAM_REVISION,
        fetch=args.fetch,
    )
    print("[symtrs] next: symtrs train --task domain_adapt --model cyclegan "
          "--data-root <root> --set model_kwargs.direction=day2night")
    return 0
