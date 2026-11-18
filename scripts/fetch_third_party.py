#!/usr/bin/env python3
"""Check out the four shadow-removal upstreams (git submodules under third_party/).

    python scripts/fetch_third_party.py           # git submodule update --init third_party
    python scripts/fetch_third_party.py --check   # verify every checkout is at its pin; change nothing

The pins are the submodule gitlinks, not a table: `.gitmodules` holds each URL and git records
the commit, so `git submodule status` is the single source of truth (third_party/PINS.md keeps
the notes - dates, licence, weights - and no hashes). This wrapper exists so the command the
error messages and older docs name keeps working, and so `--check` reads as a pass/fail list.
A submodule cannot carry local edits: none of the four is modified, and if one ever has to be,
fork it and point the submodule at the fork rather than editing the checkout.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: `git submodule status` prefixes: ' ' at the pin, '+' another commit checked out,
#: '-' not initialised, 'U' merge conflict.
STATES = {" ": "ok", "+": "BAD (checked out at another commit)", "-": "missing (run without --check)",
          "U": "BAD (merge conflict)"}


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if not args.check:
        result = git("submodule", "update", "--init", "third_party")
        if result.returncode:
            print(result.stderr, end="")
            return 1

    result = git("submodule", "status", "third_party")
    if result.returncode or not result.stdout.strip():
        print(result.stderr or "no submodules under third_party/ (is .gitmodules present?)")
        return 1
    bad = 0
    for line in result.stdout.splitlines():
        state, commit, path = line[0], *line[1:].split()[:2]
        label = STATES.get(state, f"BAD (unknown state {state!r})")
        print(f"[{label.split()[0]}] {Path(path).name} at {commit[:10]}"
              + ("" if state == " " else f"  - {label}"))
        bad += state != " "
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
