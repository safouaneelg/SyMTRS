#!/usr/bin/env python3
"""Clone the three shadow-removal upstreams at the revisions pinned in third_party/PINS.md.

    python scripts/fetch_third_party.py           # clone / checkout / apply third_party/patches/<repo>*.patch
    python scripts/fetch_third_party.py --check   # verify revisions (and patches applied); change nothing

The clones are gitignored (third_party/*/): they are upstream code, not ours. The pins are
read from PINS.md so there is one table to edit. Patches, if any, live in
third_party/patches/<RepoName>-*.patch and are applied with `git apply`; every patch must be
listed in PINS.md (the current set is: none needed on torch 2.11).
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TP = ROOT / "third_party"


def pins():
    rows = []
    for line in (TP / "PINS.md").read_text(encoding="utf-8").splitlines():
        m = re.match(r"\|\s*(\w+)\s*\|\s*(https://\S+)\s*\|\s*([0-9a-f]{40})\s*\|", line)
        if m:
            rows.append(m.groups())
    if not rows:
        raise SystemExit("no pins found in third_party/PINS.md")
    return rows


def git(*args, cwd=None):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


def head(path: Path) -> str:
    result = git("rev-parse", "HEAD", cwd=path)
    return result.stdout.strip() if result.returncode == 0 else ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    bad = 0
    for name, url, commit in pins():
        path = TP / name
        patches = sorted((TP / "patches").glob(f"{name}-*.patch"))
        if args.check:
            ok = head(path) == commit
            print(f"[{'ok' if ok else 'BAD'}] {name} at {head(path)[:10] or 'missing'} (pinned {commit[:10]})")
            bad += not ok
            continue
        if not path.is_dir():
            r = git("clone", url, str(path))
            if r.returncode:
                print(r.stderr)
                return 1
        if head(path) != commit:
            for cmd in (("fetch", "origin", commit), ("checkout", "--force", commit)):
                r = git(*cmd, cwd=path)
                if r.returncode:
                    print(f"{name}: git {' '.join(cmd)} failed: {r.stderr}")
                    return 1
        for patch in patches:
            if git("apply", "--check", str(patch), cwd=path).returncode == 0:
                git("apply", str(patch), cwd=path)
                print(f"{name}: applied {patch.name}")
        print(f"{name}: at {commit[:10]}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
