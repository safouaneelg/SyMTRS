"""Allow `python -m symtrs ...` without installing the package."""

from __future__ import annotations

from symtrs.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
