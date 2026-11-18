"""JSON that other tools can read: non-finite floats become `null`.

Python's `json.dumps` writes a NaN as the bare token `NaN`, which is not JSON. `jq`, every
browser and most other parsers reject the whole file, and Python itself only reads it back
because its parser is lenient. A run whose validation metric went NaN used to leave exactly
such a `summary.json`, so one diverged run could break the results table for the rest.

`null` is also the honest value: "this metric could not be computed" is not a number, and
must not be mistaken for zero when a table is read months later.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def json_safe(value: Any) -> Any:
    """A copy of `value` with every NaN/inf float (also inside dicts and lists) as None.

    Tensors and numpy scalars are not special-cased: anything with `item()` is read through
    it, because metric dictionaries pick those up from the loops and `json.dumps` would
    otherwise raise on them instead of writing them.
    """
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item") and callable(value.item) and getattr(value, "ndim", 1) == 0:
        return json_safe(value.item())
    return value


def dumps(value: Any, **kwargs: Any) -> str:
    """`json.dumps` that cannot emit `NaN`/`Infinity`; `allow_nan=False` is the backstop."""
    kwargs.setdefault("default", str)
    return json.dumps(json_safe(value), allow_nan=False, **kwargs)


def write_json(path: Any, value: Any, indent: int = 2) -> None:
    Path(path).write_text(dumps(value, indent=indent) + "\n", encoding="utf-8")


__all__ = ["dumps", "json_safe", "write_json"]
