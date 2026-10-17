"""CSV metric logging."""

from __future__ import annotations

import csv
import math
import os
from typing import Mapping


def append_csv_row(path: str, row: Mapping) -> None:
    """Append one row, writing the header on first use.

    The header is derived from the first row's keys. Callers must keep their key
    set stable across epochs; `MetricTracker` guarantees that by construction.

    A NaN/inf cell is written empty, the CSV spelling of the `null` that `summary.json`
    uses: "could not be computed" must not read back as a number (`row_as_floats` leaves an
    empty cell as a string, so nothing downstream averages it as zero).
    """
    row = {
        key: ("" if isinstance(value, float) and not math.isfinite(value) else value)
        for key, value in row.items()
    }
    directory = os.path.dirname(str(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    fieldnames = list(row.keys())
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def read_csv_rows(path: str):
    """Read a metrics CSV back as a list of dicts, for the table generator."""
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def row_as_floats(row: Mapping) -> dict:
    """Cast every numeric-looking cell to float; leave the rest as strings."""
    out = {}
    for key, value in row.items():
        try:
            out[key] = float(value)
        except (TypeError, ValueError):
            out[key] = value
    return out
