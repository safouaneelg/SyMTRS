"""CSV metric logging."""

from __future__ import annotations

import csv
import math
import os
from typing import Mapping


def append_csv_row(path: str, row: Mapping) -> None:
    """Append one row, writing the header on first use and widening it when a key is new.

    The header starts as the first row's keys. A later row may bring a key the header does
    not have - `val_skipped_images` only exists in an epoch where an image was skipped, and
    `evaluate_run` appends a `split` column to a training run's file. Appending such a row
    under the old header used to make `DictWriter` raise, or (with the cell written as an
    extra) leave an unlabeled trailing value that read back as the wrong column. So the file
    is rewritten once with the union of the keys, earlier rows getting an empty cell, which
    is the same "not measured" spelling as a NaN. A row that lacks a header key gets an empty
    cell for it, for the same reason.

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
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    if exists:
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            header = list(reader.fieldnames or [])
            existing = list(reader) if header else []
        new_keys = [key for key in fieldnames if key not in header]
        fieldnames = header + new_keys
        if new_keys:
            # Rewrite through a temporary file so a crash mid-rewrite cannot truncate the
            # history of a run that is hours in.
            tmp = f"{path}.tmp"
            with open(tmp, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="")
                writer.writeheader()
                writer.writerows(existing)
            os.replace(tmp, path)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="")
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
