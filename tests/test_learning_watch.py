"""The learning watcher must judge a run by its newest log, not by every log it ever wrote.

Why this exists: `_non_finite` globbed `*<run name>*.log` and summed over all of them, so the
float16 warnings of 2026-10-22 (midas_dpt_beit_large_512: 114; the 2048 MiDaS group: 12052)
kept flagging the bfloat16 runs that replaced them - a NONFINITE flag every four hours on two
runs that had logged nothing non-finite in days. A log that stopped being written is history.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "learning_watch.py"


def load_module(monkeypatch, root: Path):
    spec = importlib.util.spec_from_file_location("learning_watch_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "REPO", root)
    return module


def write_log(root: Path, name: str, text: str, mtime: float) -> Path:
    path = root / "logs" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    os.utime(path, (mtime, mtime))
    return path


def test_an_older_log_with_warnings_does_not_flag_the_newer_run(tmp_path, monkeypatch):
    module = load_module(monkeypatch, tmp_path)
    write_log(tmp_path, "midas_small_2048_20261022.log",
              "[symtrs] skipped 4978 non-finite samples\n", mtime=1_000)
    write_log(tmp_path, "midas_small_2048_20261023.log",
              "[symtrs] epoch 3 (1200.0s) lr=1.00e-05\n", mtime=2_000)
    assert module._non_finite(tmp_path / "midas_small_2048") == 0


def test_warnings_in_the_newest_log_are_still_counted(tmp_path, monkeypatch):
    module = load_module(monkeypatch, tmp_path)
    write_log(tmp_path, "midas_small_2048_20261022.log", "", mtime=1_000)
    write_log(tmp_path, "midas_small_2048_20261023.log",
              "[symtrs] skipped 3 non-finite samples\n"
              "[symtrs] skipped 4 non-finite samples\n", mtime=2_000)
    assert module._non_finite(tmp_path / "midas_small_2048") == 2


def test_only_the_newest_launch_inside_a_file_counts(tmp_path, monkeypatch):
    """The queue appends a restart to the same log under a new `=== ` header."""
    module = load_module(monkeypatch, tmp_path)
    write_log(
        tmp_path,
        "midas_small_2048_20261023.log",
        "=== 2026-10-23 08:00 float16 attempt\n"
        "[symtrs] skipped 7074 non-finite samples\n"
        "=== 2026-10-23 20:00 bfloat16 attempt\n"
        "[symtrs] epoch 1 (600.0s) lr=1.00e-05\n",
        mtime=2_000,
    )
    assert module._non_finite(tmp_path / "midas_small_2048") == 0


def test_a_group_log_is_counted_per_member(tmp_path, monkeypatch):
    module = load_module(monkeypatch, tmp_path)
    write_log(
        tmp_path,
        "midas_small_2048+midas_dpt_hybrid_2048+midas_dpt_large_2048_20261025.log",
        "[symtrs] [midas_dpt_hybrid] skipped 5 non-finite samples\n"
        "[symtrs] [midas_dpt_hybrid] skipped 1 non-finite samples\n"
        "[symtrs] [midas_small] skipped 9 non-finite samples\n",
        mtime=3_000,
    )
    assert module._non_finite(tmp_path / "midas_dpt_hybrid_2048") == 2
    assert module._non_finite(tmp_path / "midas_small_2048") == 1
    assert module._non_finite(tmp_path / "midas_dpt_large_2048") == 0


def test_a_run_with_no_log_is_not_an_error(tmp_path, monkeypatch):
    module = load_module(monkeypatch, tmp_path)
    assert module._non_finite(tmp_path / "never_logged_512") == 0
