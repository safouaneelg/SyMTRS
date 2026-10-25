"""A queue must not resume a run another process is writing.

Why this exists: on 2026-10-25 two overlapping queues both saw "no summary.json, has
last.pt" and started the same runs. `srcnn_x2` had two trainers on one directory for fifteen
minutes, and `midas_small_256` was trained by two processes at once - its metrics.csv carries
epochs 10-19 twice and its summary came from an interleaved trajectory, so the run had to be
discarded. `inflight` is the guard, tested against a fake /proc tree so it needs no processes.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def load_run_queue():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_queue.py"
    spec = importlib.util.spec_from_file_location("run_queue_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def run_queue():
    return load_run_queue()


def fake_proc(root: Path, pid: str, argv: list[str]) -> None:
    entry = root / pid
    entry.mkdir(parents=True)
    (entry / "cmdline").write_bytes("\x00".join(argv).encode() + b"\x00")


def test_finds_a_live_trainer(tmp_path, run_queue):
    proc = tmp_path / "proc"
    run = tmp_path / "runs" / "srcnn_x2"
    fake_proc(proc, "4242", [
        ".venv/bin/python", "-m", "symtrs.cli", "train",
        "configs/superresolution/srcnn_x2.yaml", "--out-dir", str(run),
    ])
    found = run_queue.inflight(run, proc_root=proc)
    assert found and "4242" in found


def test_matches_the_relative_spelling_the_queue_passes(tmp_path, run_queue, monkeypatch):
    # The queue builds `--out-dir <out_root>/<config stem>` from what the caller typed, so the
    # child's argv usually holds `runs/campaign2/srcnn_x2`, not the resolved path.
    monkeypatch.chdir(tmp_path)
    proc = tmp_path / "proc"
    fake_proc(proc, "77", ["python", "-m", "symtrs.legacy.superres.train",
                           "--out_dir", "runs/campaign2/srcnn_x2"])
    assert run_queue.inflight(Path("runs/campaign2/srcnn_x2"), proc_root=proc)


def test_finds_a_group_member_by_its_config(tmp_path, run_queue):
    """`train-group` writes to `<out_root>/<stem>` and names members by config path only."""
    proc = tmp_path / "proc"
    run = tmp_path / "runs" / "midas_small_2048"
    fake_proc(proc, "9", ["python", "-m", "symtrs.cli", "train-group",
                          "configs/depth/2048/midas_small_2048.yaml",
                          "configs/depth/2048/midas_dpt_hybrid_2048.yaml",
                          "--out-dir", str(tmp_path / "runs")])
    assert run_queue.inflight(run, "configs/depth/2048/midas_small_2048.yaml", proc_root=proc)
    # The same tree without the config hint misses it - which is why run_one passes it.
    assert run_queue.inflight(run, proc_root=proc) is None


def test_ignores_unrelated_mentions(tmp_path, run_queue):
    """An editor, a `ls`, or a prefix-sharing run name must not block a queue."""
    proc = tmp_path / "proc"
    run = tmp_path / "runs" / "srcnn_x2"
    fake_proc(proc, "1", ["vim", str(run / "config.yaml")])          # not symtrs
    fake_proc(proc, "2", ["python", "-m", "symtrs.cli", "train",     # a different run
                          "--out-dir", str(run) + "_x8"])
    fake_proc(proc, "3", [])                                          # zombie: empty cmdline
    assert run_queue.inflight(run, proc_root=proc) is None


def test_missing_proc_root_is_not_an_error(tmp_path, run_queue):
    """Reading a fake tree that vanished (or a non-Linux box) must skip, not crash the queue."""
    assert run_queue.inflight(tmp_path / "runs" / "srcnn_x2", proc_root=tmp_path / "absent") is None
