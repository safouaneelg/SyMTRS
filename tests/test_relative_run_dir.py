"""A relative run directory must reach every subprocess as an absolute path.

The failure these tests exist for: the 2026-10-17 validation queue ran
`symtrs train <cfg> --out-dir runs/validation/<name>` from the repo root. Shadow removal
and domain adaptation launch their subprocess with the *upstream checkout* as cwd, so

  * the shadow worker died on `FileNotFoundError: 'runs/validation/restormer/config.yaml'`
    (and would have derived a wrong run directory from the relative `out_dir` in it), and
  * upstream's `train.py --dataroot runs/.../staging/train` could not see `trainA/trainB`,
    fell into its auto-split branch and raised "Could not find trainA/trainB or raw
    directories for auto-splitting" (its relative `--checkpoints_dir`/`--results_dir` would
    have landed inside the vendored tree).

The paths a user types are relative to their shell, so `RunConfig.anchored()` resolves them
against that cwd once, before any launch. The launchers are replaced here by recorders, so
nothing is trained and no dataset is needed; the cwd is moved to a temp dir that is *not*
the repo, so "resolved against the wrong directory" cannot pass by coincidence.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from conftest import REPO_ROOT
from symtrs.config import RunConfig

RELATIVE_OUT = os.path.join("runs", "validation", "x")


@pytest.fixture
def caller_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The directory the user typed their relative paths in; not the repo, not upstream."""
    here = tmp_path / "caller"
    here.mkdir()
    monkeypatch.chdir(here)
    return here


def _assert_absolute_under(value: str, expected: Path) -> None:
    assert os.path.isabs(value), f"{value!r} is relative; the subprocess cwd would change its meaning"
    assert Path(value).resolve() == expected.resolve()


# ------------------------------------------------------------------------- the config


def test_anchored_resolves_against_the_callers_cwd_and_leaves_the_original(caller_cwd: Path) -> None:
    cfg = RunConfig(task="shadow_removal", model="restormer").with_overrides(
        {
            "out_dir": RELATIVE_OUT,
            "data.root": "data",
            "shadow.init_weights": os.path.join(".cache", "w.pth"),
            "train.resume": os.path.join("old", "last.pt"),
        }
    )
    anchored = cfg.anchored()
    _assert_absolute_under(anchored.out_dir, caller_cwd / RELATIVE_OUT)
    _assert_absolute_under(anchored.data.root, caller_cwd / "data")
    _assert_absolute_under(anchored.shadow.init_weights, caller_cwd / ".cache" / "w.pth")
    _assert_absolute_under(anchored.train.resume, caller_cwd / "old" / "last.pt")
    assert cfg.out_dir == RELATIVE_OUT, "anchored() must copy, not mutate the caller's config"


def test_anchored_covers_the_default_run_directory_and_keeps_empty_paths_empty(caller_cwd: Path) -> None:
    """With no `--out-dir` the canonical default (`runs/<task>/<model>/<res>`) is relative too."""
    anchored = RunConfig(task="shadow_removal", model="restormer").anchored()
    _assert_absolute_under(anchored.out_dir, caller_cwd / "runs" / "shadow_removal" / "restormer" / "2048")
    assert anchored.shadow.init_weights == "" and anchored.train.resume == "" and anchored.data.root == ""


# --------------------------------------------------------------------- shadow removal


def _shadow_cfg() -> RunConfig:
    cfg = RunConfig.from_yaml(str(REPO_ROOT / "configs/shadow_removal/restormer.yaml"))
    return cfg.with_overrides(
        {
            "out_dir": RELATIVE_OUT,
            "data.root": "data",
            "shadow.init_weights": os.path.join(".cache", "real_denoising.pth"),
            "train.resume": os.path.join("prev", "last.pt"),
        }
    )


def test_shadow_worker_gets_absolute_config_and_weights(
    caller_cwd: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The worker runs in the upstream checkout: config, weights and the config's own paths must be absolute."""
    from symtrs.model.shadow import runner, staging

    launches = []
    monkeypatch.setattr(runner, "upstream_missing", lambda key: "")
    monkeypatch.setattr(runner, "_launch", lambda command, cwd: launches.append((command, cwd)))
    monkeypatch.setattr(runner, "score_split", lambda cfg, run_dir, split: {})
    staged = []
    monkeypatch.setattr(
        staging, "stage_split", lambda root, run_dir, name, *a: staged.append((root, run_dir, name))
    )

    runner.run_shadow_removal(_shadow_cfg(), mode="finetune", checkpoint=os.path.join("ckpt", "best.pt"))

    run_dir = caller_cwd / RELATIVE_OUT
    assert len(launches) == 2  # train, then predict
    for command, cwd in launches:
        _assert_absolute_under(command[command.index("--config") + 1], run_dir / "config.yaml")
        assert Path(cwd).resolve() != caller_cwd.resolve(), "the premise: cwd is the upstream repo"
    train_command, predict_command = (c for c, _ in launches)
    _assert_absolute_under(predict_command[predict_command.index("--weights") + 1], caller_cwd / "ckpt" / "best.pt")
    assert train_command.count("--weights") == 0

    # What the worker re-reads from config.yaml, with the upstream checkout as cwd.
    written = RunConfig.from_yaml(str(run_dir / "config.yaml"))
    _assert_absolute_under(written.out_dir, run_dir)
    assert written.resolved_out_dir == run_dir.resolve()
    _assert_absolute_under(written.shadow.init_weights, caller_cwd / ".cache" / "real_denoising.pth")
    _assert_absolute_under(written.train.resume, caller_cwd / "prev" / "last.pt")
    _assert_absolute_under(written.data.root, caller_cwd / "data")

    assert staged and all(os.path.isabs(run) and os.path.isabs(root) for root, run, _ in staged)
    assert (run_dir / "summary.json").is_file(), "output must land where the relative --out-dir pointed"


def test_shadow_dry_run_prints_the_absolute_config_path(
    caller_cwd: Path, capsys: pytest.CaptureFixture
) -> None:
    from symtrs.commands._common import print_shadow_commands

    print_shadow_commands(_shadow_cfg())
    printed = capsys.readouterr().out
    assert str((caller_cwd / RELATIVE_OUT / "config.yaml").resolve()) in printed


# ------------------------------------------------------------------ domain adaptation


@pytest.mark.parametrize("model", ["cyclegan", "pix2pix"])
def test_domain_upstream_gets_absolute_dataroot_checkpoints_and_results(
    caller_cwd: Path, monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    """`--dataroot`, `--checkpoints_dir` and `--results_dir` are read with cwd = the vendored tree."""
    from symtrs.model.domain import runner

    upstream = runner.upstream_module
    monkeypatch.setattr(upstream, "is_complete", lambda root=None: True)
    monkeypatch.setattr(upstream, "verify", lambda root=None: upstream.default_upstream())
    monkeypatch.setattr(upstream, "count_staged", lambda staging, split: (1, 1))
    monkeypatch.setattr(runner, "evaluate_adaptation", lambda cfg, task, run_dir: {})
    staged = []

    def fake_staging(root, run_dir, split="train", **kwargs):
        staged.append((root, run_dir))
        return Path(run_dir) / "staging" / split

    monkeypatch.setattr(upstream, "build_staging_dir", fake_staging)
    launches = []
    monkeypatch.setattr(runner, "_launch", lambda command, cwd, env=None: launches.append((command, cwd)))

    cfg = RunConfig(task="domain_adapt", model=model).with_overrides(
        {"out_dir": RELATIVE_OUT, "data.root": "data", "train.epochs": 1}
    )
    runner.run_domain_adaptation(cfg, mode="finetune")

    run_dir = caller_cwd / RELATIVE_OUT
    assert len(launches) == 1
    command, cwd = launches[0]
    assert Path(cwd).resolve() != caller_cwd.resolve(), "the premise: cwd is the vendored upstream tree"
    _assert_absolute_under(command[command.index("--dataroot") + 1], run_dir / "staging" / "train")
    _assert_absolute_under(command[command.index("--checkpoints_dir") + 1], run_dir / "checkpoints")
    assert all(os.path.isabs(root) and os.path.isabs(run) for root, run in staged)

    # `evaluate_adaptation` (stubbed above) hands `test.py` the anchored cfg's run directory.
    anchored = cfg.anchored()
    task = runner.DomainAdaptation(direction="day2night")
    task.key = model
    task.run_test(dataroot=anchored.data.root, run_dir=str(anchored.resolved_out_dir), split="test")
    test_command = launches[-1][0]
    _assert_absolute_under(test_command[test_command.index("--results_dir") + 1], run_dir / "results")
    _assert_absolute_under(test_command[test_command.index("--dataroot") + 1], run_dir / "staging" / "test")
    assert (run_dir / "summary.json").is_file()


# ------------------------------------------------------------------ super-resolution


def test_superres_trainer_gets_an_absolute_out_dir(caller_cwd: Path, layout: Path) -> None:
    """Relative-safe before only because the trainer inherits the caller's cwd; now by construction."""
    from symtrs.model.superres import run_super_resolution

    relative_root = os.path.relpath(layout, caller_cwd)
    cfg = RunConfig(task="superresolution", model="srcnn", resolution=512).with_overrides(
        {"out_dir": RELATIVE_OUT, "data.root": relative_root}
    )
    summary = run_super_resolution(cfg, mode="finetune", dry_run=True)

    command = summary["command"]
    run_dir = caller_cwd / RELATIVE_OUT
    _assert_absolute_under(command[command.index("--out_dir") + 1], run_dir)
    _assert_absolute_under(command[command.index("--lr_root") + 1], layout / "train" / "rgb_day_lr" / "x4")
    assert Path(summary["run_dir"]).resolve() == run_dir.resolve()
