"""`train.init_weights`: start a run from another run's weights, and nothing else.

The feature exists for a fine-tune, and its tests are about the difference from `--resume`,
which is the flag a fine-tune is tempted to use and cannot:

  * the config refuses both at once, because they mean opposite things;
  * `_load_init_weights` copies the weights and returns nothing - no epoch counter, no
    `best`, no history - so the run that follows is a new run;
  * a mismatch is an error naming the file and the count, because `strict=False` would
    otherwise silently train a partly random model and still report a number.

The config tests are torch-free; the loading ones build one 1x1 convolution.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import requires_torch

from symtrs.config import ConfigError, RunConfig


# ------------------------------------------------------------------------- the config


def test_resume_and_init_weights_are_refused_together() -> None:
    cfg = RunConfig(task="depth", model="midas_small").with_overrides(
        {
            "data.root": "/data/SyMTRS",
            "train.resume": "runs/a/weights/last.pt",
            "train.init_weights": "runs/b/weights/best.pt",
        }
    )
    with pytest.raises(ConfigError, match="train.init_weights are different things"):
        cfg.validate()


def test_init_weights_alone_validates() -> None:
    cfg = RunConfig(task="depth", model="midas_small").with_overrides(
        {"data.root": "/data/SyMTRS", "train.init_weights": "runs/b/weights/best.pt"}
    )
    cfg.validate()


# ------------------------------------------------------------------------ the loading


@requires_torch
def test_weights_cross_over_and_no_run_state_does(tmp_path: Path) -> None:
    import torch

    from symtrs.engine.trainer import _load_init_weights

    donor = torch.nn.Conv2d(3, 4, 1)
    with torch.no_grad():
        donor.weight.fill_(2.0)
        donor.bias.fill_(3.0)
    path = tmp_path / "best.pt"
    # A real checkpoint carries the donor's run state. None of it may follow the weights.
    torch.save({"model": donor.state_dict(), "epoch": 19, "best": 0.1234}, path)

    cfg = RunConfig(task="depth", model="midas_small").with_overrides(
        {"train.init_weights": str(path)}
    )
    target = torch.nn.Conv2d(3, 4, 1)
    with torch.no_grad():
        target.weight.zero_()
    assert _load_init_weights(cfg, target) is None
    assert torch.equal(target.weight, donor.weight) and torch.equal(target.bias, donor.bias)


@requires_torch
def test_a_mismatched_checkpoint_names_the_file_and_the_key_count(tmp_path: Path) -> None:
    import torch

    from symtrs.engine.trainer import _load_init_weights

    donor = torch.nn.Conv2d(3, 4, 1)
    path = tmp_path / "wrong.pt"
    torch.save({"model": donor.state_dict()}, path)
    cfg = RunConfig(task="depth", model="midas_small").with_overrides(
        {"train.init_weights": str(path)}
    )
    with pytest.raises(ConfigError, match="does not match midas_small"):
        _load_init_weights(cfg, torch.nn.Conv2d(3, 7, 1))


@requires_torch
def test_a_missing_checkpoint_names_the_path(tmp_path: Path) -> None:
    import torch

    from symtrs.engine.trainer import _load_init_weights

    missing = tmp_path / "nope.pt"
    cfg = RunConfig(task="depth", model="midas_small").with_overrides(
        {"train.init_weights": str(missing)}
    )
    with pytest.raises(FileNotFoundError, match="nothing to initialise from"):
        _load_init_weights(cfg, torch.nn.Conv2d(3, 4, 1))


def test_the_flag_is_on_the_train_command() -> None:
    """`--init-weights` reaches `train.init_weights` through the shared flag table."""
    from symtrs.cli import build_parser

    args = build_parser().parse_args(
        ["train", "configs/depth/256/midas_small_256.yaml", "--init-weights", "runs/a/weights/best.pt"]
    )
    assert args.init_weights == "runs/a/weights/best.pt"
