"""Super-resolution: the config -> command-line mapping and the run summary.

The models themselves live in `symtrs/legacy/superres/` and are not re-tested here. What
is tested is the part `symtrs` owns: that a resolution maps to the right pair of folders,
that the scale is derived rather than configured, and that the summary a run writes is the
shape `symtrs tables` reads.

Torch is not needed for any of this - the module builds a command line, it does not build
a network - so these run on a machine that is only checking the dataset layout.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from symtrs.config import RunConfig
from symtrs.model.superres import (
    SCALE_FOR_RESOLUTION,
    SR_MODELS,
    SuperResolution,
    SuperResolutionError,
    hr_dir,
    lr_dir,
    runner,
    scale_for,
    summary_from_run,
)
from symtrs.registry import build_model, get_spec


def _config(root: Path, resolution: int = 512, model: str = "srcnn") -> RunConfig:
    return RunConfig(task="superresolution", model=model, resolution=resolution).with_overrides(
        {"data.root": str(root)}
    ).validate()


# ------------------------------------------------------------------ the scale mapping


def test_scale_is_derived_from_the_resolution() -> None:
    assert scale_for(2048) == 1
    assert scale_for(1024) == 2
    assert scale_for(512) == 4
    assert scale_for(256) == 8


def test_scale_covers_every_resolution_the_config_schema_accepts() -> None:
    from symtrs.config import RESOLUTIONS

    assert set(SCALE_FOR_RESOLUTION) == set(RESOLUTIONS)
    assert SCALE_FOR_RESOLUTION == {2048: 1, 1024: 2, 512: 4, 256: 8}


def test_an_unknown_resolution_is_refused_by_name() -> None:
    with pytest.raises(SuperResolutionError) as excinfo:
        scale_for(384)
    assert "384" in str(excinfo.value)


# ------------------------------------------------------------------- the folder mapping


def test_lr_folder_follows_the_resolution(layout: Path) -> None:
    assert lr_dir(str(layout), "train", 2048) == layout / "train" / "rgb_day"
    assert lr_dir(str(layout), "train", 1024) == layout / "train" / "rgb_day_lr" / "x2"
    assert lr_dir(str(layout), "train", 512) == layout / "train" / "rgb_day_lr" / "x4"
    assert lr_dir(str(layout), "train", 256) == layout / "train" / "rgb_day_lr" / "x8"


def test_hr_folder_is_always_native(layout: Path) -> None:
    for resolution in (2048, 1024, 512, 256):
        assert hr_dir(str(layout), "train") == layout / "train" / "rgb_day"
        assert hr_dir(str(layout), "test") == layout / "test" / "rgb_day"


def test_x1_reads_the_same_folder_twice(layout: Path) -> None:
    """resolution: 2048 means x1: a refiner, not an upsampler. It has to say so."""
    assert lr_dir(str(layout), "train", 2048) == hr_dir(str(layout), "train")


# ------------------------------------------------------------------------- the command


def test_command_carries_the_scale_and_both_folder_pairs(layout: Path) -> None:
    cfg = _config(layout, resolution=512)
    command = build_model("srcnn").train_command(cfg, layout / "run")

    def value_of(flag: str) -> str:
        return command[command.index(flag) + 1]

    assert value_of("--scale") == "4"
    assert value_of("--lr_root") == str(layout / "train" / "rgb_day_lr" / "x4")
    assert value_of("--hr_root") == str(layout / "train" / "rgb_day")
    assert value_of("--test_lr_root") == str(layout / "test" / "rgb_day_lr" / "x4")
    assert value_of("--test_hr_root") == str(layout / "test" / "rgb_day")
    assert value_of("--model") == "srcnn"
    assert value_of("--out_dir") == str(layout / "run")
    # Launched as a module of this package, never imported: the legacy loop owns its own
    # RNG, matplotlib backend and multiprocessing start-up.
    assert command[:3] == [sys.executable, "-m", "symtrs.legacy.superres.train"]


def test_the_trainer_is_launched_as_a_module_of_this_package(layout: Path) -> None:
    """A module of this package, not a script beside it.

    Pointing at a file in the repository root meant the task could not run from an
    installed `symtrs` at all - the script is not in the distribution, and `symtrs` is
    meant to ship its tasks. The module is in the distribution.
    """
    command = build_model("srcnn").train_command(_config(layout), layout / "run")
    assert command[:3] == [sys.executable, "-m", runner.TRAIN_MODULE]
    assert not any(token.endswith(".py") for token in command), command


def test_the_front_end_always_passes_the_run_dir(layout: Path) -> None:
    """`--out_dir` defaults to a path inside the package in the trainer now.

    `symtrs` always passes its own run directory, so the default is only ever reached by a
    hand-run; this pins that the front-end does not rely on it.
    """
    command = build_model("srcnn").train_command(_config(layout), layout / "run")
    assert command[command.index("--out_dir") + 1] == str(layout / "run")


def test_a_missing_trainer_is_named_with_its_fix(monkeypatch, layout: Path) -> None:
    """A stripped install has to say what to do, not raise from a child process."""
    assert runner.trainer_missing() is False  # it ships with the package
    monkeypatch.setattr(runner, "trainer_missing", lambda: True)
    with pytest.raises(SuperResolutionError) as excinfo:
        build_model("srcnn").train_command(_config(layout), layout / "run")
    message = str(excinfo.value)
    assert runner.TRAIN_MODULE in message
    assert "pip install -e ." in message


def test_test_roots_are_skipped_when_the_split_is_absent(tmp_path: Path, capsys) -> None:
    """A training run without a test split still runs - but it has to say so."""
    root = tmp_path / "no-test"
    (root / "train" / "rgb_day").mkdir(parents=True)
    (root / "train" / "rgb_day_lr" / "x4").mkdir(parents=True)

    command = build_model("srcnn").train_command(_config(root), root / "run")

    assert "--test_lr_root" not in command
    assert "validation number only" in capsys.readouterr().out


def test_missing_test_split_is_fatal_when_scoring(tmp_path: Path) -> None:
    """The mirror case: on `eval` the test split *is* the command, so its absence is fatal."""
    root = tmp_path / "no-test"
    (root / "train" / "rgb_day").mkdir(parents=True)
    (root / "train" / "rgb_day_lr" / "x4").mkdir(parents=True)

    with pytest.raises(SuperResolutionError) as excinfo:
        build_model("srcnn").train_command(
            _config(root), root / "run", eval_only=True
        )
    assert "nothing to score" in str(excinfo.value)


def test_swinir_flags_are_confined_to_swinir(layout: Path) -> None:
    """The script implements crops and AMP for SwinIR only; passing them elsewhere lies."""
    srcnn = build_model("srcnn").train_command(_config(layout), layout / "run")
    swinir = build_model("swinir").train_command(
        _config(layout, model="swinir"), layout / "run"
    )

    assert "--swinir_embed_dim" not in srcnn
    assert "--amp" not in srcnn
    assert "--patch_size" not in srcnn

    assert "--swinir_embed_dim" in swinir
    assert "--swinir_depths" in swinir


def test_resume_becomes_the_scripts_weights_flag(layout: Path) -> None:
    cfg = _config(layout).with_overrides({"train.resume": "/runs/old/weights/best.pt"})
    command = build_model("srcnn").train_command(cfg, layout / "run")
    assert command[command.index("--weights") + 1] == "/runs/old/weights/best.pt"


def test_gpu_ids_override_the_gpu_count(layout: Path) -> None:
    command = build_model("srcnn", gpu_ids="0,1").train_command(
        _config(layout), layout / "run"
    )
    assert command[command.index("--gpus") + 1] == "2"


# ----------------------------------------------------------------------- the summary


def test_summary_is_shaped_for_symtrs_tables(layout: Path) -> None:
    cfg = _config(layout, resolution=512)
    run_dir = layout / "run"
    (run_dir / "weights").mkdir(parents=True)
    (run_dir / "weights" / "best.pt").write_bytes(b"")
    (run_dir / "test_metrics.json").write_text(
        json.dumps({"psnr": 24.5, "ssim": 0.81, "mse": 0.003, "images": 40}),
        encoding="utf-8",
    )

    summary = summary_from_run(cfg, run_dir, 0, ["python", "script"], mode="finetune")

    assert summary["task"] == "superresolution"
    assert summary["resolution"] == 512
    assert summary["scale"] == 4
    assert summary["test"]["psnr"] == 24.5
    assert summary["checkpoint"] == str(run_dir / "weights" / "best.pt")


def test_a_run_without_test_metrics_has_no_test_key(layout: Path) -> None:
    """Absent rather than empty, so a missing number is visibly missing in `tables`."""
    summary = summary_from_run(_config(layout), layout / "run", 0, ["python", "script"])
    assert "test" not in summary


# ---------------------------------------------------------------------- the description


def test_x1_says_it_is_not_super_resolution() -> None:
    described = build_model("srcnn").describe_cell(2048)
    assert "x1 refiner" in described
    assert "not super-resolution" in described


def test_higher_scales_name_both_folders() -> None:
    described = build_model("srgan").describe_cell(512)
    assert "x4" in described
    assert "rgb_day_lr" in described


# ---------------------------------------------------------------------------- registry


def test_every_super_resolution_model_is_registered() -> None:
    for key in SR_MODELS:
        spec = get_spec(key)
        assert spec.task == "superresolution", key
        assert "SuperResolution" in spec.target, key


def test_the_registry_and_the_trainer_agree_on_the_model_set() -> None:
    """The trainer has its own `choices=`; the two lists must not drift.

    Read from where the trainer lives now, not from the root shim: the shim has no model
    set to compare against, so a test pointed at it would pass while `choices=` drifted.
    `utf-8-sig` because the moved files kept their BOMs.
    """
    trainer = (
        Path(__file__).resolve().parents[1] / "symtrs" / "legacy" / "superres" / "train.py"
    ).read_text(encoding="utf-8-sig")
    for key in SR_MODELS:
        assert f'"{key}"' in trainer, key


def test_a_zero_shot_sweep_drops_the_super_resolution_row() -> None:
    """A zero-shot entry could only fail: there is nothing to load until a run exists."""
    from symtrs.commands.sweep import build_plan

    plan = build_plan("superresolution", ["srcnn"], [512], ["zero_shot", "finetune"])
    assert [entry["mode"] for entry in plan] == ["finetune"]


def test_zero_shot_is_still_planned_for_depth() -> None:
    """The exclusion is about released weights, not about the mode being unused."""
    from symtrs.commands.sweep import build_plan

    plan = build_plan("depth", ["midas_small"], [512], ["zero_shot", "finetune"])
    assert sorted(entry["mode"] for entry in plan) == ["finetune", "zero_shot"]


def test_zero_shot_is_refused_with_a_reason() -> None:
    """There are no released SyMTRS upsampling weights, so the mode cannot mean anything."""
    import inspect

    from symtrs.model.superres import run_super_resolution

    source = inspect.getsource(run_super_resolution)
    assert 'mode == "zero_shot"' in source
    assert "released SyMTRS" in source
