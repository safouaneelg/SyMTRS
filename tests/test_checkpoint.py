"""The self-describing checkpoint, and the SDK path back out of it.

This file guards the piece that makes the Python SDK work at all: `autoloader("model.pt")`
rebuilds an architecture and reproduces its postprocessing from the metadata inside the
checkpoint, with no training YAML present. If the metadata block loses a field, nothing
raises - the model is rebuilt with a default and the numbers are quietly wrong. So the
fields are checked one by one against the config that produced them.

Rebuilding a real network downloads pretrained weights, so the tests that would do that
are marked `network` and are not run by default. Everything here is about the payload.

Torch is needed, so this module skips on a machine without it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="torch is not installed")

from symtrs import CONFIG_FORMAT_VERSION  # noqa: E402
from symtrs.common.checkpoint import (  # noqa: E402
    CHECKPOINT_KEY,
    CheckpointError,
    is_symtrs_checkpoint,
    load_checkpoint,
    read_checkpoint_metadata,
    resolve_checkpoint_path,
    save_checkpoint,
)
from symtrs.config import RunConfig  # noqa: E402


def _config(**kwargs) -> RunConfig:
    payload = {
        "task": "depth",
        "model": "midas_dpt_hybrid",
        "resolution": 1024,
        "data": {"root": "/data/SyMTRS"},
    }
    payload.update(kwargs)
    return RunConfig.from_dict(payload).validate()


def _state() -> dict:
    return {"model": {"weight": torch.zeros(2, 3)}, "epoch": 3, "best": 0.5, "metrics": {"rmse": 1.0}}


# --------------------------------------------------------------------- the payload


def test_saving_stamps_the_format_version_so_an_old_checkpoint_is_detectable(tmp_path: Path):
    path = tmp_path / "weights" / "best.pt"
    save_checkpoint(str(path), _state(), {"task": "depth", "model_key": "midas_dpt_hybrid"})

    block = read_checkpoint_metadata(str(path))
    assert block["format_version"] == CONFIG_FORMAT_VERSION
    assert block["task"] == "depth"


def test_an_explicit_format_version_is_not_overwritten(tmp_path: Path):
    path = tmp_path / "best.pt"
    save_checkpoint(str(path), _state(), {"format_version": 99, "task": "depth"})
    assert read_checkpoint_metadata(str(path))["format_version"] == 99


def test_saving_creates_the_weights_directory(tmp_path: Path):
    """The trainer writes `weights/best.pt` before it has ever made the folder itself."""
    path = tmp_path / "runs" / "deep" / "weights" / "best.pt"
    save_checkpoint(str(path), _state(), {"task": "depth"})
    assert path.is_file()


def test_the_write_leaves_no_temporary_file_behind(tmp_path: Path):
    """A half-written `best.pt` is worse than no checkpoint: it is a file that exists and
    cannot be read, and a resumed run picks it up."""
    path = tmp_path / "best.pt"
    save_checkpoint(str(path), _state(), {"task": "depth"})

    assert not Path(f"{path}.tmp").exists()
    assert [p.name for p in tmp_path.iterdir()] == ["best.pt"]


def test_the_state_dict_survives_the_round_trip(tmp_path: Path):
    path = tmp_path / "best.pt"
    save_checkpoint(str(path), _state(), {"task": "depth"})

    payload = load_checkpoint(str(path))
    assert torch.equal(payload["model"]["weight"], torch.zeros(2, 3))
    assert payload["epoch"] == 3
    assert payload["metrics"] == {"rmse": 1.0}
    assert CHECKPOINT_KEY in payload


def test_a_plain_torch_file_is_reported_as_not_a_symtrs_checkpoint(tmp_path: Path):
    """Upstream CycleGAN checkpoints are plain `torch.save` dicts with a `G_A` key. The
    error has to say where they belong instead, because the obvious next move - passing
    one to `autoloader` - would otherwise fail three frames down in a KeyError."""
    path = tmp_path / "latest_net_G_A.pth"
    torch.save({"G_A": torch.zeros(1)}, str(path))

    assert is_symtrs_checkpoint(str(path)) is False
    with pytest.raises(CheckpointError, match="domain_adapt"):
        read_checkpoint_metadata(str(path))


def test_a_missing_checkpoint_names_the_path(tmp_path: Path):
    with pytest.raises(CheckpointError, match="checkpoint not found"):
        load_checkpoint(str(tmp_path / "absent.pt"))
    assert is_symtrs_checkpoint(str(tmp_path / "absent.pt")) is False


# --------------------------------------------------------------- path resolution


def test_a_run_directory_resolves_to_best_then_last(tmp_path: Path):
    """`autoloader(run_dir)` is the documented shorthand, and `best.pt` is the one a
    result table reports, so it has to win over `last.pt`."""
    run = tmp_path / "runs" / "depth" / "midas_dpt_hybrid" / "1024"
    (run / "weights").mkdir(parents=True)
    (run / "weights" / "last.pt").write_bytes(b"")
    (run / "weights" / "best.pt").write_bytes(b"")

    assert resolve_checkpoint_path(str(run)) == str(run / "weights" / "best.pt")

    (run / "weights" / "best.pt").unlink()
    assert resolve_checkpoint_path(str(run)) == str(run / "weights" / "last.pt")


def test_a_file_resolves_to_itself_and_an_empty_directory_to_nothing(tmp_path: Path):
    path = tmp_path / "best.pt"
    path.write_bytes(b"")
    assert resolve_checkpoint_path(str(path)) == str(path)
    assert resolve_checkpoint_path(str(tmp_path / "empty_run_dir")) is None
    assert resolve_checkpoint_path(str(tmp_path / "no_such_thing")) is None


# ------------------------------------------- the metadata the autoloader rebuilds from


def test_the_metadata_carries_everything_autoloader_needs_without_the_yaml(tmp_path: Path):
    """Field by field, against the config that produced it. A missing field here is a
    model silently rebuilt with a default."""
    from symtrs.engine.trainer import checkpoint_metadata

    cfg = _config(resolution=512, model_kwargs={"variant": "K"}, train={"freeze": "encoder"})

    class _Stub:
        protocol = "relative"
        default_align = "scale"

    metadata = checkpoint_metadata(_Stub(), cfg, {"metrics": {"rmse": 1.0}})

    assert metadata["task"] == "depth"
    assert metadata["model_key"] == "midas_dpt_hybrid"
    assert metadata["resolution"] == 512
    assert metadata["model_kwargs"] == {"variant": "K"}
    assert metadata["protocol"] == "relative"
    assert metadata["default_align"] == "scale"
    assert metadata["metrics"] == {"rmse": 1.0}


def test_the_config_inside_the_metadata_reparses_to_the_same_run(tmp_path: Path):
    """`autoloader` does `RunConfig.from_dict(metadata["config"])`. If that round trip is
    lossy, evaluating a checkpoint happens at the wrong resolution or against the wrong
    dataset without anything saying so."""
    from symtrs.engine.trainer import checkpoint_metadata

    cfg = _config(
        resolution=256,
        train={"epochs": 12, "loss": "berhu", "loss_kwargs": {"threshold": 0.2}},
        eval={"align": "median", "tile_size": 512, "tile_overlap": 64},
    )
    metadata = checkpoint_metadata(object(), cfg)

    path = tmp_path / "best.pt"
    save_checkpoint(str(path), _state(), metadata)
    restored = RunConfig.from_dict(read_checkpoint_metadata(str(path))["config"]).validate()

    assert restored.to_dict() == cfg.to_dict()
    assert restored.resolution == 256
    assert restored.eval.align == "median"
    assert restored.train.loss_kwargs == {"threshold": 0.2}


def test_segmentation_metadata_records_the_class_count():
    """A checkpoint that forgets `num_classes` rebuilds a 21-class VOC head and then loads
    a 6-class state dict into it - which `strict=False` reports as a warning and carries
    on from, with every class wrong."""
    from symtrs.engine.trainer import checkpoint_metadata

    cfg = _config(
        task="segmentation", model="segformer_b2", data={"root": "/data/SyMTRS", "num_classes": 6}
    )

    class _Stub:
        class_names = ["ground", "road", "building", "veg", "water", "sky"]

    metadata = checkpoint_metadata(_Stub(), cfg)
    assert metadata["num_classes"] == 6
    assert metadata["ignore_index"] == 255
    assert len(metadata["class_names"]) == 6


def test_a_non_depth_model_gets_no_protocol_field(tmp_path: Path):
    """`protocol` drives which loss and which alignment are applied. A segmentation model
    claiming one would be scored as a depth model."""
    from symtrs.engine.trainer import checkpoint_metadata

    class _Stub:  # no `protocol` attribute at all
        pass

    metadata = checkpoint_metadata(_Stub(), _config(task="segmentation", model="unet_small"))
    assert "protocol" not in metadata


# ------------------------------------------------------- reading the flags back in


def _namespace(**kwargs):
    import argparse

    return argparse.Namespace(**kwargs)


def test_a_checkpoint_passed_to_model_is_read_as_a_checkpoint(capsys):
    """`--model runs/seg/weights/best.pt` is the natural mistake. Left as a model key it
    becomes a run directory whose path is that file with `/2048` appended, and the reader
    gets `NotADirectoryError: .../best.pt/2048`, which names neither the flag nor the fix.
    """
    from symtrs.commands._common import redirect_checkpoint_flag

    args = _namespace(model="runs/seg/weights/best.pt", checkpoint="")
    redirect_checkpoint_flag(args)

    assert args.checkpoint == "runs/seg/weights/best.pt"
    assert args.model is None
    assert "--checkpoint" in capsys.readouterr().out


def test_a_registry_key_is_left_alone(capsys):
    from symtrs.commands._common import redirect_checkpoint_flag

    for key in ("midas_dpt_hybrid", "zoedepth_k", "segformer_b2", "unet_small"):
        args = _namespace(model=key, checkpoint="")
        redirect_checkpoint_flag(args)
        assert args.model == key, key
        assert args.checkpoint == ""
    assert capsys.readouterr().out == ""


def test_an_explicit_checkpoint_wins_over_model(capsys):
    """So `--model` can still name the architecture to rebuild a checkpoint into."""
    from symtrs.commands._common import redirect_checkpoint_flag

    args = _namespace(model="segformer_b2", checkpoint="runs/seg/weights/best.pt")
    redirect_checkpoint_flag(args)

    assert args.model == "segformer_b2"
    assert args.checkpoint == "runs/seg/weights/best.pt"
    assert capsys.readouterr().out == ""


def test_a_command_without_a_checkpoint_flag_refuses_instead_of_blanking_model():
    """`train` has no `--checkpoint`, so there is nowhere to redirect to. Blanking `--model`
    anyway would surface `model is required` and hide that the value was a file; letting it
    through as a key gives `NotADirectoryError: .../best.pt/256/weights`. `train` does have
    `--resume`, so the useful answer names it."""
    from symtrs.commands._common import redirect_checkpoint_flag
    from symtrs.config import ConfigError

    args = _namespace(model="runs/seg/weights/best.pt", resume="")
    with pytest.raises(ConfigError) as caught:
        redirect_checkpoint_flag(args)

    message = str(caught.value)
    assert "--model takes a registry key" in message
    assert "--resume runs/seg/weights/best.pt" in message
    assert "eval --checkpoint runs/seg/weights/best.pt" in message
    assert args.model == "runs/seg/weights/best.pt"  # left as it was, not silently cleared


@pytest.mark.parametrize(
    "value,expected",
    [
        ("midas_dpt_hybrid", False),
        ("unet_small", False),
        ("segformer_b0", False),
        ("best.pt", True),
        ("weights/best.pt", True),
        ("runs\\seg\\best.pt", True),
        ("./run", True),
        ("", False),
        (None, False),
    ],
)
def test_what_counts_as_a_path_rather_than_a_key(value, expected):
    """A bare identifier is a key; a separator or a weight extension is a path. Checked
    without touching the filesystem, so `--dry-run` and a not-yet-downloaded checkpoint
    both still work, and a genuinely unknown key still reaches the registry."""
    from symtrs.commands._common import looks_like_a_checkpoint_path

    assert looks_like_a_checkpoint_path(value) is expected


def _trained_checkpoint(tmp_path: Path, **kwargs) -> Path:
    from symtrs.engine.trainer import checkpoint_metadata

    cfg = _config(**kwargs)
    metadata = checkpoint_metadata(object(), cfg)
    path = tmp_path / "weights" / "best.pt"
    save_checkpoint(str(path), _state(), metadata)
    return path


def test_a_lone_checkpoint_supplies_the_model_the_task_and_the_resolution(tmp_path: Path, capsys):
    """`symtrs eval --checkpoint runs/.../best.pt` is the documented invocation, with no
    `--model`. Without this it fails validation on a missing model - and, worse, a
    checkpoint trained at 1024 would be evaluated at the default 2048, which is a
    different point on the axis this benchmark is about."""
    from symtrs.commands._common import adopt_checkpoint_settings

    path = _trained_checkpoint(tmp_path, resolution=1024, model="midas_dpt_hybrid")

    cfg = adopt_checkpoint_settings(RunConfig(), _namespace(checkpoint=str(path)), {})
    assert cfg.model == "midas_dpt_hybrid"
    assert cfg.task == "depth"
    assert cfg.resolution == 1024
    assert "taking" in capsys.readouterr().out


def test_an_explicit_flag_still_wins_over_the_checkpoint(tmp_path: Path):
    """`overrides` is the caller's record of what the command line already stated, so it is
    read, never re-applied. The config handed in already carries those values - which is
    what `build_config` does - and the checkpoint must not overwrite them."""
    from symtrs.commands._common import adopt_checkpoint_settings

    path = _trained_checkpoint(tmp_path, resolution=1024, model="midas_dpt_hybrid")

    stated = {"resolution": 256, "task": "segmentation"}
    base = RunConfig().with_overrides(stated)
    cfg = adopt_checkpoint_settings(base, _namespace(checkpoint=str(path)), stated)

    assert cfg.resolution == 256  # the caller asked for this, the checkpoint did not get a vote
    assert cfg.task == "segmentation"
    assert cfg.model == "midas_dpt_hybrid"  # nothing else said which model it is


def test_the_class_count_is_taken_from_a_segmentation_checkpoint(tmp_path: Path):
    """Scoring a 6-class head against a 21-class label space counts every label as wrong,
    and nothing about the run says so."""
    from symtrs.commands._common import adopt_checkpoint_settings

    path = _trained_checkpoint(
        tmp_path, task="segmentation", model="unet_small",
        data={"root": "/data/SyMTRS", "num_classes": 6},
    )

    cfg = adopt_checkpoint_settings(RunConfig(), _namespace(checkpoint=str(path)), {})
    assert cfg.data.num_classes == 6
    assert cfg.data.ignore_index == 255


def test_a_plain_torch_file_is_left_for_the_loader_to_explain(tmp_path: Path, capsys):
    """A CycleGAN checkpoint has no metadata block to read. Adopting nothing and letting
    `autoloader` report it keeps one clear error instead of two confusing ones."""
    from symtrs.commands._common import adopt_checkpoint_settings

    path = tmp_path / "latest_net_G_A.pth"
    torch.save({"G_A": torch.zeros(1)}, str(path))

    cfg = adopt_checkpoint_settings(RunConfig(), _namespace(checkpoint=str(path)), {})
    assert cfg.model == ""
    assert capsys.readouterr().out == ""
