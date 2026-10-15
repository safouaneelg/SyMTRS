"""The run config: one schema behind YAML, the command line and Python kwargs.

The thing these tests are really protecting is that all three front-ends produce the same
`RunConfig`. When they drift, a YAML run and the equivalent CLI invocation silently train
different things, and nobody notices until the numbers disagree.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from symtrs.config import (
    MODES,
    RESOLUTIONS,
    TASKS,
    ConfigError,
    DataConfig,
    EvalConfig,
    RunConfig,
    TrainConfig,
    load_config,
)


def _base(**kwargs) -> RunConfig:
    """A minimal valid config, so each test only has to state what it is testing."""
    payload = {"task": "depth", "model": "midas_dpt_hybrid", "data": {"root": "/data/SyMTRS"}}
    payload.update(kwargs)
    return RunConfig.from_dict(payload).validate()


def test_the_task_list_is_exactly_what_the_pipeline_implements():
    """`TASKS` gates `--task`, so it must not advertise a task nothing can run.

    `superresolution` is in the list because it is a real SyMTRS task with its own
    pre-existing script, but it is deliberately *not* implemented by this pipeline.
    `symtrs.engine.trainer` raises a named error for it rather than building a depth
    model and calling it super-resolution; this assertion pins that arrangement so that
    adding a task to the list is a deliberate act.
    """
    assert set(TASKS) == {"superresolution", "depth", "segmentation", "domain_adapt"}


def test_defaults_validate():
    assert _base().resolution == 2048


@pytest.mark.parametrize("resolution", RESOLUTIONS)
def test_every_named_resolution_is_accepted(resolution):
    assert _base(resolution=resolution).resolution == resolution


def test_a_resolution_between_the_named_ones_is_refused():
    with pytest.raises(ConfigError, match="resolution must be one of"):
        _base(resolution=1536)


def test_train_may_not_equal_a_validation_split():
    """Otherwise the checkpoint selected as "best" was selected on data it trained on."""
    for split in ("val", "test"):
        with pytest.raises(ConfigError, match="must differ"):
            _base(data={"root": "/data/SyMTRS", "split": split})


def test_model_is_required():
    with pytest.raises(ConfigError, match="model is required"):
        RunConfig.from_dict({"task": "depth", "data": {"root": "/x"}}).validate()


def test_unknown_top_level_key_is_named_with_the_known_ones():
    with pytest.raises(ConfigError) as info:
        RunConfig.from_dict({"task": "depth", "model": "m", "epochs": 10})
    assert "epochs" in str(info.value)
    assert "train" in str(info.value)  # tells the caller where epochs actually lives


def test_unknown_section_key_is_named():
    with pytest.raises(ConfigError, match="unknown keys in 'train'"):
        RunConfig.from_dict(
            {"task": "depth", "model": "m", "train": {"learning_rate": 1e-4}}
        )


def test_train_section_validation():
    with pytest.raises(ConfigError, match="train.epochs"):
        _base(train={"epochs": 0})
    with pytest.raises(ConfigError, match="train.optim"):
        _base(train={"optim": "rmsprop"})
    with pytest.raises(ConfigError, match="train.freeze"):
        _base(train={"freeze": "everything"})


@pytest.mark.parametrize("freeze", ["", "encoder", "head", "last_n:3"])
def test_freeze_policies_that_are_implemented(freeze):
    assert _base(train={"freeze": freeze}).train.freeze == freeze


def test_eval_tile_overlap_must_be_smaller_than_the_tile():
    """Equal values would make the sliding window advance by zero and never terminate."""
    assert _base(eval={"tile_size": 512, "tile_overlap": 128}).eval.tile_size == 512
    with pytest.raises(ConfigError, match="tile_overlap must be smaller"):
        _base(eval={"tile_size": 512, "tile_overlap": 512})


def test_depth_backend_comes_from_the_reader_so_the_two_cannot_disagree():
    from symtrs.data.exr import VALID_BACKENDS

    assert _base(data={"root": "/x", "depth_backend": "numpy"}).data.depth_backend == "numpy"
    with pytest.raises(ConfigError, match="depth_backend"):
        _base(data={"root": "/x", "depth_backend": "tiff"})
    assert set(VALID_BACKENDS) >= {"auto", "numpy"}


def test_num_classes_must_leave_room_for_a_decision():
    with pytest.raises(ConfigError, match="num_classes"):
        _base(data={"root": "/x", "num_classes": 1})


# ------------------------------------------------------------------ yaml round trip


def test_yaml_round_trip_is_lossless(tmp_path: Path):
    original = _base(
        resolution=1024,
        model_kwargs={"variant": "K", "direction": "day2night"},
        upstream_args=["--lambda_A", "10"],
        train={"epochs": 7, "lr": 3e-5, "loss_kwargs": {"aux_weight": 0.3}},
        eval={"metrics": ["rmse", "d1"], "tile_size": 512, "tile_overlap": 64},
    )
    path = tmp_path / "run.yaml"
    original.to_yaml(str(path))

    reloaded = RunConfig.from_yaml(str(path))
    assert reloaded.to_dict() == original.to_dict()
    # The nested free-form sections are the easy thing to lose here.
    assert reloaded.model_kwargs == {"variant": "K", "direction": "day2night"}
    assert reloaded.train.loss_kwargs == {"aux_weight": 0.3}
    assert reloaded.eval.metrics == ["rmse", "d1"]


def test_empty_config_file_is_reported_not_silently_defaulted(tmp_path: Path):
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="empty"):
        RunConfig.from_yaml(str(path))


def test_shipped_example_configs_all_load():
    """Every YAML in `configs/` is a promise that that run works. Keep them honest."""
    configs = sorted((Path(__file__).resolve().parents[1] / "configs").glob("*.yaml"))
    assert configs, "no example configs found"
    for path in configs:
        cfg = RunConfig.from_yaml(str(path))
        assert cfg.model, f"{path.name} does not name a model"
        assert cfg.data.root, f"{path.name} does not name a dataset root"


# ----------------------------------------------------------------- overrides


def test_dotted_override_reaches_a_nested_field():
    cfg = _base().with_overrides({"train.epochs": 30, "data.num_workers": 2})
    assert cfg.train.epochs == 30
    assert cfg.data.num_workers == 2


def test_override_values_are_coerced_to_the_field_type():
    """The CLI hands over strings; `train.epochs` is an int and has to stay one."""
    cfg = _base().with_overrides(
        {
            "train.epochs": "12",
            "train.lr": "0.0002",
            "train.amp": "false",
            "eval.metrics": "rmse,d1",
        }
    )
    assert cfg.train.epochs == 12 and isinstance(cfg.train.epochs, int)
    assert cfg.train.lr == pytest.approx(2e-4)
    assert cfg.train.amp is False
    assert cfg.eval.metrics == ["rmse", "d1"]


def test_boolean_coercion_refuses_a_non_boolean_word():
    with pytest.raises(ConfigError, match="as a boolean"):
        _base().with_overrides({"train.amp": "maybe"})


def test_a_nested_mapping_cannot_replace_a_section():
    """`{"train": {"epochs": 20}}` is the first thing an SDK user reaches for, because the
    YAML file they just read is nested. Assigning it put a plain dict where a section object
    belongs, and nothing complained until `validate()` ran several frames later:

        AttributeError: 'dict' object has no attribute 'validate'

    which names neither the key nor the fix. The message has to name both.
    """
    with pytest.raises(ConfigError) as caught:
        _base().with_overrides({"train": {"epochs": 20, "lr": 1e-5}})

    message = str(caught.value)
    assert "'train' is a config section" in message
    assert "train.epochs" in message and "train.lr" in message


def test_a_mapping_value_inside_a_free_form_section_still_works():
    """The counterpart to the test above, and the reason the check is on the *field type*
    rather than on "a mapping was passed": inside `loss_kwargs` a mapping is a legitimate
    value, and flattening it would rewrite `class_weights` into `class_weights.0`.
    """
    cfg = _base().with_overrides({"train.loss_kwargs.class_weights": {0: 1.0, 5: 4.0}})
    assert cfg.train.loss_kwargs["class_weights"] == {0: 1.0, 5: 4.0}


def test_unknown_dotted_key_is_refused():
    with pytest.raises(ConfigError, match="unknown config key"):
        _base().with_overrides({"train.learning_rate": 1e-4})


def test_none_overrides_are_ignored():
    """So a whole parsed argparse namespace can be passed without filtering it first."""
    cfg = _base().with_overrides({"train.epochs": None, "model": None})
    assert cfg.train.epochs == TrainConfig().epochs
    assert cfg.model == "midas_dpt_hybrid"


def test_override_does_not_mutate_the_original():
    original = _base()
    original.with_overrides({"train.epochs": 99})
    assert original.train.epochs == TrainConfig().epochs


# ----------------------------------------------------- free-form sections


def test_dotted_path_into_model_kwargs_creates_the_key():
    """`--set model_kwargs.direction=day2night` has to work on an empty dict."""
    cfg = _base().with_overrides({"model_kwargs.direction": "night2day"})
    assert cfg.model_kwargs["direction"] == "night2day"


def test_values_in_free_form_sections_are_type_guessed_conservatively():
    cfg = _base().with_overrides(
        {
            "model_kwargs.metric": "true",
            "model_kwargs.domain_id": "3",
            "model_kwargs.model_id": "depth-anything/DA3-LARGE",
            "train.loss_kwargs.threshold": "0.2",
        }
    )
    assert cfg.model_kwargs["metric"] is True
    assert cfg.model_kwargs["domain_id"] == 3
    # A model id must survive untouched: no numeric parse, no path mangling.
    assert cfg.model_kwargs["model_id"] == "depth-anything/DA3-LARGE"
    assert cfg.train.loss_kwargs["threshold"] == pytest.approx(0.2)


# ------------------------------------------------------------------ run directory


def test_run_dir_is_canonical_by_default_and_overridable():
    cfg = _base(resolution=512)
    assert cfg.resolved_out_dir == Path("runs/depth/midas_dpt_hybrid/512")
    assert _base(out_dir="/tmp/here").resolved_out_dir == Path("/tmp/here")


def test_a_derived_model_key_cannot_escape_the_run_directory():
    """A key containing a slash would nest the run path one level deeper than intended."""
    cfg = _base(model="da3_metric_large")
    parts = cfg.resolved_out_dir.parts
    assert parts[:3] == ("runs", "depth", "da3_metric_large")
    assert len(parts) == 4


# ---------------------------------------------------------------------- load_config


def test_load_config_layers_yaml_then_overrides(tmp_path: Path):
    path = tmp_path / "base.yaml"
    _base(resolution=2048, train={"epochs": 20}).to_yaml(str(path))

    cfg = load_config(str(path), {"train.epochs": 5}, resolution=256)
    assert cfg.train.epochs == 5
    assert cfg.resolution == 256
    assert cfg.model == "midas_dpt_hybrid"  # untouched from the YAML


def test_modes_are_exactly_the_two_the_trainer_handles():
    assert set(MODES) == {"zero_shot", "finetune"}


def test_dataclass_sections_have_independent_defaults():
    """`field(default_factory=...)` rather than a shared mutable default."""
    first = DataConfig()
    second = DataConfig()
    first.max_samples = 5
    assert second.max_samples == 0
    assert EvalConfig().metrics is not TrainConfig().__dict__ and EvalConfig().metrics == []
