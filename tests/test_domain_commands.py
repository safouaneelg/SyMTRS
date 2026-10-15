"""The argv the domain-adaptation front-end hands upstream must be argv upstream accepts.

Nothing in this package parses those flags. Upstream's own `argparse` does, inside a
subprocess, so a flag upstream has dropped does not fail at the call site: it fails after
the staging directory has been built, with upstream's traceback and exit code 2, which
reads like an upstream bug rather than a stale wrapper. That is exactly what happened -
`--gpu_ids` and `--results_dir` were both emitted for `train.py`, and this revision of
upstream defines neither. Reading the vendored option parsers is what turns that class of
mistake into a test.

Everything here is torch-free: it is string and path work over the vendored tree.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

from symtrs.model.domain import runner, upstream

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDORED = REPO_ROOT / "pytorch-CycleGAN-and-pix2pix"

#: `parser.add_argument("--flag"` / `add_argument('--flag'`, which is how every flag in
#: the vendored tree is declared - in `options/`, and in the model and dataset option
#: setters that extend the parser at run time.
ADD_ARGUMENT = re.compile(r"""add_argument\(\s*['"]--([A-Za-z0-9_]+)""")


def accepted_flags() -> set:
    """Every flag the vendored tree's parsers define.

    Scanned from source rather than by running `--help`: running the parser needs torch,
    a model class and a dataset mode, and this test needs to work on a machine that has
    none of them. `upstream_originals/` is excluded - those are the pre-modification
    copies kept for the delta, not code that runs.
    """
    flags = set()
    for path in VENDORED.rglob("*.py"):
        if "upstream_originals" in path.parts or "scripts" in path.parts:
            continue
        flags.update(ADD_ARGUMENT.findall(path.read_text(encoding="utf-8", errors="ignore")))
    return flags


def staging(tmp_path: Path) -> Path:
    return tmp_path / "staging" / "train"


def train_command(tmp_path: Path, direction: str = "day2night", **kwargs) -> list:
    return upstream.train_command(
        upstream=VENDORED,
        staging=staging(tmp_path),
        experiment="symtrs_cyclegan_day2night",
        direction=direction,
        dataset_mode="unaligned",
        upstream_model="cycle_gan",
        epochs=40,
        batch_size=1,
        checkpoints_dir=str(tmp_path / "checkpoints"),
        **kwargs,
    )


def test_the_vendored_parser_defines_every_flag_we_pass(tmp_path: Path) -> None:
    """No flag reaches upstream that upstream cannot parse."""
    accepted = accepted_flags()
    # A sanity floor: if the regex stops matching (a reformatted upstream), this test
    # would otherwise pass by accepting nothing.
    assert {"dataroot", "n_epochs", "phase"} <= accepted

    commands = {
        "train.py": train_command(tmp_path),
        "test.py": upstream.test_command(
            upstream=VENDORED,
            staging=staging(tmp_path),
            experiment="symtrs_cyclegan_day2night",
            split="test",
            dataset_mode="unaligned",
            upstream_model="cycle_gan",
            direction="day2night",
            checkpoints_dir=str(tmp_path / "checkpoints"),
            results_dir=str(tmp_path / "results"),
        ),
    }
    for script, command in commands.items():
        for token in command:
            if token.startswith("--"):
                assert token[2:] in accepted, (
                    f"{script} would reject {token!r}: the vendored options do not define "
                    f"it. Upstream's parser is the interface - read "
                    f"options/base_options.py rather than adding a flag back."
                )


def test_the_device_is_steered_through_the_environment_not_a_flag(tmp_path: Path) -> None:
    """Upstream chooses its own device, so there is no flag to pass.

    `util.util.init_ddp()` reads `WORLD_SIZE`/`LOCAL_RANK` under DDP and otherwise takes
    `cuda:0` when a GPU is visible and `cpu` when none is. `device_env` is therefore the
    only lever, and a `--device cpu` run that forgot it would train on the GPU.
    """
    assert not ({"gpu_ids", "gpus"} & accepted_flags())
    assert "--gpu_ids" not in train_command(tmp_path)

    assert runner.device_env("cpu") == {"CUDA_VISIBLE_DEVICES": ""}
    assert runner.device_env("cuda") == {}
    assert runner.device_env("cuda", gpu_ids="1") == {"CUDA_VISIBLE_DEVICES": "1"}
    # Under DDP the ranks choose their own devices; narrowing the visible set here would
    # put every rank on the same one.
    assert runner.device_env("cuda", gpu_ids="0,1", gpus=2) == {}


def test_multi_gpu_starts_upstream_under_torch_distributed_run(tmp_path: Path) -> None:
    single = train_command(tmp_path)
    multi = train_command(tmp_path, gpus=4)

    assert single[:2] == [single[0], str(VENDORED / "train.py")]
    assert multi[1:6] == ["-m", "torch.distributed.run", "--standalone", "--nproc_per_node", "4"]
    assert multi[6] == str(VENDORED / "train.py")
    # Same flags, whichever launcher: the GPU count is in the launch, not in an argument.
    assert multi[6:] == single[1:]


def test_train_writes_a_checkpoint_the_scoring_step_can_always_load(tmp_path: Path) -> None:
    """Upstream's `save_epoch_freq` default is 5, and the score step loads `latest`.

    A run shorter than five epochs - every smoke test - would then train, write no
    checkpoint, and fail at scoring with a `FileNotFoundError` that looks like a scoring
    bug. `--save_epoch_freq 1` is what makes "the weights the run ended with" always
    exist.
    """
    command = train_command(tmp_path)
    assert command[command.index("--save_epoch_freq") + 1] == "1"

    # A caller can thin the saving out again: upstream's parser takes the last value, and
    # extra_args are appended after ours.
    overridden = train_command(tmp_path, extra_args=["--save_epoch_freq", "10"])
    assert overridden[-2:] == ["--save_epoch_freq", "10"]


def test_both_scripts_are_told_which_direction_was_trained(tmp_path: Path) -> None:
    """`TestModel` builds its generator from the direction, so test.py needs it too.

    A night-to-day run scored with the default `AtoB` generates the day-to-night side:
    it completes, and the images it scores are not the ones the run was asked to learn.
    """
    assert upstream.upstream_direction("day2night") == "AtoB"
    assert upstream.upstream_direction("night2day") == "BtoA"

    train = train_command(tmp_path, direction="night2day")
    assert train[train.index("--direction") + 1] == "BtoA"

    test = upstream.test_command(
        upstream=VENDORED,
        staging=staging(tmp_path),
        experiment="symtrs_cyclegan_night2day",
        split="test",
        dataset_mode="unaligned",
        upstream_model="cycle_gan",
        direction="night2day",
    )
    assert test[test.index("--direction") + 1] == "BtoA"


def test_generated_images_are_found_in_the_flat_layout_upstream_writes(tmp_path: Path) -> None:
    """Upstream writes `<image>_<label>.png` into `images/`, not `images/<label>/`.

    Reading the older per-label layout finds nothing and scores nothing, which is a
    silent drop rather than an error: the run reports no metrics.
    """
    images = tmp_path / "results" / "run" / "test_latest" / "images"
    images.mkdir(parents=True)
    for name in ("000000", "000001"):
        for label in ("real_A", "real_B", "fake_A", "fake_B", "rec_A", "rec_B"):
            (images / f"{name}_{label}.png").write_bytes(b"")

    day2night = runner.DomainAdaptation(direction="day2night", name="run")
    night2day = runner.DomainAdaptation(direction="night2day", name="run")

    assert [p.name for p in day2night.generated_paths(str(tmp_path), "test")] == [
        "000000_fake_B.png",
        "000001_fake_B.png",
    ]
    assert [p.name for p in night2day.generated_paths(str(tmp_path), "test")] == [
        "000000_fake_A.png",
        "000001_fake_A.png",
    ]

    # The old shape - one folder per label - is *not* what this revision writes, and
    # finding nothing there is the reason `generated_paths` globs by filename.
    legacy = tmp_path / "results" / "run" / "test_latest" / "images" / "fake_B"
    legacy.mkdir()
    (legacy / "000000.png").write_bytes(b"")
    assert len(day2night.generated_paths(str(tmp_path), "test")) == 2


def test_missing_upstream_imports_are_reported_by_name(monkeypatch, tmp_path: Path) -> None:
    """The two start-up imports are not SyMTRS dependencies, so the check is the fix."""
    declared = tuple(upstream.REQUIRED_IMPORTS)
    assert "dominate" in declared
    assert "wandb" in declared

    monkeypatch.setattr(upstream, "REQUIRED_IMPORTS", ("symtrs_not_a_real_package",))
    assert upstream.missing_imports() == ["symtrs_not_a_real_package"]

    # Installed packages are not reported, however long the list is.
    monkeypatch.setattr(upstream, "REQUIRED_IMPORTS", declared + ("symtrs_not_a_real_package",))
    assert upstream.missing_imports() == ["symtrs_not_a_real_package"]

    # A broken install reports as missing rather than raising out of the check: the point
    # is to name the package, whatever state it is in.
    real_find_spec = importlib.util.find_spec

    def explode(name: str):
        if name == "wandb":
            raise ImportError("broken install")
        return real_find_spec(name)

    monkeypatch.setattr(importlib.util, "find_spec", explode)
    monkeypatch.setattr(upstream, "REQUIRED_IMPORTS", ("dominate", "wandb"))
    assert upstream.missing_imports() == ["wandb"]


def test_a_run_without_the_upstream_imports_stops_with_the_install_command(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(upstream, "REQUIRED_IMPORTS", ("symtrs_not_a_real_package",))
    with pytest.raises(upstream.UpstreamError) as excinfo:
        runner._launch(["python", "train.py"], cwd=tmp_path)
    message = str(excinfo.value)
    assert "symtrs_not_a_real_package" in message
    assert "pip install -e '.[adapt]'" in message


def test_multi_process_training_refuses_a_cpu_device(tmp_path: Path) -> None:
    """Upstream's multi-process path is NCCL-only, so `gpus > 1` on cpu is a config error.

    Raised here rather than by waiting for a distributed init to time out.
    """
    from symtrs.config import RunConfig

    cfg = RunConfig(task="domain_adapt", model="cyclegan", resolution=2048).with_overrides(
        {
            "data.root": str(tmp_path),
            "out_dir": str(tmp_path / "run"),
            "train.gpus": 2,
            "train.device": "cpu",
        }
    )
    with pytest.raises(ValueError) as excinfo:
        runner.run_domain_adaptation(cfg, mode="finetune")
    assert "NCCL" in str(excinfo.value)


def test_doctor_and_pyproject_agree_on_where_the_upstream_imports_come_from() -> None:
    """The error message names an extra, so that extra has to be the one that installs them."""
    from symtrs.commands.doctor import OPTIONAL

    for name in upstream.REQUIRED_IMPORTS:
        assert name in OPTIONAL, f"{name} is not reported by `symtrs doctor`"
        assert OPTIONAL[name][1] == "adapt"

    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    extra = re.search(r"^adapt = \[(.*?)\]", text, re.MULTILINE | re.DOTALL)
    assert extra, "pyproject.toml no longer declares the `adapt` extra"
    for name in upstream.REQUIRED_IMPORTS:
        assert f'"{name}"' in extra.group(1)
