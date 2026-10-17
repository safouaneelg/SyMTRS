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
    # BtoA swaps the inputs, not the output label: the translation is still fake_B.
    assert [p.name for p in night2day.generated_paths(str(tmp_path), "test")] == [
        "000000_fake_B.png",
        "000001_fake_B.png",
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


# ------------------------------------------------- the shadow <-> shadow-free pair

SHADOW_DIRECTIONS = ("shadow2noshadow", "noshadow2shadow")


def _shadow_root(tmp_path: Path, count: int = 2, size: int = 32) -> Path:
    """A real-layout root: `rgb_day` with a dark square, `rgb_noshadow` without it."""
    import numpy as np
    from PIL import Image

    root = tmp_path / "SyMTRS"
    for role in ("rgb_day", "rgb_noshadow", "rgb_night"):
        (root / "test" / role).mkdir(parents=True)
    for index in range(count):
        clean = np.full((size, size, 4), 200, np.uint8)
        dark = clean.copy()
        dark[size // 4 : 3 * size // 4, size // 4 : 3 * size // 4, :3] = 70
        key = f"loc{index:03d}"
        Image.fromarray(dark).save(root / "test" / "rgb_day" / f"rgb_day_2048_{key}.png")
        Image.fromarray(clean).save(root / "test" / "rgb_noshadow" / f"rgb_noshadow_2048_{key}.png")
        night = np.full((size, size, 4), 100, np.uint8)
        Image.fromarray(night).save(root / "test" / "rgb_night" / f"rgb_night_2048_{key}.png")
    return root


def test_shadow_directions_stage_day_as_A_and_noshadow_as_B_paired_by_key(tmp_path: Path) -> None:
    """pix2pix pairs by exact basename, so both folders must be staged under the key.

    Both shadow directions stage the same way - A is `rgb_day` - and differ only in which
    way upstream translates, exactly like the day/night pair.
    """
    pytest.importorskip("PIL")
    from PIL import Image

    root = _shadow_root(tmp_path)
    for direction in SHADOW_DIRECTIONS:
        staging = upstream.build_staging_dir(
            str(root), str(tmp_path / direction), "test", direction=direction
        )
        a = sorted(p.name for p in (staging / "testA").iterdir())
        b = sorted(p.name for p in (staging / "testB").iterdir())
        assert a == b == ["loc000.png", "loc001.png"], direction
        # A really is the shadowed image: its dark square is what rgb_noshadow lacks.
        assert Image.open(staging / "testA" / "loc000.png").convert("RGB").getpixel((16, 16)) == (70, 70, 70)
        assert Image.open(staging / "testB" / "loc000.png").convert("RGB").getpixel((16, 16)) == (200, 200, 200)


def test_the_day_night_staging_is_unchanged_by_the_shadow_pair(tmp_path: Path) -> None:
    """The default direction still stages rgb_night as B."""
    pytest.importorskip("PIL")
    root = _shadow_root(tmp_path)
    staging = upstream.build_staging_dir(str(root), str(tmp_path / "run"), "test")
    from PIL import Image

    assert Image.open(staging / "testB" / "loc000.png").convert("RGB").getpixel((2, 2)) == (100, 100, 100)
    assert Image.open(staging / "testA" / "loc000.png").convert("RGB").getpixel((2, 2)) == (200, 200, 200)


def test_directions_map_to_the_upstream_side_and_the_real_target() -> None:
    expected = {
        "day2night": ("AtoB", "night"),
        "night2day": ("BtoA", "day"),
        "shadow2noshadow": ("AtoB", "noshadow"),
        "noshadow2shadow": ("BtoA", "day"),
    }
    assert set(runner.DIRECTIONS) == set(expected)
    for direction, (side, target) in expected.items():
        assert upstream.upstream_direction(direction) == side
        assert upstream.target_role(direction) == target
    with pytest.raises(ValueError, match="unknown direction"):
        runner.DomainAdaptation(direction="day2dusk")


def test_shadow_directions_pass_only_flags_the_vendored_parser_accepts(tmp_path: Path) -> None:
    """The pair is data; the argv is the same as for day/night, side flag aside."""
    accepted = accepted_flags()
    for direction in SHADOW_DIRECTIONS:
        commands = [
            train_command(tmp_path, direction=direction),
            upstream.test_command(
                upstream=VENDORED,
                staging=staging(tmp_path),
                experiment=f"symtrs_cyclegan_{direction}",
                split="test",
                dataset_mode="unaligned",
                upstream_model="cycle_gan",
                direction=direction,
                checkpoints_dir=str(tmp_path / "checkpoints"),
                results_dir=str(tmp_path / "results"),
            ),
        ]
        for command in commands:
            for token in command:
                if token.startswith("--"):
                    assert token[2:] in accepted, f"{direction}: upstream would reject {token!r}"
    forward = train_command(tmp_path, direction="shadow2noshadow")
    backward = train_command(tmp_path, direction="noshadow2shadow")
    assert forward[forward.index("--direction") + 1] == "AtoB"
    assert backward[backward.index("--direction") + 1] == "BtoA"


def test_generated_images_of_each_shadow_direction_are_fake_B(tmp_path: Path) -> None:
    images = tmp_path / "results" / "run" / "test_latest" / "images"
    images.mkdir(parents=True)
    for label in ("real_A", "real_B", "fake_A", "fake_B"):
        (images / f"loc000_{label}.png").write_bytes(b"")
    forward = runner.DomainAdaptation(direction="shadow2noshadow", name="run")
    backward = runner.DomainAdaptation(direction="noshadow2shadow", name="run")
    assert [p.name for p in forward.generated_paths(str(tmp_path), "test")] == ["loc000_fake_B.png"]
    assert [p.name for p in backward.generated_paths(str(tmp_path), "test")] == ["loc000_fake_B.png"]


def _upstream_source(name: str) -> str:
    return (VENDORED / "models" / name).read_text(encoding="utf-8")


@pytest.mark.parametrize("model_file", ["pix2pix_model.py", "cycle_gan_model.py"])
def test_the_requested_translation_is_fake_B_whatever_the_direction(model_file: str) -> None:
    """The premise behind `runner.GENERATED_LABEL`, read from the vendored source.

    `--direction BtoA` makes upstream swap the *inputs* (`real_A` is the folder-B image) and
    then generate `fake_B = G(real_A)`. If a bump of the vendored tree changes either line,
    selecting `fake_B` silently scores the wrong side, so this fails instead. The patterns
    tolerate whitespace, not meaning.
    """
    source = _upstream_source(model_file)
    assert runner.GENERATED_LABEL == "fake_B"
    assert re.search(r'real_A\s*=\s*input\[\s*"A"\s+if\s+AtoB\s+else\s+"B"\s*\]', source), (
        f"{model_file}: set_input no longer takes real_A from the folder the direction names"
    )
    assert re.search(r'real_B\s*=\s*input\[\s*"B"\s+if\s+AtoB\s+else\s+"A"\s*\]', source)
    assert re.search(r"self\.fake_B\s*=\s*self\.netG(_A)?\(\s*self\.real_A\s*\)", source), (
        f"{model_file}: fake_B is no longer the generator applied to real_A"
    )
    # The image name follows the same swap, which is why the staged B files (named by the
    # same normalised key as the A ones) still pair with their day counterpart.
    assert re.search(r'image_paths\s*=\s*input\[\s*"A_paths"\s+if\s+AtoB\s+else\s+"B_paths"\s*\]', source)


def test_cyclegan_fake_A_is_the_opposite_translation() -> None:
    """Why `fake_A` must not be read for BtoA: it is the other generator on the other input."""
    source = _upstream_source("cycle_gan_model.py")
    assert re.search(r"self\.fake_A\s*=\s*self\.netG_B\(\s*self\.real_B\s*\)", source)


@pytest.mark.parametrize("model", ["cyclegan", "pix2pix"])
@pytest.mark.parametrize("direction", runner.DIRECTIONS)
def test_every_direction_of_both_models_is_scored_on_fake_B(
    model: str, direction: str, tmp_path: Path
) -> None:
    """All four directions x both models, with `fake_A` present to be picked by mistake."""
    task = runner.DomainAdaptation(direction=direction, name="run")
    task.key = model
    images = task.generated_dir(str(tmp_path), "test")
    images.mkdir(parents=True)
    for key in ("loc000", "loc001"):
        for label in ("real_A", "real_B", "fake_A", "fake_B", "rec_A", "rec_B"):
            (images / f"{key}_{label}.png").write_bytes(b"")
    names = [p.name for p in task.generated_paths(str(tmp_path), "test")]
    assert names == ["loc000_fake_B.png", "loc001_fake_B.png"]


def test_a_pix2pix_style_output_without_fake_A_is_still_found_for_BtoA(tmp_path: Path) -> None:
    """The case that scored nothing: pix2pix writes no fake_A at all (smoke run, real data)."""
    task = runner.DomainAdaptation(direction="night2day", name="run")
    images = task.generated_dir(str(tmp_path), "test")
    images.mkdir(parents=True)
    for label in ("real_A", "real_B", "fake_B"):
        (images / f"loc000_{label}.png").write_bytes(b"")
    assert [p.name for p in task.generated_paths(str(tmp_path), "test")] == ["loc000_fake_B.png"]


@pytest.mark.parametrize("direction", runner.DIRECTIONS)
def test_nothing_generated_is_an_error_naming_what_was_looked_for(
    direction: str, tmp_path: Path
) -> None:
    """A run that generated nothing must not finish with `test: {}`."""
    task = runner.DomainAdaptation(direction=direction, name="run")
    with pytest.raises(RuntimeError, match=r"no `\*_fake_B\.png` generated images") as excinfo:
        task.generated_paths(str(tmp_path), "test")  # the folder does not exist
    message = str(excinfo.value)
    assert direction in message and "results" in message and "test.py" in message

    images = task.generated_dir(str(tmp_path), "test")
    images.mkdir(parents=True)
    (images / "loc000_fake_A.png").write_bytes(b"")  # only the other side: still nothing
    with pytest.raises(RuntimeError, match="the folder has no such file"):
        task.generated_paths(str(tmp_path), "test")


def test_scoring_without_real_target_images_is_an_error(tmp_path: Path) -> None:
    """The second silent `{}`: no real images of the target domain to compare against."""
    pytest.importorskip("PIL")
    from symtrs.config import RunConfig

    root = _shadow_root(tmp_path, count=1, size=16)
    run = tmp_path / "run"
    cfg = RunConfig(task="domain_adapt", model="cyclegan", resolution=2048).with_overrides(
        {
            "data.root": str(root), "out_dir": str(run),
            "shadow.mask_cache": str(tmp_path / "mask_cache"),
        }
    )
    task = runner.DomainAdaptation(direction="night2day", name="run")
    task.run_test = lambda **kwargs: None  # upstream is not launched here
    images = task.generated_dir(str(run), "test")
    images.mkdir(parents=True)
    (images / "loc000_fake_B.png").write_bytes(b"")
    day = root / "test" / "rgb_day"  # night2day's target
    for image in day.iterdir():
        image.unlink()
    with pytest.raises(RuntimeError, match="no real day images to score night2day"):
        runner.evaluate_adaptation(cfg, task, run)


@pytest.mark.parametrize("direction", SHADOW_DIRECTIONS)
def test_shadow_directions_are_refused_below_native_resolution(direction: str, tmp_path: Path) -> None:
    """`rgb_noshadow` has no `_lr` twin, so 1024 would mean approximating the target."""
    from symtrs.config import RunConfig

    cfg = RunConfig(task="domain_adapt", model="cyclegan", resolution=1024).with_overrides(
        {"model_kwargs.direction": direction, "data.root": str(tmp_path), "out_dir": str(tmp_path / "run")}
    )
    with pytest.raises(ValueError, match="2048-only") as excinfo:
        runner.run_domain_adaptation(cfg, mode="finetune")
    assert "rgb_noshadow" in str(excinfo.value)


def test_only_the_shadow_removal_direction_gets_the_paired_metrics() -> None:
    """`noshadow2shadow` has a ground truth too, but a shadow-removal column describing it
    would measure how well a shadow was painted under a name that says it was removed."""
    assert runner.SHADOW_REMOVAL_DIRECTIONS == ("shadow2noshadow",)


def test_the_paired_scoring_runs_the_shadow_removal_metrics_on_a_resampled_output(
    tmp_path: Path, capsys
) -> None:
    """Upstream writes small images; the metrics are those of `symtrs.metrics.shadow` at
    the ground-truth size, so a shadow-free output scores perfectly and an output that kept
    the shadow scores badly in the shadow region only."""
    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    from symtrs.config import RunConfig

    root = _shadow_root(tmp_path, count=2, size=64)
    run = tmp_path / "run"
    cfg = RunConfig(task="domain_adapt", model="pix2pix", resolution=2048).with_overrides(
        {
            "data.root": str(root), "out_dir": str(run), "eval.num_visualizations": 0,
            "shadow.mask_cache": str(tmp_path / "mask_cache"),  # never the repo's own cache
        }
    )
    task = runner.DomainAdaptation(direction="shadow2noshadow", name="run")
    images = task.generated_dir(str(run), "test")
    images.mkdir(parents=True)

    def write(shaded: bool) -> None:
        for key in ("loc000", "loc001"):
            array = np.full((16, 16, 3), 200, np.uint8)  # upstream's size is not the GT's
            if shaded:
                array[4:12, 4:12] = 70
            Image.fromarray(array).save(images / f"{key}_fake_B.png")

    write(shaded=False)
    perfect = runner.score_shadow_removal(cfg, task, run, task.generated_paths(str(run), "test"))
    assert perfect["psnr"] > 40 and perfect["ssim_y"] > 0.99
    assert perfect["images"] == 2 and perfect["images_with_shadow"] == 2
    assert "resampled" in capsys.readouterr().out
    # The scorer's own input contract: a full-size PNG per key.
    assert Image.open(run / "predictions" / "test" / "loc000.png").size == (64, 64)

    write(shaded=True)  # the shadow was not removed
    unchanged = runner.score_shadow_removal(cfg, task, run, task.generated_paths(str(run), "test"))
    assert unchanged["psnr_shadow"] < perfect["psnr_shadow"] - 10
    assert unchanged["psnr_nonshadow"] > unchanged["psnr_shadow"]


def test_the_paired_scoring_names_missing_generated_images(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    from symtrs.config import RunConfig

    root = _shadow_root(tmp_path, count=2, size=64)
    run = tmp_path / "run"
    cfg = RunConfig(task="domain_adapt", model="pix2pix", resolution=2048).with_overrides(
        {
            "data.root": str(root), "out_dir": str(run),
            "shadow.mask_cache": str(tmp_path / "mask_cache"),
        }
    )
    task = runner.DomainAdaptation(direction="shadow2noshadow", name="run")
    task.generated_dir(str(run), "test").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="no generated counterpart"):
        runner.score_shadow_removal(cfg, task, run, [])


def test_tables_tell_the_four_directions_apart_and_line_up_with_shadow_removal(tmp_path: Path) -> None:
    """Two runs of one model differ only by direction; without the column they are one row."""
    import json

    from symtrs.commands import tables

    for direction, extra in (("day2night", {}), ("shadow2noshadow", {"psnr": 20.0})):
        run = tmp_path / direction
        run.mkdir()
        (run / "summary.json").write_text(json.dumps({
            "task": "domain_adapt", "model": "pix2pix", "direction": direction,
            "resolution": 2048, "mode": "finetune", "test": {"fid": 100.0, **extra},
        }), encoding="utf-8")
    rows = tables.collect(str(tmp_path))
    columns = tables._columns(rows, "domain_adapt")
    assert {r["direction"] for r in rows} == {"day2night", "shadow2noshadow"}
    assert "direction" in columns and columns.index("fid") < columns.index("psnr")


def test_pix2pix_staging_writes_the_AB_pairs_the_vendored_aligned_dataset_reads(tmp_path: Path) -> None:
    """`AlignedDataset` reads `<dataroot>/<phase>/*.png` as `[A | B]` and ignores `<phase>A`.

    With only the links it falls through to its raw-directory auto-split, looking for
    `hr/` and `night/` under the staging directory - the failure every pix2pix run died of
    before the pairs were written. The A half is the shadowed image, the B half the clean one.
    """
    pytest.importorskip("PIL")
    from PIL import Image

    root = _shadow_root(tmp_path, count=2, size=32)
    staging = upstream.build_staging_dir(
        str(root), str(tmp_path / "run"), "test", direction="shadow2noshadow", aligned=True
    )
    pairs = sorted(p.name for p in (staging / "test").iterdir())
    assert pairs == ["loc000.png", "loc001.png"]
    pair = Image.open(staging / "test" / "loc000.png").convert("RGB")
    assert pair.size == (64, 32)
    assert pair.getpixel((16, 16)) == (70, 70, 70)  # left half: rgb_day, the shadowed square
    assert pair.getpixel((32 + 16, 16)) == (200, 200, 200)  # right half: rgb_noshadow
    # The unpaired front-end is untouched: CycleGAN reads trainA/trainB and nothing else.
    unaligned = upstream.build_staging_dir(str(root), str(tmp_path / "other"), "test")
    assert not (unaligned / "test").exists()

    # A pair whose scene left the split must not survive in the folder.
    (root / "test" / "rgb_day" / "rgb_day_2048_loc001.png").unlink()
    (root / "test" / "rgb_noshadow" / "rgb_noshadow_2048_loc001.png").unlink()
    again = upstream.build_staging_dir(
        str(root), str(tmp_path / "run"), "test", direction="shadow2noshadow", aligned=True
    )
    assert sorted(p.name for p in (again / "test").iterdir()) == ["loc000.png"]


def test_pix2pix_selects_the_aligned_dataset_mode_and_cyclegan_the_unaligned_one() -> None:
    cyclegan = runner.DomainAdaptation(direction="shadow2noshadow")
    cyclegan.key = "cyclegan"
    pix2pix = runner.DomainAdaptation(direction="shadow2noshadow")
    pix2pix.key = "pix2pix"
    assert cyclegan.dataset_mode == "unaligned" and pix2pix.dataset_mode == "aligned"


def test_a_partial_translation_of_the_split_is_an_error(tmp_path: Path) -> None:
    """KID/FID take the whole real folder: fewer translations than real images would compare
    two different scene sets, so a truncated upstream test pass must not be scored."""
    pytest.importorskip("PIL")
    from symtrs.config import RunConfig

    root = _shadow_root(tmp_path, count=2, size=16)
    run = tmp_path / "run"
    cfg = RunConfig(task="domain_adapt", model="cyclegan", resolution=2048).with_overrides(
        {
            "data.root": str(root), "out_dir": str(run),
            "shadow.mask_cache": str(tmp_path / "mask_cache"),
        }
    )
    task = runner.DomainAdaptation(direction="night2day", name="run")
    task.run_test = lambda **kwargs: None  # upstream is not launched here
    images = task.generated_dir(str(run), "test")
    images.mkdir(parents=True)
    (images / "loc000_fake_B.png").write_bytes(b"")  # one of the two test frames
    with pytest.raises(RuntimeError, match="did not translate the whole split"):
        runner.evaluate_adaptation(cfg, task, run)
