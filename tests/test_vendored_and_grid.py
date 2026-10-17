"""The vendored upstream tree, and the config grid it has to sit beside.

Two things are checked here that no other test covers, both of them the kind of thing that
breaks quietly:

  * **The vendored CycleGAN/pix2pix tree is complete.** It is committed rather than cloned
    at run time, so "upstream is present" is now a property of this repository that can
    regress - a `.gitignore` that grows a `models/` rule, a `git clean` in the wrong place,
    a hand-edit that deletes a file. `symtrs setup-adapt --check` answers the same question
    for a user; this asks it in CI.

  * **The config grid matches the tables it is generated from.** 88 files, hand-edited one
    at a time, is how a benchmark ends up comparing two protocols without noticing.

Both run without torch: they read files and import path constants.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

from symtrs.model.domain import upstream  # noqa: E402
from symtrs.model.superres import scale_for  # noqa: E402
from symtrs.registry import get_spec  # noqa: E402


def _grid_module():
    """Load `configs/generate_grid.py`, which is a script rather than a package module."""
    path = REPO_ROOT / "configs" / "generate_grid.py"
    spec = importlib.util.spec_from_file_location("symtrs_config_grid", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------- the vendored tree


def test_the_vendored_tree_is_present_and_complete() -> None:
    root = upstream.default_upstream()
    assert root.is_dir(), f"{root} is missing; the upstream tree is meant to be committed"
    assert upstream.missing_files(root) == []
    assert upstream.is_complete(root)


def test_the_tree_was_not_half_vendored() -> None:
    """A tree with entry points but no model subpackage is the failure mode to catch."""
    root = upstream.default_upstream()
    for relative in ("models/networks.py", "util/util.py", "options/base_options.py"):
        assert (root / relative).is_file(), relative


def test_the_vendored_tree_is_not_a_git_checkout() -> None:
    """It is a copy, not a submodule or a nested repository."""
    root = upstream.default_upstream()
    assert not (root / ".git").exists()


def test_the_gallery_is_the_only_thing_left_out() -> None:
    root = upstream.default_upstream()
    for excluded in upstream.VENDORED_EXCLUDES:
        assert not (root / excluded).exists(), excluded
    # and the licence travelled with the code, which is the reason imgs/ could be dropped
    assert (root / "LICENSE").is_file()


def test_every_deviation_is_documented_with_a_reason() -> None:
    for name, why in upstream.DEVIATIONS.items():
        assert why.strip(), name
    assert set(upstream.MODIFIED_FILES) == {
        "data/unaligned_dataset.py",
        "data/aligned_dataset.py",
    }


def test_modified_loaders_have_their_upstream_originals_on_disk() -> None:
    """The delta has to be auditable offline, without fetching the revision again."""
    root = upstream.default_upstream()
    for name in upstream.MODIFIED_FILES:
        original = root / "upstream_originals" / name
        assert original.is_file(), f"no upstream original kept for {name}"
        assert original.read_text(encoding="utf-8") != (root / name).read_text(encoding="utf-8")


def test_the_revision_is_pinned_to_a_sha() -> None:
    assert len(upstream.UPSTREAM_REVISION) == 40
    assert all(character in "0123456789abcdef" for character in upstream.UPSTREAM_REVISION)


def test_verify_reports_the_tree_rather_than_writing_to_it() -> None:
    """`setup` used to clone; it must not any more, and `verify` must not either."""
    before = sorted(p.name for p in upstream.default_upstream().iterdir())
    assert upstream.verify() == upstream.default_upstream()
    after = sorted(p.name for p in upstream.default_upstream().iterdir())
    assert before == after


# --------------------------------------------------------------------- the config grid


def _configs() -> list[Path]:
    return sorted(
        path
        for path in (REPO_ROOT / "configs").rglob("*.yaml")
        if not path.name.endswith(".upstream.yaml")
    )


def test_the_grid_is_not_empty() -> None:
    assert len(_configs()) >= 88


@pytest.mark.parametrize("path", _configs(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_every_config_loads_and_validates(path: Path) -> None:
    from symtrs.config import RunConfig

    cfg = RunConfig.from_yaml(str(path))
    assert cfg.model
    assert cfg.resolution in (2048, 1024, 512, 256)


@pytest.mark.parametrize("path", _configs(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_every_config_names_a_registered_model_of_its_own_task(path: Path) -> None:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    spec = get_spec(raw["model"])
    assert spec.task == raw["task"], f"{path.name}: {spec.key} is a {spec.task} model"


def test_resolution_folders_match_their_contents() -> None:
    """`configs/depth/1024/...yaml` must say resolution: 1024, or the folder lies."""
    for path in _configs():
        parent = path.parent.name
        if not parent.isdigit():
            continue
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert raw["resolution"] == int(parent), path
        assert path.name.endswith(f"_{parent}.yaml"), path


def test_super_resolution_filenames_match_their_derived_scale() -> None:
    folder = REPO_ROOT / "configs" / "superresolution"
    for path in sorted(folder.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert raw["task"] == "superresolution"
        assert path.stem.endswith(f"_x{scale_for(raw['resolution'])}"), path.name


def test_domain_adaptation_has_both_directions_for_every_model() -> None:
    """A one-directional claim about a translation says nothing about the other way."""
    folder = REPO_ROOT / "configs" / "domain_adapt"
    directions = set()
    for path in sorted(folder.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert raw["resolution"] == 2048, "rgb_night has no _lr folders"
        direction = raw["model_kwargs"]["direction"]
        assert path.stem.endswith(direction), path.name
        directions.add(direction)
    assert directions == {"day2night", "night2day", "shadow2noshadow", "noshadow2shadow"}
    stems = {path.stem for path in folder.glob("*.yaml")}
    assert stems == {
        f"{model}_{direction}"
        for model in ("cyclegan", "pix2pix")
        for direction in directions
    }


def test_the_generator_and_the_runtime_agree_on_which_folders_a_direction_stages() -> None:
    """`generate_grid.py` cannot import the package, so it keeps a copy of the pair table."""
    grid = _grid_module()
    assert set(grid.DOMAIN_DIRECTIONS) == set(upstream.DIRECTION_ROLES)
    for direction, (a_role, b_role, _) in upstream.DIRECTION_ROLES.items():
        assert grid.DOMAIN_FOLDERS[direction] == (a_role, b_role), direction


def test_depth_protocol_is_consistent_down_each_row() -> None:
    """Loss and alignment follow (protocol, output space) at EVERY resolution - rule 3.

    The expected pair comes from `symtrs.registry.depth_pairing`, not from a table in this
    test, so the generator (which keeps its own copy because it cannot import the package),
    the registry and the committed YAMLs are all held to one statement. MiDaS emits
    disparity `s / z + t`, so it needs a disparity-space scale-and-shift fit; ZoeD-N and DA3
    emit depth, so a scale is enough; metric models take none.
    """
    from symtrs.registry import depth_pairing

    folder = REPO_ROOT / "configs" / "depth"
    seen: dict[str, set[tuple[str, str]]] = {}
    for path in sorted(folder.rglob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        key = (raw["train"]["loss"], raw["eval"]["align"])
        seen.setdefault(raw["model"], set()).add(key)

    assert seen, "no depth configs found"
    for model, protocols in seen.items():
        assert len(protocols) == 1, f"{model} switches protocol between resolutions: {protocols}"
        spec = get_spec(model)
        expected = depth_pairing(spec.protocol, spec.output_space)
        assert next(iter(protocols)) == expected, (
            f"{model} ({spec.protocol}, {spec.output_space}) should be (loss, align) = {expected}, "
            f"configs say {next(iter(protocols))}"
        )


def test_midas_is_aligned_in_disparity_space_at_every_resolution() -> None:
    """The reason, not just the value: a scale alignment on `1 / (s / z + t)` cannot remove
    the shift `t`, which dominates on narrow-depth nadir views. All five MiDaS rows at all four
    resolutions must say so, and nothing else may use the disparity fit."""
    folder = REPO_ROOT / "configs" / "depth"
    for path in sorted(folder.rglob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        is_midas = raw["model"].startswith("midas_")
        assert (raw["eval"]["align"] == "scale_shift_disparity") == is_midas, path.name
        assert (raw["train"]["loss"] == "ssi") == is_midas, path.name


def test_da3_metric_configs_state_the_camera_and_zoed_k_states_its_depth_range() -> None:
    """DA3-metric needs the camera's field of view to give metres; ZoeD-K needs a ceiling that
    covers the data (its bins stop at max_depth). Both are written into the generated files."""
    for resolution in (2048, 1024, 512, 256):
        base = REPO_ROOT / "configs" / "depth" / str(resolution)
        da3 = yaml.safe_load((base / f"da3_metric_large_{resolution}.yaml").read_text(encoding="utf-8"))
        assert da3["data"]["camera_hfov_deg"] == 60.0, resolution
        zoe = yaml.safe_load((base / f"zoedepth_k_{resolution}.yaml").read_text(encoding="utf-8"))
        assert zoe["model_kwargs"]["max_depth"] == 300.0, resolution
        assert "max_depth" not in yaml.safe_load(
            (base / f"zoedepth_nk_{resolution}.yaml").read_text(encoding="utf-8")
        ).get("model_kwargs", {}), "NK has an unbounded head; max_depth would be refused"


def test_segmentation_always_states_the_class_count_and_ignore_index() -> None:
    for path in sorted((REPO_ROOT / "configs" / "segmentation").rglob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert raw["data"]["num_classes"] == 5, path
        assert raw["data"]["label_step"] == 63, path
        assert raw["data"]["ignore_index"] == 255, path


def test_the_committed_tree_matches_the_generator() -> None:
    """The 88 files are output; the tables are the source. They must not diverge.

    Fails when someone edits a generated YAML by hand, or changes a table without
    re-running `python3 configs/generate_grid.py`.
    """
    grid = _grid_module()
    for path, text in grid.build_tree().items():
        assert path.is_file(), f"{path} was never written"
        assert path.read_text(encoding="utf-8") == text, (
            f"{path} differs from configs/generate_grid.py; re-run it"
        )


def test_da3_crops_to_512_wherever_the_image_is_larger_than_that() -> None:
    """DA3-Large is fully fine-tuned (411M parameters, ~6.5GB of weights, grads and Adam
    state), and a 1024px frame is 4x the tokens of a 512px one: with patch_size 0 at 1024 it
    ran out of memory on a 12GB card even at batch 1, while the 512px crop used at 2048 fit.
    A crop never exceeds the image (nothing is upsampled), so 512 and 256 use the whole frame."""
    expected = {"2048": 512, "1024": 512, "512": 0, "256": 0}
    for model in ("da3_large", "da3_metric_large"):
        for resolution, patch in expected.items():
            path = REPO_ROOT / "configs" / "depth" / resolution / f"{model}_{resolution}.yaml"
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            assert raw["train"]["patch_size"] == patch, f"{model} at {resolution}"
            assert raw["train"]["patch_size"] <= int(resolution)
