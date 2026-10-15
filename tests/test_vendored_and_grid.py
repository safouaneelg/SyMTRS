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
    assert directions == {"day2night", "night2day"}
    stems = {path.stem for path in folder.glob("*.yaml")}
    assert stems == {
        "cyclegan_day2night",
        "cyclegan_night2day",
        "pix2pix_day2night",
        "pix2pix_night2day",
    }


def test_depth_protocol_is_consistent_down_each_row() -> None:
    """A relative model is align: scale at every resolution, or the row is not comparable."""
    folder = REPO_ROOT / "configs" / "depth"
    seen: dict[str, set[tuple[str, str]]] = {}
    for path in sorted(folder.rglob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        key = (raw["train"]["loss"], raw["eval"]["align"])
        seen.setdefault(raw["model"], set()).add(key)

    assert seen, "no depth configs found"
    for model, protocols in seen.items():
        assert len(protocols) == 1, f"{model} switches protocol between resolutions: {protocols}"
        loss, align = next(iter(protocols))
        expected_align = "scale" if get_spec(model).protocol == "relative" else "none"
        assert align == expected_align, f"{model}: {loss}/{align} against {get_spec(model).protocol}"


def test_segmentation_always_states_the_class_count_and_ignore_index() -> None:
    for path in sorted((REPO_ROOT / "configs" / "segmentation").rglob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert raw["data"]["num_classes"] == 6, path
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
