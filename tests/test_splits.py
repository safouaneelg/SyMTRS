"""The resolution axis: 2048/1024/512/256 map to the right directories, and nothing else.

These are the tests that catch a mis-generated dataset before you spend GPU hours on it,
so they need no torch and no images - only the directory names.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from symtrs.data.splits import (
    RESOLUTION_TO_SCALE,
    RESOLUTION_TO_SUFFIX,
    RESOLUTIONS,
    SPLITS,
    MissingDirectoryError,
    available_resolutions,
    describe_layout,
    resolve_split_dirs,
    scale_for_resolution,
    subdir_for,
)


@pytest.mark.parametrize(
    "resolution,expected",
    [
        (2048, ("rgb_day", "depth", "semantic")),
        (1024, ("rgb_day_lr/x2", "depth_lr/x2", "semantic_lr/x2")),
        (512, ("rgb_day_lr/x4", "depth_lr/x4", "semantic_lr/x4")),
        (256, ("rgb_day_lr/x8", "depth_lr/x8", "semantic_lr/x8")),
    ],
)
def test_subdirs_follow_the_layout(resolution, expected):
    assert (subdir_for("day", resolution), subdir_for("depth", resolution),
            subdir_for("semantic", resolution)) == expected


def test_night_is_native_only():
    """rgb_night has no low-resolution copy, which is what makes domain adaptation 2048-only."""
    for resolution in RESOLUTIONS:
        assert subdir_for("night", resolution) == "rgb_night"


def test_resolution_scale_is_the_inverse_of_the_pixel_size():
    for resolution, scale in RESOLUTION_TO_SCALE.items():
        assert resolution * scale == max(RESOLUTIONS)
        assert RESOLUTION_TO_SUFFIX[resolution] == (None if scale == 1 else f"x{scale}")


def test_scale_for_unknown_resolution_says_what_is_supported():
    with pytest.raises(ValueError, match="expected one of"):
        scale_for_resolution(4096)


def test_subdir_for_unknown_role():
    with pytest.raises(ValueError, match="unknown role"):
        subdir_for("lidar", 2048)


@pytest.mark.parametrize("resolution", RESOLUTIONS)
def test_resolve_split_dirs_points_at_real_directories(layout: Path, resolution: int):
    for split in SPLITS:
        directories = resolve_split_dirs(str(layout), split, resolution)
        assert directories.day.is_dir()
        assert directories.depth.is_dir()
        assert directories.semantic.is_dir()
        assert directories.day == layout / split / subdir_for("day", resolution)
        # The `path_for` accessor is what every caller actually uses, and it is the one
        # that turns a wrong layout into a named error instead of an empty file list.
        assert directories.path_for("day") == directories.day


def test_night_is_none_below_native(layout: Path):
    assert resolve_split_dirs(str(layout), "train", 1024).night is None
    assert resolve_split_dirs(str(layout), "train", 2048).night is not None


def test_missing_directory_error_names_the_path_and_the_fix(tmp_path: Path):
    root = tmp_path / "empty"
    (root / "train" / "rgb_day").mkdir(parents=True)
    directories = resolve_split_dirs(str(root), "train", 1024)
    with pytest.raises(MissingDirectoryError) as info:
        directories.path_for("depth")
    message = str(info.value)
    assert "depth_lr/x2" in message
    # The hint has to name a command that runs. `symtrs inspect --root` was here, and
    # `inspect` takes a positional checkpoint, so the flag was rejected - on the message a
    # wrong `--data-root` produces, i.e. the first one a new user meets.
    assert "symtrs doctor --data-root" in message
    assert info.value.resolution == 1024


def test_missing_root_is_reported_not_created(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="dataset root does not exist"):
        resolve_split_dirs(str(tmp_path / "nowhere"), "train", 2048)


def test_unknown_split_name():
    with pytest.raises(ValueError, match="unknown split"):
        # Even with a root that exists, a bad split name must not silently resolve.
        resolve_split_dirs("/tmp", "training", 2048)


def test_available_resolutions_reports_what_is_there(layout: Path, partial_layout: Path):
    assert available_resolutions(str(layout), "train") == (  # largest first
        2048, 1024, 512, 256,
    )
    # Only the native folders and rgb_day_lr/x2 exist in the partial dataset.
    assert available_resolutions(str(partial_layout), "train") == (2048, 1024)


def test_describe_layout_marks_each_role(layout: Path, partial_layout: Path):
    text = describe_layout(str(partial_layout))
    assert "dataset root" in text
    assert "2048" in text and "1024" in text
    # A role that is absent at a resolution is marked, not omitted: a quietly short table
    # is how a missing semantic_lr tree goes unnoticed.
    assert "semantic=missing" in text
    assert "night=-" in text  # None below native resolution


def test_describe_layout_on_a_root_with_nothing(tmp_path: Path):
    text = describe_layout(str(tmp_path / "absent"))
    assert "MISSING" in text
