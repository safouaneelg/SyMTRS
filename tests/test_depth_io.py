"""Reading depth: the validity mask, the unit scale, and matching depths to images.

The validity rule is the one that matters. Rendered depth files use 0 for "no geometry"
(sky, and anything outside the rendered frustum), and a metric that treats 0 as a distance
of zero metres is wrong in a way that looks like a modest accuracy loss rather than a bug.
So the rule is pinned here, along with the `.npy` path that lets the whole pipeline run
before the EXR tooling exists.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from symtrs.data.exr import (
    DEPTH_EXTENSIONS,
    DepthReadError,
    depth_summary,
    depth_to_tensor,
    guess_scale_from_summary,
    list_depth_files,
    read_depth_file,
)

pytestmark = pytest.mark.skipif(
    __import__("importlib").util.find_spec("numpy") is None, reason="numpy is not installed"
)


def test_depth_extensions_cover_npy_so_the_pipeline_runs_without_exr_tooling():
    assert ".npy" in DEPTH_EXTENSIONS
    assert ".exr" in DEPTH_EXTENSIONS


def test_validity_is_finite_and_strictly_positive():
    raw = np.array([[0.0, 1.5, np.nan, np.inf, -2.0, 3.0]], dtype=np.float32)
    metres, valid = depth_to_tensor(raw)
    assert valid.tolist() == [[False, True, False, False, False, True]]
    # Invalid pixels are zeroed in the metres array so that a downstream `mean` over
    # everything cannot be pulled by a NaN or a negative.
    assert metres[0, 0] == 0.0 and metres[0, 2] == 0.0 and metres[0, 4] == 0.0
    assert metres[0, 1] == pytest.approx(1.5)


def test_scale_converts_centimetres_to_metres():
    raw = np.array([[150.0, 0.0]], dtype=np.float32)
    metres, valid = depth_to_tensor(raw, scale=0.01)
    assert metres[0, 0] == pytest.approx(1.5)
    # Scaling must not resurrect an invalid pixel: 0 * 0.01 is still 0, and it stays masked.
    assert not valid[0, 1]


def test_an_all_invalid_map_summarises_without_raising():
    """A nadir tile over water is a legitimate thing to point `symtrs inspect` at."""
    summary = depth_summary(np.zeros((4, 4), dtype=np.float32))
    assert summary["invalid_fraction"] == 1.0
    assert not np.isfinite(summary["p50"])
    assert guess_scale_from_summary(summary) is None


def test_summary_reports_the_distribution_of_the_valid_pixels():
    raw = np.concatenate([np.linspace(1.0, 100.0, 1000), np.zeros(1000)]).astype(np.float32)
    summary = depth_summary(raw.reshape(40, 50))
    assert summary["invalid_fraction"] == pytest.approx(0.5, abs=1e-3)
    assert 1.0 <= summary["p1"] < summary["p50"] < summary["p99"] <= 100.0


def test_summary_flags_negative_values_separately():
    """A negative depth is a different bug from a missing one and should be visible."""
    raw = np.array([[-1.0, 5.0, 5.0, 5.0]], dtype=np.float32)
    assert depth_summary(raw)["negative_fraction"] == pytest.approx(0.25)


@pytest.mark.parametrize(
    "median,expected",
    [(120.0, 1.0), (3000.0, 1.0), (8000.0, 0.01), (40000.0, 0.01)],
)
def test_scale_guess_only_fires_on_centimetre_scale_values(median, expected):
    """A plausible city-scale median reads as metres; only a four-digit median is
    suspicious enough to suggest centimetres. `None` is reserved for "nothing to go on"
    (an all-invalid map), not for "metres"."""
    assert guess_scale_from_summary({"p50": median}) == expected


@pytest.mark.parametrize("p50", [None, float("nan"), 0.0, -5.0])
def test_scale_guess_declines_when_there_is_nothing_to_go_on(p50):
    assert guess_scale_from_summary({"p50": p50}) is None


# ------------------------------------------------------------------- .npy round trip


def test_npy_depth_round_trips_at_full_precision(tmp_path: Path):
    """`.npy` is the format the pipeline can be debugged on before the EXR writer exists,
    so it must not be a lossy path: float32 in, float32 out, exact."""
    rng = np.random.default_rng(0)
    depth = rng.uniform(1.0, 500.0, size=(32, 32)).astype(np.float32)
    depth[0, 0] = 0.0
    depth[0, 1] = np.nan
    path = tmp_path / "RS.100.png.depth.npy"
    np.save(path, depth)

    loaded = read_depth_file(path)
    assert loaded.dtype == np.float32
    assert loaded.shape == (32, 32)
    # NaN survives, because "no geometry" has to be distinguishable from "zero metres".
    assert np.isnan(loaded[0, 1])
    # Bit-exact, not "close": nothing on this path is allowed to quantise a depth map.
    assert np.array_equal(loaded[1:, :], depth[1:, :])


def test_npy_read_squeezes_a_singleton_channel_axis(tmp_path: Path):
    """Some writers emit [H, W, 1]; the rest of the pipeline wants [H, W]."""
    path = tmp_path / "scene.npy"
    np.save(path, np.full((8, 8, 1), 4.0, dtype=np.float32))
    assert read_depth_file(path).shape == (8, 8)


def test_npy_read_keeps_the_spatial_axes_the_right_way_round(tmp_path: Path):
    """`[H, W, 1]` squeezed from the wrong end gives `[W, 1]`, which is a transposed and
    truncated depth map - it would surface as a mysteriously bad model, not as an error."""
    depth = np.arange(8 * 5, dtype=np.float32).reshape(8, 5)
    path = tmp_path / "scene.npy"
    np.save(path, depth[..., None])
    loaded = read_depth_file(path)
    assert loaded.shape == (8, 5)
    assert np.array_equal(loaded, depth)


def test_npy_read_accepts_a_saved_batch_of_one(tmp_path: Path):
    depth = np.arange(8 * 5, dtype=np.float32).reshape(8, 5)
    path = tmp_path / "scene.npy"
    np.save(path, depth[None, ...])
    assert np.array_equal(read_depth_file(path), depth)


def test_npy_read_refuses_an_array_it_cannot_interpret_as_one_depth_map(tmp_path: Path):
    """Silently taking channel 0 of an RGB-ish array would train on red-channel values."""
    path = tmp_path / "scene.npy"
    np.save(path, np.zeros((8, 8, 3), dtype=np.float32))
    with pytest.raises(DepthReadError, match="expected"):
        read_depth_file(path)


def test_a_missing_depth_file_is_reported_with_its_path(tmp_path: Path):
    with pytest.raises(DepthReadError, match="depth file not found"):
        read_depth_file(tmp_path / "absent.npy")


def test_an_unsupported_extension_names_the_supported_ones(tmp_path: Path):
    path = tmp_path / "depth.tif"
    path.write_bytes(b"not a depth map")
    with pytest.raises(DepthReadError) as info:
        read_depth_file(path)
    assert ".exr" in str(info.value) and ".npy" in str(info.value)


def test_the_numpy_backend_refuses_an_exr_rather_than_guessing(tmp_path: Path):
    """`backend="numpy"` reads .npy only. Silently returning something else for an EXR
    would be worse than refusing."""
    path = tmp_path / "depth.exr"
    path.write_bytes(b"\x76\x2f\x31\x01" + bytes(64))  # EXR magic, then junk
    with pytest.raises(DepthReadError, match="numpy"):
        read_depth_file(path, backend="numpy")


def test_an_unreadable_exr_lists_what_each_backend_said(tmp_path: Path):
    path = tmp_path / "depth.exr"
    path.write_bytes(b"\x00" * 64)
    with pytest.raises(DepthReadError) as info:
        read_depth_file(path, backend="auto")
    message = str(info.value)
    assert "could not read" in message
    # Each attempted backend is named, so a missing OpenEXR is distinguishable from a
    # corrupt file without re-running under a debugger.
    assert "openexr" in message and "install one of" in message


def test_unknown_backend_name(tmp_path: Path):
    path = tmp_path / "depth.exr"
    path.write_bytes(b"\x00" * 64)
    with pytest.raises(ValueError, match="unknown depth backend"):
        read_depth_file(path, backend="pillow")


# -------------------------------------------------------------------- file listing


def test_list_depth_files_finds_both_extensions_sorted(tmp_path: Path):
    (tmp_path / "b.npy").write_bytes(b"")
    (tmp_path / "a.exr").write_bytes(b"")
    # Not depth, and not named as depth: a stray file in the folder is not an error, it
    # just is not listed.
    (tmp_path / "notes.txt").write_text("not depth", encoding="utf-8")

    names = [p.name for p in list_depth_files(tmp_path)]
    assert names == ["a.exr", "b.npy"]


def test_list_depth_files_does_not_descend_into_a_resolution_subtree(tmp_path: Path):
    """Every depth folder in the layout is flat, and `depth_lr/x2` sits *inside* `depth`
    only when someone nests them by mistake. A recursive walk would silently merge the
    native and the low-resolution maps into one training set."""
    (tmp_path / "a.npy").write_bytes(b"")
    (tmp_path / "x2").mkdir()
    (tmp_path / "x2" / "b.npy").write_bytes(b"")

    names = [p.name for p in list_depth_files(tmp_path)]
    assert names == ["a.npy"]


def test_list_depth_files_on_a_missing_directory_is_empty_not_an_error(tmp_path: Path):
    """`symtrs inspect` walks a dataset that may be half-generated; a missing folder is a
    thing to report, not a reason to abort the walk."""
    assert list_depth_files(tmp_path / "absent") == []
