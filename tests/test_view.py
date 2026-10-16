"""`symtrs view`: sibling resolution, mask decoding and --save.

The path tests need nothing but pathlib; the pixel tests skip when numpy / PIL /
matplotlib are missing. There is no EXR writer in the dependency set, so the --save test
stores depth as `.npy` (which `symtrs.data.exr` also reads) and points the planner at it.
"""

from __future__ import annotations

import argparse
import importlib.util

import pytest

from symtrs.commands import view

KEY = "loc008_rot101deg_h00_x+039546_y+054449_z+007073"


def _have(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _touch_layout(root):
    """Empty files with the real names, one transform in `val`."""
    val = root / "val"
    for sub, name in [
        ("rgb_day", f"rgb_day_2048_{KEY}.png"),
        ("rgb_day_lr/x4", f"rgb_day_512_{KEY}.png"),
        ("rgb_night", f"rgb_night_2048_{KEY}.png"),
        ("rgb_noshadow", f"rgb_noshadow_2048_{KEY}.png"),
        ("semantic", f"seg_day_2048_{KEY}.png"),
        ("semantic_lr/x4", f"seg_day_512_{KEY}.png"),
        ("depth", f"depth_day_2048_{KEY}.exr"),
        ("depth_lr/x4", f"depth_day_512_{KEY}.exr"),
    ]:
        (val / sub).mkdir(parents=True, exist_ok=True)
        (val / sub / name).touch()
    return val


def test_sibling_resolves_every_modality_from_any_file(tmp_path):
    val = _touch_layout(tmp_path)
    for start in (val / "rgb_day" / f"rgb_day_2048_{KEY}.png",
                  val / "depth_lr/x4" / f"depth_day_512_{KEY}.exr"):
        assert view.sibling(start, "seg", "day", 2048) == (val / "semantic" / f"seg_day_2048_{KEY}.png").resolve()
        assert view.sibling(start, "depth", "day", 512) == (val / "depth_lr/x4" / f"depth_day_512_{KEY}.exr").resolve()
        assert view.sibling(start, "rgb", "day", 512) == (val / "rgb_day_lr/x4" / f"rgb_day_512_{KEY}.png").resolve()


def test_plan_takes_night_and_noshadow_at_native_resolution(tmp_path):
    val = _touch_layout(tmp_path)
    steps = view.plan(val / "rgb_day_lr/x4" / f"rgb_day_512_{KEY}.png")
    by_title = {t: p for t, _, p in steps}
    assert by_title["RGB night (2048)"].name == f"rgb_night_2048_{KEY}.png"
    assert by_title["RGB day, no shadows (2048)"].parent.name == "rgb_noshadow"
    assert by_title["RGB day"].parent.as_posix().endswith("rgb_day_lr/x4")
    # --res overrides the file's own resolution
    assert view.plan(val / "rgb_day" / f"rgb_day_2048_{KEY}.png", 512)[4][2].parent.name == "x4"


def test_modality_dir_follows_the_resolution_mapping():
    assert view.modality_dir("rgb", "day", 2048) == "rgb_day"
    assert view.modality_dir("seg", "day", 1024) == "semantic_lr/x2"
    assert view.modality_dir("depth", "day", 256) == "depth_lr/x8"


def test_foreign_file_and_foreign_folder_are_refused_with_a_reason(tmp_path):
    with pytest.raises(ValueError, match="does not look like a SyMTRS file"):
        view.parse_name("photo.png")
    with pytest.raises(ValueError, match="not under a train/, val/ or test/"):
        view.split_root(tmp_path / f"rgb_day_2048_{KEY}.png")


def test_list_transforms_is_sorted_and_missing_folder_is_an_error(tmp_path):
    _touch_layout(tmp_path)
    assert [p.name for p in view.list_transforms(tmp_path, "val")] == [f"rgb_day_2048_{KEY}.png"]
    with pytest.raises(FileNotFoundError, match="symtrs doctor"):
        view.list_transforms(tmp_path, "test")


@pytest.mark.skipif(not _have("numpy"), reason="numpy missing")
def test_seg_decode_colorize_and_percentages():
    import numpy as np

    gray = np.array([[0, 63], [189, 252]], dtype=np.uint8)
    ids, odd = view.seg_ids(gray)
    assert ids.tolist() == [[0, 1], [3, 4]] and odd.size == 0
    assert view.colorize(ids)[1, 1].tolist() == [255, 0, 0]  # car is red
    assert view.class_percentages(ids) == [25.0, 25.0, 0.0, 25.0, 25.0]
    _, odd = view.seg_ids(np.array([[100]], dtype=np.uint8))
    assert odd.tolist() == [100]  # a resized mask is reported, not silently rounded


@pytest.mark.skipif(not (_have("numpy") and _have("PIL") and _have("matplotlib")),
                    reason="numpy / PIL / matplotlib missing")
def test_missing_sibling_is_named_not_blanked(tmp_path):
    val = _touch_layout(tmp_path)
    (val / "rgb_night" / f"rgb_night_2048_{KEY}.png").unlink()
    with pytest.raises(FileNotFoundError, match="rgb_night_2048"):
        view.build_figure(val / "rgb_day" / f"rgb_day_2048_{KEY}.png")


@pytest.mark.skipif(not (_have("numpy") and _have("PIL") and _have("matplotlib")),
                    reason="numpy / PIL / matplotlib missing")
def test_save_writes_a_png(tmp_path):
    import numpy as np
    from PIL import Image

    val = _touch_layout(tmp_path)
    rgba = Image.fromarray(np.full((8, 8, 4), 200, dtype=np.uint8), "RGBA")
    for p in val.rglob("*.png"):
        if p.name.startswith("seg_"):
            Image.fromarray(np.array([[0, 63] * 4] * 8, dtype=np.uint8)).save(p)
        else:
            rgba.save(p)
    # Depth as .npy: the package's reader accepts it, and the name keeps the .exr stem
    # contract by pointing the plan at a file that exists. Patch the planner for this one.
    depth = np.linspace(100, 5000, 64, dtype=np.float32).reshape(8, 8)  # centimetres
    for p in val.rglob("*.exr"):
        np.save(p.with_suffix(".npy"), depth)
    orig = view.sibling

    def sibling_npy(path, kind, light, res):
        out = orig(path, kind, light, res)
        return out.with_suffix(".npy") if kind == "depth" else out

    view.sibling = sibling_npy
    try:
        out = tmp_path / "fig" / "view.png"
        # Parsed through the command's own parser, not symtrs.cli: the full CLI imports
        # every subcommand, and some of them import torch.
        root = argparse.ArgumentParser()
        view.add_parser(root.add_subparsers())
        args = root.parse_args(["view", "--data-root", str(tmp_path), "--split", "val",
                                "--index", "0", "--save", str(out)])
        code = args.handler(args)
    finally:
        view.sibling = orig
    assert code == 0
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
