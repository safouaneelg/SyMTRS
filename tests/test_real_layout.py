"""The real SyMTRS export: prefixed file names, x63 masks, centimetre depth, noshadow.

`symtrs make-dummy` used to write `000000.png` everywhere, raw class ids, six classes and
metres, so every guard below was invisible until the first run on the real dataset. These
tests pin the real conventions; the last group checks that the dummy now writes them too,
which is what makes "it works on the dummy" mean "it will work on the real data".
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_numpy, requires_pil, requires_torch
from symtrs.data.io import check_pairing, coverage_report, match_by_stem, normalize_stem
from symtrs.data.splits import RESOLUTIONS, resolve_split_dirs, subdir_for

KEY = "loc000_rot152deg_h00_x-049013_y+035369_z+009230"


# ------------------------------------------------------------------ D1: names


@pytest.mark.parametrize(
    "stem",
    [
        f"rgb_day_2048_{KEY}",
        f"rgb_day_512_{KEY}",
        f"rgb_night_2048_{KEY}",
        f"rgb_noshadow_2048_{KEY}",
        f"seg_day_1024_{KEY}",
        f"depth_day_256_{KEY}",
    ],
)
def test_normalize_stem_strips_the_real_prefix_whatever_the_modality_or_size(stem):
    assert normalize_stem(stem) == KEY


def test_normalize_stem_leaves_unprefixed_names_alone():
    assert normalize_stem("000123") == "000123"
    assert normalize_stem("RS.depth.100200") == "RS.100200"


def test_match_by_stem_pairs_an_lr_input_with_its_native_target():
    """The super-resolution pairing: `rgb_day_512_<key>` goes with `rgb_day_2048_<key>`."""
    lr = [Path(f"rgb_day_512_{KEY}.png")]
    hr = [Path(f"rgb_day_2048_{KEY}.png")]
    assert match_by_stem(lr, hr) == [(lr[0], hr[0])]


def test_check_pairing_names_doctor_when_nothing_matches():
    images = [Path("rgb_day_2048_a.png"), Path("rgb_day_2048_b.png")]
    depths = [Path("something_else_1.exr")]
    with pytest.raises(RuntimeError, match=r"symtrs doctor --data-root"):
        check_pairing(match_by_stem(images, depths), images, depths, "image/depth", "train", 2048)


def test_check_pairing_rejects_a_large_mismatch_but_tolerates_a_stray_orphan():
    images = [Path(f"rgb_day_2048_{i}.png") for i in range(20)]
    labels = [Path(f"seg_day_2048_{i}.png") for i in range(19)]  # one orphan: 5%
    check_pairing(match_by_stem(images, labels), images, labels, "image/semantic", "train", 2048)

    labels = [Path(f"seg_day_2048_{i}.png") for i in range(10)]  # half the split vanishes
    with pytest.raises(RuntimeError, match="10 of 20 files matched"):
        check_pairing(match_by_stem(images, labels), images, labels, "image/semantic", "train", 2048)


def test_coverage_report_matches_on_the_key():
    report = coverage_report([Path(f"rgb_day_2048_{KEY}.png")], [Path(f"seg_day_2048_{KEY}.png")])
    assert report["matched"] == 1


# ------------------------------------------------------------- D6: noshadow


def test_noshadow_is_native_only_and_says_why():
    assert subdir_for("noshadow", 2048) == "rgb_noshadow"
    for resolution in RESOLUTIONS[1:]:
        with pytest.raises(ValueError, match="native resolution only"):
            subdir_for("noshadow", resolution)


def test_noshadow_dir_is_none_below_native(layout: Path):
    assert resolve_split_dirs(str(layout), "train", 1024).noshadow is None
    native = resolve_split_dirs(str(layout), "train", 2048)
    assert native.noshadow == layout / "train" / "rgb_noshadow"
    assert native.path_for("noshadow").is_dir()


# ------------------------------------------------------------ D3: x63 labels


@requires_torch
def test_decode_labels_divides_by_the_step():
    import torch

    from symtrs.data.semantic_dataset import decode_labels

    raw = torch.tensor([[0, 63, 126, 189, 252]])
    assert decode_labels(raw, 63).tolist() == [[0, 1, 2, 3, 4]]


@requires_torch
def test_decode_labels_marks_off_grid_values_ignored_instead_of_rounding_them():
    """100 is not a multiple of 63 (and 255 is 3 away from 252): rounding would invent a class."""
    import torch

    from symtrs.data.semantic_dataset import decode_labels

    raw = torch.tensor([[100, 255, 64, 61, 31]])
    decoded = decode_labels(raw, 63, ignore_index=255).tolist()[0]
    assert decoded[:2] == [255, 255]
    assert decoded[2:4] == [1, 1], "one grey level of PNG/resampling noise is still class 1"
    assert decoded[4] == 255


@requires_torch
def test_label_step_one_means_raw_class_ids():
    import torch

    from symtrs.data.semantic_dataset import decode_labels

    raw = torch.tensor([[0, 1, 5, 255]])
    assert decode_labels(raw, 1).tolist() == raw.tolist()


def test_the_defaults_are_the_real_dataset_and_label_step_is_validated():
    from symtrs.config import ConfigError, DataConfig

    data = DataConfig(root="x")
    assert (data.num_classes, data.label_step, data.depth_scale) == (5, 63, 0.01)
    assert data.max_depth is None
    with pytest.raises(ConfigError, match="label_step"):
        DataConfig(root="x", label_step=0).validate()
    with pytest.raises(ConfigError, match="max_depth"):
        DataConfig(root="x", max_depth=-1.0).validate()


# --------------------------------------------------------- D4: cm + max_depth


@requires_numpy
def test_depth_in_centimetres_becomes_metres_and_sky_is_invalid_past_max_depth():
    from symtrs.data.exr import depth_to_tensor

    raw = np.array([[5000.0, 100000.0, 1.0e7, 0.0]], dtype=np.float32)  # cm
    metres, valid = depth_to_tensor(raw, scale=0.01, max_depth=1000.0)
    assert valid.tolist() == [[True, True, False, False]]
    assert metres[0, :2].tolist() == pytest.approx([50.0, 1000.0])
    assert metres[0, 2] == 0.0, "an invalid pixel holds 0, never the sky distance"

    _, unbounded = depth_to_tensor(raw, scale=0.01)
    assert unbounded.tolist() == [[True, True, True, False]], "max_depth is opt-in"


@requires_numpy
def test_max_depth_is_applied_after_scaling_not_to_file_units():
    from symtrs.data.exr import depth_to_tensor

    raw = np.array([[500.0]], dtype=np.float32)
    assert depth_to_tensor(raw, scale=0.01, max_depth=10.0)[1].tolist() == [[True]]  # 5 m
    assert depth_to_tensor(raw, scale=1.0, max_depth=10.0)[1].tolist() == [[False]]  # 500 m


# ------------------------------------------------- D2: staging for pix2pix


def _real_pair_root(tmp_path: Path, count: int = 3) -> Path:
    root = tmp_path / "root"
    day = root / "train" / "rgb_day"
    night = root / "train" / "rgb_night"
    day.mkdir(parents=True)
    night.mkdir(parents=True)
    for index in range(count):
        key = KEY.replace("loc000", f"loc{index:03d}")
        (day / f"rgb_day_2048_{key}.png").write_bytes(b"day")
        (night / f"rgb_night_2048_{key}.png").write_bytes(b"night")
    return root


def test_staged_day_and_night_names_match_so_pix2pix_can_pair_them(tmp_path: Path):
    from symtrs.model.domain.upstream import build_staging_dir

    staging = build_staging_dir(str(_real_pair_root(tmp_path)), str(tmp_path / "run"), "train")
    a = sorted(p.name for p in (staging / "trainA").iterdir())
    b = sorted(p.name for p in (staging / "trainB").iterdir())
    assert a == b and len(a) == 3, "AlignedDataset pairs by exact basename"
    assert all(name.startswith("loc") for name in a)
    assert (staging / "trainA" / a[0]).read_bytes() == b"day"
    assert (staging / "trainB" / b[0]).read_bytes() == b"night"


def test_staging_with_no_matching_names_names_the_doctor(tmp_path: Path):
    from symtrs.model.domain.upstream import UpstreamError, build_staging_dir

    root = _real_pair_root(tmp_path, count=0)
    (root / "train" / "rgb_day" / "rgb_day_2048_a.png").write_bytes(b"x")
    (root / "train" / "rgb_night" / "rgb_night_2048_b.png").write_bytes(b"x")
    with pytest.raises(UpstreamError, match="symtrs doctor --data-root"):
        build_staging_dir(str(root), str(tmp_path / "run"), "train")


def test_staging_falls_back_from_symlink_to_hardlink_to_copy(tmp_path: Path, monkeypatch, capsys):
    """Windows without Developer Mode refuses `os.symlink`; staging must still work."""
    from symtrs.model.domain import upstream

    source = tmp_path / "src.png"
    source.write_bytes(b"pixels")

    def no_symlink(self, target, *args, **kwargs):
        raise OSError("A required privilege is not held by the client")

    monkeypatch.setattr(Path, "symlink_to", no_symlink)
    assert upstream.link_file(source, tmp_path / "hard.png") == "hardlink"
    assert (tmp_path / "hard.png").read_bytes() == b"pixels"

    def no_link(*args, **kwargs):
        raise OSError("cross-device link")

    monkeypatch.setattr(os, "link", no_link)
    assert upstream.link_file(source, tmp_path / "copy.png") == "copy"
    assert (tmp_path / "copy.png").read_bytes() == b"pixels"

    # And end to end, with the fallback reported rather than silent.
    upstream.build_staging_dir(str(_real_pair_root(tmp_path / "e2e")), str(tmp_path / "run"), "train")
    assert "Developer" in capsys.readouterr().out


# ------------------------------------- the dummy writes the real conventions


@requires_torch
@requires_pil
def test_the_dummy_dataset_follows_the_real_conventions(tmp_path: Path):
    from PIL import Image

    from symtrs.commands.make_dummy import CLASSES, _write_split
    from symtrs.data.depth_dataset import DepthTaskDataset
    from symtrs.data.exr import read_depth_file
    from symtrs.data.semantic_dataset import SemanticDataset

    root = tmp_path / "dummy"
    _write_split(root, "train", 2, 64, "npy", True, np.random.default_rng(0))
    split = root / "train"

    assert len(CLASSES) == 5
    assert not (split / "rgb_night_lr").exists(), "night has no downsampled twin"
    assert not (split / "rgb_noshadow_lr").exists()
    assert (split / "rgb_noshadow").is_dir()
    names = sorted(p.name for p in (split / "rgb_day").iterdir())
    assert names[0].startswith("rgb_day_64_loc000_"), names
    assert sorted(p.name for p in (split / "rgb_day_lr" / "x2").iterdir())[0].startswith("rgb_day_32_")
    assert Image.open(split / "rgb_day" / names[0]).mode == "RGBA"

    grey = np.unique(np.array(Image.open(next((split / "semantic").iterdir()))))
    assert set(grey.tolist()) <= {0, 63, 126, 189, 252}

    depth = read_depth_file(next((split / "depth").iterdir()))
    assert depth.max() == 1.0e7, "sky is a huge value in centimetres, as in the real renders"

    seg = SemanticDataset(str(root), "train", 2048)
    sample = seg[0]
    assert sample["image"].shape[0] == 3, "RGBA is dropped to RGB"
    assert int(sample["mask"].max()) <= 4 and set(sample["mask"].unique().tolist()) <= {0, 1, 2, 3, 4}

    dep = DepthTaskDataset(str(root), "train", 2048, max_depth=1000.0)[0]
    assert not dep["valid"][dep["depth"] > 1000.0].any()
    assert dep["depth"][dep["valid"]].max() <= 1000.0
    assert dep["depth"][dep["valid"]].min() > 0.0
