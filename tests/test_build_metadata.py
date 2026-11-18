"""The release manifests (`scripts/build_metadata.py`): parsing, units, and what it writes.

Two kinds of assertion live here. The parsing ones pin the filename grammar, because every
field of `metadata/views.csv` is read straight out of a name and a grammar that drifts would
silently relabel a release. The end-to-end one pins the *reporting*: a view missing one of its
fourteen files must be named in the run's output, since a manifest that lists a view whose night
render never exported is worse than no manifest at all.

Torch-free on purpose (AGENTS.md, Testing): this runs on a machine that has only the layout.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location("build_metadata",
                                               ROOT / "scripts" / "build_metadata.py")
build_metadata = importlib.util.module_from_spec(_spec)
sys.modules["build_metadata"] = build_metadata
_spec.loader.exec_module(build_metadata)

NAME = "rgb_day_2048_loc495_rot271deg_h00_x+061996_y+014713_z+007062"


def write_view(split_root: Path, key: str, skip: set[str] = frozenset()) -> None:
    """Every one of the fourteen export names for one view, minus the skipped ones."""
    for modality, light, resolution in build_metadata.EXPECTED:
        if (modality, light, resolution) in skip:
            continue
        folder = split_root / f"{modality}_{light}_{resolution}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{modality}_{light}_{resolution}_{key}.png").write_bytes(b"")


KEY = "loc495_rot271deg_h00_x+061996_y+014713_z+007062"


def test_parse_name_reads_every_field_off_the_name():
    fields = build_metadata.parse_name(NAME)
    assert fields["location"] == "495"
    assert fields["rotation"] == "271"
    assert fields["altitude"] == "00"
    # The export writes the engine position in centimetres; the manifest is in metres.
    assert fields["x"] == pytest.approx(619.96)
    assert fields["z"] == pytest.approx(70.62)


def test_parse_name_rejects_a_name_that_is_not_an_export():
    assert build_metadata.parse_name("rgb_day_2048") is None
    assert build_metadata.parse_name("README") is None


def test_ground_sample_distance_is_the_footprint_over_the_pixel_count():
    # 70.62 m with a 60 deg horizontal field of view covers 2 * 70.62 * tan(30) = 81.5 m,
    # which is 3.98 cm at 2048 px -- the figure the manuscript quotes for the native level.
    assert build_metadata.gsd_cm(70.62, 2048) == pytest.approx(3.98, abs=0.01)
    for coarse in (1024, 512, 256):
        assert build_metadata.gsd_cm(70.62, coarse) == pytest.approx(
            build_metadata.gsd_cm(70.62, 2048) * 2048 / coarse)


def test_focal_length_matches_the_value_the_manuscript_quotes():
    # f = R / (2 tan 30) is 1773.6 px at 2048 and halves per level; the manuscript says so.
    assert build_metadata.focal_px(2048) == pytest.approx(1773.6, abs=0.1)
    assert build_metadata.focal_px(1024) == pytest.approx(886.8, abs=0.1)
    assert build_metadata.focal_px(256) == pytest.approx(221.7, abs=0.1)


def test_main_writes_the_manifests_and_names_a_view_that_is_incomplete(tmp_path, monkeypatch,
                                                                      capsys):
    for split in ("train", "val"):
        split_root = tmp_path / split
        split_root.mkdir()
        write_view(split_root, KEY)
    test_root = tmp_path / "test"
    test_root.mkdir()
    write_view(test_root, "loc001_rot090deg_h01_x+000100_y+000200_z+010000")
    write_view(test_root, KEY, skip={("depth", "day", "256")})  # one file short

    out = tmp_path / "metadata"
    monkeypatch.setattr(sys, "argv", ["build_metadata.py", "--data-root", str(tmp_path),
                                      "--out", str(out)])
    build_metadata.main()

    assert (out / "splits" / "test.txt").read_text().split() == sorted([KEY, "loc001_rot090deg_h01_x+000100_y+000200_z+010000"])
    rows = (out / "views.csv").read_text().splitlines()
    assert len(rows) == 1 + 4  # header + two views in each of train and val, two in test
    assert {"view_key", "split", "location", "rotation_deg", "altitude_code", "x_m",
            "gsd_2048_cm", "gsd_256_cm"} <= set(rows[0].split(","))

    camera = (out / "camera.yaml").read_text()
    assert "horizontal_fov_deg: 60.0" in camera
    assert "1773.6" in camera
    classes = (out / "classes.yaml").read_text()
    assert "ignore_index: 255" in classes
    assert "car: 252" in classes  # 4 * label_step 63

    # The incomplete view is reported, with the file it is missing.
    reported = capsys.readouterr().out
    assert "missing  256 px day      depth" in reported


def test_main_refuses_a_root_it_does_not_recognise(tmp_path, monkeypatch):
    (tmp_path / "train").mkdir()
    (tmp_path / "val").mkdir()
    (tmp_path / "test").mkdir()
    monkeypatch.setattr(sys, "argv", ["build_metadata.py", "--data-root", str(tmp_path),
                                      "--out", str(tmp_path / "metadata")])
    with pytest.raises(SystemExit) as error:
        build_metadata.main()
    assert "--data-root" in str(error.value)
