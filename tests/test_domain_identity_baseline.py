"""The "do nothing" domain-adaptation reference (`scripts/domain_identity_baseline.py`).

The pure parts - pairing by key, the two deterministic noise-floor splits - run without torch
by injecting the key function; the parts that need the repository's own key and direction
table are marked `requires_torch` (importing `symtrs.model` pulls torch in).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT, requires_torch

sys.path.insert(0, str(REPO_ROOT / "scripts"))

import domain_identity_baseline as baseline  # noqa: E402


def _by_name(path: Path) -> str:
    return path.stem.split("_")[-1]


def _files(prefix: str, keys) -> list:
    return [Path(f"{prefix}_{k}.png") for k in keys]


def test_pairs_follow_the_key_not_the_listing_order():
    inputs = _files("rgb_day", ["c", "a", "b"])
    targets = _files("rgb_night", ["b", "c", "a"])
    pairs = baseline.pair_by_key(inputs, targets, key=_by_name)
    assert [k for k, _, _ in pairs] == ["a", "b", "c"]
    assert all(_by_name(i) == _by_name(t) == k for k, i, t in pairs)


def test_unpaired_files_are_dropped_and_said_so(capsys):
    pairs = baseline.pair_by_key(
        _files("d", ["a", "b", "z"]), _files("n", ["a", "b", "y"]), key=_by_name
    )
    assert [k for k, _, _ in pairs] == ["a", "b"]
    assert "2 file(s) have no counterpart" in capsys.readouterr().out


def test_no_shared_key_is_an_error_naming_the_check():
    with pytest.raises(ValueError, match="symtrs doctor"):
        baseline.pair_by_key(_files("d", ["a"]), _files("n", ["b"]), key=_by_name)


def test_a_duplicated_key_is_refused_instead_of_dropping_a_file():
    with pytest.raises(ValueError, match="share the pairing key"):
        baseline.pair_by_key(
            [Path("x/a_1.png"), Path("y/a_1.jpg")], _files("n", ["1"]), key=_by_name
        )


def test_halves_are_by_sorted_key_and_do_not_depend_on_input_order():
    keys = ["k5", "k1", "k4", "k2", "k3", "k0"]
    first, second = baseline.split_halves(_files("t", keys), key=_by_name)
    assert [_by_name(p) for p in first] == ["k0", "k1", "k2"]
    assert [_by_name(p) for p in second] == ["k3", "k4", "k5"]
    # deterministic: a shuffled listing gives the same halves
    again = baseline.split_halves(_files("t", reversed(keys)), key=_by_name)
    assert (first, second) == again


def test_odd_counts_put_the_extra_image_in_the_second_half():
    first, second = baseline.split_halves(_files("t", "abcde"), key=_by_name)
    assert (len(first), len(second)) == (2, 3)


def test_interleaved_split_mixes_neighbours():
    even, odd = baseline.split_interleaved(_files("t", "dcbaef"), key=_by_name)
    assert [_by_name(p) for p in even] == ["a", "c", "e"]
    assert [_by_name(p) for p in odd] == ["b", "d", "f"]


@pytest.mark.parametrize("split", [baseline.split_halves, baseline.split_interleaved])
def test_too_few_images_cannot_give_a_noise_floor(split):
    with pytest.raises(ValueError, match="at least 4 images"):
        split(_files("t", "abc"), key=_by_name)


@requires_torch
def test_default_key_is_the_runners_pairing_key():
    day = Path("rgb_day_2048_loc010_rot045deg_h00_x+057749.png")
    night = Path("rgb_night_2048_loc010_rot045deg_h00_x+057749.png")
    assert baseline.default_key(day) == baseline.default_key(night) == "loc010_rot045deg_h00_x+057749"


@requires_torch
def test_roles_come_from_the_direction_table_and_target_is_what_the_runner_scores_against():
    from symtrs.model.domain import upstream

    assert {d: baseline.roles_for(d) for d in upstream.DIRECTION_ROLES} == {
        "day2night": ("day", "night"),
        "night2day": ("night", "day"),
        "shadow2noshadow": ("day", "noshadow"),
        "noshadow2shadow": ("noshadow", "day"),
    }
    for direction in upstream.DIRECTION_ROLES:
        assert baseline.roles_for(direction)[1] == upstream.target_role(direction)


@requires_torch
def test_report_adds_identity_rows_and_the_noise_floor(tmp_path):
    import json

    import results_report

    base = tmp_path / "_baselines"
    base.mkdir()
    entry = {"kid": 0.2306, "kid_std": None, "fid": 319.0, "fid_rank_deficient": 1.0, "n_fake": 24.0}
    (base / "domain_identity.json").write_text(json.dumps({
        "directions": {"day2night": entry},
        "noise_floor": {"night": {"kid": 0.1277, "kid_interleaved": 0.0091, "n_first": 12, "n_second": 12}},
    }), encoding="utf-8")
    rows = results_report._domain_identity_rows(
        tmp_path, results_report.COLUMNS["domain_adapt"], ["day2night", "night2day"]
    )
    assert len(rows) == 1  # a direction that was not scored gets no row
    assert rows[0].startswith("| identity (input as output) | day2night | 0.2306 | n/a |")
    note = " ".join(results_report._domain_noise_floor_note(tmp_path))
    assert "a translator must beat this KID" in note and "night 0.1277" in note
    # nothing scored yet -> no rows and no note, not an error
    assert results_report._domain_identity_rows(tmp_path / "none", results_report.COLUMNS["domain_adapt"], ["day2night"]) == []
    assert results_report._domain_noise_floor_note(tmp_path / "none") == []
