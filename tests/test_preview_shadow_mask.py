"""`scripts/preview_shadow_mask.py`: it must show the mask the runs use, for the pair asked for.

The preview exists so a person can judge the derived mask; if it picked a different pair than
the one named, or thresholded with other settings than a run, it would show something no model
ever saw.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT, requires_torch

sys.path.insert(0, str(REPO_ROOT / "scripts"))

PAIRS = [
    ("loc010_rot045deg_h00_x+1", Path("a"), Path("b")),
    ("loc010_rot045deg_h01_x+1", Path("c"), Path("d")),
    ("loc011_rot010deg_h00_x+2", Path("e"), Path("f")),
]


@requires_torch
def test_a_key_substring_picks_the_first_matching_pair_and_an_unknown_one_stops():
    import preview_shadow_mask as preview

    assert preview.pick(PAIRS, "loc011", 0)[0] == "loc011_rot010deg_h00_x+2"
    assert preview.pick(PAIRS, "h01", 0)[0] == "loc010_rot045deg_h01_x+1"
    with pytest.raises(SystemExit, match="--list"):
        preview.pick(PAIRS, "loc999", 0)


@requires_torch
def test_an_index_out_of_range_stops_instead_of_wrapping():
    import preview_shadow_mask as preview

    assert preview.pick(PAIRS, "", 2)[0] == "loc011_rot010deg_h00_x+2"
    # A negative index would silently show the last pair in Python; the script refuses it.
    with pytest.raises(SystemExit, match="--index"):
        preview.pick(PAIRS, "", -1)


@requires_torch
def test_the_default_settings_are_the_runs_settings(monkeypatch, tmp_path):
    """The defaults come from RunConfig().shadow, so the preview matches what a run derives."""
    import preview_shadow_mask as preview
    from symtrs.config import RunConfig

    seen = {}

    def fake_derive(day, clean, threshold, open_size):
        seen.update(threshold=threshold, open_size=open_size)
        raise SystemExit(0)

    monkeypatch.setattr(preview, "pairs_for", lambda root, split: PAIRS)
    monkeypatch.setattr(preview, "read_rgb", lambda path: None)
    monkeypatch.setattr(preview, "derive_mask", fake_derive)
    monkeypatch.setattr(sys, "argv", ["preview", "--data-root", str(tmp_path),
                                      "--save", str(tmp_path / "x.png")])
    with pytest.raises(SystemExit):
        preview.main()
    shadow = RunConfig().shadow
    assert seen == {"threshold": shadow.mask_threshold, "open_size": shadow.mask_open}
