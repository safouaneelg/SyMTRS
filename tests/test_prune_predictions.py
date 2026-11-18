"""`scripts/prune_predictions.py` must be conservative: it deletes the only copy of an image.

The policy it applies is `eval.retain_predictions` (`tests/test_shadow_removal.py` covers the
scorer side). What these tests pin is the rule that makes it safe to point at a runs tree that
may contain runs still going: a split is touched only when it was scored (`per_image_*.json`
exists), its prediction directory is complete, and it has not already been retained.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from conftest import REPO_ROOT, requires_numpy, requires_pil

SCRIPT = REPO_ROOT / "scripts" / "prune_predictions.py"


def load_module():
    spec = importlib.util.spec_from_file_location("prune_predictions_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_run(root: Path, keys, scored=True, size=600):
    """A run directory in the shape the pruner walks: config.yaml + predictions/<split>."""
    import numpy as np
    from PIL import Image

    run = root / "campaign2" / "model_test"
    run.mkdir(parents=True)
    (run / "config.yaml").write_text(
        (REPO_ROOT / "configs/shadow_removal/shadowformer.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    rng = np.random.default_rng(0)
    for key in keys:
        path = run / "predictions" / "test" / f"{key}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rng.integers(0, 256, (size, size, 3), np.uint8)).save(path)
    if scored:
        rows = [{"key": key, "shadow_fraction": 0.4, "psnr": 20.0} for key in keys]
        (run / "per_image_test.json").write_text(json.dumps(rows), encoding="utf-8")
    return run


KEYS = [
    "loc001_rot000deg_h00_x+0_y+0_z+007000",
    "loc002_rot000deg_h00_x+0_y+0_z+008000",
    "loc003_rot000deg_h01_x+0_y+0_z+010000",
]


@requires_numpy
@requires_pil
def test_an_unscored_split_is_never_touched(tmp_path, capsys):
    # The images are the only copy of the predictions; a run whose scoring did not finish
    # (or is still predicting) must be left exactly as it is.
    module = load_module()
    run = make_run(tmp_path, KEYS, scored=False)
    before = sorted(p.name for p in (run / "predictions" / "test").iterdir())

    module.main(["--runs", str(tmp_path / "campaign2")])

    assert sorted(p.name for p in (run / "predictions" / "test").iterdir()) == before
    assert "not scored" in capsys.readouterr().out


@requires_numpy
@requires_pil
def test_a_dry_run_changes_nothing_and_apply_retains(tmp_path, capsys):
    from PIL import Image

    from symtrs.model.shadow.runner import RETAINED_SIZE, RETAIN_MARKER

    module = load_module()
    run = make_run(tmp_path, KEYS)
    predictions = run / "predictions" / "test"

    module.main(["--runs", str(tmp_path / "campaign2")])
    assert "would prune" in capsys.readouterr().out
    assert not (predictions / RETAIN_MARKER).exists()
    with Image.open(next(predictions.glob("*.png"))) as image:
        assert image.size == (600, 600), "a dry run must not shrink anything"

    module.main(["--runs", str(tmp_path / "campaign2"), "--apply"])
    record = json.loads((predictions / RETAIN_MARKER).read_text(encoding="utf-8"))
    assert record["size"] == RETAINED_SIZE
    # One scene per camera height stays at full resolution, under preview/full/.
    full = sorted((run / "preview" / "full").glob("test_*.png"))
    assert len(full) == 2
    with Image.open(full[0]) as image:
        assert image.size == (600, 600)
    with Image.open(next(predictions.glob("*.png"))) as image:
        assert image.size == (RETAINED_SIZE, RETAINED_SIZE)
    assert json.loads((run / "per_image_test.json").read_text())[0]["key"] == KEYS[0]

    # Idempotent: a second pass knows there is nothing left to do.
    module.main(["--runs", str(tmp_path / "campaign2"), "--apply"])
    assert "already retained" in capsys.readouterr().out


@requires_numpy
@requires_pil
def test_apply_backfills_a_summary_written_before_the_change(tmp_path, capsys):
    # A run scored before `aggregate` reported the spread has a summary.json with means
    # only. The images it was scored from are about to shrink, so the summary is the last
    # place the distribution can live; the pruner adds it, and only additively.
    from symtrs.model.shadow.runner import RETAINED_SIZE

    from symtrs.common.jsonio import dumps as json_dumps_safe

    module = load_module()
    run = make_run(tmp_path, KEYS)
    original = {"test": {"psnr": 20.0, "ssim_y": 0.9}, "train": {"loss": 1.0}}
    (run / "summary.json").write_text(json_dumps_safe(original), encoding="utf-8")

    module.main(["--runs", str(tmp_path / "campaign2")])
    assert json.loads((run / "summary.json").read_text()) == original, "a dry run must not write"

    module.main(["--runs", str(tmp_path / "campaign2"), "--apply"])
    written = json.loads((run / "summary.json").read_text())
    block = written["test"]
    assert block["psnr"] == pytest.approx(20.0), "the mean already there is unchanged"
    assert block["psnr_std"] == pytest.approx(0.0)
    assert block["shadow_fraction_median"] == pytest.approx(0.4)
    assert block["images"] == 3
    assert written["train"] == {"loss": 1.0}, "other splits are left alone"
    assert written["predictions"]["retain"] == "downsample"
    assert written["predictions"]["retained_size"] == RETAINED_SIZE
    assert len(written["predictions"]["full_resolution"]) == 2

    # A second pass has nothing to shrink but still backfills, so pointing it at a tree
    # twice is harmless.
    module.main(["--runs", str(tmp_path / "campaign2"), "--apply"])
    assert "already retained, summary refreshed" in capsys.readouterr().out
