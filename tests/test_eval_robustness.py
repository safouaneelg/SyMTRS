"""Small failures that took a whole evaluation or a results file with them.

  * an image with a single valid pixel passed the metrics and then crashed the alignment fit;
  * a metrics column that first appears in a later epoch (`val_skipped_images`) was written
    under a header that did not have it;
  * `--set data.max_depth=1000` was stored as a string because the field's current value is
    None, and validation then failed comparing it with 0.
"""

from __future__ import annotations

import csv

import pytest

from conftest import requires_torch


# ------------------------------------------------------------------- the CSV header


def test_a_new_metric_column_widens_the_header_and_keeps_earlier_rows(tmp_path):
    from symtrs.common.table import append_csv_row, read_csv_rows

    path = str(tmp_path / "metrics.csv")
    append_csv_row(path, {"epoch": 0, "val_rmse": 1.5})
    append_csv_row(path, {"epoch": 1, "val_rmse": 1.2, "val_skipped_images": 2.0})
    append_csv_row(path, {"epoch": 2, "val_rmse": 1.0})

    with open(path, newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    assert header == ["epoch", "val_rmse", "val_skipped_images"]

    rows = read_csv_rows(path)
    assert [row["epoch"] for row in rows] == ["0", "1", "2"]
    # The value sits under its own name, not as an unlabeled trailing cell.
    assert rows[1]["val_skipped_images"] == "2.0"
    assert rows[0]["val_skipped_images"] == "" and rows[2]["val_skipped_images"] == ""
    assert None not in rows[1], "no unlabeled overflow cell"


# --------------------------------------------------------------------- --set coercion


def test_set_coerces_an_optional_float_that_is_currently_none():
    from symtrs.config import RunConfig

    cfg = RunConfig()
    assert cfg.data.max_depth is None
    updated = cfg.with_overrides({"data.max_depth": "1000", "data.root": "."})
    assert updated.data.max_depth == 1000.0 and isinstance(updated.data.max_depth, float)
    updated.data.validate()  # used to raise TypeError comparing str <= 0


def test_set_can_unset_an_optional_field_and_rejects_a_non_number():
    from symtrs.config import ConfigError, RunConfig

    cfg = RunConfig().with_overrides({"data.max_depth": "80"})
    assert cfg.with_overrides({"data.max_depth": "none"}).data.max_depth is None
    with pytest.raises(ConfigError, match="number"):
        RunConfig().with_overrides({"data.max_depth": "far"})


# ------------------------------------------------- one valid pixel in the depth eval


@requires_torch
def test_an_image_with_one_valid_pixel_is_skipped_and_counted_not_fatal():
    torch = pytest.importorskip("torch")
    from symtrs.config import RunConfig
    from symtrs.engine.loop import evaluate_depth

    class _Depth(torch.nn.Module):
        protocol = "metric"
        default_align = "scale"
        align = "scale"

        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))

        def preprocess_image(self, image):
            return image

        def forward(self, image):
            return image[:, :1] + 1.0

        def to_metres(self, prediction):
            return prediction

    def sample(valid_pixels: int):
        valid = torch.zeros(1, 1, 8, 8, dtype=torch.bool)
        valid.view(-1)[:valid_pixels] = True
        return {
            "image": torch.rand(1, 3, 8, 8),
            "depth": torch.rand(1, 1, 8, 8) + 1.0,
            "valid": valid,
        }

    cfg = RunConfig()
    cfg.task = "depth"
    cfg.eval.amp = False
    cfg.eval.align = "scale"
    results = evaluate_depth(
        _Depth(), [sample(1), sample(40)], cfg, torch.device("cpu"), align="scale"
    )
    assert results["skipped_images"] == 1.0
    assert "rmse" in results, "the scorable image still counts"
