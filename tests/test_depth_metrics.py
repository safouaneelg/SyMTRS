"""Depth metrics and per-image alignment.

The expectations below are hand-computed from the definitions, not captured from a run of
the code, so a change to a formula has to be argued for in these numbers rather than
absorbed silently. That matters more here than almost anywhere else in the repo: these
are the numbers the result tables are made of, and two of the metrics (`silog`, `abs_rel`)
are only comparable to published tables if the exact convention is held fixed.

Torch is needed, so this module skips on a machine without it.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch", reason="torch is not installed")

from symtrs.metrics.depth import (  # noqa: E402
    ALIGNMENTS,
    DEPTH_METRIC_NAMES,
    AlignmentFit,
    AllInvalidError,
    align_prediction,
    compute_depth_metrics,
    summarise_alignment,
)


def _quad(values):
    """A `[1, 1, 1, N]` float32 tensor, the shape the pipeline actually feeds these."""
    return torch.tensor([[[list(values)]]], dtype=torch.float32)


def _mask(shape, fill=True):
    return torch.full(tuple(shape), fill, dtype=torch.bool)


# ------------------------------------------------------------------- metric formulas


def test_none_alignment_metrics_match_hand_computation():
    """pred [1, 2, 4] against target [1, 4, 2]: signed errors 0, -2, +2."""
    metrics = compute_depth_metrics(_quad([1.0, 2.0, 4.0]), _quad([1.0, 4.0, 2.0]), align="none")

    assert set(metrics) == set(DEPTH_METRIC_NAMES)
    assert metrics["rmse"] == pytest.approx(math.sqrt(8.0 / 3.0), abs=1e-6)
    assert metrics["mae"] == pytest.approx(4.0 / 3.0, abs=1e-6)
    # |d|/t = 0, 2/4, 2/2
    assert metrics["abs_rel"] == pytest.approx(0.5, abs=1e-6)
    # d^2/t = 0, 4/4, 4/2
    assert metrics["sq_rel"] == pytest.approx(1.0, abs=1e-6)
    # ratio = max(p/t, t/p) = 1, 2, 2. 2 exceeds 1.25^3 = 1.953, so all three thresholds
    # hold exactly one of three pixels.
    assert metrics["d1"] == pytest.approx(1.0 / 3.0, abs=1e-6)
    assert metrics["d2"] == pytest.approx(1.0 / 3.0, abs=1e-6)
    assert metrics["d3"] == pytest.approx(1.0 / 3.0, abs=1e-6)
    # log errors are 0, +ln2, -ln2: mean 0, mean square 2(ln2)^2/3.
    assert metrics["silog"] == pytest.approx(56.5952, abs=1e-2)


def test_a_metric_model_is_not_silently_rescaled_by_default():
    """`align` defaults to `none`, so a model that already predicts metres is scored as
    it stands. If the default ever became `scale`, every metric table would improve and
    the improvement would be an artefact of the scoring code."""
    pred = _quad([2.0, 4.0, 8.0])
    target = _quad([1.0, 2.0, 4.0])
    assert compute_depth_metrics(pred, target)["abs_rel"] == pytest.approx(1.0, abs=1e-6)
    # Aligned, the same prediction is exact - which is exactly why the default matters.
    assert compute_depth_metrics(pred, target, align="scale")["abs_rel"] == pytest.approx(
        0.0, abs=1e-6
    )


def test_the_validity_mask_comes_from_the_ground_truth():
    """The third pixel is "no geometry" in the target. Counting it as 0 m would give
    mae = 34 instead of 1, and rmse would be dominated by a pixel that has no depth."""
    metrics = compute_depth_metrics(_quad([1.0, 2.0, 100.0]), _quad([1.0, 4.0, 0.0]))
    assert metrics["mae"] == pytest.approx(1.0, abs=1e-6)


def test_an_explicit_mask_is_intersected_with_the_validity_rule():
    pred = _quad([1.0, 2.0, 4.0])
    target = _quad([1.0, 4.0, 2.0])
    mask = _mask((1, 1, 1, 3))
    mask[..., 1] = False
    mask[..., 2] = False
    metrics = compute_depth_metrics(pred, target, mask=mask)
    # Only the first pixel survives, and it is exact. Both of the pixels the mask drops
    # are wrong, so a mask that was ignored would show up as error here.
    assert metrics["mae"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["d1"] == pytest.approx(1.0, abs=1e-6)


def test_an_all_invalid_batch_is_an_error_not_a_zero():
    """Reporting 0 m of error for a tile with no geometry would look like a perfect
    prediction."""
    with pytest.raises(AllInvalidError):
        compute_depth_metrics(torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 2, 2))


def test_a_shape_mismatch_is_refused_before_any_arithmetic():
    with pytest.raises(ValueError, match="does not match"):
        compute_depth_metrics(torch.ones(1, 1, 2, 3), torch.ones(1, 1, 3, 2))


def test_metric_names_are_exactly_the_keys_that_are_returned():
    """`DEPTH_METRIC_NAMES` drives every CSV column and result table, so a metric that is
    computed but not named - or named but not computed - is a silently shifted table."""
    metrics = compute_depth_metrics(_quad([1.0, 2.0]), _quad([1.0, 4.0]))
    assert tuple(metrics) == DEPTH_METRIC_NAMES


# ------------------------------------------------------------------------ alignment


def test_scale_shift_recovers_an_exact_affine_map():
    pred = _quad([1.0, 2.0, 3.0, 4.0])
    target = 3.0 * pred + 5.0
    aligned, fits = align_prediction(pred, target, _mask(pred.shape), mode="scale_shift")

    assert fits[0].mode == "scale_shift"
    assert fits[0].scale == pytest.approx(3.0, abs=1e-5)
    assert fits[0].shift == pytest.approx(5.0, abs=1e-5)
    assert torch.allclose(aligned, target, atol=1e-4)


def test_scale_only_uses_the_least_squares_scale_with_no_shift():
    """`<p, t> / <p, p>` with the means removed from neither. For p = [1..4] and
    t = 3p + 5 that is 140/30, not the 3.0 that a shifted model would recover."""
    pred = _quad([1.0, 2.0, 3.0, 4.0])
    target = 3.0 * pred + 5.0
    _, fits = align_prediction(pred, target, _mask(pred.shape), mode="scale")

    assert fits[0].scale == pytest.approx(140.0 / 30.0, abs=1e-5)
    assert fits[0].shift == 0.0


def test_median_alignment_uses_the_ratio_of_medians():
    """The convention several relative-depth papers use. Odd length here on purpose:
    `torch.median` on an even-length input takes the lower middle value rather than the
    average of the two, and this test should be about our convention, not torch's."""
    pred = _quad([1.0, 2.0, 3.0, 4.0, 5.0])
    target = _quad([10.0, 20.0, 30.0, 40.0, 50.0])
    aligned, fits = align_prediction(pred, target, _mask(pred.shape), mode="median")

    assert fits[0].scale == pytest.approx(10.0, abs=1e-6)
    assert fits[0].shift == 0.0
    assert torch.allclose(aligned, target, atol=1e-5)


def test_alignment_is_fitted_per_image_so_one_image_cannot_rescale_another():
    """A shared fit across the batch would give both images a compromise scale, and a
    model that is right on one image and wrong on the other would score as two mediocre
    images instead of one right and one wrong."""
    pred = _quad([1.0, 2.0, 3.0, 4.0]).repeat(2, 1, 1, 1)
    target = torch.cat([2.0 * pred[:1], 10.0 * pred[1:]], dim=0)
    _, fits = align_prediction(pred, target, _mask(pred.shape), mode="scale")

    assert fits[0].scale == pytest.approx(2.0, abs=1e-5)
    assert fits[1].scale == pytest.approx(10.0, abs=1e-5)


def test_none_alignment_returns_the_prediction_untouched():
    pred = _quad([1.0, 2.0, 3.0, 4.0])
    target = _quad([4.0, 3.0, 2.0, 1.0])
    aligned, fits = align_prediction(pred, target, _mask(pred.shape), mode="none")

    assert aligned is pred
    assert fits[0] == AlignmentFit(mode="none", scale=1.0, shift=0.0)


def test_a_three_dimensional_batch_keeps_its_shape():
    """The engine passes `[B, H, W]` in some paths and `[B, 1, H, W]` in others; an
    unsqueezed-aligned-squeezed round trip that loses the channel axis is an easy slip."""
    pred = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]], dtype=torch.float32)
    target = 2.0 * pred
    aligned, fits = align_prediction(pred, target, _mask(pred.shape), mode="scale")

    assert aligned.shape == pred.shape
    assert fits[0].scale == pytest.approx(2.0, abs=1e-5)


def test_alignment_needs_at_least_two_pixels_to_fit():
    pred = torch.ones(1, 1, 2, 2)
    target = torch.ones(1, 1, 2, 2)
    mask = _mask(pred.shape, fill=False)
    mask[0, 0, 0, 0] = True
    with pytest.raises(AllInvalidError, match="at least 2"):
        align_prediction(pred, target, mask, mode="scale")


def test_an_unknown_alignment_is_refused():
    pred = torch.ones(1, 1, 2, 2)
    with pytest.raises(ValueError, match="unknown alignment"):
        align_prediction(pred, pred, _mask(pred.shape), mode="l1")
    assert "none" in ALIGNMENTS and "scale_shift" in ALIGNMENTS


def test_an_image_too_small_to_fit_is_left_alone_rather_than_aborting_the_batch():
    """One tile with almost no geometry must not take down an evaluation run. Its
    unaligned prediction shows up as error, which is the honest outcome."""
    pred = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]], [[[20.0, 0.0, 0.0, 0.0]]]])
    target = torch.tensor([[[[2.0, 4.0, 6.0, 8.0]]], [[[9.0, 0.0, 0.0, 0.0]]]])
    metrics = compute_depth_metrics(pred, target, align="scale")

    # Image 0 aligns exactly (scale 2). Image 1 has one valid pixel, so it is untouched
    # and carries an 11 m error. 5 valid pixels in total.
    assert metrics["mae"] == pytest.approx(11.0 / 5.0, abs=1e-5)
    assert metrics["rmse"] == pytest.approx(math.sqrt(121.0 / 5.0), abs=1e-4)


def test_summarise_alignment_reports_the_mean_fit():
    fits = [AlignmentFit(mode="scale", scale=2.0), AlignmentFit(mode="scale", scale=4.0)]
    summary = summarise_alignment(fits)
    assert summary["align_scale_mean"] == pytest.approx(3.0)
    assert summary["align_shift_mean"] == pytest.approx(0.0)
    assert summarise_alignment([]) == {}
