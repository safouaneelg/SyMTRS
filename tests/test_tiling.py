"""Sliding-window inference: coverage, and the stitch.

A bad stitch does not crash. It shows up as faint seams along tile boundaries, which at
2048x2048 with a 512 tile is a grid every quarter of the image - and on a depth map that
reads as a model that is slightly worse than it is. So the properties pinned here are the
ones that would fail silently: every pixel covered at least once, no gap or overhang at
the right and bottom edges, and a pointwise model reproduced exactly through the blend.

Torch is needed, so this module skips on a machine without it.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="torch is not installed")

from symtrs.engine.tiling import (  # noqa: E402
    hann_window,
    plan_tiling,
    predict_tiled,
    tile_grid,
    tile_positions,
)


def _covered_mask(height, width, boxes):
    """Which pixels the tile grid touches, as a boolean array laid out by hand."""
    mask = [[0] * width for _ in range(height)]
    for top, left, tile_h, tile_w in boxes:
        for y in range(top, top + tile_h):
            for x in range(left, left + tile_w):
                mask[y][x] += 1
    return mask


# ---------------------------------------------------------------------- positions


def test_tile_positions_end_flush_with_the_edge():
    """The last window is clamped to `length - tile`, not allowed to overhang: an
    overhanging window is mostly padding, and the padding is exactly what the network is
    worst at."""
    assert tile_positions(1000, 512, 384) == [0, 384, 488]
    assert tile_positions(1000, 512, 384)[-1] == 1000 - 512


def test_tile_positions_when_the_tile_covers_the_whole_axis():
    assert tile_positions(256, 512, 384) == [0]


def test_every_pixel_is_covered_at_least_once():
    """A gap is not a seam, it is a zero - and a zero in a depth map is "no geometry"."""
    for height, width in ((2048, 2048), (2048, 1365), (700, 300)):
        boxes = tile_grid(height, width, 512, 128)
        mask = _covered_mask(height, width, boxes)
        assert all(count > 0 for row in mask for count in row), (height, width)


def test_no_window_overhangs_the_image():
    for height, width in ((2048, 2048), (2048, 1365), (700, 300)):
        for top, left, tile_h, tile_w in tile_grid(height, width, 512, 128):
            assert top >= 0 and left >= 0
            assert top + tile_h <= height and left + tile_w <= width


def test_the_overlap_is_the_overlap_that_was_asked_for():
    """`overlap` is the number of pixels neighbouring windows share, so the step is
    `tile - overlap`. An off-by-one here shifts every seam in the image."""
    boxes = tile_grid(2048, 2048, 512, 128)
    tops = sorted({box[0] for box in boxes})
    assert tops[1] - tops[0] == 512 - 128


def test_an_overlap_as_large_as_the_tile_still_advances():
    """`config` refuses this, but the function is also called from `plan_tiling`, and a
    zero step would loop forever rather than raise."""
    boxes = tile_grid(1024, 1024, 512, 512)
    tops = sorted({box[0] for box in boxes})
    assert tops[1] - tops[0] == 1


def test_a_non_positive_tile_is_refused():
    with pytest.raises(ValueError, match="tile_size"):
        tile_grid(512, 512, 0, 0)
    with pytest.raises(ValueError, match="tile_overlap"):
        tile_grid(512, 512, 128, -1)


# ------------------------------------------------------------------------- window


def test_the_window_peaks_in_the_middle_and_never_reaches_zero():
    window = hann_window(64, 64)
    assert window.shape == (64, 64)
    # Not exactly 1: with `periodic=False` over 64 samples the peak lands between samples.
    assert float(window.max()) == pytest.approx(1.0, abs=2e-3)
    assert window[32, 32] > window[0, 32] > 0.0
    assert float(window.min()) > 0.0


def test_the_window_is_separable_so_it_weights_both_axes():
    """A 1-D window broadcast along one axis only would down-weight seams on one side of
    every tile and leave the other untouched."""
    window = hann_window(64, 64)
    assert float(window[0, 32]) == pytest.approx(float(window[32, 0]), abs=1e-6)
    assert float(window[0, 0]) < float(window[0, 32])


# -------------------------------------------------------------------------- stitch


def _pointwise(image: torch.Tensor) -> torch.Tensor:
    """A stand-in for a depth model: pointwise, so the stitched result is exact."""
    return image * image


def test_a_pointwise_model_is_reproduced_exactly_through_the_blend():
    """The strongest statement available without a real network. Every window that covers
    a pixel contributes the same value there, so the weighted average is that value - no
    matter how the weights vary. Any seam, gap or double-count in the stitch shows up
    here as a difference from the untiled answer."""
    image = torch.linspace(0.0, 1.0, 1024 * 1024).reshape(1, 1, 1024, 1024)
    stitched = predict_tiled(image, _pointwise, tile_size=256, overlap=64)
    assert stitched.shape == image.shape
    assert torch.allclose(stitched, _pointwise(image), atol=1e-5)


def test_a_tile_larger_than_the_image_is_a_single_window_passthrough():
    image = torch.rand(1, 1, 128, 96)
    stitched = predict_tiled(image, _pointwise, tile_size=512, overlap=64)
    assert torch.allclose(stitched, _pointwise(image), atol=1e-6)


def test_a_non_square_image_stitches_back_to_its_own_shape():
    """The right and bottom edges are where a clamped tile grid goes wrong, and a
    non-square image is what makes it visible."""
    image = torch.rand(1, 1, 700, 300)
    stitched = predict_tiled(image, _pointwise, tile_size=256, overlap=64)
    assert stitched.shape == (1, 1, 700, 300)


def test_a_batch_is_tiled_one_image_at_a_time_and_stacked_back():
    image = torch.rand(3, 2, 300, 300)
    stitched = predict_tiled(image, _pointwise, tile_size=128, overlap=32)
    assert stitched.shape == (3, 2, 300, 300)
    for index in range(3):
        assert torch.allclose(stitched[index], _pointwise(image[index]), atol=1e-5)


def test_a_layout_style_input_keeps_its_channel_axis():
    image = torch.rand(3, 300, 300)
    assert predict_tiled(image, _pointwise, tile_size=128, overlap=32).shape == (1, 3, 300, 300)


def test_a_multichannel_prediction_is_blended_per_channel():
    """Segmentation logits, not indices: the blend has to happen in logit space and keep
    every class channel, or the argmax afterwards is taken on a blended label."""
    image = torch.rand(1, 1, 256, 256)
    stitched = predict_tiled(image, lambda tile: tile.repeat(1, 4, 1, 1), tile_size=128, overlap=32)
    assert stitched.shape == (1, 4, 256, 256)
    for channel in range(4):
        assert torch.allclose(stitched[0, channel], image[0, 0], atol=1e-5)


def test_an_output_at_a_different_resolution_is_resized_before_blending():
    """These networks downsample internally - a ViT-L at 512 emits a 32x32 map. Resizing
    after the blend would be the same answer; resizing the wrong thing is not."""
    image = torch.rand(1, 1, 400, 400)
    half = lambda tile: torch.nn.functional.avg_pool2d(tile, 2)  # noqa: E731
    stitched = predict_tiled(image, half, tile_size=200, overlap=50)
    assert stitched.shape == (1, 1, 400, 400)
    # avg_pool2d over 2x2 is a box filter, and applying it per tile then blending is not
    # the same as applying it once to the whole image - but the result must lie inside
    # the input's range, not outside it, which a mis-scaled resize would violate.
    assert float(stitched.min()) >= float(image.min()) - 1e-5
    assert float(stitched.max()) <= float(image.max()) + 1e-5


def test_a_non_positive_tile_is_refused_before_any_forward_pass():
    called = []
    with pytest.raises(ValueError, match="tile_size must be positive"):
        predict_tiled(torch.rand(1, 1, 64, 64), lambda tile: called.append(1), tile_size=0)
    assert called == []


# -------------------------------------------------------------------- plan_tiling


def test_plan_tiling_counts_the_windows_and_the_overlap_they_cost():
    plan = plan_tiling(2048, 2048, 512, 128)
    # 2048 with tile 512 and step 384: positions 0, 384, 768, 1152, 1536 = 5 per axis.
    assert plan["tiles"] == 25
    assert plan["coverage"] == pytest.approx(25 * 512 * 512 / (2048 * 2048))
    assert plan["coverage"] > 1.0  # overlapping tiles cover some pixels more than once


def test_plan_tiling_for_a_whole_image_pass_is_one_tile():
    plan = plan_tiling(512, 512, 0, 0)
    assert plan["tiles"] == 1 and plan["coverage"] == 1.0


# ------------------------------------------------ window matching: robustness and families

from symtrs.engine.tiling import SLOPE_RANGE, fit_overlap, match_mode  # noqa: E402


def _textured(size=96, seed=0):
    generator = torch.Generator().manual_seed(seed)
    noise = torch.nn.functional.interpolate(
        torch.rand(1, 1, 12, 12, generator=generator), size=(size, size), mode="bilinear"
    )
    return 1.0 + noise  # strictly positive, smooth-ish texture everywhere


def _per_window_distortion(field, scales, shifts):
    """forward_fn that applies a different (scale, shift) on each call, in raster order."""
    calls = {"n": 0}
    image = field.expand(1, 3, *field.shape[-2:]).contiguous()

    def forward(tile):
        n = calls["n"]
        calls["n"] += 1
        return tile[:, :1] * scales[n % len(scales)] + shifts[n % len(shifts)]

    return image, forward


def _global_affine_error(field, stitched):
    x = field.flatten()
    design = torch.stack([x, torch.ones_like(x)], dim=1)
    solution = torch.linalg.lstsq(design, stitched.flatten().unsqueeze(1)).solution
    return float((design @ solution - stitched.flatten().unsqueeze(1)).abs().max() / stitched.abs().max())


def test_affine_matching_recovers_random_per_window_scale_and_shift():
    field = _textured()
    image, forward = _per_window_distortion(field, [1.0, 1.7, 0.6, 2.3], [0.0, 0.4, -0.3])
    stitched = predict_tiled(image, forward, 48, 24, match="affine")
    assert _global_affine_error(field, stitched) < 1e-3

    image, forward = _per_window_distortion(field, [1.0, 1.7, 0.6, 2.3], [0.0, 0.4, -0.3])
    raw = predict_tiled(image, forward, 48, 24)
    assert _global_affine_error(field, raw) > 0.05, "the distortion must be visible without matching"


def test_scale_matching_recovers_scales_and_never_fits_a_shift():
    """Depth up to a scale: windows differ by a scale only. A fitted shift would be an offset the
    final scale-only alignment cannot remove, so the mode must return shift == 0 always."""
    field = _textured()
    image, forward = _per_window_distortion(field, [1.0, 1.7, 0.6, 2.3], [0.0])
    stitched = predict_tiled(image, forward, 48, 24, match="scale")
    scale = (stitched * field).sum() / (field * field).sum()
    assert float((stitched - scale * field).abs().max() / stitched.abs().max()) < 1e-3, (
        "a pure scale ambiguity must be removed by a pure scale"
    )
    source = torch.rand(400) + 1.0
    slope, shift, _ = fit_overlap(source, 2.0 * source + 5.0, torch.ones(400), "scale")
    assert shift == 0.0, "the offset in the reference is NOT absorbed: that is the point"


def test_affine_fit_on_a_flat_overlap_is_shift_only_and_counted():
    source = torch.full((400,), 3.0) + 1e-9 * torch.rand(400)
    reference = torch.full((400,), 5.0)
    slope, shift, degraded = fit_overlap(source, reference, torch.ones(400), "affine")
    assert slope == 1.0 and shift == pytest.approx(2.0, abs=1e-6) and degraded


def test_the_flat_threshold_is_relative_to_the_units_of_the_model():
    """An absolute 1e-12 variance threshold treats a field in 1e-6 units as flat, and one in
    1e6 units with the same relative texture as textured. Scaling the field must not change it."""
    base = torch.rand(400) * 0.2 + 1.0
    for unit in (1e-6, 1.0, 1e6):
        fit = fit_overlap(base * unit, (2.0 * base + 0.3) * unit, torch.ones(400), "affine")
        assert fit is not None and not fit[2], unit
        assert fit[0] == pytest.approx(2.0, rel=1e-3)


def test_the_fitted_slope_is_clamped_and_the_clamp_is_counted():
    source = torch.rand(400)
    slope, shift, degraded = fit_overlap(source, 50.0 * source, torch.ones(400), "affine")
    assert slope == SLOPE_RANGE[1] and degraded
    assert shift == pytest.approx(float((50.0 * source).mean() - SLOPE_RANGE[1] * source.mean()), rel=1e-4)


def test_an_anticorrelated_overlap_is_not_flipped():
    source = torch.rand(400)
    slope, shift, degraded = fit_overlap(source, -source, torch.ones(400), "affine")
    assert slope == 1.0 and degraded, "a relative model is monotone in the truth: do not flip"


def test_noise_does_not_bias_the_slope_toward_zero():
    """OLS cov/var shrinks the slope by var_signal / (var_signal + var_noise); the symmetric
    slope does not."""
    generator = torch.Generator().manual_seed(1)
    truth = torch.rand(20000, generator=generator)
    source = truth + 0.3 * torch.randn(20000, generator=generator)
    reference = 2.0 * truth + 0.6 * torch.randn(20000, generator=generator)
    slope, _, _ = fit_overlap(source, reference, torch.ones(20000), "affine")
    ols = float(((source - source.mean()) * (reference - reference.mean())).mean() / source.var(unbiased=False))
    assert abs(slope - 2.0) < abs(ols - 2.0)


def test_a_flat_half_is_no_worse_than_no_matching_and_does_not_explode():
    field = _textured().clone()
    field[..., :, 48:] = 2.0  # flat right half: overlaps there carry no slope information
    image, forward = _per_window_distortion(field, [1.0, 1.7, 0.6, 2.3], [0.0, 0.4, -0.3])
    stats = {}
    matched = predict_tiled(image, forward, 48, 24, match="affine", stats=stats)
    image, forward = _per_window_distortion(field, [1.0, 1.7, 0.6, 2.3], [0.0, 0.4, -0.3])
    raw = predict_tiled(image, forward, 48, 24)

    assert torch.isfinite(matched).all()
    assert _global_affine_error(field, matched) <= _global_affine_error(field, raw) + 1e-6
    assert float(matched.abs().max()) < 4 * float(raw.abs().max()), "slopes are clamped: no blow-up"
    assert stats["fallbacks"] > 0 and stats["windows"] == len(tile_grid(96, 96, 48, 24))


def test_stats_account_for_every_window():
    field = _textured()
    image, forward = _per_window_distortion(field, [1.0, 1.5], [0.0])
    stats = {}
    predict_tiled(image, forward, 48, 24, match="affine", stats=stats)
    assert stats["windows"] == len(tile_grid(96, 96, 48, 24))
    assert stats["matched"] + stats["unmatched"] == stats["windows"] - 1, "the first window anchors"
    assert stats["fallbacks"] <= stats["matched"]


def test_unknown_match_mode_is_refused():
    with pytest.raises(ValueError, match="match mode"):
        predict_tiled(torch.rand(1, 3, 64, 64), lambda t: t[:, :1], 32, 8, match="shift")


def test_match_mode_follows_output_space():
    class Model:
        def __init__(self, protocol, output_space="depth"):
            self.protocol, self.output_space = protocol, output_space

    assert match_mode(Model("relative", "disparity")) == "affine"
    assert match_mode(Model("relative", "depth")) == "scale"
    assert match_mode(Model("metric", "depth")) == "none"
    assert match_mode(object()) == "none"
