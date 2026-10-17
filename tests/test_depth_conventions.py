"""What each depth model's output means, and that it is scored and trained accordingly.

These guard the 2026-10-17 audit findings on the real data, each of which produced a
plausible-looking but meaningless number:

  * MiDaS emits disparity `s / z + t`; scoring `1 / disparity` with a scale alone cannot
    remove the shift (silog 41-111 against ~27 for a constant plane).
  * Depth Anything 3 metric with `focal_px` unset returns canonical depth, not metres
    (abs_rel 0.91-0.96, d1 = 0).
  * ZoeD-N produced NaN under float16 autocast (training loss non-finite, no valid score).
  * ZoeD-K's bin head is bounded, and the hub built it with 10 m instead of the 80 m it was
    trained with.

Each test states the mistake it prevents, so removing the guard fails here instead of
quietly changing a table.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch", reason="torch is not installed")

from symtrs.config import ConfigError, DataConfig, RunConfig, focal_px_for_resolution  # noqa: E402
from symtrs.metrics.depth import (  # noqa: E402
    ALIGNMENTS,
    DISPARITY_ALIGNMENTS,
    AllInvalidError,
    align_prediction,
    compute_depth_metrics,
)
from symtrs.model.depth.base import DepthAdapter  # noqa: E402
from symtrs.model.depth.losses import DEPTH_LOSSES, build_depth_loss, ssi_loss  # noqa: E402
from symtrs.registry import REGISTRY, depth_pairing, get_spec  # noqa: E402


def _scene(seed: int = 0, size: int = 24):
    """A nadir-like depth map: 70-160 m, narrow relative spread."""
    generator = torch.Generator().manual_seed(seed)
    depth = 70.0 + 90.0 * torch.rand(1, 1, size, size, generator=generator)
    return depth, torch.ones_like(depth, dtype=torch.bool)


# ----------------------------------------------------------------- pairing (rule 3)


def test_pairing_follows_protocol_and_output_space():
    assert depth_pairing("metric", "depth") == ("berhu", "none")
    assert depth_pairing("relative", "depth") == ("silog", "scale")
    assert depth_pairing("relative", "disparity") == ("ssi", "scale_shift_disparity")
    with pytest.raises(Exception, match="output_space"):
        depth_pairing("relative", "inverse")


def test_registry_and_adapter_classes_agree_on_the_midas_convention():
    """The registry copy exists so `symtrs models` needs no weights; it must not drift from
    the class attributes the adapter actually runs with."""
    for spec in REGISTRY.values():
        if spec.task == "depth" and spec.key.startswith("midas"):
            cls = spec.load_class()
            assert cls.default_align == spec.default_align == "scale_shift_disparity", spec.key
            assert cls.output_space == spec.output_space == "disparity", spec.key


def test_all_alignments_are_known_and_disparity_ones_are_a_subset():
    assert set(DISPARITY_ALIGNMENTS) <= set(ALIGNMENTS)
    for spec in REGISTRY.values():
        if spec.task == "depth":
            assert spec.default_align in ALIGNMENTS, spec.key


def test_ssi_is_a_registered_loss_and_the_default_for_disparity_models():
    from symtrs.model.depth.losses import default_loss_for_protocol

    assert "ssi" in DEPTH_LOSSES
    assert default_loss_for_protocol("relative", "disparity") == "ssi"
    assert default_loss_for_protocol("relative", "depth") == "silog"
    assert default_loss_for_protocol("relative") == "silog"
    assert default_loss_for_protocol("metric") == "berhu"


# ------------------------------------------------- MiDaS: disparity-space alignment


def test_disparity_alignment_recovers_depth_exactly_from_an_affine_disparity():
    """MiDaS output is `s / z + t`. Fitting scale and shift against 1/GT inverts it exactly."""
    depth, mask = _scene()
    raw = 37.0 / depth + 0.004  # unknown scale AND shift
    aligned, fits = align_prediction(raw, depth, mask, mode="scale_shift_disparity")
    assert torch.allclose(aligned, depth, rtol=1e-3)
    assert fits[0].scale == pytest.approx(1.0 / 37.0, rel=1e-3)


def test_the_old_pipeline_could_not_remove_the_shift():
    """The reason this protocol exists: invert first, align by scale, and a shift in the
    disparity leaves a large error that no scale can fix."""
    depth, mask = _scene()
    raw = 37.0 / depth + 0.4  # a shift of the size of the signal, as MiDaS produces
    inverted = 1.0 / raw  # what to_metres gave the old scale alignment
    old, _ = align_prediction(inverted, depth, mask, mode="scale")
    new, _ = align_prediction(raw, depth, mask, mode="scale_shift_disparity")
    old_error = (old - depth).abs().mean()
    new_error = (new - depth).abs().mean()
    assert new_error < 0.05
    assert old_error > 1000 * new_error and old_error > 3.0


def test_disparity_alignment_clamps_the_far_field_instead_of_exploding():
    """A fitted disparity at or below zero would invert to infinity or a negative depth; it is
    clamped to 1/max_depth, the way the MiDaS evaluation caps depth at the dataset ceiling."""
    depth, mask = _scene()
    raw = 1.0 / depth
    raw[..., 0, 0] = -5.0  # a sky-like pixel the fit will map below zero
    aligned, _ = align_prediction(raw, depth, mask, mode="scale_shift_disparity", max_depth=500.0)
    assert torch.isfinite(aligned).all()
    assert aligned.max() <= 500.0 * (1 + 1e-6)
    assert aligned.min() > 0


def test_metrics_with_disparity_alignment_score_an_affine_disparity_as_near_perfect():
    depth, mask = _scene()
    metrics = compute_depth_metrics(37.0 / depth + 0.004, depth, mask, align="scale_shift_disparity")
    assert metrics["abs_rel"] < 1e-3 and metrics["d1"] == 1.0


def test_a_disparity_alignment_on_an_unfittable_image_refuses_instead_of_scoring_disparity_as_metres():
    depth, _ = _scene()
    one_pixel = torch.zeros_like(depth, dtype=torch.bool)
    one_pixel[..., 0, 0] = True
    with pytest.raises(AllInvalidError, match="disparity"):
        compute_depth_metrics(1.0 / depth, depth, one_pixel, align="scale_shift_disparity")


# ------------------------------------------------------------- alignment validation


class _Disparity(DepthAdapter):
    protocol = "relative"
    output_space = "disparity"
    default_align = "scale_shift_disparity"

    def forward(self, image):  # pragma: no cover - never run
        raise NotImplementedError


class _RelDepth(DepthAdapter):
    protocol = "relative"
    output_space = "depth"
    default_align = "scale"

    def forward(self, image):  # pragma: no cover
        raise NotImplementedError


def test_a_disparity_model_refuses_a_depth_space_alignment():
    for wrong in ("scale", "median", "scale_shift", "none"):
        with pytest.raises(ValueError, match="outputs disparity"):
            _Disparity(align=wrong)
    assert _Disparity().align == "scale_shift_disparity"
    assert _Disparity().loss_name == "ssi"


def test_a_depth_model_refuses_the_disparity_alignment():
    with pytest.raises(ValueError, match="outputs depth"):
        _RelDepth(align="scale_shift_disparity")
    assert _RelDepth().loss_name == "silog"


def test_scoring_input_hands_raw_disparity_to_the_disparity_fit_and_metres_to_the_rest():
    model = _Disparity()
    raw = torch.tensor([[[[-1.0, 0.0, 2.0]]]])
    assert torch.equal(model.scoring_input(raw, "scale_shift_disparity"), raw)  # sign kept
    # to_metres zeroes non-positive disparity, which would throw away what the fit needs
    assert model.scoring_input(raw, "scale")[0, 0, 0, 0] == 0.0


def test_disparity_targets_are_inverse_depth_and_depth_targets_are_depth():
    depth, mask = _scene()
    assert torch.allclose(_Disparity().prepare_target(depth, mask).values, 1.0 / depth, rtol=1e-6)
    assert torch.equal(_RelDepth().prepare_target(depth, mask).values, depth)


# ------------------------------------------------------------------------ ssi loss


def test_ssi_ignores_any_scale_and_shift_of_the_prediction():
    depth, mask = _scene()
    target = 1.0 / depth
    prediction = 55.0 * target - 0.3
    assert ssi_loss(prediction, target, mask).item() < 1e-3
    assert ssi_loss(torch.full_like(target, 3.0), target, mask).item() == pytest.approx(1.0, abs=0.05)


def test_ssi_is_dimensionless_so_the_target_scale_does_not_matter():
    """Disparity at 1/70..1/270 m is ~1e-2; an unnormalised loss would be ~1e-6 and Adam's
    epsilon would swamp its gradient."""
    depth, mask = _scene(seed=1)
    noise = torch.randn_like(depth) * 0.05
    a = ssi_loss(1.0 / depth + noise / depth, 1.0 / depth, mask)
    b = ssi_loss(1000.0 * (1.0 / depth + noise / depth), 1000.0 / depth, mask)
    assert a.item() == pytest.approx(b.item(), rel=2e-2)
    assert a.item() > 0.01


def test_ssi_has_a_gradient_and_survives_a_constant_prediction():
    depth, mask = _scene()
    prediction = (1.0 / depth + 0.01 * torch.randn_like(depth)).requires_grad_(True)
    loss = ssi_loss(prediction, 1.0 / depth, mask)
    loss.backward()
    assert torch.isfinite(prediction.grad).all() and prediction.grad.abs().sum() > 0
    flat = torch.ones_like(depth, requires_grad=True)
    ssi_loss(flat, 1.0 / depth, mask).backward()
    assert torch.isfinite(flat.grad).all()
    assert build_depth_loss("ssi") is ssi_loss


# ----------------------------------------------------------- DA3 metric focal length


def test_focal_follows_the_resolution_at_60_degrees():
    """f = (W / 2) / tan(30 deg) = 0.866 * W: 1773.6 px at 2048, halving with each level."""
    assert focal_px_for_resolution(2048, 60.0) == pytest.approx(1773.62, abs=0.01)
    for resolution in (1024, 512, 256):
        assert focal_px_for_resolution(resolution, 60.0) == pytest.approx(
            focal_px_for_resolution(2048, 60.0) * resolution / 2048
        )
    assert focal_px_for_resolution(1000, 90.0) == pytest.approx(500.0)


def _da3_cfg(resolution=1024, **kwargs):
    return RunConfig(task="depth", model="da3_metric_large", resolution=resolution, **kwargs)


def test_build_from_config_derives_focal_px_from_the_resolution():
    from symtrs.model.wrapper import _fill_camera_kwargs

    for resolution in (2048, 1024, 512, 256):
        cfg = _da3_cfg(resolution)
        _fill_camera_kwargs(cfg)
        assert cfg.model_kwargs["focal_px"] == pytest.approx(
            0.5 * resolution / math.tan(math.radians(30.0))
        ), resolution


def test_focal_px_does_not_depend_on_the_training_crop_or_the_tiling():
    """A crop or a tile keeps the pixel size, so the focal length in pixels is unchanged. A
    value tied to the crop would be 4x off at 2048 -> 512."""
    from symtrs.model.wrapper import _fill_camera_kwargs

    plain, cropped = _da3_cfg(2048), _da3_cfg(2048)
    cropped.train.patch_size = 512
    cropped.eval.tile_size = 512
    _fill_camera_kwargs(plain)
    _fill_camera_kwargs(cropped)
    assert plain.model_kwargs["focal_px"] == cropped.model_kwargs["focal_px"]


def test_the_assumed_field_of_view_is_the_one_knob_and_an_explicit_focal_wins():
    from symtrs.model.wrapper import _fill_camera_kwargs

    cfg = _da3_cfg(1024)
    cfg.data.camera_hfov_deg = 90.0
    _fill_camera_kwargs(cfg)
    assert cfg.model_kwargs["focal_px"] == pytest.approx(512.0)

    explicit = _da3_cfg(1024, model_kwargs={"focal_px": 123.0})
    _fill_camera_kwargs(explicit)
    assert explicit.model_kwargs["focal_px"] == 123.0

    with pytest.raises(ConfigError, match="camera_hfov_deg"):
        DataConfig(root="r", camera_hfov_deg=0.0).validate()


def test_only_models_that_need_the_camera_get_a_focal_length():
    from symtrs.model.wrapper import _fill_camera_kwargs

    assert get_spec("da3_metric_large").focal_kwarg == "focal_px"
    for key in ("da3_large", "midas_small", "zoedepth_k", "depth_pro"):
        cfg = RunConfig(task="depth", model=key, resolution=1024)
        _fill_camera_kwargs(cfg)
        assert cfg.model_kwargs == {}, key


def test_da3_metric_scales_canonical_depth_by_focal_over_300():
    """metres = canonical * focal_px / 300; unset focal leaves canonical depth (the mistake)."""
    from symtrs.model.depth.da3 import CANONICAL_FOCAL, DepthAnything3Adapter

    class _Net(torch.nn.Module):
        def forward(self, x):
            return torch.ones(x.shape[0], 1, 4, 4)

    def _adapter(focal):
        adapter = DepthAnything3Adapter.__new__(DepthAnything3Adapter)
        torch.nn.Module.__init__(adapter)
        adapter.model, adapter.protocol, adapter.focal_px = _Net(), "metric", focal
        adapter.model_id, adapter.key = "depth-anything/DA3METRIC-LARGE", "da3_metric_large"
        return adapter

    image = torch.zeros(1, 3, 4, 4)
    assert CANONICAL_FOCAL == 300.0
    assert _adapter(None).forward(image).mean().item() == pytest.approx(1.0)
    assert _adapter(1773.6).forward(image).mean().item() == pytest.approx(1773.6 / 300.0)


# ------------------------------------------------------------------------- ZoeDepth


def test_zoedepth_k_default_range_is_the_one_the_weights_were_trained_with():
    from symtrs.model.depth.zoedepth import ZOED_K_TRAINED_MAX_DEPTH

    assert ZOED_K_TRAINED_MAX_DEPTH == 80.0
    assert get_spec("zoedepth_k").default_kwargs["max_depth"] == ZOED_K_TRAINED_MAX_DEPTH
    assert "max_depth" not in get_spec("zoedepth_nk").default_kwargs


@pytest.mark.parametrize("variant", ["N", "NK"])
def test_a_depth_ceiling_is_refused_where_the_head_is_unbounded(variant):
    """ZoeD-N and ZoeD-NK use softplus bin centres, for which `max_depth` is documented
    'not used' upstream. Accepting the key would suggest a cap that does not exist."""
    from symtrs.model.depth.zoedepth import ZoeDepthAdapter

    with pytest.raises(ValueError, match="no effect"):
        ZoeDepthAdapter(variant=variant, max_depth=300.0)


def test_zoedepth_k_rejects_a_non_positive_ceiling_before_touching_the_network():
    from symtrs.model.depth.zoedepth import ZoeDepthAdapter

    with pytest.raises(ValueError, match="positive"):
        ZoeDepthAdapter(variant="K", max_depth=0.0)


def test_only_zoed_n_is_kept_out_of_float16():
    """Measured: fp16 autocast turned ~10% of ZoeD-N's output into NaN; K and NK were finite.
    bf16 has fp32's range, and fp32 is the fallback on a GPU without it."""
    from symtrs.model.depth import zoedepth

    assert zoedepth._FP16_UNSAFE_VARIANTS == frozenset({"N"})


def test_the_precision_guard_leaves_cpu_and_the_other_variants_alone():
    import contextlib

    from symtrs.model.depth.zoedepth import ZoeDepthAdapter

    adapter = ZoeDepthAdapter.__new__(ZoeDepthAdapter)
    torch.nn.Module.__init__(adapter)
    cpu_image = torch.zeros(1, 3, 8, 8)
    for variant in ("N", "K", "NK"):
        adapter.variant = variant
        assert isinstance(adapter._safe_precision(cpu_image), contextlib.nullcontext)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_zoed_n_runs_in_bfloat16_inside_a_float16_autocast_on_cuda():
    from symtrs.model.depth.zoedepth import ZoeDepthAdapter

    seen = {}

    class _Spy(torch.nn.Module):
        def forward(self, x, **kwargs):
            seen["dtype"] = torch.get_autocast_dtype("cuda")
            return {"metric_depth": torch.ones(x.shape[0], 1, 4, 4, device=x.device)}

    adapter = ZoeDepthAdapter.__new__(ZoeDepthAdapter)
    torch.nn.Module.__init__(adapter)
    adapter.variant, adapter.model, adapter._domain_kwarg, adapter.domain_id = "N", _Spy(), None, 0
    adapter.key = "zoedepth_n"
    image = torch.zeros(1, 3, 8, 8, device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        out = adapter.forward(image)
    if torch.cuda.is_bf16_supported():
        assert seen["dtype"] == torch.bfloat16
    assert out.dtype == torch.float32
