"""The evaluation protocol: what a validation or test number is a number *of*.

Four mistakes were found in smoke runs on the real capture, each of which produced a
plausible-looking table instead of an error:

  * val and test at 2048 were scored on ONE RANDOM 512px crop per image (`val_patch_size: 0`
    fell back to `train.patch_size`), so the 2048 rows were neither reproducible nor
    comparable with the whole-image 1024/512/256 rows, and "best" was picked on noise;
  * a NaN validation metric became `best.pt` and wrote a bare `NaN` into `summary.json`;
  * a class absent from the data reported IoU 0.0 instead of "not measurable";
  * relative-depth tiles were blended although each has its own unknown scale and shift.

Each test asserts the reason, so removing the guard fails here rather than moving a number.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from conftest import requires_numpy, requires_pil, requires_torch

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _no_bare_constant(name):
    pytest.fail(f"bare {name} in what should be valid JSON")


# ----------------------------------------------------------- 1. whole image, not a crop


def _cfg(**train):
    from symtrs.config import RunConfig

    return RunConfig.from_dict(
        {
            "task": "segmentation",
            "model": "unet_small",
            "resolution": 2048,
            "data": {"root": "/x"},
            "train": train,
        }
    ).validate()


@requires_torch
def test_evaluation_never_inherits_the_training_crop():
    """The bug: `val_patch_size or train.patch_size`. 0 has to mean the whole image."""
    from symtrs.data import factory

    seen = {}

    def spy(**kwargs):
        seen[kwargs["train"]] = kwargs["patch_size"]
        return object()

    original = factory.dataloader
    factory.dataloader = spy
    try:
        cfg = _cfg(patch_size=512, val_patch_size=0)
        factory.from_config(cfg, train=True)
        factory.from_config(cfg, train=False)
        factory.from_config(cfg, split="test", train=False)
    finally:
        factory.dataloader = original

    assert seen[True] == 512, "training still crops"
    assert seen[False] == 0, "val/test must be the whole image, not the training patch"


@requires_torch
def test_a_positive_val_patch_size_is_honoured_for_evaluation():
    from symtrs.data import factory

    seen = {}
    original = factory.dataloader
    factory.dataloader = lambda **kw: seen.setdefault(kw["train"], kw["patch_size"])
    try:
        factory.from_config(_cfg(patch_size=512, val_patch_size=256), train=False)
    finally:
        factory.dataloader = original
    assert seen[False] == 256


def test_the_config_comment_that_promised_the_whole_image_is_now_true():
    """Every generated 2048 config says `val_patch_size: 0` and trains on a smaller crop."""
    text = (CONFIGS / "segmentation" / "2048" / "unet_small_2048.yaml").read_text(encoding="utf-8")
    assert "patch_size: 512" in text and "val_patch_size: 0" in text
    assert "tile_size: 512" in text, "the whole image is evaluated in training-sized windows"


def _write_pair_dataset(root: Path, size: int = 64, count: int = 3):
    import numpy as np

    from symtrs.commands.make_dummy import _write_split

    _write_split(root, "val", count, size, "npy", False, np.random.default_rng(0))


@requires_torch
@requires_numpy
@requires_pil
@pytest.mark.parametrize("task", ["segmentation", "depth"])
def test_evaluation_crops_are_deterministic_centre_crops_and_training_crops_are_random(
    tmp_path: Path, task
):
    """Same pixels on every pass (so two evaluations of a checkpoint are comparable);
    training keeps its random crop (that is what augmentation is)."""
    import torch

    from symtrs.data.factory import build_dataset

    root = tmp_path / "d"
    _write_pair_dataset(root)

    kwargs = dict(task=task, root=str(root), split="val", resolution=2048, patch_size=32,
                  max_depth=1000.0 if task == "depth" else None)
    evaluation = build_dataset(train=False, **kwargs)
    first, second = evaluation[0]["image"], evaluation[0]["image"]
    assert first.shape[-2:] == (32, 32)
    assert torch.equal(first, second), "an evaluation sample must not change between reads"

    full = build_dataset(train=False, **{**kwargs, "patch_size": 0})[0]["image"]
    assert full.shape[-2:] == (64, 64), "patch_size 0 is the whole image"
    assert torch.equal(first, full[..., 16:48, 16:48]), "the crop is the centre one"

    training = build_dataset(train=True, **kwargs)
    crops = {tuple(training[0]["image"].flatten()[:8].tolist()) for _ in range(40)}
    assert len(crops) > 1, "training crops must stay random"


@requires_torch
def test_val_and_test_share_one_evaluation_function_and_one_tiling_setting():
    """`evaluate` is one function for both splits and reads only `cfg.eval`; this pins that
    there is no second code path where val could be tiled and test not."""
    import inspect

    from symtrs.engine import loop, trainer

    source = inspect.getsource(trainer._run)
    assert "evaluate(model, val_loader, cfg, resolved)" in source
    assert "evaluate(\n        model, test_loader, cfg, resolved" in source.replace("\r\n", "\n")
    assert inspect.getsource(loop.evaluate_depth).count("cfg.eval.tile_size") == 1
    assert inspect.getsource(loop.evaluate_segmentation).count("cfg.eval.tile_size") == 1


# -------------------------------------------------------- tile policy in the generator


def _generator():
    import importlib.util

    spec = importlib.util.spec_from_file_location("generate_grid", CONFIGS / "generate_grid.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tile_policy_is_the_training_patch_when_the_row_crops_else_whole_image():
    grid = _generator()
    assert grid.eval_tiling(512, 2048) == (512, 128)
    assert grid.eval_tiling(384, 2048) == (384, 96)
    assert grid.eval_tiling(0, 1024) == (0, 128), "trained on whole images -> scored whole"
    assert grid.eval_tiling(512, 512) == (0, 128), "a crop as large as the image is no crop"


def test_every_generated_config_tiles_exactly_when_it_trains_on_crops():
    yaml = pytest.importorskip("yaml")

    paths = sorted(CONFIGS.glob("depth/*/*.yaml")) + sorted(CONFIGS.glob("segmentation/*/*.yaml"))
    assert paths
    for path in paths:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        patch, resolution = raw["train"]["patch_size"], raw["resolution"]
        assert raw["train"]["val_patch_size"] == 0, path
        tile = raw["eval"]["tile_size"]
        if patch and patch < resolution:
            assert tile == patch, f"{path}: evaluation windows must be the training patch"
            assert raw["eval"]["tile_overlap"] < tile
        else:
            assert tile == 0, path


# ----------------------------------------------- tiling vs relative depth / alignment


@requires_torch
def test_relative_tiles_are_matched_before_blending_so_the_field_is_one_surface():
    """Each forward pass of a relative model has its own scale and shift. Stitching the raw
    outputs leaves steps at every seam; fitting each window onto what is already stitched
    recovers the field up to ONE global affine map - which is all the per-image alignment
    can undo."""
    import torch

    from symtrs.engine.tiling import predict_tiled

    ramp = torch.linspace(0.5, 2.0, 96).view(1, 1, 96, 1) * torch.linspace(1.0, 1.5, 96).view(1, 1, 1, 96)
    image = ramp.expand(1, 3, 96, 96).contiguous()
    calls = {"n": 0}

    def forward(tile):
        calls["n"] += 1
        scale, shift = 0.5 + 0.3 * (calls["n"] % 4), 0.2 * (calls["n"] % 3)
        return tile[:, :1] * scale + shift  # every window has its own affine ambiguity

    def error(stitched):
        x = ramp.flatten()
        design = torch.stack([x, torch.ones_like(x)], dim=1)
        coeffs = torch.linalg.lstsq(design, stitched.flatten().unsqueeze(1)).solution
        return float((design @ coeffs - stitched.flatten().unsqueeze(1)).abs().max())

    raw = predict_tiled(image, forward, 48, 16, match_affine=False)
    calls["n"] = 0
    matched = predict_tiled(image, forward, 48, 16, match_affine=True)
    assert error(matched) < 0.05
    assert error(raw) > 5 * error(matched), "without matching the windows disagree at the seams"


@requires_torch
def test_absolute_outputs_are_not_matched_by_default():
    """Metric depth and logits are absolute: a default call must reproduce them exactly."""
    import torch

    from symtrs.engine.tiling import predict_tiled

    image = torch.rand(1, 3, 80, 80)
    stitched = predict_tiled(image, lambda tile: tile[:, :1] * 3.0, 40, 10)
    assert torch.allclose(stitched, image[:, :1] * 3.0, atol=1e-5)


@requires_torch
def test_forward_native_matches_tiles_by_output_space_not_by_protocol_alone():
    """Disparity (MiDaS) is ambiguous in scale AND shift; depth up to a scale (ZoeD-N, DA3
    relative) in scale only - a fitted shift there is an offset the final `scale` alignment cannot
    remove; metric depth and logits are absolute and must never be matched."""
    import torch

    import symtrs.engine.tiling as tiling
    from symtrs.engine import loop

    seen = {}
    current = {}

    class Stub:
        def __init__(self, protocol, output_space=None):
            self.protocol = protocol
            if output_space is not None:
                self.output_space = output_space

        def preprocess_image(self, image):
            return image

        def forward(self, image):
            return image[:, :1]

    original = tiling.predict_tiled

    def spy(image, fn, size, overlap, **kwargs):
        seen[current["case"]] = kwargs["match"]
        return original(image, fn, size, overlap, **kwargs)

    tiling.predict_tiled = spy
    cases = {
        "midas": Stub("relative", "disparity"),
        "depth_up_to_scale": Stub("relative", "depth"),
        "metric": Stub("metric", "depth"),
        "segmentation": Stub(""),
    }
    try:
        for name, model in cases.items():
            current["case"] = name
            loop.forward_native(model, torch.rand(1, 3, 64, 64), tile_size=32, tile_overlap=8)
    finally:
        tiling.predict_tiled = original
    assert seen == {
        "midas": "affine", "depth_up_to_scale": "scale", "metric": "none", "segmentation": "none",
    }


@requires_torch
def test_real_adapters_expose_the_output_space_match_mode_reads():
    """`match_mode` reads `protocol` / `output_space` off the adapter. If an adapter stopped
    exposing them, a disparity model would silently be matched with a shift-free fit."""
    from symtrs.engine.tiling import match_mode
    from symtrs.registry import REGISTRY

    for key, spec in REGISTRY.items():
        if spec.task != "depth":
            continue

        class FromSpec:
            protocol = spec.protocol
            output_space = spec.output_space

        expected = "none" if spec.protocol != "relative" else (
            "affine" if spec.output_space == "disparity" else "scale"
        )
        assert match_mode(FromSpec) == expected, key


@requires_torch
def test_padding_is_cropped_not_stretched_when_the_model_pads_to_a_multiple():
    """DA3 pads a 512px tile to 518. Resizing the 518px answer back to 512 shifts every
    pixel against the ground truth; the pad has to be cut off."""
    import torch

    from symtrs.engine import loop

    class Padded:
        protocol = "metric"
        size_multiple = 14

        def preprocess_image(self, image):
            return image

        def _pad_to_multiple(self, image):
            return torch.nn.functional.pad(image, (0, 6, 0, 6), mode="replicate")

        def forward(self, image):
            return image[:, :1]

    image = torch.rand(1, 3, 512, 512)
    out = loop.forward_native(Padded(), image)
    assert out.shape[-2:] == (512, 512)
    assert torch.equal(out, image[:, :1]), "a stretch would have changed every pixel"


# -------------------------------------------------------------- 2. NaN is never "best"


def test_a_non_finite_value_is_never_an_improvement_not_even_the_first():
    pytest.importorskip("torch")
    from symtrs.engine.trainer import is_improvement

    for bad in (float("nan"), float("inf"), float("-inf")):
        assert not is_improvement(bad, None, "min"), "the first epoch used to win by default"
        assert not is_improvement(bad, 1.0, "min")
        assert not is_improvement(bad, 1.0, "max")
    assert is_improvement(2.0, None, "min")
    assert is_improvement(0.5, 1.0, "min") and not is_improvement(1.5, 1.0, "min")
    assert is_improvement(0.6, 0.5, "max")


def test_json_with_non_finite_floats_is_valid_json_and_null():
    from symtrs.common.jsonio import dumps

    summary = {"best": float("nan"), "test": {"rmse": float("inf"), "d1": 0.5}, "h": [{"x": float("-inf")}]}
    text = dumps(summary)
    assert "NaN" not in text and "Infinity" not in text
    parsed = json.loads(text, parse_constant=_no_bare_constant)
    assert parsed == {"best": None, "test": {"rmse": None, "d1": 0.5}, "h": [{"x": None}]}


def test_write_json_refuses_to_emit_a_bare_nan(tmp_path: Path):
    from symtrs.common.jsonio import write_json

    write_json(tmp_path / "summary.json", {"a": float("nan")})
    text = (tmp_path / "summary.json").read_text(encoding="utf-8")
    assert json.loads(text, parse_constant=_no_bare_constant) == {"a": None}


def test_csv_rows_write_non_finite_cells_empty(tmp_path: Path):
    from symtrs.common.table import append_csv_row, read_csv_rows, row_as_floats

    path = str(tmp_path / "metrics.csv")
    append_csv_row(path, {"epoch": 0, "val_rmse": float("nan"), "val_d1": 0.25})
    row = row_as_floats(read_csv_rows(path)[0])
    assert row["val_rmse"] == "", "a missing metric must not read back as a number"
    assert row["val_d1"] == 0.25


def test_tables_cope_with_null_metrics(tmp_path: Path):
    from symtrs.commands.tables import _format, collect, render

    run = tmp_path / "runs" / "depth" / "m" / "2048"
    run.mkdir(parents=True)
    # Older runs wrote a bare NaN; new ones write null. Both must load and print as n/a.
    (run / "summary.json").write_text(
        '{"task": "depth", "model": "m", "resolution": 2048, "mode": "finetune", '
        '"best": null, "best_epoch": -1, "test": {"rmse": null, "abs_rel": NaN, "d1": 0.5}}',
        encoding="utf-8",
    )
    rows = collect(str(tmp_path / "runs"))
    assert len(rows) == 1
    table = render(rows, ["model", "rmse", "abs_rel", "d1"], "markdown")
    assert "n/a" in table and "None" not in table and "nan" not in table
    assert _format(None) == "n/a" and _format(float("nan")) == "n/a" and _format(0.5) == "0.5000"
    json.loads(render(rows, ["model", "abs_rel"], "json"), parse_constant=_no_bare_constant)


# -------------------------------------------------------- 3. mIoU and empty classes


@requires_torch
def test_a_class_with_empty_union_is_null_and_excluded_from_miou():
    import torch

    from symtrs.metrics.semantic import ConfusionMatrix

    matrix = ConfusionMatrix(num_classes=5, ignore_index=255)
    target = torch.tensor([[1, 1, 2, 2, 3, 3, 4, 255]])
    matrix.update(target.clone(), target)  # perfect prediction; class 0 exists nowhere

    summary = matrix.summary()
    assert math.isnan(summary["iou_0"]) and math.isnan(summary["dice_0"]), (
        "class 0 has no pixel and no prediction: not measurable, which is not IoU 0"
    )
    assert summary["miou"] == pytest.approx(1.0), "the empty class must not drag the mean down"
    assert summary["miou_classes"] == 4


@requires_torch
def test_a_class_predicted_but_absent_from_the_ground_truth_is_reported_but_not_averaged():
    """The mIoU denominator is a property of the data (classes with ground truth in the split),
    not of the model. Otherwise one stray class-0 pixel moved `miou_classes` from 4 to 5 and cut
    the mIoU by 20% relative, which made the models of one table incomparable."""
    import torch

    from symtrs.metrics.semantic import ConfusionMatrix

    matrix = ConfusionMatrix(num_classes=3, ignore_index=255)
    matrix.update(torch.tensor([[0, 1, 1, 1]]), torch.tensor([[1, 1, 1, 1]]))  # predicts 0 once

    summary = matrix.summary()
    assert summary["iou_0"] == 0.0, "union > 0: the hallucination is visible per class..."
    assert summary["fp_pixels_0"] == 1.0 and summary["support_0"] == 0.0
    assert math.isnan(summary["iou_2"]), "class 2 is nowhere"
    assert summary["miou_classes"] == 1, "...but class 0 has no ground truth, so it is not averaged"
    assert summary["miou"] == pytest.approx(3 / 4)
    assert summary["mean_dice"] == pytest.approx(2 * 3 / (4 + 3))


@requires_torch
def test_miou_denominator_is_identical_for_every_model_on_the_same_split():
    import torch

    from symtrs.metrics.semantic import ConfusionMatrix

    target = torch.tensor([[1, 1, 2, 2, 3, 3, 4, 4]])  # SyMTRS-like: class 0 exists nowhere
    perfect = ConfusionMatrix(num_classes=5, ignore_index=255)
    perfect.update(target.clone(), target)
    stray = ConfusionMatrix(num_classes=5, ignore_index=255)
    prediction = target.clone()
    prediction[0, 0] = 0  # one pixel of "background"
    stray.update(prediction, target)

    assert perfect.summary()["miou_classes"] == stray.summary()["miou_classes"] == 4
    # Class 1 lost one of two pixels (IoU 1/2); the other three classes are perfect. Dividing by 5
    # (the old rule) would give 0.7 instead of 0.875 for the same single mistake.
    assert stray.summary()["miou"] == pytest.approx((0.5 + 1.0 + 1.0 + 1.0) / 4)
    assert stray.summary()["fp_pixels_0"] == 1.0


@requires_torch
def test_null_per_class_iou_survives_into_summary_json():
    import torch

    from symtrs.common.jsonio import dumps
    from symtrs.metrics.semantic import ConfusionMatrix

    matrix = ConfusionMatrix(num_classes=3, ignore_index=255)
    matrix.update(torch.tensor([[1, 1]]), torch.tensor([[1, 1]]))
    parsed = json.loads(dumps({"test": matrix.summary()}), parse_constant=_no_bare_constant)
    assert parsed["test"]["iou_0"] is None and parsed["test"]["iou_1"] == 1.0
