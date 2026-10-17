"""Shadow removal: config grid, guards, mask derivation, metrics, staging.

Torch-free by design (the worker is exercised by the smoke runs, not here): these tests
protect the decisions that make a shadow-removal number mean something.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from conftest import REPO_ROOT, requires_numpy, requires_pil, requires_torch

CONFIGS = sorted((REPO_ROOT / "configs" / "shadow_removal").glob("*.yaml"))
KEYS = ("shadowformer", "restormer", "uformer", "shadowmaskformer")


def test_one_config_per_model():
    assert sorted(p.stem for p in CONFIGS) == sorted(KEYS)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_configs_load_native_resolution(path):
    from symtrs.config import RunConfig

    cfg = RunConfig.from_yaml(str(path))
    assert cfg.task == "shadow_removal"
    assert cfg.resolution == 2048  # rgb_noshadow has no downsampled twin
    assert cfg.train.patch_size and cfg.eval.tile_size


def test_registry_has_the_four_models():
    from symtrs.registry import get_spec

    for key in KEYS:
        assert get_spec(key).task == "shadow_removal"


def test_archs_match_upstream_signatures():
    """The recorded constructor kwargs must exist in the pinned upstream source."""
    from symtrs.model.shadow.archs import ARCHS, upstream_missing

    for key, arch in ARCHS.items():
        if upstream_missing(key):
            pytest.skip("third_party checkouts absent (scripts/fetch_third_party.py)")
        tree = ast.parse(arch.source_path.read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == arch.cls)
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        names = {a.arg for a in init.args.args + init.args.kwonlyargs}
        for name in arch.kwargs:
            assert name in names, f"{arch.cls}.__init__ has no {name!r}"
        assert ("img_size" in names) == arch.takes_img_size


def test_unknown_model_message_names_the_fix():
    from symtrs.model.shadow.archs import arch_for

    with pytest.raises(ValueError, match="symtrs models --task shadow_removal"):
        arch_for("midas_small")


def _cfg(**over):
    from symtrs.config import RunConfig

    cfg = RunConfig.from_yaml(str(REPO_ROOT / "configs/shadow_removal/restormer.yaml"))
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def test_guards():
    from symtrs.model.shadow.runner import check_config

    check_config(_cfg(), "finetune")
    with pytest.raises(ValueError, match="2048-only"):
        check_config(_cfg(resolution=1024), "finetune")
    with pytest.raises(ValueError, match="zero-shot does not apply"):
        check_config(_cfg(), "zero_shot")
    cfg = _cfg()
    cfg.train.patch_size = 0
    with pytest.raises(ValueError, match="train.patch_size"):
        check_config(cfg, "finetune")
    cfg = _cfg()
    cfg.eval.tile_size = 0
    with pytest.raises(ValueError, match="eval.tile_size"):
        check_config(cfg, "score")


def test_worker_command_shape():
    from symtrs.model.shadow.runner import worker_command

    cmd = worker_command("c.yaml", "predict", "val,test", "w.pt")
    assert cmd[1:3] == ["-m", "symtrs.model.shadow.worker"]
    assert cmd[-4:] == ["--splits", "val,test", "--weights", "w.pt"]


@requires_numpy
def test_derive_mask_finds_the_darkened_region_and_drops_specks():
    import numpy as np

    from symtrs.metrics.shadow import derive_mask

    clean = np.full((64, 64, 3), 0.8, np.float32)
    shadowed = clean.copy()
    shadowed[10:40, 10:40] *= 0.4
    shadowed[55, 55] *= 0.4  # a one-pixel speck: the opening removes it
    mask = derive_mask(shadowed, clean, 0.1, 3)
    assert mask[20, 20] and not mask[55, 55] and not mask[50, 5]
    assert mask.sum() == pytest.approx(30 * 30, rel=0.1)


@requires_numpy
def test_score_image_perfect_and_regions():
    import numpy as np

    from symtrs.metrics.shadow import aggregate, score_image

    rng = np.random.default_rng(0)
    target = rng.random((48, 48, 3)).astype(np.float32)
    shadow = np.zeros((48, 48), bool)
    shadow[:24] = True
    perfect = score_image(target.copy(), target, shadow)
    assert perfect["psnr"] > 100 and perfect["ssim_y"] == pytest.approx(1.0, abs=1e-4)
    assert perfect["rmse_lab"] == pytest.approx(0.0, abs=1e-4)

    pred = target.copy()
    pred[:24] = np.clip(pred[:24] + 0.2, 0, 1)  # error only inside the shadow
    row = score_image(pred, target, shadow)
    assert row["psnr_shadow"] < row["psnr"] < row["psnr_nonshadow"]

    empty = score_image(pred, target, np.zeros((48, 48), bool))
    assert empty["psnr_shadow"] is None  # no shadow pixels: absent, not zero
    assert "psnr" in aggregate([row, empty])


@requires_numpy
@requires_pil
def test_stage_split_pairs_masks_and_idempotence(tmp_path):
    import json

    import numpy as np
    from PIL import Image

    from symtrs.model.shadow.staging import SUBDIRS, stage_split, staged_keys

    root = tmp_path / "SyMTRS"
    (root / "train" / "rgb_day").mkdir(parents=True)
    (root / "train" / "rgb_noshadow").mkdir(parents=True)
    for i in range(2):
        clean = np.full((32, 32, 4), 200, np.uint8)
        dark = clean.copy()
        dark[8:24, 8:24, :3] = 70
        Image.fromarray(dark).save(root / "train" / "rgb_day" / f"rgb_day_2048_loc{i:03d}.png")
        Image.fromarray(clean).save(
            root / "train" / "rgb_noshadow" / f"rgb_noshadow_2048_loc{i:03d}.png"
        )
    run = tmp_path / "run"
    staging = stage_split(str(root), str(run), "train", 0.1, 3)
    assert all((staging / s).is_dir() for s in SUBDIRS)
    assert staged_keys(staging) == ["loc000", "loc001"]
    mask = np.asarray(Image.open(staging / "mask" / "loc000.png"))
    assert mask[16, 16] == 255 and mask[2, 2] == 0
    first = json.loads((staging / "mask_params.json").read_text())
    stage_split(str(root), str(run), "train", 0.1, 3)  # idempotent
    assert json.loads((staging / "mask_params.json").read_text()) == first


# ------------------------------------------------------------ ShadowMaskFormer


def test_shadowmaskformer_is_declared_as_a_from_scratch_signed_range_mask_model():
    from symtrs.model.shadow.archs import arch_for
    from symtrs.model.shadow.runner import describe

    arch = arch_for("shadowmaskformer")
    assert arch.takes_mask and not arch.takes_img_size
    assert arch.input_range == "signed"
    assert arch.released_weights is False
    # the other three keep the [0, 1] contract they always had
    for key in ("shadowformer", "restormer", "uformer"):
        assert arch_for(key).input_range == "unit" and arch_for(key).released_weights
    text = describe("shadowmaskformer")
    assert "no released weights" in text and "trained from scratch" in text
    assert "like-for-like" in text


def test_shadowmaskformer_config_says_it_is_trained_from_scratch():
    from symtrs.config import RunConfig

    path = REPO_ROOT / "configs" / "shadow_removal" / "shadowmaskformer.yaml"
    comment_lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.startswith("#")]
    header = " ".join(l.lstrip("# ") for l in comment_lines)
    assert "NO RELEASED WEIGHTS" in header and "not like-for-like" in header
    cfg = RunConfig.from_yaml(str(path))
    assert cfg.model == "shadowmaskformer"
    assert cfg.shadow.init_weights == ""  # random init is allowed and is the only option


def test_shadowmaskformer_still_refuses_zero_shot_and_allows_empty_init_weights():
    from symtrs.config import RunConfig
    from symtrs.model.shadow.runner import check_config

    cfg = RunConfig.from_yaml(str(REPO_ROOT / "configs/shadow_removal/shadowmaskformer.yaml"))
    assert cfg.shadow.init_weights == ""
    check_config(cfg, "finetune")  # no init_weights is not an error
    with pytest.raises(ValueError, match="zero-shot does not apply"):
        check_config(cfg, "zero_shot")


@requires_numpy
def test_signed_range_round_trip_and_mask_polarity():
    """[0, 1] image / {0, 1} mask go in as [-1, 1] with shadow = +1, and come back to [0, 1]."""
    import numpy as np

    from symtrs.model.shadow.archs import arch_for, from_network_range, to_network_range

    arch = arch_for("shadowmaskformer")
    image = np.array([0.0, 0.25, 0.5, 1.0])
    mask = np.array([0.0, 1.0, 1.0, 0.0])
    net_image, net_mask = to_network_range(arch, image, mask)
    assert net_image.tolist() == [-1.0, -0.5, 0.0, 1.0]
    assert net_mask.tolist() == [-1.0, 1.0, 1.0, -1.0]  # shadow (1) -> +1
    assert np.allclose(from_network_range(arch, net_image), image)
    # an unclamped network output outside [-1, 1] stays outside [0, 1]: the caller clamps
    assert from_network_range(arch, np.array([-1.5, 1.5])).tolist() == [-0.25, 1.25]

    unit = arch_for("shadowformer")
    same_image, same_mask = to_network_range(unit, image, mask)
    assert same_image is image and same_mask is mask
    assert from_network_range(unit, image) is image


@requires_torch
def test_worker_forward_hands_the_network_signed_inputs_and_returns_unit_range():
    import torch

    from symtrs.model.shadow.archs import arch_for
    from symtrs.model.shadow.worker import forward

    seen = {}

    class Probe(torch.nn.Module):
        def forward(self, x, m):
            seen["x"], seen["m"] = x, m
            return x * 0.0 + 1.0  # a network that outputs +1 = white

    image = torch.rand(1, 3, 8, 8)
    mask = (torch.rand(1, 1, 8, 8) > 0.5).float()
    out = forward(Probe(), arch_for("shadowmaskformer"), image, mask)
    assert torch.allclose(seen["x"], image * 2 - 1) and torch.equal(seen["m"], mask * 2 - 1)
    assert float(seen["m"].min()) == -1.0 and float(seen["m"].max()) == 1.0
    assert torch.allclose(out, torch.ones_like(out))  # +1 -> 1.0 white

    class UnitProbe(torch.nn.Module):
        def forward(self, x, m):
            seen["x"], seen["m"] = x, m
            return x

    forward(UnitProbe(), arch_for("shadowformer"), image, mask)
    assert torch.equal(seen["x"], image) and torch.equal(seen["m"], mask)


@requires_torch
def test_shadowmaskformer_builds_with_about_2_3_million_parameters_and_runs_on_cpu():
    import importlib.util
    import warnings

    import torch

    from symtrs.model.shadow.archs import arch_for, upstream_missing
    from symtrs.model.shadow.worker import forward

    if upstream_missing("shadowmaskformer"):
        pytest.skip("third_party checkouts absent (scripts/fetch_third_party.py)")
    arch = arch_for("shadowmaskformer")
    spec = importlib.util.spec_from_file_location("symtrs_test_smf", arch.source_path)
    module = importlib.util.module_from_spec(spec)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # timm.models.layers FutureWarning
        spec.loader.exec_module(module)
    net = getattr(module, arch.cls)(**arch.kwargs)
    count = sum(p.numel() for p in net.parameters())
    assert 2.2e6 < count < 2.4e6, count  # 2.28 M is what the paper reports for -b
    net.eval()
    image = torch.rand(1, 3, 36, 38)  # not a multiple of 4: the network pads and crops back
    mask = (torch.rand(1, 1, 36, 38) > 0.5).float()
    with torch.no_grad():
        out = forward(net, arch, image, mask)
    assert out.shape == image.shape and torch.isfinite(out).all()
