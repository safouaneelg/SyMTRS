"""Regression tests for the three adapters that could not be built offline.

Each test asserts the reason for the change, so removing the protection fails here and not
as a download-time surprise: SegFormer's encoder source, the Depth Anything 3 loading path,
and the MiDaS Swin-V2 timm pin and transform pairing.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from symtrs.registry import get_spec

from conftest import requires_torch

ROOT = Path(__file__).resolve().parents[1]
HAVE_TRANSFORMERS = importlib.util.find_spec("transformers") is not None


def _extras():
    try:
        import tomllib
    except ImportError:  # pragma: no cover - python < 3.11
        pytest.skip("tomllib needs python >= 3.11")
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return data["project"]["optional-dependencies"]


@pytest.mark.parametrize("key", ["segformer_b0", "segformer_b1", "segformer_b2"])
def test_segformer_requires_transformers_not_timm(key):
    # timm has never shipped MiT, so listing it as the requirement made `doctor` say
    # "dependencies ok" for a model that could not be built.
    requires = get_spec(key).requires
    assert "transformers" in requires
    assert "timm" not in requires


@pytest.mark.parametrize("key", ["da3_large", "da3_metric_large"])
def test_da3_requires_the_package_not_transformers(key):
    # The DA3 Hub checkpoints are depth_anything_3 object graphs; transformers' Auto*
    # classes cannot load them.
    requires = get_spec(key).requires
    assert "depth_anything_3" in requires
    assert "transformers" not in requires


def test_extras_match_the_new_loading_paths():
    extras = _extras()
    assert extras["segformer"] == ["transformers"]
    # depth-anything-3 must not be an extra: its wheel pins numpy<2 and 3D-export packages.
    assert not any("depth-anything-3" in dep for dep in extras["da3"])
    assert {"omegaconf", "addict", "safetensors"} <= set(extras["da3"])


@requires_torch
@pytest.mark.skipif(not HAVE_TRANSFORMERS, reason="transformers is not installed")
@pytest.mark.parametrize(
    "variant,channels", [("b0", [32, 64, 160, 256]), ("b1", [64, 128, 320, 512])]
)
def test_segformer_builds_from_config_without_download(variant, channels, monkeypatch):
    import torch

    # pretrained=False must not touch the Hub: forbid it, so a regression is a failure
    # rather than a silent download.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    from symtrs.model.semantic.segformer import SegFormerSegmentation

    model = SegFormerSegmentation(variant=variant, pretrained=False).eval()
    with torch.no_grad():
        logits = model(torch.zeros(1, 3, 64, 96))
    assert logits.shape == (1, 6, 64, 96)
    assert [proj.in_features for proj in model.head.projections] == channels


@requires_torch
def test_da3_rejects_random_init_and_names_the_install_when_package_missing(monkeypatch):
    from symtrs.model.depth.da3 import DepthAnything3Adapter

    with pytest.raises(ValueError, match="pretrained=False"):
        DepthAnything3Adapter(pretrained=False)

    # `sys.modules[name] = None` makes `import name` raise ImportError.
    monkeypatch.setitem(sys.modules, "depth_anything_3", None)
    monkeypatch.setitem(sys.modules, "depth_anything_3.cfg", None)
    with pytest.raises(ImportError, match="--no-deps"):
        DepthAnything3Adapter()


@requires_torch
def test_da3_tolerates_only_the_unused_aux_head_keys():
    from symtrs.model.depth.da3 import _TOLERATED_MISSING

    assert _TOLERATED_MISSING.match("head.scratch.output_conv2_aux.1.2.weight")
    assert _TOLERATED_MISSING.match("head.scratch.output_conv2_aux.3.2.bias")
    # the aux branch that feeds the pose/ray output, and any backbone key, are real misses
    assert not _TOLERATED_MISSING.match("head.scratch.output_conv2_aux.0.2.weight")
    assert not _TOLERATED_MISSING.match("backbone.pretrained.blocks.0.attn.qkv.weight")


@requires_torch
def test_midas_swin2_tiny_uses_the_256_transform():
    # dpt_transform resizes to 384, and the 256-pixel checkpoint then asserts
    # "Input image height (384) doesn't match model (256)" in its first forward.
    from symtrs.model.depth.midas import _TRANSFORM_NAMES

    assert _TRANSFORM_NAMES["DPT_SwinV2_T_256"] == "swin256_transform"
    assert _TRANSFORM_NAMES["DPT_SwinV2_L_384"] == "swin384_transform"


def test_swin2_tiny_is_listed_with_the_timm_pinned_models():
    from symtrs.commands.doctor import TIMM_PINNED_KEYS

    assert "midas_dpt_swin2_tiny" in TIMM_PINNED_KEYS
    for key in TIMM_PINNED_KEYS:
        get_spec(key)  # every listed key exists in the registry


def test_beit_large_is_listed_with_the_timm_pinned_models_and_the_smoke_runner_agrees():
    # MiDaS's backbones/beit.py patches timm's Block.forward with a copy that calls
    # `self.drop_path`; timm >= 0.9 removed it, so the model builds and then dies at the
    # first forward. The doctor warning and the venv that scripts/smoke_all.py picks must
    # name the same models, or a queue sends it to the environment where it cannot run.
    from symtrs.commands.doctor import TIMM_PINNED_KEYS

    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import smoke_all
    finally:
        sys.path.remove(str(ROOT / "scripts"))

    assert "midas_dpt_beit_large" in TIMM_PINNED_KEYS
    assert set(smoke_all.ZOE_VENV_MODELS) == set(TIMM_PINNED_KEYS)


@requires_torch
def test_midas_beit_384_uses_the_384_transform():
    from symtrs.model.depth.midas import _TRANSFORM_NAMES, MIDAS_MODELS

    assert get_spec("midas_dpt_beit_large").default_kwargs["hub_model"] == "DPT_BEiT_L_384"
    assert MIDAS_MODELS["DPT_BEiT_L_384"] == 384
    assert _TRANSFORM_NAMES["DPT_BEiT_L_384"] == "dpt_transform"  # Resize(384, 384)


@requires_torch
def test_midas_beit_on_a_too_new_timm_is_refused_at_build_time_naming_the_venv():
    # A block without `drop_path` is what timm >= 0.9 builds; the error has to surface at
    # construction (before the GPU is reserved) and say where the model does run.
    import types

    from symtrs.model.depth.midas import MiDaSAdapter

    new_timm_block = types.SimpleNamespace()  # no drop_path attribute
    backbone = types.SimpleNamespace(
        pretrained=types.SimpleNamespace(model=types.SimpleNamespace(blocks=[new_timm_block]))
    )
    with pytest.raises(RuntimeError, match=r"\.venv-zoe.*timm==0\.6\.12"):
        MiDaSAdapter._require_beit_compatible_timm("DPT_BEiT_L_384", backbone)

    # the old-timm layout, and every non-BEiT model, pass through untouched
    old_timm_block = types.SimpleNamespace(drop_path=object())
    old = types.SimpleNamespace(
        pretrained=types.SimpleNamespace(model=types.SimpleNamespace(blocks=[old_timm_block]))
    )
    MiDaSAdapter._require_beit_compatible_timm("DPT_BEiT_L_384", old)
    MiDaSAdapter._require_beit_compatible_timm("DPT_Hybrid", backbone)
