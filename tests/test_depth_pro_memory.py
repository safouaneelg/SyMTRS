"""Depth Pro only fits a 12GB card because `train.freeze: encoder` actually freezes.

The adapters stored the freeze policy but nothing on the training path applied it, so the
run printed "freeze='encoder' -> 952.0M trainable" and kept an autograd graph through the
35 ViT-L passes of the encoder (out of memory above 11GiB at batch 1; with the encoder
frozen the same step peaks at ~7.9GiB, measured on an RTX 5070 Ti Laptop). These tests pin
the pieces that keep that from regressing without needing the 1.9GB checkpoint or a GPU.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import yaml

from tests.conftest import requires_torch

REPO_ROOT = Path(__file__).resolve().parents[1]


def _depth_pro_like():
    """The module-name layout of the adapter around `depth_pro.DepthPro`, without weights."""
    import torch.nn as nn

    def tiny():
        return nn.Linear(2, 2)

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_encoder = tiny()
            self.image_encoder = tiny()
            self.upsample0 = tiny()
            self.fuse_lowres = tiny()

    class Fov(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(tiny(), tiny())
            self.downsample = tiny()
            self.head = tiny()

    class DepthPro(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = Encoder()
            self.decoder = tiny()
            self.head = tiny()
            self.fov = Fov()

    class Adapter(nn.Module):
        key = "depth_pro"

        def __init__(self):
            super().__init__()
            self.model = DepthPro()

    return Adapter()


def _cfg(freeze: str):
    from symtrs.config import RunConfig

    return RunConfig.from_dict(
        {"task": "depth", "model": "depth_pro", "data": {"root": "/data/SyMTRS"},
         "train": {"freeze": freeze}}
    ).validate()


@requires_torch
def test_freeze_encoder_leaves_only_decoder_head_and_fov_head_trainable():
    from symtrs.engine.trainer import apply_run_freeze

    adapter = _depth_pro_like()
    apply_run_freeze(adapter, _cfg("encoder"))

    trainable = {name.split(".")[1] for name, p in adapter.named_parameters() if p.requires_grad}
    frozen = {name for name, p in adapter.named_parameters() if not p.requires_grad}
    # Everything under model.encoder (patch/image ViTs and their projections) and the FOV
    # ViT (model.fov.encoder) must be frozen; the part that adapts to a new sensor must not.
    assert trainable == {"decoder", "head", "fov"}
    assert all(
        name.startswith(("model.encoder.", "model.fov.encoder.")) for name in frozen
    ), frozen
    assert not any(
        p.requires_grad for n, p in adapter.named_parameters() if n.startswith("model.encoder.")
    )
    assert adapter.model.fov.head.weight.requires_grad


@requires_torch
def test_an_empty_policy_changes_nothing():
    from symtrs.engine.trainer import apply_run_freeze

    adapter = _depth_pro_like()
    apply_run_freeze(adapter, _cfg(""))
    assert all(p.requires_grad for p in adapter.parameters())


@requires_torch
def test_a_policy_that_freezes_everything_names_the_fix(monkeypatch):
    """The optimiser would fail later with a less useful message; say it where it happened."""
    import symtrs.model.base as base
    from symtrs.engine.trainer import apply_run_freeze

    def freeze_all(model, policy, **kwargs):
        for parameter in model.parameters():
            parameter.requires_grad = False

    monkeypatch.setattr(base, "apply_freeze_policy", freeze_all)
    with pytest.raises(RuntimeError, match=r"symtrs doctor --model depth_pro --freeze encoder"):
        apply_run_freeze(_depth_pro_like(), _cfg("encoder"))


@requires_torch
def test_the_trainer_applies_the_policy_before_the_model_moves_or_is_wrapped():
    """DDP builds its reducer from the parameters that require grad at wrap time, and the
    optimiser is built from them: applying the policy after either is the original bug."""
    from symtrs.engine import trainer

    source = inspect.getsource(trainer._run)
    freeze_at = source.index("apply_run_freeze(model, cfg)")
    assert freeze_at < source.index("model.to(resolved)")
    assert freeze_at < source.index("DistributedDataParallel")
    assert freeze_at < source.index("build_optimizer(model, cfg)")


def test_depth_pro_configs_are_batch_one_uncropped_and_encoder_frozen_at_every_resolution():
    """`infer` resizes any input to the fixed 1536px the architecture asserts, so the crop
    and the resolution do not shrink the step; only the batch and the frozen encoder do."""
    for resolution in (2048, 1024, 512, 256):
        path = REPO_ROOT / "configs" / "depth" / str(resolution) / f"depth_pro_{resolution}.yaml"
        train = yaml.safe_load(path.read_text(encoding="utf-8"))["train"]
        assert train["batch_size"] == 1, f"{resolution}: batch 2 does not fit in 12GB"
        assert train["patch_size"] == 0, f"{resolution}: a crop is resized to 1536 anyway"
        assert train["freeze"] == "encoder", f"{resolution}: unfrozen it is >11GiB at batch 1"
        assert train["amp"] is True
        assert train["batch_size"] * train["grad_accum"] == 8, f"{resolution}: effective batch"


@requires_torch
def test_midas_freeze_encoder_keeps_its_decoder_trainable():
    """MiDaS keeps the ViT (`backbone.pretrained`) and the decoder (`backbone.scratch`) under
    one `backbone.` attribute. The default prefixes matched all of it, so once the policy was
    really applied `freeze: encoder` left zero trainable parameters on every MiDaS config."""
    import torch.nn as nn

    from symtrs.model.base import apply_freeze_policy
    from symtrs.model.depth.midas import MiDaSAdapter

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.pretrained = nn.Linear(2, 2)
            self.scratch = nn.Linear(2, 2)

    class Fake(nn.Module):
        freeze_encoder_prefixes = MiDaSAdapter.freeze_encoder_prefixes

        def __init__(self):
            super().__init__()
            self.backbone = Backbone()

    fake = Fake()
    apply_freeze_policy(fake, "encoder")
    assert not fake.backbone.pretrained.weight.requires_grad
    assert fake.backbone.scratch.weight.requires_grad

    # And the default really would have frozen everything: that is what the attribute is for.
    default = Fake()
    default.freeze_encoder_prefixes = None
    apply_freeze_policy(default, "encoder")
    assert not any(p.requires_grad for p in default.parameters())
