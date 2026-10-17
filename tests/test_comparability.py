"""Guards that keep the domain-adaptation and shadow-removal numbers comparable.

Each test asserts the reason for a guard, not only a value: FID/KID going through one resize
path on both sides, KID leading the columns, a shadow init that cannot silently be a
from-scratch run, SSIM named for what it is, and the smoke batch search keeping the grid's
effective batch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT, requires_numpy, requires_pil, requires_torch

sys.path.insert(0, str(REPO_ROOT / "scripts"))


# ------------------------------------------------------------------ FID / KID


@requires_torch
def test_real_images_are_antialiased_before_the_shared_resize():
    """High-frequency detail is pure aliasing at this scale: antialiasing flattens it, plain
    bilinear subsampling keeps it. This is the sharpness the real side used to leak."""
    import torch
    import torch.nn.functional as F

    from symtrs.metrics.domain import inception_preprocess, match_resolution

    # Vertical stripes of period 4 px at 1024 px: far above the Nyquist limit of 256 / 299.
    stripes = (torch.arange(1024) % 4 < 2).float().view(1, 1, 1, -1).expand(1, 3, 1024, 1024)
    real = stripes.contiguous()
    aliased = F.interpolate(real, size=(299, 299), mode="bilinear", align_corners=False)
    assert aliased.std() > 0.3  # plain bilinear subsampling keeps the pattern: the old bug
    assert match_resolution(real, (256, 256)).std() < 0.05
    # inception_preprocess itself is antialiased too (it is the one path both sides share)
    assert (inception_preprocess(real) * 0.226).std() < 0.05


@requires_torch
def test_match_resolution_leaves_the_generated_side_untouched():
    import torch

    from symtrs.metrics.domain import match_resolution

    generated = torch.rand(2, 3, 256, 256)
    assert match_resolution(generated, 256) is generated
    assert match_resolution(generated, (256, 256)) is generated


@requires_torch
def test_extract_features_resizes_only_when_asked():
    import torch

    from symtrs.metrics.domain import extract_features

    seen = []

    def extractor(images):
        seen.append(tuple(images.shape[-2:]))
        return images.mean(dim=(1, 2, 3)).unsqueeze(1)

    loader = [{"image": torch.rand(2, 3, 128, 128)}]
    extract_features(loader, extractor=extractor, progress=False, resize_to=(64, 64))
    extract_features(loader, extractor=extractor, progress=False)
    assert seen == [(64, 64), (128, 128)]


@requires_pil
def test_generated_size_is_read_from_the_files_and_mixed_sizes_raise(tmp_path: Path):
    from PIL import Image

    from symtrs.model.domain.runner import generated_image_size

    a, b = tmp_path / "a.png", tmp_path / "b.png"
    Image.new("RGB", (40, 30)).save(a)
    Image.new("RGB", (40, 30)).save(b)
    assert generated_image_size([a, b]) == (30, 40)  # (height, width), not hard-coded 256
    c = tmp_path / "c.png"
    Image.new("RGB", (64, 64)).save(c)
    with pytest.raises(RuntimeError, match="share a size"):
        generated_image_size([a, c])


@requires_numpy
def test_kid_leads_and_fid_is_flagged_rank_deficient_for_small_n():
    import numpy as np

    pytest.importorskip("torch")
    from symtrs.metrics import domain

    rng = np.random.default_rng(0)
    real = rng.normal(size=(24, 2048))
    fake = rng.normal(size=(24, 2048)) + 0.1
    metrics = domain.distribution_metrics(real, fake)
    keys = list(metrics)
    assert keys.index(domain.PRIMARY_METRIC) < keys.index("fid")
    assert metrics["fid_rank_deficient"] == 1.0
    # a covariance with more samples than dimensions is not flagged
    enough = domain.distribution_metrics(rng.normal(size=(60, 8)), rng.normal(size=(60, 8)))
    assert enough["fid_rank_deficient"] == 0.0


@requires_numpy
def test_the_full_set_kid_reports_no_spread_rather_than_a_zero_one():
    """`num_subsets=0` draws no subsets, so no standard deviation exists. It used to be written
    as 0.0, which reads as a measured zero spread next to N=24; it must be `None` (`null` in
    summary.json) and the KID itself must be the same single full-set estimate as before."""
    import numpy as np

    pytest.importorskip("torch")
    from symtrs.common.jsonio import dumps
    from symtrs.metrics import domain

    rng = np.random.default_rng(0)
    real, fake = rng.normal(size=(24, 64)), rng.normal(size=(24, 64)) + 0.2
    full = domain.compute_kid(real, fake)
    assert full["kid_std"] is None
    assert domain.distribution_metrics(real, fake)["kid_std"] is None
    assert '"kid_std": null' in dumps(full)
    # the estimate is untouched: it is the unbiased MMD^2 over all samples
    assert full["kid"] == domain.compute_kid(real, fake, num_subsets=0)["kid"]
    # drawing subsets is what produces a spread
    subsets = domain.compute_kid(real, fake, num_subsets=5, subset_size=12)
    assert subsets["kid_std"] is not None and subsets["kid_std"] > 0


def test_a_legacy_zero_kid_std_is_not_printed_as_a_measurement():
    """Summaries written before the fix still hold `kid_std: 0.0`; the report shows n/a."""
    import results_report

    assert results_report._cell("kid_std", {"kid_std": 0.0}, {}, {}) == "n/a"
    assert results_report._cell("kid_std", {"kid_std": None}, {}, {}) == "n/a"
    assert results_report._cell("kid_std", {"kid_std": 0.0123}, {}, {}) == "0.0123"


def test_the_runner_and_the_metric_module_agree_on_the_primary_metric():
    pytest.importorskip("torch")
    from symtrs.metrics.domain import PRIMARY_METRIC
    from symtrs.model.domain import runner

    assert runner.PRIMARY_METRIC == PRIMARY_METRIC == "kid"
    assert "rank-deficient" in runner.FID_NOTE


def test_tables_put_kid_before_fid():
    from symtrs.commands import tables

    order = tables.COLUMN_ORDER["domain_adapt"]
    assert order.index("kid") < order.index("fid")


# --------------------------------------------------------------- shadow init


@requires_torch
def test_load_matching_refuses_a_mostly_unmatched_file(tmp_path: Path):
    import torch

    from symtrs.model.shadow.worker import load_matching

    net = torch.nn.Sequential(*[torch.nn.Linear(4, 4) for _ in range(5)])  # 10 tensors
    wrong = tmp_path / "wrong.pth"
    torch.save({"params": {"other.weight": torch.zeros(4, 4)}}, wrong)
    with pytest.raises(RuntimeError) as info:
        load_matching(net, str(wrong))
    message = str(info.value)
    assert "wrong.pth" in message and "0 of the model's 10" in message and "90%" in message

    # 8 of 10 tensors matching is still below 90%
    state = net.state_dict()
    partial = {k: v for i, (k, v) in enumerate(state.items()) if i < 8}
    half = tmp_path / "partial.pth"
    torch.save({"params": partial}, half)
    with pytest.raises(RuntimeError, match="8 of the model's 10"):
        load_matching(net, str(half))


@requires_torch
def test_load_matching_accepts_a_full_file_and_reports_counts(tmp_path: Path):
    import torch

    from symtrs.model.shadow.worker import load_matching

    source = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
    path = tmp_path / "ok.pth"
    torch.save({"params": {f"module.{k}": v for k, v in source.state_dict().items()}}, path)
    target = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
    assert load_matching(target, str(path)) == (4, 4)
    assert torch.equal(target[0].weight, source[0].weight)


def test_every_shadow_model_declares_what_its_released_weights_were_for():
    from symtrs.model.shadow.archs import ARCHS

    assert "shadow removal" in ARCHS["shadowformer"].init_source_task
    for key in ("restormer", "uformer"):
        task = ARCHS[key].init_source_task
        assert "denoising" in task and "no shadow-removal weights" in task
    # ShadowMaskFormer has no weights at all: its row is always a from-scratch row.
    smf = ARCHS["shadowmaskformer"]
    assert not smf.released_weights and "trained from scratch" in smf.init_source_task


@requires_numpy
def test_ssim_is_named_for_luma_and_the_note_travels_with_the_summary():
    import numpy as np

    from symtrs.metrics import shadow

    image = np.random.default_rng(0).random((32, 32, 3)).astype(np.float32)
    row = shadow.score_image(image, image, np.zeros((32, 32), bool))
    assert "ssim_y" in row and "ssim" not in row
    assert "luma" in shadow.SSIM_NOTE and "RGB channels" in shadow.SSIM_NOTE
    assert "ssim_y" in shadow.METRICS


# --------------------------------------------------------------- smoke batch


def _smoke():
    import smoke_all

    return smoke_all


def test_halving_keeps_the_effective_batch_for_accumulating_tasks():
    smoke = _smoke()
    assert smoke.halve(32, 1, 32, "segmentation") == (16, 2)
    assert smoke.halve(16, 2, 32, "depth") == (8, 4)
    assert smoke.halve(1, 4, 4, "depth") == (1, 4)
    # a task that ignores grad_accum cannot compensate, so accum is left alone
    assert smoke.halve(8, 1, 8, "superresolution") == (4, 1)


def test_load_rows_reads_old_nine_column_and_new_twelve_column_rows(tmp_path: Path):
    smoke = _smoke()
    tsv = tmp_path / "smoke.tsv"
    old = "\t".join(["t", "unet_small_512", "ok", "8", "1.0", ".venv", "segmentation", "l.log", ""])
    new = "\t".join(["t", "midas_small_512", "ok", "4", "1.0", ".venv", "depth", "l.log", "",
                    "4", "16", "16"])
    failed = "\t".join(["t", "x", "FAIL rc=1", "4", "1.0", ".venv", "depth", "l.log", "eval OOM",
                        "2", "8", "8"])
    tsv.write_text("\n".join([old, new, failed]) + "\n", encoding="utf-8")
    rows = smoke.load_rows(tsv)
    assert rows == {"unet_small_512": {"batch": 8, "accum": None},
                    "midas_small_512": {"batch": 4, "accum": 4}}
    assert smoke.load_done(tsv) == {"unet_small_512": 8, "midas_small_512": 4}


TRAIN_OOM = '''=== cmd
Traceback (most recent call last):
  File "x/symtrs/engine/trainer.py", line 478, in train
    train_metrics = train_one_epoch(
  File "x/symtrs/engine/loop.py", line 190, in train_one_epoch
    loss = _training_step(model, batch, device, task)
  File "x/symtrs/engine/loop.py", line 181, in _training_step
    prediction = forward_native(model, image, tile_size=0)
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB
'''

EVAL_OOM = '''=== cmd
Traceback (most recent call last):
  File "x/symtrs/engine/trainer.py", line 524, in train
    test_metrics = evaluate(
  File "x/symtrs/engine/loop.py", line 500, in evaluate
    return evaluate_depth(model, loader, cfg, device)
  File "x/symtrs/engine/loop.py", line 320, in evaluate_depth
    prediction = forward_native(model, image, cfg.eval.tile_size, cfg.eval.tile_overlap)
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB
'''


def test_oom_is_attributed_to_training_or_whole_image_evaluation():
    smoke = _smoke()
    assert smoke.classify_oom(TRAIN_OOM) == "train"
    assert smoke.classify_oom(EVAL_OOM) == "eval"
    assert smoke.classify_oom("RuntimeError: CUDA out of memory") == "unknown"
    assert smoke.classify_oom("nothing wrong here") == "unknown"


def test_read_header_reads_grad_accum(tmp_path: Path):
    smoke = _smoke()
    config = tmp_path / "c.yaml"
    config.write_text("task: depth\nmodel: midas_small\ntrain:\n  batch_size: 4\n  grad_accum: 8\n",
                      encoding="utf-8")
    assert smoke.read_header(config) == {"task": "depth", "model": "midas_small",
                                         "batch_size": 4, "grad_accum": 8}
