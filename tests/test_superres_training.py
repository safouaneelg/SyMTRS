"""Super-resolution: a crashed run must fail loudly, train on crops, and be scored whole.

Each test names the mistake it exists to prevent:

  * the front-end wrote `summary.json` with `returncode: 1` and exited 0, so a smoke pass
    recorded 13 out-of-memory runs as `ok` and the queue would have skipped them for ever;
  * SRCNN / the autoencoder / SRGAN trained on whole 2048 px frames because the trainer
    only cropped SwinIR (and cropped after the bicubic upsample, which is the 31.75 GiB tensor);
  * validation and test centre-cropped to the TRAINING patch at `val_patch_size: 0`, so a
    SwinIR "test number" was one 64 px window per image;
  * `predict_tiled` could not stitch an upscaler: its windows came back `scale` times larger.
"""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import REPO_ROOT, requires_torch
from symtrs.config import RunConfig
from symtrs.model.superres import SuperResolutionError, runner
from symtrs.registry import build_model

sys.path.insert(0, str(REPO_ROOT / "scripts"))


def _config(root: Path, run_dir: Path, **overrides) -> RunConfig:
    cfg = RunConfig(task="superresolution", model="srcnn", resolution=512, out_dir=str(run_dir))
    return cfg.with_overrides({"data.root": str(root), **overrides}).validate()


def _flag(command, name):
    return command[command.index(name) + 1]


# ------------------------------------------------------ a failed run must not look finished


def test_a_crashed_trainer_raises_and_leaves_no_summary_json(monkeypatch, layout: Path, tmp_path: Path) -> None:
    """The queue treats summary.json as "done"; a failed run must not leave one behind."""
    run = tmp_path / "run"
    monkeypatch.setattr(runner.subprocess, "run", lambda command: SimpleNamespace(returncode=1))

    with pytest.raises(SuperResolutionError) as excinfo:
        runner.run_super_resolution(_config(layout, run))

    assert not (run / "summary.json").exists(), "the queue would skip this run for ever"
    failed = json.loads((run / "summary_failed.json").read_text(encoding="utf-8"))
    assert failed["returncode"] == 1
    assert "exited with code 1" in str(excinfo.value)
    assert "train.batch_size" in str(excinfo.value), "an OOM message must name the knobs"


def test_symtrs_train_exits_non_zero_when_the_trainer_crashes(monkeypatch, layout: Path, tmp_path: Path, capsys) -> None:
    from symtrs.cli import main

    monkeypatch.setattr(runner.subprocess, "run", lambda command: SimpleNamespace(returncode=1))
    code = main([
        "train", "--task", "superresolution", "--model", "srcnn", "--resolution", "512",
        "--data-root", str(layout), "--out-dir", str(tmp_path / "run"),
    ])

    assert code == 2, "a crashed trainer is a reported error, not a bug in the package (3)"
    assert "SuperResolutionError" in capsys.readouterr().err


def test_a_successful_run_still_writes_summary_json(monkeypatch, layout: Path, tmp_path: Path) -> None:
    run = tmp_path / "run"

    def fake_run(command):
        (run / "test_metrics.json").write_text(
            json.dumps({"psnr": 24.0, "ssim": 0.8, "mse": 0.004, "images": 2}), encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    summary = runner.run_super_resolution(_config(layout, run))

    assert summary["test"]["psnr"] == 24.0
    assert json.loads((run / "summary.json").read_text(encoding="utf-8"))["returncode"] == 0
    assert not (run / "summary_failed.json").exists()


def test_a_training_run_writes_config_yaml_and_a_dry_run_does_not(monkeypatch, layout: Path, tmp_path: Path) -> None:
    """AGENTS.md section 5: every run leaves a resolved `config.yaml`, training included.

    The runner wrote it only when re-scoring, so the 16 super-resolution validation runs had
    none and `scripts/runs_report.py` (which keys on it) silently dropped them from RUNS.md
    (80 runs listed instead of 96).
    """
    import yaml

    monkeypatch.setattr(runner.subprocess, "run", lambda command: SimpleNamespace(returncode=0))

    dry = tmp_path / "dry"
    runner.run_super_resolution(_config(layout, dry), dry_run=True)
    assert not (dry / "config.yaml").exists(), "a dry run must leave nothing behind"

    run = tmp_path / "run"
    runner.run_super_resolution(_config(layout, run, **{"train.epochs": 3}))
    recorded = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    assert recorded["task"] == "superresolution"
    assert recorded["model"] == "srcnn"
    assert recorded["resolution"] == 512
    assert recorded["train"]["epochs"] == 3
    assert Path(recorded["out_dir"]).is_absolute(), "the anchored config is what ran"


def test_a_stale_test_metrics_file_is_not_folded_into_a_new_run(monkeypatch, layout: Path, tmp_path: Path) -> None:
    """Exit 0 with nothing scored must not inherit the previous run's number."""
    run = tmp_path / "run"
    run.mkdir()
    (run / "test_metrics.json").write_text(json.dumps({"psnr": 99.0}), encoding="utf-8")
    monkeypatch.setattr(runner.subprocess, "run", lambda command: SimpleNamespace(returncode=0))

    summary = runner.run_super_resolution(_config(layout, run))

    assert "test" not in summary


def test_eval_that_scores_nothing_is_an_error(monkeypatch, layout: Path, tmp_path: Path) -> None:
    """On `symtrs eval` the test number is the whole point; exit 0 without one is a failure."""
    run = tmp_path / "run"
    (run / "weights").mkdir(parents=True)
    (run / "weights" / "best.pt").write_bytes(b"")
    monkeypatch.setattr(runner.subprocess, "run", lambda command: SimpleNamespace(returncode=0))

    with pytest.raises(SuperResolutionError) as excinfo:
        runner.run_super_resolution(_config(layout, run), mode="eval")

    assert "nothing was scored" in str(excinfo.value)
    assert not (run / "summary.json").exists()


# ---------------------------------------------------------- the queue's definition of "ok"


def test_run_verdict_demands_a_finished_scored_run(tmp_path: Path) -> None:
    import run_queue

    def write(summary) -> Path:
        run = tmp_path / f"run{len(list(tmp_path.iterdir()))}"
        run.mkdir()
        if summary is not None:
            (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return run

    scored = {"returncode": 0, "test": {"psnr": 20.0}}
    assert run_queue.run_verdict(0, write(scored)) == (True, "")
    assert run_queue.run_verdict(0, write({"test": {"miou": 0.5}})) == (True, ""), "no returncode key"

    # the three ways the old check said "ok" about a run that was not
    ok, why = run_queue.run_verdict(0, write({"returncode": 1}))
    assert not ok and why == "summary returncode=1"
    ok, why = run_queue.run_verdict(0, write({"returncode": 0, "test": {}}))
    assert not ok and why == "summary has no test metrics"
    ok, why = run_queue.run_verdict(0, write(None))
    assert not ok and why == "no summary.json"

    ok, why = run_queue.run_verdict(1, write(scored))
    assert not ok and why == "rc=1", "a non-zero exit wins over a summary left behind"
    ok, why = run_queue.run_verdict("timeout", write(scored))
    assert not ok and why == "rc=timeout"

    # an eval-only job writes metrics elsewhere and is allowed an empty test dict
    assert run_queue.run_verdict(0, write({"returncode": 0}), eval_only=True)[0]


def test_run_verdict_rejects_a_summary_that_is_not_json(tmp_path: Path) -> None:
    import run_queue

    (tmp_path / "summary.json").write_text("{not json", encoding="utf-8")
    assert run_queue.run_verdict(0, tmp_path) == (False, "summary.json unreadable")


def test_smoke_rows_are_reverified_against_the_run_directory(tmp_path: Path) -> None:
    """smoke.tsv is a ledger: an `ok` row whose summary says returncode 1 must be re-run."""
    import smoke_all

    good, bad = tmp_path / "good", tmp_path / "bad"
    good.mkdir(), bad.mkdir()
    (good / "summary.json").write_text(json.dumps({"test": {"psnr": 1.0}}), encoding="utf-8")
    (bad / "summary.json").write_text(json.dumps({"returncode": 1}), encoding="utf-8")
    rows = {"good": {"batch": 4, "accum": None}, "bad": {"batch": 4, "accum": None}, "gone": {"batch": 1, "accum": None}}

    assert set(smoke_all.verified(rows, tmp_path)) == {"good"}


def test_an_oom_in_the_legacy_trainers_scoring_is_attributed_to_evaluation() -> None:
    """The legacy loop's frames are `eval_one_epoch` / `train_one_epoch`; without the former in
    EVAL_FRAMES an evaluation OOM read as "unknown" and was halved like a training one."""
    import smoke_all

    def traceback(frame: str) -> str:
        return (
            "Traceback (most recent call last):\n"
            '  File "train.py", line 600, in main_worker\n    x\n'
            f'  File "train.py", line 340, in {frame}\n    y\n'
            "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 31.75 GiB\n"
        )

    assert smoke_all.classify_oom(traceback("train_one_epoch")) == "train"
    assert smoke_all.classify_oom(traceback("eval_one_epoch")) == "eval"
    assert smoke_all.classify_oom(traceback("super_resolve")) == "eval"


# ----------------------------------------------------------------- what the command carries


def test_every_model_gets_the_crop_and_the_eval_window(layout: Path, tmp_path: Path) -> None:
    """The crop used to be passed for SwinIR only, so the other three trained on whole frames."""
    cfg = _config(layout, tmp_path / "run", **{
        "train.patch_size": 64, "eval.tile_size": 512, "eval.tile_overlap": 64})
    for key in ("srcnn", "autoencoder", "srgan", "swinir"):
        command = build_model(key).train_command(cfg, tmp_path / "run")
        assert _flag(command, "--patch_size") == "64", key
        assert _flag(command, "--eval_tile_size") == "512", key
        assert _flag(command, "--eval_tile_overlap") == "64", key


def test_the_validation_split_comes_from_the_dataset_not_from_train(layout: Path, tmp_path: Path) -> None:
    """Ground rule 6: never carve a validation set out of train/."""
    command = build_model("srcnn").train_command(_config(layout, tmp_path / "run"), tmp_path / "run")
    assert _flag(command, "--val_lr_root") == str(layout / "val" / "rgb_day_lr" / "x4")
    assert _flag(command, "--val_hr_root") == str(layout / "val" / "rgb_day")


def test_scoring_only_does_not_ask_for_a_validation_split(layout: Path, tmp_path: Path) -> None:
    command = build_model("srcnn").train_command(_config(layout, tmp_path / "run"), tmp_path / "run", eval_only=True)
    assert "--val_lr_root" not in command


def test_generated_configs_crop_every_model_and_tile_the_evaluation() -> None:
    """The grid is the source of truth for the patch sizes; this pins the reason for each."""
    for scale in (1, 2, 4, 8):
        for key in ("srcnn", "autoencoder", "srgan"):
            text = (REPO_ROOT / "configs" / "superresolution" / f"{key}_x{scale}.yaml").read_text(encoding="utf-8")
            assert f"patch_size: {256 // scale}\n" in text, f"{key} x{scale}: a fixed 256 px target region"
        swinir = (REPO_ROOT / "configs" / "superresolution" / f"swinir_x{scale}.yaml").read_text(encoding="utf-8")
        assert "patch_size: 64\n" in swinir, "SwinIR: a fixed 64 px LR patch, i.e. a fixed token count"
        for key in ("srcnn", "autoencoder", "srgan", "swinir"):
            text = (REPO_ROOT / "configs" / "superresolution" / f"{key}_x{scale}.yaml").read_text(encoding="utf-8")
            assert "  val_patch_size: 0\n" in text
            # The autoencoder runs at the target size, so its window is capped at 1024 target
            # pixels: a whole 2048 px frame in one pass ran out of memory at x4 / x8.
            expected = min(512, 1024 // scale) if key == "autoencoder" else 512
            assert f"  tile_size: {expected}\n" in text, (key, scale)


# ------------------------------------------------------------- tiling an upscaling network


@requires_torch
def test_predict_tiled_stitches_an_upscaler() -> None:
    """A pointwise x3 upsample stitched from overlapping windows equals the one-pass answer."""
    import torch
    import torch.nn.functional as F

    from symtrs.engine.tiling import predict_tiled

    image = torch.rand(1, 3, 40, 52)

    def forward(tile):
        return F.interpolate(tile, scale_factor=3, mode="nearest")

    whole = forward(image)
    tiled = predict_tiled(image, forward, tile_size=16, overlap=4, scale=3)

    assert tiled.shape == (1, 3, 120, 156)
    assert torch.allclose(tiled, whole, atol=1e-5)


@requires_torch
def test_predict_tiled_refuses_a_window_of_the_wrong_size() -> None:
    """Resizing would blend a misregistered window into its neighbours, so it is an error."""
    import torch

    from symtrs.engine.tiling import predict_tiled

    with pytest.raises(ValueError) as excinfo:
        predict_tiled(torch.rand(1, 1, 32, 32), lambda tile: tile, tile_size=16, overlap=4, scale=2)
    assert "exactly scale x its input" in str(excinfo.value)


@requires_torch
def test_scale_one_is_the_old_behaviour() -> None:
    import torch

    from symtrs.engine.tiling import predict_tiled

    image = torch.rand(1, 1, 33, 29)
    assert torch.allclose(predict_tiled(image, lambda t: t * 2, 16, 4), image * 2, atol=1e-5)


# ------------------------------------------------------------------ the legacy trainer


def _trainer():
    from symtrs.legacy.superres import train

    return train


def _args(**kw) -> Namespace:
    base = dict(
        model="swinir", scale=2, use_y_channel=False, amp=False, patch_size=0, val_patch_size=0,
        eval_tile_size=0, eval_tile_overlap=0, is_main=False, adv_weight=1e-3, swinir_window=7,
        swinir_checkpoint=False,
    )
    base.update(kw)
    return Namespace(**base)


class _Spy:
    """A stand-in network that records the spatial size it is given and upsamples nearest."""

    def __init__(self, scale: int, margin: int = 0) -> None:
        self.scale, self.margin, self.seen = scale, margin, []

    def eval(self):
        return self

    def __call__(self, x):
        import torch.nn.functional as F

        self.seen.append(tuple(x.shape[-2:]))
        out = F.interpolate(x, scale_factor=self.scale, mode="nearest") if self.scale > 1 else x
        m = self.margin
        return out[..., m:-m, m:-m] if m else out


@requires_torch
@pytest.mark.parametrize("model", ["srcnn", "autoencoder", "srgan", "swinir"])
def test_every_model_trains_on_crops_not_whole_frames(model: str) -> None:
    """The reason the SRCNN smoke asked for 31.75 GiB: only SwinIR was ever cropped."""
    import torch

    train = _trainer()
    scale, patch = 2, 8
    args = _args(model=model, scale=scale, patch_size=patch)
    seen = []

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.ones(1))

        def forward(self, x):
            seen.append(tuple(x.shape[-2:]))
            out = torch.nn.functional.interpolate(x, scale_factor=scale, mode="nearest") if model in ("srgan", "swinir") else x
            return out * self.w

    net = Net()
    batch = {"lr": torch.rand(2, 3, 32, 32), "hr": torch.rand(2, 3, 64, 64)}
    optim = torch.optim.SGD(net.parameters(), lr=0.1)
    disc = None
    if model == "srgan":
        class Disc(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch.ones(1))

            def forward(self, x):
                return x.mean(dim=(1, 2, 3), keepdim=True) * self.w

        disc = Disc()
    train.train_one_epoch(args, net, disc, [batch], optim, torch.optim.SGD(disc.parameters(), lr=0.1) if disc else None,
                          torch.device("cpu"))

    expected = patch * scale if model in ("srcnn", "autoencoder") else patch
    assert seen and all(s == (expected, expected) for s in seen), (model, seen)


@requires_torch
def test_validation_scores_the_whole_image_not_the_training_patch() -> None:
    """`val_patch_size: 0` used to mean "use the training patch": one 64 px window per image."""
    import torch

    train = _trainer()
    args = _args(model="swinir", scale=2, patch_size=8, val_patch_size=0)
    spy = _Spy(2)
    batch = {"lr": torch.rand(2, 3, 32, 24), "hr": torch.rand(2, 3, 64, 48)}

    loss, psnr, mse, ssim = train.eval_one_epoch(args, spy, None, [batch], torch.device("cpu"))

    assert spy.seen == [(32, 24), (32, 24)], "each image alone, at full size, never the 8 px patch"
    assert 0.0 <= ssim <= 1.0 and mse > 0


@requires_torch
def test_a_positive_val_patch_size_is_still_a_centre_crop() -> None:
    import torch

    train = _trainer()
    spy = _Spy(2)
    batch = {"lr": torch.rand(1, 3, 32, 32), "hr": torch.rand(1, 3, 64, 64)}
    train.eval_one_epoch(_args(scale=2, val_patch_size=16), spy, None, [batch], torch.device("cpu"))
    assert spy.seen == [(16, 16)]


@requires_torch
def test_windowed_scoring_equals_one_pass_scoring() -> None:
    """Val and test share one path, and windows only decide how the image is pushed through."""
    import torch

    train = _trainer()
    spy_whole, spy_tiled = _Spy(4), _Spy(4)
    batch = {"lr": torch.rand(1, 3, 40, 40), "hr": torch.rand(1, 3, 160, 160)}
    cpu = torch.device("cpu")

    whole = train.eval_one_epoch(_args(scale=4), spy_whole, None, [batch], cpu)
    tiled = train.eval_one_epoch(_args(scale=4, eval_tile_size=16, eval_tile_overlap=4), spy_tiled, None, [batch], cpu)

    assert max(max(s) for s in spy_tiled.seen) <= 16, "the windows were really used"
    assert len(spy_tiled.seen) > 1 and spy_whole.seen == [(40, 40)]
    for a, b in zip(whole, tiled):
        assert a == pytest.approx(b, rel=1e-4)


@requires_torch
def test_psnr_is_averaged_per_image_not_taken_from_the_batch_mean_mse() -> None:
    """Two images with MSE 0.01 and 0.0001: per-image PSNR is 30 dB, a batch-MSE PSNR is 22.97."""
    import torch

    train = _trainer()
    hr = torch.full((2, 3, 16, 16), 0.5)
    lr = hr.clone()
    lr[0] += 0.1
    lr[1] += 0.01
    spy = _Spy(1)
    _, psnr, _, _ = train.eval_one_epoch(_args(scale=1), spy, None, [{"lr": lr, "hr": hr}], torch.device("cpu"))

    assert psnr == pytest.approx((20.0 + 40.0) / 2, abs=0.1)


@requires_torch
def test_srcnn_is_padded_so_its_window_comes_back_the_size_it_went_in() -> None:
    """SRCNN's valid convolutions eat 12 px; a stitched window has to be exactly scale x its input."""
    import torch

    from symtrs.legacy.superres.models import SRCNN

    train = _trainer()
    args = _args(model="srcnn", scale=2, use_y_channel=True)
    out = train.super_resolve(args, SRCNN(1, 1).eval(), torch.rand(1, 3, 20, 24))
    assert out.shape == (1, 1, 40, 48)


@requires_torch
def test_srgan_generator_builds_at_x1() -> None:
    """srgan_x1 never ran: the generator refused scale 1, a ValueError the smoke pass misread as OOM."""
    import torch

    from symtrs.legacy.superres.models import SRGANGenerator

    net = SRGANGenerator(scale=1, num_residual_blocks=1).eval()
    assert net(torch.rand(1, 3, 16, 16)).shape == (1, 3, 16, 16)
    with pytest.raises(ValueError, match="scale must be one of"):
        SRGANGenerator(scale=3)


@requires_torch
def test_swinir_is_built_for_the_training_crop_not_the_whole_frame() -> None:
    """A 2048 px frame means ~10 GB of precomputed attention masks, stored in every checkpoint."""
    import torch

    train = _trainer()
    sample = torch.zeros(3, 2048, 2048)
    assert train.swinir_img_size(_args(patch_size=64), sample) == (70, 70)
    assert train.swinir_img_size(_args(patch_size=0), sample) == (2051, 2051), "no crop: the old behaviour"


# ------------------------------------------------------------ fixes from the review pass


def test_a_failed_rerun_removes_the_older_summary(monkeypatch, layout: Path, tmp_path: Path) -> None:
    """An old success beside summary_failed.json would still read as finished to the queue."""
    run = tmp_path / "run"
    run.mkdir()
    (run / "summary.json").write_text('{"test": {"psnr": 30.0}}', encoding="utf-8")
    monkeypatch.setattr(runner.subprocess, "run", lambda command: SimpleNamespace(returncode=1))

    with pytest.raises(SuperResolutionError):
        runner.run_super_resolution(_config(layout, run))

    assert not (run / "summary.json").exists()
    assert (run / "summary_failed.json").exists()


@requires_torch
def test_a_patch_larger_than_the_frame_is_an_error_not_a_whole_frame_step() -> None:
    import torch

    train = _trainer()
    lr, hr = torch.rand(1, 3, 16, 16), torch.rand(1, 3, 32, 32)
    with pytest.raises(ValueError, match="lower train.patch_size"):
        train.random_crop_pair(lr, hr, scale=2, patch_size=32)


@requires_torch
def test_scoring_clamps_the_output_to_the_saved_image_range() -> None:
    """An output overshooting [0, 1] is scored as the PNG it would be saved as."""
    import torch

    train = _trainer()
    hr = torch.ones(1, 3, 16, 16)
    lr = torch.full((1, 3, 16, 16), 1.5)  # the identity "network" overshoots by 0.5
    _, psnr, mse, _ = train.eval_one_epoch(_args(scale=1), _Spy(1), None, [{"lr": lr, "hr": hr}], torch.device("cpu"))

    assert mse == pytest.approx(0.0, abs=1e-12), "clamped to 1.0, the overshoot is not an error"


def test_runs_report_lists_a_run_that_has_no_config_yaml(tmp_path: Path) -> None:
    """16 super-resolution runs had no config.yaml and the report dropped them without a word.

    A run with a summary.json is listed from what it recorded (the trainer argv carries batch
    size, learning rate, crop and tile); it is never invented from nothing.
    """
    import runs_report

    summ = {
        "task": "superresolution", "model": "srcnn", "resolution": 512, "epochs": 2,
        "command": ["python", "-m", "x", "--batch_size", "16", "--lr", "0.0001",
                    "--patch_size", "64", "--eval_tile_size", "512", "--use_y_channel"],
    }
    cfg = runs_report.config_from_summary(summ)
    assert (cfg["task"], cfg["model"], cfg["resolution"]) == ("superresolution", "srcnn", 512)
    assert cfg["train"]["batch_size"] == 16 and cfg["train"]["patch_size"] == 64
    assert cfg["train"]["lr"] == 0.0001 and cfg["eval"]["tile_size"] == 512
    assert runs_report.argv_flags(summ["command"])["use_y_channel"] is True
    assert "2026-10-17" in runs_report.NO_CONFIG_NOTE
