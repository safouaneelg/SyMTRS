"""The learning-rate schedule, resume, and which weights the test pass scores.

Three things the trainer got wrong without a number looking wrong:

  * The scheduler was sized in optimiser steps and stepped once per epoch, so the learning rate
    stayed on the first few percent of the warmup ramp (a 2-epoch run trained at 4-8% of the
    configured rate, and nothing ever decayed). These tests run the real `build_scheduler`
    against the real `train_one_epoch` on a tiny CPU model and count both sides.
  * `--resume` forgot `best`, so the first epoch afterwards overwrote `best.pt` even when worse.
  * `summary["test"]` was scored with the last epoch's weights next to a "best" that came from
    another epoch.

Nothing here reads a dataset or downloads weights; the loaders are lists of tensors.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import requires_torch

torch = pytest.importorskip("torch", reason="torch is not installed")

from symtrs.config import RunConfig  # noqa: E402
from symtrs.engine import loop, trainer  # noqa: E402

pytestmark = requires_torch


class _TinySeg(torch.nn.Module):
    """The smallest model the segmentation loop accepts: one 1x1 convolution."""

    task = "segmentation"
    key = "tiny"

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 6, 1)

    def preprocess_image(self, image):
        return image

    def forward(self, image):
        return self.conv(image)

    def compute_loss(self, logits, mask):
        loss = torch.nn.functional.cross_entropy(logits, mask, ignore_index=255)
        return loss, {}

    def describe(self) -> str:
        return "tiny"


def _loader(batches: int):
    generator = torch.Generator().manual_seed(0)
    return [
        {
            "image": torch.rand(2, 3, 8, 8, generator=generator),
            "mask": torch.randint(0, 6, (2, 8, 8), generator=generator),
        }
        for _ in range(batches)
    ]


def _cfg(tmp_path: Path, **train) -> RunConfig:
    cfg = RunConfig()
    cfg.task = "segmentation"
    cfg.model = "tiny"
    cfg.out_dir = str(tmp_path / "run")
    cfg.data.root = str(tmp_path)  # never read: the data factory is replaced
    cfg.train.device = "cpu"
    cfg.train.amp = False
    cfg.train.lr = 1e-2
    cfg.train.save_samples_every = 0
    cfg.train.log_every = 0
    cfg.eval.num_visualizations = 0
    for key, value in train.items():
        setattr(cfg.train, key, value)
    return cfg


# ------------------------------------------------------------------ the schedule


def _run_schedule(tmp_path: Path, batches: int, **train):
    """Train `epochs` epochs exactly as the trainer does; return the LR trace and the counts."""
    cfg = _cfg(tmp_path, **train)
    model = _TinySeg()
    optimizer = trainer.build_optimizer(model, cfg)
    scheduler = trainer.build_scheduler(
        optimizer, cfg, trainer.optimizer_steps_per_epoch(batches, cfg.train.grad_accum)
    )

    counts = {"optimizer": 0, "scheduler": 0}
    real_scheduler_step = scheduler.step

    def count_scheduler(*args, **kwargs):
        counts["scheduler"] += 1
        return real_scheduler_step(*args, **kwargs)

    # A post-hook rather than replacing `optimizer.step`: torch warns when the method is
    # swapped after the scheduler was built, and the hook counts the real calls.
    optimizer.register_step_post_hook(lambda *_: counts.__setitem__("optimizer", counts["optimizer"] + 1))
    scheduler.step = count_scheduler

    # `lr` the optimiser is *about to use* at each step: the value at the start of the step.
    seen = [optimizer.param_groups[0]["lr"]]
    data = _loader(batches)
    scaler = loop.make_grad_scaler(torch.device("cpu"), False)
    for epoch in range(cfg.train.epochs):
        loop.train_one_epoch(
            model, data, optimizer, scaler, cfg, torch.device("cpu"), epoch=epoch,
            scheduler=scheduler,
        )
        seen.append(optimizer.param_groups[0]["lr"])
    return cfg, seen, counts, optimizer, scheduler


def test_the_schedule_reaches_the_configured_peak_after_warmup_and_decays(tmp_path):
    """The failure this guards: a per-epoch `scheduler.step()` against a per-step schedule.

    With 4 epochs, 1 warmup epoch and 3 optimiser steps per epoch, the learning rate has to be
    at the configured value when warmup ends and at `min_lr_ratio` times it at the end. Per
    epoch it was at 0.08 of it after the first epoch and never decayed.
    """
    cfg, seen, counts, _, _ = _run_schedule(
        tmp_path, batches=5, epochs=4, warmup_epochs=1, grad_accum=2, sched="cosine"
    )
    base, floor = cfg.train.lr, cfg.train.lr * cfg.train.min_lr_ratio

    # After the warmup epoch (3 steps) the multiplier is exactly 1.
    assert seen[1] == pytest.approx(base), "warmup must end at the configured learning rate"
    # Strictly decaying afterwards, ending at the floor.
    assert seen[1] > seen[2] > seen[3] > seen[4]
    assert seen[-1] == pytest.approx(floor), "the cosine must reach min_lr_ratio at the last step"
    assert max(seen) <= base * (1 + 1e-9)
    # What the per-epoch bug produced after one epoch of a warmup this size.
    assert seen[1] > 0.5 * base


def test_scheduler_steps_equal_optimizer_steps_including_the_partial_group(tmp_path):
    """5 batches with `grad_accum=2` are 3 optimiser steps (2+2+1), not `5 // 2 = 2`.

    The sizing used a floor division while the loop steps the last partial group too, so the
    schedule's length and the number of steps taken could never agree.
    """
    _, _, counts, _, scheduler = _run_schedule(
        tmp_path, batches=5, epochs=3, warmup_epochs=1, grad_accum=2
    )
    assert trainer.optimizer_steps_per_epoch(5, 2) == 3
    assert counts["optimizer"] == counts["scheduler"] == 3 * 3
    assert scheduler.last_epoch == 9


def test_a_warmup_longer_than_the_run_is_capped_so_it_still_decays(tmp_path, capsys):
    """The 2-epoch validation runs carry `warmup_epochs: 2`. Uncapped, they ramp up for the
    whole run and finish at the peak without a single decay step; capped at half of the run,
    one epoch warms up and one decays, and the cap is announced."""
    cfg, seen, _, _, _ = _run_schedule(
        tmp_path, batches=4, epochs=2, warmup_epochs=2, sched="cosine"
    )
    assert "warming up for" in capsys.readouterr().out
    assert seen[1] == pytest.approx(cfg.train.lr), "peak after the (capped) warmup"
    assert seen[2] < seen[1], "and a decay after it"
    assert seen[2] == pytest.approx(cfg.train.lr * cfg.train.min_lr_ratio)


def test_a_warmup_within_half_the_run_is_left_as_configured(tmp_path, capsys):
    _run_schedule(tmp_path, batches=4, epochs=4, warmup_epochs=2)
    assert "warming up for" not in capsys.readouterr().out


def test_the_trainer_does_not_step_the_scheduler_itself():
    """The per-epoch `scheduler.step()` in `_run` was the bug; the loop owns the stepping."""
    import inspect

    source = inspect.getsource(trainer._run)
    assert "scheduler.step()" not in source
    assert "scheduler=scheduler" in source


# ------------------------------------------------- resume, best and the test weights


@pytest.fixture
def scripted(monkeypatch):
    """Replace the data factory and `evaluate`: val metrics follow a script, and every test
    pass records the weights it was handed.

    Returns a namespace with `val` (the values to hand out, in order) and `test_seen`.
    """

    class Script:
        val: list = []
        test_seen: list = []

    script = Script()
    script.val, script.test_seen = [], []

    import symtrs.data.factory as factory

    monkeypatch.setattr(factory, "from_config", lambda *args, **kwargs: _loader(2))

    def fake_evaluate(model, loader, cfg, device, align="", num_visualizations=0,
                      vis_dir=None, vis_prefix="eval"):
        if vis_prefix == "test":
            inner = getattr(model, "module", model)
            script.test_seen.append(inner.conv.weight.detach().clone())
            return {"miou": 0.123}
        return {"miou": script.val.pop(0)}

    monkeypatch.setattr(loop, "evaluate", fake_evaluate)
    return script


def _weights(run_dir: Path, tag: str):
    return torch.load(run_dir / "weights" / f"{tag}.pt", map_location="cpu", weights_only=False)


def test_the_test_pass_scores_best_weights_and_says_so(tmp_path, scripted):
    """val miou 0.5 -> 0.9 -> 0.3: epoch 1 is best, the last epoch is worse, and the test
    number must come from epoch 1's weights."""
    scripted.val = [0.5, 0.9, 0.3]
    cfg = _cfg(tmp_path, epochs=3, batch_size=2)
    summary = trainer.train(cfg, model=_TinySeg())

    run_dir = Path(summary["run_dir"])
    best, last = _weights(run_dir, "best"), _weights(run_dir, "last")
    assert summary["best"] == pytest.approx(0.9) and summary["best_epoch"] == 1
    assert summary["test_weights"] == "best"
    assert not torch.equal(best["model"]["conv.weight"], last["model"]["conv.weight"])
    assert torch.equal(scripted.test_seen[0], best["model"]["conv.weight"])
    on_disk = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert on_disk["test_weights"] == "best"


def test_the_test_pass_falls_back_to_last_when_no_epoch_was_finite(tmp_path, scripted):
    scripted.val = [float("nan"), float("nan")]
    cfg = _cfg(tmp_path, epochs=2, batch_size=2)
    summary = trainer.train(cfg, model=_TinySeg())

    run_dir = Path(summary["run_dir"])
    assert summary["best"] is None and summary["test_weights"] == "last"
    assert not (run_dir / "weights" / "best.pt").exists()
    assert torch.equal(scripted.test_seen[0], _weights(run_dir, "last")["model"]["conv.weight"])


def test_resume_restores_best_and_does_not_overwrite_best_with_a_worse_epoch(tmp_path, scripted):
    """Crash after epoch 1 (best 0.9), resume, and epoch 2 scores 0.3. `best.pt` must still be
    epoch 1's, `summary.json` must still say 0.9 at epoch 1, and the history must keep the
    epochs from before the resume."""
    scripted.val = [0.5, 0.9]
    first = trainer.train(_cfg(tmp_path, epochs=2, batch_size=2), model=_TinySeg())
    run_dir = Path(first["run_dir"])
    best_before = _weights(run_dir, "best")["model"]["conv.weight"].clone()

    scripted.val = [0.3]
    resumed_cfg = _cfg(tmp_path, epochs=3, batch_size=2, resume=str(run_dir / "weights" / "last.pt"))
    summary = trainer.train(resumed_cfg, model=_TinySeg())

    assert summary["best"] == pytest.approx(0.9) and summary["best_epoch"] == 1
    assert torch.equal(_weights(run_dir, "best")["model"]["conv.weight"], best_before)
    assert [row["epoch"] for row in summary["history"]] == [0, 1, 2]
    assert summary["test_weights"] == "best"
    assert torch.equal(scripted.test_seen[-1], best_before)


def test_resume_still_takes_a_genuine_improvement(tmp_path, scripted):
    scripted.val = [0.5, 0.6]
    first = trainer.train(_cfg(tmp_path, epochs=2, batch_size=2), model=_TinySeg())
    run_dir = Path(first["run_dir"])

    scripted.val = [0.8]
    cfg = _cfg(tmp_path, epochs=3, batch_size=2, resume=str(run_dir))
    summary = trainer.train(cfg, model=_TinySeg())
    assert summary["best"] == pytest.approx(0.8) and summary["best_epoch"] == 2


def test_resume_restores_the_scheduler_position(tmp_path, scripted):
    """The scheduler's state is saved after the per-step updates, so a resumed run continues the
    ramp where it stopped instead of restarting it."""
    scripted.val = [0.5, 0.6]
    cfg = _cfg(tmp_path, epochs=4, warmup_epochs=1, batch_size=2)
    cfg.train.epochs = 2  # the first leg only; the schedule below is for the 4-epoch run
    first = trainer.train(cfg, model=_TinySeg())
    run_dir = Path(first["run_dir"])
    payload = _weights(run_dir, "last")
    assert payload["scheduler_unit"] == "optimizer_step"
    saved_position = payload["scheduler"]["last_epoch"]
    assert saved_position == 2 * trainer.optimizer_steps_per_epoch(2, 1)

    model = _TinySeg()
    optimizer = trainer.build_optimizer(model, cfg)
    scheduler = trainer.build_scheduler(optimizer, cfg, 2)
    start, restored = trainer._resume(
        cfg.set_many(**{"train.resume": str(run_dir)}), model, optimizer, scheduler, run_dir, 2, "miou"
    )
    assert start == 2 and scheduler.last_epoch == saved_position
    assert restored["best"] == pytest.approx(0.6)


def test_a_checkpoint_from_the_per_epoch_scheduler_is_fast_forwarded_not_loaded(tmp_path):
    """Its `last_epoch` counts epochs; read as steps it would restart the warmup."""
    cfg = _cfg(tmp_path, epochs=4, warmup_epochs=1)
    model = _TinySeg()
    optimizer = trainer.build_optimizer(model, cfg)
    scheduler = trainer.build_scheduler(optimizer, cfg, 3)
    trainer._fast_forward_scheduler(scheduler, 2 * 3)
    assert scheduler.last_epoch == 6
    assert optimizer.param_groups[0]["lr"] == pytest.approx(
        scheduler.base_lrs[0] * scheduler.lr_lambdas[0](6)
    )
