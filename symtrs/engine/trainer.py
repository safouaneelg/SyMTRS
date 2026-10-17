"""The trainer: everything around the loop.

Optimiser, schedule, mixed precision, gradient accumulation, distributed launch,
checkpoints, the metrics CSV, the curves, and the end-of-run test pass. The loop
itself lives in `symtrs.engine.loop` and knows none of this.

Two design points worth stating because they are easy to get wrong:

  * The validation split is never trained on. `data.split` must differ from
    `data.val_split`, and that is enforced at config-validation time rather than left
    to the caller's discipline.
  * Every run writes a `config.yaml` next to its checkpoints. The checkpoint metadata
    also carries the resolved config, so a result can be reproduced from the `.pt`
    alone even if the run directory is lost.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from symtrs.common.checkpoint import load_checkpoint, save_checkpoint
from symtrs.common.jsonio import write_json
from symtrs.common.ddp import (
    barrier,
    cleanup_ddp,
    is_main_process,
    local_rank_from_env,
    setup_ddp,
    world_size_from_env,
)
from symtrs.common.meters import MetricTracker
from symtrs.common.seed import describe_device, resolve_device, set_seed
from symtrs.common.table import append_csv_row
from symtrs.config import MODES, RunConfig

#: `superresolution` is a real SyMTRS task, but its training loop is not this one. The
#: architectures, the loading, the metrics and the loop all predate this package and now
#: live inside it, under `symtrs/legacy/superres/` and `symtrs/legacy/dataloader/`.
#: `symtrs.model.superres` launches that trainer as a subprocess rather than reimplementing
#: it, so
#: a run reaches it exactly the way domain adaptation reaches upstream:
#
#     task == "superresolution" -> symtrs.model.superres.run_super_resolution(cfg, mode)
#
# What must *not* happen is this module's own pipeline handling it. `from_config` would
# build a depth or segmentation dataset, `forward_native` would run a depth or
# segmentation model over it, and the run would complete and print a plausible number
# under a task name with nothing to do with what it did. The routing below is what makes
# that unreachable; there is no separate guard any more, because there is no longer a
# code path here that could be reached by mistake.
_SUPERRESOLUTION_TASK = "superresolution"

#: Which validation metric decides "best", and whether lower is better. Used when
#: `eval.metrics` names nothing the task recognises.
MONITORS = {
    "depth": ("rmse", "min"),
    "segmentation": ("miou", "max"),
}

#: Every metric name that can decide a "best" checkpoint, and its direction. Lower is
#: better for errors; higher is better for the threshold accuracies, the IoUs and the
#: accuracies. The fitted alignment statistics (`align_scale_mean`, `align_shift_mean`)
#: are deliberately absent: they describe the alignment, not the quality of the model,
#: and a run whose "best" checkpoint was chosen by a drifting scale would be choosing
#: noise.
MONITOR_DIRECTIONS = {
    "rmse": "min",
    "mae": "min",
    "abs_rel": "min",
    "sq_rel": "min",
    "silog": "min",
    "log10": "min",
    "loss": "min",
    "d1": "max",
    "d2": "max",
    "d3": "max",
    "miou": "max",
    "pixel_acc": "max",
    "mean_acc": "max",
    "mean_dice": "max",
    "fwiou": "max",
    "acc": "max",
}


def resolve_monitor(cfg: RunConfig) -> tuple:
    """`(metric_name, "min"|"max")` for this run, defaulting per task.

    A name in `eval.metrics` that is not a monitor (a per-class IoU, say, or a fitted
    alignment statistic) is skipped rather than guessed at. If nothing matches, the
    task default applies and, when the caller asked for something specific and did not
    get it, that is said out loud - silently selecting a different checkpoint than the
    config asked for is the kind of thing only noticed on the final table.
    """
    recognised = [c.lower() for c in cfg.eval.metrics if c.lower() in MONITOR_DIRECTIONS]
    if recognised:
        name = recognised[0]
        return name, MONITOR_DIRECTIONS[name]

    if cfg.eval.metrics:
        print(
            f"[symtrs] none of eval.metrics={cfg.eval.metrics} can decide a best "
            f"checkpoint; falling back to {MONITORS.get(cfg.task, ('loss', 'min'))[0]!r}. "
            f"Choose from {sorted(MONITOR_DIRECTIONS)} to pick the checkpoint yourself."
        )
    return MONITORS.get(cfg.task, ("loss", "min"))


def is_improvement(value: float, best: Optional[float], direction: str) -> bool:
    """Whether `value` should replace `best` as the checkpoint to keep.

    A non-finite value never does, **including the first epoch**. The early return for
    `best is None` used to come first, so a run whose very first validation was NaN saved
    `best.pt` with rmse=NaN - and then nothing could beat it (every comparison with NaN is
    False), leaving the "best" checkpoint to be the one from the diverged epoch.
    """
    if not math.isfinite(value):
        return False
    if best is None:
        return True
    return value < best if direction == "min" else value > best


def apply_run_freeze(model, cfg: RunConfig) -> None:
    """Turn `train.freeze` into `requires_grad`, once, before anything reads it.

    Every adapter stores the policy it was built with, but only `trainable_parameters()`
    / `apply_freeze()` ever applied it and nothing on the training path called either:
    `build_optimizer` walks `named_parameters()` directly. So `freeze: encoder` was
    accepted, printed as "freeze='encoder' -> 952.0M trainable" (all of them) and did
    nothing - which is how Depth Pro, whose frozen encoder is what makes it fit in 12 GB
    at all, kept an autograd graph through 35 ViT-L passes and ran out of memory.

    It runs before the model is moved or wrapped in DDP (DDP builds its reducer from the
    parameters that require grad at wrap time) and before the optimiser is built, and it
    is the one place the policy is applied; the adapters' own helpers are left for
    `symtrs doctor`, which has no trainer.
    """
    if not cfg.train.freeze:
        return
    from symtrs.model.base import apply_freeze_policy

    inner = getattr(model, "module", model)
    apply_freeze_policy(inner, cfg.train.freeze)
    if not any(parameter.requires_grad for parameter in inner.parameters()):
        raise RuntimeError(
            f"train.freeze={cfg.train.freeze!r} left no trainable parameter in "
            f"{getattr(inner, 'key', type(inner).__name__)!r}: the policy recognises the "
            f"encoder by module-name prefix. Run `symtrs doctor --model "
            f"{getattr(inner, 'key', '<key>')} --freeze {cfg.train.freeze}` to see the counts."
        )


def build_optimizer(model, cfg: RunConfig):
    """AdamW/Adam/SGD over the trainable parameters, with weight decay off for
    biases and normalisation scales.

    Decaying a LayerNorm scale pulls it toward zero for no reason, and on a
    transformer backbone that is a measurable slowdown. Splitting the groups is cheap
    and removes the question entirely.
    """
    decay: List[nn.Parameter] = []
    no_decay: List[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)

    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": cfg.train.weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    if not groups:
        raise RuntimeError(
            "no trainable parameters. A freeze policy has probably frozen everything; "
            "run `symtrs doctor --model <key> --freeze <policy>` to see the counts."
        )

    if cfg.train.optim == "adamw":
        return torch.optim.AdamW(groups, lr=cfg.train.lr)
    if cfg.train.optim == "adam":
        return torch.optim.Adam(groups, lr=cfg.train.lr)
    return torch.optim.SGD(groups, lr=cfg.train.lr, momentum=0.9, nesterov=True)


def optimizer_steps_per_epoch(num_batches: int, grad_accum: int) -> int:
    """How many times `train_one_epoch` calls `optimizer.step()` for a loader of this length.

    A ceiling, not a floor division: the last, partial accumulation group is stepped too
    (`(step + 1) == len(loader)` in the loop), so 5 batches with `grad_accum=2` are 3 steps.
    The schedule is sized with this number and stepped next to every `optimizer.step()`, which
    is what keeps the two in agreement.
    """
    accumulation = max(1, int(grad_accum))
    return max(1, -(-int(num_batches) // accumulation))


def build_scheduler(optimizer, cfg: RunConfig, steps_per_epoch: int):
    """Warmup then cosine or polynomial decay, all in *optimiser steps*.

    The scheduler must be stepped once per optimiser step (`train_one_epoch(...,
    scheduler=...)` does it). It used to be stepped once per epoch while sized in steps, so
    the learning rate sat on the first few percent of the warmup ramp for the whole run.

    `steps_per_epoch` is `optimizer_steps_per_epoch(len(loader), grad_accum)`.

    Warmup is capped at half of the run. `warmup_epochs` is a per-run-length choice made for
    the full-length config; a short run (the validation queue trains 2 epochs with
    `warmup_epochs: 2`) would otherwise spend all of it ramping up and end at the peak without
    ever decaying, which is neither the configured schedule nor a useful short one. Capping
    keeps "warm up, then decay" true at any length, and leaves every run with
    `warmup_epochs <= epochs / 2` exactly as configured.
    """
    if cfg.train.sched == "none":
        return None

    steps_per_epoch = max(1, int(steps_per_epoch))
    total_steps = max(1, cfg.train.epochs * steps_per_epoch)
    warmup_steps = max(0, cfg.train.warmup_epochs * steps_per_epoch)
    if warmup_steps > total_steps // 2:
        capped = total_steps // 2
        print(
            f"[symtrs] train.warmup_epochs={cfg.train.warmup_epochs} is more than half of "
            f"train.epochs={cfg.train.epochs}; warming up for {capped} of {total_steps} steps "
            "instead so the schedule still decays"
        )
        warmup_steps = capped
    min_ratio = cfg.train.min_lr_ratio

    def factor(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            # Start at 1% rather than 0 so the very first step still moves.
            return max(0.01, (step + 1) / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        if cfg.train.sched == "cosine":
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_ratio + (1.0 - min_ratio) * cosine
        exponent = 0.9
        return min_ratio + (1.0 - min_ratio) * ((1.0 - progress) ** exponent)

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: factor(step) if step >= 0 else 1.0
    )


def checkpoint_metadata(model, cfg: RunConfig, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The self-describing block written into every checkpoint.

    This is what makes `autoloader("model.pt")` work: the architecture key, the kwargs
    it was built with, the depth protocol and the alignment convention are all here, so
    the model can be rebuilt and interpreted without the training YAML.
    """
    metadata: Dict[str, Any] = {
        "task": cfg.task,
        "model_key": cfg.model,
        "resolution": cfg.resolution,
        "seed": cfg.seed,
        "align": cfg.eval.align or getattr(model, "align", ""),
        "model_kwargs": dict(cfg.model_kwargs),
        "config": cfg.to_dict(),
    }
    protocol = getattr(model, "protocol", "")
    if protocol:
        metadata["protocol"] = protocol
        metadata["default_align"] = getattr(model, "default_align", "")
        # The weights live in this space; a reloaded checkpoint must not be read in another.
        metadata["output_space"] = getattr(model, "output_space", "depth")
    if cfg.task == "segmentation":
        metadata["num_classes"] = cfg.data.num_classes
        metadata["ignore_index"] = cfg.data.ignore_index
        metadata["label_step"] = cfg.data.label_step
        metadata["class_names"] = getattr(model, "class_names", None)
    if extra:
        metadata.update(extra)
    return metadata


# ------------------------------------------------------------------- run directory


def prepare_run_dir(cfg: RunConfig) -> Path:
    run_dir = cfg.resolved_out_dir
    for sub in ("weights", "preview", "logs"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    if is_main_process():
        cfg.to_yaml(str(run_dir / "config.yaml"))
    return run_dir


def save_run(
    model,
    cfg: RunConfig,
    run_dir: Path,
    state: Dict[str, Any],
    tag: str,
    metrics: Optional[Dict[str, float]] = None,
) -> None:
    """Write `weights/<tag>.pt` with the full metadata block."""
    extra: Dict[str, Any] = {"tag": tag, "metrics": metrics or {}}
    save_checkpoint(
        str(run_dir / "weights" / f"{tag}.pt"),
        state,
        checkpoint_metadata(model, cfg, extra),
    )


# ------------------------------------------------------------------------ the run


def train(
    cfg: RunConfig,
    model=None,
    mode: str = "finetune",
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Train, validate, checkpoint, then evaluate on the test split.

    Args:
        cfg: a validated run config.
        model: an already-built model, or None to build one from `cfg.model`.
        mode: "finetune" trains first; "zero_shot" skips training entirely and just
            evaluates the released weights, which is the baseline half of the run
            matrix and shares this whole code path.
        device: overrides `train.device`.

    Returns:
        `{"run_dir", "best", "monitor", "epochs", "val", "test", "history"}`.
    """
    if mode not in MODES:
        raise ValueError(f"unknown run mode {mode!r}; expected one of {MODES}")
    cfg.validate()

    if cfg.task == _SUPERRESOLUTION_TASK:
        from symtrs.model.superres import run_super_resolution

        return run_super_resolution(cfg, mode=mode)

    # Domain adaptation has no model of its own to build: the network belongs to the
    # upstream CycleGAN/pix2pix project, which is driven as a subprocess.
    if cfg.task == "domain_adapt":
        from symtrs.model.domain import run_domain_adaptation

        return run_domain_adaptation(cfg, mode=mode)

    # Shadow removal likewise: the networks are upstream's, trained by a worker process.
    if cfg.task == "shadow_removal":
        from symtrs.model.shadow import run_shadow_removal

        return run_shadow_removal(cfg, mode=mode)

    # A configured multi-GPU run that has not been launched by torchrun yet has to
    # spawn its own processes; a run already inside a distributed context must not.
    world_size = world_size_from_env()
    if cfg.train.gpus > 1 and world_size == 1 and mode == "finetune":
        return _spawn(cfg, model, mode, device)

    return _run(cfg, model=model, mode=mode, device=device, world_size=world_size)


def _spawn(cfg: RunConfig, model, mode: str, device: Optional[str]) -> Dict[str, Any]:
    import torch.multiprocessing as mp

    print(f"[symtrs] launching {cfg.train.gpus} processes with mp.spawn")
    mp.spawn(
        _worker,
        args=(cfg.train.gpus, cfg, model, mode, device),
        nprocs=cfg.train.gpus,
        join=True,
    )
    run_dir = cfg.resolved_out_dir
    return _read_summary(run_dir)


def _worker(rank: int, world_size: int, cfg: RunConfig, model, mode: str, device: Optional[str]) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29511")
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    try:
        _run(cfg, model=model, mode=mode, device=device, world_size=world_size)
    finally:
        cleanup_ddp()


def _run(
    cfg: RunConfig,
    model,
    mode: str,
    device: Optional[str],
    world_size: int,
) -> Dict[str, Any]:
    from symtrs.data.factory import from_config
    from symtrs.engine.loop import (
        evaluate,
        format_metrics,
        make_grad_scaler,
        train_one_epoch,
    )

    distributed = world_size > 1
    rank = setup_ddp() if distributed else 0
    requested = device or cfg.train.device
    if distributed:
        torch.cuda.set_device(rank)
        resolved = torch.device("cuda", rank)
    else:
        resolved = resolve_device(requested)
    set_seed(cfg.seed + rank)

    run_dir = prepare_run_dir(cfg)
    if is_main_process():
        print(f"[symtrs] {cfg.task} / {cfg.model} / {cfg.resolution}px / mode={mode}")
        print(f"[symtrs] device: {describe_device(resolved)}")
        print(f"[symtrs] run dir: {run_dir}")

    if model is None:
        from symtrs.model.wrapper import build_from_config

        model = build_from_config(cfg)
    apply_run_freeze(model, cfg)
    model.to(resolved)
    if distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    if is_main_process():
        inner = getattr(model, "module", model)
        print(f"[symtrs] {inner.describe()}")
        if cfg.train.freeze:
            counts = sum(p.numel() for p in inner.parameters() if p.requires_grad)
            print(f"[symtrs] freeze={cfg.train.freeze!r} -> {counts / 1e6:.1f}M trainable")

    val_loader = from_config(cfg, train=False)
    test_loader = from_config(cfg, split=cfg.data.test_split, train=False)

    history: List[Dict[str, Any]] = []
    monitor, direction = resolve_monitor(cfg)
    best: Optional[float] = None
    best_epoch = -1
    # Epochs whose monitored value was NaN/inf. They are kept out of "best" and reported, so
    # a diverged run is visible in summary.json instead of looking like a run that merely
    # saved a checkpoint with rmse=nan.
    non_finite_epochs: List[int] = []

    if mode == "finetune":
        train_loader = from_config(cfg, train=True)
        if is_main_process():
            print(
                f"[symtrs] {len(train_loader)} train batches, {len(val_loader)} val batches, "
                f"batch_size={cfg.train.batch_size}"
            )

        optimizer = build_optimizer(model, cfg)
        steps_per_epoch = optimizer_steps_per_epoch(len(train_loader), cfg.train.grad_accum)
        scheduler = build_scheduler(optimizer, cfg, steps_per_epoch)
        scaler = make_grad_scaler(resolved, cfg.train.amp)

        start_epoch = 0
        if cfg.train.resume:
            start_epoch, restored = _resume(
                cfg, model, optimizer, scheduler, run_dir, steps_per_epoch, monitor
            )
            # Without these, the first epoch after a resume starts with `best=None` and
            # overwrites `best.pt` even when it is worse than the epoch the crash interrupted
            # (`run_queue.py` resumes on every crash or timeout), and `summary.json` loses
            # the epochs before it.
            best = restored["best"]
            best_epoch = restored["best_epoch"]
            history = restored["history"]
            non_finite_epochs = restored["non_finite_epochs"]
            if is_main_process():
                print(
                    f"[symtrs] resumed from {cfg.train.resume} at epoch {start_epoch} "
                    f"(best {monitor}={best} at epoch {best_epoch})"
                )

        for epoch in range(start_epoch, cfg.train.epochs):
            started = time.time()
            train_metrics = train_one_epoch(
                model, train_loader, optimizer, scaler, cfg, resolved, epoch=epoch,
                scheduler=scheduler,
            )

            val_metrics = evaluate(model, val_loader, cfg, resolved)
            lr = float(optimizer.param_groups[0]["lr"])
            elapsed = time.time() - started

            row: Dict[str, Any] = {"epoch": epoch, "lr": lr, "seconds": round(elapsed, 1)}
            row.update({f"train_{k}": v for k, v in train_metrics.items()})
            row.update({f"val_{k}": v for k, v in val_metrics.items()})
            history.append(row)
            if is_main_process():
                append_csv_row(str(run_dir / "metrics.csv"), row)

            if is_main_process():
                print(f"[symtrs] epoch {epoch} ({elapsed:.1f}s) lr={lr:.2e}")
                print(f"[symtrs]   train {format_metrics(train_metrics)}")
                print(f"[symtrs]   val   {format_metrics(val_metrics)}")

                value = float(val_metrics.get(monitor, float("nan")))
                if not math.isfinite(value):
                    non_finite_epochs.append(epoch)
                    print(
                        f"[symtrs]   warning: val {monitor}={value} is not finite; epoch {epoch} "
                        "cannot be the best checkpoint (weights/last.pt is still written)"
                    )
                improved = is_improvement(value, best, direction)
                if improved:
                    best = value
                    best_epoch = epoch

                # `best` and the history go into the payload *after* this epoch is decided, so
                # that a resume from either checkpoint knows what it is up against.
                state = _state(
                    model, optimizer, scheduler, epoch, val_metrics,
                    best=best, best_epoch=best_epoch, history=history,
                    non_finite_epochs=non_finite_epochs,
                )
                save_run(model, cfg, run_dir, state, "last", val_metrics)
                if improved:
                    save_run(model, cfg, run_dir, state, "best", val_metrics)
                    print(f"[symtrs]   new best {monitor}={value:.4f} -> weights/best.pt")

                if cfg.train.save_samples_every and (epoch + 1) % cfg.train.save_samples_every == 0:
                    _save_previews(model, val_loader, cfg, resolved, run_dir, epoch)

                _save_curves(history, run_dir)

            barrier()

    # The test pass scores the checkpoint the summary reports as "best", not whatever the
    # last epoch left in memory: `best` and `test` sit side by side in summary.json, and the
    # other tasks (shadow removal) score best.pt, so a number from the last-epoch weights
    # made the protocols differ between tasks. Which weights were scored is recorded rather
    # than implied.
    test_weights = _load_test_weights(model, run_dir, mode, best, resolved)
    test_align = ""
    test_metrics = evaluate(
        model, test_loader, cfg, resolved,
        align=test_align,
        num_visualizations=cfg.eval.num_visualizations,
        vis_dir=str(run_dir / "preview"),
        vis_prefix="test",
    )

    summary = {
        "run_dir": str(run_dir),
        "task": cfg.task,
        "model": cfg.model,
        "resolution": cfg.resolution,
        "mode": mode,
        "monitor": monitor,
        "best": best,
        "best_epoch": best_epoch,
        "epochs": cfg.train.epochs if mode == "finetune" else 0,
        "test_weights": test_weights,
        "test": test_metrics,
        "history": history,
    }
    if non_finite_epochs:
        summary["non_finite_monitor_epochs"] = non_finite_epochs
    if is_main_process():
        print(f"[symtrs] test {format_metrics(test_metrics)}")
        # Non-finite floats are written as null: a bare NaN is not JSON, and one diverged
        # run must not make the whole results directory unreadable by `jq` or a browser.
        write_json(run_dir / "summary.json", summary)
        if mode == "finetune" and best is not None:
            print(f"[symtrs] best {monitor}={best:.4f} at epoch {best_epoch}")
        elif mode == "finetune":
            print(
                f"[symtrs] no epoch produced a finite val {monitor}: no weights/best.pt was "
                "written and \"best\" is null in summary.json. The test numbers above come "
                "from the last-epoch weights (test_weights=\"last\"); treat them as a failed run."
            )
    return summary


def _state(
    model,
    optimizer,
    scheduler,
    epoch: int,
    metrics: Dict[str, float],
    best: Optional[float] = None,
    best_epoch: int = -1,
    history: Optional[List[Dict[str, Any]]] = None,
    non_finite_epochs: Optional[List[int]] = None,
) -> Dict[str, Any]:
    inner = getattr(model, "module", model)
    return {
        "model": inner.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        # The scheduler is stepped per optimiser step. A checkpoint from before that was
        # stepped per epoch, and its position means something else; `_resume` tells them apart.
        "scheduler_unit": "optimizer_step",
        "epoch": epoch,
        "metrics": metrics,
        "best": best,
        "best_epoch": best_epoch,
        "history": list(history or []),
        "non_finite_epochs": list(non_finite_epochs or []),
    }


def _load_test_weights(model, run_dir: Path, mode: str, best: Optional[float], device) -> str:
    """Put the weights the test pass should score into `model`; return which they are.

    `"best"` after a training run with a finite best, `"last"` when no epoch had a finite
    monitor (there is no best.pt then, and the summary says so), `"released"` for zero-shot
    where nothing was trained. In a distributed run every rank reaches this after the
    epoch loop's barrier, so best.pt is complete on disk.
    """
    if mode != "finetune":
        return "released"
    path = run_dir / "weights" / "best.pt"
    if best is None or not path.is_file():
        return "last"
    payload = load_checkpoint(str(path), map_location="cpu")
    inner = getattr(model, "module", model)
    inner.load_state_dict(payload["model"])
    inner.to(device)
    return "best"


def _fast_forward_scheduler(scheduler, steps: int) -> None:
    """Put a `LambdaLR` at `steps` optimiser steps without replaying them one by one."""
    scheduler.last_epoch = steps
    for group, base, lambda_ in zip(
        scheduler.optimizer.param_groups, scheduler.base_lrs, scheduler.lr_lambdas
    ):
        group["lr"] = base * lambda_(steps)
    scheduler._last_lr = [group["lr"] for group in scheduler.optimizer.param_groups]


def _resume(
    cfg: RunConfig,
    model,
    optimizer,
    scheduler,
    run_dir: Path,
    steps_per_epoch: int,
    monitor: str,
):
    """Restore a run from a checkpoint. Returns `(start_epoch, restored_run_state)`.

    The run state is `best`, `best_epoch`, `history` and `non_finite_epochs`, taken from the
    checkpoint. Checkpoints written before they were stored fall back to what is on disk
    beside them: `best.pt`'s own metrics for the best value, `metrics.csv` for the history.
    """
    from symtrs.common.checkpoint import resolve_checkpoint_path
    from symtrs.common.table import read_csv_rows, row_as_floats

    path = resolve_checkpoint_path(cfg.train.resume)
    if path is None:
        raise FileNotFoundError(f"nothing to resume from at {cfg.train.resume}")
    payload = load_checkpoint(path, map_location="cpu")
    inner = getattr(model, "module", model)
    missing, unexpected = inner.load_state_dict(payload["model"], strict=False)
    if missing or unexpected:
        print(
            f"[symtrs] resume note: {len(missing)} missing and {len(unexpected)} unexpected "
            "keys; the architecture may differ from the one that was saved"
        )
    if payload.get("optimizer") and optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    start_epoch = int(payload.get("epoch", 0)) + 1
    if scheduler is not None:
        if payload.get("scheduler") and payload.get("scheduler_unit") == "optimizer_step":
            scheduler.load_state_dict(payload["scheduler"])
        else:
            # A per-epoch position (or none) is meaningless to a per-step schedule: loading
            # it would restart the warmup. Recompute where the run would be instead.
            _fast_forward_scheduler(scheduler, start_epoch * steps_per_epoch)

    best = payload.get("best")
    best_epoch = payload.get("best_epoch", -1)
    if "best" not in payload:
        best, best_epoch = None, -1
        best_path = Path(path).with_name("best.pt")
        if best_path.is_file():
            best_payload = load_checkpoint(str(best_path), map_location="cpu")
            value = (best_payload.get("metrics") or {}).get(monitor)
            if isinstance(value, (int, float)) and math.isfinite(value):
                best, best_epoch = float(value), int(best_payload.get("epoch", -1))
    history = payload.get("history")
    if history is None:
        history = [
            row_as_floats(row)
            for row in read_csv_rows(str(run_dir / "metrics.csv"))
            if row.get("epoch", "") not in ("", "-1")
        ]
    # Resuming from an earlier checkpoint than the last one written drops the later epochs:
    # they are about to be trained again.
    history = [row for row in history if float(row.get("epoch", -1)) < start_epoch]
    non_finite = [e for e in payload.get("non_finite_epochs", []) if e < start_epoch]
    return start_epoch, {
        "best": best if best is None or math.isfinite(best) else None,
        "best_epoch": int(best_epoch),
        "history": history,
        "non_finite_epochs": non_finite,
    }


def _save_previews(model, loader, cfg: RunConfig, device, run_dir: Path, epoch: int) -> None:
    """A handful of qualitative panels per epoch, so a collapsed prediction is visible
    in the run directory rather than only in a metric that moved by 0.001."""
    # Imported here rather than taken from `train`'s local scope: this is a module-level
    # function, and reaching for a name that only exists inside the caller raises
    # NameError - which the handler below then reports as "preview skipped", turning a
    # wiring mistake into a quietly preview-less training run.
    from symtrs.engine.loop import evaluate

    count = cfg.eval.num_visualizations or 2
    try:
        evaluate(
            model, loader, cfg, device,
            num_visualizations=count,
            vis_dir=str(run_dir / "preview"),
            vis_prefix=f"epoch{epoch:03d}",
        )
    except Exception as exc:  # visuals must never take down a training run
        print(f"[symtrs] preview generation skipped: {type(exc).__name__}: {exc}")
    finally:
        model.train()


#: Set once a plotting failure has been reported, so a 40-epoch run says it 40 times less.
_PLOT_FAILURE_REPORTED = False


def _save_curves(history, run_dir: Path) -> None:
    """Training curves, into the run's `logs/`.

    Guarded for the same reason the previews are: matplotlib is imported lazily and can be
    absent or unable to start a backend on a headless machine, and this runs at the end of
    every epoch - after the checkpoints are saved but before the test evaluation. Letting it
    raise means a finished training run never reports a test number.
    """
    global _PLOT_FAILURE_REPORTED

    from symtrs.common.plots import save_curves

    try:
        save_curves(history, str(run_dir / "logs"))
    except Exception as exc:  # noqa: BLE001 - a plot is never a reason to lose a run
        if not _PLOT_FAILURE_REPORTED:
            _PLOT_FAILURE_REPORTED = True
            print(
                f"[symtrs] training curves skipped: {type(exc).__name__}: {exc}\n"
                "[symtrs]   install matplotlib for curves; checkpoints and metrics are "
                "unaffected. This is reported once."
            )


def _read_summary(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "summary.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"run_dir": str(run_dir), "note": "training finished; summary.json was not written"}


# --------------------------------------------------------------- zero-shot / eval


def evaluate_run(
    cfg: RunConfig,
    model=None,
    split: Optional[str] = None,
    checkpoint: str = "",
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate without training: released weights (zero-shot) or a trained `.pt`.

    This is the other half of the run matrix. It shares `evaluate` with the training
    loop, so a zero-shot number and a finetuned number are always measured the same way.
    """
    from symtrs.data.factory import from_config
    from symtrs.engine.loop import evaluate, format_metrics

    cfg.validate()

    if cfg.task == _SUPERRESOLUTION_TASK:
        # "eval", not "zero_shot": the script has no released SyMTRS weights to run, so
        # this re-scores the run directory's own weights/best.pt on the test split. Same
        # shape as domain adaptation's "score" mode.
        from symtrs.model.superres import run_super_resolution

        return run_super_resolution(cfg, mode="eval")

    if cfg.task == "domain_adapt":
        from symtrs.model.domain import run_domain_adaptation

        # "score" rather than "zero_shot": there are no released adaptation weights, so
        # this re-generates and re-measures whatever the run directory already holds.
        return run_domain_adaptation(cfg, mode="score")

    if cfg.task == "shadow_removal":
        from symtrs.model.shadow import run_shadow_removal

        # Re-predicts and re-scores the run's own weights (or `--checkpoint`) on one split.
        return run_shadow_removal(cfg, mode="score", split=split, checkpoint=checkpoint)

    resolved = resolve_device(device or cfg.train.device)
    run_dir = cfg.resolved_out_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    if model is None:
        if checkpoint:
            from symtrs.model.wrapper import autoloader

            model = autoloader(checkpoint).net
        else:
            from symtrs.model.wrapper import build_from_config

            model = build_from_config(cfg)
    model.to(resolved)

    loader = from_config(cfg, split=split, train=False)
    if is_main_process():
        print(f"[symtrs] evaluating {cfg.task} / {cfg.model} on split {split or cfg.data.test_split}")

    metrics = evaluate(
        model, loader, cfg, resolved,
        align=cfg.eval.align,
        num_visualizations=cfg.eval.num_visualizations,
        vis_dir=str(run_dir / "preview"),
        vis_prefix="eval",
    )
    if is_main_process():
        print(f"[symtrs] {format_metrics(metrics)}")
        append_csv_row(
            str(run_dir / "metrics.csv"),
            {"epoch": -1, "split": split or cfg.data.test_split, **metrics},
        )
    return {"run_dir": str(run_dir), "split": split or cfg.data.test_split, "metrics": metrics}


def run(cfg: RunConfig, mode: str = "finetune", **kwargs) -> Dict[str, Any]:
    """Entry point the commands and the SDK both call."""
    if mode == "zero_shot":
        return evaluate_run(cfg, **kwargs)
    return train(cfg, mode=mode, **kwargs)


__all__ = [
    "MONITORS",
    "MONITOR_DIRECTIONS",
    "apply_run_freeze",
    "build_optimizer",
    "build_scheduler",
    "checkpoint_metadata",
    "evaluate_run",
    "is_improvement",
    "optimizer_steps_per_epoch",
    "prepare_run_dir",
    "resolve_monitor",
    "run",
    "save_run",
    "train",
]
