"""One training epoch, one evaluation pass.

This is the task-agnostic half of the engine. It knows the two batch shapes, the two
metric accumulations, and nothing else: no optimizer setup, no checkpointing, no
distributed launch. Keeping it that small is what makes the loop readable, and it is
also the only part of the engine that a test can exercise with a stub model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F

from symtrs.common.meters import MetricTracker
from symtrs.config import ConfigError, RunConfig

try:  # torch >= 2.0
    from torch.amp import GradScaler, autocast as _autocast
except ImportError:  # pragma: no cover - torch 1.x
    from torch.cuda.amp import GradScaler, autocast as _autocast


def autocast_context(device: torch.device, enabled: bool):
    """Mixed precision, but only where it is actually supported.

    `enabled=False` on CPU keeps one code path: the caller never has to branch on the
    device, and nothing silently runs fp16 on a CPU tensor.
    """
    device_type = "cuda" if device.type == "cuda" else "cpu"
    return _autocast(device_type=device_type, enabled=bool(enabled) and device.type == "cuda")


def make_grad_scaler(device: torch.device, enabled: bool) -> GradScaler:
    try:
        return GradScaler(device="cuda", enabled=bool(enabled) and device.type == "cuda")
    except TypeError:  # pragma: no cover - torch 1.x signature
        return GradScaler(enabled=bool(enabled) and device.type == "cuda")


# --------------------------------------------------------------------- inference


def _resize(prediction: torch.Tensor, size) -> torch.Tensor:
    """Bilinear resize, used only when a model's output does not already match."""
    if prediction.dim() == 3:
        prediction = prediction.unsqueeze(1)
    if tuple(prediction.shape[-2:]) == tuple(size):
        return prediction
    return F.interpolate(prediction.float(), size=tuple(size), mode="bilinear", align_corners=False)


def prepare_input(model, image: torch.Tensor) -> torch.Tensor:
    """`[B, 3, H, W]` in `[0, 1]` to whatever the network wants.

    This is the **only** place a raw image becomes model input. Preprocessing and the
    pad-to-a-multiple fixup both happen here, and `forward_at_size` is documented to
    expect the result - so an adapter never has to remember to normalise, and no path
    through the engine can feed a network an unnormalised image.
    """
    prepared = model.preprocess_image(image)
    pad = getattr(model, "_pad_to_multiple", None)
    if callable(pad):
        prepared = pad(prepared)
    return prepared


def forward_native(
    model,
    image: torch.Tensor,
    tile_size: int = 0,
    tile_overlap: int = 0,
    stats=None,
    tile_batch: int = 1,
) -> torch.Tensor:
    """Raw prediction for a whole batch, at the input's spatial size.

    Takes a raw `[0, 1]` image; preprocessing is applied here. Returns the model's *own*
    output space - disparity for a relative depth model, logits for segmentation -
    because that is the space the loss and any blending must work in. `to_metres` or
    `argmax` is applied by the caller.

    The final resize is here, and not left to the adapter, because this is the last frame in
    which the *raw* image's size is still in scope. `prepare_input` may change that size: a
    MiDaS transform resizes whatever it is given to 256 or 384, and `_pad_to_multiple` rounds
    the sides up. An adapter handed the preprocessed tensor can only restore its own input,
    which is the transformed size - so a 2048px benchmark image came back as a 384px
    prediction and was then compared against a 2048px ground truth, and `compute_depth_metrics`
    refused it with "prediction shape (1, 1, 384, 384) does not match target (1, 1, 2048,
    2048)". Both the training step and the evaluation pass reach the network through this
    function, so both were affected at every resolution.
    """
    # Relative depth is only defined up to a scale and shift per forward pass, so tiles of
    # one such image must be brought onto each other before they are blended (see
    # `engine.tiling`). Metric depth and logits are absolute and are blended as they are.
    # The kind of matching follows the model's OUTPUT SPACE, not just its protocol: disparity gets
    # scale+shift, depth-up-to-scale gets scale only (a shift could not be undone by the `scale`
    # alignment), metric depth and logits get none. `stats` collects the fit fallbacks.
    if tile_size and tile_size > 0 and max(image.shape[-2:]) > tile_size:
        from symtrs.engine.tiling import match_mode, predict_tiled

        def forward_fn(tile: torch.Tensor) -> torch.Tensor:
            return _restore_size(model, model.forward(prepare_input(model, tile)), tile)

        stitched = predict_tiled(
            image, forward_fn, tile_size, tile_overlap, match=match_mode(model), stats=stats,
            tile_batch=tile_batch,
        )
        return _resize(stitched, image.shape[-2:])

    prepared = prepare_input(model, image)
    method = getattr(model, "forward_at_size", None)
    prediction = method(prepared) if callable(method) else model.forward(prepared)
    return _restore_size(model, prediction, image, prepared)


def _restore_size(model, prediction: torch.Tensor, raw: torch.Tensor, prepared=None) -> torch.Tensor:
    """Bring a prediction back onto the raw image's pixel grid.

    When the only thing `prepare_input` did to the size was pad bottom/right up to the
    model's multiple, the prediction is the padded size and the padding is *cropped* off:
    resizing it instead stretches the whole field by up to a few pixels (DA3 pads a 512px
    tile to 518), which shifts every pixel against the ground truth. Any other size change
    (a MiDaS transform resizing to 384, a head at 1/4 resolution) is a genuine resample and
    gets the bilinear resize.
    """
    height, width = raw.shape[-2:]
    multiple = int(getattr(model, "size_multiple", 1) or 1)
    padded = (height + (-height) % multiple, width + (-width) % multiple)
    if (
        tuple(prediction.shape[-2:]) == padded
        and padded != (height, width)
        and (prepared is None or tuple(prepared.shape[-2:]) == padded)
    ):
        if prediction.dim() == 3:
            prediction = prediction.unsqueeze(1)
        return prediction[..., :height, :width]
    return _resize(prediction, (height, width))


# ------------------------------------------------------------------ training step


#: The words in torch's own BatchNorm batch-size check. Matched rather than catching every
#: `ValueError`, so nothing else is reinterpreted.
_BATCHNORM_BATCH_SIZE = "more than 1 value per channel"


def _as_batch_size_error(exc: BaseException, model, batch_size: int) -> BaseException:
    """Re-raise torch's BatchNorm batch-size failure as something the reader can act on.

    torchvision's DeepLabV3 puts a global average pooling in front of a BatchNorm inside its
    ASPP, so that BatchNorm always sees a 1x1 feature map and has only the batch axis left to
    compute its statistics from. A training batch of one image is therefore not valid for it -
    not something a flag can talk it out of, and not something `--grad-accum` helps with,
    because accumulation repeats forwards rather than widening one. `--freeze` does not help
    either: the pooling and its BatchNorm are in the head, so they stay in training mode.

    Left alone, the reader gets `ValueError: Expected more than 1 value per channel when
    training, got input size torch.Size([1, 256, 1, 1])`, which names neither the batch size
    nor the model, and arrives as a one-line `symtrs ValueError: ...`.

    Batch size 1 is fine for *evaluation*: BatchNorm uses its running statistics in eval mode
    and never runs this check.
    """
    if _BATCHNORM_BATCH_SIZE not in str(exc) or batch_size != 1:
        return exc
    # `ConfigError`, not `RuntimeError`: the batch size is the reader's to change, and the
    # CLI reports `ConfigError` as a one-line message while anything else is announced as an
    # unexpected internal bug and sent off to a bug report.
    return ConfigError(
        f"{type(model).__name__} cannot be trained with a batch of one image: a BatchNorm "
        "inside it is handed a feature map with no spatial extent (torchvision's DeepLabV3 "
        "ASPP does this by design), so the batch axis is all it has to compute its statistics "
        "from. Raise --batch-size to at least 2. Neither --grad-accum nor --freeze avoids "
        "this: the first repeats forwards instead of widening one, and the offending layers "
        "are in the head, not the encoder. Models without the restriction: unet_small and "
        "segformer_b0/b1/b2."
    )


def _training_step(model, batch: Mapping[str, Any], device: torch.device, task: str):
    """Return `(loss, component_dict)` for one batch. Shared by both training loops."""
    image = batch["image"].to(device, non_blocking=True)

    try:
        if task == "depth":
            ground_truth = batch["depth"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)
            prediction = forward_native(model, image, tile_size=0)
            target = model.prepare_target(ground_truth, valid)
            return model.compute_loss(prediction, target)

        if task == "segmentation":
            mask = batch["mask"].to(device, non_blocking=True)
            logits = forward_native(model, image, tile_size=0)
            return model.compute_loss(logits, mask)
    except ValueError as exc:
        raise _as_batch_size_error(exc, model, int(image.shape[0])) from exc

    raise ValueError(f"task {task!r} has no training step; it is not a supervised task")


class Trainee:
    """One model's optimiser state inside a training pass.

    `train_one_epoch` drives exactly one; `train_one_epoch_many` drives several over the
    *same* batch. That is how a co-trained group pays for a 2048px sample once: decoding a
    2048 PNG + EXR is 0.21 s of CPU, while one forward+backward of the cheapest depth model
    at a 384 crop is 0.15 s of GPU on an A6000, so several models can consume the same
    decoded batch before the GPU catches up with the loader. The model's own loss,
    optimiser, scaler and scheduler stay private to it.
    """

    def __init__(self, model, optimizer, scaler, cfg: RunConfig, label: str = "", scheduler=None,
                 windows_per_batch: int = 1):
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.scheduler = scheduler
        self.cfg = cfg
        self.label = label
        self.tracker = MetricTracker()
        self.accumulation = max(1, int(cfg.train.grad_accum))
        # How many of this member's optimiser windows one shared batch holds. 1 for every
        # member whose effective batch is the group's; 2 for a member whose effective batch
        # is half of it (DeepLabV3 beside SegFormer), which then steps twice per shared batch.
        self.windows_per_batch = max(1, int(windows_per_batch))
        self.optimizer.zero_grad(set_to_none=True)

    @property
    def prefix(self) -> str:
        return f"[{self.label}] " if self.label else ""

    def split(self, batch: Mapping[str, Any]) -> Tuple[Tuple[Mapping[str, Any], bool], ...]:
        """Cut one shared batch into this member's micro-batches, each with "closes a window".

        Only a co-trained group calls this. The group's loader is built at the *largest*
        effective batch among its members (`batch_size * grad_accum`), so one shared batch is
        `windows_per_batch` whole accumulation windows for this member. It is cut into those
        windows first, in order, and each window into the member's `grad_accum`
        micro-batches; the last micro-batch of each window carries True, and that is where
        the optimiser steps. A member asking for 4 images over 2 accumulation steps in a group
        whose window is 16 therefore takes two 8-sample windows, four 4-image forwards and two
        optimiser steps per shared batch - the samples per step its config describes, and the
        whole shared batch is consumed exactly once.

        The cuts are as even as the batch allows, so a short last batch still yields the
        member's full number of windows and micro-batches rather than a ragged one.
        """
        out: List[Tuple[Mapping[str, Any], bool]] = []
        for window in _even_cuts(batch, self.windows_per_batch):
            micro = _even_cuts(window, self.accumulation)
            out.extend((chunk, index == len(micro) - 1) for index, chunk in enumerate(micro))
        return tuple(out)

    def step(self, batch: Mapping[str, Any], device: torch.device, epoch: int, step: int,
             batch_count: int, close_window: Optional[bool] = None, log: bool = True) -> None:
        """One forward+backward on `batch`, and an optimiser step when the window closes.

        `close_window` is for the group driver: there, `step` counts *shared* batches, one
        of which is already a whole accumulation window for this member, so the window is
        closed explicitly rather than inferred from the step index. `log=False` suppresses
        the progress line for a micro-batch in the middle of a window, which would otherwise
        print the same step number once per micro-batch.
        """
        cfg = self.cfg
        with autocast_context(device, cfg.train.amp):
            loss, components = _training_step(self.model, batch, device, cfg.task)
            # Divide before backward so the accumulated gradient is a mean over the
            # accumulation window rather than a sum that grows with it.
            scaled_loss = loss / self.accumulation

        if not torch.isfinite(scaled_loss):
            # A non-finite loss means NaN weights if it is allowed to backpropagate,
            # and NaN weights never recover. Skip the step and say so loudly.
            print(f"[symtrs] {self.prefix}warning: non-finite loss at epoch {epoch} step {step}; "
                  "skipping batch")
            self.optimizer.zero_grad(set_to_none=True)
            return

        if self.scaler is not None and self.scaler.is_enabled():
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        batch_size = int(batch["image"].shape[0])
        self.tracker.update("loss", float(loss.detach()), n=batch_size)
        self.tracker.update_many(components, n=batch_size)

        # A group passes `close_window` explicitly and it is the only signal: `step` counts
        # *shared* batches there, so the solo rule below fired on every odd shared batch for a
        # 2-way-accumulating member and stepped its optimiser (and schedule) mid-window - six
        # steps for four windows. None is the solo shape, where the index decides.
        if close_window is None:
            close_window = (step + 1) % self.accumulation == 0 or (step + 1) == batch_count
        if close_window:
            if cfg.train.grad_clip and cfg.train.grad_clip > 0:
                if self.scaler is not None and self.scaler.is_enabled():
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], cfg.train.grad_clip
                )
            if self.scaler is not None and self.scaler.is_enabled():
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            if self.scheduler is not None:
                self.scheduler.step()

        if log and cfg.train.log_every and (step + 1) % cfg.train.log_every == 0:
            print(
                f"[symtrs] {self.prefix}epoch {epoch} step {step + 1}/{batch_count} "
                f"loss={self.tracker.get('loss'):.4f}"
            )


def _even_cuts(batch: Mapping[str, Any], parts: int) -> List[Mapping[str, Any]]:
    """`batch` cut into `parts` consecutive, near-equal pieces (fewer if it is smaller)."""
    size = int(batch["image"].shape[0])
    if parts <= 1 or size <= 1:
        return [batch]
    parts = min(parts, size)
    base, extra = divmod(size, parts)
    out: List[Mapping[str, Any]] = []
    start = 0
    for index in range(parts):
        stop = start + base + (1 if index < extra else 0)
        out.append(_slice_batch(batch, start, stop, size))
        start = stop
    return out


def _slice_batch(batch: Mapping[str, Any], start: int, stop: int, size: int) -> Dict[str, Any]:
    """`batch` restricted to samples `[start, stop)`.

    Tensors are cut on the batch axis. The datasets also carry per-sample metadata as a
    list of strings (`image_path`, `depth_path`, ...); a list is cut only when its length is
    the batch size, so a scalar-ish field is passed through untouched. Nothing in a training
    step reads those paths, but a member must still see its own samples' if anything later
    does.
    """
    out: Dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value[start:stop]
        elif isinstance(value, (list, tuple)) and len(value) == size:
            out[key] = list(value[start:stop])
        else:
            out[key] = value
    return out


def train_one_epoch_many(
    trainees: List[Trainee],
    loader,
    device: torch.device,
    epoch: int = 0,
    max_steps: int = 0,
    windows: bool = False,
) -> List[Dict[str, float]]:
    """One pass over `loader`, stepping every trainee on every batch.

    The loader is read once. Each trainee sees exactly the same images, crops and flips
    (see `Trainee` for why that is the point), and returns its own averaged metrics. The
    tensors are shared, so a model's preprocessing must not write into them in place; no
    adapter does, and a new one must not.

    `windows=True` is the co-trained group's shape: the loader yields the group's largest
    accumulation window per batch (`batch_size * grad_accum`), and each trainee cuts that
    into its own windows and micro-batches (`Trainee.split`), stepping its optimiser once
    per window of its own. `windows=False` is the solo shape, where a loader batch *is* a
    micro-batch and the optimiser steps every `grad_accum` of them. The arithmetic -
    scaling, clipping, stepping, logging - is `Trainee.step` in both cases; only the
    batching in front of it differs, because that is what the two loaders disagree about.
    """
    for trainee in trainees:
        trainee.model.train()
        # A trainee outlives the pass in a co-trained group (it carries the optimiser and
        # scheduler across epochs), so the running averages are cleared here, not in the
        # constructor: each epoch's row is that epoch's mean.
        trainee.tracker.reset()

    batch_count = len(loader)
    for step, batch in enumerate(loader):
        if max_steps and step >= max_steps:
            break
        for trainee in trainees:
            if not windows:
                trainee.step(batch, device, epoch, step, batch_count)
                continue
            pieces = trainee.split(batch)
            last = len(pieces) - 1
            for index, (chunk, closes) in enumerate(pieces):
                trainee.step(chunk, device, epoch, step, batch_count,
                             close_window=closes, log=index == last)
    return [trainee.tracker.averages() for trainee in trainees]


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    cfg: RunConfig,
    device: torch.device,
    epoch: int = 0,
    max_steps: int = 0,
    scheduler=None,
) -> Dict[str, float]:
    """Train for one pass over `loader` and return the averaged metrics.

    `scheduler`, when given, is stepped once per *optimiser* step, here and nowhere else.
    `trainer.build_scheduler` sizes its warmup and decay in optimiser steps, and the trainer
    used to step it once per epoch instead: the learning rate then crawled through the first
    `epochs` points of a schedule meant for `epochs * steps_per_epoch`, so a 2-epoch run
    trained at 4-8% of the configured rate and no run ever decayed. Stepping next to
    `optimizer.step()` keeps the two counts equal by construction - including for the last,
    partial accumulation group, and for a step the loss-scaler skipped, which still
    advances the schedule so the epoch ends where `build_scheduler` expects.
    """
    trainee = Trainee(model, optimizer, scaler, cfg, scheduler=scheduler)
    return train_one_epoch_many([trainee], loader, device, epoch, max_steps)[0]


# ------------------------------------------------------------------ evaluation


@torch.no_grad()
def evaluate_depth(
    model,
    loader,
    cfg: RunConfig,
    device: torch.device,
    align: str = "",
    num_visualizations: int = 0,
    vis_dir: Optional[str] = None,
    vis_prefix: str = "eval",
) -> Dict[str, float]:
    """Metrics for the depth task, averaged per image.

    Per image rather than pooled over the dataset, because that is how the MiDaS and
    ZoeDepth tables this benchmark is compared against are computed: each image is
    aligned and scored on its own, then the scores are averaged. Pooling first would
    let one large image dominate and would make the numbers incomparable.
    """
    from symtrs.metrics.depth import (
        AllInvalidError,
        align_prediction,
        compute_depth_metrics,
        summarise_alignment,
    )

    model.eval()
    align = align or cfg.eval.align or getattr(model, "align", "") or getattr(model, "default_align", "none")
    # An override (`--align`, `eval.align`) can contradict the model's output space; refuse it
    # here as well as at build time, because this function is also reached with a model built
    # elsewhere. Adapters without the method (test doubles) are scored as before.
    validate_alignment = getattr(model, "validate_alignment", None)
    if callable(validate_alignment):
        validate_alignment(align)
    scoring_input = getattr(model, "scoring_input", None)
    tracker = MetricTracker()
    skipped = 0
    saved = 0
    tile_stats: Dict[str, int] = {}

    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        ground_truth = batch["depth"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)

        with autocast_context(device, cfg.eval.amp):
            prediction = forward_native(
                model, image, cfg.eval.tile_size, cfg.eval.tile_overlap, stats=tile_stats,
                tile_batch=cfg.eval.tile_batch,
            )
            metres = model.to_metres(prediction)

        metres = metres.float()
        # What the alignment fits: metres for most, the raw disparity for MiDaS (whose shift
        # a depth-space fit cannot remove). `metres` stays the human-readable map for previews.
        scored = scoring_input(prediction.float(), align) if callable(scoring_input) else metres
        ground_truth = ground_truth.float()
        valid = valid.bool() & torch.isfinite(ground_truth) & (ground_truth > 0)

        for index in range(image.shape[0]):
            single_prediction = scored[index : index + 1]
            single_target = ground_truth[index : index + 1]
            single_valid = valid[index : index + 1]
            try:
                metrics = compute_depth_metrics(
                    single_prediction, single_target, single_valid, align=align,
                    max_depth=cfg.data.max_depth,
                )
                # The fitted scale and shift are logged alongside the metrics: a scale that
                # drifts epoch to epoch is the first sign that an alignment is hiding a
                # model that is not actually predicting the right thing. The fit sits inside
                # the same `try` as the metrics: it raises `AllInvalidError` for an image
                # with fewer than two valid pixels, which `compute_depth_metrics` tolerates
                # (it leaves that image unaligned), so a single-valid-pixel image used to get
                # through the metrics and then crash the whole evaluation here. Both results
                # are only recorded once both exist, so a skipped image leaves no half-row.
                _, fits = align_prediction(
                    single_prediction, single_target, single_valid, mode=align,
                    max_depth=cfg.data.max_depth,
                )
            except AllInvalidError as exc:
                # An all-sky tile has nothing to measure. Counting it as a failure
                # would be wrong and crashing would be worse.
                skipped += 1
                if skipped <= 3:
                    print(f"[symtrs] skipping an unscorable image: {exc}")
                continue

            tracker.update_many(metrics, n=1)
            tracker.update_many(summarise_alignment(fits), n=1)

            if vis_dir and saved < num_visualizations:
                saved_dir = vis_dir
                vis_dir = _render_visual(
                    lambda path: _save_depth_visual(
                        image[index], metres[index], ground_truth[index], valid[index], path
                    ),
                    saved_dir,
                    f"{vis_prefix}_{saved:02d}.png",
                )
                saved += 1

    results = tracker.averages()
    if skipped:
        results["skipped_images"] = float(skipped)
    if tile_stats.get("windows"):
        # Totals over the split, only when tiling ran: a window whose overlap was too flat or
        # too uncorrelated for a full fit (`tile_match_fallbacks`) or could not be fitted at all
        # (`tile_match_unmatched`) is a stitching seam risk the number should disclose.
        results["tile_match_fallbacks"] = float(tile_stats["fallbacks"])
        results["tile_match_unmatched"] = float(tile_stats["unmatched"])
        results["tile_windows"] = float(tile_stats["windows"])
    return results


@torch.no_grad()
def evaluate_segmentation(
    model,
    loader,
    cfg: RunConfig,
    device: torch.device,
    num_visualizations: int = 0,
    vis_dir: Optional[str] = None,
    vis_prefix: str = "eval",
) -> Dict[str, float]:
    """Metrics for the segmentation task, from one confusion matrix over the split.

    Accumulating a confusion matrix rather than averaging per-batch mIoU is not a
    detail: a class absent from a batch contributes no information, and averaging
    per-batch would let a batch with two classes count as much as a batch with six.
    """
    from symtrs.metrics.semantic import ConfusionMatrix

    model.eval()
    num_classes = cfg.data.num_classes
    ignore_index = cfg.data.ignore_index
    confusion = ConfusionMatrix(num_classes, ignore_index=ignore_index)
    saved = 0
    total_loss = 0.0
    batches = 0

    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        with autocast_context(device, cfg.eval.amp):
            logits = forward_native(
                model, image, cfg.eval.tile_size, cfg.eval.tile_overlap,
                tile_batch=cfg.eval.tile_batch,
            )
            loss, _ = model.compute_loss(logits, mask)

        total_loss += float(loss.detach())
        batches += 1

        if tuple(logits.shape[-2:]) != tuple(mask.shape[-2:]):
            logits = _resize(logits, mask.shape[-2:])

        prediction = logits.argmax(dim=1)
        confusion.update(prediction, mask)

        if vis_dir:
            for index in range(image.shape[0]):
                if saved >= num_visualizations:
                    break
                from symtrs.common.plots import save_segmentation_preview

                saved_dir = vis_dir
                vis_dir = _render_visual(
                    lambda path: save_segmentation_preview(
                        image[index], prediction[index], mask[index], num_classes, path,
                        ignore_index=ignore_index,
                    ),
                    saved_dir,
                    f"{vis_prefix}_{saved:02d}.png",
                )
                saved += 1

    results = confusion.summary()
    results["loss"] = total_loss / max(batches, 1)
    return results


def _render_visual(render, vis_dir: Optional[str], name: str) -> Optional[str]:
    """Render one qualitative panel, returning the directory to keep using, or None.

    Plots go through matplotlib and Pillow, which are the two dependencies most likely to
    be missing or unhappy on a headless GPU box. A preview that cannot be drawn is a lost
    convenience; letting it raise out of the evaluation is a lost run, because the failure
    lands after the metrics have been computed and before they are written. So the first
    failure turns visuals off and says what to install, and the numbers carry on.
    """
    if not vis_dir:
        return None
    try:
        render(str(Path(vis_dir) / name))
    except Exception as exc:  # noqa: BLE001 - any plotting failure is non-fatal by design
        print(f"[symtrs] visual output disabled: {type(exc).__name__}: {exc}")
        if isinstance(exc, ImportError):
            print(
                "[symtrs]   install matplotlib and pillow for preview panels; "
                "metrics are unaffected"
            )
        else:
            # Anything else is a rendering bug, not a missing package. Saying "install
            # matplotlib" here sent the reader to pip for an `IndexError`, which is where this
            # message used to point for every failure - including the two real bugs it hid.
            print(
                "[symtrs]   this is a bug in the preview, not in the numbers; the metrics "
                "were computed already and are unaffected. Run with --traceback for the stack"
            )
        return None
    return vis_dir


def _save_depth_visual(image, metres, ground_truth, valid, path) -> None:
    """One RGB / predicted / ground-truth panel. `path` is the finished file name.

    The tensors are passed at the rank the evaluation loop holds them in - `image[index]` is
    `[3, H, W]` and the rest are `[1, H, W]` - and `save_depth_preview` normalises them.
    Adding a batch axis here used to break the panel: `_to_uint8` recognises `[3, H, W]` and
    `[H, W]` but not `[1, 3, H, W]`, so it passed the image through unchanged and the four-
    dimensional panel then failed in the padding below with a broadcast error naming neither
    the tensor nor the caller.
    """
    from symtrs.common.plots import save_depth_preview

    save_depth_preview(image, metres, ground_truth, valid, str(path))


def evaluate(
    model,
    loader,
    cfg: RunConfig,
    device: torch.device,
    align: str = "",
    num_visualizations: int = 0,
    vis_dir: Optional[str] = None,
    vis_prefix: str = "eval",
) -> Dict[str, float]:
    """Dispatch to the right evaluation for the run's task."""
    if cfg.task == "depth":
        return evaluate_depth(
            model, loader, cfg, device, align=align,
            num_visualizations=num_visualizations, vis_dir=vis_dir, vis_prefix=vis_prefix,
        )
    if cfg.task == "segmentation":
        return evaluate_segmentation(
            model, loader, cfg, device,
            num_visualizations=num_visualizations, vis_dir=vis_dir, vis_prefix=vis_prefix,
        )
    raise ValueError(
        f"task {cfg.task!r} has no evaluation loop; domain adaptation is evaluated by "
        "its upstream script, and its numbers are collected from the run directories by "
        "`symtrs tables --metric domain_adapt`"
    )


def format_metrics(results: Mapping[str, float], prefix: str = "") -> str:
    """One line of `key=value` pairs, ordered so the headline metrics come first.

    Counts stay integers. A super-resolution run reports `images=2` and
    `checkpoint_epoch=1` alongside its PSNR, and `2.0000` in a log line reads like a
    measurement rather than a count - which is exactly the confusion the count is there
    to prevent.
    """
    order: List[str] = [
        "loss", "miou", "pixel_acc", "mean_acc", "mean_dice", "fwiou",
        "rmse", "mae", "abs_rel", "sq_rel", "d1", "d2", "d3", "silog",
        "psnr", "ssim", "mse",
    ]
    keys = [key for key in order if key in results]
    keys += [key for key in results if key not in keys]
    body = " ".join(
        f"{key}={value:d}" if isinstance(value, int) and not isinstance(value, bool)
        else f"{key}={value:.4f}"
        for key in keys
        for value in (results[key],)
    )
    return f"{prefix}{body}".strip()


__all__ = [
    "autocast_context",
    "evaluate",
    "evaluate_depth",
    "evaluate_segmentation",
    "format_metrics",
    "forward_native",
    "make_grad_scaler",
    "prepare_input",
    "train_one_epoch",
]
