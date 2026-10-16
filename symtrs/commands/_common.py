"""Shared command-line plumbing.

Every command builds the same `RunConfig` through the same two functions, so a flag
means the same thing everywhere and the YAML/CLI/Python front-ends cannot drift apart.
Adding an option means adding one line to `FLAG_TO_PATH` and one to `add_common_args`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from symtrs.config import RESOLUTIONS, TASKS, ConfigError, RunConfig
from symtrs.data.exr import VALID_BACKENDS

#: CLI flag dest -> dotted path into the run config.
FLAG_TO_PATH: Dict[str, str] = {
    "task": "task",
    "model": "model",
    "resolution": "resolution",
    "seed": "seed",
    "out_dir": "out_dir",
    "data_root": "data.root",
    "split": "data.split",
    "val_split": "data.val_split",
    "test_split": "data.test_split",
    "depth_scale": "data.depth_scale",
    "max_depth": "data.max_depth",
    "label_step": "data.label_step",
    "depth_channel": "data.depth_channel",
    "depth_backend": "data.depth_backend",
    "num_classes": "data.num_classes",
    "ignore_index": "data.ignore_index",
    "num_workers": "data.num_workers",
    "max_samples": "data.max_samples",
    "limit_train_batches": "data.limit_train_batches",
    "limit_val_batches": "data.limit_val_batches",
    "epochs": "train.epochs",
    "batch_size": "train.batch_size",
    "lr": "train.lr",
    "weight_decay": "train.weight_decay",
    "optim": "train.optim",
    "sched": "train.sched",
    "warmup_epochs": "train.warmup_epochs",
    "patch_size": "train.patch_size",
    "val_patch_size": "train.val_patch_size",
    "gpus": "train.gpus",
    "grad_clip": "train.grad_clip",
    "grad_accum": "train.grad_accum",
    "loss": "train.loss",
    "freeze": "train.freeze",
    "log_every": "train.log_every",
    "save_samples_every": "train.save_samples_every",
    "resume": "train.resume",
    "device": "train.device",
    "eval_batch_size": "eval.batch_size",
    "tile_size": "eval.tile_size",
    "tile_overlap": "eval.tile_overlap",
    "align": "eval.align",
    "num_visualizations": "eval.num_visualizations",
}


def add_common_args(parser: argparse.ArgumentParser, training: bool = True) -> None:
    """Flags shared by every command. `training=False` drops the optimiser options."""
    parser.add_argument("--config", "-c", default=None, help="YAML run config to start from")

    core = parser.add_argument_group("run")
    core.add_argument("--task", choices=TASKS, default=None)
    core.add_argument("--model", default=None, help="registry key, e.g. midas_dpt_hybrid")
    core.add_argument("--resolution", type=int, choices=RESOLUTIONS, default=None,
                      help="2048 native, or 1024/512/256 for the _lr/x2/x4/x8 folders")
    core.add_argument("--data-root", dest="data_root", default=None, help="SyMTRS dataset root")
    core.add_argument("--split", default=None)
    core.add_argument("--val-split", dest="val_split", default=None)
    core.add_argument("--test-split", dest="test_split", default=None)
    core.add_argument("--out-dir", dest="out_dir", default=None)
    core.add_argument("--seed", type=int, default=None)
    core.add_argument("--device", default=None, help="cuda, cuda:1, cpu (default: cuda)")

    data = parser.add_argument_group("data")
    data.add_argument("--depth-scale", dest="depth_scale", type=float, default=None,
                      help="multiply raw depth values by this to get metres (default 0.01: the real EXRs hold centimetres)")
    data.add_argument("--max-depth", dest="max_depth", type=float, default=None,
                      help="metres after --depth-scale; farther pixels (sky) are invalid")
    data.add_argument("--depth-channel", dest="depth_channel", type=int, default=None)
    data.add_argument("--depth-backend", dest="depth_backend",
                      choices=list(VALID_BACKENDS), default=None)
    data.add_argument("--num-classes", dest="num_classes", type=int, default=None)
    data.add_argument("--label-step", dest="label_step", type=int, default=None,
                      help="grey-level step of the semantic masks (real data: 63; 1 = raw class ids)")
    data.add_argument("--ignore-index", dest="ignore_index", type=int, default=None)
    data.add_argument("--num-workers", dest="num_workers", type=int, default=None)
    data.add_argument("--max-samples", dest="max_samples", type=int, default=None)
    data.add_argument("--limit-train-batches", dest="limit_train_batches", type=int, default=None)
    data.add_argument("--limit-val-batches", dest="limit_val_batches", type=int, default=None)

    evaluation = parser.add_argument_group("evaluation")
    evaluation.add_argument("--eval-batch-size", dest="eval_batch_size", type=int, default=None)
    evaluation.add_argument("--tile-size", dest="tile_size", type=int, default=None,
                            help="sliding-window inference; 0 runs the whole image at once")
    evaluation.add_argument("--tile-overlap", dest="tile_overlap", type=int, default=None)
    evaluation.add_argument("--align", default=None,
                            help="depth alignment: none | median | scale | scale_shift")
    evaluation.add_argument("--num-visualizations", dest="num_visualizations", type=int, default=None)

    if training:
        optimiser = parser.add_argument_group("training")
        optimiser.add_argument("--epochs", type=int, default=None)
        optimiser.add_argument("--batch-size", dest="batch_size", type=int, default=None)
        optimiser.add_argument("--lr", type=float, default=None)
        optimiser.add_argument("--weight-decay", dest="weight_decay", type=float, default=None)
        optimiser.add_argument("--optim", choices=["adamw", "adam", "sgd"], default=None)
        optimiser.add_argument("--sched", choices=["cosine", "poly", "none"], default=None)
        optimiser.add_argument("--warmup-epochs", dest="warmup_epochs", type=int, default=None)
        optimiser.add_argument("--patch-size", dest="patch_size", type=int, default=None,
                               help="random-crop patch training; 0 trains on the full image")
        optimiser.add_argument("--val-patch-size", dest="val_patch_size", type=int, default=None)
        optimiser.add_argument("--gpus", type=int, default=None)
        optimiser.add_argument("--grad-clip", dest="grad_clip", type=float, default=None)
        optimiser.add_argument("--grad-accum", dest="grad_accum", type=int, default=None)
        optimiser.add_argument("--loss", default=None)
        optimiser.add_argument("--freeze", default=None,
                               help="'', 'encoder', 'head' or 'last_n:<k>'")
        optimiser.add_argument("--log-every", dest="log_every", type=int, default=None)
        optimiser.add_argument("--save-samples-every", dest="save_samples_every", type=int, default=None)
        optimiser.add_argument("--resume", default=None)

    misc = parser.add_argument_group("misc")
    misc.add_argument("--set", action="append", default=None, metavar="KEY=VALUE",
                      help="any config key by dotted path, e.g. --set train.loss_kwargs.gradient_weight=0.5")
    misc.add_argument("--print-config", action="store_true", help="print the resolved config and exit")
    misc.add_argument("--dry-run", action="store_true",
                      help="check the config and the dataset layout without building the model")


def parse_set(items: Optional[List[str]]) -> Dict[str, Any]:
    """`["train.lr=1e-5", "data.num_workers=8"]` -> `{"train.lr": "1e-5", ...}`."""
    overrides: Dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ConfigError(f"--set expects KEY=VALUE, got {item!r}")
        key, _, value = item.partition("=")
        overrides[key.strip()] = value.strip()
    return overrides


def looks_like_a_checkpoint_path(value: Any) -> bool:
    """Whether a `--model` value is a file on disk rather than a registry key.

    Every registry key is a bare identifier (`midas_dpt_hybrid`, `unet_small`), so a
    separator or a weight-file extension is enough to tell them apart. Checked without
    touching the filesystem: `--dry-run` must not need the checkpoint to exist yet, and an
    unknown *key* still has to reach the registry so its error can list the near matches.
    """
    if not value:
        return False
    text = str(value)
    return text.endswith((".pt", ".pth", ".ckpt")) or "/" in text or "\\" in text


def redirect_checkpoint_flag(args: argparse.Namespace) -> None:
    """Read a checkpoint passed to `--model` as what it is.

    `--model` takes a registry key, but a file on disk is the more natural thing to reach
    for, and on `eval` and `infer` there is a `--checkpoint` flag for exactly that. Left
    alone, `--model runs/seg/weights/best.pt` is treated as a model *key*, which then
    becomes a run directory whose path is that file with the resolution appended - and the
    error that surfaces is `NotADirectoryError: .../best.pt/2048`, which says nothing about
    the flag.

    Where there is no `--checkpoint` to redirect to - `train` and `sweep` - the value is
    refused outright rather than quietly dropped, because those two do have somewhere for a
    checkpoint to go (`--resume`), and the useful answer is to name it.
    """
    # `hasattr`, not `getattr(args, "checkpoint", None)`: the two are not the same question.
    # A command with no `--checkpoint` at all would read as "empty", this would conjure a
    # `checkpoint` attribute onto its namespace that nothing consumes, and it would blank
    # `--model` on the way - losing the model and surfacing `model is required`, which names
    # neither the flag that was wrong nor what the file actually is.
    has_checkpoint = hasattr(args, "checkpoint")

    model = getattr(args, "model", None)
    if not looks_like_a_checkpoint_path(model):
        return
    if has_checkpoint and args.checkpoint:
        return  # an explicit --checkpoint wins; --model is then the architecture to build

    text = str(model)
    if has_checkpoint:
        args.checkpoint = model
        args.model = None
        print(f"[symtrs] reading {text!r} as --checkpoint; --model takes a registry key "
              f"(`symtrs models` lists them)")
        return

    raise ConfigError(
        f"--model takes a registry key, but {text!r} is a checkpoint file.\n"
        f"[symtrs]   to keep training it:  symtrs train --model <key> --resume {text}\n"
        f"[symtrs]   to score it:          symtrs eval --checkpoint {text}\n"
        "[symtrs]   `symtrs models` lists the registry keys"
    )


def adopt_checkpoint_settings(cfg: RunConfig, args: argparse.Namespace, overrides: Dict[str, Any]) -> RunConfig:
    """Take the run identity from a checkpoint when the command line did not state it.

    `symtrs eval --checkpoint runs/seg/weights/best.pt` is the documented invocation, and
    the model key, the task and the resolution are all recorded inside that file. Without
    this, the command fails validation on a missing `model`, and - worse - a checkpoint
    trained at 1024 is silently evaluated at the config default of 2048, which is a
    different point on the very axis this benchmark is about.

    Anything the caller stated on the command line wins, and a file that is not a symtrs
    checkpoint is left alone so the loader can explain itself.
    """
    checkpoint = getattr(args, "checkpoint", None)
    if not checkpoint:
        return cfg

    from symtrs.common.checkpoint import is_symtrs_checkpoint, read_checkpoint_metadata
    from symtrs.common.checkpoint import resolve_checkpoint_path

    path = resolve_checkpoint_path(checkpoint)
    if path is None or not is_symtrs_checkpoint(path):
        return cfg

    metadata = read_checkpoint_metadata(path)
    adopted: Dict[str, Any] = {}

    if not cfg.model and metadata.get("model_key"):
        adopted["model"] = metadata["model_key"]
    if "task" not in overrides and metadata.get("task") in TASKS:
        adopted["task"] = metadata["task"]
    if "resolution" not in overrides and metadata.get("resolution") in RESOLUTIONS:
        adopted["resolution"] = metadata["resolution"]
    # A 6-class head scored against a 21-class label space reads every label as wrong.
    if "num_classes" not in overrides and metadata.get("num_classes"):
        adopted["data.num_classes"] = metadata["num_classes"]
    if "label_step" not in overrides and metadata.get("label_step"):
        adopted["data.label_step"] = metadata["label_step"]
    if "ignore_index" not in overrides and metadata.get("ignore_index") is not None:
        adopted["data.ignore_index"] = metadata["ignore_index"]

    if not adopted:
        return cfg
    names = ", ".join(f"{k}={v}" for k, v in adopted.items())
    print(f"[symtrs] taking {names} from {path}")
    return cfg.with_overrides(adopted)


def build_config(args: argparse.Namespace) -> RunConfig:
    """Assemble a validated `RunConfig` from a YAML base plus the command line."""
    redirect_checkpoint_flag(args)

    if getattr(args, "config", None):
        cfg = RunConfig.from_yaml(args.config)
    else:
        cfg = RunConfig()

    overrides: Dict[str, Any] = {}
    for flag, dotted in FLAG_TO_PATH.items():
        value = getattr(args, flag, None)
        if value is not None:
            overrides[dotted] = value
    overrides.update(parse_set(getattr(args, "set", None)))

    cfg = cfg.with_overrides(overrides) if overrides else cfg
    cfg = adopt_checkpoint_settings(cfg, args, overrides)
    return cfg.validate()


def should_stop_early(args: argparse.Namespace, cfg: RunConfig) -> bool:
    """Handle `--print-config` and `--dry-run`. Returns True when the caller should stop."""
    import yaml

    if getattr(args, "print_config", False):
        print(yaml.safe_dump(cfg.to_dict(), sort_keys=False, default_flow_style=False))
        return True
    if getattr(args, "dry_run", False):
        if cfg.task == "superresolution":
            print_superresolution_command(cfg)
        elif cfg.task == "shadow_removal":
            print_shadow_commands(cfg)
        else:
            print_preflight(cfg)
        return True
    return False


def print_shadow_commands(cfg: RunConfig) -> None:
    """The two worker launches a shadow-removal run makes, and whether upstream is there."""
    from symtrs.model.shadow import arch_for, worker_command
    from symtrs.model.shadow.archs import upstream_missing
    from symtrs.model.shadow.runner import check_config

    print("[symtrs] dry run - nothing will be staged or trained")
    check_config(cfg, "finetune")
    config_path = cfg.resolved_out_dir / "config.yaml"
    print(f"[symtrs] cwd: {arch_for(cfg.model).repo_path}")
    for phase, splits in (("train", ""), ("predict", f"{cfg.data.val_split},{cfg.data.test_split}")):
        print("[symtrs] " + " ".join(worker_command(str(config_path), phase, splits)))
    print(f"[symtrs] {upstream_missing(cfg.model) or 'upstream checkout present'}")


def print_superresolution_command(cfg: RunConfig) -> None:
    """The argv a super-resolution run would launch.

    This is the one task a dry run can be exact about, because the training happens in
    another process and the whole translation is a command line. Printing it is also the
    thing worth eyeballing before a 40-epoch run: it is where a wrong input folder or a
    wrong scale shows up.
    """
    from symtrs.model.superres import TRAIN_MODULE, SuperResolutionError, scale_for

    print("[symtrs] dry run - nothing will be trained")
    print(f"[symtrs] {cfg.task} / {cfg.model} at {cfg.resolution}px (x{scale_for(cfg.resolution)})")
    try:
        from symtrs.model.wrapper import build_from_config

        model = build_from_config(cfg)
        print(f"[symtrs] {model.describe_cell(cfg.resolution)}")
        # `[3:]` rather than `[2:]`: the first three tokens are the interpreter, `-m` and the
        # module, and printing the module as part of the flags would read as a flag.
        print(f"[symtrs] python -m {TRAIN_MODULE} \\")
        for item in model.train_command(cfg, cfg.resolved_out_dir)[3:]:
            separator = " \\" if not item.startswith("--") else ""
            print(f"    {item}{separator}")
    except SuperResolutionError as exc:
        print(f"[symtrs] {exc}")


def print_preflight(cfg: RunConfig) -> None:
    """What a run would do, before any weights are downloaded or GPU is touched.

    Deliberately does not build the model: constructing a MiDaS or Depth Pro checkpoint
    downloads hundreds of megabytes, and the point of a dry run is to find out that the
    dataset path is wrong *without* that.
    """
    import yaml

    from symtrs.data.splits import describe_layout
    from symtrs.registry import get_spec

    print("[symtrs] dry run - nothing will be trained or downloaded")
    print(yaml.safe_dump(cfg.to_dict(), sort_keys=False, default_flow_style=False))

    try:
        spec = get_spec(cfg.model)
        print(f"[symtrs] model: {spec.key} ({spec.task})")
        print(f"[symtrs]   {spec.description}")
        if spec.protocol:
            print(f"[symtrs]   protocol={spec.protocol} default_align={spec.default_align or '-'}")
        if spec.requires:
            print(f"[symtrs]   requires: {', '.join(spec.requires)}")
        if spec.task != cfg.task:
            print(
                f"[symtrs] WARNING: --task is {cfg.task!r} but {spec.key!r} is a "
                f"{spec.task!r} model"
            )
    except Exception as exc:
        print(f"[symtrs] model check failed: {exc}")

    if cfg.data.root:
        print(describe_layout(cfg.data.root))
        # Only the two tensor tasks have a loader for `from_config` to build.
        # `domain_adapt` hands file lists to the vendored upstream loaders instead, and
        # `superresolution` is served by its own script, so asking either one for a
        # dataloader answers a question that was never being asked.
        if cfg.task not in ("depth", "segmentation"):
            print(f"[symtrs] {cfg.task} does not use this package's dataloader; "
                  "the layout above is what the run reads")
            return
        try:
            from symtrs.data.factory import from_config

            loader = from_config(cfg, train=True)
            print(f"[symtrs] train batches per epoch: {len(loader)}")
        except Exception as exc:
            print(f"[symtrs] could not build the training loader: {exc}")
    else:
        print("[symtrs] data.root is not set, so the dataset layout was not checked")


def run_dir_for(cfg: RunConfig) -> Path:
    return cfg.resolved_out_dir


__all__ = [
    "FLAG_TO_PATH",
    "add_common_args",
    "build_config",
    "parse_set",
    "print_preflight",
    "run_dir_for",
    "should_stop_early",
]
