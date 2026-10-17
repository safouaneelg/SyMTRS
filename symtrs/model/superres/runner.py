"""Super-resolution: ×2, ×4 and ×8 from the `_lr` folders, driven by the legacy trainer.

    symtrs train configs/superresolution/srcnn_x4.yaml --data-root /data/SyMTRS
    symtrs eval  --task superresolution --model swinir --resolution 512 --data-root /data/SyMTRS

This is the one task whose training code predates this package. The trainer lives at
`symtrs/legacy/superres/train.py` (this repository's own code, moved inside the package
unchanged - see `symtrs/legacy/README.md`) and owns the model set (`srcnn`, `autoencoder`,
`srgan`, `swinir`), the data loading and the metrics; rewriting it inside `symtrs` would be
a second implementation of it, and the two would drift. So the same arrangement as domain
adaptation applies: `symtrs` translates a `RunConfig` into the trainer's command line,
launches it, and reads the result back.

What this module adds on top of calling the script by hand:

  * **The folder mapping.** `resolution` already names an input scale everywhere else in
    this package (2048 native, 1024 `_lr/x2`, 512 `_lr/x4`, 256 `_lr/x8`), so it names one
    here too: the LR root is `rgb_day_lr/xN` for that resolution and the HR root is always
    native `rgb_day`. `scale` is derived rather than configured, which is why a config
    cannot say `resolution: 512` and `scale: 2` at the same time and mean something else.

  * **The test split.** The script carves its own validation set out of `--lr_root`, so on
    its own it never touches the SyMTRS `test` split. This module passes `test/` in as the
    held-out set and folds `test_metrics.json` into the `summary.json` every other task
    writes, so a super-resolution row in `symtrs tables` is a real held-out number.

  * **A run directory.** Weights, the metrics CSV, sample panels and the resolved config
    all land under one run directory, as they do for the other three tasks.

`resolution: 2048` means ×1: LR and HR are the same native folder, and the model is an
image-to-image refiner rather than an upsampler. It is a legitimate control - it isolates
how much of a ×4 result is restoration rather than resolution - but it is not
super-resolution, and the run says so in its own summary.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Dict, List, Optional

from symtrs.config import RunConfig
from symtrs.common.jsonio import dumps as json_dumps_safe
from symtrs.data.splits import RESOLUTION_TO_SCALE, resolve_split_dirs, subdir_for

#: The model set of the legacy trainer (`symtrs/legacy/superres/train.py`), mirrored here
#: so the registry, the CLI and the trainer's own `choices=` cannot disagree about what
#: exists.
SR_MODELS = ("srcnn", "autoencoder", "srgan", "swinir")

#: The scale each native-derived resolution implies. Shared with the rest of the package
#: rather than restated, so `--resolution 512` cannot mean ×8 in one task and ×4 in
#: another.
SCALE_FOR_RESOLUTION = dict(RESOLUTION_TO_SCALE)


class SuperResolutionError(RuntimeError):
    """The legacy super-resolution trainer is missing or refused to run."""


#: The legacy trainer, now a module of this package rather than a script in the repository
#: root (`symtrs/legacy/superres/train.py`; the move is recorded in
#: `symtrs/legacy/README.md`). Launched as a subprocess rather than imported: it owns its
#: own RNG, its own matplotlib backend and its own `torch.multiprocessing` start-up, and a
#: crash in it must not take the CLI down with it.
TRAIN_MODULE = "symtrs.legacy.superres.train"


def trainer_missing() -> bool:
    """Whether the legacy trainer is importable, checked without importing it.

    False for every working install, since the trainer ships with this package. It is here
    because the failure it prevents - a stripped or hand-copied `symtrs` that has the
    front-end and not the implementation - used to surface as a `FileNotFoundError` for a
    script in the repository root, and the message that replaced it names the fix.
    """
    try:
        return find_spec(TRAIN_MODULE) is None
    except (ImportError, ValueError):  # a broken install reports as missing
        return True


def scale_for(resolution: int) -> int:
    try:
        return SCALE_FOR_RESOLUTION[resolution]
    except KeyError:
        raise SuperResolutionError(
            f"unsupported resolution {resolution!r}; expected one of "
            f"{sorted(SCALE_FOR_RESOLUTION)}"
        ) from None


def lr_dir(root: str, split: str, resolution: int) -> Path:
    """The degraded input folder: `rgb_day_lr/xN`, or `rgb_day` at native resolution."""
    return Path(resolve_split_dirs(root, split, resolution).day)


def hr_dir(root: str, split: str) -> Path:
    """The target folder. Always native: super-resolution is defined against it."""
    return Path(resolve_split_dirs(root, split, 2048).day)


def expected_test_dirs(root: str, split: str, resolution: int) -> "tuple[Path, Path]":
    """Where the held-out pair for `split` would be, whether or not it exists yet.

    `lr_dir` and `hr_dir` go through `resolve_split_dirs`, which raises when the *split*
    directory is missing. That is the right answer when the run needs the split; here it
    is the question being asked, so the paths are built directly and the caller tests
    them with `is_dir()`. A dataset that is still being generated has a `train/` and no
    `test/`, and a training run should say so rather than die on it.
    """
    split_root = Path(root) / split
    return (
        split_root / subdir_for("day", resolution),
        split_root / subdir_for("day", 2048),
    )


class SuperResolution:
    """A task descriptor, so `symtrs models` and `symtrs doctor` stay uniform.

    Deliberately not an `nn.Module`: the network is built inside the legacy script's
    process, and holding a second copy of it here would be the drift this module exists
    to avoid. This object only knows how to turn a config into a command line.
    """

    task = "superresolution"
    key = ""
    requires = ("torch",)
    default_loss = ""

    def __init__(
        self,
        use_y_channel: bool = False,
        adv_weight: float = 1e-3,
        gpu_ids: str = "",
        extra_args: Optional[List[str]] = None,
        swinir_embed_dim: int = 96,
        swinir_depths: Optional[List[int]] = None,
        swinir_num_heads: Optional[List[int]] = None,
        swinir_window: int = 7,
        swinir_mlp_ratio: float = 4.0,
        swinir_upsampler: str = "pixelshuffle",
        swinir_checkpoint: bool = False,
        **kwargs: Any,
    ) -> None:
        self.use_y_channel = bool(use_y_channel)
        self.adv_weight = float(adv_weight)
        self.gpu_ids = gpu_ids
        self.extra_args = list(extra_args or [])
        # SwinIR's architecture is fixed at build time and has to match the checkpoint,
        # so these are recorded in the run's config rather than left to the script's
        # defaults.
        self.swinir_embed_dim = int(swinir_embed_dim)
        self.swinir_depths = list(swinir_depths or [6, 6, 6, 6])
        self.swinir_num_heads = list(swinir_num_heads or [6, 6, 6, 6])
        self.swinir_window = int(swinir_window)
        self.swinir_mlp_ratio = float(swinir_mlp_ratio)
        self.swinir_upsampler = str(swinir_upsampler)
        self.swinir_checkpoint = bool(swinir_checkpoint)
        self.extra_kwargs = kwargs

    # ------------------------------------------------------------------ description

    @property
    def model_name(self) -> str:
        return self.key or "srcnn"

    def describe(self) -> str:
        return (
            f"{self.model_name} ({self.task}): trained by the legacy trainer "
            "(`symtrs/legacy/superres/train.py`), which owns its own model set and data "
            "loading"
        )

    def describe_cell(self, resolution: int) -> str:
        scale = scale_for(resolution)
        if scale == 1:
            return (
                f"{self.model_name} at {resolution}px: LR and HR are both native rgb_day, "
                "so this is a x1 refiner, not super-resolution"
            )
        return (
            f"{self.model_name} at {resolution}px: x{scale} from rgb_day_lr/"
            f"x{scale} up to native rgb_day"
        )

    # -------------------------------------------------------------------- commands

    def train_command(
        self,
        cfg: RunConfig,
        run_dir: Path,
        with_test: bool = True,
        eval_only: bool = False,
    ) -> List[str]:
        """argv for the legacy trainer, from a run config.

        Every flag the trainer accepts has a home in `RunConfig`, so nothing here is a
        hard-coded constant a config cannot reach: the training knobs come from
        `cfg.train`, the scoring knobs from `cfg.eval`, and the model-specific ones from
        `model_kwargs`.
        """
        scale = scale_for(cfg.resolution)
        root = cfg.data.root

        if trainer_missing():
            raise SuperResolutionError(
                f"{TRAIN_MODULE} is not importable, so there is nothing to launch. The "
                "super-resolution trainer predates this package but is part of it: "
                "reinstall with `pip install -e .`, and run `symtrs doctor` to see what is "
                "installed."
            )

        command = [
            sys.executable, "-m", TRAIN_MODULE,
            "--lr_root", str(lr_dir(root, cfg.data.split, cfg.resolution)),
            "--hr_root", str(hr_dir(root, cfg.data.split)),
            "--model", self.model_name,
            "--scale", str(scale),
            "--out_dir", str(run_dir),
            "--epochs", str(cfg.train.epochs),
            "--batch_size", str(cfg.train.batch_size),
            "--lr", str(cfg.train.lr),
            "--seed", str(cfg.seed),
            "--num_workers", str(cfg.data.num_workers),
            "--device", cfg.train.device,
            "--gpus", str(cfg.train.gpus),
            "--save_samples_every", str(cfg.train.save_samples_every),
            "--adv_weight", str(self.adv_weight),
        ]

        # The script only implements AMP and gradient checkpointing for SwinIR, and only
        # crops pairs for SwinIR, so those flags are passed with the same restriction
        # rather than implying they apply everywhere.
        if self.model_name == "swinir":
            if cfg.train.patch_size > 0:
                command += ["--patch_size", str(cfg.train.patch_size)]
            if cfg.train.val_patch_size > 0:
                command += ["--val_patch_size", str(cfg.train.val_patch_size)]
            if cfg.train.amp:
                command += ["--amp"]
            if self.swinir_checkpoint:
                command += ["--swinir_checkpoint"]
            command += [
                "--swinir_embed_dim", str(self.swinir_embed_dim),
                "--swinir_depths", *[str(d) for d in self.swinir_depths],
                "--swinir_num_heads", *[str(h) for h in self.swinir_num_heads],
                "--swinir_window", str(self.swinir_window),
                "--swinir_mlp_ratio", str(self.swinir_mlp_ratio),
                "--swinir_upsampler", self.swinir_upsampler,
            ]

        if self.use_y_channel:
            command += ["--use_y_channel"]

        if cfg.train.resume:
            command += ["--weights", cfg.train.resume]

        if with_test:
            test_lr, test_hr = expected_test_dirs(root, cfg.data.test_split, cfg.resolution)
            if test_lr.is_dir() and test_hr.is_dir():
                command += ["--test_lr_root", str(test_lr), "--test_hr_root", str(test_hr)]
            elif eval_only:
                # The opposite of the training case: here the test split is the entire
                # point of the command, so its absence is fatal rather than a warning.
                raise SuperResolutionError(
                    f"the {cfg.data.test_split} split has no {test_lr} or no {test_hr}, so "
                    "there is nothing to score. Check --data-root."
                )
            else:
                # Not fatal, but it has to be said out loud: without this the run's only
                # number comes from a validation split carved out of the training folder.
                print(
                    f"[symtrs] no test split at {test_lr} / {test_hr}, so this run will "
                    "report a validation number only and write no test metrics"
                )

        if eval_only:
            command += ["--eval_only"]

        if self.gpu_ids:
            command[command.index("--gpus") + 1] = str(len(self.gpu_ids.split(",")))

        return command + list(self.extra_args)


def summary_from_run(
    cfg: RunConfig,
    run_dir: Path,
    returncode: int,
    command: List[str],
    mode: str = "finetune",
) -> Dict[str, Any]:
    """The run summary, in the shape `symtrs tables` reads for every other task."""
    metrics: Dict[str, Any] = {}
    test_path = run_dir / "test_metrics.json"
    if test_path.is_file():
        metrics = json.loads(test_path.read_text(encoding="utf-8"))

    best = run_dir / "weights" / "best.pt"
    return {
        "run_dir": str(run_dir),
        "task": cfg.task,
        "model": cfg.model,
        "resolution": cfg.resolution,
        "scale": scale_for(cfg.resolution),
        "mode": mode,
        "seed": cfg.seed,
        "epochs": cfg.train.epochs,
        "returncode": returncode,
        "checkpoint": str(best) if best.is_file() else "",
        "command": command,
        # `test` is what `symtrs tables` prints. Absent rather than empty when the test
        # split was not evaluated, so a missing number is visibly missing.
        **({"test": metrics} if metrics else {}),
    }


def run_super_resolution(cfg: RunConfig, mode: str = "finetune", dry_run: bool = False) -> Dict[str, Any]:
    """Train a super-resolution model, or score an existing run, and summarise it.

    `mode="finetune"` trains. `mode="eval"` skips training and scores the run directory's
    `weights/best.pt` on the held-out test split, which is the re-measurement path the
    other tasks get from `symtrs eval`.

    Launched as a subprocess rather than imported, even though it is a module of this
    package now and could be imported: the legacy loop owns its own RNG (`set_seed`), its
    own matplotlib backend (`Agg`), its own `torch.multiprocessing` start-up and its own
    `argparse`, all of which would become the CLI process's state - and a crash in it would
    take the CLI down with it. The reason used to be a top-level `utils` package name this
    package must not inherit; the move removed that hazard, and the other reasons stand.
    """
    from symtrs.model.wrapper import build_from_config

    cfg.validate()
    if mode == "zero_shot":
        raise SuperResolutionError(
            "zero-shot does not apply to super-resolution: there are no released SyMTRS "
            "upsampling weights to evaluate. Train first, then score the checkpoint with "
            "`symtrs eval --task superresolution --model <key> --resolution <r>`, which "
            "loads the run directory's weights/best.pt."
        )
    if mode not in ("finetune", "eval"):
        raise SuperResolutionError(
            f"unknown run mode {mode!r} for super-resolution; expected 'finetune' or 'eval'"
        )

    eval_only = mode == "eval"
    model = build_from_config(cfg)
    run_dir = cfg.resolved_out_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[symtrs] {model.describe()}")
    print(f"[symtrs] {model.describe_cell(cfg.resolution)}")

    if eval_only:
        best = run_dir / "weights" / "best.pt"
        if not best.is_file():
            raise SuperResolutionError(
                f"there is no checkpoint at {best}, so there is nothing to score. Train "
                f"first: `symtrs train --task superresolution --model {cfg.model} "
                f"--resolution {cfg.resolution} --data-root {cfg.data.root}`"
            )
        cfg.to_yaml(str(run_dir / "config.yaml"))

    command = model.train_command(cfg, run_dir, eval_only=eval_only)
    if dry_run:
        print("[symtrs] " + " ".join(command))
        return summary_from_run(cfg, run_dir, 0, command, mode=mode)

    print(f"[symtrs] launching: {' '.join(command)}")
    completed = subprocess.run(command)
    summary = summary_from_run(cfg, run_dir, completed.returncode, command, mode=mode)
    (run_dir / "summary.json").write_text(
        json_dumps_safe(summary, indent=2) + "\n", encoding="utf-8"
    )

    if completed.returncode != 0:
        print(
            f"[symtrs] {TRAIN_MODULE} exited with {completed.returncode}; "
            f"see the output above. Partial results are in {run_dir}"
        )
        return summary

    if "test" in summary:
        test = summary["test"]
        print(
            f"[symtrs] done. test psnr={test.get('psnr', float('nan')):.4f} "
            f"ssim={test.get('ssim', float('nan')):.4f} "
            f"({test.get('images', 0)} images) -> {run_dir}/summary.json"
        )
    else:
        print(f"[symtrs] done. no test metrics were written; see {run_dir}/metrics.csv")
    return summary


__all__ = [
    "SCALE_FOR_RESOLUTION",
    "SR_MODELS",
    "SuperResolution",
    "SuperResolutionError",
    "TRAIN_MODULE",
    "expected_test_dirs",
    "hr_dir",
    "lr_dir",
    "run_super_resolution",
    "scale_for",
    "summary_from_run",
    "trainer_missing",
]
