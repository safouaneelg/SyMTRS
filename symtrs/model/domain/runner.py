"""Day-to-night and night-to-day translation, driven through the upstream code.

    symtrs train --task domain_adapt --model cyclegan --data-root /data/SyMTRS \\
                 --set model_kwargs.direction=day2night --epochs 40

    symtrs train --task domain_adapt --model pix2pix --data-root /data/SyMTRS \\
                 --set model_kwargs.direction=night2day --epochs 40

The training itself is upstream `train.py`, unmodified. This module prepares the inputs
upstream needs (see `symtrs.model.domain.upstream` for the auto-split trap that makes the
staging directory necessary), launches it, and then scores the generated images with
the same FID/KID implementation used everywhere else, so the domain-adaptation numbers
in the results table come from the same code as the ones in the text.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from symtrs.model.domain import upstream as upstream_module

#: direction -> "translate A into B" or the reverse. Stated once; everything else derives.
DIRECTIONS = ("day2night", "night2day")

#: Which upstream model class and dataset mode each SyMTRS model key uses.
UPSTREAM_MODELS = {
    "cyclegan": ("cycle_gan", "unaligned"),
    "pix2pix": ("pix2pix", "aligned"),
}


class DomainAdaptation:
    """A non-neural task descriptor, so `symtrs models` and `symtrs doctor` stay uniform.

    Deliberately not an `nn.Module`: the network belongs to the upstream project and is
    constructed and optimised entirely inside upstream's own process. Anything this
    class did with parameters would be a second, competing implementation.
    """

    task = "domain_adapt"
    key = ""
    default_loss = ""
    requires = ("torch",)

    def __init__(
        self,
        direction: str = "day2night",
        upstream: str = "",
        name: str = "",
        dataset_mode: str = "",
        gpu_ids: str = "",
        extra_args: Optional[List[str]] = None,
        decay_epochs: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        if direction not in DIRECTIONS:
            raise ValueError(f"unknown direction {direction!r}; expected one of {DIRECTIONS}")
        self.direction = direction
        self.upstream_root = upstream
        self.experiment = name
        self.override_dataset_mode = dataset_mode
        self.gpu_ids = gpu_ids
        self.extra_args = list(extra_args or [])
        # None -> divide cfg.train.epochs following upstream's own flat-then-decay shape.
        # See `upstream.epoch_split` for why this has to be passed at all.
        self.decay_epochs = None if decay_epochs is None else int(decay_epochs)
        self.extra_kwargs = kwargs

    # ------------------------------------------------------------------ description

    @property
    def upstream_model(self) -> str:
        return UPSTREAM_MODELS.get(self.key, ("cycle_gan", "unaligned"))[0]

    @property
    def dataset_mode(self) -> str:
        if self.override_dataset_mode:
            return self.override_dataset_mode
        return UPSTREAM_MODELS.get(self.key, ("cycle_gan", "unaligned"))[1]

    def describe(self) -> str:
        source, target = self.direction.split("2")
        return (
            f"{self.key} ({self.task}): translates {source} to {target} using upstream "
            f"--model {self.upstream_model} --dataset_mode {self.dataset_mode}"
        )

    def resolve_upstream(self) -> Path:
        return upstream_module.verify(self.upstream_root or None)

    # -------------------------------------------------------------------- commands

    def run_train(
        self,
        dataroot: str,
        run_dir: str,
        split: str = "train",
        epochs: int = 40,
        batch_size: int = 1,
        device: str = "",
        gpus: int = 1,
        dry_run: bool = False,
    ) -> List[str]:
        root = self.resolve_upstream()
        staging = upstream_module.build_staging_dir(dataroot, run_dir, split)
        day, night = upstream_module.count_staged(staging, split)
        print(f"[symtrs] staged {split}: {day} domain-A images, {night} domain-B images")
        if day <= 0 or night <= 0:
            raise RuntimeError(
                f"the staging directory {staging} has {day} A and {night} B images, so "
                "upstream would train on nothing. The dataset root is probably wrong, or "
                "the 2048 rgb_day / rgb_night folders are empty."
            )

        command = upstream_module.train_command(
            upstream=root,
            staging=staging,
            experiment=self.experiment_name(),
            direction=self.direction,
            dataset_mode=self.dataset_mode,
            upstream_model=self.upstream_model,
            epochs=epochs,
            batch_size=batch_size,
            checkpoints_dir=str(Path(run_dir) / "checkpoints"),
            extra_args=self.extra_args,
            decay_epochs=self.decay_epochs,
            gpus=gpus,
        )
        if not dry_run:
            _launch(command, cwd=root, env=device_env(device, self.gpu_ids, gpus))
        return command

    def run_test(
        self,
        dataroot: str,
        run_dir: str,
        split: str = "test",
        epochs: int = 40,
        checkpoint: str = "latest",
        device: str = "",
        dry_run: bool = False,
    ) -> List[str]:
        root = self.resolve_upstream()
        staging = upstream_module.build_staging_dir(dataroot, run_dir, split)

        command = upstream_module.test_command(
            upstream=root,
            staging=staging,
            experiment=self.experiment_name(),
            split=split,
            dataset_mode=self.dataset_mode,
            upstream_model=self.upstream_model,
            direction=self.direction,
            checkpoint=checkpoint,
            checkpoints_dir=str(Path(run_dir) / "checkpoints"),
            results_dir=str(Path(run_dir) / "results"),
            extra_args=self.extra_args,
        )
        if not dry_run:
            _launch(command, cwd=root, env=device_env(device, self.gpu_ids, gpus=1))
        return command

    def experiment_name(self) -> str:
        """Upstream keys its checkpoints and results by `--name`, so the name has to
        carry everything that distinguishes one run from another - the direction above
        all, since day-to-night and night-to-day would otherwise share a checkpoint
        directory and silently overwrite each other."""
        if self.experiment:
            return self.experiment
        return f"symtrs_{self.key}_{self.direction}"

    # --------------------------------------------------------------- generated images

    def generated_dir(self, run_dir: str, split: str, checkpoint: str = "latest") -> Path:
        """The folder upstream `test.py` writes this run's translations into.

        Flat, not one subfolder per visual: upstream's `save_images` at this revision
        writes `<image>_<label>.png` into `images/`, so the generated side is selected by
        filename (`*_fake_B.png`), which is what `generated_paths` does. An older
        upstream layout used `images/fake_B/`, and reading it that way silently scores
        nothing.
        """
        return Path(run_dir) / "results" / self.experiment_name() / f"{split}_{checkpoint}" / "images"

    def generated_paths(self, run_dir: str, split: str, checkpoint: str = "latest") -> List[Path]:
        """The files holding *this direction's* generated images.

        `AtoB` (day to night) writes `fake_B`; `BtoA` writes `fake_A`. Both are written
        in every run, so taking the wrong one would score the translation the run was not
        asked to learn.
        """
        side = "B" if self.direction == "day2night" else "A"
        return sorted(self.generated_dir(run_dir, split, checkpoint).glob(f"*_fake_{side}.png"))


def device_env(device: str = "", gpu_ids: str = "", gpus: int = 1) -> Dict[str, str]:
    """The environment that makes upstream's own device choice match the config's.

    Upstream has no `--gpu_ids` flag at this revision: `util.util.init_ddp()` reads
    `WORLD_SIZE`/`LOCAL_RANK` under DDP and otherwise takes `cuda:0` if a GPU is visible
    and `cpu` if none is. So the only levers are the ones the process is started with:

      * `train.gpus > 1` -> `torch.distributed.run` (see `upstream.launcher`) sets the DDP
        variables; nothing to add here, and `CUDA_VISIBLE_DEVICES` must be left alone or
        the ranks would land on the wrong devices.
      * `model_kwargs.gpu_ids` -> selects the visible devices, which is what "run on GPU
        1" means when the script picks the device itself. A string, as in `"0,1"`.
      * `train.device: cpu` -> hides every GPU, so `torch.cuda.is_available()` is false
        and upstream takes its CPU branch. Without this a `--device cpu` run would
        silently train on the GPU.
    """
    if gpus and gpus > 1:
        return {}
    if gpu_ids:
        return {"CUDA_VISIBLE_DEVICES": gpu_ids}
    if str(device).lower() == "cpu":
        return {"CUDA_VISIBLE_DEVICES": ""}
    return {}


def _launch(command: List[str], cwd: Path, env: Optional[Dict[str, str]] = None) -> None:
    missing = upstream_module.missing_imports()
    if missing:
        raise upstream_module.UpstreamError(
            f"upstream imports {', '.join(missing)} before it parses a single option, and "
            f"{'it is' if len(missing) == 1 else 'they are'} not installed in "
            f"{sys.executable}. Install the extra that provides them: "
            f"`pip install -e '.[adapt]'`"
        )
    environment = dict(os.environ)
    environment.update(env or {})
    print(f"[symtrs] {' '.join(command)}")
    completed = subprocess.run(command, cwd=str(cwd), env=environment)
    if completed.returncode != 0:
        raise RuntimeError(
            f"the upstream script exited with code {completed.returncode}. Its own output "
            "above says why; this wrapper does not reinterpret it."
        )


# ------------------------------------------------------------------- the entry point


def run_domain_adaptation(cfg, mode: str = "finetune") -> Dict[str, Any]:
    """Train and/or score one direction of a domain translation.

    Called by `symtrs.engine.trainer` before it builds anything, because there is no
    SyMTRS model to build: the network is upstream's.
    """
    from symtrs.registry import get_spec

    spec = get_spec(cfg.model)
    descriptors = [
        DomainAdaptation(
            direction=cfg.model_kwargs.get("direction", "day2night"),
            upstream=cfg.model_kwargs.get("upstream", ""),
            name=cfg.model_kwargs.get("name", ""),
            dataset_mode=cfg.model_kwargs.get("dataset_mode", ""),
            gpu_ids=str(cfg.model_kwargs.get("gpu_ids", "")),
            decay_epochs=cfg.model_kwargs.get("decay_epochs"),
            extra_args=list(cfg.upstream_args) + list(cfg.model_kwargs.get("extra_args", []) or []),
        )
    ]
    task = descriptors[0]
    task.key = spec.key

    if cfg.resolution != 2048:
        raise ValueError(
            f"domain adaptation is 2048-only: the rgb_night folders are not downsampled, "
            f"so there is no night imagery at {cfg.resolution}px. Got resolution "
            f"{cfg.resolution} from the config."
        )

    run_dir = cfg.resolved_out_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_yaml(str(run_dir / "config.yaml"))

    print(f"[symtrs] {task.describe()}")

    if not upstream_module.is_complete(task.upstream_root or None):
        print("[symtrs] the upstream checkout is not ready; running setup")
        upstream_module.setup(task.upstream_root or None)

    if mode == "zero_shot":
        raise ValueError(
            "zero-shot does not apply to domain adaptation: an untrained CycleGAN "
            "generates noise, and there are no released SyMTRS adaptation weights to "
            "evaluate instead. Train first, then score the checkpoint with "
            "`symtrs eval --task domain_adapt`."
        )

    summary: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "task": cfg.task,
        "model": cfg.model,
        "direction": task.direction,
        "resolution": cfg.resolution,
        "mode": mode,
        "epochs": cfg.train.epochs,
    }

    if mode == "finetune":
        if cfg.train.gpus > 1 and str(cfg.train.device).lower() == "cpu":
            raise ValueError(
                f"train.gpus={cfg.train.gpus} with train.device=cpu is not possible: "
                "upstream's multi-process path initialises NCCL, which needs CUDA. Use "
                "one process, or a GPU device."
            )
        task.run_train(
            dataroot=cfg.data.root,
            run_dir=str(run_dir),
            split=cfg.data.split,
            epochs=cfg.train.epochs,
            batch_size=cfg.train.batch_size,
            device=cfg.train.device,
            gpus=cfg.train.gpus,
        )
    elif mode != "score":
        raise ValueError(f"unknown domain-adaptation mode {mode!r}; expected finetune or score")

    summary["test"] = evaluate_adaptation(cfg, task, run_dir)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def evaluate_adaptation(cfg, task: "DomainAdaptation", run_dir: Path) -> Dict[str, float]:
    """Generate, then score the generated images against the real target domain.

    FID and KID are the numbers that matter here. A cycle-consistency loss going down
    says the translation is invertible; it says nothing about whether the output looks
    like night. Feature-space distance to the real target domain is the direct question,
    and it is also what the reviewer comment asked for.
    """
    task.run_test(
        dataroot=cfg.data.root,
        run_dir=str(run_dir),
        split=cfg.data.test_split,
        epochs=cfg.train.epochs,
        device=cfg.train.device,
    )

    # Which side of the translation is "generated" depends on the direction: AtoB writes
    # fake_B, BtoA writes fake_A, and the real target domain is the other one.
    target_domain = "night" if task.direction == "day2night" else "day"
    generated = task.generated_paths(str(run_dir), cfg.data.test_split)

    if not generated:
        print(
            f"[symtrs] no generated images in "
            f"{task.generated_dir(str(run_dir), cfg.data.test_split)}; upstream may have "
            f"failed to write them, or used a different output layout. Metrics will be "
            f"empty."
        )
        return {}

    real_directory = Path(cfg.data.root) / cfg.data.test_split / f"rgb_{target_domain}"
    if not real_directory.is_dir():
        print(f"[symtrs] no real {target_domain} images at {real_directory}; cannot score")
        return {}

    from symtrs.metrics.domain import distribution_metrics, extract_features

    real_features = extract_features(
        _folder_loader(_image_paths(real_directory), cfg.eval.batch_size, cfg.data.num_workers)
    )
    fake_features = extract_features(
        _folder_loader(generated, cfg.eval.batch_size, cfg.data.num_workers)
    )
    metrics = distribution_metrics(real_features, fake_features)
    print(f"[symtrs] fid={metrics['fid']:.2f} kid={metrics.get('kid', float('nan')):.5f} "
          f"(real={metrics['n_real']:.0f}, generated={metrics['n_fake']:.0f})")
    return metrics


#: Suffixes the metric code will read. Kept next to the readers so a new format cannot
#: be half-added: an image the loader skips is an image silently missing from a score.
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def _image_paths(directory: Path) -> List[Path]:
    """Every image directly inside `directory`, in a stable order."""
    return sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


class _ImageFolder:
    """A dataset over explicit image paths, for the metric code.

    Module level, not a class defined inside a function: DataLoader workers are started
    with `spawn` on macOS and have to pickle the dataset, and a locally-defined class
    cannot be pickled. `read_image` is imported per item so importing this module does
    not pull in an image backend.
    """

    def __init__(self, paths: List[Path]) -> None:
        self.paths = list(paths)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        from symtrs.data.io import read_image

        return {"image": read_image(self.paths[index])}


def _folder_loader(paths: List[Path], batch_size: int, num_workers: int = 0):
    """A minimal loader over a list of image paths, for the metric code."""
    import torch
    from torch.utils.data import DataLoader

    if not paths:
        raise RuntimeError("no images to score")
    return DataLoader(
        _ImageFolder(paths), batch_size=max(1, batch_size), num_workers=max(0, num_workers)
    )


__all__ = [
    "DIRECTIONS",
    "DomainAdaptation",
    "IMAGE_SUFFIXES",
    "UPSTREAM_MODELS",
    "device_env",
    "evaluate_adaptation",
    "run_domain_adaptation",
]
