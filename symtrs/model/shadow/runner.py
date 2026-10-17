"""Shadow removal driven through the upstream networks.

    symtrs train configs/shadow_removal/shadowformer.yaml --data-root /data/SyMTRS
    symtrs train --task shadow_removal --model restormer --data-root /data/SyMTRS \\
                 --epochs 20 --set shadow.iterations=500

Why a worker instead of the upstream `train.py` scripts: all three were written for
256-320 px ISTD crops. They train by epoch over a loader with hard-coded folder names
(`train_A/B/C`), hard-code CUDA and `DataParallel`, and validate on the *whole* image in
one forward pass - at 2048 px that does not fit in 12 GB for ShadowFormer or Restormer.
Driving three different loops and parsers would also give three different notions of
"best" and of "score". So the worker (`worker.py`) imports only each upstream's network
class and runs one loop and one tiled inference for all three; the metric code is
`symtrs.metrics.shadow`, run here in the parent on the PNGs the worker wrote. The networks
are not reimplemented: a model whose class is not in the pinned checkout fails with the
command that fetches it.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from symtrs.model.shadow.archs import arch_for, upstream_missing
from symtrs.common.jsonio import dumps as json_dumps_safe

WORKER_MODULE = "symtrs.model.shadow.worker"

#: Every column `metrics.csv` can hold, fixed up front: the worker appends training rows
#: and the parent appends scoring rows, and a CSV whose header was taken from the first
#: row would misalign the others.
CSV_FIELDS = (
    "iteration", "split", "loss", "lr", "seconds", "vram_gb",
    "psnr", "ssim", "mae_lab", "rmse_lab",
    "psnr_shadow", "ssim_shadow", "mae_lab_shadow", "rmse_lab_shadow",
    "psnr_nonshadow", "ssim_nonshadow", "mae_lab_nonshadow", "rmse_lab_nonshadow",
    "shadow_fraction", "images", "images_with_shadow",
)


def append_metrics_row(path: Path, row: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow(row)


def describe(key: str) -> str:
    arch = arch_for(key)
    return (
        f"{key} (shadow_removal): upstream {arch.repo}/{arch.source}:{arch.cls}, "
        "rgb_day -> rgb_noshadow at 2048 (trained on crops, tiled at inference)"
    )


class ShadowRemoval:
    """A non-neural task descriptor, so `symtrs models` and `symtrs doctor` stay uniform.

    Deliberately not an `nn.Module`: the network is upstream's and is built inside the
    worker process. Anything built here would be a second, competing implementation.
    """

    task = "shadow_removal"
    key = ""
    default_loss = "charbonnier"
    requires = ("torch", "einops", "timm")

    def __init__(self, **kwargs: Any) -> None:
        self.extra_kwargs = kwargs

    def describe(self) -> str:
        return describe(self.key)


def worker_command(config_path: str, phase: str, splits: str = "", weights: str = "") -> List[str]:
    command = [sys.executable, "-m", WORKER_MODULE, "--config", str(config_path), "--phase", phase]
    if splits:
        command += ["--splits", splits]
    if weights:
        command += ["--weights", str(weights)]
    return command


def _launch(command: List[str], cwd: Path) -> None:
    environment = dict(os.environ)
    environment.setdefault("PYTHONUTF8", "1")
    print(f"[symtrs] {' '.join(command)}")
    completed = subprocess.run(command, cwd=str(cwd), env=environment)
    if completed.returncode != 0:
        raise RuntimeError(
            f"the shadow-removal worker exited with code {completed.returncode}. Its own "
            "output above says why; this wrapper does not reinterpret it."
        )


# ---------------------------------------------------------------------- scoring


def score_split(cfg, run_dir: Path, split: str) -> Dict[str, float]:
    """Score `<run_dir>/predictions/<split>` against the staged targets and masks."""
    import numpy as np
    from PIL import Image

    from symtrs.metrics.shadow import aggregate, score_image
    from symtrs.model.shadow.staging import staged_keys

    staging = run_dir / "staging" / split
    keys = staged_keys(staging)
    predictions = run_dir / "predictions" / split
    missing = [k for k in keys if not (predictions / f"{k}.png").is_file()]
    if missing:
        raise RuntimeError(
            f"{len(missing)} of {len(keys)} {split} predictions are missing from "
            f"{predictions} (first: {missing[0]}); the worker did not finish. Re-score with "
            f"`symtrs eval <config> --eval-split {split}` once it has."
        )

    rows = []
    per_image = []
    for index, key in enumerate(keys):
        prediction = np.asarray(Image.open(predictions / f"{key}.png").convert("RGB"), np.float32) / 255.0
        target = np.asarray(Image.open(staging / "target" / f"{key}.png").convert("RGB"), np.float32) / 255.0
        mask = np.asarray(Image.open(staging / "mask" / f"{key}.png").convert("L")) > 127
        row = score_image(prediction, target, mask)
        rows.append(row)
        per_image.append({"key": key, **row})
        if index < cfg.eval.num_visualizations:
            _write_preview(run_dir / "preview" / f"{split}_{key}.png", staging, key, prediction, mask)

    (run_dir / f"per_image_{split}.json").write_text(json_dumps_safe(per_image, indent=1), encoding="utf-8")
    metrics = aggregate(rows)
    append_metrics_row(run_dir / "metrics.csv", {"iteration": -1, "split": split, **metrics})
    print(
        f"[symtrs] {split}: psnr={metrics['psnr']:.3f} ssim={metrics['ssim']:.4f} "
        f"mae_lab={metrics['mae_lab']:.3f} | shadow psnr={metrics.get('psnr_shadow', float('nan')):.3f} "
        f"non-shadow psnr={metrics.get('psnr_nonshadow', float('nan')):.3f} ({len(keys)} images)"
    )
    return metrics


def _write_preview(path: Path, staging: Path, key: str, prediction, mask) -> None:
    """input | prediction | target | mask, each downsampled to 512 px, in one strip."""
    import numpy as np
    from PIL import Image

    def fit(image: "Image.Image") -> "Image.Image":
        return image.convert("RGB").resize((512, 512), Image.BILINEAR)

    tiles = [
        fit(Image.open(staging / "input" / f"{key}.png")),
        fit(Image.fromarray((prediction * 255).round().astype(np.uint8))),
        fit(Image.open(staging / "target" / f"{key}.png")),
        fit(Image.fromarray(mask.astype(np.uint8) * 255)),
    ]
    strip = Image.new("RGB", (512 * len(tiles), 512))
    for i, tile in enumerate(tiles):
        strip.paste(tile, (512 * i, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    strip.save(path)


# ------------------------------------------------------------------ the entry point


def check_config(cfg, mode: str) -> None:
    """Refuse the configurations that cannot work, each with the flag that fixes it."""
    if cfg.resolution != 2048:
        raise ValueError(
            f"shadow removal is 2048-only: rgb_noshadow has no downsampled twin, so there "
            f"is no target at {cfg.resolution}px. Got resolution {cfg.resolution} from the "
            "config; use --resolution 2048."
        )
    if mode == "zero_shot":
        raise ValueError(
            "zero-shot does not apply to shadow removal: the released weights were trained "
            "on ISTD, not SyMTRS, so scoring them would not be a SyMTRS baseline. Train "
            "first, or score a checkpoint with `symtrs eval --task shadow_removal`."
        )
    if mode not in ("finetune", "score"):
        raise ValueError(f"unknown shadow-removal mode {mode!r}; expected finetune or score")
    if not cfg.train.patch_size:
        raise ValueError(
            "shadow removal trains on crops of the 2048 images and needs train.patch_size "
            "(256 or 320 in the shipped configs); 0 would feed a whole 2048 px image to "
            "the network. Set it with `--set train.patch_size=256`."
        )
    if not cfg.eval.tile_size:
        raise ValueError(
            "shadow removal predicts 2048 px images tile by tile and needs eval.tile_size "
            "(the training patch); 0 would run a whole 2048 px image in one pass, which "
            "does not fit in 12 GB. Set it with `--set eval.tile_size=256`."
        )


def run_shadow_removal(
    cfg, mode: str = "finetune", split: Optional[str] = None, checkpoint: str = ""
) -> Dict[str, Any]:
    """Train and/or score one shadow-removal model.

    Called by `symtrs.engine.trainer` before it builds anything: there is no SyMTRS
    network to build, the class is upstream's.
    """
    from symtrs.model.shadow.staging import stage_split
    from symtrs.registry import get_spec

    check_config(cfg, mode)
    spec = get_spec(cfg.model)
    arch = arch_for(spec.key)
    problem = upstream_missing(spec.key)
    if problem:
        raise FileNotFoundError(problem)

    run_dir = cfg.resolved_out_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    # The worker runs with the upstream checkout as its working directory, so every path
    # it reads from config.yaml or argv must already be absolute.
    if cfg.shadow.init_weights:
        cfg.shadow.init_weights = str(Path(cfg.shadow.init_weights).resolve())
    if checkpoint:
        checkpoint = str(Path(checkpoint).resolve())
    cfg.data.root = str(Path(cfg.data.root).resolve())
    config_path = run_dir / "config.yaml"
    cfg.to_yaml(str(config_path))
    print(f"[symtrs] {describe(spec.key)}")

    shadow = cfg.shadow
    scored = [split] if split else [cfg.data.val_split, cfg.data.test_split]
    wanted = ([cfg.data.split] if mode == "finetune" else []) + scored
    for name in dict.fromkeys(wanted):
        stage_split(cfg.data.root, str(run_dir), name, shadow.mask_threshold, shadow.mask_open)

    summary: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "task": cfg.task,
        "model": cfg.model,
        "resolution": cfg.resolution,
        "mode": mode,
        "epochs": cfg.train.epochs,
        "upstream": {"repo": arch.repo, "class": arch.cls},
        "mask": {"threshold": shadow.mask_threshold, "open": shadow.mask_open},
    }

    if mode == "finetune":
        _launch(worker_command(str(config_path), "train"), arch.repo_path)
    stats = run_dir / "train_stats.json"
    if stats.is_file():
        summary["train"] = json.loads(stats.read_text(encoding="utf-8"))
        summary["best_epoch"] = summary["train"].get("best_iteration", "")

    if not shadow.skip_final_eval:
        _launch(
            worker_command(str(config_path), "predict", ",".join(scored), checkpoint),
            arch.repo_path,
        )
        for name in scored:
            label = "test" if name == cfg.data.test_split else "val" if name == cfg.data.val_split else name
            summary[label] = score_split(cfg, run_dir, name)
    (run_dir / "summary.json").write_text(json_dumps_safe(summary, indent=2), encoding="utf-8")
    return summary


__all__ = [
    "CSV_FIELDS",
    "ShadowRemoval",
    "WORKER_MODULE",
    "append_metrics_row",
    "check_config",
    "describe",
    "run_shadow_removal",
    "score_split",
    "worker_command",
]
