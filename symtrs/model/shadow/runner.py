"""Shadow removal driven through the upstream networks.

    symtrs train configs/shadow_removal/shadowformer.yaml --data-root /data/SyMTRS
    symtrs train --task shadow_removal --model restormer --data-root /data/SyMTRS \\
                 --epochs 20 --set shadow.iterations=500

Why a worker instead of the upstream `train.py` scripts: all of them were written for
256-320 px ISTD crops. They train by epoch over a loader with hard-coded folder names
(`train_A/B/C`), hard-code CUDA and `DataParallel`, and validate on the *whole* image in
one forward pass - at 2048 px that does not fit in 12 GB for ShadowFormer or Restormer.
Driving four different loops and parsers would also give three different notions of
"best" and of "score". So the worker (`worker.py`) imports only each upstream's network
class and runs one loop and one tiled inference for all four; the metric code is
`symtrs.metrics.shadow`, run here in the parent on the PNGs the worker wrote. The networks
are not reimplemented: a model whose class is not in the pinned checkout fails with the
command that fetches it.
"""

from __future__ import annotations

import csv
import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from symtrs.metrics.shadow import SSIM_NOTE
from symtrs.model.shadow.archs import arch_for, upstream_missing
from symtrs.common.jsonio import dumps as json_dumps_safe

WORKER_MODULE = "symtrs.model.shadow.worker"

#: Upper bound on the processes `score_split` uses (see the comment there).
SCORE_WORKERS_MAX = 4

#: Every column `metrics.csv` can hold, fixed up front: the worker appends training rows
#: and the parent appends scoring rows, and a CSV whose header was taken from the first
#: row would misalign the others.
CSV_FIELDS = (
    "iteration", "split", "loss", "lr", "seconds", "vram_gb",
    "psnr", "ssim_y", "mae_lab", "rmse_lab",
    "psnr_shadow", "ssim_y_shadow", "mae_lab_shadow", "rmse_lab_shadow",
    "psnr_nonshadow", "ssim_y_nonshadow", "mae_lab_nonshadow", "rmse_lab_nonshadow",
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
    text = (
        f"{key} (shadow_removal): upstream {arch.repo}/{arch.source}:{arch.cls}, "
        "rgb_day -> rgb_noshadow at 2048 (trained on crops, tiled at inference)"
    )
    if not arch.released_weights:
        text += (
            "; no released weights: trained from scratch, so not like-for-like with a model "
            "that starts from a checkpoint"
        )
    return text


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

    from symtrs.metrics.shadow import aggregate
    from symtrs.model.shadow.staging import staged_keys

    staging = run_dir / "staging" / split
    keys = staged_keys(staging)
    predictions = run_dir / "predictions" / split
    marker = predictions / RETAIN_MARKER
    if marker.is_file():
        record = json.loads(marker.read_text(encoding="utf-8"))
        raise RuntimeError(
            f"the {split} predictions in {predictions} were reduced to {record.get('size')} px "
            "when they were last scored (eval.retain_predictions="
            f"{record.get('mode')!r}); scoring them now would measure them against "
            f"{staging / 'target'} at the ground-truth size. Re-run prediction first: "
            f"`symtrs eval <config> --eval-split {split}` regenerates them from the checkpoint."
        )
    missing = [k for k in keys if not (predictions / f"{k}.png").is_file()]
    if missing:
        raise RuntimeError(
            f"{len(missing)} of {len(keys)} {split} predictions are missing from "
            f"{predictions} (first: {missing[0]}); the worker did not finish. Re-score with "
            f"`symtrs eval <config> --eval-split {split}` once it has."
        )

    # One 2048 px image costs ~24 s of single-threaded numpy (the SSIM filter dominates), so
    # 3000 test images scored serially took ~20 h and held a GPU slot of the queue the whole
    # time (pix2pix_shadow2noshadow, 2026-10-27). The images are independent, so they are
    # scored in processes. The count follows shadow.mask_workers (the other CPU-side shadow
    # step) but is capped at SCORE_WORKERS_MAX: scoring runs beside GPU jobs whose loaders
    # need the cores. `spawn`, not fork: the parent may already hold torch/CUDA state.
    jobs = [(str(predictions / f"{k}.png"), str(staging / "target" / f"{k}.png"),
             str(staging / "mask" / f"{k}.png")) for k in keys]
    workers = max(1, min(SCORE_WORKERS_MAX, cfg.shadow.mask_workers, len(jobs)))
    if workers == 1:
        rows = [_score_files(job) for job in jobs]
    else:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        print(f"[symtrs] scoring {len(jobs)} {split} images in {workers} processes")
        with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            rows = list(pool.map(_score_files, jobs, chunksize=8))
    per_image = [{"key": key, **row} for key, row in zip(keys, rows)]
    scene_keys = preview_keys(per_image, cfg.eval.num_visualizations)
    for key in scene_keys:
        prediction = np.asarray(Image.open(predictions / f"{key}.png").convert("RGB"), np.float32) / 255.0
        mask = np.asarray(Image.open(staging / "mask" / f"{key}.png").convert("L")) > 127
        _write_preview(run_dir / "preview" / f"{split}_{key}.png", staging, key, prediction, mask)

    (run_dir / f"per_image_{split}.json").write_text(json_dumps_safe(per_image, indent=1), encoding="utf-8")
    metrics = aggregate(rows)
    append_metrics_row(run_dir / "metrics.csv", {"iteration": -1, "split": split, **metrics})
    print(
        f"[symtrs] {split}: psnr={metrics['psnr']:.3f} ssim_y={metrics['ssim_y']:.4f} "
        f"mae_lab={metrics['mae_lab']:.3f} | shadow psnr={metrics.get('psnr_shadow', float('nan')):.3f} "
        f"non-shadow psnr={metrics.get('psnr_nonshadow', float('nan')):.3f} ({len(keys)} images)"
    )
    # Last, and only once every row, preview and CSV line above is on disk: the images are
    # the input to scoring, not an output of it.
    retain_predictions(cfg, run_dir, split, keys, scene_keys)
    return metrics


def _score_files(paths) -> Dict[str, Optional[float]]:
    """`score_image` on one (prediction, target, mask) triple of PNG paths.

    Top-level and path-based so a spawned process can run it: it reads its own images rather
    than receiving three 2048 px arrays through a pipe.
    """
    import numpy as np
    from PIL import Image

    from symtrs.metrics.shadow import score_image

    prediction_path, target_path, mask_path = paths
    prediction = np.asarray(Image.open(prediction_path).convert("RGB"), np.float32) / 255.0
    target = np.asarray(Image.open(target_path).convert("RGB"), np.float32) / 255.0
    mask = np.asarray(Image.open(mask_path).convert("L")) > 127
    return score_image(prediction, target, mask)


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


# --------------------------------------------------------- what survives scoring

#: Longest side of the copy kept when `eval.retain_predictions` is "downsample".
RETAINED_SIZE = 512

#: Written into a downsampled `<run_dir>/predictions/<split>/` so `score_split` refuses to
#: measure it: the files are still named `<key>.png`, but they are not the size the metrics
#: assume. Kept in the directory itself, where whoever re-runs scoring will see it.
RETAIN_MARKER = "_retained.json"

_HEIGHT = re.compile(r"_h(\d{2})_")


def preview_keys(rows: List[Dict[str, Any]], count: int) -> List[str]:
    """Up to `count` scenes to preview and to keep at full resolution: one per camera height,
    each the one whose shadow fraction is closest to that height's median.

    These used to be the first `count` keys. The keys sort by location first, so that was the
    same location at one altitude `count` times over - a preview strip that showed one scene
    four times, and, once the rest are downsampled, the only scene a figure could zoom into.
    "Typical" rather than "best" for the same reason the figures choose that way: a reader of
    the paper should see what the table's mean looks like.
    """
    if count <= 0 or not rows:
        return []
    fraction = {row["key"]: row.get("shadow_fraction") for row in rows}
    by_height: Dict[str, List[str]] = {}
    for row in rows:
        match = _HEIGHT.search(row["key"])
        if match:
            by_height.setdefault(match.group(1), []).append(row["key"])
    if not by_height:
        # No altitude in the key (a synthetic dataset): spread along the split instead.
        step = max(1, len(rows) // count)
        return [row["key"] for row in rows[::step][:count]]
    heights = sorted(by_height)
    if len(heights) > count:
        # Evenly across the altitudes, keeping order and never repeating one.
        pick = sorted({int(round(i * (len(heights) - 1) / (count - 1))) for i in range(count)}) if count > 1 else [0]
        heights = [heights[i] for i in pick]
    chosen: List[str] = []
    for height in heights:
        group = [key for key in by_height[height] if fraction.get(key) is not None]
        if not group:
            chosen.append(by_height[height][0])
            continue
        median = statistics.median(fraction[key] for key in group)
        chosen.append(min(group, key=lambda key: abs(fraction[key] - median)))
    return chosen


def retain_predictions(cfg, run_dir: Path, split: str, keys: List[str], scene_keys: List[str]) -> None:
    """Apply `eval.retain_predictions` to the PNGs `score_split` just read.

    Only the shadow and domain scorers write predictions to disk, and one 2048 px lossless
    PNG is ~15 MB: the four shadow-removal runs held 170 GB of them for numbers that are all
    in `per_image_<split>.json`. The images are the scorer's input, not its output, so what
    a reader needs afterwards is a look at any scene ("downsample"), the scenes the figures
    zoom into, and the metrics. Deleting more is safe: re-scoring re-runs prediction.
    """
    mode = cfg.eval.retain_predictions
    if mode == "all":
        return
    import shutil

    predictions = run_dir / "predictions" / split
    kept: List[str] = []
    if mode == "downsample":
        full = run_dir / "preview" / "full"
        for key in scene_keys:
            full.mkdir(parents=True, exist_ok=True)
            source = predictions / f"{key}.png"
            if source.is_file():
                source.replace(full / f"{split}_{key}.png")
                kept.append(f"preview/full/{split}_{key}.png")
    if mode == "none":
        # Every file in it is a prediction image; nothing else is written there.
        shutil.rmtree(predictions)
        print(
            f"[symtrs] {split}: removed {len(keys)} prediction images "
            "(eval.retain_predictions=none); metrics and previews are kept"
        )
        return

    todo = [str(predictions / f"{key}.png") for key in keys if key not in set(scene_keys)]
    workers = max(1, min(SCORE_WORKERS_MAX, cfg.shadow.mask_workers, len(todo)))
    if workers == 1:
        freed = sum(_shrink_file(path) for path in todo)
    else:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            freed = sum(pool.map(_shrink_file, todo, chunksize=8))
    (predictions / RETAIN_MARKER).write_text(
        json_dumps_safe(
            {
                "mode": mode,
                "size": RETAINED_SIZE,
                "split": split,
                "full_resolution": kept,
                "note": "these are 512 px copies; re-run prediction before scoring them again",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"[symtrs] {split}: predictions downsampled to {RETAINED_SIZE} px, {freed / 2**30:.1f} GB "
        f"freed, {len(kept)} scene(s) kept at full resolution in preview/full/"
    )


def _shrink_file(path: str) -> int:
    """Rewrite one prediction PNG at `RETAINED_SIZE`, returning the bytes it gave back.

    Top-level and path-based so a spawned process can run it; it reads its own image rather
    than receiving a 2048 px array through a pipe.
    """
    from PIL import Image

    path = Path(path)
    before = path.stat().st_size
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((RETAINED_SIZE, RETAINED_SIZE), Image.LANCZOS)
        image.save(path)
    return before - path.stat().st_size


def retention_record(cfg, run_dir: Path) -> Dict[str, Any]:
    """What `summary.json` says about the images this run kept, and where to look."""
    full = run_dir / "preview" / "full"
    mode = cfg.eval.retain_predictions
    return {
        "retain": mode,
        "retained_size": RETAINED_SIZE if mode == "downsample" else None,
        "full_resolution": sorted(str(path.relative_to(run_dir)) for path in full.glob("*.png")),
        "note": (
            "every metric is in per_image_<split>.json; re-scoring re-runs prediction from "
            "the checkpoint, so deleting prediction images costs GPU time, never a number"
        ),
    }


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

    # The worker runs with the upstream checkout as its working directory, so every path
    # it reads from config.yaml or argv must already be absolute (out_dir included: the
    # worker derives its run directory from the config it reads).
    cfg = cfg.anchored()
    if checkpoint:
        checkpoint = str(Path(checkpoint).resolve())
    run_dir = cfg.resolved_out_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.yaml"
    cfg.to_yaml(str(config_path))
    print(f"[symtrs] {describe(spec.key)}")

    shadow = cfg.shadow
    scored = [split] if split else [cfg.data.val_split, cfg.data.test_split]
    wanted = ([cfg.data.split] if mode == "finetune" else []) + scored
    for name in dict.fromkeys(wanted):
        stage_split(
            cfg.data.root, str(run_dir), name, shadow.mask_threshold, shadow.mask_open,
            cache=shadow.mask_cache, workers=shadow.mask_workers,
        )

    summary: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "task": cfg.task,
        "model": cfg.model,
        "resolution": cfg.resolution,
        "mode": mode,
        "epochs": cfg.train.epochs,
        "upstream": {"repo": arch.repo, "class": arch.cls},
        "mask": {"threshold": shadow.mask_threshold, "open": shadow.mask_open},
        # Comparability: the models do not start from comparable weights, so every
        # result row states what it started from (None = random init).
        "init_weights": shadow.init_weights or None,
        "init_source_task": arch.init_source_task if shadow.init_weights else None,
        "metric_notes": {"ssim_y": SSIM_NOTE},
    }

    if mode == "finetune":
        _launch(worker_command(str(config_path), "train"), arch.repo_path)
    stats = run_dir / "train_stats.json"
    if stats.is_file():
        summary["train"] = json.loads(stats.read_text(encoding="utf-8"))
        summary["best_epoch"] = summary["train"].get("best_iteration", "")
        summary["init_tensors"] = {
            "loaded": summary["train"].get("init_tensors_loaded"),
            "total": summary["train"].get("init_tensors_total"),
        }

    if not shadow.skip_final_eval:
        # Start prediction from an empty folder: the predictor rewrites every key, but a
        # leftover from an earlier pass - a downsampled file, a retention marker, a key the
        # split no longer has - would either be scored as if it were fresh or refuse to be.
        import shutil

        for name in scored:
            shutil.rmtree(run_dir / "predictions" / name, ignore_errors=True)
        _launch(
            worker_command(str(config_path), "predict", ",".join(scored), checkpoint),
            arch.repo_path,
        )
        for name in scored:
            label = "test" if name == cfg.data.test_split else "val" if name == cfg.data.val_split else name
            summary[label] = score_split(cfg, run_dir, name)
        summary["predictions"] = retention_record(cfg, run_dir)
    (run_dir / "summary.json").write_text(json_dumps_safe(summary, indent=2), encoding="utf-8")
    return summary


__all__ = [
    "CSV_FIELDS",
    "RETAINED_SIZE",
    "RETAIN_MARKER",
    "ShadowRemoval",
    "WORKER_MODULE",
    "append_metrics_row",
    "check_config",
    "describe",
    "preview_keys",
    "retain_predictions",
    "retention_record",
    "run_shadow_removal",
    "score_split",
    "worker_command",
]
