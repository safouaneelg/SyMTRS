"""The process that trains and runs the upstream shadow-removal networks.

Launched by `runner.py` as `python -m symtrs.model.shadow.worker`, with the upstream
checkout as the working directory. That is the whole reason it is a subprocess:
ShadowFormer's `model.py` does `import utils`, and `utils` is a name the `symtrs` process
must never inherit (AGENTS.md section 7). With the checkout as cwd, `-m` puts it first on
`sys.path` without this package writing to `sys.path`.

It imports the *network class* from the pinned upstream source and nothing else of theirs
(see `runner.py` for why their training scripts are not used), then:

  --phase train    random-crop training with one loop for all three networks, validation on
                   a fixed set of val crops to pick `weights/best.pt`;
  --phase predict  tiled full-image inference of the val/test splits into
                   `<run_dir>/predictions/<split>/<key>.png`, scored by the parent.

Training recipe: Charbonnier loss, AdamW, linear warmup then cosine decay per iteration,
bf16 autocast, random flips and 90-degree rotations - the recipe the three papers share.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from symtrs.config import RunConfig
from symtrs.model.shadow.archs import Arch, arch_for
from symtrs.model.shadow.runner import append_metrics_row
from symtrs.model.shadow.staging import staged_keys


# ------------------------------------------------------------------------ network


def build_network(cfg: RunConfig) -> Tuple[torch.nn.Module, Arch]:
    """Import the upstream class from its pinned source file and construct it."""
    from symtrs.registry import get_spec

    spec = get_spec(cfg.model)
    arch = arch_for(spec.key)
    if not arch.source_path.is_file():
        raise FileNotFoundError(
            f"{arch.source_path} is missing; run `python scripts/fetch_third_party.py`"
        )
    module_spec = importlib.util.spec_from_file_location(
        f"symtrs_upstream_{arch.repo.lower()}", arch.source_path
    )
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    kwargs: Dict[str, Any] = dict(arch.kwargs)
    kwargs.update(cfg.model_kwargs.get("arch_kwargs", {}) or {})
    if arch.takes_img_size:
        kwargs["img_size"] = cfg.train.patch_size
    return getattr(module, arch.cls)(**kwargs), arch


def forward(net: torch.nn.Module, arch: Arch, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return net(image, mask) if arch.takes_mask else net(image)


#: Fraction of the model's tensors an init file must supply. A checkpoint of the wrong
#: architecture (BiasFree vs WithBias Restormer, another `embed_dim`) matches a handful of
#: shape-compatible tensors by name and used to "succeed", silently starting from scratch
#: while the summary claimed an initialisation. Below this, the run stops instead.
MIN_INIT_FRACTION = 0.9


def load_matching(
    net: torch.nn.Module, path: str, min_fraction: float = MIN_INIT_FRACTION
) -> Tuple[int, int]:
    """Initialise from an upstream checkpoint, keeping only tensors whose name and shape fit.

    Returns `(loaded, total)` tensor counts. Raises `RuntimeError` naming the file and the
    counts when fewer than `min_fraction` of the model's tensors were loaded.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model") or payload.get("state_dict") or payload.get("params") or payload
    state = {k.removeprefix("module."): v for k, v in state.items() if hasattr(v, "shape")}
    own = net.state_dict()
    kept = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    total = len(own)
    if total == 0 or len(kept) < min_fraction * total:
        raise RuntimeError(
            f"{path}: only {len(kept)} of the model's {total} tensors match by name and shape "
            f"({len(state)} tensors in the file); at least {min_fraction:.0%} are required. "
            "The file is for another architecture or configuration (see symtrs/model/shadow/"
            "archs.py for the constructor arguments), or is not a weights file; starting from "
            "it would be a silent from-scratch run."
        )
    net.load_state_dict(kept, strict=False)
    print(f"[symtrs] initialised {len(kept)}/{total} tensors from {path}")
    return len(kept), total


# ------------------------------------------------------------------------- data


def read_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))


def read_mask(path: Path) -> np.ndarray:
    from PIL import Image

    return (np.asarray(Image.open(path).convert("L")) > 127).astype(np.uint8)


class Store:
    """A split held in RAM as uint8 arrays: `{key: (input, target, mask)}`.

    PNG decoding of three 2048 px images costs about a second, which would dominate a
    crop-based iteration; 96 training triples are ~3 GB, loaded once by a thread pool
    (PIL decodes without holding the GIL).
    """

    def __init__(self, staging: Path, limit: int = 0) -> None:
        self.keys = staged_keys(staging)
        if limit:
            self.keys = self.keys[:limit]
        if not self.keys:
            raise RuntimeError(f"nothing staged in {staging}")

        def load(key: str):
            return (
                read_rgb(staging / "input" / f"{key}.png"),
                read_rgb(staging / "target" / f"{key}.png"),
                read_mask(staging / "mask" / f"{key}.png"),
            )

        started = time.time()
        with ThreadPoolExecutor(max_workers=8) as pool:
            self.items = list(pool.map(load, self.keys))
        print(f"[symtrs] loaded {len(self.keys)} images from {staging} in {time.time() - started:.0f}s")

    def __len__(self) -> int:
        return len(self.keys)

    def crop(self, index: int, top: int, left: int, size: int):
        image, target, mask = self.items[index]
        sl = (slice(top, top + size), slice(left, left + size))
        return image[sl], target[sl], mask[sl]


def to_tensor(arrays: List[np.ndarray], device: torch.device, channels: bool) -> torch.Tensor:
    stacked = np.stack(arrays)
    tensor = torch.from_numpy(stacked).to(device)
    tensor = tensor.permute(0, 3, 1, 2) if channels else tensor.unsqueeze(1)
    return tensor.float() / (255.0 if channels else 1.0)


def augment(arrays: Tuple[np.ndarray, ...], rng: np.random.Generator) -> Tuple[np.ndarray, ...]:
    """One of the eight flips / 90-degree rotations, applied identically to every array."""
    k = int(rng.integers(4))
    flip = bool(rng.integers(2))
    out = []
    for a in arrays:
        a = np.rot90(a, k)
        out.append(np.ascontiguousarray(a[:, ::-1] if flip else a))
    return tuple(out)


def sample_batch(store: Store, size: int, batch: int, order: List[int], rng, device):
    images, targets, masks = [], [], []
    for _ in range(batch):
        index = order.pop()
        h, w = store.items[index][0].shape[:2]
        top = int(rng.integers(0, h - size + 1))
        left = int(rng.integers(0, w - size + 1))
        i, t, m = augment(store.crop(index, top, left, size), rng)
        images.append(i), targets.append(t), masks.append(m)
    return (
        to_tensor(images, device, True),
        to_tensor(targets, device, True),
        to_tensor(masks, device, False),
    )


def charbonnier(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt((prediction - target) ** 2 + eps * eps).mean()


def psnr_of(prediction: torch.Tensor, target: torch.Tensor) -> float:
    mse = ((prediction.clamp(0, 1) - target) ** 2).flatten(1).mean(1).clamp_min(1e-10)
    return float((10 * torch.log10(1.0 / mse)).mean())


# ----------------------------------------------------------------------- training


def schedule(step: int, total: int, warmup: int, floor: float) -> float:
    """Learning-rate multiplier: linear warmup, then cosine decay to `floor`."""
    if warmup and step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def amp_context(cfg: RunConfig, device: torch.device):
    enabled = bool(cfg.train.amp) and device.type == "cuda"
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled)


def save_state(path: Path, net, optimizer, iteration: int, best: float, cfg: RunConfig, metrics) -> None:
    from symtrs.common.checkpoint import save_checkpoint

    save_checkpoint(
        str(path),
        {"model": net.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration,
         "best": best, "metrics": metrics},
        {"task": cfg.task, "model_key": cfg.model, "resolution": cfg.resolution, "seed": cfg.seed,
         "model_kwargs": dict(cfg.model_kwargs), "config": cfg.to_dict()},
    )


def validate_crops(net, arch, store: Store, crops, device, cfg) -> float:
    net.eval()
    scores = []
    with torch.no_grad(), amp_context(cfg, device):
        for start in range(0, len(crops), max(1, cfg.eval.batch_size) * 4):
            chunk = crops[start : start + max(1, cfg.eval.batch_size) * 4]
            images = to_tensor([store.crop(*c)[0] for c in chunk], device, True)
            targets = to_tensor([store.crop(*c)[1] for c in chunk], device, True)
            masks = to_tensor([store.crop(*c)[2] for c in chunk], device, False)
            scores.append(psnr_of(forward(net, arch, images, masks).float(), targets))
    net.train()
    return float(np.mean(scores))


def train(cfg: RunConfig) -> None:
    from symtrs.common.seed import describe_device, resolve_device, set_seed

    set_seed(cfg.seed)
    device = resolve_device(cfg.train.device)
    print(f"[symtrs] {describe_device(device)}")
    run_dir = cfg.resolved_out_dir
    weights = run_dir / "weights"
    weights.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "metrics.csv"

    net, arch = build_network(cfg)
    init_counts = None
    if cfg.shadow.init_weights:
        init_counts = load_matching(net, cfg.shadow.init_weights)
    net.to(device).train()
    print(f"[symtrs] {arch.repo}.{arch.cls}: {sum(p.numel() for p in net.parameters()) / 1e6:.1f}M parameters")

    optimizers = {"adamw": torch.optim.AdamW, "adam": torch.optim.Adam}
    if cfg.train.optim not in optimizers:
        raise ValueError(
            f"train.optim={cfg.train.optim!r} is not supported for shadow removal; "
            f"use one of {sorted(optimizers)}"
        )
    optimizer = optimizers[cfg.train.optim](
        net.parameters(), lr=cfg.train.lr, betas=(0.9, 0.999), eps=1e-8,
        weight_decay=cfg.train.weight_decay,
    )

    train_store = Store(run_dir / "staging" / cfg.data.split, cfg.data.max_samples)
    val_store = Store(run_dir / "staging" / cfg.data.val_split, cfg.data.max_samples)
    batch = cfg.train.batch_size
    size = cfg.train.patch_size
    per_epoch = math.ceil(len(train_store) / batch)
    total = cfg.shadow.iterations or cfg.train.epochs * per_epoch
    warmup = cfg.train.warmup_epochs * per_epoch
    warmup = min(warmup, total // 2)

    rng = np.random.default_rng(cfg.seed)
    val_rng = np.random.default_rng(cfg.seed + 1)
    crops = []
    for _ in range(cfg.shadow.val_crops):
        index = int(val_rng.integers(len(val_store)))
        h, w = val_store.items[index][0].shape[:2]
        crops.append((index, int(val_rng.integers(0, h - size + 1)), int(val_rng.integers(0, w - size + 1)), size))

    start = 0
    best = -1.0
    best_iteration = 0
    if cfg.train.resume:
        from symtrs.common.checkpoint import load_checkpoint

        state = load_checkpoint(cfg.train.resume)
        net.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start, best = int(state["iteration"]), float(state.get("best", -1.0))
        print(f"[symtrs] resumed {cfg.train.resume} at iteration {start}/{total}")

    print(
        f"[symtrs] training {total} iterations (batch {batch}, patch {size}, "
        f"{per_epoch} it/epoch, warmup {warmup}) on {len(train_store)} images"
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    order: List[int] = []
    loss_sum, loss_n, window_start = 0.0, 0, time.time()
    train_started = time.time()
    timed_seconds, timed_iterations = 0.0, 0

    for iteration in range(start, total):
        for group in optimizer.param_groups:
            group["lr"] = cfg.train.lr * schedule(iteration, total, warmup, cfg.train.min_lr_ratio)
        if len(order) < batch:
            order = list(rng.permutation(len(train_store))) + order
        step_started = time.time()
        images, targets, masks = sample_batch(train_store, size, batch, order, rng, device)
        with amp_context(cfg, device):
            prediction = forward(net, arch, images, masks)
        loss = charbonnier(prediction.float().clamp(0, 1), targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.train.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.train.grad_clip)
        optimizer.step()
        loss_sum += float(loss)  # also synchronises, so the timing below is real
        loss_n += 1
        if iteration - start >= 2:  # the first iterations include cudnn/kernel warm-up
            timed_seconds += time.time() - step_started
            timed_iterations += 1

        done = iteration + 1
        if done % max(1, cfg.train.log_every) == 0 or done == total:
            elapsed = time.time() - window_start
            vram = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"[symtrs] it {done}/{total} loss={loss_sum / loss_n:.4f} lr={lr:.2e} "
                f"{elapsed / loss_n:.2f}s/it peak_vram={vram:.2f}GB"
            )
            append_metrics_row(csv_path, {
                "iteration": done, "split": "train", "loss": loss_sum / loss_n, "lr": lr,
                "seconds": elapsed / loss_n, "vram_gb": vram,
            })
            loss_sum, loss_n, window_start = 0.0, 0, time.time()

        every = cfg.shadow.val_every
        if done == total or (every and done % every == 0):
            score = validate_crops(net, arch, val_store, crops, device, cfg)
            append_metrics_row(csv_path, {"iteration": done, "split": "val_crops", "psnr": score})
            print(f"[symtrs] it {done}: val-crop psnr={score:.3f} (best {max(best, score):.3f})")
            if score > best:
                best, best_iteration = score, done
                save_state(weights / "best.pt", net, optimizer, done, best, cfg, {"val_crop_psnr": score})
            save_state(weights / "last.pt", net, optimizer, done, best, cfg, {"val_crop_psnr": score})

    stats = {
        "iterations": total,
        "iterations_per_epoch": per_epoch,
        "batch_size": batch,
        "patch_size": size,
        "best_iteration": best_iteration,
        "best_val_crop_psnr": best,
        "seconds_per_iteration": timed_seconds / max(1, timed_iterations),
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0,
        "train_seconds": time.time() - train_started,
        "init_tensors_loaded": init_counts[0] if init_counts else None,
        "init_tensors_total": init_counts[1] if init_counts else None,
    }
    (run_dir / "train_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"[symtrs] trained: {json.dumps(stats)}")


# --------------------------------------------------------------------- prediction


def predict(cfg: RunConfig, splits: List[str], weights: str) -> None:
    from PIL import Image

    from symtrs.common.checkpoint import load_checkpoint
    from symtrs.common.seed import resolve_device
    from symtrs.engine.tiling import predict_tiled

    device = resolve_device(cfg.train.device)
    run_dir = cfg.resolved_out_dir
    net, arch = build_network(cfg)
    path = weights or str(run_dir / "weights" / "best.pt")
    state = load_checkpoint(path)
    net.load_state_dict(state["model"])
    net.to(device).eval()
    print(f"[symtrs] predicting with {path} (iteration {state.get('iteration')})")

    def tile_forward(tile: torch.Tensor) -> torch.Tensor:
        # The mask rides along as a fourth channel so the generic tiler can crop it with
        # the image; only ShadowFormer reads it.
        with amp_context(cfg, device):
            return forward(net, arch, tile[:, :3], tile[:, 3:]).float()

    tile, overlap = cfg.eval.tile_size, cfg.eval.tile_overlap
    for split in splits:
        staging = run_dir / "staging" / split
        out = run_dir / "predictions" / split
        out.mkdir(parents=True, exist_ok=True)
        keys = staged_keys(staging)
        started = time.time()
        for n, key in enumerate(keys, 1):
            image = torch.from_numpy(read_rgb(staging / "input" / f"{key}.png")).to(device)
            mask = torch.from_numpy(read_mask(staging / "mask" / f"{key}.png")).to(device)
            packed = torch.cat([image.permute(2, 0, 1).float() / 255.0, mask[None].float()], dim=0)[None]
            with torch.no_grad():
                restored = predict_tiled(packed, tile_forward, tile, overlap)
            array = (restored[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
            Image.fromarray(array).save(out / f"{key}.png")
            if n % 5 == 0 or n == len(keys):
                print(f"[symtrs] {split}: {n}/{len(keys)} predicted ({(time.time() - started) / n:.1f}s/img)")
    if device.type == "cuda":
        print(f"[symtrs] predict peak_vram={torch.cuda.max_memory_allocated() / 2**30:.2f}GB")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", choices=["train", "predict"], required=True)
    parser.add_argument("--splits", default="val,test")
    parser.add_argument("--weights", default="")
    args = parser.parse_args(argv)

    cfg = RunConfig.from_yaml(args.config)
    if args.phase == "train":
        train(cfg)
    else:
        predict(cfg, [s for s in args.splits.split(",") if s], args.weights)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
