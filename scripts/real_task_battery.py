#!/usr/bin/env python3
"""Zero-shot runs of the campaign checkpoints on *real* aerial imagery, task by task.

`scripts/zeroshot_real_data.py` answers R1.4 for semantic segmentation, the one real task
for which a labelled benchmark exists in the same conventions (ISPRS Potsdam). The other
four tasks have no labelled real counterpart in hand: there is no real day/night pair, no
real shadow-free ground truth, no real metric depth map and no real native multi-scale
render. This script therefore does the only honest thing that is left - it runs the released
checkpoints on real imagery and reports, per task:

  * **super-resolution** is measured. A real orthophoto crop is the HR target and its
    bicubic downsample is the LR input, so PSNR/SSIM still have a target. The LR is a
    downsample and not a real sensor's output; that is the one difference from the dataset
    and it is stated with every number. The bicubic baseline is recomputed here, with the
    same channel convention as the model it is compared against.
  * **monocular depth, day-to-night translation and shadow removal** have no target, so
    they are written out as images and read visually. For depth this is unavoidable: a true
    orthophoto has no perspective to recover metric depth from, and the oblique Swiss
    photographs have no depth ground truth.

Everything is fed at the ground sampling distance the checkpoint was trained at. Potsdam is
5 cm/px; the SyMTRS 2048 px level taken from 160 m is 9.03 cm/px, so a Potsdam source crop is
resized by 5/9.03 before cropping. Without that, a model would see a scene 1.8x finer than
its training data and the comparison would measure scale, not transfer. The Swiss
photographs have no stated GSD and are used as they are, for the qualitative rows only.

    python scripts/real_task_battery.py --potsdam top_potsdam_7_9_RGB.tif \\
        --swiss swiss_topo_test_image3.png --out runs/real_tasks

Writes `<out>/inputs.json`, `<out>/sr/sr.json`, `<out>/depth/`, `<out>/translation/`,
`<out>/shadow/`. Figures and the paper text are rendered from these by
`notebooks/real_tasks_figure.py`.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]
POTSDAM_GSD_M = 0.05        # ISPRS Potsdam true orthophoto, 5 cm/px
SYMTRS_2048_GSD_M = 0.0903  # 2048 px from 160 m AGL, 60 deg HFOV (AGENTS.md ground rule 3)
MATCH = POTSDAM_GSD_M / SYMTRS_2048_GSD_M

#: Runs under the campaign that this battery uses. The name is the run directory, so the
#: checkpoint, its config and its evaluation protocol come from one place.
SR_MODELS = ("srcnn", "autoencoder", "srgan", "swinir")
SR_SCALES = (2, 4, 8)
DEPTH_RUNS = ("midas_small_2048", "midas_dpt_hybrid_2048", "da3_metric_large_2048")
TRANSLATIONS = (
    ("day2night", "cyclegan_day2night"),
    ("day2night", "pix2pix_day2night"),
    ("shadow2noshadow", "cyclegan_shadow2noshadow"),
    ("shadow2noshadow", "pix2pix_shadow2noshadow"),
)
SHADOW_RUNS = ("restormer", "uformer", "shadowformer", "shadowmaskformer")

#: Upstream modules that import a sibling by its bare name and therefore need their own
#: checkout on `PYTHONPATH`. ShadowFormer's `model.py` does `import utils`; loading it in
#: this process would leave a top-level `utils` importable to the rest of the script, which
#: is the name collision `symtrs/model/shadow/worker.py` avoids by running in its own
#: process with the checkout as cwd. We do the same: one child per such model.
SHADOW_LOCAL_IMPORTS = {"shadowformer": "ShadowFormer"}


# --------------------------------------------------------------------------- inputs


def matched_potsdam(path: Path) -> "Any":
    """The Potsdam tile resampled to the SyMTRS 2048 px GSD (5 cm -> 9.03 cm)."""
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    image = Image.open(path).convert("RGB")
    side = (round(image.width * MATCH), round(image.height * MATCH))
    return image.resize(side, Image.LANCZOS)


def crop_at(image, origin: Tuple[int, int], side: int):
    left, top = origin
    return image.crop((left, top, left + side, top + side))


#: The four quadrants of the matched tile: each crop starts at an extreme corner, so the set
#: covers the roof/street block, the multi-lane junction, the water-side buildings and the
#: tree-lined park with its long cast shadows. Each crop is 2048 px, ~185 m like a SyMTRS view.
POTSDAM_FRACTIONS = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0))


def write_inputs(args) -> Dict[str, Any]:
    """Write the shared real-image set once, so every task reads the same pixels."""
    from PIL import Image

    out = args.out / "inputs"
    out.mkdir(parents=True, exist_ok=True)
    records: Dict[str, Any] = {"potsdam_scale": MATCH, "potsdam_gsd_m": SYMTRS_2048_GSD_M}

    tile = matched_potsdam(args.potsdam)
    names = []
    side = 2048
    for i, (fx, fy) in enumerate(POTSDAM_FRACTIONS):
        left = round(fx * (tile.width - side))
        top = round(fy * (tile.height - side))
        crop = crop_at(tile, (left, top), side)
        name = f"potsdam_{i}"
        crop.save(out / f"{name}.png")
        names.append(name)
    records["potsdam"] = names

    swiss = []
    for path in args.swiss:
        image = Image.open(path).convert("RGB")
        # Long side 2048: the model sees the same pixel count as a training view. No GSD is
        # claimed for these; they are the qualitative rows only.
        side = 2048
        ratio = side / max(image.size)
        image = image.resize((round(image.width * ratio), round(image.height * ratio)),
                             Image.LANCZOS)
        name = f"swiss_{Path(path).stem}"
        image.save(out / f"{name}.png")
        swiss.append(name)
    records["swiss"] = swiss
    (args.out / "inputs.json").write_text(json.dumps(records, indent=2) + "\n")
    print(f"[inputs] {len(names)} Potsdam crops, {len(swiss)} Swiss photographs -> {out}")
    return records


# ---------------------------------------------------------------- super-resolution


def _sr_data_root(out: Path, crops: List[Path]) -> Path:
    """A fake SyMTRS root with the real crops as HR and their bicubic LR at every scale.

    The front-end's own command builder turns `data.root` into `--lr_root/--hr_root`
    (`rgb_day_lr/xN` and `rgb_day`), so laying the real pairs out in that shape is what
    lets this reuse the trainer's evaluation code and its channel conventions rather than
    reimplementing them here.
    """
    from PIL import Image

    root = out / "sr_data"
    for split in ("train", "test"):
        for scale in SR_SCALES:
            (root / split / "rgb_day").mkdir(parents=True, exist_ok=True)
            (root / split / "rgb_day_lr" / f"x{scale}").mkdir(parents=True, exist_ok=True)
        for crop in crops:
            image = Image.open(crop)
            dst = root / split / "rgb_day" / crop.name
            if not dst.exists():
                shutil.copyfile(crop, dst)
            for scale in SR_SCALES:
                side = image.width // scale
                image.resize((side, side), Image.BICUBIC).save(
                    root / split / "rgb_day_lr" / f"x{scale}" / crop.name)
    return root


def _bicubic_baseline(hr: Path, lr: Path, use_y: bool) -> Tuple[float, float]:
    """PSNR/SSIM of the bicubic upsample, in the same convention as the model beside it."""
    import numpy as np
    import torch
    from PIL import Image

    from symtrs.legacy.superres.utils import psnr, rgb_to_y, ssim

    target = Image.open(hr).convert("RGB")
    source = Image.open(lr).convert("RGB").resize(target.size, Image.BICUBIC)
    a = torch.from_numpy(np.array(source)).permute(2, 0, 1).float()[None] / 255.0
    b = torch.from_numpy(np.array(target)).permute(2, 0, 1).float()[None] / 255.0
    if use_y:
        a, b = rgb_to_y(a), rgb_to_y(b)
    return float(psnr(a, b)), float(ssim(a, b))


def run_sr(args) -> Dict[str, Any]:
    import numpy as np

    from symtrs.config import RunConfig
    from symtrs.model.superres.runner import SuperResolution, expected_test_dirs

    out = args.out / "sr"
    out.mkdir(parents=True, exist_ok=True)
    crops = sorted((args.out / "inputs").glob("potsdam_*.png"))
    if not crops:
        raise SystemExit("run `--task inputs` first; there are no Potsdam crops")
    root = _sr_data_root(out, crops)

    results: Dict[str, Dict[str, Any]] = {"crops": [c.name for c in crops], "runs": {}}
    for model in SR_MODELS:
        for scale in SR_SCALES:
            run_dir = args.runs / f"{model}_x{scale}"
            config = run_dir / "config.yaml"
            if not config.is_file():
                print(f"[sr] no {config}; skipped")
                continue
            cfg = RunConfig.from_yaml(str(config))
            cfg = cfg.with_overrides({
                "data.root": str(root), "data.split": "train",
                "data.test_split": "test", "data.val_split": "val",
                "out_dir": str(out / f"{model}_x{scale}"),
                "train.device": args.device, "data.num_workers": args.workers,
            })
            target = Path(cfg.resolved_out_dir)
            (target / "weights").mkdir(parents=True, exist_ok=True)
            best = target / "weights" / "best.pt"
            if not best.exists():
                os.symlink((run_dir / "weights" / "best.pt").resolve(), best)
            descriptor = SuperResolution(**cfg.model_kwargs)
            descriptor.key = model
            command = descriptor.train_command(cfg, target, eval_only=True)
            print("[sr] " + " ".join(command))
            completed = subprocess.run(command)
            if completed.returncode != 0:
                raise RuntimeError(f"{model} x{scale} exited {completed.returncode}; see its output above")
            metrics = json.loads((target / "test_metrics.json").read_text())

            use_y = bool(cfg.model_kwargs.get("use_y_channel"))
            test_lr, test_hr = expected_test_dirs(str(root), "test", cfg.resolution)
            pairs = [(test_hr / c.name, test_lr / c.name) for c in crops]
            base = [_bicubic_baseline(hr, lr, use_y) for hr, lr in pairs]
            record = {
                "model": model, "scale": scale, "use_y_channel": use_y,
                "psnr": metrics["psnr"], "ssim": metrics["ssim"],
                "bicubic_psnr": float(np.mean([p for p, _ in base])),
                "bicubic_ssim": float(np.mean([s for _, s in base])),
                "images": metrics["images"],
            }
            results["runs"][f"{model}_x{scale}"] = record
            print(f"[sr] {model:11s} x{scale}  model {record['psnr']:.2f} dB / "
                  f"{record['ssim']:.3f}   bicubic {record['bicubic_psnr']:.2f} dB / "
                  f"{record['bicubic_ssim']:.3f}", flush=True)

    (out / "sr.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


# --------------------------------------------------------------------------- depth


def run_depth(args) -> Dict[str, Any]:
    import gc

    import numpy as np
    import torch
    from PIL import Image

    from symtrs.model.wrapper import autoloader

    out = args.out / "depth"
    out.mkdir(parents=True, exist_ok=True)
    images = [args.out / "inputs" / f"{name}.png" for name in args.depth_images]
    results: Dict[str, Any] = {}
    record_path = out / "depth.json"
    if record_path.is_file():
        results.update(json.loads(record_path.read_text()))
    for run in args.depth_models:
        weights = args.runs / run / "weights" / "best.pt"
        if not weights.is_file():
            print(f"[depth] no {weights}; skipped")
            continue
        model = autoloader(str(weights), device=args.device)
        for image in images:
            prediction = np.asarray(model.predict(Image.open(image).convert("RGB"),
                                                  tile_size=args.depth_tile))
            np.save(out / f"{image.stem}_{run}.npy", prediction)
            _save_depth_png(prediction, out / f"{image.stem}_{run}.png")
            finite = prediction[np.isfinite(prediction)]
            results[f"{image.stem}_{run}"] = {
                "image": image.name, "model": run,
                "min": float(finite.min()), "median": float(np.median(finite)),
                "max": float(finite.max()),
            }
            print(f"[depth] {run:24s} {image.stem:18s} "
                  f"p2 {np.percentile(finite, 2):.2f} "
                  f"median {np.median(finite):.2f} p98 {np.percentile(finite, 98):.2f}",
                  flush=True)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    (out / "depth.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def _save_depth_png(depth, path: Path) -> None:
    """A colour preview on a per-image 2-98 percentile stretch.

    The relative models have no metric scale without a ground truth, so the only honest
    picture is the ordering they produce. A shared stretch across models would make the
    comparison about the scale of each head rather than about what each one sees.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    finite = depth[np.isfinite(depth)]
    lo, hi = np.percentile(finite, 2), np.percentile(finite, 98)
    scaled = np.clip((depth - lo) / max(hi - lo, 1e-9), 0, 1)
    colour = (plt.get_cmap("turbo")(scaled)[..., :3] * 255).astype(np.uint8)
    Image.fromarray(colour).save(path)


# ------------------------------------------------------------------ translation


def _stage_translation(stage: Path, direction: str, aligned: bool, crops: List[Path],
                       partner: List[Path]) -> Path:
    """`testA`/`testB` (unaligned) or `test/` of A|B pairs (aligned), as upstream reads them.

    `crops` are PIL images (the real 256 px views) and `partner` are paths to SyMTRS images
    of the other domain; only the A side reaches `fake_B`, the B side is what the loader
    needs to exist.
    """
    from PIL import Image

    stage.mkdir(parents=True, exist_ok=True)
    if aligned:
        folder = stage / "test"
        folder.mkdir(exist_ok=True)
        for i, (a, b) in enumerate(zip(crops, partner)):
            left = a.convert("RGB").resize((256, 256), Image.LANCZOS)
            right = Image.open(b).convert("RGB").resize((256, 256), Image.LANCZOS)
            combo = Image.new("RGB", (512, 256))
            combo.paste(left, (0, 0))
            combo.paste(right, (256, 0))
            combo.save(folder / f"real_{i}.png")
    else:
        a_dir, b_dir = stage / "testA", stage / "testB"
        a_dir.mkdir(exist_ok=True)
        b_dir.mkdir(exist_ok=True)
        for i, (a, b) in enumerate(zip(crops, partner)):
            a.convert("RGB").resize((256, 256), Image.LANCZOS).save(a_dir / f"real_{i}.png")
            Image.open(b).convert("RGB").resize((256, 256), Image.LANCZOS).save(
                b_dir / f"real_{i}.png")
    return stage


def _symtrs_partners(root: Path, direction: str, count: int) -> List[Path]:
    """Source-domain images for the B side; `fake_B` never reads them, the loader does."""
    source = "rgb_night" if direction == "day2night" else "rgb_noshadow"
    files = sorted((root / "test" / source).glob("*.png"))[:count]
    if len(files) < count:
        raise SystemExit(f"need {count} images under {root}/test/{source}")
    return files


def run_translation(args) -> Dict[str, Any]:
    from PIL import Image

    upstream = REPO / "pytorch-CycleGAN-and-pix2pix"
    out = args.out / "translation"
    out.mkdir(parents=True, exist_ok=True)
    # The two real crops the translation rows use: one roof/street block and one shadowed
    # park crop, at the translator's own 256 px / 23 m working size.
    sources = [args.out / "inputs" / "potsdam_0.png",
               args.out / "inputs" / "potsdam_2.png"]
    crops = []
    for i, source in enumerate(sources):
        image = Image.open(source)
        crops.append(image.crop((300 + 256 * i, 300, 300 + 256 * i + 256, 556)))
        crops[-1].save(out / f"input_{i}.png")

    results: Dict[str, Any] = {}
    for direction, run in TRANSLATIONS:
        run_dir = args.runs / run
        weights = run_dir / "checkpoints" / f"symtrs_{run}"
        if not (weights / "latest_net_G.pth").exists() and not (weights / "latest_net_G_A.pth").exists():
            print(f"[translation] no checkpoint under {weights}; skipped")
            continue
        model = "pix2pix" if run.startswith("pix2pix") else "cyclegan"
        aligned = model == "pix2pix"
        partner = _symtrs_partners(args.data_root, direction, len(crops))
        stage = out / f"staging_{direction}_{model}"
        if stage.exists():
            shutil.rmtree(stage)
        _stage_translation(stage, direction, aligned, crops, partner)
        results_dir = out / f"results_{direction}_{model}"
        if results_dir.exists():
            shutil.rmtree(results_dir)
        dataroot = stage
        command = [
            sys.executable, str(upstream / "test.py"),
            "--dataroot", str(dataroot),
            "--name", f"symtrs_{run}",
            "--model", "pix2pix" if aligned else "cycle_gan",
            "--dataset_mode", "aligned" if aligned else "unaligned",
            "--direction", "AtoB",
            "--phase", "test", "--num_test", "1000",
            "--checkpoints_dir", str(run_dir / "checkpoints"),
            "--results_dir", str(results_dir),
        ]
        print("[translation] " + " ".join(command))
        completed = subprocess.run(
            command, cwd=str(upstream),
            env={**os.environ, "CUDA_VISIBLE_DEVICES": args.device.split(":")[-1]})
        if completed.returncode != 0:
            raise RuntimeError(f"{run} exited {completed.returncode}; see its output above")
        generated = sorted((results_dir / f"symtrs_{run}" / "test_latest" / "images").glob("*_fake_B.png"))
        keep = out / f"{direction}_{model}"
        keep.mkdir(exist_ok=True)
        for i, path in enumerate(generated):
            shutil.copyfile(path, keep / f"crop_{i}_fake_B.png")
        results[f"{direction}_{model}"] = {"run": run, "generated": len(generated)}
        print(f"[translation] {direction:16s} {model:8s} {len(generated)} frames", flush=True)
    (out / "translation.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


# --------------------------------------------------------------- shadow removal


def _shadow_in_child(args, run: str) -> None:
    """Re-enter this script for one model whose upstream module imports a local `utils`.

    The child gets the checkout on `PYTHONPATH` and the same CLI, so the import resolves to
    that checkout's `utils` and never to anything this process or another package exposes.
    """
    checkout = REPO / "third_party" / SHADOW_LOCAL_IMPORTS[run]
    if not checkout.is_dir():
        raise SystemExit(f"{checkout} is missing; run `git submodule update --init third_party`")
    env = {**os.environ, "SYMTRS_REAL_SHADOW_CHILD": run}
    env["PYTHONPATH"] = str(checkout) + (os.pathsep + env["PYTHONPATH"]
                                         if env.get("PYTHONPATH") else "")
    command = [sys.executable, str(Path(__file__).resolve()), "--task", "shadow",
               "--shadow-models", run, "--out", str(args.out), "--runs", str(args.runs),
               "--device", args.device, "--workers", str(args.workers)]
    print(f"[shadow] {run}: importing from {checkout} in a child process")
    completed = subprocess.run(command, env=env)
    if completed.returncode != 0:
        raise RuntimeError(f"{run} exited {completed.returncode}; see its output above")


def run_shadow(args) -> Dict[str, Any]:
    import numpy as np
    import torch
    from PIL import Image

    from symtrs.common.checkpoint import load_checkpoint
    from symtrs.config import RunConfig
    from symtrs.engine.tiling import predict_tiled
    from symtrs.model.shadow.worker import build_network, forward

    out = args.out / "shadow"
    out.mkdir(parents=True, exist_ok=True)
    source = args.out / "inputs" / "potsdam_2.png"   # the park with long cast shadows
    image = Image.open(source).convert("RGB")
    crops = [image.crop((0, 0, 1024, 1024)), image.crop((1024, 0, 2048, 1024))]
    for i, crop in enumerate(crops):
        crop.save(out / f"input_{i}.png")

    results: Dict[str, Any] = {}
    record_path = out / "shadow.json"
    if record_path.is_file():
        results.update(json.loads(record_path.read_text()))
    for run in args.shadow_models:
        if run in SHADOW_LOCAL_IMPORTS and os.environ.get("SYMTRS_REAL_SHADOW_CHILD") != run:
            _shadow_in_child(args, run)
            # The child wrote its own record; merge it so this process's final write does not
            # drop it.
            if record_path.is_file():
                results.update(json.loads(record_path.read_text()))
            continue
        config = args.runs / run / "config.yaml"
        weights = args.runs / run / "weights" / "best.pt"
        if not (config.is_file() and weights.is_file()):
            print(f"[shadow] no run under {args.runs / run}; skipped")
            continue
        cfg = RunConfig.from_yaml(str(config))
        net, arch = build_network(cfg)
        net.load_state_dict(load_checkpoint(str(weights))["model"])
        net.to(args.device).eval()

        for i, crop in enumerate(crops):
            array = np.array(crop)
            # Real data carries no shadow mask. The two mask-conditioned networks need one,
            # so they are given a luminance proposal (dark relative to the local mean); the
            # two image-only networks ignore the channel. The proposal is not a detector and
            # is reported as such: a real deployment would need one.
            mask = _luma_shadow(array) if arch.takes_mask else np.zeros(array.shape[:2], np.float32)
            packed = torch.cat([
                torch.from_numpy(array).permute(2, 0, 1).float().contiguous() / 255.0,
                torch.from_numpy(mask)[None].float(),
            ], dim=0)[None].to(args.device)

            def tile_forward(tile, net=net, arch=arch):
                with torch.no_grad():
                    return forward(net, arch, tile[:, :3], tile[:, 3:]).float()

            tile = 320 if run == "shadowformer" else 256
            with torch.no_grad():
                restored = predict_tiled(packed, tile_forward, tile, 64)
            restored = (restored[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
            Image.fromarray(restored).save(out / f"crop_{i}_{run}.png")
            print(f"[shadow] {run:18s} crop {i} done", flush=True)
        results[run] = {"mask": "luminance proposal" if arch.takes_mask else "none"}
        del net

    (out / "shadow.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def _luma_shadow(array) -> "Any":
    import numpy as np
    from scipy.ndimage import uniform_filter

    grey = array.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    local = uniform_filter(grey, 31)
    return ((grey < 0.75 * local) & (grey < 110)).astype(np.float32)


# ---------------------------------------------------------------------------- cli


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--potsdam", type=Path, default=REPO / "top_potsdam_7_9_RGB.tif")
    parser.add_argument("--swiss", type=Path, nargs="*", default=[
        REPO / "swiss_topo_test_image3.png", REPO / "swiss_topo_test_image2.png"])
    parser.add_argument("--runs", type=Path, default=REPO / "runs" / "campaign2")
    parser.add_argument("--data-root", dest="data_root", type=Path, default=Path("/mnt/ssd1/SyMTRS"),
                        help="the SyMTRS root, for the partner-domain images the translators load")
    parser.add_argument("--out", type=Path, default=REPO / "runs" / "real_tasks")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--depth-tile", dest="depth_tile", type=int, default=0)
    parser.add_argument("--depth-models", dest="depth_models", nargs="*", default=list(DEPTH_RUNS),
                        help="which depth runs to load; separating them lets a large metric "
                             "model build in a fresh process when the card is already shared")
    parser.add_argument("--shadow-models", dest="shadow_models", nargs="*", default=list(SHADOW_RUNS),
                        help="which shadow runs to load; ShadowFormer is re-launched in a child "
                             "with its checkout on PYTHONPATH because its module imports `utils`")
    parser.add_argument("--task", default="all",
                        choices=["inputs", "sr", "depth", "translation", "shadow", "all"])
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.task in ("inputs", "all"):
        records = write_inputs(args)
    else:
        records = json.loads((args.out / "inputs.json").read_text()) if (args.out / "inputs.json").is_file() else {}
    args.depth_images = list(records.get("potsdam", []))[:2] + list(records.get("swiss", []))[:2]

    if args.task in ("sr", "all"):
        run_sr(args)
    if args.task in ("depth", "all"):
        run_depth(args)
    if args.task in ("translation", "all"):
        run_translation(args)
    if args.task in ("shadow", "all"):
        run_shadow(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
