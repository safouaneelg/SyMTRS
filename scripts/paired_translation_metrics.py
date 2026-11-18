#!/usr/bin/env python3
"""Paired metrics for the day<->night translations: PSNR, SSIM and LPIPS against the target.

The translation table of the manuscript reports KID and FID, which compare two *distributions*
and say nothing about one generated image and the scene it was generated for. Reviewer R2.3 asks
for exactly that second half, and SyMTRS is what makes it possible: the night render of a view
has the same geometry as its day render, so a translated frame can be compared, pixel by pixel,
with the image it should have produced. This script computes that comparison over the whole test
split, for the four campaign runs, and writes one row per run.

Two conventions worth stating, because both change the number:

  * the target is the 2048 px render resized to the size of the generated image with BICUBIC,
    i.e. through the same `transforms.Resize` upstream's test script applied to the input, so
    what is compared is the generator's output against the image it was asked to produce;
  * SSIM is `symtrs.metrics.shadow.ssim_map`, the luma SSIM (Gaussian 11x11, sigma 1.5, valid
    region) the shadow-removal table already uses, so one number in the paper does not mean two
    things. It is not skimage's default `structural_similarity`, which scores RGB channels.

    python scripts/paired_translation_metrics.py --device cuda:2
    python scripts/paired_translation_metrics.py --runs cyclegan_night2day --dry-run

`lpips` is an analysis-only import (`uv pip install --no-deps lpips`); nothing in `symtrs`
imports it. Writes `runs/campaign2/translation_paired.json` and prints the table.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "campaign2"
OUT = RUNS / "translation_paired.json"
DATA = Path("/mnt/ssd1/SyMTRS")

#: campaign run -> (direction, the `rgb_<role>` folder its output should look like)
TRANSLATIONS = {
    "cyclegan_day2night": ("day2night", "night"),
    "pix2pix_day2night": ("day2night", "night"),
    "cyclegan_night2day": ("night2day", "day"),
    "pix2pix_night2day": ("night2day", "day"),
}
LABEL = "_fake_B"
BATCH = 64


def _load(job: tuple[str, str, int | None]) -> tuple:
    """One (generated, target) pair as two HxWx3 uint8 arrays, both at the generated size.

    `size` is None for a generated frame, whose own size is the reference, and the generator's
    output size for the untranslated reference pair, which has to be brought to it first.
    """
    import numpy as np
    from PIL import Image

    generated_path, target_path, size = job
    with Image.open(generated_path) as image:
        generated = image.convert("RGB")
        if size is not None:
            generated = generated.resize((size, size), Image.BICUBIC)
        generated = np.asarray(generated)
    with Image.open(target_path) as image:
        target = image.convert("RGB")
        if target.size != (generated.shape[1], generated.shape[0]):
            target = target.resize((generated.shape[1], generated.shape[0]), Image.BICUBIC)
        target = np.asarray(target)
    return generated, target


def pairs(run: Path, role: str) -> list[tuple[str, str, None]]:
    """`(generated, target)` for every frame the test script wrote, in a stable order."""
    images = next((run / "results").glob("*/test_latest/images"))
    found = []
    for generated in sorted(images.glob(f"*{LABEL}.png")):
        stem = generated.name[: -len(f"{LABEL}.png")]
        target = DATA / "test" / f"rgb_{role}" / f"rgb_{role}_2048_{stem}.png"
        if not target.is_file():
            raise SystemExit(f"no target for {generated.name}: expected {target}")
        found.append((str(generated), str(target), None))
    if not found:
        raise SystemExit(f"{images} holds no *{LABEL}.png; was the test script run?")
    return found


def reference_pairs(size: int = 256) -> list[tuple[str, str, int]]:
    """The day render against the night render of every test view, at the generator's size.

    This is what the same metrics read when the frame is left alone, and it is the row a
    translation has to beat. It is one row and not two: PSNR is symmetric in its two images,
    and the day and the night render of a view are the same view.
    """
    day = sorted((DATA / "test" / "rgb_day").glob("rgb_day_2048_*.png"))
    found = []
    for path in day:
        stem = path.name[len("rgb_day_2048_") :]
        night = DATA / "test" / "rgb_night" / f"rgb_night_2048_{stem}"
        if not night.is_file():
            raise SystemExit(f"no night render for {path.name}: expected {night}")
        found.append((str(path), str(night), size))
    if not found:
        raise SystemExit(f"no test views under {DATA / 'test' / 'rgb_day'}")
    return found


def score(jobs: list[tuple[str, str, int | None]], device: str, workers: int) -> dict:
    """PSNR, SSIM and LPIPS over `jobs`, loading each pair once."""
    import numpy as np
    import torch
    from symtrs.metrics.shadow import luminance, ssim_map

    import lpips as lpips_module

    model = lpips_module.LPIPS(net="alex", verbose=False).to(device).eval()
    psnr, ssim, distances = [], [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(jobs), BATCH):
            batch = list(pool.map(_load, jobs[start : start + BATCH], chunksize=1))
            generated = np.stack([pair[0] for pair in batch])
            target = np.stack([pair[1] for pair in batch])
            a, b = generated.astype(np.float64) / 255.0, target.astype(np.float64) / 255.0
            for left, right in zip(a, b):
                mse = float(np.mean((left - right) ** 2))
                psnr.append(10.0 * np.log10(1.0 / max(mse, 1e-12)))
                ssim.append(float(np.mean(ssim_map(luminance(left), luminance(right)))))
            # lpips wants [-1, 1] in NCHW, whatever the input size.
            with torch.no_grad():
                value = model(
                    torch.from_numpy(generated.astype(np.float32) / 127.5 - 1.0)
                    .permute(0, 3, 1, 2).to(device),
                    torch.from_numpy(target.astype(np.float32) / 127.5 - 1.0)
                    .permute(0, 3, 1, 2).to(device),
                )
            distances.extend(value.flatten().tolist())
            print(f"  {len(psnr)}/{len(jobs)}", flush=True)
    return {
        "frames": len(psnr),
        "psnr": float(np.mean(psnr)),
        "ssim": float(np.mean(ssim)),
        "lpips": float(np.mean(distances)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", nargs="+", default=None,
                        choices=sorted(TRANSLATIONS))
    parser.add_argument("--reference", action="store_true",
                        help="also score the untranslated pair (day render against night render)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # `--reference` alone re-scores only the baseline and keeps the runs already on disk, so a
    # reference row can be added without re-running the four 3 000-frame passes.
    names = args.runs if args.runs is not None else ([] if args.reference else list(TRANSLATIONS))
    rows = json.loads(args.out.read_text()) if args.out.is_file() else {}

    for name in names:
        direction, role = TRANSLATIONS[name]
        jobs = pairs(RUNS / name, role)
        print(f"{name} ({direction}): {len(jobs)} frames against rgb_{role}", flush=True)
        if args.dry_run:
            continue
        rows[name] = {"direction": direction, "target": f"rgb_{role}", **score(jobs, args.device, args.workers)}

    if args.reference:
        jobs = reference_pairs()
        print(f"identity: {len(jobs)} test views, day render against night render", flush=True)
        if not args.dry_run:
            rows["identity"] = {"direction": "either", "target": "the other illumination",
                                **score(jobs, args.device, args.workers)}

    if args.dry_run:
        return

    args.out.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"\n{'run':22s} {'PSNR':>8s} {'SSIM':>8s} {'LPIPS':>8s}")
    for name, row in rows.items():
        print(f"{name:22s} {row['psnr']:8.2f} {row['ssim']:8.4f} {row['lpips']:8.4f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
