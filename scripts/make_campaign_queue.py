"""Write the benchmark campaign queue, with every co-trainable model grouped per resolution.

    python scripts/make_campaign_queue.py --out runs/queue_campaign.txt
    python scripts/make_campaign_queue.py --check-groups      # group_conflicts on every group

Why a script: the 2048 runs are data-bound (decoding a 2048 PNG+EXR is ~0.2 s of CPU, a
training step ~0.15 s of GPU), so one decoded batch feeding six segmentation models costs
little more than feeding one (`symtrs train-group`, AGENTS.md rule 12). The groups below are
the largest the rules allow: one environment (timm 0.6.12 vs 1.x, rule 10), one crop size
(the crop is cut inside the shared loader), and effective batches that divide the largest
one (`symtrs.engine.trainer.group_conflicts`). Which models are in it is a decision, not a
derivation: a model enters only once `scripts/overfit_check.py` says it can learn as
configured. The ones that cannot yet (`PENDING`) are listed in the file's header so the
queue never reads as the whole benchmark.

Super-resolution, domain adaptation and shadow removal are subprocesses with their own data
loading, so they stay one job per config; their order and flags are those of
`runs/queue.txt` and `runs/queue_shadow.txt`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESOLUTIONS = (2048, 1024, 512, 256)

SEGMENTATION = ("unet_small", "segformer_b0", "segformer_b1", "segformer_b2",
                "deeplabv3_r50", "deeplabv3_r101")
# MiDaS Swin-V2 and BEiT-Large need timm 0.6.12; small, DPT-Hybrid and DPT-Large run on
# both, so they join them there. Split by crop size below (the large two crop 512 at 1024).
MIDAS = ("midas_small", "midas_dpt_hybrid", "midas_dpt_swin2_tiny", "midas_dpt_large",
         "midas_dpt_beit_large")
DA3 = ("da3_large", "da3_metric_large")  # depth_anything_3 is installed in .venv only
#: Did not learn as configured on the overfit check (2026-10-22); added once a setting that
#: learns is chosen. See runs/overfit/ and the PR description.
PENDING = ()
#: Dropped from the benchmark (2026-10-22, the dataset owner's call): ZoeD-N/K/NK did not learn
#: usefully on the overfit check even with the MiDaS core released (N flat, K slow, NK stuck at
#: ~8% of the true depth), and each attempt cost a 12-15 min slot. Depth Pro (same day, same
#: call): its zero-shot scale is ~16x short on SyMTRS and only full fine-tuning (952 M
#: parameters, ~3 h per 512px epoch) closed it. Adapters and configs removed.
DROPPED = ("zoedepth_n", "zoedepth_k", "zoedepth_nk", "depth_pro")

SR = ("srcnn", "autoencoder", "srgan", "swinir")
SR_SCALE = {2048: "x1", 1024: "x2", 512: "x4", 256: "x8"}
DOMAIN = [f"{m}_{d}" for m in ("pix2pix", "cyclegan")
          for d in ("day2night", "night2day", "noshadow2shadow", "shadow2noshadow")]


def _flags(resolution: int, tile_batch: int) -> str:
    # val_every=2 at >= 1024: a pass over the 3000 val images costs several 2048 epochs.
    flags = ["--set train.val_every=2"] if resolution >= 1024 else []
    flags.append(f"--set eval.tile_batch={tile_batch}")
    return "  ".join(flags)


def groups():
    """(venv, configs, flags) for every co-trained group, native resolution first."""
    for res in RESOLUTIONS:
        yield (".venv", [f"configs/segmentation/{res}/{m}_{res}.yaml" for m in SEGMENTATION],
               _flags(res, 8))
        for configs in _by_patch([f"configs/depth/{res}/{m}_{res}.yaml" for m in MIDAS]):
            yield (".venv-zoe", configs, _flags(res, 8 if len(configs) > 2 else 4))
        yield (".venv", [f"configs/depth/{res}/{m}_{res}.yaml" for m in DA3], _flags(res, 4))


def _by_patch(configs):
    """`configs` split into lists of equal `train.patch_size` (the crop is cut in the loader)."""
    import yaml

    buckets = {}
    for config in configs:
        patch = yaml.safe_load((REPO / config).read_text(encoding="utf-8"))["train"]["patch_size"]
        buckets.setdefault(patch, []).append(config)
    return [members for members in buckets.values()]


def solos():
    for res in RESOLUTIONS:
        for model in SR:
            yield f".venv  configs/superresolution/{model}_{SR_SCALE[res]}.yaml  --set eval.tile_batch=8"
    for name in DOMAIN:
        yield f".venv  configs/domain_adapt/{name}.yaml"
    shadow = REPO / "runs" / "queue_shadow.txt"
    if shadow.exists():
        for raw in shadow.read_text(encoding="utf-8").splitlines():
            line = raw.split("#", 1)[0].strip()
            if line:
                yield line


def render() -> str:
    lines = [
        "# Benchmark campaign queue, written by scripts/make_campaign_queue.py - edit that, not this.",
        "# Groups: every co-trainable model per resolution, one decoded batch for all members.",
        *(["# Not yet in the queue (did not learn as configured on scripts/overfit_check.py):",
           "#   " + ", ".join(PENDING)] if PENDING else []),
        "# Dropped from the benchmark: " + ", ".join(DROPPED),
        "# Launch: scripts/run_queue.py --jobs 3 --devices 0,1,2 --num-workers 4 --omp-threads 2",
        "",
    ]
    grouped = [f"group  {venv}  " + "  ".join(cfgs) + f"  {flags}" for venv, cfgs, flags in groups()]
    solo = list(solos())
    # Groups are the long jobs; interleave so the three lanes are not all on solos at the end.
    while grouped or solo:
        if grouped:
            lines.append(grouped.pop(0))
        for _ in range(3):
            if solo:
                lines.append(solo.pop(0))
    return "\n".join(lines) + "\n"


def check_groups() -> int:
    from symtrs.config import RunConfig
    from symtrs.engine.trainer import group_conflicts, shared_window, effective_batch

    bad = 0
    for venv, cfgs, _ in groups():
        loaded = [RunConfig.from_yaml(str(REPO / c)) for c in cfgs]
        for cfg, path in zip(loaded, cfgs):
            cfg.out_dir = str(REPO / "runs" / "check" / Path(path).stem)
        conflicts = group_conflicts(loaded)
        window = shared_window(loaded)
        steps = ", ".join(f"{c.model} x{window // effective_batch(c)}" for c in loaded)
        print(f"{'OK ' if not conflicts else 'BAD'} {venv:9s} {loaded[0].resolution:4d} "
              f"window {window:2d}: {steps}")
        for conflict in conflicts:
            print(f"      {conflict}")
        bad += bool(conflicts)
    return bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="runs/queue_campaign.txt")
    ap.add_argument("--check-groups", action="store_true")
    args = ap.parse_args()
    if args.check_groups:
        raise SystemExit(check_groups())
    Path(args.out).write_text(render(), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
