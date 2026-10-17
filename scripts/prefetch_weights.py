"""Download every pretrained weight / hub repo the training runs need, into <repo>/.cache.

Run it with the venv that can build the keys you ask for (ZoeDepth keys need .venv-zoe):

    .venv/Scripts/python.exe scripts/prefetch_weights.py                 # everything but ZoeDepth
    .venv-zoe/Scripts/python.exe scripts/prefetch_weights.py --keys zoedepth_n zoedepth_k zoedepth_nk
    .venv/Scripts/python.exe scripts/prefetch_weights.py --offline       # verify: no network allowed

It is idempotent: the underlying loaders (torch.hub, huggingface_hub, timm, torchvision)
skip what is already cached. Each key is built through symtrs.registry.build_model, the
same call the trainer uses, so exactly the files training reads are the files cached.

Runs must set the same env (see cache_env()); `source scripts/cache_env.sh` is not needed,
the values are printed by `--print-env`. Depth Pro's checkpoint goes to
<repo>/checkpoints/depth_pro.pt because depth_pro's default checkpoint_uri is the
CWD-relative './checkpoints/depth_pro.pt': launch Depth Pro runs from the repo root.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"


def cache_env(offline: bool = False) -> dict:
    env = {
        "TORCH_HOME": str(CACHE / "torch"),
        "HF_HOME": str(CACHE / "huggingface"),
        "XDG_CACHE_HOME": str(CACHE / "xdg"),
    }
    if offline:
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    return env


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.exists() else 0


def fmt(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}"
        n /= 1024


def depth_pro_checkpoint(offline: bool) -> None:
    target = ROOT / "checkpoints" / "depth_pro.pt"
    if target.exists() and target.stat().st_size > 1_000_000_000:
        return
    from huggingface_hub import hf_hub_download

    target.parent.mkdir(exist_ok=True)
    hf_hub_download("apple/DepthPro", "depth_pro.pt", local_dir=str(target.parent))


def extra_inception() -> None:
    from torchvision.models import Inception_V3_Weights, inception_v3

    inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
    try:  # pytorch-fid weights: not used by symtrs/metrics/domain.py, cached as insurance
        from pytorch_fid.inception import FID_WEIGHTS_URL
        import torch

        torch.hub.load_state_dict_from_url(FID_WEIGHTS_URL, progress=False)
    except ImportError:
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", nargs="*", help="registry keys (default: every depth/segmentation key except zoedepth*)")
    ap.add_argument("--offline", action="store_true", help="forbid network; verifies the cache is sufficient")
    ap.add_argument("--print-env", action="store_true")
    ap.add_argument("--no-extras", action="store_true", help="skip Inception / FID weights")
    args = ap.parse_args()

    env = cache_env(args.offline)
    if args.print_env:
        for k, v in env.items():
            print(f"{k}={v}")
        return 0
    os.environ.update(env)
    for v in env.values():
        Path(v).mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)  # depth_pro resolves ./checkpoints/depth_pro.pt against the CWD
    sys.path.insert(0, str(ROOT))

    from symtrs.registry import REGISTRY, build_model  # noqa: E402

    keys = args.keys or [
        k for k, s in REGISTRY.items()
        if s.task in ("depth", "segmentation") and not k.startswith("zoedepth")
    ]
    rows = []
    for key in keys:
        t0, before = time.time(), dir_size(CACHE)
        try:
            if key == "depth_pro" and not args.offline:
                depth_pro_checkpoint(args.offline)
            build_model(key)
            status = "OK"
        except Exception as exc:  # report and continue: one bad key must not hide the rest
            status = f"FAIL {type(exc).__name__}: {str(exc)[:150]}"
        rows.append((key, status, dir_size(CACHE) - before, time.time() - t0))
    if not args.no_extras and not args.keys:
        try:
            extra_inception()
            rows.append(("inception_v3+fid", "OK", 0, 0))
        except Exception as exc:
            rows.append(("inception_v3+fid", f"FAIL {type(exc).__name__}: {exc}"[:170], 0, 0))

    print(f"\n{'key':<22}{'status':<12}{'new bytes':>11}{'secs':>7}")
    for key, status, size, secs in rows:
        print(f"{key:<22}{status:<12}{fmt(size):>11}{secs:>7.0f}")
    total = dir_size(CACHE) + dir_size(ROOT / "checkpoints")
    print(f"\ncache total (.cache + checkpoints/): {fmt(total)}")
    return 0 if all(r[1] == "OK" for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
