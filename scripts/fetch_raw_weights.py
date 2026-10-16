"""Download the heavy weight FILES directly (no model build), so they reach the disk even for
models whose build is still broken in this environment (DA3 image processor, ZoeDepth venv).
Idempotent: a file already complete is skipped. Same cache layout as prefetch_weights.py."""
import os, sys, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CK = ROOT / ".cache" / "torch" / "hub" / "checkpoints"
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

URLS = {
    "ZoeD_M12_N.pt": "https://github.com/isl-org/ZoeDepth/releases/download/v1.0/ZoeD_M12_N.pt",
    "ZoeD_M12_K.pt": "https://github.com/isl-org/ZoeDepth/releases/download/v1.0/ZoeD_M12_K.pt",
    "ZoeD_M12_NK.pt": "https://github.com/isl-org/ZoeDepth/releases/download/v1.0/ZoeD_M12_NK.pt",
    "deeplabv3_resnet101_coco-586e9e4e.pth": "https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth",
}

def get(name, url):
    dst = CK / name
    tmp = dst.with_suffix(dst.suffix + ".part")
    req = urllib.request.Request(url, method="HEAD")
    size = int(urllib.request.urlopen(req).headers.get("Content-Length", 0))
    if dst.exists() and dst.stat().st_size == size:
        print(f"skip {name}", flush=True); return
    print(f"get  {name} ({size/2**20:.0f} MB)", flush=True)
    CK.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dst)
    print(f"ok   {name}", flush=True)

for n, u in URLS.items():
    try: get(n, u)
    except Exception as e: print(f"FAIL {n}: {e}", flush=True)

from huggingface_hub import snapshot_download
for repo in ("depth-anything/DA3-LARGE", "depth-anything/DA3METRIC-LARGE"):
    try:
        p = snapshot_download(repo); print(f"ok   {repo} -> {p}", flush=True)
    except Exception as e: print(f"FAIL {repo}: {e}", flush=True)
print("RAW_DONE", flush=True)
