# Start every pre-download as detached processes that outlive the shell / Claude session
# that launched them (a session ending killed the first attempt). Idempotent: re-run it any
# time; finished downloads are skipped. Logs: logs\prefetch_*.log, logs\install_timm06_*.log.
$repo = Split-Path -Parent $PSScriptRoot
$stamp = Get-Date -Format "yyyyMMdd_HHmm"
New-Item -ItemType Directory -Force (Join-Path $repo "logs") | Out-Null

# MiDaS's hubconf calls torch.hub.load on two more repos WITHOUT trust_repo, and torch then
# waits on stdin for "Do you trust this repository? (y/N)" - a hidden process hangs there
# forever (the first prefetch did). Pre-trusting them in the repo-local TORCH_HOME also
# keeps the training runs, which use the same TORCH_HOME, from hanging the same way.
$trusted = Join-Path $repo ".cache\torch\hub\trusted_list"
New-Item -ItemType Directory -Force (Split-Path $trusted) | Out-Null
$have = @(); if (Test-Path $trusted) { $have = Get-Content $trusted }
foreach ($r in "intel-isl_MiDaS", "rwightman_gen-efficientnet-pytorch", "facebookresearch_WSL-Images",
               ) {
    if ($have -notcontains $r) { Add-Content -Encoding ascii $trusted $r }
}

# 1. Every weight the main venv can build (all depth/seg keys except the timm 0.6.12 MiDaS two, + Inception/FID).
$main = Join-Path $repo "logs\prefetch_main_$stamp.log"
Start-Process powershell -WorkingDirectory $repo -WindowStyle Hidden -ArgumentList @(
    "-NoProfile", "-Command",
    "& '.venv\Scripts\python.exe' scripts\prefetch_weights.py *> '$main'; 'PREFETCH_MAIN_DONE' | Add-Content '$main'")

# 2. MiDaS Swin-V2 and BEiT-Large: their own Python 3.10 venv (timm 0.6.12 breaks on 3.12), then
#    their weights. The venv keeps the name .venv-zoe from when ZoeDepth was its first user.
$zoe = Join-Path $repo "logs\install_timm06_$stamp.log"
$zoeCmd = @"
`$ErrorActionPreference = 'Continue'
python -m uv python install 3.10
if (-not (Test-Path '.venv-zoe\Scripts\python.exe')) { python -m uv venv .venv-zoe --python 3.10 }
python -m uv pip install --python .venv-zoe\Scripts\python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m uv pip install --python .venv-zoe\Scripts\python.exe -e '.[dev,exr,cv2]' 'timm==0.6.12'
& '.venv-zoe\Scripts\python.exe' -c 'import torch,timm;print(torch.__version__,torch.cuda.is_available(),timm.__version__)'
if (`$LASTEXITCODE -eq 0) { 'TIMM06_OK' }
& '.venv-zoe\Scripts\python.exe' scripts\prefetch_weights.py --keys midas_dpt_swin2_tiny midas_dpt_beit_large --no-extras
'PREFETCH_TIMM06_DONE'
"@
Start-Process powershell -WorkingDirectory $repo -WindowStyle Hidden -ArgumentList @(
    "-NoProfile", "-Command", "& { $zoeCmd } *> '$zoe'")

"started: $main"
"started: $zoe"
