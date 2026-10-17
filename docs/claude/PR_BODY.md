## Summary

Makes the `symtrs` package trainable and scoreable on the **real SyMTRS capture** (previously only the synthetic `make-dummy` dataset was exercised), adds a **shadow-removal task** with three third-party models and **shadow <-> no-shadow translation**, and fixes a set of evaluation-protocol bugs found by auditing the first real runs. Everything was exercised end to end on a 12 GB laptop GPU with short *validation* runs; **no benchmark number is claimed** (see "What is NOT claimed").

### Real-data support (D1-D6)
- **D1** `normalize_stem`: real files are prefixed (`rgb_day_2048_<key>.png`); pairing is by normalised key. `check_pairing` raises on 0 pairs or >10 % unmatched instead of silently training on nothing.
- **D2** Segmentation labels decoded as `round(gray / label_step)` (`label_step: 63`, recorded in the checkpoint); off-grid gray -> `ignore_index` 255.
- **D3** Depth EXR is centimetres: `depth_scale: 0.01` -> metres; `max_depth` masks sky/invalid pixels.
- **D4** `noshadow` is a first-class native-only role (`rgb_noshadow`).
- **D5** Domain-adaptation staging keyed on the normalised stem, with symlink -> hardlink -> copy fallback (Windows).
- **D6** Legacy super-resolution pairs prefixed LR/HR files; `make-dummy` now writes the real layout; `symtrs doctor` reports pairing, the gray histogram and depth units.

### New: `shadow_removal` task (3 models)
ShadowFormer, Restormer and Uformer, run through an in-package worker that imports only the upstream network class (upstream repos are cloned at pinned commits by `scripts/fetch_third_party.py`, listed in `third_party/PINS.md`; not vendored). `rgb_day` -> `rgb_noshadow` at 2048; the shadow mask is derived from the pair (relative darkening + opening). Metrics: PSNR, luma-only SSIM (`ssim_y`, deliberately named so it is not confused with published per-channel SSIM), and LAB MAE/RMSE over all / shadow / non-shadow pixels. Models are fine-tuned from released weights; each summary records `init_weights` and `init_source_task`, and loading fails if fewer than 90 % of tensors match.

### New: shadow <-> no-shadow translation
CycleGAN and pix2pix reuse the domain front-end with two new directions (`shadow2noshadow`, `noshadow2shadow`; the A/B pair is data in `DIRECTION_ROLES`). `shadow2noshadow` additionally reports the same paired metrics as `shadow_removal`, so rows line up. 95 generated configs in total.

### Other additions
- `symtrs view` (port of the Unreal-side `view_dataset` tool, headless `--save`) and `notebooks/view_dataset.ipynb`.
- Offline builds for every registry key and a two-environment split: `.venv` (Python 3.12, torch cu128) and `.venv-zoe` (Python 3.10, timm 0.6.12 for ZoeDepth, MiDaS Swin2-T and BEiT-L). SegFormer now builds its MiT encoder from `transformers` (timm never shipped `mit_b*`); Depth Anything 3 is built from its own package; Depth Pro is fine-tunable on 12 GB (the `train.freeze` policy was never applied before).
- **VRAM guard**: `SYMTRS_CUDA_MEM_FRACTION` (set to 0.92 by the queue) turns a silent WDDM spill into system RAM (100-500x slower) into an immediate OOM.
- **Resumable tooling** (the dev laptop crashed repeatedly): `scripts/smoke_all.py` (smoke every config, find the batch that fits while keeping batch x accum equal to the grid's) and `scripts/run_queue.py` / `run_queue.ps1` (skip finished runs, resume from `last.pt`, kill the whole process tree on timeout, keep the machine awake, `runs/STOP` to stop cleanly). The dataset root is a parameter / `SYMTRS_DATA_ROOT`, never hard-coded.
- **Reporting**: `scripts/dataset_manifest.py` (dataset fingerprint, per-folder counts/sizes), `scripts/runs_report.py` (how each run was launched: resolved config, command, env, versions), `scripts/results_report.py` (TEST metrics per task). Absolute paths are redacted to `<DATA_ROOT>` / `<REPO>` by default (`--redact-root`).
- `docs/claude/`: working journal (PROGRESS, PLAN, BLOCKERS, IMPROVEMENTS, SUGGESTIONS, AUDIT_smoke, REVIEW_night, DATASET, RUNS, RESULTS).

### Bugs found and fixed
A smoke audit of the first real runs and an independent review of the diff found these; each now has a test that fails on the old behaviour.
1. **LR scheduler (pre-existing, present in the original commit)**: it was sized in optimizer steps but stepped once per *epoch*, so a 20-epoch run was a very long warmup with no decay and a 2-epoch run trained at ~4-8 % of the configured LR. Now stepped per optimizer step.
2. **val/test at 2048 scored a random 512 crop** (`val_patch_size: 0` fell back to the train patch). Evaluation now uses the whole image or a deterministic crop; val and test share one path.
3. **MiDaS was scored as inverse depth aligned by scale only**, but MiDaS emits `s/z + t`. New `scale_shift_disparity` alignment (least squares in disparity space, then invert) and a scale-and-shift-invariant `ssi` loss; the output space is declared in the registry and the (loss, align) pairing derives from it.
4. **DA3-metric ignored the focal length** (output ~4 % of GT). The focal is derived from the capture FOV and applied (`metres = canonical * f_px / 300`).
5. **ZoeD-N produced NaN under fp16 autocast** and a NaN epoch could be saved as `best.pt`; NaN is now rejected as "best" and written as `null`, and ZoeD-N is forced to a safe precision.
6. **mIoU counted absent classes as 0**, so the denominator varied per model; classes without ground-truth support are excluded (`miou_classes` reports how many were averaged).
7. **Resume lost the best metric** (a worse epoch could overwrite `best.pt`), and **test was scored with the last-epoch weights** rather than `best.pt`; both fixed, and `test_weights` is recorded in `summary.json`.
8. **Non-positive depth predictions were silently dropped** from the metrics; they are now scored at `min_depth` and reported as `invalid_pred_fraction`.
9. **Tiled affine matching** for 2048 inference was fragile on flat tiles (unit-dependent variance threshold, unclamped slope); fixed with a unit-independent threshold, a clamped slope and a shift-only fallback; depth-up-to-scale models (DA3 relative, ZoeD-N) are matched by scale only, never shifted.
10. **FID/KID domain asymmetry**: real 2048 images were downscaled without antialiasing while generated 256 px images were upsampled, so FID partly measured sharpness. Both sides are now brought to the generated size and resized identically; **KID is the primary metric** (N=24 makes the 2048-d FID covariance rank-deficient).
11. **pix2pix could never start** on the vendored `AlignedDataset` (it reads `[A|B]` side-by-side pairs, not `trainA/trainB`); staging now writes those pairs.
12. Smaller: halving the batch on OOM silently changed the effective batch (now compensated by `--grad-accum`); an eval-time OOM was halved down to batch 1; `--set` coercion of `None` defaults; CSV header drift on late columns; `silog` NaN from a tiny negative `sqrt`; DA3 1024 patch 0 -> 512 to fit 12 GB; BEiT-L routed to `.venv-zoe`.

## How to reproduce

Windows + NVIDIA GPU shown; `uv` is used for the environments. `<DATA_ROOT>` is the folder containing `train/`, `val/`, `test/`.

```powershell
# environments
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv\Scripts\python.exe -e ".[dev,exr,cv2,da3,adapt,timm,segformer]"
uv pip install --python .venv\Scripts\python.exe --no-deps depth-anything-3==0.1.1
uv venv .venv-zoe --python 3.10        # ZoeDepth, MiDaS Swin2-T, MiDaS BEiT-L
uv pip install --python .venv-zoe\Scripts\python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-zoe\Scripts\python.exe -e ".[dev,exr,cv2]" "timm==0.6.12"

# third-party shadow repos at pinned commits, then every pretrained weight (idempotent)
.venv\Scripts\python.exe scripts\fetch_third_party.py
powershell -File scripts\prefetch_all.ps1

# checks
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m symtrs doctor --data-root <DATA_ROOT>

# smoke every config, find the batch that fits, write runs\queue.txt (resumable)
.venv\Scripts\python.exe -u scripts\smoke_all.py --data-root <DATA_ROOT>

# the validation queue (detached, resumable; `New-Item runs\STOP` stops it cleanly)
$env:SYMTRS_DATA_ROOT = "<DATA_ROOT>"; powershell -File scripts\run_queue.ps1

# documents
.venv\Scripts\python.exe scripts\dataset_manifest.py --data-root <DATA_ROOT>
.venv\Scripts\python.exe scripts\runs_report.py
.venv\Scripts\python.exe scripts\results_report.py
```

## What is NOT claimed

- **No benchmark result.** Per `AGENTS.md` the dataset is under revision. All numbers come from *validation runs* (2 epochs; shadow removal 500 iterations) on the **preview dataset** (144 transforms, 96/24/24). They show that each model trains and is scored on real data, nothing more, and are not comparable to converged runs.
- The splits are geographic and **shift in distribution** (test is deeper: median ~208 m vs ~92 m in train; class shares differ strongly, e.g. val is 63 % road). Val/test gaps are partly a property of the data.
- Shadow-removal rows are **not a fair model comparison**: ShadowFormer starts from ISTD+ shadow-removal weights, Restormer and Uformer from denoising weights (no shadow weights are released).
- CycleGAN / pix2pix are trained and predict at 256 px and are scored at 2048; KID, not FID, is the primary metric.

## Known limitations / open items

- **Camera FOV (60 deg horizontal) is an assumption** (the Unreal capture actor default), isolated in `data.camera_hfov_deg`. It sets the focal length for DA3-metric; it needs confirming against the level that rendered the dataset. Depth Pro keeps `f_px: null` (it estimates its own FOV); giving it the derived focal is a one-line registry change.
- **DA3-metric zero-shot is ~25x too small** on nadir views at 70-270 m even with the correct focal: a domain gap, not a unit bug. Only fine-tuning recovers it. ZoeD-K/NK are structurally capped by their bin ranges (80 m / 10 m) on scenes up to ~270 m.
- Shadow-removal init heterogeneity (above); a common from-scratch row with a real budget is the fix.
- The development laptop blue-screened repeatedly (hardware/firmware, predating this work); hence the resumable, idempotent smoke/queue tooling. A few hours of runs were lost to sleep before `keep_awake` was added.
- Windows-first tooling (`.ps1`, `Scripts\python.exe`); the Python entry points are portable but were only exercised on Windows.
- Shadow inference is tiled sequentially (~10 s per 2048 image); batching tiles would speed evaluation ~5-10x.

## Tests

654 tests pass (`python -m pytest -q`, CPU only, no network, no dataset). `python configs/generate_grid.py --check` is clean (95 generated configs). Every protocol fix above has a test verified to fail on the old code.

## Validation results

Per-task test metrics: [docs/claude/RESULTS.md](docs/claude/RESULTS.md) (generated by `scripts/results_report.py`).
How each run was configured and launched: [docs/claude/RUNS.md](docs/claude/RUNS.md) (generated by `scripts/runs_report.py`).
Dataset fingerprint and layout: [docs/claude/DATASET.md](docs/claude/DATASET.md).

_To be filled in from those files when the validation queue finishes; nothing here is a benchmark result._

🤖 Generated with [Claude Code](https://claude.com/claude-code)
