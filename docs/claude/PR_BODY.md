## Summary

Makes the `symtrs` package trainable and scoreable on the **real SyMTRS capture** (previously only the synthetic `make-dummy` dataset was exercised), adds a **shadow-removal task** with four third-party models (ShadowFormer, Restormer, Uformer, ShadowMaskFormer) and **shadow <-> no-shadow translation**, and fixes a set of evaluation-protocol bugs found by auditing the first real runs. Everything was exercised end to end on a 12 GB laptop GPU with short *validation* runs; **no benchmark number is claimed** (see "What is NOT claimed"). The goal of this work is mainly to check that the repository provides appropriate models and that the data is in an appropriate format; training quality is secondary.

### Real-data support (D1-D6)
- **D1** `normalize_stem`: real files are prefixed (`rgb_day_2048_<key>.png`); pairing is by normalised key. `check_pairing` raises on 0 pairs or >10 % unmatched instead of silently training on nothing.
- **D2** Segmentation labels decoded as `round(gray / label_step)` (`label_step: 63`, recorded in the checkpoint); off-grid gray -> `ignore_index` 255.
- **D3** Depth EXR is centimetres: `depth_scale: 0.01` -> metres; `max_depth` masks sky/invalid pixels.
- **D4** `noshadow` is a first-class native-only role (`rgb_noshadow`).
- **D5** Domain-adaptation staging keyed on the normalised stem, with symlink -> hardlink -> copy fallback (Windows).
- **D6** Legacy super-resolution pairs prefixed LR/HR files; `make-dummy` now writes the real layout; `symtrs doctor` reports pairing, the gray histogram and depth units.

### New: `shadow_removal` task (4 models)
ShadowFormer, Restormer, Uformer and ShadowMaskFormer (TAI 2025, 2.28 M parameters, a light mask-aware model added last), run through an in-package worker that imports only the upstream network class (upstream repos are cloned at pinned commits by `scripts/fetch_third_party.py`, listed in `third_party/PINS.md`; not vendored). `rgb_day` -> `rgb_noshadow` at 2048; the shadow mask is derived from the pair (relative darkening + opening). Metrics: PSNR, luma-only SSIM (`ssim_y`, deliberately named so it is not confused with published per-channel SSIM), and LAB MAE/RMSE over all / shadow / non-shadow pixels. ShadowFormer, Restormer and Uformer are fine-tuned from released weights; ShadowMaskFormer releases none and trains from scratch. Each summary records `init_weights` and `init_source_task`, and loading fails if fewer than 90 % of tensors match. ShadowFormer and ShadowMaskFormer receive the derived mask at test time too (as in ISTD); Restormer and Uformer get none.

The derived masks are computed once per pair and mask setting and shared by every run (`shadow.mask_cache`, default `.cache/shadow_masks`, gitignored; valid while the sources' size/mtime match; filled by a bounded process pool; never written into the dataset). `scripts/precompute_shadow_masks.py` builds the cache upfront (144 pairs in 35 s with 8 workers; masks byte-identical to the per-run ones, and a ShadowFormer re-score through the cache reproduced all 15 test metrics). `scripts/preview_shadow_mask.py` shows one pair as `rgb_day`, `rgb_noshadow`, the darkening map, the mask and an overlay, using the runs' own settings.

### New: shadow <-> no-shadow translation
CycleGAN and pix2pix reuse the domain front-end with two new directions (`shadow2noshadow`, `noshadow2shadow`; the A/B pair is data in `DIRECTION_ROLES`). `shadow2noshadow` additionally reports the same paired metrics as `shadow_removal`, so rows line up. 96 generated configs in total. `scripts/domain_identity_baseline.py` scores the "do nothing" reference (untranslated input vs target KID, plus a split-half noise floor) so a translator can be read against it.

### Other additions
- `symtrs view` (port of the Unreal-side `view_dataset` tool, headless `--save`) and `notebooks/view_dataset.ipynb`.
- Offline builds for every registry key and a two-environment split: `.venv` (Python 3.12, torch cu128) and `.venv-zoe` (Python 3.10, timm 0.6.12 for ZoeDepth, MiDaS Swin2-T and BEiT-L). SegFormer now builds its MiT encoder from `transformers` (timm never shipped `mit_b*`); Depth Anything 3 is built from its own package; Depth Pro is fine-tunable on 12 GB (the `train.freeze` policy was never applied before).
- **VRAM guard**: `SYMTRS_CUDA_MEM_FRACTION` (set to 0.92 by the queue) turns a silent WDDM spill into system RAM (100-500x slower) into an immediate OOM.
- **Resumable tooling** (the dev laptop crashed repeatedly): `scripts/smoke_all.py` (smoke every config, find the batch that fits while keeping batch x accum equal to the grid's) and `scripts/run_queue.py` / `run_queue.ps1` (skip finished runs, resume from `last.pt`, kill the whole process tree on timeout, keep the machine awake, `runs/STOP` to stop cleanly). The dataset root is a parameter / `SYMTRS_DATA_ROOT`, never hard-coded.
- **Reporting**: `scripts/dataset_manifest.py` (dataset fingerprint, per-folder counts/sizes), `scripts/runs_report.py` (how each run was launched: resolved config, command, env, versions), `scripts/results_report.py` (TEST metrics per task, with the identity / bicubic / noise-floor reference rows). Absolute paths are redacted to `<DATA_ROOT>` / `<REPO>` by default (`--redact-root`).
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
12. **Failed super-resolution runs were reported as successful.** The legacy trainer's non-zero exit was written into `summary.json` and `symtrs train` exited 0, so the queue counted 13 of 16 crashed SR smoke runs as "ok" (and would have skipped them forever). A failed run now raises (exit 2) and leaves `summary_failed.json`; one `run_verdict` (exit 0, summary present, `returncode` 0, non-empty `test`) decides "ok" for both the queue and the smoke harness.
13. **Super-resolution did not fit and was not scored on the protocol**: SRCNN / autoencoder / SRGAN trained on whole 2048 px frames (crops existed for SwinIR only), SwinIR precomputed ~10 GB of attention masks for the whole frame, SRGAN refused x1, validation was carved 80/20 out of `train/`, and SwinIR's val/test scored a centre crop. Now: aligned random crops for every model, `val/` as the validation split (missing = error), whole-image val and test in windows (`predict_tiled(scale=)`), per-image metrics on the clamped output, `best_psnr` kept across a resume.
14. **Domain adaptation `BtoA` (night2day, noshadow2shadow) scored the wrong images**: upstream's `--direction BtoA` swaps the inputs and still writes the requested translation as `fake_B`; reading `fake_A` made pix2pix score nothing (`test: {}`) and CycleGAN score the opposite translation. Empty or partial generated sets are now errors.
15. **Relative run directory broke the shadow and domain runs.** Both subprocesses run with the upstream checkout as cwd, so a relative `--out-dir` (`runs/validation`) resolved against the wrong tree: 11 of the 95 first-pass validation jobs failed (`FileNotFoundError ... config.yaml`; "Could not find trainA/trainB"). `RunConfig.anchored()` now makes the paths absolute at the entry of the three runners (`tests/test_relative_run_dir.py`).
16. **`kid_std` was a hard-coded `0.0`** in the domain summaries (no subset spread was ever computed). It is now `null`, and legacy zeros are shown as n/a.
17. **Super-resolution training runs wrote no `config.yaml`**, so the resolved config of a run was not in its run directory. Fixed in the SR front-end.
18. Smaller: halving the batch on OOM silently changed the effective batch (now compensated by `--grad-accum`); an eval-time OOM was halved down to batch 1; `--set` coercion of `None` defaults; CSV header drift on late columns; `silog` NaN from a tiny negative `sqrt`; DA3 1024 patch 0 -> 512 to fit 12 GB; BEiT-L routed to `.venv-zoe`.

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
- Shadow-removal rows are **not a fair model comparison**: ShadowFormer starts from ISTD+ shadow-removal weights, Restormer and Uformer from denoising weights (no shadow weights are released), ShadowMaskFormer from scratch. ShadowFormer and ShadowMaskFormer also get the ground-truth-derived mask at test time; Restormer and Uformer do not.
- CycleGAN / pix2pix are trained and predict at 256 px; KID / FID compare them with the real images resized to 256 px (the paired shadow2noshadow metrics upsample to 2048). KID, not FID, is the primary metric.
- Shadow-removal PSNR must be read against the **identity row** (the shadowed input scored as the prediction: test PSNR 12.4 dB, 7.9 dB in shadow, 27.9 dB outside; `scripts/shadow_identity_baseline.py`). Super-resolution against bicubic (test x2 22.9 / x4 20.9 / x8 19.3 dB). Two-epoch SR runs are a dozen optimiser steps and sit below bicubic. Translators are read against the untranslated-input KID (below).

## Known limitations / open items

- **Camera FOV is 60 deg horizontal, confirmed by the dataset owner (2026-10-17)**, isolated in `data.camera_hfov_deg`. It sets the focal length for DA3-metric. Depth Pro keeps `f_px: null` (it estimates its own FOV); giving it the derived focal is a one-line registry change.
- **DA3-metric zero-shot is ~25x too small** on nadir views at 70-270 m even with the correct focal: a domain gap, not a unit bug. Only fine-tuning recovers it. ZoeD-K's bin head is bounded (80 m in the released weights; the generated fine-tuning configs use `max_depth: 300` to cover scenes up to ~280 m). Depth Pro and ZoeD-NK also predict ~10x too shallow zero-shot.
- Shadow-removal init heterogeneity (above); a common start (all from scratch with a real budget) is the fix.
- `scripts/smoke_all.py` picks the batch from a few-frame smoke with no whole-image scoring pass, so it can choose a batch that OOMs in a real run (4 depth rows did; they were re-run with a smaller batch and a raised `--grad-accum`, same effective batch).
- The development laptop blue-screened repeatedly (hardware/firmware, predating this work); hence the resumable, idempotent smoke/queue tooling. A few hours of runs were lost to sleep before `keep_awake` was added.
- Windows-first tooling (`.ps1`, `Scripts\python.exe`); the Python entry points are portable but were only exercised on Windows.
- Shadow inference is tiled sequentially (~10 s per 2048 image); batching tiles would speed evaluation ~5-10x.

## Dataset findings

Checked on the preview dataset (details and the per-location table in [docs/claude/DATASET.md](docs/claude/DATASET.md)).

- **Depth format is fine**: the EXR is planar scene depth in centimetres, with the same convention in train, val and test.
- **Camera altitude is relative to the first surface hit**, not to the ground (capture actor, `ECC_Visibility` trace). At test loc010/loc011 the centre is a skyscraper roof, so the camera is 70 m above the roof and the street is 152-275 m away; train/val ground sits at 70-160 m (to ~182 m).
- **Effect on the test split**: test depths (to ~275 m) lie largely outside the training range. This makes the test split harder for a capture reason and explains a large part of the metric-depth test error.
- **FOV is not stored in the dataset.** The owner confirmed 60 deg; a car-size estimate over 838 cars (54-63 deg for 4.2-5.0 m cars) agrees. A capture with another FOV must set `data.camera_hfov_deg`.
- The dataset owner's questions (FOV, data each model needs, why Restormer/Uformer, the shadow mask, BMNet, the altitude rule) and the answers are in [docs/claude/QA.md](docs/claude/QA.md).
- For the dataset regeneration: the altitude rule is kept (decided with the dataset owner). All three split zones contain skyscrapers; in the preview, none of the 8 train locations happened to fall on a tower while both test locations did, an accident of the small capture that more than 20,000 views remove (see `QA.md`). No per-frame metadata or UE shadow mask is needed: the pose is in the file names, the FOV is always 60 deg, and the derived mask is close to exact (bimodal darkening).

## Tests

The full suite passes: 735 tests (`python -m pytest -q`, CPU only, no network, no dataset); in `.venv-zoe` one test fails because that environment does not install the domain-adaptation extras (`dominate`, `wandb`). `python configs/generate_grid.py --check` is clean (96 generated configs). Each fix comes with a test that asserts the reason, not just the value (AGENTS.md section 8); the relative-run-dir tests were checked to fail with the fix disabled.

## Validation results

**96 of 96 validation runs finished and pass `run_verdict`** (exit 0, `summary.json` present, `returncode` 0, non-empty `test`): segmentation 24, depth 44, super-resolution 16, shadow removal 4, domain adaptation 8. Wall clock on the 12 GB laptop, 2026-10-17: first pass 08:43 -> 12:20 (3 h 37; 80 ok, 15 failed: 4 OOM and 11 relative-path), then 4 OOM re-runs (11 min), 11 shadow/domain re-runs (68 min) and the ShadowMaskFormer run (12 min). Job time of the 96 final runs: segmentation 22 min, depth 2 h 18, super-resolution 59 min, shadow removal 60 min, domain adaptation 20 min (4 h 59 in total).

Per-task test metrics: [docs/claude/RESULTS.md](docs/claude/RESULTS.md) (generated by `scripts/results_report.py`).
How each run was configured and launched: [docs/claude/RUNS.md](docs/claude/RUNS.md) (generated by `scripts/runs_report.py`).
Dataset fingerprint and layout: [docs/claude/DATASET.md](docs/claude/DATASET.md).

- **Segmentation** (test mIoU over 4 classes): 0.08-0.47. DeepLabV3-r50 rises from 0.08 to 0.47 between 256 and 2048; U-Net small stays at 0.11-0.13.
- **Depth** (test abs_rel): relative models 0.05-0.22 (MiDaS BEiT-L 0.05 at 256; the 1024 rows beat the 2048 rows, which train on crops and are scored in tiles: a note, not a claim); DA3 relative 0.16-0.20. Metric heads barely trained: Depth Pro and ZoeD-NK (frozen encoder, lr 5e-6) stay at 0.97-0.98, effectively zero-shot; ZoeD-K 0.65-0.70; DA3-metric (unfrozen) reaches 0.27-0.31 at 1024/2048 but stays at 0.87-0.96 at 256/512, where an effective batch of 64 on 96 images is 2 optimizer steps per epoch.
- **Super-resolution** (test PSNR, x8 / x4 / x2 / x1): SwinIR 17.8 / 19.3 / 21.9 / 26.6 dB; the other three models 14.7-19.6 dB at x2-x8. Bicubic scores 19.3 / 20.9 / 22.9 dB at x8 / x4 / x2, so no two-epoch model beats it. x1 is a refiner control, not super-resolution.
- **Shadow removal** (test PSNR all / shadow / non-shadow, 500 iterations): ShadowFormer from ISTD+ 23.39 / 19.48 / 30.05; Uformer 21.51 / 17.90 / 26.46; Restormer 20.90 / 17.39 / 25.49; ShadowMaskFormer from scratch 20.81 / 16.97 / 26.93; identity (input as output) 12.40 / 7.87 / 27.91. All four beat identity on shadow pixels; only ShadowFormer also beats it outside the shadow (the others damage non-shadow pixels, notably on val).
- **Domain adaptation** (test KID, lower is better; N=24): the untranslated input scores day->night 0.231, night->day 0.238, shadow->noshadow 0.104, noshadow->shadow 0.096. All 8 two-epoch translators are worse (CycleGAN 0.251 / 0.246 / 0.280 / 0.378 and pix2pix 0.369 / 0.607 / 0.177 / 0.362, in the same direction order), i.e. no better than doing nothing. Paired shadow2noshadow PSNR: CycleGAN 15.6 dB, pix2pix 18.8 dB (identity 12.4).

Caveats, because they decide how to read the numbers:
- These are validation runs on the preview dataset, **not benchmark numbers**.
- The shadow-removal starts are heterogeneous, and ShadowFormer / ShadowMaskFormer get the ground-truth-derived mask at test time; do not rank those rows.
- The translators are trained for 2 epochs and are no better than the identity. The test split has 2 locations, so KID is mostly a scene-shift measure, and the split-half noise floor (up to 0.18) is as large as most gaps.
- Metric depth heads (Depth Pro, ZoeD-NK, DA3-metric at 256/512) barely trained; `best_epoch` is 0 on several relative rows (the second epoch at a decayed LR did not help). Two epochs say nothing about convergence.
- The test split is deeper than train for a capture reason (altitude rule, see "Dataset findings"), which inflates the metric-depth test error of every model.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
