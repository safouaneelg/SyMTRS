# Smoke test audit (2026-10-17, read-only, 49 runs at the time of the audit)

## A. Confirmed problems
1. **The 2048 test (and val) are evaluated on a random 512 crop, not on the whole image.**
   `symtrs/data/factory.py` (from_config): `patch_size = val_patch_size or train.patch_size`. The 2048 configs have
   `patch_size: 512, val_patch_size: 0`, so 0 falls back to 512. The trainer (`engine/trainer.py:376-377`) builds val AND
   test with `train=False`, and `sample_crop_params` (data/transforms.py:151) draws a random crop without a seed.
   Evidence: preview `da3_large_2048/preview/test_00.png` = 3 panels of 512 px. At 1024/512/256, `patch_size: 0`
   (whole image, e.g. `midas_dpt_hybrid_1024/preview/test_00.png` = 1024 px). Consequences: 2048 test not reproducible,
   not comparable with the other resolutions, `best.pt` chosen on 2 crops. The comment in the configs ("0 = whole image,
   like the published numbers") is wrong. Fix: in from_config, with `train=False` use `val_patch_size` alone
   (0 = whole image); deterministic center crop if we want to bound memory, or `eval.tile_size`.
   Also affects 2048 segmentation (patch 512).
2. **Evaluation of the MiDaS models: the disparity is inverted and then aligned by scale alone.**
   `model/depth/base.py::to_metres` does `1/disp`, then `eval.align: scale` (metrics/depth.py:107). MiDaS outputs
   `s/z + t` (unknown scale AND shift); `1/(s/z+t)` is not proportional to z, and a scale does not correct t.
   On nadir views with narrow depth (CV std/mean ~0.25) t dominates. Clues: silog 41-111 for MiDaS against
   ~27 for a constant plane; align_scale_mean varies from 2e3 to 1.7e4 between val and test of the same model
   (hybrid_1024: 1.28e4 vs 2.13e3). The MiDaS numbers cannot be interpreted. Fix: align in disparity space
   (least-squares scale+shift against 1/GT), then invert and clamp; keep `scale` for the models that output a depth
   (DA3, ZoeD-N). Rule 3 of AGENTS.md is still respected (relative -> alignment) but the *type* of alignment
   must depend on the output space.
3. **DA3-metric: `focal_px: null` -> canonical depth not rescaled** (da3.py:156, 307-309).
   abs_rel 0.91-0.96, d1=0: pred ~4 % of the GT. The number is meaningless until focal_px is set (FOV of the
   UE camera not documented in Docs/; to be read from the capture blueprint).
4. **ZoeD-N: NaN outputs on GPU in the run** (zoedepth_n_1024/2048: "non-finite loss" at the 3 steps, val and test
   "every valid pixel had a non-positive prediction", test = `{skipped_images: 2.0}`). In fp32 on CPU (tested, .venv-zoe,
   test image 256) the output is finite (0.90-2.36 m) so the NaN comes from the fp16 autocast (hypothesis, not tested on GPU).
   Try `--set train.amp=false --set eval.amp=false`. In addition `best.pt` is saved with rmse=NaN (trainer:
   `new best rmse=nan`) and `summary.json` contains `NaN` (invalid JSON): refuse NaN as "best".
5. **Segmentation mIoU: class 0 absent from the data.** Checked on real PNGs (train/val/test, 2048 and 256): values
   only {63,126,189,252} -> classes 1-4; class 0 = 0 % everywhere (doc: "rare after fill"). Decoding OK
   (`round(v/63)`). But `iou_0=0.0` is written instead of NaN (metrics/semantic.py:`summary`) and, if the model predicts
   class 0, union>0 so an IoU of 0 is counted in the mIoU (untrained models). Fix: exclude from the mIoU the classes
   without GT support, write null for the absent ones.

## B. Split distribution (explains val vs test, not a units bug)
Median depth per image (m, x8): train 28-170 (med. 92), val 70-160 (med. 103), **test 137-265 (med. 208)**
(higher cameras: towers). No invalid pixel, max 279 m (< max_depth 1000). Segmentation: green 45 % train,
2 % val, 20 % test; road 22/63/30 %. Splits by location with few locations -> strong shift; val/test mIoU are hardly comparable.
- **DA3-large 2048: val rmse 1.43 vs test 31.7.** Units are correct: relative DA3 outputs a depth of arbitrary scale,
  scale 75-138 for 100-200 m is plausible. The gap comes from 2 random 512 crops in val (probably flat,
  d1=1.0) against 24 crops in test, plus the depth shift. At 256 (whole image): test abs_rel 0.31, val rmse 8.4.
- **ZoeD-K/NK**: abs_rel ~0.98 because the output saturates (max bins 80 m K, 10 m NYU) and is ~1-4 m here (CPU fp32: mean
  1.75 m) for scenes at 100-270 m; frozen encoder + 3 steps change nothing. Structural ceiling: the bin range
  has to be set before any serious fine-tuning.
- ZoeD-N is actually a **metric** NYU model (ZoeD_M12_N), not "without metric supervision" as the docstring of
  `zoedepth.py` says; treating it as relative (align scale) is defensible but the label is wrong.

## C. Plausible / to check
- The depth previews show the "predicted" panel in a uniform color: it plots the *unaligned* prediction (MiDaS ~1e-2,
  DA3 ~1), with the colormap set on the GT. Misleading, not a metric bug. Fix: plot the aligned prediction.
- `compute_depth_metrics` silently discards the pixels with prediction <=0 after alignment (favorable bias); log the
  discarded fraction.
- Add a reference row "constant plane aligned by the median" (silog ~27 on this dataset): sanity floor.
- `train.warmup_epochs: 1` with 1 epoch: scheduler.step warning before optimizer.step and lr 1e-5/5e-6 never applied at the 1st step.
- Smoke: 1 epoch/3 batches, the values measure nothing but the plumbing.

## D. Verified OK
- Units: EXR in cm, `depth_scale 0.01` -> m; max_depth not reached. Depth along the axis (doc): consistent.
- RGBA -> RGB: alpha = 255 everywhere (sample), `convert("RGB")` lossless. Pairing by stem OK (24 val/24 test).
- Normalization: single `prepare_input`; MiDaS via the hub transform (384, uint8 round trip), Zoe identity (normalizes
  internally), DA3 ImageNet + multiple of 14. Return to the input size in `forward_native`.
- Depth metrics: per-image alignment, metrics in m, `align none` for metric models (scale_mean=1.0 confirmed),
  val and test go through the same function `evaluate_depth` (same align, same tiling); only the crop size is suspect (A1).
- Segmentation: accumulated confusion matrix, ignore 255, preview GT consistent (buildings, road, thin cars).
- Not yet audited (no run): SR (LR/HR), domain adapt, shadow removal.
