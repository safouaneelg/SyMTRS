# Skeptical review, 76dd299..HEAD (symtrs, scripts, generate_grid)

Read-only review. Items were checked by reading the code; items 1 and 4 were also run on synthetic CPU tensors.

## Findings, most severe first

**1. The LR schedule never does what the config says (corrupts every depth and segmentation number).**
`symtrs/engine/trainer.py:198-228` builds the scheduler in *optimizer steps*: `total = epochs*steps_per_epoch` and `warmup = warmup_epochs*steps_per_epoch`. But `trainer.py:455` calls `scheduler.step()` once per *epoch*, and nothing in `loop.py` steps it.
- Check: epochs=20, warmup_epochs=1, 24 steps/epoch. The LR multiplier is 0.04, 0.08, ... 0.83 over the 20 epochs. It is a very long warmup with no cosine decay.
- Queue runs with `--epochs 2` train at about 4% and 8% of the configured LR.
- Present since 76dd299.
- Fix: step per optimizer step inside `train_one_epoch`, or call `build_scheduler(..., steps_per_epoch=1)`. Then re-run the validation queue.

**2. Resume and best checkpoint.**
- `trainer.py:423` and `:445-447`: `best` and `history` are not restored on `--resume`. The first epoch after a resume has `best=None`, so it overwrites `best.pt` even when it is worse than the pre-crash best. `run_queue.py` resumes on every crash or timeout.
- `trainer.py:496`: `summary["test"]` is scored with the **last-epoch** weights, not `best.pt`. The summary shows `best` next to a test number that does not come from it. Shadow removal scores `best.pt`, so the protocols differ across tasks.
- Fix: restore `best` from the checkpoint payload (it holds `val_metrics`). Reload `best.pt` before the test pass, or document "last" and make all tasks follow it.

**3. Segmentation mIoU has a different denominator per model.**
- `metrics/semantic.py:79-95`: class 0 is absent from the ground truth but the head has 5 outputs. A model that predicts a single class-0 pixel gets IoU_0 = 0, so `miou_classes` goes from 4 to 5 and mIoU drops by 20% relative. Models in the same table are not comparable.
- Fix: average only over classes with ground-truth support (`matrix.sum(1) > 0`). Or remap ids to 0..3 and use `num_classes=4`.

**4. Tiled affine matching is fragile (`engine/tiling.py:93-116`, `loop.py:91`).**
- The math is correct on textured fields: random per-tile a,b gave 1e-6 relative error after matching, versus 0.95 without.
- Flat or low-variance overlap: the threshold `variance <= 1e-12` is absolute and unit-dependent. Above it, the slope `cov/var` is noise-dominated and not clamped. At or below it, the tile is silently left on its own random scale and shift. A synthetic field with a flat half gave 0.97 relative error, the same as no matching.
- `match_affine` is keyed on `protocol == "relative"`. ZoeD-N and DA3-relative emit depth known **up to scale only**. Fitting a shift adds an offset that the final `scale` alignment cannot remove. Key on `output_space`: affine for disparity, through-origin scale (or log-scale) for depth.
- The raster order anchors on tile 0, so errors accumulate (order dependence). Slope is a plain OLS regression and is biased toward 0 under noise.
- Severity: medium. It affects relative models at 2048 only, where tiling is on (tile 384 or 512).
- Fix: fall back to shift-only when variance is low. Clamp the slope to about [0.25, 4]. Add a "fallback used" counter to the logs.

**5. Non-positive predictions are silently dropped from the metrics (`metrics/depth.py:224-228`).**
- `to_metres` clamps negatives to 0 for metric and depth-space models. The mask then removes those pixels with no count and no warning, so RMSE and abs_rel are optimistic for a model that predicts ≤ 0 in places.
- Fix: report `dropped_fraction`, or score them as `min_depth`.

**6. FID/KID domain asymmetry (`metrics/domain.py:42`, `domain/runner.py` `evaluate_adaptation`).**
- Real 2048 images are bilinear-downscaled to 299 **without antialias**, so they are aliased. Generated 256-px images are upsampled to 299, so they are smooth. FID then measures sharpness, not translation quality. N=24 per set also makes FID's 2048-d covariance singular.
- Fix: `antialias=True`. Resize real images to the generated size (256) first. Report KID only for N=24.

**7. Smoke batch search and the validation queue (`scripts/smoke_all.py`).**
- Halving the batch keeps `grad_accum` and the LR. The queue therefore trains at an effective batch of `batch*accum`, which differs from the grid. This is not recorded in the summary.
- The OOM regex matches torch's messages. It cannot tell a train-time OOM from a val/test OOM at whole-frame 2048 (`eval.batch_size`). The latter wastes halvings down to 1 and then records FAIL.
- Fix: keep `batch*accum` constant by raising `--grad-accum`. Write the effective batch into the queue and the report.

**8. Shadow benchmark comparability (`worker.py:71-80`, `smoke_all.py:SHADOW_INIT`).**
- ShadowFormer starts from ISTD+ (shadow-removal) weights. Restormer and Uformer start from *denoising* weights (`real_denoising`, `Uformer_B_SIDD`).
- `load_matching` proceeds with any number of matched tensors, even 0.
- Fix: raise below a fraction (for example 90%) of tensors loaded. State the init in the RESULTS table. Add a from-scratch or common-init row.

**9. Low severity.**
- `metrics/depth.py:244`: silog can take `sqrt` of a tiny negative number and return NaN, which makes the whole epoch non-finite. Clamp at 0.
- `config.py:_coerce`: `--set data.max_depth=1000` yields a string, because the current value is `None`. Validation then raises `TypeError`.
- `loop.py:338`: the second `align_prediction` call is outside the `try`. An image with exactly one valid pixel crashes the eval.
- `common/table.py`: the CSV header comes from each row's own keys, so a late `val_skipped_images` column becomes an unlabeled trailing cell.
- `metrics/shadow.py`: SSIM is computed on luma only. ShadowFormer's test script scores per channel, so the numbers are not comparable with published ones. The module docstring covers only the regional metrics.
- `loop.py:99`: in the tiled path `_restore_size` is called with `prepared=None`. A model that resizes to exactly the padded size would be cropped instead of resized. No current adapter does this.

## Checked and found correct

- Tile grid: coverage, last tile flush with the image edge, Hann floor, accumulate/normalize. Metric depth and logits are never matched: segmentation models have no `protocol`, and metric models are not `"relative"`.
- `_restore_size`: crop for pad-only changes (DA3 518 → 512), resize otherwise.
- Alignment math: `scale`, `scale_shift_disparity` (LS in disparity space, floor clamp 1/max_depth, inversion), per-image scoring, and `validate_alignment` at build and at eval.
- Losses: `ssi`, `silog` and `berhu` masks and `eps`. `prepare_target` per output space, `scoring_input`, DA3 focal scaling, ZoeD-N bf16 `_safe_precision` (output cast to float).
- Freeze:
  - `apply_run_freeze` runs before the optimizer and DDP; resume loads the same param groups.
  - Prefixes verified per adapter: MiDaS `backbone.pretrained`, ZoeDepth `pretrained`, DA3 `backbone`, SegFormer `encoder`.
  - Depth Pro `encoder.*` and `fov.encoder.0` (the second ViT) are frozen, verified in the installed `depth_pro` package.
  - Frozen encoders are all LayerNorm ViTs, so there is no BatchNorm eval-mode issue.
- Eval protocol:
  - Eval uses `val_patch_size=0` (the whole image) or a deterministic centre crop; val and test share the code path.
  - `is_improvement` rejects NaN, including the first epoch; `jsonio` writes null.
  - Segmentation label decode is int64 with no overflow, and off-grid values are ignored.
- Domain adaptation:
  - `DIRECTION_ROLES`, `target_role` and the fake_A/fake_B choice are consistent.
  - The pix2pix A|B pair is A on the left and B on the right, matching AtoB. Keys are normalized, and a size mismatch raises.
  - Staging lives in separate sub-folders from the shadow staging.
- Shadow removal: mask derivation (relative darkening, opening), LAB conversion and operator order, PSNR on the 0..1 range, SSIM valid-region cropping, tiled predict (mask as a 4th channel, 3-channel output), iterations budget, and `--set shadow.*` coercion.
- Scripts: `ZOE_VENV_MODELS` equals `TIMM_PINNED_KEYS`. `--set shadow.iterations` and `--set shadow.init_weights` parse. The relative init path resolves against the queue's cwd, which is REPO. The resume flag is honoured by the superres and shadow runners, and domain adaptation resumes inside upstream.
