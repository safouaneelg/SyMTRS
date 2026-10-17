# SUGGESTIONS

Future leads, not done.

- When the 20k-transform dataset is ready: re-run the grid with the epochs of the generated configs
  (the current runs are 2-epoch validation runs).
- `SyMTRS_Pipeline.md` contradicts itself (gray ×51 vs ×63, 6 vs 5 classes): to be fixed on the Docs side.
- Sensitivity of the derived shadow-mask threshold (ShadowFormer, per-region metrics).
- Tiled shadow-removal inference: ~10 s per 2048 image (Restormer, 256 tiles one at a time, 0.6 GB VRAM) → batching the tiles together would speed up evaluation ~5–10×.
- Super-resolution: with crops, one epoch = 1 crop per training image (96 → 6 steps at batch 16).
  Before the real runs, add a "crops per image per epoch" option or revisit the epochs/batches
  of the grid (fixed when training was done on the whole image).
- `swinir_x1`: ~15 min per smoke epoch, almost all of it in evaluation (masks recomputed for every
  512 window on CPU) → cache the masks or validate less often.
- Systematic reference rows in RESULTS: "identity" for shadow removal
  (`scripts/shadow_identity_baseline.py`, test PSNR 12.4 dB) and bicubic for SR (x2 22.9 / x4 20.9 /
  x8 19.3 dB measured by the SR agent), and for domain translation the input-vs-target KID
  (`scripts/domain_identity_baseline.py`: day->night 0.231, night->day 0.238, shadow->noshadow 0.104,
  noshadow->shadow 0.096; both are in RESULTS.md); without them a PSNR or KID does not say whether the
  model does better than nothing.
- Metric depth (Depth Pro, ZoeD-NK, DA3-metric): predictions ~10x too close in zero-shot. The 2-epoch
  validation runs show the frozen-encoder models do not leave that regime: Depth Pro and ZoeD-NK
  (frozen encoder, lr 5e-6, decayed to 5e-8 over the compressed 2-epoch cosine) move val abs_rel
  0.937 -> 0.928 (Depth Pro 2048) and 0.949 -> 0.949 (ZoeD-NK 1024), and score test abs_rel ~0.97-0.98
  (predicting ~2% of the true depth): effectively zero-shot. This is not a code bug: DA3-metric
  (unfrozen, lr 1e-5) goes 0.81 -> 0.50 val abs_rel in one epoch and lands at test abs_rel 0.27-0.31,
  which also supports `depth_scale: 0.01` and `camera_hfov_deg: 60` being roughly right. For the
  real runs, unfreeze the encoder or raise the LR for the metric heads, and run enough epochs.
- Dataset regeneration, camera altitude (see DATASET.md, "Depth range and camera altitude"): the
  actor places the camera above the first `ECC_Visibility` hit, so at loc010/loc011 (test) it sits
  70 m above a skyscraper roof and the ground is 152-275 m away, outside the training range (to
  ~182 m). Either trace against terrain/streets only (or take the lowest hit), or keep
  "above the first surface" on purpose and balance it across train/val/test.
- Dataset regeneration, metadata: write a per-frame JSON with the altitude above ground, the camera
  position and yaw, and the intrinsics (FOV). The dataset has none today, and the package's
  `data.camera_hfov_deg: 60` is an assumption confirmed only by a car-size estimate (54-63 deg).
- Shadow removal, validation cadence: `shadow.val_every` is 500 in the validation runs, i.e. one validation
  point, so `best` is `last` and the curve is empty. For full runs use ~100 (validation on 2048 images is
  tiled and slow, so weigh it against the iteration count).
- Effective batch against dataset size: the grid batches are chosen for the 20k-transform dataset. On the
  96-image preview an effective batch of 64 is 2 optimizer steps per epoch, both inside the warmup (DA3-metric
  256/512 show identical epoch-0 and epoch-1 validation, abs_rel 0.87-0.96). Fine for the real runs,
  meaningless here; for any run on a small subset, cap the effective batch (or the accumulation) so an epoch is
  at least a few dozen steps.
- Domain translation metrics: KID on 24 images from 2 test locations is mostly a scene-shift measure (the
  split-half noise floor is up to 0.18). With the larger dataset use more locations and more test images, KID
  with several subsets (so `kid_std` is a real spread), and keep the identity row beside every translator.
- Shadow removal comparison: the four models start from different weights (ISTD+ shadow removal; SIDD
  denoising x2; scratch). For a ranking, give every model the same start (all from scratch with a real
  budget, or all from the best available released weights) and report the start in the table.
- `scripts/smoke_all.py` batch probe: add one whole-image evaluation to the probe. The few-frame smoke does
  not score a full image, so it chose batches that OOMed in the real run (`da3_large` 256/512/1024,
  `zoedepth_nk_512`; patched by hand with a smaller batch and a raised `grad_accum`).
- Next light shadow model: BMNet (0.43 M parameters, weights in its repo, two-stage training, the path to
  `condition.pth` is hard-coded upstream). Cheaper than ShadowMaskFormer (2.28 M) and ShadowFormer (11.4 M),
  which makes it a useful low-compute point.
