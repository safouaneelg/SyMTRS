# IMPROVEMENTS

Changes made in the repo, and why.

- `.gitignore`: `logs/`, `checkpoints/`, `*.pt`, `*.pth`, `*.ckpt`, `.venv-*/`, `third_party/*/`
  (the third-party repos are re-cloned at the revisions of `third_party/PINS.md`, not committed).
- `symtrs view` (symtrs/commands/view.py): port of Tools/view_dataset.py, EXR reading via symtrs.data.exr, no-shadow panel, headless `--save`, `--data-root/--split/--index`.
- D1–D6 (real data): `normalize_stem` strips the `<modality>_<light>_<res>_` prefix;
  `check_pairing` raises an error if 0 pairs / >10% unpaired; segmentation decoding
  `round(v/label_step)` (63, 5 classes, off-grid → 255, `label_step` in the checkpoint);
  `max_depth` (m, after `depth_scale` 0.01 cm→m) invalidates the sky; native `noshadow` role;
  domain staging by normalized key + symlink→hardlink→copy fallback (Windows); legacy SR pairs
  prefixed LR/HR; `make-dummy` now writes the real layout; doctor reports the
  pairing, the gray histogram and the depth in cm. Configs regenerated (69).
- `config.py` defaults: `num_classes 5`, `label_step 63`, `depth_scale 0.01`.
- `tests/test_real_layout.py` (24 tests), `tests/test_view.py` (9); two tests of `test_splits.py`
  made Windows-portable (`/tmp`, `\` separator). Suite: 486 passed.
- SegFormer b0/b1/b2 (`model/semantic/segformer.py`): the MiT encoder now comes from
  `transformers.SegformerModel` (`nvidia/mit-b0|b1|b2`, `output_hidden_states` → 4 feature
  maps). timm never provided MiT (`Unknown model (mit_b0)` on all versions). MLP head
  unchanged; `pretrained=False` builds from a `SegformerConfig` of the right size,
  without network. `requires` → `transformers`, extra `segformer = ["transformers"]`, rule 10 of
  AGENTS.md and README corrected (SegFormer no longer constrains timm).
- Depth Anything 3 (`model/depth/da3.py`): the `depth-anything/DA3*` checkpoints are object graphs
  of the `depth_anything_3` package (not transformers models: no preprocessor
  config, `model.backbone.*` keys). The adapter builds `DepthAnything3Net` via
  `create_object(config.json)` and loads the safetensors (`strict=False` limited to the 6 keys
  `output_conv2_aux.{1,2,3}.2` missing from the checkpoint, as the package does; any other missing
  key raises). Explicit preprocessing = ImageNet normalization of the base + `size_multiple=14`
  (no resize to 504). Optional `focal_px` for canonical metric depth
  (`× focal/300`). Installation: `pip install --no-deps depth-anything-3==0.1.1` (the wheel imposes
  numpy<2, xformers, open3d… useless here) + extra `da3` (omegaconf, addict, safetensors).
- MiDaS Swin-V2-T: (1) loads only under timm 0.6.12 (move of `downsample.reduction`
  between stages in timm >= 0.9) → `.venv-zoe`, explicit error message in the adapter,
  `TIMM_PINNED_KEYS` in `doctor`; (2) bug found at the first forward: the `dpt_transform` (384)
  transform was associated with a 256 checkpoint; mapping aligned with the upstream `hubconf`
  (`swin256_transform`, `swin384_transform`, `beit512_transform`).
- `.cache/torch/hub/isl-org_ZoeDepth_main` contained 9 files filled with zeros (corruption after
  BSOD) → repo deleted and re-downloaded; null MiDaS `.pyc` deleted. `.venv-zoe` (uv CPython 3.10)
  imports torch + timm 0.6.12 correctly (CUDA available), no need for python.org.
- `tests/test_offline_build_fixes.py` (12 tests). Suite: 516 passed.
- CUDA memory guard: `SYMTRS_CUDA_MEM_FRACTION` (read in `symtrs/common/seed.resolve_device`) →
  `torch.cuda.set_per_process_memory_fraction`. Under Windows/WDDM, a VRAM overflow silently spills
  into RAM (×100–500 slower); with the guard (0.92, set by `run_env`) we get a clean OOM in a few
  seconds. Verified: U-Net bs32 → OOM in 6 s. Tests:
  `tests/test_memory_guard.py`.
- `scripts/run_queue.py`: reusable `run_env()` / `run_job()`; on timeout, kills the process **tree**
  (`taskkill /T`), otherwise the subprocesses (upstream CycleGAN, shadow worker)
  keep the GPU.
- `scripts/smoke_all.py`: smoke test of each config on the real dataset, search for the batch
  that fits (halving on OOM), resumable (`runs/smoke/smoke.tsv`), writes `runs/queue.txt`
  with the validation budget (2 epochs; shadow 500 iterations) and the measured batch.
- Fixes coming from the smoke test (2026-10-17):
  - DA3 (`da3_large`, `da3_metric_large`) at 1024: `patch_size` 0 → 512 (`DEPTH_MODELS` table
    of `configs/generate_grid.py`, configs regenerated). Full fine-tuning of a ViT-L of 411 M
    parameters: ~6.5 GB of weights + gradients + Adam states resident, and a 1024 image has 4×
    the tokens of a 512 → OOM on 12 GB even at batch 1; the 512 crop (already used at 2048,
    passes) fits. At 512 and 256 the value stays 0: the 512 crop is the whole image there (a crop never
    exceeds the image, nothing is upsampled — rule 1). Test:
    `test_da3_crops_to_512_wherever_the_image_is_larger_than_that`. Note: the grid batches
    (4/8/16 at 512 for DA3) remain calibrated for a bigger card; `smoke_all.py` measures
    the batch that fits.
  - `midas_dpt_beit_large` routed to `.venv-zoe` (timm 0.6.12) like swin2: MiDaS's `backbones/beit.py`
    replaces `Block.forward` with a copy that calls `self.drop_path`, which timm >= 0.9
    split into `drop_path1/2` → `AttributeError` at the first forward. Added to `TIMM_PINNED_KEYS`
    (doctor), `ZOE_VENV_MODELS` (smoke_all), `run_queue.py` docstring, rule 10 of AGENTS.md,
    README. The MiDaS adapter now raises at construction (`_require_beit_compatible_timm`,
    test of the `drop_path` attribute rather than of the version) a `RuntimeError` naming `.venv-zoe`.
    MiDaS hub code not touched. Verified on CPU: `.venv-zoe`, `DPT_BEiT_L_384` builds and
    a 1×3×384×384 forward gives 1×1×384×384; `dpt_transform` transform = 384, consistent.
    The other MiDaS (small, DPT-Hybrid, DPT-Large) only patch the forward of the head ViT
    (`forward_flex`), never a timm `Block`: OK under timm 1.0 (smoke ok at 1024/2048).
    Tests: 3 added in `tests/test_offline_build_fixes.py`. Suite: 524 passed.

## 2026-10-17 — Depth Pro fits on 12 GB: `train.freeze` was never applied

- **Root cause**: `train.freeze` was read by `build_from_config` and stored by each
  adapter (`self.freeze`), but only `trainable_parameters()` / `apply_freeze()` applied the
  policy, and nothing on the training path called them (`build_optimizer` walks
  `named_parameters()` directly). The log nevertheless said `freeze='encoder' -> 952.0M trainable`
  (= everything). Result for Depth Pro: the 35 ViT-L passes (5×5 + 3×3 + 1 patches of 384 px) were
  recorded in the autograd graph → OOM > 11 GiB even at batch 1, whatever the crop.
  The crop could not help: `DepthPro.forward` imposes 1536×1536 (`assert H == W == img_size`,
  position embedding of 384 px × 4 fixed by the architecture) and `infer` resizes any input to it.
- **Fix**: `symtrs.engine.trainer.apply_run_freeze(model, cfg)`, called in `_run` before
  `model.to()`, DDP and `build_optimizer` (DDP freezes the set of parameters with `requires_grad`
  at the time of the wrap). Raises a `RuntimeError` naming `symtrs doctor --model <key> --freeze <policy>`
  if the policy leaves nothing trainable. Not called in `evaluate_run` (no gradient).
  A frozen encoder no longer needs an explicit `no_grad`: no input and no upstream parameter
  requires a gradient, so autograd records nothing (verified: 7.9 GiB measured).
- **Side effect fixed**: MiDaS puts its whole network under `backbone.` (ViT in
  `backbone.pretrained`, decoder in `backbone.scratch`); the default prefix `backbone`
  therefore also froze the decoder → 0 trainable parameters as soon as the policy became real.
  `apply_freeze_policy` now reads the class attribute `freeze_encoder_prefixes`;
  `MiDaSAdapter.freeze_encoder_prefixes = ("backbone.pretrained",)` → DPT-Large: 16.5 M trainable
  out of 344 M. ZoeDepth is not affected (`.pretrained` prefix correct; its MiDaS core is
  already frozen at construction anyway: 0.48 M trainable).
  **Consequence**: the MiDaS DPT-Large / BEiT-L and ZoeDepth configs with `freeze: encoder`
  were so far training as full fine-tuning; the smoke tests / batch measurements made before this
  fix for these models are to be redone (their memory drops, their maximum batch goes up).
- **Measurements** (RTX 5070 Ti Laptop 12 GB, amp fp16, `torch.cuda.max_memory_allocated`, 1 fwd+bwd
  step (+ AdamW) at batch 1):
  | case | peak |
  |---|---|
  | before: encoder not frozen, crop 512 | OOM at 11.1 GiB |
  | after: encoder frozen, crop 512 | 7.82 GiB |
  | after: encoder frozen, whole 2048 image (no crop) | 7.94 GiB |
  | native 2048 evaluation, tile_size 0, batch 1, amp | 5.74 GiB |
  fp32 weights: 3.6 GiB; trainable: 20.5 M (decoder 19.7 + head 0.4 + FOV head 0.4).
  End-to-end smoke through the trainer (`depth_pro_1024.yaml`, 2 steps, val + test): OK with
  `SYMTRS_CUDA_MEM_FRACTION=0.8`; at 0.7 (8.3 GiB) it fails narrowly (7.3 GiB allocated +
  ~1 GiB of fragmentation) — the 0.92 ceiling of `run_env` leaves the margin.
  AMP: fp16 + GradScaler works (finite loss); neither bf16 nor decoder checkpointing were
  needed (7.9 GiB < 10.5 GiB).
- **Configs** (`configs/generate_grid.py`, `depth_pro` row): batch 1 at every resolution
  (before 1/2/4/8: batch 2 does not fit), `patch_size: 0` everywhere (a 512 crop costs as much as
  the whole image, is upsampled ×3 to 1536 at training while evaluation
  downsamples: train/test shift), `grad_accum: 8` unchanged → uniform effective batch 8.
  Note text regenerated with these measurements.
- Tests: `tests/test_depth_pro_memory.py` (6) — policy applied on the module layout of Depth Pro
  (only decoder/head/FOV head trainable), empty policy inert, explicit error if everything is frozen,
  order `apply_run_freeze` < `model.to` < DDP < `build_optimizer` in
  `_run`, Depth Pro configs (batch 1, no crop, `freeze: encoder`, amp, effective batch 8),
  MiDaS keeps its decoder trainable.
- **Validation recommendation**: Depth Pro is fine-tunable on 12 GB, no need for a zero-shot
  fallback. Launch from the repo root (checkpoint `./checkpoints/depth_pro.pt`) with
  `SYMTRS_CUDA_MEM_FRACTION=0.92`, `--batch-size 1` (value of the configs; do not let
  `smoke_all` try 2). Zero-shot remains available:
  `symtrs eval --task depth --model depth_pro --resolution 2048 --data-root <root> --eval-split test`.

## Evaluation protocol (depth + segmentation) — 2026-10-17

Smoke audit (`docs/claude/AUDIT_smoke.md` A1, A4, A5): three protocol errors fixed on the
evaluation side, plus two defects discovered along the way. None of these fixes aims to improve
a number: they change what the number measures.

- **Val and test at 2048 were scored on ONE random 512 crop** (`factory.from_config`:
  `val_patch_size or train.patch_size`, and the datasets drew a random crop even outside
  training). Consequences: test not reproducible, not comparable with the 1024/512/256 rows
  (whole image), `best.pt` chosen on 2 windows. Fix:
  - `from_config`: evaluation never inherits `train.patch_size` anymore; `val_patch_size: 0`
    = whole image (`symtrs/config.py` documents the semantics); a value > 0 is a
    **deterministic central** crop (`transforms.center_crop_params`), never random.
  - `DepthTaskDataset` / `SemanticDataset`: random crop only if `augment` (hence
    training), central crop otherwise.
- **Tiling policy** (`configs/generate_grid.py::eval_tiling`): a row that trains on
  crops smaller than the image is evaluated, val and test, on the WHOLE image in sliding windows
  of the size of the training patch (`eval.tile_size = patch`, overlap = patch/4,
  Hann blend of `engine/tiling.py`): the network sees the field of view and the GSD of training,
  the memory stays that of a training step, every pixel is covered. A row trained on
  whole images is evaluated whole (`tile_size: 0`). Concretely: 2048 patch 512 → tiles
  512/128 (25 per image: U-Net, DeepLabV3, SegFormer, DA3, ZoeD-K/NK, ZoeD-N); 2048 patch 384 →
  384/96 (49: MiDaS small, DPT-Hybrid, DPT-Large, Swin2-tiny, BEiT-L); 1024 patch 512 → 512/128
  (9: segmentation, DA3, MiDaS DPT-Large and BEiT-L); all the others, including Depth Pro, `0`.
  Val and test go through the same `evaluate` function and read the same `cfg.eval`.
- **Tiling and relative depth**: the alignment (`align_prediction`) was already computed once
  per full image, after stitching — correct. But each pass of a relative model
  (MiDaS disparity, DA3 relative depth) has its own scale AND its own shift: blending the raw
  tiles leaves steps at the seams that no global alignment corrects.
  `predict_tiled(..., match_affine=True)` fits (weighted least squares) each new window
  onto the already stitched area before blending; `forward_native` enables it only if
  `model.protocol == "relative"` (metric depth and logits are absolute: never fitted).
- **Padding cropped, no longer stretched** (`loop._restore_size`): when `prepare_input` only
  pads to a multiple (DA3: 512 → 518), the output is cropped instead of resized —
  resizing shifted every pixel of the GT (up to 6 px) on the tiles and the image.
- **NaN is never "best"** (`trainer.is_improvement`): the `best is None → True` branch came
  before the NaN test, so a NaN first epoch was saved as `best.pt` (rmse=NaN) and nothing
  could beat it. A non-finite value never wins, not even the first one; it is flagged
  (per-epoch warning, `non_finite_monitor_epochs` in `summary.json`); if no epoch is
  finite, no `best.pt` is written, `best` is `null` and the message says to treat the run as
  failed (the test numbers then come from the weights of the last epoch).
- **Valid JSON**: new `symtrs/common/jsonio.py` (`json_safe`, `dumps`, `write_json`,
  `allow_nan=False` as a safeguard). All the `summary.json` (trainer, domain, shadow, super-resolution)
  and `per_image_*.json` write `null` for NaN/inf. `metrics.csv`: empty cell. `symtrs tables`
  reads `null` (and the old `NaN`) and displays `n/a`; sorting puts missing values last;
  the `--format json` output is valid.
- **mIoU** (`metrics/semantic.py`): a class with a null union (absent from the GT and from the predictions)
  is NaN → `null` in `summary.json` (`iou_k`, `dice_k`) instead of `0.0`, and stays excluded from the
  mean; a class absent from the GT but predicted keeps its IoU 0 (union > 0, standard
  behavior). New field `miou_classes` = number of averaged classes. Class 0 (background) has
  no pixel in any split of the current capture: the mIoU is therefore the mean of the classes
  present, and two mIoUs with different `miou_classes` are not comparable (note added in
  each segmentation config via `SEG_CLASS_NOTE`). The old test that required `iou_2 == 0.0`
  for an absent class was inverted (that was precisely the defect).
- **Tests** (`tests/test_eval_protocol.py`, 20): evaluation does not inherit the crop (spy on
  `dataloader`); eval crop = deterministic center, training crop always random (dummy dataset,
  depth and segmentation); val and test use the same function; tiling policy (unit + checked on the 91 generated YAMLs);
  relative tiles are fitted (the seam error is ≥ 5× lower than with raw
  blending) and absolute outputs are not; `forward_native` fits only `protocol == "relative"`; padding cropping;
  NaN/inf never "best", even at the first epoch; valid JSON (`parse_constant` that fails on NaN); CSV;
  `tables` with `null`/`NaN`; null-union class excluded, predicted-but-absent class counted,
  `null` all the way down to the JSON.
- **GPU probe** (untrained smoke ckpts, val split 2048, 24 images, 50% ceiling, machine shared
  with the running smoke so durations are indicative):
  | model | windows | allocated peak | duration |
  |---|---|---|---|
  | `unet_small` 2048 | 512/128 (25/img) | 0.65 GiB | 15 s |
  | `unet_small` 2048 | whole image | 5.21 GiB | 10 s |
  | `midas_small` 2048 | 384/96 (49/img) | 0.51 GiB | 39 s |
  | `midas_small` 2048 | whole image | 0.51 GiB | 16 s |
  For U-Net, tiled and whole give the same mIoU (0.0958) to 4 decimals: stitching
  changes nothing for a convolutional network. MiDaS changes (rmse 28.2 vs 29.9): normal, it does not
  see the same thing (native 384 tile vs image reduced to 384), and it is the tile that matches
  training. Cost: ~1.6 s/image for MiDaS-small, so ~40 s per validation pass of
  24 images — the big MiDaS/BEiT-L will cost more; lower `eval.tile_overlap` or
  `val_patch_size` (central crop) if per-epoch validation becomes the bottleneck.
- **Known limitation**: an existing smoke run (`runs/smoke/*`) was scored with the old protocol;
  its 2048 numbers are not comparable with those produced after this change.

## 2026-10-17 - Output conventions of the depth models (MiDaS, DA3-metric, ZoeDepth)
- **MiDaS: alignment in disparity space (rule 3 of AGENTS.md rewritten).** MiDaS outputs `s/z + t`
  (unknown scale AND shift). Before: `to_metres` inverted then `eval.align: scale`; `1/(s/z+t)`
  is not proportional to `z` and a scale does not remove `t`. Now: new alignment
  `scale_shift_disparity` (`metrics/depth.py`) = least-squares scale+shift on the raw disparity
  against `1/GT`, fitted disparity with floor `1/data.max_depth` (1000 m; `DEFAULT_MAX_DEPTH` if
  not provided), then inversion. The output space is declared in the registry (`ModelSpec.output_space`,
  `registry.depth_pairing(protocol, output_space)` = the only definition of (loss, align)); the adapters
  refuse an inconsistent `align` at construction and in `evaluate_depth`
  (`DepthAdapter.validate_alignment`); `DepthAdapter.scoring_input` provides the raw disparity (and not
  the metres) to these alignments; the space is written in the checkpoint metadata.
  The MiDaS configs (5 models x 4 resolutions) switch to `align: scale_shift_disparity` + `loss: ssi`.
- **`ssi` loss** (`model/depth/losses.py`): L1 on the residual of a per-image closed-form scale+shift fit
  (differentiable), divided by the spread of the target -> dimensionless (~1 for a constant
  plane, 0 if perfect; without it the loss would be ~1e-6 on disparities 1/70-1/270 and
  Adam's epsilon would smother the gradient). Before, `silog` on the disparity only forgave the
  scale. Neither the median/MAD normalization nor the multi-scale gradient term of MiDaS (not reproduced;
  `loss_kwargs.gradient_weight` adds an edge term).
- **Real probe** (midas_small 256, 6 test images, GPU, old vs new protocol on the same
  predictions): silog 47.9 -> 29.2; abs_rel 0.358 -> 0.239; rmse 56.5 -> 34.8; d1 0.39 -> 0.49.
  Unit test of the reason: on a disparity `37/z + 0.4`, the old path keeps an error
  >1000x greater than the new one (which recovers `z` exactly).
- **DA3-metric: `focal_px` derived from the field of view.** `DataConfig.camera_hfov_deg = 60.0`
  (config.py), `focal_px_for_resolution(res, fov) = (res/2)/tan(fov/2)`: 1773.6 px at 2048, 886.8 at
  1024, 443.4 at 512, 221.7 at 256. `build_from_config` (`_fill_camera_kwargs`) writes it into
  `cfg.model_kwargs.focal_px` if `ModelSpec.focal_kwarg` asks for it and if it is empty: it therefore follows
  the resolution automatically, depends neither on the crop nor on the tiles (same pixel size), is recorded
  in the checkpoint and `config.yaml` (so `autoloader` finds it again), and an explicit value wins.
  The adapter warns if a metric instance is built without a focal length. **Confirmed by the dataset owner on 2026-10-17 (was a hypothesis):**
  horizontal FOV 60 deg = default value of the capture actor (`FieldOfView = 60.f`,
  SyMTRSDatasetCaptureActor.h:97), editable per level; always 60 for the SyMTRS capture.
  **Real probe:** with the focal length applied, the zero-shot stays ~25x too small (predicted median
  2-9 m against 150 m of GT at 256/512/1024; abs_rel 0.95, d1 = 0). The focal length was necessary
  (without it the output is still ~3x smaller) but is not the main cause: DA3-metric
  reads the nadir view at 70-270 m as a close-up scene (domain gap). Only fine-tuning will fix it.
- **ZoeD-N: NaN under fp16, confirmed and fixed.** GPU probe (RTX 5070 Ti, 1 real test image, forward
  only): fp16 autocast -> ~10% of non-finite pixels at 512 and at 1024 (finite: 0.903 / 0.906); fp32 and
  bf16 entirely finite and identical to 3 digits (mean 1.674 m). ZoeD-K and ZoeD-NK finite in
  the 3 modes. Fix in the adapter (no new config key):
  `ZoeDepthAdapter._safe_precision` runs the N variant in bf16 (fp32 if the GPU does not support it)
  outside any CUDA autocast and returns float32; `amp: true` stays identical to the other configs.
  Verified: finite `silog` loss (2.49) and finite gradients in training; 1024 evaluation with
  `eval.amp: true` on 3 test images: abs_rel 0.19, d1 0.55 (before: "every valid pixel had a
  non-positive prediction").
- **ZoeD-K/NK: the "bin ceiling" is only true for K, and the hub has the wrong value.**
  Read in the pinned code: `bin_centers_type` is `normed` for K (bins bounded to `[min,max]_depth`)
  but `softplus` for N and NK (`SeedBinRegressorUnnormed`: "max_depth: Not used"), so
  N and NK are NOT capped (their 1-4 m are a misjudgment, not a ceiling). Worse: the hub
  entry `ZoeD_K` calls `get_config(..., config_version="kitti")` without `dataset="kitti"`, so the
  `max_depth: 80` is never merged and the head is built with 10 m (class default) whereas the weights
  were trained at 80. Fix: `ZoeDepthAdapter(max_depth=...)`, default
  `ZOED_K_TRAINED_MAX_DEPTH = 80` (faithful zero-shot; the key is also in the registry's `default_kwargs`,
  with an equality test), refused (`ValueError`) for N/NK which have no ceiling. The
  `zoedepth_k_*` configs set `model_kwargs.max_depth: 300.0` (covers the max of 279 m; the
  parameters keep their shapes, the normalized layout of the bins is simply stretched: starting
  point, not calibration). Real probe (zoedepth_k 512, 3 test images, GT median ~152 m):
  max_depth 10 (hub) -> pred. median 1.4-2.8 m, abs_rel 0.98; 80 -> 11-22 m, abs_rel 0.88;
  300 -> 41-84 m, abs_rel 0.53; silog identical (27.4) because the range only stretches the output. The
  K zero-shot must therefore use the default 80, not the value of the training config.
- Docstring of `zoedepth.py` corrected: ZoeD-N is the NYU metric model (ZoeD_M12_N), scored here
  as relative by protocol choice (indoor metres not transferable), not "without metric supervision".
- **Tests**: `tests/test_depth_conventions.py` (29): pairing by (protocol, space), exact return
  of `z` from an affine disparity, the old path fails, floor/clamp, refusal of inconsistent
  alignments (messages verified), `scoring_input` keeps the sign, `ssi` loss (scale+shift
  invariance, dimensionless, finite gradient, constant prediction), focal length per resolution /
  independent of the crop and tiles / FOV the only setting / explicit value takes priority, DA3
  `focal/300` scale, default K range = 80, refusal of `max_depth` for N/NK, N alone excluded from fp16,
  effective bf16 on CUDA. Grid and registry tests rewritten to enforce the `depth_pairing` pairing
  at every resolution and MiDaS as `ssi`/`scale_shift_disparity` everywhere. 581 tests
  green, `generate_grid.py --check` OK.
- **Consequence for the existing**: the `config.yaml` of the MiDaS smoke runs (`align: scale`) are
  now refused at construction (explicit message); re-launch from the regenerated configs.
  The MiDaS numbers of the old runs are not comparable.
- **Not done / to note**: the depth preview still plots the unaligned prediction (`to_metres`), so
  for MiDaS a map in 1/disparity without scale; Depth Pro keeps `f_px: null` (it estimates its own
  FOV): `focal_kwarg="f_px"` in the registry would suffice to give it the same derived focal length if
  we want to anchor this baseline (this changes the nature of the test, to be decided).

## 2026-10-17 - Shadow ⇄ no-shadow translation via the domain front-end (optional goal 3)

- **The pair is data.** `upstream.DIRECTION_ROLES` (`symtrs/model/domain/upstream.py`) gives, per
  direction, the A/B folders and the upstream direction: `day2night`/`night2day` = `rgb_day`/`rgb_night`,
  `shadow2noshadow`/`noshadow2shadow` = `rgb_day`/`rgb_noshadow` (A is always `rgb_day`). Staging
  (`build_staging_dir(..., direction=)`), `upstream_direction`, real target of the FID/KID (`target_role`) and
  (the generated side is NOT derived from it: it is `fake_B` everywhere, see the "Generated side" entry below) derive from it; no more hard-coded `== "day2night"`. The
  day/night behavior is unchanged (existing configs and tests identical; the 4 existing YAMLs regenerated
  byte for byte). Still 2048 only (explicit refusal otherwise, the message cites `rgb_noshadow`).
- **4 generated configs** (`cyclegan|pix2pix` x `shadow2noshadow|noshadow2shadow`) from the tables of
  `configs/generate_grid.py` (`DOMAIN_DIRECTIONS`, `DOMAIN_FOLDERS`, `DomainModel.shadow_note`):
  95 files in total (`--check` OK). `tests/test_vendored_and_grid.py` compares the generator's table
  (which cannot import the package) to `DIRECTION_ROLES`.
- **Paired scoring** for `shadow2noshadow` only: `runner.score_shadow_removal` reuses as is
  `stage_split` (mask derivation), `score_split` and `symtrs.metrics.shadow` of `shadow_removal`
  (PSNR/SSIM/LAB error on all/shadow/non-shadow, same columns) in addition to FID/KID; the numbers are
  therefore comparable with the ShadowFormer/Restormer/Uformer rows. Size: the front-end passes no
  `--preprocess`/`--load_size` (unchanged); `test.py` therefore writes 256x256 (whole image resized),
  resampled bicubic to the size of the ground truth before scoring. The loss of detail
  counts against the model; the note is written in `summary.json` (`paired_scoring`) and in the header
  of the config. `noshadow2shadow`: FID/KID only (a "shadow area" PSNR would measure the quality
  of a shading, not of a removal).
- **Pre-existing bug fixed: pix2pix could not start.** The vendored `AlignedDataset` ignores
  `trainA/trainB` and reads `<dataroot>/<phase>/*.png` as side-by-side `[A | B]` images; without that it
  falls back to its `hr/`/`night/` auto-split and crashes (`FileNotFoundError`). No day/night pix2pix run
  could therefore ever run. `build_staging_dir(aligned=True)` (pix2pix) now writes these pairs
  (`write_aligned_pairs`, `combine_A_and_B.py` format of upstream, reused if newer than their
  sources, stale ones deleted). The raw-dir + `--train_ratio` would have lost images (sum < 1 required).
  Cost: one lossless PNG per scene in `<run>/staging/<split>/<split>/`.
- `symtrs tables`: `direction` column (4 rows per model otherwise indistinguishable) and PSNR/SSIM/LAB columns
  in the `domain_adapt` group. `sweep`: the native-only message cites `rgb_noshadow` (the plan still does not vary
  the direction: pass `--extra model_kwargs.direction=...` AND one `--out-dir` per direction).
  Docs: AGENTS.md §1/§2.5/tree (95), `configs/README.md`, `README.md`, registry descriptions.
- **Tests** (+15, 603 green): staging A=`rgb_day`/B=`rgb_noshadow` paired by key for both directions,
  day/night unchanged, direction -> upstream side/target table, argv of both directions accepted by the
  vendored parser, generated side per direction, refusal at 1024 (message), paired scoring on a synthetic pair
  (16 px output resampled; shadow kept -> shadow PSNR drops, non-shadow unchanged) and named error
  if images are missing, pix2pix `[A|B]` pairs (halves, cleanup), `tables` columns.
- **Smoke** (dummy 512 px, 6/2/2, `--epochs 1`, ~37 s each): `pix2pix_shadow2noshadow` finishes, FID/KID
  + PSNR 15.7 / SSIM 0.39 / mae_lab 37.4 (meaningless dummy numbers); `cyclegan_noshadow2shadow` finishes
  (FID/KID only). Real launch: `symtrs train configs/domain_adapt/pix2pix_shadow2noshadow.yaml
  --data-root <root>` (same for `cyclegan_*`), after the queue, single-GPU, `train.epochs: 40`.

## 2026-10-17 - FID/KID comparability, smoke batch, shadow model init (REVIEW_night #6, #7, #8)

- **#6 Asymmetric FID/KID fixed** (`symtrs/metrics/domain.py`, `model/domain/runner.py`). Before, the
  real 2048 images were reduced to 299 without antialiasing (aliased) and the generated 256 ones enlarged
  (smooth): the FID partly measured sharpness. Now: the generated size is read from the files
  (`generated_image_size`, named error if sizes are mixed), the real ones are resized to this size
  with antialiasing (`match_resolution`, `extract_features(resize_to=...)`), then a single antialiased
  299 resize identical for both sides (`inception_preprocess`). **KID becomes the main number**
  (`PRIMARY_METRIC`, listed before `fid` in the dict, `summary["primary_metric"]`, `symtrs tables`
  columns: `kid, kid_std, fid, fid_rank_deficient`): with N=24 per split and 2048 dimensions, the
  FID covariance has rank <= N-1, hence is biased; the FID is still computed, with the
  `fid_rank_deficient` flag and a note (`metric_notes.fid`). Summary: `feature_resize_h/w`. The old FIDs are
  not comparable with the new ones.
- **#7 `scripts/smoke_all.py`**: halving the batch now doubles `--grad-accum` (depth and
  segmentation, the only tasks whose loop accumulates) to keep batch x accum = effective batch of the
  grid; otherwise the effective batch silently changed at constant LR. For superres / shadow / domain
  (no accumulation) the effective batch really decreases: the `eff_batch` < `grid_eff_batch` column and the
  queue line (`# eff_batch=... (SHRUNK)`) say so. `smoke.tsv`: 3 columns added at the end
  (`accum`, `eff_batch`, `grid_eff_batch`); `load_rows` reads the old 9-column rows (accum
  unknown = value of the config, the queue line then does not emit `--grad-accum`), `load_done` keeps its
  old form. OOM during whole-image evaluation (eval batch = 1) distinguished from a training
  OOM via the traceback (`classify_oom`: innermost frame among train_one_epoch /
  _training_step versus evaluate* / validate_crops / predict): immediate `FAIL ... eval OOM` instead of
  going down to 1. Without a traceback -> old behavior (we divide). Warning: the `ok` rows already
  written by the pass in progress with a reduced batch ran with unchanged accum; they show up as
  `SHRUNK` in the queue, to be re-run with `--only` if the effective batch matters.
- **#8 Shadow**: `load_matching` raises a `RuntimeError` (file + counters) if fewer than 90% of the model's
  tensors are loaded (before: 0 tensors went through silently, a de facto "from scratch" run). The summary
  records `init_weights`, `init_source_task` (ShadowFormer: ISTD+ shadow removal; Restormer: real SIDD
  denoising; Uformer: SIDD denoising, neither of the two publishes shadow-removal weights),
  `init_tensors{loaded,total}` and `metric_notes.ssim_y`. To be written in RESULTS next to each row.
  **SSIM renamed `ssim_y`** (Rec.709 luma only; ShadowFormer averages per RGB channel, hence not
  comparable with the published numbers): keys `ssim_y`, `ssim_y_shadow`, `ssim_y_nonshadow` in
  `metrics/shadow.py`, `CSV_FIELDS`, `tables` and the tests; the old `summary.json`/`metrics.csv`
  keep `ssim`.
- **Tests**: `tests/test_comparability.py` (15): antialiasing (high-frequency stripes), generated size read
  and mixed sizes refused, KID before FID and rank-deficient flag, consistent note/main key,
  `load_matching` (0 and 8/10 tensors refused, full accepted with counters), init source per model,
  `ssim_y`, `halve` (effective batch kept), reading of `smoke.tsv` rows with 9 and 12 columns,
  train/eval attribution of an OOM, reading of `grad_accum`. Suite: 639 green; only failure,
  `test_eval_robustness.py::test_set_coerces_an_optional_float_that_is_currently_none` (concurrent work
  on `config.py`/`--set`, outside this scope). `generate_grid.py --check` OK.

## 2026-10-17 - Review: LR scheduling, resume, test weights, robustness (trainer / loop / table / config)

**Changes the semantics of the depth and segmentation runs: the validation runs already done are to be redone**
(the effective LR was wrong since 76dd299).

- **#1 LR scheduling** (`engine/trainer.py`, `engine/loop.py`): `build_scheduler` was sized in
  optimizer steps (`epochs * steps_per_epoch`) but `scheduler.step()` was called once per EPOCH: the LR
  stayed on the first percent of the warmup ramp (a 2-epoch run trained at 4-8% of the
  configured LR, never any cosine decay). The scheduler step is now done in
  `train_one_epoch(..., scheduler=)` right after each `optimizer.step()` (including the last
  partial accumulation group; a step skipped by the GradScaler still advances the schedule). The size
  per epoch comes from `optimizer_steps_per_epoch(len(loader), grad_accum)` = ceiling (5 batches, accum 2 = 3
  steps; the old `//` counted 2). The warmup is capped at half of the run (message shown): the 2-epoch
  validation runs with `warmup_epochs: 2` otherwise ended at the peak without ever decaying; any run
  with `warmup_epochs <= epochs/2` is unchanged. The scheduler state is saved after the steps, with
  `scheduler_unit: "optimizer_step"`; an old checkpoint (per-epoch step) is not loaded as is but the
  scheduler is repositioned to `start_epoch * steps_per_epoch`.
- **#2 Resume and test weights**: `best`, `best_epoch`, `history`, `non_finite_epochs` are in the
  payload of `last.pt`/`best.pt` and restored by `--resume` (fallback to the neighboring `best.pt` and
  `metrics.csv` for old checkpoints): a worse epoch after resume no longer overwrites `best.pt`. The test pass
  reloads `weights/best.pt` before evaluating (`summary["test_weights"]` = `best`, `last` if no finite
  epoch, `released` in zero-shot); as for shadow, `test` therefore corresponds to `best`.
- **#9**: the alignment fit of `evaluate_depth` is inside the `try`: an image with < 2 valid
  pixels is skipped and counted (`skipped_images`) instead of crashing the evaluation; `append_csv_row`
  rewrites the header with the union of the keys when a column appears later (`val_skipped_images`), the
  old rows get an empty cell; `--set data.max_depth=1000` gives a float (type read from the
  dataclass annotation for an `Optional` field whose value is None; `none`/`null` resets to None).
  The `silog` loss (`model/depth/losses.py`) already clamped the variance to 0: nothing to change; the
  `silog` metric of `metrics/depth.py` (sqrt of a small negative) is left to the metrics agent.
- **Tests**: `tests/test_trainer_schedule.py` (real `build_scheduler` + real `train_one_epoch` on a
  small CPU model: peak reached after the warmup, decay down to `min_lr_ratio`, number of scheduler steps =
  number of optimizer steps including the partial group, warmup cap, test on `best` / `last` fallback, resume
  without overwriting `best.pt`, real improvement taken after resume, scheduler position restored) and
  `tests/test_eval_robustness.py` (CSV header, `--set` coercion, 1-valid-pixel image).

## 2026-10-17 - Metrics and tile stitching: night review (points 3, 4, 5, 9)

- **mIoU: denominator fixed by the data (point 3).** `ConfusionMatrix.supported_classes()` = classes
  having at least one ground-truth pixel in the confusion matrix of the whole split (never per image).
  `miou`, `mean_dice` and `mean_acc` average over this set, identical for all the models of a same
  split. Before, a single predicted "class 0" pixel (0 exists in no split, the head has 5 outputs) gave
  IoU_0 = 0 and made `miou_classes` go from 4 to 5, i.e. about -20% relative for a one-pixel error.
  `iou_k` stays NaN/null without ground truth or prediction, is 0 for a class predicted without ground truth (visible),
  but does not enter the mean. New numeric keys: `support_k` (ground-truth pixels) and `fp_pixels_k`
  (pixels predicted k whose ground truth is another class); the averaged identifiers are the `k` with
  `support_k > 0` (numeric keys, because this dict goes as is into the CSV). `format_report` displays
  `fp_pixels`. The header of the 24 segmentation configs (regenerated, `--check` OK) describes the new rule.
- **Tile stitching by output space (point 4a).** `tiling.match_mode(model)`: disparity (MiDaS) ->
  `affine` (scale + shift); depth up to a scale (relative DA3, ZoeD-N scored as relative) ->
  `scale` (regression through the origin, never a shift: the final `scale` alignment could not remove it);
  metric depth and logits -> `none`. Read from `protocol` + `output_space` of the adapter (the test
  walks the whole registry). `predict_tiled(match=..., stats=...)`; `match_affine=True` is still accepted
  (= `affine`). `forward_native(..., stats=)`.
- **Robustness of the fit (4b).** `fit_overlap` replaces `fit_affine`: **relative** variance threshold
  (`var <= 1e-6 * mean^2`, unit-independent), "shift only" fallback (slope 1) if the overlap is flat,
  correlation < 0.5 or non-positive slope (never a flip), slope bounded to [0.25; 4] (shift
  recomputed on the means), symmetric slope `std(ref)/std(src)` instead of OLS (biased toward 0 under
  noise). Counters `windows/matched/fallbacks/unmatched`; `evaluate_depth` writes `tile_match_fallbacks`,
  `tile_match_unmatched` and `tile_windows` (split totals, only if tiling took place).
- **Anchoring drift (4c).** Verified: each window is fitted on the overlap with the already assembled field
  (Hann blend of ALL the windows laid down, not just the neighbor). The anchoring stays the first
  window (raster order): the global frame is an affine of the true one, which the per-image alignment removes.
  Documented in the module; no global optimization.
- **Non-positive predictions scored, no longer removed (point 5).** `compute_depth_metrics` replaces any
  prediction <= 0 or non-finite (after alignment) by `min_depth` (default `DEFAULT_MIN_DEPTH` = 1e-3 m)
  instead of removing it from the mask, and returns `invalid_pred_fraction` (added to `DEPTH_METRIC_NAMES`). An
  entirely non-positive image is no longer "ignored" (`AllInvalidError`): it is badly scored, as it should be.
- **silog (point 9, metric part).** Variance bounded to >= 0 before the root (real case found:
  -2.2e-16 -> NaN that made the epoch non-finite).
- **Effect on the numbers already produced.** mIoU: a model that predicted class 0 pixels sees its mIoU
  go back up (up to x1.25); those that did not are unchanged. Depth: models having pixels <= 0
  see RMSE/abs_rel increase (before: optimistic); the others are unchanged. Stitching: MiDaS
  2048 in tiles -> same fields on normal textures, more robust on flat areas; relative DA3 / ZoeD-N
  no longer receive a shift (real change for them at 2048). Any earlier tiled run is to be re-evaluated.
- **Tests**: `test_segmentation_metrics.py` (same denominator for all the models, union over the whole
  split, visible ghost class), `test_eval_protocol.py` (stitching mode per output space, complete
  registry), `test_tiling.py` (random a,b per tile recovered, flat half without blow-up, no shift
  in `scale` mode, unit-independent threshold, bounded slope, non-flip, OLS bias), `test_depth_metrics.py`
  (negative/NaN/inf prediction scored, `invalid_pred_fraction`, silog without NaN on a case with variance -2e-16).

## 2026-10-17 - Domain adaptation: the generated side is `fake_B` in every direction

- **Bug.** `DomainTask.generated_paths` read `*_fake_A.png` for `night2day` and `noshadow2shadow`
  (`BtoA`). But upstream's `set_input` (`models/pix2pix_model.py`, `models/cycle_gan_model.py`) does
  `real_A = input["A" if AtoB else "B"]` and `forward` does `fake_B = G(real_A)`: `--direction BtoA` swaps
  the INPUTS, the requested translation remains `fake_B`. Consequences observed on the real set: pix2pix
  only writes `fake_B`/`real_A`/`real_B`, the scoring found nothing and returned `test: {}` (a "successful" run
  without a metric); CycleGAN scored `fake_A = netG_B(real_B)`, the OPPOSITE translation (day -> night), against
  real day images: a wrong number without an error.
- **Fix.** `runner.GENERATED_LABEL = "fake_B"` (explanation and pointer to the upstream lines in the
  comment), used by `generated_paths` and `score_shadow_removal`. The names come from `B_paths` (the
  staged night/no-shadow files), named by the same normalized key as the day ones: the key
  pairing of the paired scoring is unchanged.
- **No more silent `{}`.** `generated_paths` raises `RuntimeError` (folder, searched pattern, probable cause)
  if nothing is generated; `evaluate_adaptation` also raises if the real folder (`rgb_<target>`) is missing or
  empty. `summary["test"]` can no longer be `{}` on the domain side.
- **Tests.** `tests/test_domain_commands.py`: reading of the upstream sources (the two facts the
  choice depends on), `fake_B` for the 4 directions x 2 models, named errors. The old tests that expected
  `fake_A` for `BtoA` were corrected.
- **To redo.** Any domain `summary.json` produced before this fix for `night2day`/`noshadow2shadow` is
  to be rejected (CycleGAN: wrong direction; pix2pix: empty) and re-evaluated with
  `symtrs eval <config> --eval-split test` (the scoring re-runs `test.py` then re-reads `*_fake_B.png`).

## 2026-10-17 - Super-resolution: silent failures, crops, whole-image evaluation

- **Visible failure.** `symtrs/model/superres/runner.py`: a non-zero exit code of the legacy trainer raises
  `SuperResolutionError` (CLI: exit 2) and writes `summary_failed.json` instead of `summary.json`; an
  `--eval_only` without `test_metrics.json` also raises; a stale `test_metrics.json` is deleted before the
  launch. A training without `val/` is refused (rule 6) instead of splitting 80/20 inside `train/`.
- **Single verdict.** `scripts/run_queue.py:run_verdict` (rc 0, readable `summary.json`, `returncode` absent
  or 0, non-empty `test` except eval-only), used by the queue and by `smoke_all.py`; `verified()` rechecks
  the `ok` rows of the ledger. Legacy eval frames added to the OOM classification.
- **Legacy trainer** (details in `symtrs/legacy/README.md`, "Changes after the move"): aligned crops
  for the 4 models (existing `random_crop_pair`); validation = `val/`; val and test on the whole image,
  one image at a time, in windows (`symtrs/engine/tiling.predict_tiled`, new `scale` parameter);
  per-image PSNR/SSIM then mean; SRGAN accepts x1; SwinIR built with `img_size` = crop.
- **Grid.** LR patch = 256/scale for SRCNN/AE/SRGAN (fixed target region of 256 px), 64 for SwinIR;
  eval window 512 px LR, overlap 64 (`SR_EVAL_TILE`, deliberate departure from "window = crop").
- **Tests.** `tests/test_superres_training.py` (new) + `tests/test_superres.py`.
- **Domain.** In addition to the `fake_B` fix: the scoring requires as many generated images as real images
  (`test_a_partial_translation_of_the_split_is_an_error`).
- **Shadow removal reference.** `scripts/shadow_identity_baseline.py`: the shadowed input as the prediction
  (same pairs, mask and metrics); val 12.15 / test 12.40 dB PSNR, shadow 7.7 / 7.9, non-shadow 30.0 / 27.9.

## 2026-10-17 - Relative run directories for the subprocess tasks

- **Bug.** The validation queue ran `symtrs train <cfg> --out-dir runs/validation/<name>` (relative).
  Shadow removal (worker, cwd = upstream repo under `third_party/`) failed on
  `FileNotFoundError: runs/validation/restormer/config.yaml`; CycleGAN/pix2pix (upstream `train.py`,
  cwd = `pytorch-CycleGAN-and-pix2pix/`) got a relative `--dataroot`, did not find `trainA/trainB` and fell
  into the auto-split branch. Super-resolution passed only because its trainer inherits the caller's cwd.
- **Fix, in one place.** `RunConfig.anchored()` (`symtrs/config.py`) returns a copy with `out_dir`
  (including the canonical default `runs/<task>/<model>/<res>`), `data.root`, `shadow.init_weights` and
  `train.resume` resolved against the caller's cwd. `run_shadow_removal`, `run_domain_adaptation`,
  `run_super_resolution` and the shadow `--dry-run` printer call it first, so staging, `config.yaml` (which the
  worker re-reads), `--checkpoints_dir`, `--results_dir` and the score step all derive from absolute paths.
  The output folder is the same as before relative to the caller's cwd. The old per-field `.resolve()` in the
  shadow runner is gone; `train.resume` (also read by the worker) was not covered before.
- **Tests.** `tests/test_relative_run_dir.py` (7): the config copy, the shadow worker argv and written
  `config.yaml`, the shadow dry run, the CycleGAN/pix2pix `--dataroot`/`--checkpoints_dir`/`--results_dir`,
  the SR `--out_dir`/`--lr_root`; the cwd is a temp dir that is neither the repo nor upstream, and the tests
  fail without `anchored()`.
- **Side effect.** `config.yaml` of the three subprocess tasks now stores an absolute `out_dir` and `data.root`.


## ShadowMaskFormer, a fourth shadow-removal model (2026-10-17)

- **What.** `shadowmaskformer` (IEEE TAI 2025, lizhh268/ShadowMaskFormer, pinned at `9c505e1`,
  2025-05-08), the `-b` variant: 2,284,608 parameters measured (ShadowFormer 11,372,617 counted the
  same way). Registry key, `ARCHS` entry, `SHADOW_MODELS` row in `configs/generate_grid.py`
  (patch 256, batch 4, AdamW lr 2e-4, wd 0.02, cosine to 1e-6 = upstream's recipe on the shared
  Charbonnier loss), `configs/shadow_removal/shadowmaskformer.yaml`; 96 generated configs (was 95;
  AGENTS.md, README.md, configs/README.md updated). The class is imported from the checkout like the
  other three; `third_party/PINS.md` and `scripts/fetch_third_party.py` carry the pin.
- **Range conversion is per architecture.** The network takes image and mask in [-1, 1] (shadow =
  +1) and returns an unclamped [-1, 1]. The worker keeps everything in [0, 1] (loss, validation PSNR,
  tiler, PNG writer) and converts at the network boundary only: `Arch.input_range` ("unit" |
  "signed") with `to_network_range` / `from_network_range` in `archs.py`, applied inside
  `worker.forward`, so training, validation and tiled prediction cannot disagree. The three existing
  models are "unit" and behave exactly as before.
- **No released weights, no licence.** `Arch.released_weights = False`: the model is always trained
  from scratch (`shadow.init_weights` empty is accepted, zero-shot still refused), `describe()` and the
  config header say "no released weights: trained from scratch ... not like-for-like". The checkout
  is gitignored and fetched by script (PINS.md notes the missing licence); nothing of it is committed.
- **Dependencies.** torch + `timm.models.layers` only (FutureWarning on timm 1.0.30);
  upstream's `train.py`/`test.py` (pytorch_msssim) are never imported.
- **Tests.** `tests/test_shadow_removal.py`: the declaration, the config header, zero-shot refused with
  empty init, range round-trip and mask polarity, `worker.forward` with probe networks (signed vs unit),
  a CPU forward on a size that is not a multiple of 4, parameter count in 2.2-2.4 M;
  `tests/test_comparability.py`: from-scratch declaration.
- **Not changed.** `scripts/smoke_all.py` `SHADOW_INIT` has no entry on purpose;
  `runs/queue_shadowmaskformer.txt` (gitignored) holds the one validation line, without init weights.


## `kid_std` no longer a fake zero, and a "do nothing" reference for domain adaptation (2026-10-17)

- **`kid_std`.** `compute_kid(num_subsets=0)` (what the domain runner uses) returned `kid_std: 0.0`,
  which every validation summary then showed next to N=24 as if a zero spread had been measured. It
  is now `None` (`null` in `summary.json`; tables print `n/a`); the KID itself is unchanged (same
  full-set unbiased MMD^2, same default). `scripts/results_report.py` also prints `n/a` for the
  legacy `0.0` in summaries written before this change. Tests: `tests/test_comparability.py`.
- **Identity reference.** `scripts/domain_identity_baseline.py` scores the untranslated input
  images of the test split against the real target-domain images for the four directions, with the
  runner's own feature code (`extract_features`, `distribution_metrics`, 256 px target path), the
  input resized like upstream's `test.py` (PIL bicubic to 256), and a symmetric-resize sensitivity
  column. It also writes a per-domain noise floor (first vs second half by sorted key, plus even vs
  odd): the test split holds two locations, so the contiguous halves measure a scene shift as well as
  sampling noise, and the pair brackets what "same domain" looks like here. Output:
  `runs/validation/_baselines/domain_identity.json` (gitignored); `scripts/results_report.py` appends
  the identity rows and the noise floor to the domain table. Tests:
  `tests/test_domain_identity_baseline.py` (pairing and splits torch-free).


## Super-resolution training runs now write `config.yaml`; the runs report never drops a run (2026-10-17)

- **Runner.** `run_super_resolution` wrote `config.yaml` only when re-scoring (`eval_only`), so the 16
  super-resolution validation runs had none, `scripts/runs_report.py` (which keys on it) dropped them
  and `RUNS.md` listed 80 runs instead of 96. The (anchored) config is now written for training and
  re-scoring alike, after the dry-run return: a dry run writes nothing, as for the other tasks.
  Test: `tests/test_superres_training.py::test_a_training_run_writes_config_yaml_and_a_dry_run_does_not`.
- **Report.** A run directory with `summary.json` but no `config.yaml` is listed from the task, model,
  resolution, epochs and trainer argv that `summary.json` recorded, marked with a dagger and the note
  "config.yaml not written by the SR front-end before 2026-10-17 fix; settings from summary.json
  `command`"; the recorded argv is printed under the commands and redacted like the logged ones
  (`<DATA_ROOT>`/`<REPO>`). A directory with neither file is named on stderr; directories starting
  with `_` are skipped. The 16 old runs were not given a fabricated `config.yaml`. `RUNS.md` now
  reports 96 runs.
