# AGENTS.md

How to set up, run and extend this repository. Written for an automated coder working here
with no prior context; a person should find it useful for the same reasons.

**The dataset is under revision.** The SyMTRS dataset is being regenerated in response to
reviewer comments and is not final. Every number produced so far came from
`symtrs make-dummy` — a synthetic dataset in the real layout — and is a smoke test, not a
result. Do not report a benchmark number as meaningful unless it came from a real dataset
root, and do not "fix" the code to make a dummy number look better.

---

## 1. What this is

`symtrs` is the reproducibility package for the SyMTRS benchmark: a multi-task synthetic
aerial dataset benchmark with five tasks.

| Task | Registry models | Input → target |
| --- | --- | --- |
| `depth` | 7 (MiDaS ×5, Depth Anything 3 ×2) | `rgb_day{_lr}` → `depth{_lr}` |
| `segmentation` | 6 (U-Net, DeepLabV3 ×2, SegFormer ×3) | `rgb_day{_lr}` → `semantic{_lr}` |
| `domain_adapt` | 2 (CycleGAN, pix2pix) × 4 directions | `rgb_day` ⇄ `rgb_night` (`day2night`, `night2day`) and `rgb_day` ⇄ `rgb_noshadow` (`shadow2noshadow`, `noshadow2shadow`), native only |
| `shadow_removal` | 4 (ShadowFormer, Restormer, Uformer, ShadowMaskFormer), 2048 only | `rgb_day` → `rgb_noshadow` |
| `superresolution` | 4 (SRCNN, autoencoder, SRGAN, SwinIR) × 4 scales | `rgb_day_lr/xN` → native `rgb_day` |

Two of the five are not trained by this package at all and are driven as subprocesses (shadow
removal is a third case, below):

- **domain adaptation** runs the vendored `pytorch-CycleGAN-and-pix2pix/` tree. The pair is
  data (`DIRECTION_ROLES` in `symtrs/model/domain/upstream.py`): `rgb_day` is always A, B is
  `rgb_night` or `rgb_noshadow`, and the direction only picks `AtoB`/`BtoA`. `shadow2noshadow`
  is also scored like a shadow-removal result (the `shadow_removal` staging, mask and metrics,
  after resampling upstream's 256 px output to the ground-truth size), so it sits beside
  ShadowFormer / Restormer / Uformer / ShadowMaskFormer in `symtrs tables`.
- **super-resolution** runs the trainer at `symtrs/legacy/superres/train.py` (launched as
  `python -m symtrs.legacy.superres.train`), which predates the package and was moved
  inside it unchanged in 2026-10-15, then changed on purpose on 2026-10-17 (crops for every
  model, whole-image tiled evaluation, `val/` as the validation set, SRGAN x1, SwinIR
  `img_size`) — `symtrs/legacy/README.md`, "Changes after the move", lists every edit. A
  failed trainer run exits non-zero and leaves `summary_failed.json`, never `summary.json`.

**Shadow removal** takes only the network *class* from four upstream checkouts under
`third_party/` (git submodules, pinned by their gitlinks and unmodified; `git submodule update
--init third_party`, `python scripts/fetch_third_party.py --check` verifies; notes and licence in
`third_party/PINS.md`). `symtrs/model/shadow/` stages `rgb_day`/`rgb_noshadow` pairs plus a *derived* shadow
mask (the dataset has none; `shadow.mask_threshold`/`mask_open`; derived once per pair and settings into the shared,
gitignored `shadow.mask_cache` (`.cache/shadow_masks`, valid while the sources' size/mtime match,
filled by `shadow.mask_workers` processes, never inside the dataset) and only linked into each run,
see `symtrs/model/shadow/mask_cache.py`; `python scripts/precompute_shadow_masks.py --data-root
<root>` builds them upfront), runs one shared worker loop
(`python -m symtrs.model.shadow.worker`, cwd = the upstream repo, so paths must be absolute),
and scores the written PNGs with `symtrs/metrics/shadow.py` (PSNR/SSIM/LAB on all, shadow and
non-shadow pixels). 2048 only; zero-shot is refused (the released weights are ISTD/SIDD, not
SyMTRS); `shadow.init_weights` loads an upstream checkpoint. On a 12 GB card Restormer needs
batch 1 (batch 2 spills). ShadowMaskFormer (the 2.28 M light model; ShadowFormer is 11.4 M) has
**no released weights** and no licence file: it is always trained from scratch, so its rows are
not like-for-like with ShadowFormer's (which starts from ISTD+). It is written for [-1, 1]
inputs and the worker converts at the network boundary only (`Arch.input_range`, applied in
`worker.forward`); do not convert anywhere else.

For both, `symtrs` owns exactly three things: the folder mapping, the launch, and folding the
result into the `summary.json` shape every task writes. Do not reimplement their models in
`symtrs/` — a second copy is a second thing to keep in step with the checkpoints.

The CLI, the YAML loader and the Python SDK all build the same `RunConfig` and reach the
same entry points, so all five tasks are reachable from all three:
`autoloader("srcnn").train(...)` launches that trainer and
`autoloader("cyclegan").train(...)` the vendored upstream, exactly as `symtrs train` does.

## 2. Ground rules

These are the invariants that break silently when they are broken. Read them before
changing anything.

1. **`resolution` names the pixel size of the data, not a scale factor.**
   `2048` → native folders, `1024` → `_lr/x2`, `512` → `_lr/x4`, `256` → `_lr/x8`. In depth
   and segmentation the input *and* the ground truth come from that level, so
   `--resolution 1024` trains and scores at 1024. Nothing upsamples anything.

2. **Super-resolution is the one exception, and it is derived, never configured.** There
   `resolution` names the LR input and `scale = 2048 / resolution` follows from it
   (`configs/superresolution/srcnn_x4.yaml` is `resolution: 512`, factor ×4, target native
   `rgb_day`). `resolution: 2048` there means ×1 — LR and HR are the same folder, a refiner
   control row, explicitly *not* super-resolution. Never add a `scale:` key to a config.

3. **Depth models have a protocol and an output space, and the loss and the alignment follow
   from both.** The single statement is `symtrs.registry.depth_pairing(protocol, output_space)`:

   | protocol | raw output | loss | `eval.align` | models |
   | --- | --- | --- | --- | --- |
   | `metric` | metres | `berhu` | `none` | DA3 metric |
   | `relative` | depth up to a scale | `silog` | `scale` | DA3 relative |
   | `relative` | **disparity** `s/z + t` | `ssi` | `scale_shift_disparity` | MiDaS x5 |

   `scale_shift_disparity` is the MiDaS protocol: scale **and shift** are fitted per image by
   least squares in disparity space against `1 / ground truth`, the fitted disparity is clamped
   to at least `1 / data.max_depth`, and only then inverted to metres. Inverting first and
   aligning with a scale alone (the earlier setup) cannot remove the shift `t`, which dominates
   on narrow-depth nadir views; the resulting MiDaS numbers were meaningless. `ssi` is the
   matching scale-and-shift-invariant loss (the same closed-form fit as the metric, divided by
   the target's spread so it is dimensionless). The registry is the source of truth
   (`spec.protocol`, `spec.output_space`); adapters refuse a mismatched `align` at build time and
   in `evaluate_depth` (`DepthAdapter.validate_alignment`), and the tests in
   `tests/test_vendored_and_grid.py` and `tests/test_depth_conventions.py` assert the pairing at
   every resolution. A relative model scored without alignment looks broken; a metric model
   scored with one looks suspiciously good. Both mistakes have been made here.

   Two more facts that decide whether a metric-model number means anything. *DA3 metric*
   returns canonical depth, `metres = canonical * focal_px / 300`; `focal_px` is derived, not
   typed: `build_from_config` sets it to `(resolution / 2) / tan(data.camera_hfov_deg / 2)`
   (1773.6 px at 2048, halving per level; crops and tiles keep it because they keep the pixel
   size). `data.camera_hfov_deg: 60` is **confirmed**: the dataset owner stated on 2026-10-17 that
   the capture camera's horizontal FOV is always 60 deg for the SyMTRS capture (a car-size
   estimate over 838 cars, 54-63 deg, agrees). The actor property (`FieldOfView = 60.f`) is
   editable per instance and the dataset does not store it, so a future capture with another
   FOV must set `data.camera_hfov_deg`.

   Two families were dropped on 2026-10-22 (adapters, configs and tests removed; `git log` has
   them). ZoeDepth (N/K/NK): its released configs froze the whole MiDaS core (`train_midas:
   false`, forward under `set_grad_enabled(False)`), and even with the core released it did not
   learn usefully on `scripts/overfit_check.py`. Depth Pro: even given the true focal length
   its zero-shot scale was ~16x short on SyMTRS, and only full fine-tuning (952 M parameters,
   ~3 h per 512px epoch) closed the gap.

4. **Five segmentation classes, `ignore_index: 255`.** Background, building, green, road,
   car (`data.num_classes: 5`); the masks store `class_id * data.label_step` (63) as 8-bit
   gray, and an off-grid gray value decodes to the ignore index. All three are recorded in the
   checkpoint. If the mask generator ever writes `0` for "no class", change
   `data.ignore_index` — never make the metric guess which pixels to ignore.

5. **`rgb_night` and `rgb_noshadow` are native-only.** There is no downsampled night or
   shadow-free imagery, which is why
   domain adaptation is a 2048-only task and `symtrs sweep` drops the other resolutions for
   it rather than approximating.

6. **The split lives in the dataset.** `train/`, `val/` and `test/` are part of the layout,
   and `data.split` must differ from the evaluation splits. Never carve a validation set out
   of `train/` in code — upstream's loaders will happily do that if you let them (see §7).

7. **`configs/` is generated output.** 80 of the YAML files are written by
   `configs/generate_grid.py` from one table per task. Change the table and re-run it;
   `python3 configs/generate_grid.py --check` and the test suite both fail if a committed
   file has drifted. `configs/train_sr_x2.yaml` is the one hand-written file (the
   quick-start whose filename is the command).

8. **The vendored upstream tree is committed and pinned.** `pytorch-CycleGAN-and-pix2pix/`
   is a full copy at revision `2a7afba` (2025-08-06), not a submodule and not cloned at run
   time. Exactly two files deviate from upstream — `data/unaligned_dataset.py` and
   `data/aligned_dataset.py` — plus the README. `symtrs setup-adapt --fetch` warns on any
   other drift; `--check` verifies the tree is complete.

9. **Dependencies stay inside torch + torchvision + timm + numpy/PIL/matplotlib/tqdm/yaml.**
   SegFormer (`transformers`), Depth Anything 3 (the `depth_anything_3` package, installed
   with `pip install --no-deps depth-anything-3==0.1.1` because its wheel pins numpy<2) are
   optional extras, imported lazily
   inside the adapter that needs them so `symtrs doctor` can report the model as unavailable
   instead of the whole package failing to import. The one addition this rule has taken:
   the shadow-removal extra (`scipy`, `einops`, `[shadow]` in `pyproject.toml`). Those are
   not `symtrs` requirements — the *vendored* ShadowFormer/Uformer/Restormer modules import
   them at module level, so without them the upstream class cannot even be constructed.
   Install them only in the environment that runs shadow removal. `uv pip install scipy`
   pulls numpy 2 in as a side effect; keep the environment's `numpy==1.26.4` (pin it in the
   same command) or the Depth Anything 3 adapter breaks.

10. **timm cannot serve everything from one environment.** The MiDaS Swin-V2 checkpoint needs
    `timm==0.6.12` (`midas_dpt_swin2_tiny`: timm
    >= 0.9 moved the Swin-V2 patch-merging between stages, so its state_dict loads with a
    missing `downsample.reduction` key), and so does MiDaS DPT-BEiT-Large
    (`midas_dpt_beit_large`: MiDaS patches timm's `Block.forward` with a copy that calls
    `self.drop_path`, which timm >= 0.9 split into `drop_path1`/`drop_path2`, so it builds and
    then fails at the first forward; the adapter refuses at build time and names `.venv-zoe`).
    The other MiDaS models (small, DPT-Hybrid, DPT-Large) work on both: they patch only the
    top-level ViT's forward, never a timm `Block`. SegFormer is no
    longer part of the conflict: timm has never shipped MiT, so its encoder comes from
    `transformers` (`nvidia/mit-b{0,1,2}`). `symtrs doctor` reports which side you are on.
    Plan on two environments (`.venv` and `.venv-zoe` - the name is left from ZoeDepth, its
    first user); `scripts/run_queue.py` takes the environment per job.

11. **Evaluation scores the whole image, with the same path for val and test.** `train.val_patch_size: 0`
    means the full image, never `train.patch_size` (the crop is a memory device; inheriting it
    scored the 2048 rows on one random 512px window). A row that trains on crops smaller than its
    images is evaluated in sliding windows of that crop (`eval.tile_size`, set by
    `eval_tiling` in the generator); relative-depth windows are fitted onto each other before
    blending and the alignment is computed once per stitched image. A NaN/inf validation metric is
    never "best", and non-finite metrics are written as `null`, never as a bare `NaN`. mIoU
    averages the classes with a non-empty union (`miou_classes` says how many; class 0 is absent
    from the current capture).

12. **Co-training is an execution strategy, never a protocol.** `symtrs train-group` (and a
    `group` line in a queue file, §3's `scripts/run_queue.py`) runs several member configs in
    one process on one loader, handing every member the *same* decoded batch: at 2048 one
    sample costs ~0.2 s of CPU to decode and ~0.15 s of GPU per depth model on an A6000, so a
    solo run spends most of its wall clock waiting for images. The sharing unit is the
    **optimiser window**, not the forward pass: the loader is built at the members' largest
    effective batch (`batch_size × grad_accum`, `shared_window`), every member's effective
    batch must divide it, and each member cuts a shared batch into its own windows and
    micro-batches (`Trainee.split`, driven by `train_one_epoch_many(..., windows=True)`). So a
    `batch_size: 4, grad_accum: 2` member and a `batch_size: 8, grad_accum: 1` member share the
    same 8 images per optimiser step while keeping their own memory footprint, and a
    `batch_size: 4, grad_accum: 1` member beside them takes two optimiser steps per shared
    batch (DeepLabV3 beside SegFormer at every resolution). In a group a window closes only
    where `split` says: the solo `(step + 1) % grad_accum` rule counts shared batches there and
    once gave a 2x2 member six steps for four windows. What a member must not lose:
    its own optimiser, schedule (`steps_per_epoch = len(shared loader) × windows per shared
    batch`, never divided by `grad_accum` again — that split is inside the window), checkpoints, `metrics.csv`,
    previews and `summary.json`
    (which records `cogroup`), and the same validation/test path a solo run uses
    (`_finish_epoch` / `_score_and_summarise` — a group must never grow its own eval pass;
    `tests/test_eval_protocol.py` pins this). Members therefore have to agree on everything
    that shapes the window, which is `symtrs.engine.trainer.group_conflicts` (task, resolution,
    data root/splits and the data semantics, patch size, workers, an effective batch that divides the window); a
    disagreement is refused, never resolved in favour of the first member's loader.
    `train.epochs` is per member: one that finishes early leaves the shared pass and the
    others keep going. Only depth and segmentation co-train: domain adaptation, shadow removal
    and super-resolution are subprocesses and read their own data. A group's `--set` applies
    to every member, which is why a per-member resume is a flag (`--resume-existing`) rather
    than a config value.

## 3. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev,timm]'          # core + tests; add exr, da3 as needed
pip install -e '.[adapt]'             # domain adaptation: dominate + wandb
symtrs doctor                         # what is installed, which models can build
symtrs models                         # every registry key, protocol and requirement
```

`adapt` is not optional-extra-in-name-only: the vendored `train.py` and `test.py` both
import `util/visualizer.py`, which imports `dominate` and `wandb` before it parses an
option, so a domain-adaptation run cannot start without them. `symtrs doctor` reports
both; a run started without them stops with the install command rather than a traceback.

CPU works everywhere, slowly. Everything is CUDA-ready: `--device cuda`, `train.gpus` for
DDP in the legacy SR script, `--gpus` for the vendored upstream runs.

No dataset is required to exercise the code end to end:

```bash
symtrs make-dummy --out /tmp/symtrs_dummy --size 64 --train 6 --val 2 --test 2
symtrs doctor --data-root /tmp/symtrs_dummy     # dataset layout and per-cell coverage
symtrs train configs/depth/512/midas_small_512.yaml --data-root /tmp/symtrs_dummy \
    --device cpu --epochs 1 --batch-size 2 --num-workers 0 --out-dir /tmp/run
```

`make-dummy` writes the real layout, so a config that fails against it is a config that will
fail against the real dataset. `--size` must be ≥ the largest patch a config crops.

## 4. Cheat sheet

```bash
# one run, from a config, with a dry run first
symtrs train configs/depth/2048/midas_dpt_hybrid_2048.yaml --data-root /data/SyMTRS --dry-run
symtrs train configs/depth/2048/midas_dpt_hybrid_2048.yaml --data-root /data/SyMTRS

# one run, from flags (no config file)
symtrs train --task segmentation --model segformer_b2 --resolution 1024 \
    --data-root /data/SyMTRS --epochs 40

# zero-shot: released weights, no training
symtrs eval --task depth --model midas_dpt_hybrid --resolution 2048 \
    --data-root /data/SyMTRS --eval-split test

# re-score a finished run (any task, including the two subprocess ones)
symtrs eval configs/superresolution/srcnn_x4.yaml --data-root /data/SyMTRS --eval-split test

# the whole grid: plan first, then execute
symtrs sweep --task depth --data-root /data/SyMTRS --out-dir runs
symtrs sweep --task depth --data-root /data/SyMTRS --out-dir runs --execute --jobs 1

# several compatible models on one shared batch (one loader, one decoded batch per step)
symtrs train-group configs/segmentation/2048/unet_small_2048.yaml \
                  configs/segmentation/2048/segformer_b0_2048.yaml \
                  --data-root /data/SyMTRS --out-dir runs/group1 --device cuda:0 --dry-run
symtrs train-group configs/segmentation/2048/unet_small_2048.yaml \
                  configs/segmentation/2048/segformer_b0_2048.yaml \
                  --data-root /data/SyMTRS --out-dir runs/group1 --device cuda:0

# collect the results table
symtrs tables runs --task depth --format markdown

# prediction on a single image or a folder
symtrs infer --checkpoint runs/depth/midas_small/1024/weights/best.pt \
    --images-dir /data/SyMTRS/test/rgb_day --out-dir /tmp/preds

# environment / dataset / model checks
symtrs doctor --data-root /data/SyMTRS
symtrs doctor --model da3_metric_large --build     # downloads weights
symtrs inspect runs/depth/midas_small/1024/weights/best.pt
```

A bare `*.yaml` as the first argument is read as `train`, which is why
`symtrs train_sr_x2.yaml` works.

## 5. Where things live

```text
symtrs/
  config.py            RunConfig: the single schema every front-end builds
  registry.py          model keys -> spec (task, protocol, requires, default kwargs)
  cli.py, commands/    one module per subcommand; _common.py holds the shared flags
  data/                splits.py (the resolution mapping), factory.py, exr.py, datasets
  engine/              trainer.py (routing), loop.py (train/eval loops, metrics)
  model/
    base.py            the model interface the adapters implement
    wrapper.py         build_from_config, SymTRSModel, autoloader (the SDK surface)
    depth/             one adapter per depth family
    semantic/          segmentation models
    domain/            the CycleGAN/pix2pix subprocess front-end + vendored-tree check
    superres/          the SR subprocess front-end
configs/               80 generated YAMLs + generate_grid.py (see configs/README.md)
symtrs/legacy/          the SR implementation and datasets: moved inside, then edited (see its README)
train_superresolution.py, dataloader/   shims that re-export from symtrs/legacy/
pytorch-CycleGAN-and-pix2pix/                           vendored upstream, pinned
tests/                 pytest; conftest.py holds the layout and data fixtures
```

Run artefacts, uniformly for every task: `<run_dir>/weights/best.pt`, `weights/last.pt`,
`summary.json`, `metrics.csv`, `config.yaml`, `logs/`, `preview/`. `symtrs tables` reads
`summary.json`; a missing `test` key means the test split was never scored, and it is
absent rather than zero on purpose.

## 6. Making changes

**Add a model.** One `_spec` entry in `symtrs/registry.py` (key, task, protocol if it is a
depth model, `requires`, target class) plus the adapter class. Then run
`python3 configs/generate_grid.py` after adding a row to the matching table in the
generator, so the model has a config at every resolution it supports.

**Add a config.** Prefer editing the tables in `configs/generate_grid.py` over writing a
YAML by hand; if it is genuinely a standalone file (a new task, an experiment), put it in
the right subfolder and say in its header why it is not generated.

**Change a default.** `symtrs/config.py` is the only place defaults live. The CLI flags,
the YAML loader and the SDK all build the same `RunConfig`, so changing it in one place is
enough — and changing it anywhere else is a bug.

**Add a task.** `TASKS` in `symtrs/config.py`, a `commands/` entry if it needs one, a
descriptor in `symtrs/model/<task>/` with `describe()`, `task` and `requires`, and the
routing branch in `engine/trainer.py` (`train` and `evaluate_run`) if it is driven as a
subprocess. Add it to §1 of this file and to the table in `configs/README.md`.

## 7. Traps that have already bitten

- **The upstream auto-split.** `UnalignedDataset`/`AlignedDataset` re-split the data by
  `--train_ratio` when the phase folders (`trainA`, `valA`, …) are absent, which silently
  discards the SyMTRS split. That is why the domain front-end stages symlinks into
  `<run_dir>/staging/`. Do not remove that.
- **Upstream's flags move between revisions, and its parser is the interface.** The
  vendored revision has no `--gpu_ids` and `train.py` has no `--results_dir`; both were
  passed for a while and every domain-adaptation run died at argv parsing. Device and
  multi-GPU are now steered through the environment (`device_env`, `upstream.launcher`).
  Anything added to `upstream.train_command` has to exist in
  `pytorch-CycleGAN-and-pix2pix/options/*.py`; `tests/test_domain_commands.py` parses those
  files and fails if it does not. Upstream writes its generated images **flat** in
  `results/<name>/<phase>_<epoch>/images/` as `<image>_fake_B.png` — not one folder per
  label, which is the other way it was read once and scored nothing.
- **The generated image is `fake_B` in every direction, `BtoA` included.** Upstream's
  `set_input` (`models/pix2pix_model.py`, `models/cycle_gan_model.py`) does
  `real_A = input["A" if AtoB else "B"]` and `forward` does `fake_B = G(real_A)`, so
  `--direction BtoA` swaps the *inputs*, not the output label; `fake_A` is CycleGAN's other
  generator on the other input, i.e. the opposite translation. Reading `fake_A` for
  `night2day`/`noshadow2shadow` made pix2pix score nothing (`test: {}`) and CycleGAN score
  day-to-night against real day images. `DomainAdaptation.generated_paths` now selects
  `GENERATED_LABEL` and raises when it finds nothing; `tests/test_domain_commands.py` parses
  the two upstream files and fails if the premise changes.
- **Upstream's `--save_epoch_freq` default is 5, and the score step loads `latest`.** A run
  shorter than five epochs trains and then fails at scoring with a `FileNotFoundError`. The
  front-end passes `--save_epoch_freq 1` for that reason; a caller can thin it out again,
  since extra args are appended last and upstream's parser takes the last value.
- **DeepLabV3 and batch size 1.** torchvision's atrous head has BatchNorm and cannot train
  with a batch of one; the trainer converts the resulting error into a `ConfigError` that
  says so. Keep `batch_size >= 2` for the DeepLabV3 configs.
- **`train.freeze` used to be a no-op.** Adapters stored the policy and nothing applied it;
  `symtrs.engine.trainer.apply_run_freeze` now does, before `.to()`/DDP/the optimiser. It
  recognises the encoder by module-name prefix (`encoder`, `backbone`, `pretrained`), so a
  model that keeps everything under one of those sets `freeze_encoder_prefixes` (MiDaS:
  `backbone.pretrained`). Check the "freeze=... -> N M trainable" line. (It was found because
  Depth Pro, since dropped, ran out of memory with a "frozen" encoder that still trained.)
- **`data.depth_scale`.** Metric depth models predict metres. If the EXR stores millimetres
  as uint16, every metric is wrong by a constant factor and it looks like a broken model.
  `symtrs doctor --data-root` checks the first file it finds.
- **A top-level `utils` is why the trainer was subprocessed, and it is gone.**
  `train_superresolution.py` used to put the repository root on `sys.path` to reach `models/`,
  `utils/` and `dataloader/`; `utils` is a name this package must never inherit, and that is
  why the trainer was launched as a script and never imported. It now lives at
  `symtrs/legacy/superres/train.py` and imports its neighbours by name, so the hazard is gone.
  It is *still* launched as a subprocess — the legacy loop owns its own RNG, matplotlib
  backend and multiprocessing start-up — and `tests/test_legacy_layout.py` fails if any module
  under `symtrs/` writes to `sys.path` again.
- **`symtrs` must not read the `.exr` files as anything other than depth.** `--depth-backend
  numpy` refuses an EXR rather than returning whatever numpy guessed.
- **A missing resolution folder is an error, never a fallback.** A silent fallback to a
  coarser level changes what the reported number means.
- **A relative run directory and a subprocess with another cwd.** Shadow removal and domain
  adaptation launch with the upstream checkout as cwd (and the shadow worker re-reads
  `config.yaml` there), so a relative `--out-dir runs/validation/x` made the worker fail on
  its own `config.yaml` and upstream's `train.py` miss `staging/trainA` (it then falls into its
  auto-split branch; its relative `--checkpoints_dir`/`--results_dir` would also have been
  written into the vendored tree). Found by the 2026-10-17 validation queue. The runners now
  start with `cfg = cfg.anchored()` (`RunConfig.anchored`: `out_dir`, `data.root`,
  `shadow.init_weights`, `train.resume` resolved against the *caller's* cwd, once, on a copy).
  Never build a path for a subprocess from `cfg.resolved_out_dir` on an un-anchored config, and
  never resolve a user-typed path inside a launcher. `tests/test_relative_run_dir.py` covers
  the three subprocess tasks. Paths inside `upstream_args`/`model_kwargs.extra_args` are
  passed through verbatim: give them absolute.

## 8. Testing

```bash
python -m pytest -q                     # the whole suite; no GPU, no network, no dataset
python -m pytest -q -m 'not network'    # explicit, though the markers are off by default
python -m pytest tests/test_vendored_and_grid.py -q
python -m compileall -q symtrs train_superresolution.py
```

The suite must stay green, and it is expected to grow with every change: a new
resolution mapping, a new protocol pairing or a new registry key belongs in a test. Two
conventions matter when adding one:

- **Torch-free tests where possible.** `tests/conftest.py` offers a `layout` fixture that
  creates directories only, so path/split/suffix tests run anywhere — that is the machine
  state you are in when you are checking a freshly generated dataset. Use `requires_torch`
  as a skip marker, never a module-level torch import.
- **Assert the reason, not just the value.** Where behaviour exists to prevent a specific
  mistake (the protocol pairing, the ×1 row not being super-resolution, the zero-shot
  exclusion for SR), assert the message or the guard, so removing the protection fails a
  test rather than quietly changing a number.

## 9. Style

The codebase is written for a reader who will arrive years later to check whether a
published number is real. Match it:

- Comments explain **why**, not what. Every non-obvious guard in this repository exists
  because something went wrong once; the comment says what.
- Error messages name the fix and the command that shows it (`run symtrs doctor
  --data-root <root>`), not just the failure.
- Docstrings on modules say what the module is for and what it deliberately does not do.
- Prefer one implementation over two. Where a task is owned by other code, call it.
- No new runtime dependencies without a strong reason, and never a hard import of an
  optional one.
