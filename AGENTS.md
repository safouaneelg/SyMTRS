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
aerial dataset benchmark with four tasks.

| Task | Registry models | Input → target |
| --- | --- | --- |
| `depth` | 11 (MiDaS ×5, ZoeDepth ×3, Depth Anything 3 ×2, Depth Pro) | `rgb_day{_lr}` → `depth{_lr}` |
| `segmentation` | 6 (U-Net, DeepLabV3 ×2, SegFormer ×3) | `rgb_day{_lr}` → `semantic{_lr}` |
| `domain_adapt` | 2 (CycleGAN, pix2pix) × 2 directions | `rgb_day` ⇄ `rgb_night`, native only |
| `superresolution` | 4 (SRCNN, autoencoder, SRGAN, SwinIR) × 4 scales | `rgb_day_lr/xN` → native `rgb_day` |

Two of the four are not trained by this package at all and are driven as subprocesses:

- **domain adaptation** runs the vendored `pytorch-CycleGAN-and-pix2pix/` tree.
- **super-resolution** runs the trainer at `symtrs/legacy/superres/train.py` (launched as
  `python -m symtrs.legacy.superres.train`), which predates the package and was moved
  inside it unchanged in 2026-10-15 — see `symtrs/legacy/README.md`.

For both, `symtrs` owns exactly three things: the folder mapping, the launch, and folding the
result into the `summary.json` shape every task writes. Do not reimplement their models in
`symtrs/` — a second copy is a second thing to keep in step with the checkpoints.

The CLI, the YAML loader and the Python SDK all build the same `RunConfig` and reach the
same entry points, so all four tasks are reachable from all three:
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

3. **Depth models have a protocol, and the loss and the alignment follow from it.**
   `relative` → `loss: silog` + `eval.align: scale`; `metric` → `loss: berhu` +
   `eval.align: none`. The registry is the source of truth (`spec.protocol`). A relative
   model scored without alignment looks broken; a metric model scored with it looks
   suspiciously good. Both mistakes have been made here; the tests in
   `tests/test_vendored_and_grid.py` now assert the pairing holds at every resolution.

4. **Six segmentation classes, `ignore_index: 255`.** Both are recorded in the checkpoint.
   If the mask generator ever writes `0` for "no class", change `data.ignore_index` — never
   make the metric guess which pixels to ignore.

5. **`rgb_night` is native-only.** There is no downsampled night imagery, which is why
   domain adaptation is a 2048-only task and `symtrs sweep` drops the other resolutions for
   it rather than approximating.

6. **The split lives in the dataset.** `train/`, `val/` and `test/` are part of the layout,
   and `data.split` must differ from the evaluation splits. Never carve a validation set out
   of `train/` in code — upstream's loaders will happily do that if you let them (see §7).

7. **`configs/` is generated output.** 88 of the YAML files are written by
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
   Depth Anything 3 (`transformers`) and Depth Pro are optional extras, imported lazily
   inside the adapter that needs them so `symtrs doctor` can report the model as unavailable
   instead of the whole package failing to import.

10. **timm cannot serve everything from one environment.** ZoeDepth's released weights need
    `timm==0.6.12`; SegFormer needs a timm with `mit_b0`, which 0.6.12 does not have. MiDaS
    works on both. `symtrs doctor` reports which side you are on. Plan on two environments.

## 3. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev,timm]'          # core + tests; add exr, da3, depthpro as needed
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
configs/               88 generated YAMLs + generate_grid.py (see configs/README.md)
symtrs/legacy/          the SR implementation and datasets: moved inside, byte-identical
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
- **Upstream's `--save_epoch_freq` default is 5, and the score step loads `latest`.** A run
  shorter than five epochs trains and then fails at scoring with a `FileNotFoundError`. The
  front-end passes `--save_epoch_freq 1` for that reason; a caller can thin it out again,
  since extra args are appended last and upstream's parser takes the last value.
- **DeepLabV3 and batch size 1.** torchvision's atrous head has BatchNorm and cannot train
  with a batch of one; the trainer converts the resulting error into a `ConfigError` that
  says so. Keep `batch_size >= 2` for the DeepLabV3 configs.
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
