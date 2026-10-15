# SyMTRS multi-task SDK and benchmark design

Date: 2026-10-15
Status: agreed with author, implementation in progress

**Superseded in one respect:** the super-resolution layout. `models/`, `utils/` and
`dataloader/` no longer sit at the repository root; they were moved inside the package,
byte-identical, by
`docs/superpowers/specs/2026-10-15-symtrs-superres-relocation-design.md`. The rest of this
document is unchanged history.

## Purpose

The revised SyMTRS dataset adds semantic segmentation labels and reorganises the
tree into explicit `train/val/test` splits with a low-resolution pyramid. This
document specifies the code added to the repository to (a) train and evaluate
monocular depth models and semantic segmentation models on that dataset at
multiple resolutions, (b) drive the existing day/night translation pipeline from
the same interface, and (c) expose all of it through a `symtrs` command line and a
Python SDK.

Primary outcome: benchmark baselines for the revised paper. Secondary: the same
code is the public reproducibility artifact, so it is structured for release
rather than as throwaway scripts.

## Dataset contract

```text
<root>/
  train/ | val/ | test/
    rgb_day/            # 2048x2048 PNG
    rgb_night/          # 2048x2048 PNG
    rgb_day_lr/x2|x4|x8 # 1024 | 512 | 256 PNG
    depth/              # 2048x2048 EXR, single channel, metres
    depth_lr/x2|x4|x8
    semantic/           # 2048x2048 PNG, grayscale labels
    semantic_lr/x2|x4|x8
```

Resolution is an explicit axis of every task run, named `2048`, `1024`, `512`,
`256` and mapped to directories by `symtrs.data.splits.resolve_split_dirs`:

| resolution | day | night | depth | semantic |
|---|---|---|---|---|
| 2048 | `rgb_day` | `rgb_night` | `depth` | `semantic` |
| 1024 | `rgb_day_lr/x2` | - | `depth_lr/x2` | `semantic_lr/x2` |
| 512 | `rgb_day_lr/x4` | - | `depth_lr/x4` | `semantic_lr/x4` |
| 256 | `rgb_day_lr/x8` | - | `depth_lr/x8` | `semantic_lr/x8` |

Input and ground truth are always taken at the *same* resolution. These runs are
not super-resolution runs; the low-resolution pyramid is a degradation axis for
the depth and segmentation benchmarks.

`night` exists only at 2048, which is why domain adaptation is 2048-only.

Semantic ground truth has **6 classes**.

## Architecture

Three layers, so that the CLI, the YAML files and the Python SDK cannot drift
apart:

1. **Config schema** (`symtrs.config`). `RunConfig` is the single source of
   truth, built from a YAML file, from CLI flags, or from Python keyword
   arguments. CLI flags override a YAML base. The fully resolved config is
   written into every run directory and into every checkpoint.

2. **Tasks** (`symtrs.data`, `symtrs.model`, `symtrs.metrics`, `symtrs.engine`).
   Each task supplies a dataset, a model adapter, a loss and a metric set. Depth
   and segmentation are implemented natively; domain adaptation wraps the
   existing vendored CycleGAN/pix2pix code; super-resolution is a thin
   pass-through to the pre-existing `train_superresolution.py`.

3. **Front-ends** (`symtrs.cli`, `symtrs.commands`, `symtrs.model.autoloader`).
   Thin: they only build a config and call into layer 2.

`symtrs/` is a new namespace. The pre-existing top-level `dataloader/`,
`models/`, `utils/` packages and `train_superresolution.py` are left untouched.

## Depth protocol

Depth ground truth is metric, in metres. The four model families disagree about
what they predict, so each adapter declares a protocol and the pipeline picks the
matching target transform and metric set:

- `protocol: "metric"` (ZoeDepth `ZoeD_K`, Depth Anything 3 metric variants,
  Depth Pro): ground truth used directly in metres, loss defaults to `berhu`,
  metrics reported unaligned and additionally with per-image alignment as a
  secondary column.
- `protocol: "relative"` (MiDaS/DPT, ZoeDepth `ZoeD_N`, relative Depth Anything 3
  variants): ground truth converted to the model's native prediction space
  (disparity for MiDaS-style models), normalised per image, loss defaults to
  `silog`, metrics reported after per-image alignment following that model's own
  published convention so the numbers stay comparable to its paper.

Models, all lazy-imported so that a missing optional dependency disables one
model rather than the package:

| key | source | protocol |
|---|---|---|
| `midas_small` | torch.hub `intel-isl/MiDaS` | relative |
| `midas_dpt_hybrid` | torch.hub `intel-isl/MiDaS` | relative |
| `zoedepth_n` | torch.hub `isl-org/ZoeDepth` | relative |
| `zoedepth_k` | torch.hub `isl-org/ZoeDepth` | metric |
| `zoedepth_nk` | torch.hub `isl-org/ZoeDepth` | metric (+ domain id) |
| `da3_*` | HuggingFace / Depth Anything 3 | metric and relative |
| `depth_pro` | torch.hub `apple/DepthPro` | metric |

Depth metrics, all computed on the valid mask: RMSE, MAE, abs_rel, sq_rel,
delta1/2/3, SILog.

Invalid depth pixels are non-finite or `<= 0`. They never contribute to the loss
and never contribute to the metrics.

## Segmentation protocol

| key | construction |
|---|---|
| `deeplabv3_r50`, `deeplabv3_r101` | `torchvision` segmentation, head swapped to `num_classes` |
| `segformer_b0`, `segformer_b2` | timm `mit_b0`/`mit_b2` encoder + in-repo SegFormer decode head |
| `unet_small` | in-repo, cheap lower bound |

Dependencies stay at torch, torchvision, timm. No `transformers`, no
`segmentation-models-pytorch`.

Loss: cross-entropy with optional class weights, plus the DeepLab auxiliary loss
where the architecture provides one, plus optional Dice. `ignore_index`
configurable, default 255.

Metrics: mIoU (primary), per-class IoU, pixel accuracy, mean accuracy, Dice/F1,
confusion matrix.

Full-resolution inference on 2048 uses sliding-window tiling with overlap.

## Run matrix

Full grid, both modes:

- depth: each depth model x {2048, 1024, 512, 256} x {zero-shot, finetuned}
- segmentation: each segmentation model x {2048, 1024, 512, 256}
- domain adaptation: CycleGAN and pix2pix, both directions, at 2048

`zero-shot` evaluates released pretrained weights with no training on SyMTRS.
`finetuned` trains on `train/` and evaluates on `val/`/`test/`. Both share one
evaluation code path; `autoloader` accepts either a checkpoint path or a registry
key, and a registry key means "pretrained".

`symtrs sweep` generates one YAML per grid cell into `configs/generated/`, then
optionally runs them, so the grid doubles as a directory of reproducible configs.
`symtrs tables` collates finished runs into markdown and LaTeX.

## Hardware assumptions

CUDA-first but size-agnostic. Defaults target a single 24GB card: random-crop
patch training, AMP on, gradient checkpointing available, DDP across GPUs
available via `--gpus N`. Nothing assumes a specific GPU count.

## Domain adaptation integration

The vendored `pytorch-CycleGAN-and-pix2pix/` subtree contains only the three
modified `data/*.py` modules, the notebook and sample crops; upstream
`train.py`/`test.py`/`models/`/`options/` are absent. `symtrs setup-adapt` clones
upstream and overlays the modified modules.

Note a trap in the modified loaders: when `dataroot/<phase>A` and
`dataroot/<phase>B` are absent they fall back to raw directories *and always
apply their own ratio-based split*, even when explicit `--raw_A_dir`/`--raw_B_dir`
are given. Pointing them at the new pre-split folders would therefore silently
re-split them. The wrapper instead materialises a staging directory of symlinks
(`trainA`, `trainB`, `valA`, `valB`, `testA`, `testB`) pointing at
`<root>/{split}/rgb_{day,night}`, which takes the untouched `trainA`/`testA` code
path and uses the pre-split data exactly. Day/night direction is chosen by which
domain is staged as A.

## Testing

No GPU or dataset is required for the test suite. `symtrs make-dummy` writes a
small synthetic dataset in the exact layout above, so the whole pipeline can be
exercised before the real data exists.

Unit tests cover the resolution mapping, the depth reader and invalid masking,
stem matching, class auto-detection, config round-trips, checkpoint metadata
round-trips, sweep generation, table rendering, and metric correctness against
hand-computed values. Model-downloading tests are marked so they can be deselected
offline.

## Open items to verify on first real run

- Exact alignment convention per relative-depth model, recorded in the registry
  and checked against each model's published numbers.
- Depth Pro finetuning mode; default is frozen encoder plus head and last blocks.
- EXR channel count, units, and row order; `symtrs inspect --save-previews` prints
  statistics and writes RGB/depth side-by-side previews so a flipped or
  mis-scaled depth map is visible immediately.
