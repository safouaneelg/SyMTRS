# SyMTRS

Reproducibility guide for the `SyMTRS` workspace.

This repository currently contains:

- PyTorch dataloaders for the SyMTRS dataset
- a super-resolution training pipeline
- a local `CycleGAN-and-pix2pix/` subtree for image-to-image translation

## Repository Layout

```text
SyMTRS/
├── README.md
├── train_superresolution.py
├── dataloader/
│   ├── depth_dataset.py
│   ├── domain_adapt_dataset.py
│   ├── superres_dataset.py
│   └── split.py
├── models/superresolution/
│   ├── srcnn.py
│   ├── autoencoder_sr.py
│   ├── srgan.py
│   └── swinir.py
├── utils/superresolution/
└── CycleGAN-and-pix2pix/
```

## Supported Tasks

### 1. Super-resolution

Entry point:

```bash
python train_superresolution.py ...
```

Implemented models:

- `srcnn`
- `autoencoder`
- `srgan`
- `swinir`

### 2. Image-to-image translation

Implemented inside [CycleGAN-and-pix2pix/README.md](CycleGAN-and-pix2pix/README.md):

- CycleGAN on unpaired day/night imagery
- pix2pix on paired day/night imagery

### 3. Dataset loading utilities

Available dataloaders:

- `SuperResDataset`
- `DomainAdaptDataset`
- `DepthDataset`

## Environment Setup

The root project does not ship an environment file, so install the dependencies required by the present code:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision numpy pillow matplotlib tqdm timm
```

Notes:

- `timm` is required by `models/superresolution/swinir.py`.
- CUDA is optional. The training script falls back to CPU if CUDA is unavailable.
- For image-to-image translation, use the separate environment described in [CycleGAN-and-pix2pix/README.md](CycleGAN-and-pix2pix/README.md).

## Expected Dataset Layout

The code assumes flat directories whose filenames match by stem.

```text
/path/to/SyMTRS/
├── hr/
│   ├── RS.0.png
│   ├── RS.100200.png
│   └── ...
├── lr/
│   ├── x2/
│   │   ├── RS.0.png
│   │   ├── RS.100200.png
│   │   └── ...
│   ├── x4/
│   └── x8/
├── night/
│   ├── RS.0.png
│   ├── RS.100200.png
│   └── ...
└── depth/
    ├── RS.depth.0.npy
    ├── RS.depth.100200.npy
    └── ...
```

Matching rules:

- `SuperResDataset` matches LR and HR files by identical filename stem.
- `DomainAdaptDataset` matches day and night files by identical filename stem when `paired=True`.
- `DepthDataset` expects depth files like `RS.depth.100200.npy` to match image `RS.100200.png`.

## Super-resolution Reproducibility

### Basic command

```bash
python train_superresolution.py \
  --lr_root /path/to/SyMTRS/lr/x4 \
  --hr_root /path/to/SyMTRS/hr \
  --model srcnn \
  --scale 4 \
  --out_dir runs/srcnn_x4
```

### Common examples

SRCNN:

```bash
python train_superresolution.py --lr_root /path/to/SyMTRS/lr/x2 --hr_root /path/to/SyMTRS/hr --model srcnn --scale 2 --out_dir runs/srcnn_x2
python train_superresolution.py --lr_root /path/to/SyMTRS/lr/x4 --hr_root /path/to/SyMTRS/hr --model srcnn --scale 4 --out_dir runs/srcnn_x4
python train_superresolution.py --lr_root /path/to/SyMTRS/lr/x8 --hr_root /path/to/SyMTRS/hr --model srcnn --scale 8 --out_dir runs/srcnn_x8
```

Autoencoder:

```bash
python train_superresolution.py --lr_root /path/to/SyMTRS/lr/x2 --hr_root /path/to/SyMTRS/hr --model autoencoder --scale 2 --out_dir runs/autoencoder_x2
python train_superresolution.py --lr_root /path/to/SyMTRS/lr/x4 --hr_root /path/to/SyMTRS/hr --model autoencoder --scale 4 --out_dir runs/autoencoder_x4
python train_superresolution.py --lr_root /path/to/SyMTRS/lr/x8 --hr_root /path/to/SyMTRS/hr --model autoencoder --scale 8 --out_dir runs/autoencoder_x8
```

SRGAN:

```bash
python train_superresolution.py --lr_root /path/to/SyMTRS/lr/x2 --hr_root /path/to/SyMTRS/hr --model srgan --scale 2 --out_dir runs/srgan_x2
python train_superresolution.py --lr_root /path/to/SyMTRS/lr/x4 --hr_root /path/to/SyMTRS/hr --model srgan --scale 4 --out_dir runs/srgan_x4
python train_superresolution.py --lr_root /path/to/SyMTRS/lr/x8 --hr_root /path/to/SyMTRS/hr --model srgan --scale 8 --out_dir runs/srgan_x8
```

SwinIR:

```bash
python train_superresolution.py --lr_root /path/to/SyMTRS/lr/x2 --hr_root /path/to/SyMTRS/hr --model swinir --scale 2 --out_dir runs/swinir_x2
python train_superresolution.py --lr_root /path/to/SyMTRS/lr/x4 --hr_root /path/to/SyMTRS/hr --model swinir --scale 4 --out_dir runs/swinir_x4
python train_superresolution.py --lr_root /path/to/SyMTRS/lr/x8 --hr_root /path/to/SyMTRS/hr --model swinir --scale 8 --out_dir runs/swinir_x8
```

Background execution:

```bash
mkdir -p runs/srcnn_x4
nohup python train_superresolution.py \
  --lr_root /path/to/SyMTRS/lr/x4 \
  --hr_root /path/to/SyMTRS/hr \
  --model srcnn \
  --scale 4 \
  --out_dir runs/srcnn_x4 \
  > runs/srcnn_x4/train.log 2>&1 &
```

### Important CLI arguments

- `--lr_root`: low-resolution image directory
- `--hr_root`: high-resolution image directory
- `--model`: `srcnn`, `autoencoder`, `srgan`, or `swinir`
- `--scale`: upscale factor, mainly relevant for `srgan` and `swinir`
- `--batch_size`: default `4`
- `--epochs`: default `20`
- `--lr`: default `1e-4`
- `--split`: dataset split ratios, default `0.8 0.2`
- `--seed`: random seed, default `42`
- `--num_workers`: dataloader workers, default `4`
- `--use_y_channel`: train on luminance only
- `--weights`: resume from a checkpoint
- `--gpus`: number of GPUs for DDP training
- `--patch_size`: LR patch size for SwinIR training
- `--val_patch_size`: validation crop size for SwinIR
- `--amp`: mixed precision for SwinIR

Full argument list:

```bash
python train_superresolution.py --help
```

### Outputs

Each run writes to `--out_dir`:

- `metrics.csv`
- `loss_curve.png`
- `psnr_curve.png`
- `mse_curve.png`
- `ssim_curve.png`
- `weights/best.pt`
- `weights/last.pt`
- `samples/epoch_XXX.png`

### Resume training

```bash
python train_superresolution.py \
  --lr_root /path/to/SyMTRS/lr/x4 \
  --hr_root /path/to/SyMTRS/hr \
  --model swinir \
  --scale 4 \
  --weights runs/swinir_x4/weights/last.pt \
  --out_dir runs/swinir_x4_resume
```

## Dataloader Examples

### Super-resolution

```python
from dataloader import SuperResDataset

dataset = SuperResDataset(
    lr_root="/path/to/SyMTRS/lr/x4",
    hr_root="/path/to/SyMTRS/hr",
)
sample = dataset[0]
print(sample["lr"].shape, sample["hr"].shape)
```

### Domain adaptation

```python
from dataloader import DomainAdaptDataset

dataset = DomainAdaptDataset(
    day_root="/path/to/SyMTRS/hr",
    night_root="/path/to/SyMTRS/night",
    paired=False,
)
sample = dataset[0]
print(sample["day"].shape, sample["night"].shape)
```

### Depth estimation loader

```python
from dataloader import DepthDataset

dataset = DepthDataset(
    depth_root="/path/to/SyMTRS/depth",
    hr_root="/path/to/SyMTRS/hr",
    normalize_depth="raw",
)
sample = dataset[0]
print(sample["image"].shape, sample["depth"].shape)
```

## Notes and Caveats

- The current training script performs an internal train/validation split from the provided folders; separate split files are not required.
- `SRCNN` and `autoencoder` upsample LR inputs with bicubic interpolation before the model forward pass.
- `SRCNN` can be run in Y-channel mode with `--use_y_channel`.
- `AutoencoderSR` assumes spatial sizes divisible by 4.
- `SwinIR` depends on `timm` and supports patch-based training through `--patch_size`.
- Sample saving and metrics are driven by the validation loader.

## Image-to-image Translation

Use the separate guide in [CycleGAN-and-pix2pix/README.md](CycleGAN-and-pix2pix/README.md) for:

- CycleGAN day-to-night translation
- pix2pix paired translation
- the custom auto-splitting and tiling logic added in this workspace
