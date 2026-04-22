# CycleGAN and pix2pix on SyMTRS

This directory contains a local copy of the CycleGAN/pix2pix codebase adapted for the SyMTRS dataset.

The important change in this workspace is that the dataset loaders support:

- training directly from raw SyMTRS folders
- automatic train/val/test splitting
- optional image tiling for large remote-sensing images
- paired loading by filename matching for pix2pix

Clone the github repo inside SyMTRS root path:

```bash
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git
```

## What Was Modified

Local changes are concentrated in:

- `data/unaligned_dataset.py`
- `data/aligned_dataset.py`

These modifications let you train without manually creating `trainA`, `trainB`, `testA`, `testB`, or concatenated AB pair images, as long as your raw folders follow the expected structure.

## Environment

Use the provided conda environment:

```bash
conda env create -f environment.yml
conda activate pytorch-img2img
```

The environment file currently installs:

- Python 3.11
- PyTorch 2.4
- torchvision 0.19
- numpy
- scikit-image
- Pillow
- dominate
- wandb

If you want HTML visualization only, `wandb` is optional.

## Expected SyMTRS Layout

The modified loaders can work directly on raw folders like this:

```text
/path/to/SyMTRS/
├── hr/
│   ├── RS.0.png
│   ├── RS.100200.png
│   └── ...
└── night/
    ├── RS.0.png
    ├── RS.100200.png
    └── ...
```

For unpaired CycleGAN:

- domain A defaults to `hr/`
- domain B defaults to `night/`

For paired pix2pix:

- filenames must match between `hr/` and `night/`
- pairs are formed by common basename

You can override these defaults with:

- `--raw_A_dir`
- `--raw_B_dir`
- `--raw_A_subdir`
- `--raw_B_subdir`

## Auto-splitting Behavior

If the usual folders such as `trainA`, `trainB`, `testA`, `testB`, or `train/`, `test/` are missing, the custom loaders will auto-split the raw folders.

Relevant arguments:

- `--phase train|val|test`
- `--train_ratio`
- `--val_ratio`
- `--split_seed`

Default behavior:

- `train_ratio=0.9`
- `val_ratio=0.0`
- the remaining samples are used for `test`

Example:

```bash
--phase train --train_ratio 0.8 --val_ratio 0.1 --split_seed 42
```

This produces:

- 80% train
- 10% val
- 10% test

## Tiling Behavior

Large remote-sensing images can be tiled instead of resized.

Relevant arguments:

- `--tile_size`
- `--tile_mode random|center`

When `--tile_size > 0`:

- the loaders crop tiles of size `tile_size x tile_size`
- `resize_and_crop` preprocessing is bypassed for those samples

This is useful when full-resolution SyMTRS images are too large for direct CycleGAN or pix2pix training.

## CycleGAN Reproduction

CycleGAN uses the custom `unaligned_dataset.py`.

### Training

```bash
python train.py \
  --dataroot /path/to/SyMTRS \
  --raw_A_subdir hr \
  --raw_B_subdir night \
  --tile_size 512 \
  --tile_mode random \
  --model cycle_gan \
  --dataset_mode unaligned \
  --name symtrs_cyclegan_t512_e200_s42 \
  --n_epochs 100 \
  --n_epochs_decay 100 \
  --split_seed 42
```

Notes:

- `--dataset_mode unaligned` is the default for CycleGAN, but keeping it explicit is clearer.
- `--phase train` is implicit in `train.py`.
- checkpoints are saved under `checkpoints/symtrs_cyclegan_t512_e200_s42/`

### Validation or test split inference

To run inference on the held-out split produced by the same seed:

```bash
python test.py \
  --dataroot /path/to/SyMTRS \
  --raw_A_subdir hr \
  --raw_B_subdir night \
  --model cycle_gan \
  --dataset_mode unaligned \
  --name symtrs_cyclegan_t512_e200_s42 \
  --phase test \
  --split_seed 42 \
  --num_test 50
```

Results are written to:

```text
results/symtrs_cyclegan_t512_e200_s42/test_latest/
```

## pix2pix Reproduction

pix2pix uses the custom `aligned_dataset.py`.

### Training

```bash
python train.py \
  --dataroot /path/to/SyMTRS \
  --raw_A_subdir hr \
  --raw_B_subdir night \
  --tile_size 512 \
  --tile_mode random \
  --model pix2pix \
  --dataset_mode aligned \
  --direction AtoB \
  --name symtrs_pix2pix_t512_e200_s42 \
  --n_epochs 100 \
  --n_epochs_decay 100 \
  --split_seed 42
```

Notes:

- `AtoB` means `hr -> night` with the default subdirectory mapping above.
- paired samples are built by basename intersection between the two folders.

### Test

```bash
python test.py \
  --dataroot /path/to/SyMTRS \
  --raw_A_subdir hr \
  --raw_B_subdir night \
  --model pix2pix \
  --dataset_mode aligned \
  --direction AtoB \
  --name symtrs_pix2pix_t512_e200_s42 \
  --phase test \
  --split_seed 42 \
  --num_test 50
```

## Using Explicit Absolute Paths

Instead of relying on `--dataroot/hr` and `--dataroot/night`, you can point directly to the domains:

```bash
python train.py \
  --dataroot /tmp/placeholder \
  --raw_A_dir /path/to/SyMTRS/hr \
  --raw_B_dir /path/to/SyMTRS/night \
  --model cycle_gan \
  --dataset_mode unaligned \
  --name symtrs_cyclegan_abs_paths
```

`--dataroot` is still required by the base CLI, even when `--raw_A_dir` and `--raw_B_dir` are used.

## Common Options That Matter

- `--name`: experiment name
- `--checkpoints_dir`: where models are saved
- `--results_dir`: where test outputs are saved
- `--load_size` and `--crop_size`: standard preprocessing sizes when tiling is disabled
- `--preprocess`: preprocessing mode from the upstream codebase
- `--no_flip`: disable random horizontal flips
- `--batch_size`: batch size
- `--num_threads`: dataloader workers
- `--continue_train`: resume training from latest checkpoint

Full CLI help:

```bash
python train.py --help
python test.py --help
```

## Outputs

Training writes to:

```text
checkpoints/<experiment_name>/
```

Typical contents:

- saved model checkpoints
- `train_opt.txt`
- HTML training visualizations unless `--no_html` is used
- loss logs

Testing writes to:

```text
results/<experiment_name>/<phase>_<epoch>/
```

## Notebooks

This directory also includes:

- `CycleGAN.ipynb`
- `pix2pix.ipynb`
- `symtrs_inference.ipynb`

Use them for exploratory runs or visualization, but the reproducible commands are the CLI commands above.

## Known Caveats

- `test.py` forces `batch_size=1`, `num_threads=0`, `serial_batches=True`, and `no_flip=True`.
- For pix2pix, pairing depends on exact matching filenames across the two raw folders.
- For CycleGAN, the two domains are split independently, so it remains unpaired even when filenames overlap.
- If auto-splitting yields no samples, adjust `--train_ratio`, `--val_ratio`, or the folder contents.
- When `--tile_size` is disabled, the upstream resize/crop preprocessing path is used instead.
