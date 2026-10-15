# CycleGAN and pix2pix, vendored

This is the **complete** source tree of
[`junyanz/pytorch-CycleGAN-and-pix2pix`](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix),
committed into the SyMTRS repository so that domain adaptation needs no clone, no
submodule and no network access at run time.

| | |
|---|---|
| Upstream | `https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git` |
| Revision | `2a7afba2895d52556dd5dfe07e8555ef657ced6f` (2025-08-06, `master`) |
| Licence | BSD-2-Clause, see [`LICENSE`](LICENSE) |
| Excluded | `imgs/` — the README's teaser gallery (7.4 MB, almost all one animated GIF) |

Upstream's own README is kept as [`README.upstream.md`](README.upstream.md) and is the
reference for `train.py` / `test.py` flags; this file describes only what SyMTRS adds.

Check the tree at any time with:

```bash
symtrs setup-adapt            # present and complete? prints the pinned revision
symtrs setup-adapt --fetch    # clone the pinned revision and print the exact delta
```

`--fetch` never writes to this directory. It reports which files differ from upstream,
and warns if any differing file is not in `MODIFIED_FILES` below.

## What SyMTRS modified

Two files, both dataset loaders. Their upstream originals are kept beside them in
[`upstream_originals/`](upstream_originals/) so the delta is readable offline:

- `data/unaligned_dataset.py` (CycleGAN)
- `data/aligned_dataset.py` (pix2pix)

Both gained the same three things:

- **Auto-split fallback.** If `dataroot/<phase>A` and `<phase>B` are absent, the loader
  splits `--raw_A_dir` / `--raw_B_dir` by `--train_ratio`, `--val_ratio` and
  `--split_seed` instead of failing.
- **Tiling.** `--tile_size > 0` crops `tile_size x tile_size` tiles in place of the
  upstream `resize_and_crop` preprocessing, which matters for 2048px aerial imagery.
  `--tile_mode random|center` selects the tile.
- **Paired loading for pix2pix.** Aligned pairs are formed by basename intersection of
  the two raw folders, so the AB-concatenated image format is not required.

Everything else in this tree is upstream, unmodified.

> **The auto-split fallback is a trap when SyMTRS drives this code.** It fires whenever
> `trainA`/`trainB` do not exist, and it would then re-split the data itself, ignoring
> the SyMTRS `train`/`val`/`test` split. `symtrs` avoids it by materialising those six
> directories as symlinks before launching anything — see
> `symtrs/model/domain/upstream.py::build_staging_dir`. Running `train.py` here by hand
> against a raw folder does hit it, which is what the upstream README's
> `--train_ratio`/`--split_seed` examples are for.

## Running it

Prefer the `symtrs` front-end, which stages the split as symlinks, pins the checkpoints
and results directories into the run directory, translates the direction into
`--direction AtoB|BtoA`, and scores the translations with FID/KID afterwards:

```bash
symtrs train configs/domain_adapt/cyclegan_day2night.yaml --data-root /data/SyMTRS
```

`train.py` and `test.py` are also runnable directly, as upstream intends:

```bash
python train.py --dataroot <staging> --model cycle_gan --dataset_mode unaligned \
                --direction AtoB --name symtrs_cyclegan_day2night --n_epochs 100
```

## Environment

Upstream ships `environment.yml` (Python 3.11, PyTorch 2.4, torchvision 0.19, numpy,
scikit-image, Pillow, dominate, wandb). Installing it is not needed for a `symtrs` run —
the domain-adaptation task uses whatever `torch` the `symtrs` environment already has —
but **`dominate` and `wandb` are**, because `train.py` and `test.py` both import
`util/visualizer.py`, which imports them at module level before it parses a single
option. They are not optional here in practice, whatever the flags say:

```bash
pip install -e '.[adapt]'       # dominate + wandb, the two upstream start-up imports
```

`symtrs doctor` lists them, and a run started without them stops with the install
command rather than upstream's traceback.

## Known caveats

Inherited from upstream, and worth knowing before reading a result:

- `test.py` forces `batch_size=1`, `num_threads=0`, `serial_batches=True` and
  `no_flip=True`.
- For pix2pix, pairing depends on exactly matching filenames across the two raw folders.
- For CycleGAN, the two domains are split independently, so it stays unpaired even when
  filenames overlap.
- Checkpoints land in `<checkpoints_dir>/<name>/`; SyMTRS points `checkpoints_dir` at the
  run directory so a run does not scatter weights into the working directory.
