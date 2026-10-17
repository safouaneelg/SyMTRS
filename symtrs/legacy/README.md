# `symtrs/legacy/` — code that predates the package, moved inside it (then edited, see below)

Everything under this directory is this repository's own code (commits 08c9931, a2318d5,
61c644a), not a third-party tree. It used to sit beside the package at the repository root,
where top-level `utils` and `dataloader` names made it unreachable from an installed
`symtrs` and forced the super-resolution trainer to be launched as a script rather than
imported. It lives inside the package now: **relocated, not ported**, so that the network
behind a published checkpoint stays checkable against the code that trained it.

The one file whose text changed in the move is `superres/train.py`, and only its module header: a
docstring, the import block, and the two `sys.path.append` lines its script form needed. No
model, loss, metric, dataset or training-loop line changed in the move itself; the later,
deliberate edits are listed under "Changes after the move".

One later, deliberate exception (2026-10-16): `dataloader/utils.py`'s default stem function
now strips the real capture prefix (`rgb_day_512_<key>` -> `<key>`), because the regenerated
dataset names every file `<modality>_<light>_<resolution>_<key>` and exact-name matching
paired nothing. Unprefixed names still match exactly as before; no model, loss or metric
changed.

## Changes after the move (2026-10-17)

The statements above describe the move itself. Four deliberate edits were made afterwards, to
make the trainer usable on the real 2048 px dataset on a 12 GB card; the byte-identity check
below no longer holds for the files listed here (`models/srgan.py`, `train.py`), and the
digests it compares against are the ones from before the move. Nothing else under `legacy/`
changed.

1. **Every model trains on random aligned crops** (`train.py`). `--patch_size` used to apply
   to SwinIR only, and the crop happened after `prepare_batch`'s bicubic upsample, so SRCNN,
   the autoencoder and SRGAN trained on whole 2048 px frames: the first SRCNN activation at
   batch 16 asked for 31.75 GiB. The crop is now one call (`random_crop_pair`, the function
   SwinIR already used) made on the CPU, before the upsample, for every model. The LR patch is
   `--patch_size`, the HR patch `patch_size * scale`, at the matching location.
2. **Validation and test score the whole image, by one path** (`train.py`). `eval_one_epoch`
   used to centre-crop to `--val_patch_size`, or to the TRAINING patch when that was 0, for
   SwinIR (one window per image), and ran batches of whole frames for the other three. It now
   takes one image at a time through `super_resolve`: the whole image in one pass, or, with
   `--eval_tile_size`, overlapping LR windows stitched by `symtrs.engine.tiling.predict_tiled`
   (which gained a `scale` argument for this; windows are laid out in input pixels and come
   back `scale` times larger). PSNR / SSIM / MSE are computed on the stitched full image and
   averaged **per image**; they used to be computed from the batch's mean MSE. SRCNN's valid
   convolutions are compensated by a 6 px reflect pad at evaluation only, so its window comes
   back exactly `scale` times its input. `--val_patch_size > 0` remains a centre crop, for
   speed. The sample panels show the centre 512 px of the whole-image result.
3. **The validation set is the dataset's `val/`** (`train.py`). `--val_lr_root` /
   `--val_hr_root` (new; passed by `symtrs.model.superres`) replace carving 80/20 out of
   `train/`, which AGENTS.md ground rule 6 forbids. `symtrs` refuses to launch a training run
   whose dataset has no `val/`; the old carving is only reachable by calling the script by hand.
4. **SRGAN accepts scale 1** (`models/srgan.py`, one guard). The x1 control row has no
   upsampling blocks, which the constructor already produced for a scale outside {2, 4, 8}
   - it only refused it. `srgan_x1` had never been able to run; its smoke "OOM" was this
   `ValueError`. No layer, loss or parameter changed.

Smaller companion edits in `train.py`: SwinIR is built with `img_size` = the training crop
(`swinir_img_size`) instead of the whole LR frame. Its shifted-window attention masks are
precomputed for `img_size` and stored in the state dict; for a 2048 px frame that is ~10 GB
(it is why `swinir_x1` ran out of memory while constructing the model) and every checkpoint
carried it. Any other input size recomputes the mask, so the architecture, parameters and
outputs are unchanged. Checkpoints written before this edit have differently shaped
`attn_mask` buffers and do not load into a model built after it.

Review fixes, same day (`train.py`):

- Val / test metrics are computed on the output **clamped to [0, 1]**, i.e. on the image the
  sample panels and a user would save; the loop used to score the unbounded network output.
- `best.pt` survives a resume: checkpoints store `best_psnr`, and a resume restores it (older
  checkpoints fall back to their `val_psnr`, the previous behaviour). Before, a resume from
  `last.pt` reset the bar to that epoch's score, so a worse later epoch could replace `best.pt`.
- `random_crop_pair` raises when the patch does not fit the frame instead of silently
  training on the whole frame.
- Known and kept: SRCNN is scored on the full frame including the 6 px border it only ever sees
  through reflect padding (training matches the target to its valid output instead); about 1 %
  of a 2048 px frame. The autoencoder's evaluation window is capped at 1024 target pixels
  (`eval_hr_window` in `configs/generate_grid.py`), because it runs at the target size.

## Where every file came from

| now | before |
| --- | --- |
| `superres/models/{__init__,srcnn,autoencoder_sr,srgan,swinir}.py` | `models/superresolution/` |
| `superres/utils/{__init__,losses,metrics,image_utils,train_utils}.py` | `utils/superresolution/` |
| `superres/train.py` | `train_superresolution.py` (module header only) |
| `dataloader/{__init__,utils,split,superres_dataset,depth_dataset,domain_adapt_dataset}.py` | `dataloader/` |
| `dataloader/README.md` | `dataloader/README.md` |

Two files at the repository root still exist so that documented commands and imports keep
working, and they contain no logic: `train_superresolution.py` and `dataloader/__init__.py`
re-export from here.

**One import stops working**, and it is the only one: `from models.superresolution import
SRCNN` (and the other three nets). `models/` is gone, and the path is now
`symtrs.legacy.superres.models`. Nothing in this repository, in `configs/`, in `tests/` or
in the READMEs ever used it, so the only exposure is a script of the author's kept outside
this repository — written down here so that finding it is a lookup rather than a mystery.
`from dataloader import ...` is the opposite case: it keeps working through the root shim.

`utils/genai/` did **not** move: nothing imports it, and it stays where the author left it
pending a decision about whether it is still wanted. The depth and domain-adaptation
datasets did move, with the rest of `dataloader/`.

`pytorch-CycleGAN-and-pix2pix/` is not part of this tree and never was. It stays at the
repository root, byte-identical to upstream at `2a7afba`, because that byte-identity is the
claim behind every domain-adaptation number.

## Verifying the move

`$BEFORE` below is the commit the move is recorded in, so `HEAD` while the move is still
uncommitted and `HEAD^` once it is:

```bash
BEFORE=HEAD          # or HEAD^ once the move has been committed
```

**1. Byte-identity, from the digests taken before anything moved.** This one works whether
or not the move has been committed, because the "before" side is a file rather than a
commit. The digests are compared as one sorted set, not path by path: every path changed by
definition, and the question is whether any *file* changed.

```bash
grep -v ' train_superresolution.py$' /tmp/symtrs_move/before/sha256.txt \
    | awk '{print $1}' | sort > /tmp/symtrs_move/before_digests.txt
find symtrs/legacy/superres/models symtrs/legacy/superres/utils symtrs/legacy/dataloader \
    -type f \( -name '*.py' -o -name '*.md' \) \
    | xargs shasum -a 256 | awk '{print $1}' | sort > /tmp/symtrs_move/after_digests.txt
diff /tmp/symtrs_move/before_digests.txt /tmp/symtrs_move/after_digests.txt && echo IDENTICAL
```

Expected: 17 lines either side, and `IDENTICAL`.

Comparing per *basename* instead, which is the obvious way to write this, cannot work:
`__init__.py` appears three times (`models/`, `utils/`, `dataloader/`), so a `grep` for it
returns three digests whatever the files contain, and every one of them reads as changed.

**2. The same claim, checked against git.** The 17 files that are not the trainer were
unmodified relative to `HEAD` when they moved — they were tracked and clean — so the blob
each one came from is still in git and can be compared directly:

```bash
git show $BEFORE:models/superresolution/srcnn.py | shasum -a 256   # old
shasum -a 256 symtrs/legacy/superres/models/srcnn.py               # new
```

**3. The rename, with no content change**, shows as `R100` for every file except the
trainer — the one file the move is allowed to touch, and only in its header:

```bash
git diff -M --find-renames=100% --stat $BEFORE
git log --follow --oneline symtrs/legacy/superres/models/srcnn.py
```

The `--follow` line needs the move committed to have anything to follow; until then the
rename is in the working tree and `git status` is the place to look.

**4. The task still produces the number it produced before the move.** A relocation that
changes a metric has changed something it claimed not to:

```bash
diff /tmp/symtrs_move/before/test_metrics.json /tmp/symtrs_move/after/test_metrics.json
```
