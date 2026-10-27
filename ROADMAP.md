# ROADMAP

Work that is planned but not done, and the decision behind each item, so that the next
person can tell a deliberate omission from an oversight. Dates are when the decision was
taken. Nothing here is committed to the paper until it is measured.

Last updated: 2026-10-27.

---

## 1. Third-party code: git submodules

**Done 2026-10-27.** The four shadow-removal upstreams (ShadowFormer, Restormer, Uformer,
ShadowMaskFormer) are git submodules under `third_party/`, registered from the existing
checkouts at the commits they were already pinned to, so no file on disk changed. `.gitmodules`
holds the URLs (with `ignore = untracked`, for the `__pycache__` Python writes inside them),
git holds the commits, and `third_party/PINS.md` keeps the notes (dates, licence, weights) but
no hashes, so there is one source for the pin. `git submodule update --init third_party` checks
them out; `python scripts/fetch_third_party.py` wraps that and `--check` prints pass/fail. A
plain `git clone` does not fetch them, so the package still installs and tests offline.

Only the network class is taken from each (`symtrs/model/shadow/archs.py`); none of their
training scripts is run and none is modified. A submodule cannot carry local edits: if one is
ever needed, fork the repository and point the submodule at the fork (licence permitting -
ShadowMaskFormer has none, so it could not be forked publicly), or vendor it.

Adding a new third party that we only *call*: `git submodule add <url> third_party/<Repo>`,
check out the commit, add a row of notes to `PINS.md`.

The one exception is the vendored `pytorch-CycleGAN-and-pix2pix/`: a full committed tree at
`2a7afba` with exactly two files deviating (its dataset loaders are the interface we drive,
so those two edits have to survive a clean clone). That is the pattern for upstream *code we
edit*, not for upstream *code we only call*.

The rule for any new third party stays AGENTS.md §1: symtrs owns the folder mapping, the
launch and the `summary.json` fold; upstream owns the network and its training loop. Never a
second copy of the model.

## 2. A diffusion-based domain-translation baseline (planned)

**Reason.** The domain-adaptation section currently compares two GANs (CycleGAN and pix2pix)
on four directions, and a reviewer will ask whether that is a statement about the dataset or
about the model family. One diffusion row on `day2night` answers the question; it is not an
attempt to beat CycleGAN's KID, and it must not be reported as if a full sweep had been run.

**Candidate upstreams** (both published, both image-to-image with a 256 px paired-folder
loader, which is the shape our staging already produces):

| Repo | Venue | Why it fits |
| --- | --- | --- |
| BBDM | CVPR 2023 | Brownian-bridge i2i; ships day/night style configs |
| DDBM | ICLR 2024 | Denoising diffusion bridge; the same A/B folder layout |

Not yet pinned: licence, configs and a torch-2-compatible checkpoint load have to be checked
first. That check is read-only and comes before any GPU time.

**Steps.** Each is the same shape as the shadow removal front-end, which already does this
for four repos:

1. `git submodule add` it under `third_party/` at the chosen commit, plus a row of notes in
   `third_party/PINS.md`; a fork only if it needs a source change.
2. `symtrs/model/domain/` (or a sibling `diffusion/`) front-end: stage `trainA`/`trainB`
   symlinks from the split into `<run_dir>/staging/`, build the command with absolute paths,
   run it with the checkout as cwd (a relative `--out-dir` from another cwd has bitten three
   tasks already — `RunConfig.anchored()`).
3. Find the generated images the way `DomainAdaptation.generated_paths` does — by the label
   upstream actually writes, refusing rather than returning an empty list. Do not assume a
   folder per class, and do not assume the label matches the direction.
4. Score with the existing KID/FID path at 256 px, so the row is comparable with the two GANs.
5. Route it in `engine/trainer.py` (`train` and `evaluate_run`) and add the registry key.

**Cost.** Roughly two GPU-days per direction at 256 px for one model, which has to wait for
the current campaign: the box is about 1.5 days from finishing its remaining jobs.

## 3. Checkpoint hygiene for the subprocess tasks (planned)

The two GAN front-ends pass `--save_epoch_freq 1` because upstream's default of 5 makes any
run shorter than five epochs fail at the scoring step. The cost is that every epoch is kept:

| Run | Checkpoints |
| --- | --- |
| `pix2pix_day2night` | 8.8 GB (40 × 227 MB `G` + 11 MB `D`) |
| `pix2pix_night2day` | 8.8 GB |
| `pix2pix_shadow2noshadow` | 8.2 GB |
| `cyclegan_{day2night,night2day}` | 4.4 GB each |

About 36 GB now, and two more runs are queued behind it. Upstream scores `--epoch latest`,
so after a run has been scored the numbered checkpoints are needed only to re-test a
different epoch by hand.

Planned: a `--prune` step after a run's `summary.json` carries a `test` key, which keeps
`latest_net_{G,D}.pth` and `web/` and drops every numbered file (the last numbered epoch holds
the same tensors as `latest`; the files differ byte-wise only because `torch.save` records the
archive name, checked on 2026-10-27); and `--save_epoch_freq`
thinned in `model_kwargs.extra_args` for the runs still queued (it is appended last, and
upstream's parser takes the last value). The previews in `web/images` are 13 MB per run and
stay — they are the only per-epoch record and the source of the qualitative figures.

## 4. Smaller, decided, not yet done

- `SymTRSModel.predict` on a disparity model (MiDaS) returns `1 / raw`, which is not metres
  and is not what the evaluation scores. Evaluation is unaffected. Either normalise per image
  and document the output as relative, or refuse with a message naming `symtrs eval`.
- The two fixes already on disk are uncommitted: the depth squeeze in
  `symtrs/model/wrapper.py` and the Pillow-13 mode in `symtrs/commands/infer.py`.
