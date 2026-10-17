# Questions and answers (2026-10-17)

Questions the dataset owner asked during the work, with the answers given at the time. The
numbers come from the preview dataset (144 views, 96/24/24) and the short validation runs;
none of them is a benchmark result.

## Smoke test vs validation runs: "full run on what?"

There is no other data: both queues use the same 144 views. The smoke test (finished
06:47, re-run for super-resolution and domain adaptation until 08:42) trains for a few
batches and scores a couple of frames, to prove each model builds, steps and writes a
`summary.json`. The validation queue (started 08:43, `runs/validation/`) trains 2 epochs
over all 96 train views and scores all 24 val and 24 test views at full size. It is still a
short check, not a full training run.

It caught what the smoke test could not: out-of-memory failures that only appear when
2048 px images are scored whole, metric-depth models that do not move under their settings,
and relative paths that broke the subprocess tasks (see `BLOCKERS.md`).

The stated goal of the work: check that the repository provides appropriate models and that
the data is in an appropriate format. Training quality is secondary.

## Is the camera FOV really 60°?

Nothing in the dataset or the logs records it. The capture actor's default is
`FieldOfView = 60.f` (horizontal; `SyMTRSDatasetCaptureActor.h:97`), and it is `EditAnywhere`,
so a placed actor can override it. The authoritative check is in the editor: select the
capture actor in the level, Details, Dataset → Camera → Field Of View; no reset arrow next
to the value means it was never changed.

A check from the data: 838 cars in the train segmentation masks, measured in pixels and
combined with their depth, give 54/59/63° for an assumed car length of 4.2/4.6/5.0 m (55°
from a 1.85 m width). That is consistent with 60° and rules out 45° or 90°; it is not a
proof. DA3 metric, whose metres depend on the FOV, lands at a test abs_rel of ~0.3 after
fine-tuning, which a wrong FOV or depth unit would push much higher.

## Is the depth format right? (the altitude finding)

The EXR holds planar scene depth in centimetres, the same in train, val and test. But the
capture actor measures the camera altitude from the first surface hit by a downward trace,
not from the ground. At test locations 10 and 11 that surface is a skyscraper roof, so the
camera is 70 m above the roof and the street is 152-275 m away, while train/val ground sits
at 70-182 m. The test split is therefore partly outside the training depth range.
Details and the per-location table: `DATASET.md`; fixes for the regeneration:
`SUGGESTIONS.md`.

## What data do the shadow-removal models need?

ShadowFormer, Restormer and Uformer all train on pixel-aligned pairs of the same view:
`rgb_day` (with shadows) as input and `rgb_noshadow` as target. The dataset has them: every
`rgb_day_2048_<key>` has its `rgb_noshadow_2048_<key>`. Native 2048 only, since no
downscaled shadow-free imagery exists. Training uses random 320 px crops; test images are
scored whole, in tiles.

ShadowFormer also needs a **shadow mask** as a second input (the ISTD protocol it was built
for). The dataset has none, so one is derived at staging: a pixel is in shadow where the
shadowed image is darker than the shadow-free one by more than a 0.1 luminance ratio
(`shadow.mask_threshold`), then a morphological opening of 3 px (`shadow.mask_open`) removes
speckle. The same mask splits the scores into all / shadow / lit pixels. The threshold is
the weak point: it shapes both ShadowFormer's input and the per-region scores.

The semantic masks cannot replace it: they label building, green, road and car, and a
shadow falls across all of them. ShadowFormer needs a shadow / no-shadow mask, not classes.

For the regeneration, a shadow mask exported from UE (a shadow-only pass, or the difference
of lighting buffers) would replace the threshold with an exact mask.

## What data do the domain-adaptation models need?

- **CycleGAN** is unpaired: two sets of images, domain A (`rgb_day`) and domain B
  (`rgb_night` or `rgb_noshadow`). Alignment is not used in training.
- **pix2pix** is paired: the same view in both domains, pixel-aligned. Day/night and
  day/no-shadow share keys, so the dataset covers it.

Native 2048 only (night and no-shadow exist only there); upstream resizes to 286 px and
crops 256 px. Scoring compares the translated test images with the real target-domain test
images (FID, and KID as the primary metric at this sample size). `shadow2noshadow` is also
scored exactly like the shadow-removal models and sits in the same table.

No extra data is needed; 96 training pairs is very little, but the format is right.

## Are Restormer and Uformer suited to shadow removal?

They are general image-restoration networks (denoising, deblurring, deraining), not
shadow-removal models, and neither repository ships shadow weights. Here they start from
their denoising checkpoints (Restormer `real_denoising.pth`, Uformer `Uformer_B_SIDD.pth`)
and learn shadow removal from the day/no-shadow pairs alone, without a mask.

They are still the right comparison. ShadowFormer's own README says "Our implementation is
based on Uformer and Restormer", so the three form a controlled pair: the same transformer
restoration family with and without the shadow-specific design (mask input, shadow-
interaction attention). General restoration networks are also common baselines in
shadow-removal papers; if they match ShadowFormer on this data, the shadow-specific design
is not earning its keep.

Two caveats to state next to any result:

- **Heterogeneous starting weights.** ShadowFormer starts from shadow-removal weights
  (ISTD+), the other two from denoising weights, which gives ShadowFormer a head start. A fair
  comparison trains all three from the same kind of start, or reports the difference.
- **The mask is ground truth at test time.** ShadowFormer's mask is derived from the
  shadow-free target, so at test time it receives a mask a real deployment would only have
  from a shadow detector. ISTD ships ground-truth test masks too, so this matches how
  ShadowFormer is usually scored, but Restormer and Uformer do not get that information.

## Which lightweight model was added, and why

ShadowMaskFormer (IEEE TAI 2025), the `-b` variant: 2.28 M parameters against ShadowFormer's 11.37 M
(counted with the same constructor arguments the grid uses), so about a fifth. It is the most recent
light shadow-removal model with public, pure-PyTorch code (torch plus two timm helpers, no CUDA
extension), and like ShadowFormer it takes a shadow mask, which makes the comparison "same idea, much
smaller network".

Caveats to state next to any result:

- **No weights are released**, so it is trained from scratch on SyMTRS, while ShadowFormer starts from
  ISTD+ weights. A gap between the two mixes architecture and initialisation; the rows are not
  like-for-like (the config header, `describe()` and the results notes say so).
- The upstream repository has **no licence**, so the checkout is gitignored and fetched by script,
  never committed.
- It is written for [-1, 1] images and masks (shadow = +1); the shared worker converts at the network
  boundary only.

Considered and not chosen:

- **BMNet** (CVPR 2022, 0.43 M parameters, weights in the repo) is the lighter fallback, but it is
  older, trains in two stages and hardcodes a `condition.pth` path, so it does not fit the single
  shared worker loop without patching upstream.
- **RASM** (NTIRE 2025 winner, 5.2 M) needs the `natten` CUDA extension, which this repo's
  dependency rule excludes.
- **HomoFormer** (17.8 M) is not light.
- **DeS3 / ShadowDiffusion** are diffusion models: heavy to train and to sample, the opposite of the goal.
