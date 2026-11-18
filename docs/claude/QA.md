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

## Follow-up, after the PR was opened

### Is BMNet worth adding to the benchmark?

Not now. BMNet (CVPR 2022) is 0.43 M parameters and ships ISTD/SRD weights, so it would be
the only "very light *and* pretrained" row. Against that:

- **Coverage.** The benchmark already has a mask-aware transformer at two sizes (ShadowFormer
  11.37 M, ShadowMaskFormer 2.28 M) and two general restoration transformers (Restormer,
  Uformer). BMNet would add a non-transformer (invertible-network) family and one more point on
  a size curve, not a new question.
- **Fidelity.** Its published result comes from a two-stage recipe: a colour network trained
  first (`condition.pth`, loaded from a hardcoded path), then the main network with L1 +
  0.4 LAB + a reverse-pass term. The shared training loop here would train "the BMNet
  architecture with our loss", which is not what the paper reports. Matching it is real work.
- **Cost at 2048.** It never downsamples, so its activation memory is large despite the tiny
  parameter count; it needs crops and tiled inference like the others.

It becomes worth it if the paper wants a size/accuracy curve or a non-transformer baseline.
The better use of time first is training the four current models properly on the
regenerated dataset.

### Would setting the camera altitude above the ground solve everything? Why would pose matter?

*Superseded on the altitude rule: the dataset owner kept the current rule (option A, next
section). The points on pose and FOV still hold.*

It solves the main problem: with the altitude measured above the street, every location gets
the same four heights (70/100/130/160 m above the ground), and train, val and test cover the
same depth range. The per-frame JSON I suggested is not needed:

- **Pose** is already in every file name (`x`, `y`, `z`, `rot`), so it is recorded. It does
  not matter for these single-image tasks; it would only matter for multi-view or
  camera-conditioned models, or for checking that splits do not overlap in space (the split
  zones already guarantee that). That suggestion is withdrawn.
- **FOV** is always 60° (confirmed by the dataset owner), so a constant in the dataset
  documentation is enough. The package now states it as confirmed rather than assumed.

Two details to get right when changing the rule:

1. **The trace must ignore buildings and objects**, not only take the first hit: trace
   against the terrain/street only (a dedicated collision channel or object type), or take the
   lowest hit along the ray. Today's first hit was a roof at both test locations and a lower
   object at loc000 (+22 m) and loc002 (+9 m).
2. **Check the clearance.** At 70 m above the street, a tower taller than 70 m sits beside or
   around the camera: test loc010's roof is ~115 m and loc011's ~82 m, so with the new rule
   the h00 camera would be inside or next to those towers. The capture needs a guard, e.g.
   skip (or raise) a location when the tallest surface in the image footprint comes within
   some margin (say 20 m) of the camera, and log how many locations were skipped per split so
   the guard does not bias one split towards low-rise districts.

### Is exporting a shadow mask from UE worth it?

Agreed, no. The derived mask was measured to settle it (12 train pairs, darkening
`(Y_noshadow - Y_day) / Y_noshadow`):

| darkening | share of pixels |
|---|---|
| within +-5 % (lit, render noise) | ~42 % |
| above 77 % (deep shadow) | ~29 % |
| 5-30 % (penumbra, the only ambiguous band) | 7.4 % |
| brighter in the shadowed render (< -5 %) | 2.3 % |

The distribution is bimodal: moving the threshold from 0.05 to 0.3 changes the shadow fraction
only from 56 % to 49 %. Because the two renders differ almost only by the shadows, the
threshold mask is close to what a UE pass would give; a UE mask would mainly differ in the
penumbra (~7 % of pixels), i.e. in how the shadow/lit split of the metrics treats soft edges,
which should move the regional numbers a little and is unlikely to change a comparison. The threshold (`shadow.mask_threshold = 0.1`) stays documented; a
sensitivity check at 0.05 and 0.2 on the full runs is cheaper than a new export.

### Keep the altitude rule, or measure it above the street? (decided: keep it, option A)

The section above "Would setting the camera altitude above the ground solve everything?"
assumed the rule would change. A second assistant session framed two options, and the dataset
owner chose the first.

- **A. Keep the rule** (camera at a fixed height above the first surface under the location).
  The depth spread is realistic for aerial imagery; document it, check the train/val/test depth
  histograms, and fix any imbalance through the split.
- **B. Height above the street, with a clearance rule**: measure the height from the street,
  and raise the camera only where a tower in the frame would come within ~20-30 m of it,
  logging how many locations are raised per split. Raising keeps almost every location;
  skipping them (the "skip (or raise)" wording above) would push the splits towards low-rise
  districts, so raising is the better variant of B.

Points raised in the comparison:

- **B only halves the problem near towers.** The rule becomes
  `altitude = max(street + h, tallest_in_frame + clearance)`. At test loc010 (tower ~115 m) the
  h00 camera would sit ~140 m above the street instead of 70 m: half of today's deviation
  (185 m), not zero, and `h00..h03` would stop meaning fixed heights for raised frames.
- **B would need one piece of metadata after all.** The `z` in the file name is the camera's
  world height, not its height above the street, and the street level is recorded nowhere. With
  raised frames, the height above the street per frame (in the file name or a CSV row) would be
  needed to know which frames were raised and by how much.
- **A's real defect today is the split, not the spread.** Both test locations sit on towers and
  no train location does. With 12 locations (8/2/2) that cannot be balanced; with the planned
  capture of more than 20,000 views over many locations it balances largely by itself, and it
  can be checked.
- **Resume trap, if the rule ever changes.** Resume keys on file names, which include `z`. A new
  rule changes every `z`, so old frames are not recognised as done and stay in the folder next
  to the new ones: a capture with a changed rule needs a fresh output folder.

**Decision (2026-10-17): option A.** The rule stays, because the regenerated dataset will
have more than 20,000 views instead of 144, so the per-split depth distribution should even out
over many locations. A per-split depth check was proposed next and then dropped (next
section): all three zones contain skyscrapers.

### Is a per-split depth check needed? (no)

After option A was chosen, the assistant proposed a per-split depth check in
`scripts/dataset_manifest.py` and argued for it: the capture actor draws each split's
locations from its own zone (`SplitZones[Split]` in `SyMTRSDatasetCaptureActor.cpp`), so the
split is geographic, and if one zone were denser downtown, more views would make its depth
shift certain rather than average it out. The check would only have reported the per-split
depth distribution, as one sentence for the paper.

The dataset owner's answer settles it: **all three zones include skyscrapers.** The preview's
depth shift is a sampling accident of the small capture: 12 locations in total, 8 of them in
train (each multiplied by 12 through the 3 rotations and 4 heights, hence 96 train frames), and
none of the 8 train locations happened to fall on a tower while both test locations did. With
more than 20,000 views drawn from zones that all contain skyscrapers, every split samples
tower-top frames, so the check is dropped. The earlier statement that "the splits balance
largely by themselves" is right for this reason (every zone contains towers), not because the
split is random: it is not.

### Should the derived shadow masks be computed upfront?

Exporting a shadow mask from UE is still not worth it (see the section above: the derived
mask is close to what a UE pass would give). Computing the *derived* masks once is worth it,
for time and not for quality. Deriving one costs about 1 s per 2048 px pair (0.7 s decoding
the two PNGs, 0.3 s the derivation), and the result is deterministic. Until now every run
re-derived it into its own `<run_dir>/staging/<split>/mask/`: at the planned 20,000+ views
that is about 5.5 h single-threaded, paid again by each of the four shadow-removal models and
by the shadow2noshadow translators' scoring.

Now there is one shared cache (`shadow.mask_cache`, default `.cache/shadow_masks`, gitignored),
keyed by data root and by `mask_threshold` / `mask_open`, with a sidecar per mask recording the
size and mtime of the two source files. A regenerated pair, or another threshold (the 0.05 / 0.2
sensitivity runs), recomputes only what changed. It is filled in parallel (`shadow.mask_workers`,
default half the cores up to 8, because this laptop blue-screens under combined CPU and GPU
load), written atomically, and never written into the dataset (a cache inside the data root is
refused). `scripts/precompute_shadow_masks.py` fills it for whole splits before any run.

Measured on the 144-view preview (train 96, val 24, test 24; CPU only, 8 workers): 35.2 s for
all 144 masks (about 0.24 s per pair wall-clock), and 0.2 s on the second invocation, which
computed 0. The 48 masks that existing runs had already staged for val and test are
byte-identical to the cached ones. At 20,000 views the same pool is roughly an hour once,
instead of hours per run.
