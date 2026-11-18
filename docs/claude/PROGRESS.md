# PROGRESS

Full plan: [PLAN.md](PLAN.md). This file is the current state: a new session re-reads it first.

Current goal: **short validation runs** (2 epochs / ~500 iterations) of each model on the
preview dataset (144 transforms, 96/24/24). No convergence sought; no number is a result.
State: the 96 validation runs are done (step 5); what is left is the PR checklist below.

## State per step
| Step | State | Notes |
|---|---|---|
| 0 Setup (uv, venvs, .gitignore, docs) | done | uv 0.13.1, Python 3.12.15, `.venv` + `.venv-zoe` |
| 0b Pre-download (pip, weights, third-party repos) | done | ~13.6 GB, offline builds OK |
| 1 Data loader D1–D6 | done | 486 tests OK, doctor: 100% pairing |
| 2 view_dataset port | done (tests OK) | `symtrs view`, `notebooks/view_dataset.ipynb` |
| 3 shadow_removal task | done | 3 models, then ShadowMaskFormer as a 4th (2026-10-17); real smoke OK; identity reference (test 12.4 dB) |
| 4 Smoke tests | done (passes 3-5) | 95/95 configs; see journal 2026-10-17 06:48 / 08:00 |
| 5 Short runs (queue `runs/queue.txt`) | done (2026-10-17, 08:43 -> 14:03) | 96/96 ok (95 grid runs + ShadowMaskFormer), all pass `run_verdict`. First pass 3 h 37 (80 ok / 15 failed), then re-runs; job time per task: segmentation 21.5 min, depth 2 h 18, super-resolution 59 min, shadow 60 min, domain 20 min (4 h 59 in total); see journal 12:20 -> 14:03 |

## Journal
- 2026-10-16: Phase 1 analysis, plan approved. Unreal closed (VRAM free 12 GB).
- 2026-10-16: uv + Python 3.12, venvs created, torch cu128 install launched; ShadowFormer
  (f721581), Restormer (68dc6ac), Uformer (65fc970) cloned.
- 2026-10-16: `.venv` OK (torch 2.11+cu128, sm_120 detected, GPU matmul OK). `.venv-zoe` redone with
  Python 3.10 (timm 0.6.12 breaks on 3.12: dataclass mutable default). `symtrs view` ported.
  Weights pre-download delegated (script `scripts/prefetch_weights.py`, cache `.cache/`).
  Resumable launcher: `scripts/run_queue.ps1` → `scripts/run_queue.py` (skip if summary.json,
  resume from last.pt, `runs/STOP` to stop, state in `runs/validation/status.tsv`).

## Resuming after an interruption
1. `Get-Content logs\queue_*.log -Tail 20` and `runs\validation\status.tsv`: where the queue stands.
2. If the process is gone (`Get-Process -Id (Get-Content logs\run_queue.pid)`), re-launch
   `powershell -File scripts\run_queue.ps1`: finished runs are skipped, the interrupted one resumes.
- 2026-10-16: 1st real smoke OK — unet_small 512, 1 limited epoch, val+test written (113 s/epoch for 3 batches: loading to be profiled).
- 2026-10-16 19:03: previous session cut → agents and zoe install stopped. Re-launched as
  **detached processes**: `scripts/prefetch_all.ps1` (main weights + zoe venv Python 3.10 +
  ZoeDepth weights; logs `logs/prefetch_main_*.log`, `logs/install_zoe_*.log`, markers
  `PREFETCH_MAIN_DONE` / `ZOE_OK` / `PREFETCH_ZOE_DONE`). Progress bar:
  `scripts/download_progress.ps1` (separate window). shadow_removal agent re-launched.
  If a session resumes: re-launching `scripts/prefetch_all.ps1` is safe (idempotent).

### shadow_removal (agent)
- 2026-10-16 done (untested end to end): `symtrs/metrics/shadow.py`, `symtrs/model/shadow/{archs,staging,runner,worker}.py`, config `ShadowConfig` (`shadow:` section), registry keys, trainer routing, sweep/tables/models/dry-run, `SHADOW_MODELS` in `configs/generate_grid.py` -> `configs/shadow_removal/*.yaml`.
- Design: own worker imports only the upstream network class (no upstream train.py); staging = `<run>/staging/<split>/{input,target,mask}`; predictions PNG scored in parent.
- TODO: update tests/test_config.py TASKS set, new tests/test_shadow_removal.py, scripts/fetch_third_party.py (+ third_party/patches), AGENTS.md section 1 + configs/README.md, dummy smoke, real smoke (bounded), full pytest + generate_grid --check.
- 2026-10-16 19:25: **all the weights are on disk** (~13.6 GB: `.cache/` 12 GB + `checkpoints/depth_pro.pt`).
  MiDaS ×5, ZoeDepth N/K/NK (raw files), DA3 Large 1.53 GB + Metric 1.24 GB, Depth Pro,
  DeepLab r50/r101, MiT b0–b2 (HF `nvidia/mit-b*`), Inception ×2, ShadowFormer ISTD/ISTD+,
  Restormer real_denoising, Uformer-B SIDD. Signed Python 3.10.11 installer in
  `.cache/installers/` (fallback if Smart App Control blocks uv's Python 3.10).
  Still to fix (without network): DA3 build (image processor), SegFormer (timm has no mit_b*),
  MiDaS swin2-tiny (state_dict vs timm 1.0.30), zoe venv (DLL blocked by Smart App Control).
- 2026-10-16 (post-crash): TASKS test + scripts/fetch_third_party.py done; tests/test_shadow_removal.py exists (state unverified); next: run suite, dummy smoke per model.
- tests/test_shadow_removal.py rewritten (was NUL-corrupted by the crash), 12 pass. Next: full suite, generate_grid --check, docs, GPU smokes one at a time.
- 2026-10-16: slowness explained = VRAM spill (bs32 → 18.6 GB). Smoke tests: record the VRAM peak and set one batch per config < ~8 GB.
- restormer dummy smoke OK end-to-end (20 it, bs2, p256): 8.7 s/it (11.7 last), peak_vram 12.37GB (suspicious, investigating), train 236s, predict 7.8s/img first then 0.7; test psnr 34.3 (dummy, meaningless). log logs/shadow_restormer_smoke.log
- restormer VRAM: bs1 p256 = 6.4GB 0.33 s/it; bs2 = 12.4GB spills (8.7 s/it). Need per-model batch tuning in generate_grid SHADOW_MODELS.
- generate_grid: restormer batch=1 (12GB card). bs4 measured: shadowformer 8.2GB 0.28s/it, uformer 9.2GB 0.32s/it.
- config fix applied (restormer bs1). Next: full smokes shadowformer, uformer w/ predict; docs; fetch/AGENTS.
- fixed relative-path bug (worker cwd = upstream repo): runner absolutizes init_weights/checkpoint/data.root. shadowformer 50it from scratch: loss .42->.26 (converging, starts non-identity), 0.30s/it.
- init_weights verified: shadowformer istd_plus 274/274, uformer SIDD 759/759, restormer real_denoising 406/406 (after LayerNorm_type=BiasFree). All 3 models full dummy smoke OK. Next: docs (AGENTS.md, configs/README.md), pytest, final report.
- docs updated (AGENTS.md, configs/README.md).
- 2026-10-16 19:45: offline builds OK for all the keys (agent); shadow_removal task finished
  (agent, 520 tests). Real smoke tests in progress: `scripts/smoke_all.py` → `runs/smoke/smoke.tsv`
  then `runs/queue.txt`. Run documentation: `docs/claude/DATASET.md` (dataset fingerprint
  `02b1b436c745c3fc`) and `docs/claude/RUNS.md` generated by `scripts/runs_report.py` (exact command,
  resolved config, env, versions, batch, budget, test metrics per run).
- 2026-10-16 19:55: local commit `2843b46` on `features` (all the work so far). Smoke tests
  11/91 done. **To resume after a shutdown**:
  `.venv\Scripts\python.exe -u scripts\smoke_all.py --data-root "<DATA_ROOT>"`
  (skips the configs already `ok` in `runs/smoke/smoke.tsv`).
- 2026-10-16: **user decision**: shadow removal as fine-tuning from the published weights (ShadowFormer ISTD+, Restormer real_denoising, Uformer-B SIDD), written into `scripts/smoke_all.py` (SHADOW_INIT → `--set shadow.init_weights=...` in the queue).
- 2026-10-17 03:45: PC asleep ~20:00→03:39 during the smoke (7.6 h lost; da3_large_2048 shown as 460 min). Added `keep_awake()` (SetThreadExecutionState) in run_queue/smoke_all; keep the lid open. 39/91 smoke done (33 ok, 6 FAIL: DA3 1024 OOM patch 0, Depth Pro OOM, BEiT-L drop_path/timm). 3 Sonnet agents: DA3+BEiT, Depth Pro 12 GB, metrics plausibility audit.
- 2026-10-17 04:00: audit (`docs/claude/AUDIT_smoke.md`) → confirmed protocol bugs: (1) val/test
  2048 evaluated on a random 512 crop; (2) MiDaS = inverted disparity + scale-only alignment;
  (3) DA3-metric without focal length; (4) ZoeD-N NaN in fp16; (5) mIoU counts 0 for an absent class;
  + ZoeD-K/NK capped (bins 80 m / 10 m) vs scenes of 70–270 m. DA3 1024 fix (patch 512) + BEiT→.venv-zoe
  done. 2 agents: evaluation protocol; depth conventions (FOV 60° assumed, to be confirmed).
  After the fixes: **redo the whole smoke** (current depth/seg results are stale).
  Dataset finding: test is deeper (median ~208 m vs 92 train) and the classes are very different between
  splits (val: 63% road) — geographic split, to be mentioned with any number.
- 2026-10-17 04:10: evaluation protocol fixed (agent; 581 tests); Depth Pro fine-tunable (freeze never applied → fixed, 7.9 GiB). Optional goal 3 (shadow⇄no-shadow translation pix2pix/CycleGAN via the domain front-end) launched (agent). Remaining: depth conventions (agent), full re-smoke, night queue, RESULTS/RUNS, PR.

## PR checklist (to do before the PR, PUBLIC repo safouaneelg/SyMTRS)
- [x] `gh auth status`: the user's GitHub account, push right OK; local `features` ahead of `origin/features`.
- [x] Personal paths removed from the committed files (absolute paths of the machine, Unreal project folder):
      DATASET.md / RUNS.md → `<DATA_ROOT>`; `scripts/run_queue.ps1` → parameter/env `SYMTRS_DATA_ROOT`;
      PLAN.md / PROGRESS.md → placeholders `<DATA_ROOT>`, `<REPO>`, `<UNREAL_PROJECT>`. `scripts/dataset_manifest.py` and `scripts/runs_report.py` redact by default (`--redact-root`).
- [ ] Commit author email: ask the user (public personal email vs GitHub noreply).
- [ ] No .pt/.pth/log/data/`.cache`/`checkpoints`/`runs` tracked; `third_party/*/` ignored.
- [ ] Test suite green, `generate_grid.py --check`, RESULTS.md + RUNS.md up to date.
- [ ] New branch from `features`, push, PR → `features`, PR body ready; **user confirmation before creation**.
- 2026-10-17 04:50: **smoke pass 1 finished** (91 configs: 82 ok, 9 FAIL — all fixed since).
  Time: 9 h 08 wall (19:42 → 04:50) of which ~7.6 h of sleep; effective compute 78 min of jobs
  (depth ~27 min excluding the job inflated by the sleep, shadow 27, seg 10, SR 8, domain 7). Archived in
  `runs/smoke_pass1/`. Commit `a7b440d` (audit fixes). **Pass 2 launched** (95 configs, fixed
  code, keep_awake active) → `logs/smoke_all_pass2_20261017_0450.log`. Independent critical review of the diff
  in parallel (agent, read-only) → `docs/claude/REVIEW_night.md`.
- 2026-10-17 05:10: independent review (`docs/claude/REVIEW_night.md`) → **major pre-existing bug
  verified**: LR scheduler sized in steps but called per epoch (2 epochs ≈ 4–8% of the LR, no
  decay). + resume does not restore `best`, test scored with the `last` weights, mIoU denominator
  varies, tile affine matching fragile, predictions ≤0 ignored, asymmetric FID, smoke that
  changes the effective batch, heterogeneous shadow init. 3 Sonnet fix agents (trainer / metrics+
  tiling / domain+shadow+smoke). Smoke pass 2 continues (valid for plumbing and batch).
- 2026-10-17 05:25: review fixes committed (`52d07ba`, 654 tests; scheduler tests
  verified to fail on the old code). Pass 2 stopped (intermediate code, archived
  `runs/smoke_pass2_partial/`). **Clean pass 3** launched on the final code (95 configs) →
  `logs/smoke_all_pass3_*.log`, start/end timestamps in `logs/smoke_pass3_start.txt`.
- 2026-10-17 06:48: **pass 3 finished**: 1 h 39 wall (05:09 → 06:48), 94 min of cumulative compute,
  no sleep. 95/95 "ok" **but critical verification: false**. 13/16 SR had failed (training OOM
  on whole 2048 images, SwinIR x1 OOM at construction, SRGAN x1 `ValueError`) and
  `symtrs train` exited with 0 → never detected nor reduced in batch; pix2pix `night2day`/`noshadow2shadow`
  with `test: {}`; CycleGAN `night2day`/`noshadow2shadow` scored the opposite translation (`fake_A`).
  Healthy findings: depth/seg/shadow plausible; metric depth metrics ~10x too close in
  zero-shot (domain gap, to be rechecked after fine-tuning); `iou_0` = null expected (absent class).
  2 Sonnet agents (SR, domain) + 1 independent review of the SR diff; fixes + tests (703).
- 2026-10-17 08:00: **pass 4** (SR + domain, 07:24 → 08:00): domain 8/8 ok (BtoA: 24 images
  generated, consistent KIDs); SR 14/16, `autoencoder_x4/x8` OOM in evaluation (the network runs at the
  target size) → eval window capped at 1024 px target (`eval_hr_window`). Review fixes
  (clamp [0,1] before metrics, `best_psnr` kept at resume, crop too large = error, stale `summary.json`
  deleted before relaunch). Commit `4f9769e`. **Pass 5** = re-smoke of the 16 SR on this code.
- 2026-10-17 10:50: **dataset check** (goal: appropriate models, appropriate data format). Depth EXR =
  planar depth in cm, consistent across splits. Camera altitude is above the first `ECC_Visibility`
  hit, so at test loc010/loc011 (skyscraper roof) the ground is 152-275 m away vs 70-182 m in train/val
  (details: `DATASET.md`). FOV: no metadata; 60 deg is consistent with a car-size estimate (838 cars,
  54-63 deg) but unproven. Validation findings so far: Depth Pro / ZoeD-NK (frozen, lr 5e-6) do not move
  in 2 epochs (effectively zero-shot), DA3-metric does (val abs_rel 0.81 -> 0.50, test 0.27-0.31);
  ZoeD-K test abs_rel 0.65-0.68; relative models 0.095-0.22 with 1024 rows better than 2048 rows
  (note, not a claim); `best_epoch` 0 on several relative rows (2 epochs, meaningless). Suggestions
  for the regeneration added to `SUGGESTIONS.md`; PR body has a "Dataset findings" section.
- 2026-10-17: **ShadowMaskFormer added** as the fourth shadow-removal model (light, mask-aware, 2.28 M
  parameters; no released weights, so trained from scratch; per-architecture [-1, 1] range handled in
  `worker.forward`). 96 configs, pin `9c505e1` fetched; CPU only (a GPU queue was running): dry run on
  the real dataset root OK, no GPU smoke yet. Queue line in `runs/queue_shadowmaskformer.txt`
  (`--set shadow.iterations=500 --batch-size 4`). See IMPROVEMENTS.md.
- 2026-10-17 12:20: **validation queue, first pass finished**: 08:43 -> 12:20 (3 h 37 wall), 95 jobs,
  **80 ok / 15 failed**. Job time of the 80 ok runs 3 h 28, failed attempts 10 min (they die in
  6-80 s). Failures: (1) 4 CUDA OOM (rc 3): `da3_large` 256 / 512 / 1024 and `zoedepth_nk_512`, the
  batch chosen by `smoke_all` was too large for a full 2-epoch fine-tune of 411 M parameters on 12 GB;
  (2) 11 relative-path failures: all of shadow (Restormer, ShadowFormer, Uformer) and domain (CycleGAN x4,
  pix2pix x4), because the queue passed `runs/validation` relative and both subprocesses run with the
  upstream checkout as cwd (`FileNotFoundError: runs/validation/restormer/config.yaml`; "Could not find
  trainA/trainB" for the CycleGAN family). The failed attempts had left upstream option dumps in
  `pytorch-CycleGAN-and-pix2pix/runs/` (36 KB of `train_opt.txt`, gitignored): deleted;
  `symtrs setup-adapt --check` reports the tree ready. The 11 failed run dirs and logs were moved to
  `runs/validation_failed_relpath/`. Fix: `RunConfig.anchored()` at the entry of the three runners
  (`tests/test_relative_run_dir.py`).
- 2026-10-17 12:22 -> 12:33: **OOM re-runs**, 4/4 ok in 11 min (`runs/queue_oom.txt`): smaller batch with
  `grad_accum` raised so the effective batch equals the grid's (da3_large 1024: batch 1 x 8, 5.2 min;
  256: 2 x 16, 1.6 min; 512: 1 x 16, 2.5 min; zoedepth_nk_512: 2 x 16, 1.8 min). `runs/queue.txt`
  (regenerated by `smoke_all` at 13:51) was patched back to these working batches. Open note: the
  smoke picks its batch from a few frames that do not include a whole-image scoring pass, so it can
  choose a batch that OOMs in a real run (BLOCKERS, SUGGESTIONS).
- 2026-10-17 12:33 -> 13:41: **re-run of the 11 shadow/domain jobs** after the relative-path fix, 11/11 ok in
  68 min (the other 84 entries skipped as already scored): Restormer 17.3 min, ShadowFormer 18.6, Uformer
  12.5, CycleGAN 1.6-2.9 (x4), pix2pix 2.5-3.9 (x4). `kid_std` placeholder fixed in the same period (see IMPROVEMENTS.md):
  it was a hard-coded `0.0` (`num_subsets=0`), now `null`; legacy zeros are shown as n/a.
- 2026-10-17 13:42 -> 14:03: **ShadowMaskFormer** (TAI 2025, 2.28 M parameters, from scratch, 500 iterations,
  batch 4): GPU smoke 13:42 -> 13:51 ok, validation run 13:51 -> 14:03 (11.9 min) ok. Test PSNR
  20.81 all / 16.97 shadow / 26.93 non-shadow. **96/96 runs now pass `run_verdict`**; RESULTS.md
  regenerated at 14:05 from `runs/validation` (96 summaries).
- 2026-10-17 13:50 (Sonnet audit, read-only) and findings: (a) shadow removal, test PSNR all / shadow / non-shadow:
  ShadowFormer (ISTD+) 23.39 / 19.48 / 30.05, Uformer 21.51 / 17.90 / 26.46, Restormer 20.90 / 17.39 /
  25.49, ShadowMaskFormer 20.81 / 16.97 / 26.93, identity 12.40 / 7.87 / 27.91. ShadowFormer does not
  damage the non-shadow pixels on val; Uformer and Restormer beat identity on shadow pixels but lose
  non-shadow quality on val (denoising init, 500 iterations). Init weights loaded in full (274/274, 406/406,
  759/759 tensors). `val_every` 500 = a single val point, so best = last. (b) Domain "do nothing"
  baseline (`scripts/domain_identity_baseline.py`, input vs target KID): day->night 0.231, night->day
  0.238, shadow->noshadow 0.104, noshadow->shadow 0.096; all 8 two-epoch translators are worse
  (0.18-0.61). The test split has 2 locations, so split-half KID mostly measures scene shift.
  Direction and `fake_B` checked by mean brightness. (c) Depth: Depth Pro and ZoeD-NK (frozen
  encoder, lr 5e-6) stay at test abs_rel 0.97-0.98; DA3-metric 1024/2048 reach 0.27-0.31 but 256/512
  stay at 0.87-0.96 (eff_batch 64 on 96 images = 2 optimizer steps per epoch, both in warmup);
  relative models 0.05-0.22, best_epoch 0 on several rows. (d) Dataset: depth range on test is
  outside train because of the altitude rule (DATASET.md). (e) The SR front-end never wrote `config.yaml`
  for training runs: fixed (it now writes it after the dry-run return); `runs_report.py` lists the 16 earlier SR
  runs from their `summary.json` (marked †) instead of dropping them, so RUNS.md has all 96 runs.
- 2026-10-17: **FOV confirmed** by the dataset owner (capture camera horizontal FOV always 60 deg); the assumption wording was removed from AGENTS.md, README, the generator header, config.py, the results report and the docs. Default unchanged; the car-size estimate stays as a consistency check.
