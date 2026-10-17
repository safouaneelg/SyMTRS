# PROGRESS

Plan complet : [PLAN.md](PLAN.md). Ce fichier est l'état courant : une nouvelle session le relit d'abord.

Objectif actuel : **runs courtes de validation** (2 époques / ~500 itérations) de chaque modèle sur le
dataset preview (144 transforms, 96/24/24). Pas de convergence recherchée ; aucun chiffre n'est un résultat.

## État par étape
| Étape | État | Notes |
|---|---|---|
| 0 Mise en place (uv, venvs, .gitignore, docs) | fait | uv 0.13.1, Python 3.12.15, `.venv` + `.venv-zoe` |
| 0b Pré-téléchargement (pip, poids, repos tiers) | fait | ~13,6 GB, builds offline OK |
| 1 Data loader D1–D6 | fait | 486 tests OK, doctor : appariement 100 % |
| 2 Portage view_dataset | fait (tests OK) | `symtrs view`, `notebooks/view_dataset.ipynb` |
| 3 Tâche shadow_removal | fait (dummy) | 3 modèles, 12 tests ; réel = smoke en cours |
| 4 Smoke tests | en cours | `scripts/smoke_all.py` |
| 5 Runs courtes (file `runs/queue.txt`) | à faire | |

## Journal
- 2026-10-16 : analyse Phase 1, plan approuvé. Unreal fermé (VRAM libre 12 GB).
- 2026-10-16 : uv + Python 3.12, venvs créés, install torch cu128 lancée ; ShadowFormer
  (f721581), Restormer (68dc6ac), Uformer (65fc970) clonés.
- 2026-10-16 : `.venv` OK (torch 2.11+cu128, sm_120 détecté, matmul GPU OK). `.venv-zoe` refait en
  Python 3.10 (timm 0.6.12 casse sur 3.12 : dataclass mutable default). `symtrs view` porté.
  Pré-téléchargement des poids délégué (script `scripts/prefetch_weights.py`, cache `.cache/`).
  Lanceur reprenable : `scripts/run_queue.ps1` → `scripts/run_queue.py` (skip si summary.json,
  resume depuis last.pt, `runs/STOP` pour arrêter, état dans `runs/validation/status.tsv`).

## Reprendre après une coupure
1. `Get-Content logs\queue_*.log -Tail 20` et `runs\validation\status.tsv` : où en est la file.
2. Si le processus n'est plus là (`Get-Process -Id (Get-Content logs\run_queue.pid)`), relancer
   `powershell -File scripts\run_queue.ps1` : les runs finies sont sautées, l'interrompue reprend.
- 2026-10-16 : 1er smoke réel OK — unet_small 512, 1 époque limitée, val+test écrits (113 s/époque pour 3 batches : chargement à profiler).
- 2026-10-16 19:03 : session précédente coupée → agents et install zoe arrêtés. Relancé en
  **processus détachés** : `scripts/prefetch_all.ps1` (poids principaux + venv zoe Python 3.10 +
  poids ZoeDepth ; logs `logs/prefetch_main_*.log`, `logs/install_zoe_*.log`, marqueurs
  `PREFETCH_MAIN_DONE` / `ZOE_OK` / `PREFETCH_ZOE_DONE`). Barre de progression :
  `scripts/download_progress.ps1` (fenêtre séparée). Agent shadow_removal relancé.
  Si une session reprend : relancer `scripts/prefetch_all.ps1` est sans risque (idempotent).

### shadow_removal (agent)
- 2026-10-16 done (untested end to end): `symtrs/metrics/shadow.py`, `symtrs/model/shadow/{archs,staging,runner,worker}.py`, config `ShadowConfig` (`shadow:` section), registry keys, trainer routing, sweep/tables/models/dry-run, `SHADOW_MODELS` in `configs/generate_grid.py` -> `configs/shadow_removal/*.yaml`.
- Design: own worker imports only the upstream network class (no upstream train.py); staging = `<run>/staging/<split>/{input,target,mask}`; predictions PNG scored in parent.
- TODO: update tests/test_config.py TASKS set, new tests/test_shadow_removal.py, scripts/fetch_third_party.py (+ third_party/patches), AGENTS.md section 1 + configs/README.md, dummy smoke, real smoke (bounded), full pytest + generate_grid --check.
- 2026-10-16 19:25 : **tous les poids sont sur disque** (~13,6 GB : `.cache/` 12 GB + `checkpoints/depth_pro.pt`).
  MiDaS ×5, ZoeDepth N/K/NK (fichiers bruts), DA3 Large 1,53 GB + Metric 1,24 GB, Depth Pro,
  DeepLab r50/r101, MiT b0–b2 (HF `nvidia/mit-b*`), Inception ×2, ShadowFormer ISTD/ISTD+,
  Restormer real_denoising, Uformer-B SIDD. Installeur signé Python 3.10.11 dans
  `.cache/installers/` (repli si Smart App Control bloque le Python 3.10 de uv).
  Restent à corriger (sans réseau) : build DA3 (image processor), SegFormer (timm n'a pas mit_b*),
  MiDaS swin2-tiny (state_dict vs timm 1.0.30), venv zoe (DLL bloquée par Smart App Control).
- 2026-10-16 (post-crash): TASKS test + scripts/fetch_third_party.py done; tests/test_shadow_removal.py exists (state unverified); next: run suite, dummy smoke per model.
- tests/test_shadow_removal.py rewritten (was NUL-corrupted by the crash), 12 pass. Next: full suite, generate_grid --check, docs, GPU smokes one at a time.
- 2026-10-16 : lenteur expliquée = débordement VRAM (bs32 → 18,6 GB). Smoke tests : relever le pic VRAM et fixer un batch par config < ~8 GB.
- restormer dummy smoke OK end-to-end (20 it, bs2, p256): 8.7 s/it (11.7 last), peak_vram 12.37GB (suspicious, investigating), train 236s, predict 7.8s/img first then 0.7; test psnr 34.3 (dummy, meaningless). log logs/shadow_restormer_smoke.log
- restormer VRAM: bs1 p256 = 6.4GB 0.33 s/it; bs2 = 12.4GB spills (8.7 s/it). Need per-model batch tuning in generate_grid SHADOW_MODELS.
- generate_grid: restormer batch=1 (12GB card). bs4 measured: shadowformer 8.2GB 0.28s/it, uformer 9.2GB 0.32s/it.
- config fix applied (restormer bs1). Next: full smokes shadowformer, uformer w/ predict; docs; fetch/AGENTS.
- fixed relative-path bug (worker cwd = upstream repo): runner absolutizes init_weights/checkpoint/data.root. shadowformer 50it from scratch: loss .42->.26 (converging, starts non-identity), 0.30s/it.
- init_weights verified: shadowformer istd_plus 274/274, uformer SIDD 759/759, restormer real_denoising 406/406 (after LayerNorm_type=BiasFree). All 3 models full dummy smoke OK. Next: docs (AGENTS.md, configs/README.md), pytest, final report.
- docs updated (AGENTS.md, configs/README.md).
- 2026-10-16 19:45 : builds offline OK pour toutes les clés (agent) ; tâche shadow_removal finie
  (agent, 520 tests). Smoke tests réels en cours : `scripts/smoke_all.py` → `runs/smoke/smoke.tsv`
  puis `runs/queue.txt`. Documentation des runs : `docs/claude/DATASET.md` (empreinte dataset
  `02b1b436c745c3fc`) et `docs/claude/RUNS.md` généré par `scripts/runs_report.py` (commande
  exacte, config résolue, env, versions, batch, budget, métriques test par run).
- 2026-10-16 19:55 : commit local `2843b46` sur `features` (tout le travail jusqu'ici). Smoke tests
  11/91 faits. **Pour reprendre après extinction** :
  `.venv\Scripts\python.exe -u scripts\smoke_all.py --data-root "<DATA_ROOT>"`
  (saute les configs déjà `ok` dans `runs/smoke/smoke.tsv`).
- 2026-10-16 : **décision utilisateur** : shadow removal en fine-tuning depuis les poids publiés (ShadowFormer ISTD+, Restormer real_denoising, Uformer-B SIDD), inscrit dans `scripts/smoke_all.py` (SHADOW_INIT → `--set shadow.init_weights=...` dans la file).
- 2026-10-17 03:45 : PC en veille ~20:00→03:39 pendant le smoke (7,6 h perdues ; da3_large_2048 affiché 460 min). Ajout `keep_awake()` (SetThreadExecutionState) dans run_queue/smoke_all ; garder le capot ouvert. 39/91 smoke faits (33 ok, 6 FAIL : DA3 1024 OOM patch 0, Depth Pro OOM, BEiT-L drop_path/timm). 3 agents Sonnet : DA3+BEiT, Depth Pro 12 GB, audit plausibilité métriques.
- 2026-10-17 04:00 : audit (`docs/claude/AUDIT_smoke.md`) → bugs de protocole confirmés : (1) val/test
  2048 évalués sur un crop aléatoire 512 ; (2) MiDaS = disparité inversée + alignement échelle seule ;
  (3) DA3-metric sans focale ; (4) ZoeD-N NaN en fp16 ; (5) mIoU compte 0 pour classe absente ;
  + ZoeD-K/NK plafonnés (bins 80 m / 10 m) vs scènes 70–270 m. Fix DA3 1024 (patch 512) + BEiT→.venv-zoe
  faits. 2 agents : protocole d'évaluation ; conventions profondeur (FOV 60° supposé, à confirmer).
  Après corrections : **refaire tout le smoke** (résultats depth/seg actuels périmés).
  Constat dataset : test plus profond (médiane ~208 m vs 92 train) et classes très différentes entre
  splits (val : 63 % route) — split géographique, à mentionner avec tout chiffre.
- 2026-10-17 04:10 : protocole d'évaluation corrigé (agent ; 581 tests) ; Depth Pro fine-tunable (freeze jamais appliqué → corrigé, 7,9 GiB). Objectif optionnel 3 (translation ombre⇄sans ombre pix2pix/CycleGAN via le front-end domain) lancé (agent). Restant : conventions profondeur (agent), re-smoke complet, file de nuit, RESULTS/RUNS, PR.

## Checklist PR (à faire avant la PR, dépôt PUBLIC safouaneelg/SyMTRS)
- [x] `gh auth status` : compte GitHub de l'utilisateur, droit push OK ; local `features` en avance sur `origin/features`.
- [x] Chemins personnels retirés des fichiers commités (chemins absolus de la machine, dossier du projet Unreal) :
      DATASET.md / RUNS.md → `<DATA_ROOT>` ; `scripts/run_queue.ps1` → paramètre/env `SYMTRS_DATA_ROOT` ;
      PLAN.md / PROGRESS.md → placeholders `<DATA_ROOT>`, `<REPO>`, `<UNREAL_PROJECT>`. `scripts/dataset_manifest.py` et `scripts/runs_report.py` redactent par défaut (`--redact-root`).
- [ ] Email d'auteur des commits : demander à l'utilisateur (email perso public vs noreply GitHub).
- [ ] Aucun .pt/.pth/log/donnée/`.cache`/`checkpoints`/`runs` suivi ; `third_party/*/` ignoré.
- [ ] Suite de tests verte, `generate_grid.py --check`, RESULTS.md + RUNS.md à jour.
- [ ] Nouvelle branche depuis `features`, push, PR → `features`, corps de PR prêt ; **confirmation utilisateur avant création**.
- 2026-10-17 04:50 : **smoke passe 1 terminée** (91 configs : 82 ok, 9 FAIL — tous corrigés depuis).
  Temps : 9 h 08 de mur (19:42 → 04:50) dont ~7,6 h de veille ; calcul effectif 78 min de jobs
  (depth ~27 min hors job gonflé par la veille, shadow 27, seg 10, SR 8, domain 7). Archivée dans
  `runs/smoke_pass1/`. Commit `a7b440d` (corrections audit). **Passe 2 lancée** (95 configs, code
  corrigé, keep_awake actif) → `logs/smoke_all_pass2_20261017_0450.log`. Revue critique indépendante
  du diff en parallèle (agent, lecture seule) → `docs/claude/REVIEW_night.md`.
- 2026-10-17 05:10 : revue indépendante (`docs/claude/REVIEW_night.md`) → **bug majeur préexistant
  vérifié** : scheduler LR dimensionné en pas mais appelé par époque (2 époques ≈ 4–8 % du LR, pas de
  décroissance). + resume ne restaure pas `best`, test scoré avec les poids `last`, mIoU dénominateur
  variable, matching affine des tuiles fragile, prédictions ≤0 ignorées, FID asymétrique, smoke qui
  change le batch effectif, init shadow hétérogène. 3 agents Sonnet de correction (trainer / métriques+
  tuilage / domain+shadow+smoke). La passe 2 du smoke continue (valide pour plomberie et batch).
- 2026-10-17 05:25 : corrections de la revue commitées (`52d07ba`, 654 tests ; tests du scheduler
  vérifiés en échec sur l'ancien code). Passe 2 arrêtée (code intermédiaire, archivée
  `runs/smoke_pass2_partial/`). **Passe 3 propre** lancée sur le code final (95 configs) →
  `logs/smoke_all_pass3_*.log`, horodatage début/fin dans `logs/smoke_pass3_start.txt`.
