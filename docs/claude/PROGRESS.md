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
  `.venv\Scripts\python.exe -u scripts\smoke_all.py --data-root "C:\Users\bapti\Documents\Unreal Projects\SyMTRS\Saved\DatasetCaptures"`
  (saute les configs déjà `ok` dans `runs/smoke/smoke.tsv`).
- 2026-10-16 : **décision utilisateur** : shadow removal en fine-tuning depuis les poids publiés (ShadowFormer ISTD+, Restormer real_denoising, Uformer-B SIDD), inscrit dans `scripts/smoke_all.py` (SHADOW_INIT → `--set shadow.init_weights=...` dans la file).
