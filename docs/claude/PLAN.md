# Plan : entraînement réel SyMTRS + benchmark de suppression d'ombres

## Contexte
Aujourd'hui, toutes les runs du repo `symtrs` ont été faites sur `make-dummy`. Un vrai dataset existe désormais dans `…\SyMTRS\Saved\DatasetCaptures` (rendu UE 5.8, City Sample). Il est petit : 144 transforms, soit 96 / 24 / 24 pour train / val / test. C'est un aperçu de la run prévue à 20 000 transforms.

Objectifs :
1. Entraîner pour de vrai la grille du repo.
2. Ajouter un benchmark de suppression d'ombres : ShadowFormer, Restormer, Uformer.
3. Optionnel : une translation ombre ⇄ sans ombre de type CycleGAN/pix2pix.
4. Porter `view_dataset` dans le repo.

Choix validés :
- environnement **uv + Python 3.12** ;
- **nouvelle tâche symtrs `shadow_removal`** pilotée en sous-processus ;
- **grille complète, par étapes, en runs courtes** : peu d'époques, l'objectif est de valider chaque modèle sur ce dataset smoke, pas de converger ;
- sous-agents **Sonnet 5.5**.

À l'approbation, ce plan devient `docs/claude/PROGRESS.md`. Les fichiers BLOCKERS, IMPROVEMENTS, SUGGESTIONS et RESULTS.md sont créés en même temps.

## Constats de l'analyse (Phase 1)

### Repo
- Registre : depth 11, seg 6, domain 2 (×2 directions), SR 4 (×4 échelles).
- Configs : 88 YAML générés par `configs/generate_grid.py`.
- Tests : 234.
- Aucune trace d'ombre dans le code.
- Ni `third_party/`, ni `docs/claude/`.

### Dataset
- Les 17 dossiers attendus sont présents dans chaque split : `rgb_day`, `rgb_night`, `rgb_noshadow`, `depth`, `semantic`, et `*_lr/x{2,4,8}`.
- Appariement complet : 14 fichiers par transform.
- Chaque `rgb_day_2048_<key>` a son `rgb_noshadow_2048_<key>`.

### Incompatibilités bloquantes entre le code et les vraies données
| # | Problème | Où | Correctif |
|---|---|---|---|
| D1 | Les noms portent un préfixe : `rgb_day_2048_<key>`, `depth_day_512_<key>`, `seg_day_…`. `match_by_stem` ne trouve donc **aucune paire**. | `symtrs/data/io.py:16,73,80` | Étendre `normalize_stem` pour retirer `^(rgb\|seg\|depth)_(day\|night\|noshadow)_\d+_`. Ajouter un test. |
| D2 | pix2pix apparie par basename exact (jour ≠ nuit). | `pytorch-CycleGAN-and-pix2pix/data/aligned_dataset.py:93`, staging `symtrs/model/domain/upstream.py:326` | Le staging crée les liens sous la clé normalisée, sans toucher upstream. |
| D3 | Le masque de segmentation vaut gris = ID×63, sur **5 classes** : background, building, green, road, car. Le code attend 6 classes avec la valeur brute = ID. | `semantic_dataset.py`, `config.py` (num_classes) | Ajouter un paramètre `data.label_step` (63), un décodage `round(v/63)` avec les valeurs hors grille → 255, et `num_classes: 5`. La décision est notée dans la checkpoint et dans un test. |
| D4 | La profondeur est en **cm**, le ciel vaut des valeurs énormes. | `exr.py:218`, `depth_scale` | `depth_scale: 0.01`. Ajouter `data.max_depth` (par ex. 1000 m) pour invalider le ciel. |
| D5 | Les PNG RGB sont en RGBA. | `data/io.py` | Vérifier que la lecture passe bien par `convert("RGB")`, sinon l'ajouter. |
| D6 | `rgb_noshadow` n'est connu d'aucun code. | `splits.py:35,65` | Ajouter le rôle `noshadow`, natif 2048 uniquement (comme `night`). |

### Docs
- Les docs ne mentionnent aucun commentaire de reviewer.
- `SyMTRS_Pipeline.md` se contredit (×51 contre ×63). C'est le tableau §1 qui fait foi (×63, 5 classes).

### Environnement
- GPU : RTX 5070 Ti Laptop, 12 GB, sm_120. Il faut torch ≥ 2.7 avec cu128.
- Au moment du check, **10,4 GB de VRAM étaient pris par Unreal Editor**.
- CPU Ultra 9 285H, 63 GB de RAM, 946 GB libres.
- Python 3.14 sans torch. Ni uv, ni nvcc (pas nécessaire avec les wheels).
- `gh` 2.102.
- Windows : le staging du domain_adapt utilise des symlinks, ce qui demande le mode Développeur ou un repli copie/hardlink.

### Tools
- `view_dataset.py` (146 lignes, matplotlib) et un notebook. `read_depth_cm` dépend d'OpenEXR.

## Plan d'exécution

### Étape 0 — Mise en place (moi)
- Créer la branche locale `features` (déjà active).
- Créer `docs/claude/*.md` et `logs/`.
- Ajouter à `.gitignore` : `logs/`, `checkpoints/`, `*.pt`, `*.pth`, `third_party/*/` (sauf les fichiers d'épinglage), `.venv*/`.
- Installer uv (après confirmation), puis créer les environnements :
  - `.venv` : Python 3.12, torch/torchvision cu128, `-e .[dev,timm,exr,adapt]`, plus `opencv-python`, `scikit-image`, `einops` pour le shadow ;
  - `.venv-zoe` : `timm==0.6.12` pour ZoeDepth.
- `symtrs doctor --data-root <dataset>` est la référence avant chaque étape.

### Étape 0b — Tout pré-télécharger MAINTENANT, tant que la connexion est bonne (priorité 1)
Rien n'a encore été téléchargé : le mode plan interdit toute installation. Dès l'approbation, je lance ceci **avant tout le reste**, en parallèle de l'étape 1.

À télécharger :
- uv et Python 3.12 ;
- les wheels torch/torchvision cu128 (~3 GB) et toutes les dépendances pip des deux venvs ;
- le clone des 3 dépôts ombre dans `third_party/` et leurs dépendances.

Poids, dans un cache local au repo (`.cache/` gitignoré, avec `TORCH_HOME`, `HF_HOME` et `TIMM` pointés dessus) :
- MiDaS ×5 (torch.hub) ;
- ZoeDepth ×3 ;
- DA3 ×2 (HF) ;
- Depth Pro ;
- les backbones ImageNet de SegFormer (mit_b0–b2) et DeepLabV3 r50/r101 ;
- le VGG19 (perte perceptuelle de SRGAN) ;
- l'Inception (FID/KID du domain adapt) ;
- les éventuels poids pré-entraînés de Restormer, Uformer et ShadowFormer.

Ensuite :
- vérification par `symtrs doctor --model <key> --build` pour chaque clé ;
- un script `scripts/prefetch_weights.py` rejouable, qui ne télécharge que ce qui manque ;
- les runs se lancent avec `HF_HUB_OFFLINE=1` pour qu'un modèle oublié échoue vite au lieu de bloquer.

Volume estimé : ~15–25 GB. Durée : selon le débit, à faire tourner tout de suite.

### Reprise après interruption (si la session Claude est coupée)
- **Lanceur détaché** : les runs longues partent via un script `scripts/run_queue.ps1` démarré par `Start-Process` (processus indépendant). Elles continuent même si Claude est fermé ; seule l'extinction du PC les arrête.
- **File idempotente** : `runs/queue.txt` liste les configs dans l'ordre. Le lanceur saute toute run dont `summary.json` existe déjà, et reprend une run interrompue depuis `weights/last.pt` (`train.resume`). Le relancer est toujours sans risque.
- **État écrit sur disque** : chaque début et fin de run est ajouté à `docs/claude/PROGRESS.md` et à `logs/queue_<date>.log`. Une nouvelle session Claude relit ces fichiers et reprend sans contexte.
- **Arrêt propre** : un fichier `runs/STOP` fait s'arrêter le lanceur après la run en cours.

### Étape 1 — Adaptation du data loader (sous-agent Sonnet A)
- Corriger D1 à D6 en respectant AGENTS.md :
  - les défauts restent uniquement dans `config.py` ;
  - les messages d'erreur indiquent le correctif ;
  - les tests vérifient la raison, sans torch quand c'est possible.
- Ajouter des tests :
  - `normalize_stem` sur les vrais motifs de nom ;
  - le décodage ×63 ;
  - le rôle `noshadow` limité à 2048 ;
  - le `depth_scale` en cm.
- Mettre `make-dummy` à jour pour produire `rgb_noshadow`, des noms préfixés et un masque ×63, afin que le dummy reflète la vraie disposition.
- Mettre à jour les tables de `generate_grid.py` (num_classes 5, depth_scale 0.01) puis relancer la génération.
- Validation :
  - `pytest -q` passe ;
  - `generate_grid.py --check` passe ;
  - `symtrs doctor --data-root` affiche une couverture à 100 %.

### Étape 2 — Portage de view_dataset (sous-agent Sonnet B, en parallèle de l'étape 1)
- Porter vers `symtrs/commands/view.py` (`symtrs view <file> [--res]`) en réutilisant `symtrs/data/exr.py` et `splits.py`, sans dupliquer la lecture EXR.
- Ajouter le panneau no-shadow, et un mode `--save out.png` pour l'usage headless.
- Mettre à jour le notebook `data_loading_demo.ipynb` ou en ajouter un nouveau.
- Ajouter un test sur la résolution des fichiers frères.

### Étape 3 — Tâche `shadow_removal` (sous-agent Sonnet C, après l'étape 1)
- Cloner les trois dépôts dans `third_party/{ShadowFormer,Restormer,Uformer}` à un commit épinglé, noté dans `third_party/PINS.md`. Comme pour CycleGAN, toute modification est un patch minimal documenté.
- Données d'entrée et cible : `rgb_day` (2048) → `rgb_noshadow`. Le staging suit la disposition attendue par chaque dépôt (style ISTD `train_A`/`train_C`) avec des liens nommés sous la clé normalisée.
- **Masque d'ombre** : ShadowFormer en a besoin (`train_B`). On le dérive par seuil sur le ratio de luminance `noshadow/day`, puis ouverture morphologique. Il est généré au staging et mis en cache dans `<run_dir>/staging`, jamais dans le dataset. Restormer et Uformer n'en ont pas besoin.
- Entraînement par patchs aléatoires de 256 à 512. Inférence 2048 tuilée : réutiliser `engine/tiling.py`.
- Métriques communes calculées par symtrs (une seule implémentation) :
  - PSNR, SSIM, RMSE en LAB ;
  - les mêmes sur la zone ombrée et la zone non ombrée (protocole ISTD).
- Écriture dans `summary.json`.
- Registre : `shadowformer`, `restormer`, `uformer`. Ajouter `TASKS`, la branche de routage dans `trainer.py` (train et evaluate_run), une table et un rendu dans `generate_grid.py`, une ligne dans AGENTS.md §1 et dans `configs/README.md`.
- Tests : le parser upstream accepte les flags passés (même modèle que `tests/test_domain_commands.py`), et `resolution ≠ 2048` est refusé.

### Étape 4 — Smoke tests (moi, puis sous-agents par tâche)
- Unreal Editor doit être **fermé** pendant toutes les runs GPU.
- Pour chaque config : `--dry-run`, puis `--epochs 1 --limit-train-batches 4 --limit-val-batches 2` sur le vrai dataset.
- Le temps par itération est relevé pour calibrer la suite.
- Ordre : un modèle par famille d'abord (unet_small, midas_small, srcnn, cyclegan, restormer), puis tous les modèles.

### Étape 5 — Runs courtes de validation sur le vrai dataset (arrière-plan, `logs/<modele>_<date>.log`)
**But : vérifier que chaque modèle s'entraîne, s'évalue et écrit `summary.json` sur les vraies données. On ne cherche pas la convergence.**

Budget par run, passé en flags CLI (`--epochs`, `--limit-*`, `upstream_args`) sans modifier les configs générées :
- seg, depth et SR : **2 époques** ;
- CycleGAN/pix2pix : `--n_epochs 1 --n_epochs_decay 1` ;
- shadow : ~**500 itérations**.

Chaque run évalue ensuite sur val et sur test. Les configs complètes restent intactes pour le futur dataset de 20k transforms.

Ordre, du plus rapide et le plus sûr au plus long :
1. **Segmentation** (24 runs)
2. **Profondeur** (44 runs). Les relatifs utilisent silog + align scale, les métriques berhu + none, en s'appuyant sur le registre.
3. **SR** (16 runs)
4. **Suppression d'ombres** (3 runs)
5. **Domain adaptation** (4 runs)
6. Optionnel : **pix2pix et CycleGAN ombre ⇄ sans ombre**, en réutilisant le front-end domain avec la paire `rgb_day`/`rgb_noshadow`. C'est une variante de config, pas une nouvelle implémentation.

Pour chaque étape :
- lancer `symtrs sweep --execute --jobs 1` en arrière-plan ;
- surveiller les logs ;
- reporter les résultats de `symtrs tables` dans `RESULTS.md`, avec la mention « dataset preview 144 transforms ».

### Estimation de durée (runs courtes, RTX 5070 Ti Laptop, à recalibrer après les smoke tests)
L'évaluation tuilée à 2048 sur val et test (48 images) domine souvent le temps d'entraînement.

| Bloc | Runs | Par run | Total |
|---|---|---|---|
| Seg (U-Net, DeepLab, SegFormer) | 24 | 2–6 min | ~1–2 h |
| Depth petits (MiDaS small/hybrid/swin2, ZoeDepth) | 28 | 3–8 min | ~1,5–3 h |
| Depth grands (DPT-L, BEiT-L, DA3, Depth Pro) | 16 | 8–20 min | ~2–5 h |
| SR (SRCNN/AE/SRGAN/SwinIR) | 16 | 3–15 min | ~1–3 h |
| Ombres (ShadowFormer, Restormer, Uformer) | 3 | 15–40 min | ~1–2 h |
| Domain adapt (CycleGAN/pix2pix, 2 époques, FID/KID) | 4 | 10–20 min | ~1 h |
| **Total** | ~91 | | **~8–16 h de GPU** (une nuit) |

On ajoute aussi le premier téléchargement des poids pré-entraînés (DPT-L, BEiT-L, DA3, Depth Pro : plusieurs GB).

### Répartition entre sous-agents (Sonnet 5.5 : `model: "sonnet"` explicite sur chaque appel Agent)
| Agent | Mission | Dépend de |
|---|---|---|
| A | Data loader D1–D6, tests, régénération des configs | Étape 0 |
| B | Portage de view_dataset | Étape 0 (en parallèle de A) |
| C | Tâche shadow_removal et third_party | A |
| D | Smoke tests seg + depth, rapport des temps et des erreurs | A |
| E | Smoke tests SR + domain + shadow | A, C |

Mon rôle :
- orchestration et revue des diffs ;
- lancement des runs longues ;
- tenue de `docs/claude/*.md`.

### Risques
- **VRAM** : Unreal ouvert = OOM. Les modèles ViT-L à 2048 demandent batch 1, AMP et du tiling. DeepLab demande batch ≥ 2.
- **Blackwell sm_120** : il faut torch cu128. Les extensions CUDA compilées des dépôts tiers ne seront pas disponibles sans nvcc. Uformer et Restormer sont en PyTorch pur ; il faudra vérifier ShadowFormer (pas d'extension a priori). Si besoin : `pip install` des wheels ou un patch.
- **Windows** : symlinks du staging (mode Développeur, ou repli hardlink/copie), `num_workers` avec spawn, chemins contenant des espaces (le dossier du projet Unreal, `<UNREAL_PROJECT>`).
- **Dataset minuscule** (96 images) et 2 époques : les chiffres valident seulement la chaîne de bout en bout. Ils seront marqués « validation run, preview dataset, 2 epochs » dans `RESULTS.md` et ne seront jamais présentés comme résultats, conformément à AGENTS.md.
- **Dépendances anciennes** des dépôts ombre : ShadowFormer et Uformer épinglent d'anciens timm/torch. On isole dans une venv dédiée si le conflit est réel.
- **Masque d'ombre dérivé** : le seuil influence ShadowFormer et les métriques par zone. Seuil documenté et test de sensibilité dans `SUGGESTIONS.md`.
- **Le dataset va changer** (passage à 20k transforms) : tout passe par le data-root et la config, rien n'est codé en dur.
- **Écriture interdite** dans `<UNREAL_PROJECT>` : le staging et les caches vont uniquement dans `runs/` du repo.

## Fichiers critiques
- `symtrs/data/io.py`, `splits.py`, `semantic_dataset.py`, `exr.py`, `symtrs/config.py`
- `symtrs/commands/make_dummy.py`, `configs/generate_grid.py`, `symtrs/registry.py`, `symtrs/engine/trainer.py`
- nouveau : `symtrs/model/shadow/`, `symtrs/commands/view.py`, `third_party/`, `docs/claude/`
- `.gitignore`, AGENTS.md §1, `configs/README.md`

## Vérification
- `python -m pytest -q` passe, avec de nouveaux tests pour D1–D6, la tâche shadow et view.
- `python configs/generate_grid.py --check` passe.
- `symtrs doctor --data-root <dataset>` : 100 % de couverture, `depth_scale` cohérent.
- Le smoke test de chaque config se termine et écrit `summary.json`.
- `symtrs tables runs --task <t>` produit les tableaux reportés dans `RESULTS.md`.
- Avant la PR :
  - `gh auth status` ;
  - `git status` sans aucun `.pt`, log ou donnée ;
  - nouvelle branche depuis `features`, push, puis PR vers `features`.
