# IMPROVEMENTS

Changements faits dans le repo, et pourquoi.

- `.gitignore` : `logs/`, `checkpoints/`, `*.pt`, `*.pth`, `*.ckpt`, `.venv-*/`, `third_party/*/`
  (les repos tiers sont reclonés aux révisions de `third_party/PINS.md`, pas commités).
- `symtrs view` (symtrs/commands/view.py) : port de Tools/view_dataset.py, lecture EXR via symtrs.data.exr, panneau no-shadow, `--save` headless, `--data-root/--split/--index`.
- D1–D6 (données réelles) : `normalize_stem` retire le préfixe `<modalité>_<light>_<res>_` ;
  `check_pairing` lève une erreur si 0 paire / >10 % non appariés ; décodage segmentation
  `round(v/label_step)` (63, 5 classes, hors grille → 255, `label_step` dans la checkpoint) ;
  `max_depth` (m, après `depth_scale` 0.01 cm→m) invalide le ciel ; rôle `noshadow` natif ;
  staging domain par clé normalisée + repli symlink→hardlink→copie (Windows) ; legacy SR apparie
  LR/HR préfixés ; `make-dummy` écrit désormais la disposition réelle ; doctor rapporte
  l'appariement, l'histogramme des gris et la profondeur en cm. Configs régénérées (69).
- Défauts `config.py` : `num_classes 5`, `label_step 63`, `depth_scale 0.01`.
- `tests/test_real_layout.py` (24 tests), `tests/test_view.py` (9) ; deux tests de `test_splits.py`
  rendus portables Windows (`/tmp`, séparateur `\`). Suite : 486 passed.
- SegFormer b0/b1/b2 (`model/semantic/segformer.py`) : l'encodeur MiT vient maintenant de
  `transformers.SegformerModel` (`nvidia/mit-b0|b1|b2`, `output_hidden_states` → 4 cartes de
  features). timm n'a jamais fourni MiT (`Unknown model (mit_b0)` sur toutes les versions). Tête
  MLP inchangée ; `pretrained=False` construit depuis une `SegformerConfig` de la bonne taille,
  sans réseau. `requires` → `transformers`, extra `segformer = ["transformers"]`, règle 10 de
  AGENTS.md et README corrigées (SegFormer ne contraint plus timm).
- Depth Anything 3 (`model/depth/da3.py`) : les checkpoints `depth-anything/DA3*` sont des graphes
  d'objets du paquet `depth_anything_3` (pas des modèles transformers : pas de preprocessor
  config, clés `model.backbone.*`). L'adaptateur construit `DepthAnything3Net` via
  `create_object(config.json)` et charge le safetensors (`strict=False` limité aux 6 clés
  `output_conv2_aux.{1,2,3}.2` absentes du checkpoint, comme le fait le paquet ; toute autre clé
  manquante lève). Prétraitement explicite = normalisation ImageNet de la base + `size_multiple=14`
  (pas de redimensionnement à 504). `focal_px` optionnel pour la profondeur métrique canonique
  (`× focal/300`). Installation : `pip install --no-deps depth-anything-3==0.1.1` (la roue impose
  numpy<2, xformers, open3d… inutiles ici) + extra `da3` (omegaconf, addict, safetensors).
- MiDaS Swin-V2-T : (1) ne charge que sous timm 0.6.12 (déplacement de `downsample.reduction`
  entre étages dans timm >= 0.9) → `.venv-zoe`, message d'erreur explicite dans l'adaptateur,
  `TIMM_PINNED_KEYS` dans `doctor` ; (2) bug trouvé au premier forward : la transformation
  `dpt_transform` (384) était associée à un checkpoint 256 ; mapping aligné sur le `hubconf`
  amont (`swin256_transform`, `swin384_transform`, `beit512_transform`).
- `.cache/torch/hub/isl-org_ZoeDepth_main` contenait 9 fichiers remplis de zéros (corruption après
  BSOD) → dépôt supprimé et re-téléchargé ; `.pyc` MiDaS nuls supprimés. `.venv-zoe` (CPython 3.10
  uv) importe torch + timm 0.6.12 correctement (CUDA disponible), pas besoin de python.org.
- `tests/test_offline_build_fixes.py` (12 tests). Suite : 516 passed.
- Garde mémoire CUDA : `SYMTRS_CUDA_MEM_FRACTION` (lu dans `symtrs/common/seed.resolve_device`) →
  `torch.cuda.set_per_process_memory_fraction`. Sous Windows/WDDM, un dépassement de VRAM déborde
  silencieusement en RAM (×100–500 plus lent) ; avec la garde (0.92, posée par `run_env`) on
  obtient un OOM franc en quelques secondes. Vérifié : U-Net bs32 → OOM en 6 s. Tests :
  `tests/test_memory_guard.py`.
- `scripts/run_queue.py` : `run_env()` / `run_job()` réutilisables ; au timeout, tue l'**arbre**
  de processus (`taskkill /T`), sinon les sous-processus (upstream CycleGAN, worker shadow)
  gardent le GPU.
- `scripts/smoke_all.py` : smoke test de chaque config sur le vrai dataset, recherche du batch
  qui tient (halving sur OOM), reprenable (`runs/smoke/smoke.tsv`), écrit `runs/queue.txt`
  avec le budget de validation (2 époques ; shadow 500 itérations) et le batch mesuré.
