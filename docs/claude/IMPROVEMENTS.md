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
- Correctifs issus du smoke test (2026-10-17) :
  - DA3 (`da3_large`, `da3_metric_large`) à 1024 : `patch_size` 0 → 512 (table `DEPTH_MODELS`
    de `configs/generate_grid.py`, configs régénérées). Fine-tuning complet d'un ViT-L de 411 M
    de paramètres : ~6,5 Go de poids + gradients + états Adam résidents, et une image 1024 vaut 4×
    les tokens d'une 512 → OOM sur 12 Go même à batch 1 ; le crop 512 (déjà utilisé à 2048,
    passe) tient. À 512 et 256 la valeur reste 0 : le crop 512 y est l'image entière (un crop ne
    dépasse jamais l'image, rien n'est sur-échantillonné — règle 1). Test :
    `test_da3_crops_to_512_wherever_the_image_is_larger_than_that`. Note : les batchs de la grille
    (4/8/16 à 512 pour DA3) restent calibrés pour une carte plus grosse ; `smoke_all.py` mesure
    le batch qui tient.
  - `midas_dpt_beit_large` routé vers `.venv-zoe` (timm 0.6.12) comme swin2 : `backbones/beit.py`
    de MiDaS remplace `Block.forward` par une copie qui appelle `self.drop_path`, que timm >= 0.9
    a scindé en `drop_path1/2` → `AttributeError` au premier forward. Ajouté à `TIMM_PINNED_KEYS`
    (doctor), `ZOE_VENV_MODELS` (smoke_all), docstring `run_queue.py`, règle 10 d'AGENTS.md,
    README. L'adaptateur MiDaS lève désormais à la construction (`_require_beit_compatible_timm`,
    test de l'attribut `drop_path` plutôt que de la version) une `RuntimeError` nommant `.venv-zoe`.
    Code du hub MiDaS non touché. Vérifié sur CPU : `.venv-zoe`, `DPT_BEiT_L_384` se construit et
    un forward 1×3×384×384 donne 1×1×384×384 ; transform `dpt_transform` = 384, cohérent.
    Les autres MiDaS (small, DPT-Hybrid, DPT-Large) ne patchent que le forward du ViT de tête
    (`forward_flex`), jamais un `Block` timm : OK sous timm 1.0 (smoke ok à 1024/2048).
    Tests : 3 ajoutés dans `tests/test_offline_build_fixes.py`. Suite : 524 passed.

## 2026-10-17 — Depth Pro tient sur 12 Go : `train.freeze` n'était jamais appliqué

- **Cause racine** : `train.freeze` était lu par `build_from_config` et stocké par chaque
  adaptateur (`self.freeze`), mais seules `trainable_parameters()` / `apply_freeze()` appliquaient
  la politique, et rien sur le chemin d'entraînement ne les appelait (`build_optimizer` parcourt
  `named_parameters()` directement). Le log disait pourtant `freeze='encoder' -> 952.0M trainable`
  (= tout). Résultat pour Depth Pro : les 35 passes ViT-L (5×5 + 3×3 + 1 patchs de 384 px) étaient
  enregistrées dans le graphe autograd → OOM > 11 GiB même à batch 1, quel que soit le crop.
  Le crop ne pouvait pas aider : `DepthPro.forward` impose 1536×1536 (`assert H == W == img_size`,
  embedding de position 384 px × 4 fixé par l'architecture) et `infer` y redimensionne toute entrée.
- **Correctif** : `symtrs.engine.trainer.apply_run_freeze(model, cfg)`, appelée dans `_run` avant
  `model.to()`, DDP et `build_optimizer` (DDP fige l'ensemble des paramètres à `requires_grad` au
  moment du wrap). Lève une `RuntimeError` nommant `symtrs doctor --model <clé> --freeze <politique>`
  si la politique ne laisse rien d'entraînable. Pas appelée dans `evaluate_run` (pas de gradient).
  Un encodeur gelé n'a plus besoin de `no_grad` explicite : aucune entrée ni aucun paramètre en amont
  ne demande de gradient, donc autograd n'enregistre rien (vérifié : 7,9 GiB mesurés).
- **Effet de bord corrigé** : MiDaS range tout son réseau sous `backbone.` (ViT dans
  `backbone.pretrained`, décodeur dans `backbone.scratch`) ; le préfixe par défaut `backbone`
  gelait donc aussi le décodeur → 0 paramètre entraînable dès que la politique devenait réelle.
  `apply_freeze_policy` lit maintenant l'attribut de classe `freeze_encoder_prefixes` ;
  `MiDaSAdapter.freeze_encoder_prefixes = ("backbone.pretrained",)` → DPT-Large : 16,5 M entraînables
  sur 344 M. ZoeDepth n'est pas touché (préfixe `.pretrained` correct ; son noyau MiDaS est de toute
  façon déjà gelé à la construction : 0,48 M entraînables).
  **Conséquence** : les configs MiDaS DPT-Large / BEiT-L et ZoeDepth avec `freeze: encoder`
  s'entraînaient jusqu'ici en fine-tuning complet ; les smoke tests / mesures de batch faits avant ce
  correctif pour ces modèles sont à refaire (leur mémoire baisse, leur batch maximal monte).
- **Mesures** (RTX 5070 Ti Laptop 12 Go, amp fp16, `torch.cuda.max_memory_allocated`, 1 pas fwd+bwd
  (+ AdamW) à batch 1) :
  | cas | pic |
  |---|---|
  | avant : encodeur non gelé, crop 512 | OOM à 11,1 GiB |
  | après : encodeur gelé, crop 512 | 7,82 GiB |
  | après : encodeur gelé, image 2048 entière (pas de crop) | 7,94 GiB |
  | évaluation 2048 native, tile_size 0, batch 1, amp | 5,74 GiB |
  Poids fp32 : 3,6 GiB ; entraînables : 20,5 M (décodeur 19,7 + tête 0,4 + tête FOV 0,4).
  Smoke bout en bout via le trainer (`depth_pro_1024.yaml`, 2 pas, val + test) : OK avec
  `SYMTRS_CUDA_MEM_FRACTION=0.8` ; à 0.7 (8,3 GiB) il échoue de justesse (7,3 GiB alloués +
  ~1 GiB de fragmentation) — le plafond 0.92 de `run_env` laisse la marge.
  AMP : fp16 + GradScaler fonctionne (perte finie) ; ni bf16 ni checkpointing du décodeur n'ont été
  nécessaires (7,9 GiB < 10,5 GiB).
- **Configs** (`configs/generate_grid.py`, ligne `depth_pro`) : batch 1 à toutes les résolutions
  (avant 1/2/4/8 : le batch 2 ne tient pas), `patch_size: 0` partout (un crop 512 coûte autant que
  l'image entière, est sur-échantillonné ×3 vers 1536 à l'entraînement alors que l'évaluation
  sous-échantillonne : décalage train/test), `grad_accum: 8` inchangé → batch effectif 8 uniforme.
  Texte de la note régénéré avec ces mesures.
- Tests : `tests/test_depth_pro_memory.py` (6) — politique appliquée sur la disposition de
  modules de Depth Pro (seuls décodeur/tête/tête FOV entraînables), politique vide inerte, erreur
  explicite si tout est gelé, ordre `apply_run_freeze` < `model.to` < DDP < `build_optimizer` dans
  `_run`, configs Depth Pro (batch 1, pas de crop, `freeze: encoder`, amp, batch effectif 8),
  MiDaS garde son décodeur entraînable.
- **Recommandation validation** : Depth Pro est fine-tunable sur 12 Go, pas besoin de repli
  zero-shot. Lancer depuis la racine du dépôt (checkpoint `./checkpoints/depth_pro.pt`) avec
  `SYMTRS_CUDA_MEM_FRACTION=0.92`, `--batch-size 1` (valeur des configs ; ne pas laisser
  `smoke_all` essayer 2). Le zero-shot reste disponible :
  `symtrs eval --task depth --model depth_pro --resolution 2048 --data-root <root> --eval-split test`.

## Protocole d'évaluation (depth + segmentation) — 2026-10-17

Audit smoke (`docs/claude/AUDIT_smoke.md` A1, A4, A5) : trois erreurs de protocole corrigées côté
évaluation, plus deux défauts découverts en chemin. Aucune de ces corrections ne vise à améliorer
un chiffre : elles changent ce que le chiffre mesure.

- **Val et test au 2048 étaient notés sur UN crop aléatoire de 512** (`factory.from_config` :
  `val_patch_size or train.patch_size`, et les datasets tiraient un crop aléatoire même hors
  entraînement). Conséquences : test non reproductible, non comparable aux lignes 1024/512/256
  (image entière), `best.pt` choisi sur 2 fenêtres. Correctif :
  - `from_config` : l'évaluation n'hérite plus jamais de `train.patch_size` ; `val_patch_size: 0`
    = image entière (`symtrs/config.py` documente la sémantique) ; une valeur > 0 est un crop
    **central déterministe** (`transforms.center_crop_params`), jamais aléatoire.
  - `DepthTaskDataset` / `SemanticDataset` : crop aléatoire uniquement si `augment` (donc
    entraînement), crop central sinon.
- **Politique de tuiles** (`configs/generate_grid.py::eval_tiling`) : une ligne qui s'entraîne sur
  des crops plus petits que l'image est évaluée, val et test, sur l'image ENTIÈRE en fenêtres
  glissantes de la taille du patch d'entraînement (`eval.tile_size = patch`, recouvrement = patch/4,
  fondu de Hann d'`engine/tiling.py`) : le réseau voit le champ de vue et la GSD de l'entraînement,
  la mémoire reste celle d'un pas d'entraînement, chaque pixel est couvert. Une ligne entraînée sur
  image entière est évaluée entière (`tile_size: 0`). Concrètement : 2048 patch 512 → tuiles
  512/128 (25 par image : U-Net, DeepLabV3, SegFormer, DA3, ZoeD-K/NK, ZoeD-N) ; 2048 patch 384 →
  384/96 (49 : MiDaS small, DPT-Hybrid, DPT-Large, Swin2-tiny, BEiT-L) ; 1024 patch 512 → 512/128
  (9 : segmentation, DA3, MiDaS DPT-Large et BEiT-L) ; tous les autres, dont Depth Pro, `0`.
  Val et test passent par la même fonction `evaluate` et lisent le même `cfg.eval`.
- **Tuilage et profondeur relative** : l'alignement (`align_prediction`) était déjà calculé une
  fois par image complète, après recollement — correct. Mais chaque passe d'un modèle relatif
  (disparité MiDaS, profondeur relative DA3) a sa propre échelle ET son propre décalage : mélanger
  les tuiles brutes laisse des marches aux coutures qu'aucun alignement global ne corrige.
  `predict_tiled(..., match_affine=True)` ajuste (moindres carrés pondérés) chaque nouvelle fenêtre
  sur la zone déjà recollée avant le mélange ; `forward_native` l'active seulement si
  `model.protocol == "relative"` (profondeur métrique et logits sont absolus : jamais ajustés).
- **Remplissage rogné, plus étiré** (`loop._restore_size`) : quand `prepare_input` ne fait que
  compléter à un multiple (DA3 : 512 → 518), la sortie est rognée au lieu d'être redimensionnée —
  le redimensionnement décalait chaque pixel de la GT (jusqu'à 6 px) sur les tuiles et l'image.
- **NaN n'est jamais « best »** (`trainer.is_improvement`) : la branche `best is None → True` passait
  avant le test NaN, donc un premier epoch NaN était sauvé en `best.pt` (rmse=NaN) et rien ne
  pouvait le battre. Une valeur non finie ne gagne jamais, même la première ; elle est signalée
  (avertissement par epoch, `non_finite_monitor_epochs` dans `summary.json`) ; si aucun epoch n'est
  fini, aucun `best.pt` n'est écrit, `best` vaut `null` et le message dit de traiter le run comme
  échoué (les chiffres test viennent alors des poids du dernier epoch).
- **JSON valide** : nouveau `symtrs/common/jsonio.py` (`json_safe`, `dumps`, `write_json`,
  `allow_nan=False` en garde-fou). Tous les `summary.json` (trainer, domaine, shadow, super-résolution)
  et `per_image_*.json` écrivent `null` pour NaN/inf. `metrics.csv` : cellule vide. `symtrs tables`
  lit `null` (et l'ancien `NaN`) et affiche `n/a` ; le tri place les valeurs manquantes en dernier ;
  la sortie `--format json` est valide.
- **mIoU** (`metrics/semantic.py`) : une classe d'union nulle (absente de la GT et des prédictions)
  est NaN → `null` dans `summary.json` (`iou_k`, `dice_k`) au lieu de `0.0`, et reste exclue de la
  moyenne ; une classe absente de la GT mais prédite garde son IoU 0 (union > 0, comportement
  standard). Nouveau champ `miou_classes` = nombre de classes moyennées. La classe 0 (fond) n'a
  aucun pixel dans aucun split de la capture actuelle : le mIoU est donc la moyenne des classes
  présentes, et deux mIoU de `miou_classes` différents ne sont pas comparables (note ajoutée dans
  chaque config de segmentation via `SEG_CLASS_NOTE`). L'ancien test qui exigeait `iou_2 == 0.0`
  pour une classe absente a été inversé (c'était précisément le défaut).
- **Tests** (`tests/test_eval_protocol.py`, 20) : l'évaluation n'hérite pas du crop (spy sur
  `dataloader`) ; crop d'éval = centre déterministe, crop d'entraînement toujours aléatoire (jeu
  de données factice, depth et segmentation) ; val et test utilisent la même fonction ; politique
  de tuiles (unitaire + vérifiée sur les 91 YAML générés) ; les tuiles relatives sont ajustées
  (l'erreur de couture est ≥ 5× plus faible qu'en mélange brut) et les sorties absolues non ;
  `forward_native` n'ajuste que `protocol == "relative"` ; rognage du remplissage ; NaN/inf jamais
  « best », même au premier epoch ; JSON valide (`parse_constant` qui échoue sur NaN) ; CSV ;
  `tables` avec `null`/`NaN` ; classe d'union nulle exclue, classe prédite-mais-absente comptée,
  `null` jusque dans le JSON.
- **Sonde GPU** (smoke ckpt non entraînés, split val 2048, 24 images, plafond 50 %, machine partagée
  avec le smoke en cours donc durées indicatives) :
  | modèle | fenêtres | pic alloué | durée |
  |---|---|---|---|
  | `unet_small` 2048 | 512/128 (25/img) | 0,65 GiB | 15 s |
  | `unet_small` 2048 | image entière | 5,21 GiB | 10 s |
  | `midas_small` 2048 | 384/96 (49/img) | 0,51 GiB | 39 s |
  | `midas_small` 2048 | image entière | 0,51 GiB | 16 s |
  Pour U-Net, tuilé et entier donnent le même mIoU (0,0958) à 4 décimales : le recollement ne
  change rien pour un réseau convolutif. MiDaS change (rmse 28,2 vs 29,9) : normal, il ne voit
  pas la même chose (tuile 384 native vs image réduite à 384), et c'est la tuile qui correspond
  à l'entraînement. Coût : ~1,6 s/image pour MiDaS-small, donc ~40 s par passe de validation de
  24 images — les gros MiDaS/BEiT-L en coûteront plus ; baisser `eval.tile_overlap` ou
  `val_patch_size` (crop central) si la validation par epoch devient le goulot.
- **Limite connue** : un run de smoke existant (`runs/smoke/*`) a été noté avec l'ancien protocole ;
  ses chiffres 2048 ne sont pas comparables à ceux produits après ce changement.

## 2026-10-17 - Conventions de sortie des modèles de profondeur (MiDaS, DA3-métrique, ZoeDepth)
- **MiDaS : alignement en espace disparité (règle 3 d'AGENTS.md réécrite).** MiDaS sort `s/z + t`
  (échelle ET décalage inconnus). Avant : `to_metres` inversait puis `eval.align: scale` ; `1/(s/z+t)`
  n'est pas proportionnel à `z` et une échelle ne retire pas `t`. Maintenant : nouvel alignement
  `scale_shift_disparity` (`metrics/depth.py`) = moindres carrés échelle+décalage sur la disparité brute
  contre `1/GT`, disparité ajustée avec plancher `1/data.max_depth` (1000 m ; `DEFAULT_MAX_DEPTH` si
  non fourni), puis inversion. L'espace de sortie est déclaré dans le registre (`ModelSpec.output_space`,
  `registry.depth_pairing(protocol, output_space)` = seule définition de (loss, align)) ; les adaptateurs
  refusent un `align` incohérent à la construction et dans `evaluate_depth`
  (`DepthAdapter.validate_alignment`) ; `DepthAdapter.scoring_input` fournit la disparité brute (et non
  les mètres) à ces alignements ; l'espace est écrit dans les métadonnées du checkpoint.
  Les configs MiDaS (5 modèles x 4 résolutions) passent à `align: scale_shift_disparity` + `loss: ssi`.
- **Perte `ssi`** (`model/depth/losses.py`) : L1 sur le résidu d'un ajustement échelle+décalage en forme
  fermée par image (différentiable), divisé par l'étalement de la cible -> sans dimension (~1 pour un
  plan constant, 0 si parfait ; sans cela la perte vaudrait ~1e-6 sur des disparités 1/70-1/270 et
  l'epsilon d'Adam étoufferait le gradient). Avant, `silog` sur la disparité ne pardonnait que
  l'échelle. Ni normalisation médiane/MAD ni terme de gradient multi-échelle de MiDaS (non reproduits ;
  `loss_kwargs.gradient_weight` ajoute un terme de bord).
- **Sonde réelle** (midas_small 256, 6 images test, GPU, ancien vs nouveau protocole sur les mêmes
  prédictions) : silog 47,9 -> 29,2 ; abs_rel 0,358 -> 0,239 ; rmse 56,5 -> 34,8 ; d1 0,39 -> 0,49.
  Test unitaire de la raison : sur une disparité `37/z + 0,4`, l'ancienne voie garde une erreur
  >1000x supérieure à la nouvelle (qui retrouve `z` exactement).
- **DA3-métrique : `focal_px` dérivé du champ de vision.** `DataConfig.camera_hfov_deg = 60.0`
  (config.py), `focal_px_for_resolution(res, fov) = (res/2)/tan(fov/2)` : 1773,6 px à 2048, 886,8 à
  1024, 443,4 à 512, 221,7 à 256. `build_from_config` (`_fill_camera_kwargs`) l'écrit dans
  `cfg.model_kwargs.focal_px` si `ModelSpec.focal_kwarg` le demande et s'il est vide : il suit donc la
  résolution automatiquement, ne dépend ni du crop ni des tuiles (même taille de pixel), est enregistré
  dans le checkpoint et `config.yaml` (donc `autoloader` le retrouve), et une valeur explicite l'emporte.
  L'adaptateur avertit si une instance métrique est construite sans focale. **HYPOTHÈSE À CONFIRMER :**
  FOV horizontal 60 deg = valeur par défaut de l'acteur de capture (`FieldOfView = 60.f`,
  SyMTRSDatasetCaptureActor.h:97), modifiable par niveau.
  **Sonde réelle :** avec la focale appliquée, le zero-shot reste ~25x trop petit (médiane prédite
  2-9 m contre 150 m de GT à 256/512/1024 ; abs_rel 0,95, d1 = 0). La focale était nécessaire
  (sans elle la sortie est encore ~3x plus petite) mais n'est pas la cause principale : DA3-métrique
  lit la vue nadir à 70-270 m comme une scène proche (écart de domaine). Seul le fine-tuning corrigera.
- **ZoeD-N : NaN sous fp16, confirmé et corrigé.** Sonde GPU (RTX 5070 Ti, 1 image test réelle, forward
  seul) : autocast fp16 -> ~10 % de pixels non finis à 512 et à 1024 (finis : 0,903 / 0,906) ; fp32 et
  bf16 entièrement finis et identiques à 3 chiffres (moyenne 1,674 m). ZoeD-K et ZoeD-NK finis dans
  les 3 modes. Correctif dans l'adaptateur (pas de nouvelle clé de config) :
  `ZoeDepthAdapter._safe_precision` exécute le variant N en bf16 (fp32 si le GPU ne le supporte pas)
  dans tout autocast CUDA et renvoie du float32 ; `amp: true` reste identique aux autres configs.
  Vérifié : perte `silog` finie (2,49) et gradients finis à l'entraînement ; évaluation 1024 avec
  `eval.amp: true` sur 3 images test : abs_rel 0,19, d1 0,55 (avant : « every valid pixel had a
  non-positive prediction »).
- **ZoeD-K/NK : le « plafond de bins » n'est vrai que pour K, et le hub se trompe de valeur.**
  Lu dans le code épinglé : `bin_centers_type` vaut `normed` pour K (bins bornés à `[min,max]_depth`)
  mais `softplus` pour N et NK (`SeedBinRegressorUnnormed` : « max_depth: Not used »), donc N et NK
  ne sont PAS plafonnés (leurs 1-4 m sont une erreur d'appréciation, pas un plafond). Pire : l'entrée
  hub `ZoeD_K` appelle `get_config(..., config_version="kitti")` sans `dataset="kitti"`, donc le
  `max_depth: 80` n'est jamais fusionné et la tête est construite avec 10 m (défaut de la classe) alors
  que les poids ont été entraînés à 80. Correctif : `ZoeDepthAdapter(max_depth=...)`, défaut
  `ZOED_K_TRAINED_MAX_DEPTH = 80` (zero-shot fidèle ; la clé est aussi dans les `default_kwargs` du
  registre, avec un test d'égalité), refusé (`ValueError`) pour N/NK qui n'ont pas de plafond. Les
  configs `zoedepth_k_*` mettent `model_kwargs.max_depth: 300.0` (couvre le max de 279 m ; les
  paramètres gardent leurs formes, la disposition normalisée des bins est simplement étirée : point de
  départ, pas calibration). Sonde réelle (zoedepth_k 512, 3 images test, médiane GT ~152 m) :
  max_depth 10 (hub) -> pred. médiane 1,4-2,8 m, abs_rel 0,98 ; 80 -> 11-22 m, abs_rel 0,88 ;
  300 -> 41-84 m, abs_rel 0,53 ; silog identique (27,4) car la plage ne fait qu'étirer la sortie. Le
  zero-shot de K doit donc utiliser le défaut 80, pas la valeur de la config d'entraînement.
- Docstring de `zoedepth.py` corrigé : ZoeD-N est le modèle métrique NYU (ZoeD_M12_N), noté ici en
  relatif par choix de protocole (mètres intérieurs non transférables), pas « sans supervision métrique ».
- **Tests** : `tests/test_depth_conventions.py` (29) : appariement par (protocole, espace), retour exact
  de `z` depuis une disparité affine, l'ancienne voie échoue, plancher/clamp, refus d'alignements
  incohérents (messages vérifiés), `scoring_input` garde le signe, perte `ssi` (invariance
  échelle+décalage, sans dimension, gradient fini, prédiction constante), focale par résolution /
  indépendante du crop et des tuiles / FOV seul réglage / valeur explicite prioritaire, échelle
  `focal/300` de DA3, plage K par défaut = 80, refus de `max_depth` pour N/NK, N seul exclu de fp16,
  bf16 effectif sur CUDA. Tests de grille et de registre réécrits pour imposer l'appariement
  `depth_pairing` à chaque résolution et MiDaS en `ssi`/`scale_shift_disparity` partout. 581 tests
  verts, `generate_grid.py --check` OK.
- **Conséquence pour l'existant** : les `config.yaml` des smoke runs MiDaS (`align: scale`) sont
  maintenant refusés à la construction (message explicite) ; relancer depuis les configs régénérées.
  Les chiffres MiDaS des anciens runs ne sont pas comparables.
- **Non fait / à noter** : l'aperçu depth trace toujours la prédiction non alignée (`to_metres`), donc
  pour MiDaS une carte en 1/disparité sans échelle ; Depth Pro garde `f_px: null` (il estime son propre
  FOV) : `focal_kwarg="f_px"` dans le registre suffirait à lui donner la même focale dérivée si l'on
  veut ancrer cette ligne de base (cela change la nature du test, à décider).

## 2026-10-17 - Translation ombre ⇄ sans ombre via le front-end domain (objectif optionnel 3)

- **La paire est une donnée.** `upstream.DIRECTION_ROLES` (`symtrs/model/domain/upstream.py`) donne, par
  direction, les dossiers A/B et le sens upstream : `day2night`/`night2day` = `rgb_day`/`rgb_night`,
  `shadow2noshadow`/`noshadow2shadow` = `rgb_day`/`rgb_noshadow` (A est toujours `rgb_day`). Staging
  (`build_staging_dir(..., direction=)`), `upstream_direction`, cible réelle du FID/KID (`target_role`) et
  côté généré (`fake_A`/`fake_B`) en dérivent ; plus aucun `== "day2night"` en dur. Le comportement
  jour/nuit est inchangé (configs et tests existants identiques ; les 4 YAML existants régénérés à
  l'octet près). Toujours 2048 seulement (refus explicite sinon, message cite `rgb_noshadow`).
- **4 configs générées** (`cyclegan|pix2pix` x `shadow2noshadow|noshadow2shadow`) depuis les tables de
  `configs/generate_grid.py` (`DOMAIN_DIRECTIONS`, `DOMAIN_FOLDERS`, `DomainModel.shadow_note`) :
  95 fichiers au total (`--check` OK). `tests/test_vendored_and_grid.py` compare la table du générateur
  (qui ne peut pas importer le paquet) à `DIRECTION_ROLES`.
- **Notation appariée** pour `shadow2noshadow` uniquement : `runner.score_shadow_removal` réutilise tel quel
  `stage_split` (dérivation du masque), `score_split` et `symtrs.metrics.shadow` du `shadow_removal`
  (PSNR/SSIM/erreur LAB sur tout/ombre/hors ombre, mêmes colonnes) en plus de FID/KID ; les chiffres sont
  donc comparables aux lignes ShadowFormer/Restormer/Uformer. Taille : le front-end ne passe aucun
  `--preprocess`/`--load_size` (inchangé) ; `test.py` écrit donc du 256x256 (image entière redimensionnée),
  ré-échantillonné en bicubique à la taille de la vérité terrain avant le score. La perte de détail
  compte contre le modèle ; la note est écrite dans `summary.json` (`paired_scoring`) et dans l'en-tête
  de la config. `noshadow2shadow` : FID/KID seulement (une PSNR « zone d'ombre » mesurerait la qualité
  d'un ombrage, pas d'un retrait).
- **Bug pré-existant corrigé : pix2pix ne pouvait pas démarrer.** L'`AlignedDataset` vendu ignore
  `trainA/trainB` et lit `<dataroot>/<phase>/*.png` comme des images `[A | B]` côte à côte ; sans cela il
  tombe sur son auto-split `hr/`/`night/` et plante (`FileNotFoundError`). Aucun run pix2pix jour/nuit n'a
  donc jamais pu tourner. `build_staging_dir(aligned=True)` (pix2pix) écrit maintenant ces paires
  (`write_aligned_pairs`, format `combine_A_and_B.py` d'upstream, réutilisées si plus récentes que leurs
  sources, périmées supprimées). Le raw-dir + `--train_ratio` aurait perdu des images (somme < 1 exigée).
  Coût : un PNG sans perte par scène dans `<run>/staging/<split>/<split>/`.
- `symtrs tables` : colonne `direction` (4 lignes par modèle sinon indiscernables) et colonnes PSNR/SSIM/LAB
  dans le groupe `domain_adapt`. `sweep` : message natif-seul cite `rgb_noshadow` (le plan ne varie
  toujours pas la direction : passer `--extra model_kwargs.direction=...` ET un `--out-dir` par direction).
  Docs : AGENTS.md §1/§2.5/arbre (95), `configs/README.md`, `README.md`, descriptions du registre.
- **Tests** (+15, 603 verts) : staging A=`rgb_day`/B=`rgb_noshadow` appariés par clé pour les deux sens,
  jour/nuit inchangé, table direction -> côté upstream/cible, argv des deux directions accepté par le
  parseur vendu, côté généré par direction, refus à 1024 (message), scoring apparié sur paire synthétique
  (sortie 16 px ré-échantillonnée ; ombre conservée -> PSNR ombre chute, hors ombre inchangé) et erreur
  nommée si des images manquent, paires `[A|B]` pix2pix (moitiés, nettoyage), colonnes de `tables`.
- **Smoke** (dummy 512 px, 6/2/2, `--epochs 1`, ~37 s chacun) : `pix2pix_shadow2noshadow` termine, FID/KID
  + PSNR 15,7 / SSIM 0,39 / mae_lab 37,4 (chiffres dummy sans sens) ; `cyclegan_noshadow2shadow` termine
  (FID/KID seuls). Lancement réel : `symtrs train configs/domain_adapt/pix2pix_shadow2noshadow.yaml
  --data-root <racine>` (idem `cyclegan_*`), à la suite de la file, mono-GPU, `train.epochs: 40`.

## 2026-10-17 - Comparabilité FID/KID, smoke batch, init des modèles d'ombre (REVIEW_night #6, #7, #8)

- **#6 FID/KID asymétrique corrigé** (`symtrs/metrics/domain.py`, `model/domain/runner.py`). Avant, les
  images réelles 2048 étaient réduites à 299 sans antialias (aliasées) et les générées 256 agrandies
  (lisses) : le FID mesurait en partie la netteté. Maintenant : la taille générée est lue dans les fichiers
  (`generated_image_size`, erreur nommée si tailles mixtes), les réelles sont redimensionnées à cette taille
  avec antialias (`match_resolution`, `extract_features(resize_to=...)`), puis un seul redimensionnement
  299 antialiasé identique pour les deux côtés (`inception_preprocess`). **KID devient le chiffre
  principal** (`PRIMARY_METRIC`, listé avant `fid` dans le dict, `summary["primary_metric"]`, colonnes
  `symtrs tables` : `kid, kid_std, fid, fid_rank_deficient`) : avec N=24 par split et 2048 dimensions, la
  covariance du FID est de rang <= N-1, donc biaisée ; le FID reste calculé, avec le drapeau
  `fid_rank_deficient` et une note (`metric_notes.fid`). Résumé : `feature_resize_h/w`. Les anciens FID ne
  sont pas comparables aux nouveaux.
- **#7 `scripts/smoke_all.py`** : diviser le batch par 2 double maintenant `--grad-accum` (depth et
  segmentation, seules tâches dont la boucle accumule) pour garder batch x accum = batch effectif de la
  grille ; sinon le batch effectif changeait en silence à LR constant. Pour superres / ombre / domaine
  (pas d'accumulation) le batch effectif diminue vraiment : la colonne `eff_batch` < `grid_eff_batch` et la
  ligne de queue (`# eff_batch=... (SHRUNK)`) le disent. `smoke.tsv` : 3 colonnes ajoutées à la fin
  (`accum`, `eff_batch`, `grid_eff_batch`) ; `load_rows` lit les anciennes lignes à 9 colonnes (accum
  inconnu = valeur de la config, la ligne de queue n'émet alors pas `--grad-accum`), `load_done` garde son
  ancienne forme. OOM pendant l'évaluation image entière (batch d'éval = 1) distingué d'un OOM
  d'entraînement via le traceback (`classify_oom` : frame la plus interne parmi train_one_epoch /
  _training_step contre evaluate* / validate_crops / predict) : `FAIL ... eval OOM` immédiat au lieu de
  descendre à 1. Sans traceback -> comportement ancien (on divise). Attention : les lignes `ok` déjà
  écrites par la passe en cours avec un batch réduit ont tourné à accum inchangé ; elles apparaissent
  `SHRUNK` dans la queue, à relancer avec `--only` si le batch effectif compte.
- **#8 Ombre** : `load_matching` lève une `RuntimeError` (fichier + compteurs) si moins de 90 % des tenseurs
  du modèle sont chargés (avant : 0 tenseur passait en silence, run de fait « from scratch »). Le résumé
  enregistre `init_weights`, `init_source_task` (ShadowFormer : retrait d'ombre ISTD+ ; Restormer : débruitage
  réel SIDD ; Uformer : débruitage SIDD, aucun des deux ne publie de poids de retrait d'ombre),
  `init_tensors{loaded,total}` et `metric_notes.ssim_y`. À écrire dans RESULTS à côté de chaque ligne.
  **SSIM renommé `ssim_y`** (luma Rec.709 seulement ; ShadowFormer moyenne par canal RGB, donc non
  comparable aux chiffres publiés) : clés `ssim_y`, `ssim_y_shadow`, `ssim_y_nonshadow` dans
  `metrics/shadow.py`, `CSV_FIELDS`, `tables` et les tests ; les anciens `summary.json`/`metrics.csv`
  gardent `ssim`.
- **Tests** : `tests/test_comparability.py` (15) : antialias (rayures hautes fréquences), taille générée lue
  et tailles mixtes refusées, KID avant FID et drapeau rang-déficient, note/clé principale cohérentes,
  `load_matching` (0 et 8/10 tenseurs refusés, plein accepté avec compteurs), source d'init par modèle,
  `ssim_y`, `halve` (batch effectif conservé), lecture des lignes `smoke.tsv` à 9 et 12 colonnes,
  attribution train/eval d'un OOM, lecture de `grad_accum`. Suite : 639 verts ; seul échec,
  `test_eval_robustness.py::test_set_coerces_an_optional_float_that_is_currently_none` (travail concurrent
  sur `config.py`/`--set`, hors de ce périmètre). `generate_grid.py --check` OK.

## 2026-10-17 - Revue : planning du LR, reprise, poids du test, robustesse (trainer / loop / table / config)

**Change la sémantique des runs depth et segmentation : les runs de validation déjà faits sont à refaire**
(le LR effectif était faux depuis 76dd299).

- **#1 Planning du LR** (`engine/trainer.py`, `engine/loop.py`) : `build_scheduler` était dimensionné en pas
  d'optimiseur (`epochs * steps_per_epoch`) mais `scheduler.step()` était appelé une fois par EPOCH : le LR
  restait sur les premiers pourcents de la rampe de warmup (un run de 2 epochs s'entraînait à 4-8 % du LR
  configuré, jamais de décroissance cosinus). Le pas du scheduler est maintenant fait dans
  `train_one_epoch(..., scheduler=)` juste après chaque `optimizer.step()` (y compris le dernier groupe
  d'accumulation partiel ; un pas sauté par le GradScaler fait quand même avancer le planning). La taille
  par epoch vient de `optimizer_steps_per_epoch(len(loader), grad_accum)` = plafond (5 batchs, accum 2 = 3
  pas ; l'ancien `//` en comptait 2). Le warmup est plafonné à la moitié du run (message affiché) : les runs
  de validation à 2 epochs avec `warmup_epochs: 2` finissaient sinon au pic sans jamais décroître ; tout run
  avec `warmup_epochs <= epochs/2` est inchangé. L'état du scheduler est sauvé après les pas, avec
  `scheduler_unit: "optimizer_step"` ; un ancien checkpoint (pas par epoch) n'est pas chargé tel quel mais le
  scheduler est repositionné à `start_epoch * steps_per_epoch`.
- **#2 Reprise et poids du test** : `best`, `best_epoch`, `history`, `non_finite_epochs` sont dans le
  payload de `last.pt`/`best.pt` et restaurés par `--resume` (repli sur `best.pt` voisin et `metrics.csv`
  pour les anciens checkpoints) : une époque pire après reprise n'écrase plus `best.pt`. La passe de test
  recharge `weights/best.pt` avant d'évaluer (`summary["test_weights"]` = `best`, `last` si aucune époque
  finie, `released` en zero-shot) ; comme pour l'ombre, `test` correspond donc à `best`.
- **#9** : l'ajustement d'alignement de `evaluate_depth` est dans le `try` : une image avec < 2 pixels
  valides est sautée et comptée (`skipped_images`) au lieu de planter l'évaluation ; `append_csv_row`
  réécrit l'en-tête avec l'union des clés quand une colonne apparaît plus tard (`val_skipped_images`), les
  anciennes lignes ont une cellule vide ; `--set data.max_depth=1000` donne un float (type lu dans
  l'annotation du dataclass pour un champ `Optional` valant None ; `none`/`null` remet à None).
  La perte `silog` (`model/depth/losses.py`) clampait déjà la variance à 0 : rien à changer ; la métrique
  `silog` de `metrics/depth.py` (sqrt d'un petit négatif) est laissée à l'agent des métriques.
- **Tests** : `tests/test_trainer_schedule.py` (vrai `build_scheduler` + vraie `train_one_epoch` sur un
  petit modèle CPU : pic atteint après le warmup, décroissance jusqu'à `min_lr_ratio`, nb de pas scheduler =
  nb de pas optimiseur dont le groupe partiel, plafond du warmup, test sur `best` / repli `last`, reprise
  sans écrasement de `best.pt`, amélioration réelle prise après reprise, position du scheduler restaurée) et
  `tests/test_eval_robustness.py` (en-tête CSV, coercition `--set`, image à 1 pixel valide).

## 2026-10-17 - Métriques et raccord par tuiles : revue de nuit (points 3, 4, 5, 9)

- **mIoU : dénominateur fixé par les données (point 3).** `ConfusionMatrix.supported_classes()` = classes
  ayant au moins un pixel de vérité terrain dans la matrice de confusion de tout le split (jamais par image).
  `miou`, `mean_dice` et `mean_acc` moyennent sur cet ensemble, identique pour tous les modèles d'un même
  split. Avant, un seul pixel « classe 0 » prédit (0 n'existe dans aucun split, la tête a 5 sorties) donnait
  IoU_0 = 0 et faisait passer `miou_classes` de 4 à 5, soit environ -20 % relatif pour une erreur d'un pixel.
  `iou_k` reste NaN/null sans vérité ni prédiction, vaut 0 pour une classe prédite sans vérité (visible), mais
  n'entre pas dans la moyenne. Nouvelles clés numériques : `support_k` (pixels de vérité) et `fp_pixels_k`
  (pixels prédits k dont la vérité est une autre classe) ; les identifiants moyennés sont les `k` avec
  `support_k > 0` (clés numériques, car ce dict part tel quel dans le CSV). `format_report` affiche
  `fp_pixels`. L'en-tête des 24 configs de segmentation (régénérées, `--check` OK) décrit la nouvelle règle.
- **Raccord des tuiles par espace de sortie (point 4a).** `tiling.match_mode(model)` : disparité (MiDaS) ->
  `affine` (échelle + décalage) ; profondeur à une échelle près (DA3 relatif, ZoeD-N noté en relatif) ->
  `scale` (régression par l'origine, jamais de décalage : l'alignement final `scale` ne saurait l'enlever) ;
  profondeur métrique et logits -> `none`. Lu sur `protocol` + `output_space` de l'adaptateur (le test
  parcourt tout le registre). `predict_tiled(match=..., stats=...)` ; `match_affine=True` reste accepté
  (= `affine`). `forward_native(..., stats=)`.
- **Robustesse de l'ajustement (4b).** `fit_overlap` remplace `fit_affine` : seuil de variance **relatif**
  (`var <= 1e-6 * moyenne^2`, indépendant de l'unité), repli « décalage seul » (pente 1) si recouvrement plat,
  corrélation < 0,5 ou pente non positive (jamais de retournement), pente bornée à [0,25 ; 4] (décalage
  recalculé sur les moyennes), pente symétrique `std(ref)/std(src)` au lieu de l'OLS (biaisé vers 0 sous
  bruit). Compteurs `windows/matched/fallbacks/unmatched` ; `evaluate_depth` écrit `tile_match_fallbacks`,
  `tile_match_unmatched` et `tile_windows` (totaux du split, seulement si le tuilage a eu lieu).
- **Dérive d'ancrage (4c).** Vérifié : chaque fenêtre est ajustée sur le recouvrement avec le champ déjà
  assemblé (mélange de Hann de TOUTES les fenêtres posées, pas du seul voisin). L'ancrage reste la première
  fenêtre (ordre raster) : le repère global est une affine du vrai, ce que l'alignement par image retire.
  Documenté dans le module ; pas d'optimisation globale.
- **Prédictions non positives notées, plus supprimées (point 5).** `compute_depth_metrics` remplace toute
  prédiction <= 0 ou non finie (après alignement) par `min_depth` (défaut `DEFAULT_MIN_DEPTH` = 1e-3 m) au
  lieu de la retirer du masque, et renvoie `invalid_pred_fraction` (ajouté à `DEPTH_METRIC_NAMES`). Une image
  entièrement non positive n'est plus « ignorée » (`AllInvalidError`) : elle est mal notée, comme il se doit.
- **silog (point 9, partie métrique).** Variance bornée à >= 0 avant la racine (cas réel trouvé :
  -2,2e-16 -> NaN qui rendait l'époque non finie).
- **Effet sur les chiffres déjà produits.** mIoU : un modèle qui prédisait des pixels classe 0 voit son mIoU
  remonter (jusqu'à x1,25) ; ceux qui n'en prédisaient pas sont inchangés. Profondeur : les modèles ayant des
  pixels <= 0 voient RMSE/abs_rel augmenter (avant : optimistes) ; les autres sont inchangés. Raccord : MiDaS
  2048 en tuiles -> mêmes champs sur textures normales, plus robustes sur zones plates ; DA3 relatif / ZoeD-N
  ne reçoivent plus de décalage (changement réel pour eux à 2048). Tout run tuilé antérieur est à ré-évaluer.
- **Tests** : `test_segmentation_metrics.py` (dénominateur identique pour tous les modèles, union sur tout le
  split, classe fantôme visible), `test_eval_protocol.py` (mode de raccord par espace de sortie, registre
  complet), `test_tiling.py` (a,b aléatoires par tuile retrouvés, moitié plate sans explosion, pas de décalage
  en mode `scale`, seuil indépendant de l'unité, pente bornée, non-retournement, biais OLS), `test_depth_metrics.py`
  (prédiction négative/NaN/inf notée, `invalid_pred_fraction`, silog sans NaN sur un cas à variance -2e-16).
