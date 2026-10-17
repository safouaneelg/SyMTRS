# Audit smoke tests (2026-10-17, lecture seule, 49 runs au moment de l'audit)

## A. Problèmes confirmés
1. **Test (et val) du 2048 sont évalués sur un crop aléatoire de 512, pas sur l'image entière.**
   `symtrs/data/factory.py` (from_config): `patch_size = val_patch_size or train.patch_size`. Les configs 2048 ont
   `patch_size: 512, val_patch_size: 0`, donc 0 retombe sur 512. Le trainer (`engine/trainer.py:376-377`) construit val ET
   test avec `train=False`, et `sample_crop_params` (data/transforms.py:151) tire un crop aléatoire sans seed.
   Preuve : preview `da3_large_2048/preview/test_00.png` = 3 panneaux de 512 px. À 1024/512/256, `patch_size: 0`
   (image entière, ex. `midas_dpt_hybrid_1024/preview/test_00.png` = 1024 px). Conséquences : test 2048 non reproductible,
   non comparable aux autres résolutions, `best.pt` choisi sur 2 crops. Le commentaire des configs ("0 = image entière,
   comme les chiffres publiés") est faux. Fix : dans from_config, en `train=False` utiliser `val_patch_size` seul
   (0 = image entière) ; crop central déterministe si on veut borner la mémoire, ou `eval.tile_size`.
   Concerne aussi la segmentation 2048 (patch 512).
2. **Évaluation des modèles MiDaS : la disparité est inversée puis alignée par échelle seule.**
   `model/depth/base.py::to_metres` fait `1/disp`, puis `eval.align: scale` (metrics/depth.py:107). MiDaS produit
   `s/z + t` (échelle ET décalage inconnus) ; `1/(s/z+t)` n'est pas proportionnel à z, et une échelle ne corrige pas t.
   Sur des vues nadir à profondeur étroite (CV std/moy ~0,25) t domine. Indices : silog 41-111 pour MiDaS contre
   ~27 pour un plan constant ; align_scale_mean varie de 2e3 à 1,7e4 entre val et test du même modèle
   (hybrid_1024 : 1,28e4 vs 2,13e3). Les nombres MiDaS ne sont pas interprétables. Fix : aligner en espace disparité
   (scale+shift moindres carrés sur 1/GT), puis inverser et clamper ; garder `scale` pour les modèles qui sortent une
   profondeur (DA3, ZoeD-N). L'AGENTS.md règle 3 reste respectée (relative -> alignement) mais le *type* d'alignement
   doit dépendre de l'espace de sortie.
3. **DA3-metric : `focal_px: null` -> profondeur canonique non mise à l'échelle** (da3.py:156, 307-309).
   abs_rel 0,91-0,96, d1=0 : pred ~4 % de la GT. Nombre sans signification tant que focal_px n'est pas fixé (FOV de la
   caméra UE non documenté dans Docs/ ; à relever dans le blueprint de capture).
4. **ZoeD-N : sorties NaN sur GPU dans le run** (zoedepth_n_1024/2048 : « non-finite loss » aux 3 pas, val et test
   « every valid pixel had a non-positive prediction », test = `{skipped_images: 2.0}`). En fp32 sur CPU (testé, .venv-zoe,
   image test 256) la sortie est finie (0,90-2,36 m) donc le NaN vient de l'autocast fp16 (hypothèse, non testée sur GPU).
   Essayer `--set train.amp=false --set eval.amp=false`. De plus `best.pt` est enregistré avec rmse=NaN (trainer :
   `new best rmse=nan`) et `summary.json` contient `NaN` (JSON invalide) : refuser NaN comme « best ».
5. **mIoU segmentation : classe 0 absente des données.** Vérifié sur PNG réels (train/val/test, 2048 et 256) : valeurs
   seulement {63,126,189,252} -> classes 1-4 ; classe 0 = 0 % partout (doc : « rare after fill »). Décodage OK
   (`round(v/63)`). Mais `iou_0=0.0` est écrit à la place de NaN (metrics/semantic.py:`summary`) et, si le modèle prédit
   la classe 0, union>0 donc IoU 0 compté dans le mIoU (modèles non entraînés). Fix : exclure du mIoU les classes
   sans support GT, écrire null pour les absentes.

## B. Distribution des splits (explique val vs test, pas un bug d'unités)
Profondeur médiane par image (m, x8) : train 28-170 (méd. 92), val 70-160 (méd. 103), **test 137-265 (méd. 208)**
(caméras plus hautes : tours). Aucun pixel invalide, max 279 m (< max_depth 1000). Segmentation : vert 45 % train,
2 % val, 20 % test ; route 22/63/30 %. Splits par lieu avec peu de lieux -> décalage fort ; mIoU val/test peu comparables.
- **DA3-large 2048 : val rmse 1,43 vs test 31,7.** Unités correctes : DA3 relatif sort une profondeur d'échelle
  arbitraire, scale 75-138 pour 100-200 m est plausible. L'écart vient de 2 crops 512 aléatoires en val (probablement plats,
  d1=1,0) contre 24 crops en test, plus le décalage de profondeur. À 256 (image entière) : test abs_rel 0,31, val rmse 8,4.
- **ZoeD-K/NK** : abs_rel ~0,98 car la sortie plafonne (bins max 80 m K, 10 m NYU) et vaut ~1-4 m ici (CPU fp32 : moy.
  1,75 m) pour des scènes à 100-270 m ; encodeur gelé + 3 pas ne changent rien. Plafond structurel : réglage de la plage de
  bins nécessaire avant tout fine-tuning sérieux.
- ZoeD-N est en réalité un modèle **métrique** NYU (ZoeD_M12_N), pas « sans supervision métrique » comme le dit
  le docstring de `zoedepth.py` ; le traiter en relatif (align scale) est défendable mais l'étiquette est fausse.

## C. Plausible / à vérifier
- Les previews depth montrent le panneau « prédit » en couleur uniforme : il trace la prédiction *non alignée* (MiDaS ~1e-2,
  DA3 ~1), colormap calée sur la GT. Trompeur, pas un bug de métrique. Fix : tracer la prédiction alignée.
- `compute_depth_metrics` écarte silencieusement les pixels à prédiction <=0 après alignement (biais favorable) ; logguer
  la fraction écartée.
- Ajouter une ligne de référence « plan constant aligné par la médiane » (silog ~27 sur ce dataset) : plancher de sanité.
- `train.warmup_epochs: 1` avec 1 epoch : warning scheduler.step avant optimizer.step et lr 1e-5/5e-6 jamais appliqué au 1er pas.
- Smoke : 1 epoch/3 batches, les valeurs ne mesurent rien d'autre que la plomberie.

## D. Vérifié OK
- Unités : EXR en cm, `depth_scale 0.01` -> m ; max_depth non atteint. Profondeur le long de l'axe (doc) : cohérent.
- RGBA -> RGB : alpha = 255 partout (échantillon), `convert("RGB")` sans perte. Pairing par stem OK (24 val/24 test).
- Normalisation : `prepare_input` unique ; MiDaS via transform hub (384, uint8 round trip), Zoe identité (normalise en
  interne), DA3 ImageNet + multiple de 14. Retour à la taille de l'entrée dans `forward_native`.
- Métriques depth : alignement par image, métriques en m, `align none` pour métriques (scale_mean=1.0 confirmé),
  val et test passent par la même fonction `evaluate_depth` (même align, même tuilage) ; seule la taille de crop est suspecte (A1).
- Segmentation : matrice de confusion cumulée, ignore 255, GT des previews cohérente (bâtiments, route, voitures fines).
- Non encore audités (pas de run) : SR (LR/HR), domain adapt, shadow removal.
