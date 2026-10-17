# SUGGESTIONS

Pistes futures, non faites.

- Quand le dataset 20k transforms sera prêt : relancer la grille avec les époques des configs
  générées (les runs actuelles sont des runs de validation à 2 époques).
- `SyMTRS_Pipeline.md` se contredit (gris ×51 vs ×63, 6 vs 5 classes) : à corriger côté Docs.
- Sensibilité du seuil du masque d'ombre dérivé (ShadowFormer, métriques par zone).
- Inférence shadow removal tuilée : ~10 s/image 2048 (Restormer, tuiles 256 une par une, 0,6 GB VRAM) → regrouper les tuiles par batch accélérerait l'évaluation ~5–10×.
- Super-résolution : avec les crops, une époque = 1 crop par image d'entraînement (96 → 6 pas à batch 16).
  Avant les vraies runs, ajouter une option « crops par image et par époque » ou revoir époques/batchs
  de la grille (fixés quand l'entraînement se faisait sur l'image entière).
- `swinir_x1` : ~15 min par époque de smoke, presque tout en évaluation (masques recalculés pour chaque
  fenêtre 512 sur CPU) → mettre les masques en cache ou valider moins souvent.
- Lignes de référence systématiques dans RESULTS : « identité » pour shadow removal
  (`scripts/shadow_identity_baseline.py`, test PSNR 12,4 dB) et bicubique pour la SR (x2 22,9 / x4 20,9 /
  x8 19,3 dB mesurés par l'agent SR) ; sans elles un PSNR ne dit pas si le modèle fait mieux que rien.
- Profondeur métrique (Depth Pro, ZoeD-NK, DA3-métrique) : prédictions ~10x trop proches en zero-shot ;
  vérifier sur les runs de validation si 2 époques de fine-tuning déplacent l'échelle, sinon envisager
  un LR plus fort pour la tête ou un entraînement plus long.
