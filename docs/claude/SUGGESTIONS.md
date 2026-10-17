# SUGGESTIONS

Pistes futures, non faites.

- Quand le dataset 20k transforms sera prêt : relancer la grille avec les époques des configs
  générées (les runs actuelles sont des runs de validation à 2 époques).
- `SyMTRS_Pipeline.md` se contredit (gris ×51 vs ×63, 6 vs 5 classes) : à corriger côté Docs.
- Sensibilité du seuil du masque d'ombre dérivé (ShadowFormer, métriques par zone).
- Inférence shadow removal tuilée : ~10 s/image 2048 (Restormer, tuiles 256 une par une, 0,6 GB VRAM) → regrouper les tuiles par batch accélérerait l'évaluation ~5–10×.
