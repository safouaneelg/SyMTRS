# BLOCKERS

Blocages rencontrés, solutions tentées, état.

| Date | Blocage | Tentatives | État |
|---|---|---|---|
| 2026-10-16 | Python système 3.14 sans torch | uv + Python 3.12 dans `.venv` / `.venv-zoe` | résolu |
| 2026-10-16 | Noms de fichiers réels préfixés (`rgb_day_2048_<key>`) → 0 paire | correctif `normalize_stem` (étape 1) | ouvert |
| 2026-10-16 | Poids pré-entraînés ShadowFormer/Uformer/Restormer sur Google Drive | gdown (étape 0b) | ouvert |
| 2026-10-16 | timm 0.6.12 (ZoeDepth) plante à l'import sous Python 3.12 (`mutable default ... MaxxVitConvCfg`) | `.venv-zoe` recréé en Python 3.10 | en cours |
| 2026-10-16 | Fin de session Claude → tâches d'arrière-plan et sous-agents tués | lancement via `Start-Process` (processus détachés), état dans les logs/PROGRESS | contourné |
| 2026-10-16 | Prefetch bloqué à 0 MB/s : le hubconf MiDaS appelle `torch.hub.load` (gen-efficientnet, WSL-Images) sans `trust_repo` → invite y/N sur stdin, processus caché bloqué | entrées ajoutées à `.cache/torch/hub/trusted_list` (fait aussi par `prefetch_all.ps1`) ; protège aussi les runs | résolu |
| 2026-10-16 | **Correction** : les « fins de session » étaient des **plantages du PC** (BSOD + reboot). 18:35→19:00 bugcheck 0x20001 HYPERVISOR_ERROR ; ~19:07→19:08 bugcheck 0x101 CLOCK_WATCHDOG_TIMEOUT ; WHEA « fatal hardware error » à chaque reboot. Même 0x20001 (mêmes paramètres) déjà le 2026-10-07, avant ce travail. ROG Zephyrus G16 GU605CR, BIOS GU605CR.311. | rien côté code ; pistes : MAJ BIOS/chipset/ME via MyASUS, limiter la charge CPU simultanée, file idempotente pour reprendre après reboot | ouvert (matériel/firmware) |
| 2026-10-16 | Entraînement très lent (113 s pour 3 batches U-Net 512) : `batch_size: 32` des configs → pic 18,6 GB sur un GPU de 12 GB ; le pilote Windows (WDDM) déborde en mémoire système au lieu d'OOM → ×20–500 plus lent. Mesures unet_small 512 AMP : bs8 0,20 s/step (4,7 GB), bs16 4,3 s/step (9,4 GB, déjà en débordement), bs32 34–115 s/step | pour les runs de validation : `--batch-size` plafonné par config pour rester < ~8 GB alloués (mesuré au smoke test) ; recommandé : Panneau NVIDIA → « CUDA - Sysmem Fallback Policy » = « Prefer No Sysmem Fallback » pour obtenir un OOM franc au lieu d'une lenteur silencieuse | ouvert |
| 2026-10-16 | Dépôt hub ZoeDepth en cache : 9 fichiers `.py/.json` remplis de zéros (`JSONDecodeError`, `null bytes`) — corruption probable liée aux BSOD | dépôt supprimé, re-téléchargé (réseau GitHub, ~9 Mo) ; les checkpoints `.pt` n'étaient pas touchés | résolu |
| 2026-10-16 | `.venv-zoe` : DLL bloquée par Smart App Control | ré-essai : `import torch, timm` fonctionne maintenant (torch 2.11+cu128, timm 0.6.12, CUDA True) ; installateur python.org inutile | résolu |
| 2026-10-16 | Les modèles ZoeDepth et `midas_dpt_swin2_tiny` ne se construisent que dans `.venv-zoe` (timm 0.6.12) | `scripts/run_queue.py` prend le venv par job ; il faut y lancer ces 4 clés | contourné |
| 2026-10-16 | (suite VRAM) | l'utilisateur a réglé le pilote : CUDA Sysmem Fallback Policy = **Prefer No Sysmem Fallback** ; + garde `SYMTRS_CUDA_MEM_FRACTION=0.92` côté code | résolu |
