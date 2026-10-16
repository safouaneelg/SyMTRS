# Third-party pins

| Repo | URL | Commit | Date |
|---|---|---|---|
| ShadowFormer | https://github.com/guolanqing/ShadowFormer | f72158109ff3f09b4ee67d343cdc7c7b83c5ce6d | 2023-08-05 |
| Restormer | https://github.com/swz30/Restormer | 68dc6ac472db26f16361150cb7a96a1bc87da93f | 2025-10-23 |
| Uformer | https://github.com/ZhendongWang6/Uformer | 65fc970a8ffc09605faca74ed016ee93c9ad8a36 | 2025-03-21 |

The upstream requirements pin torch 1.7/1.8; we run them on the main `.venv` (torch cu128)
because none of the three builds a CUDA extension. Any patch is listed here.
