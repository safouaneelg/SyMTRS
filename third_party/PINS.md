# Third-party pins

| Repo | URL | Commit | Date |
|---|---|---|---|
| ShadowFormer | https://github.com/guolanqing/ShadowFormer | f72158109ff3f09b4ee67d343cdc7c7b83c5ce6d | 2023-08-05 |
| Restormer | https://github.com/swz30/Restormer | 68dc6ac472db26f16361150cb7a96a1bc87da93f | 2025-10-23 |
| Uformer | https://github.com/ZhendongWang6/Uformer | 65fc970a8ffc09605faca74ed016ee93c9ad8a36 | 2025-03-21 |
| ShadowMaskFormer | https://github.com/lizhh268/ShadowMaskFormer | 9c505e11ca5ab38a7473a51defca7d33b7676514 | 2025-05-08 |

The upstream requirements pin torch 1.7/1.8; we run them on the main `.venv` (torch cu128)
because none of the four builds a CUDA extension. Any patch is listed here.

**ShadowMaskFormer has no licence file** in its repository, so nothing of it is copied or committed
here: the checkout is gitignored and fetched by script, and only its network class is imported at
run time. It also publishes **no weights**, so it is always trained from scratch on SyMTRS
(`shadow.init_weights` stays empty); its comparison with ShadowFormer, which starts from ISTD+
weights, is not like-for-like. It needs only torch and `timm.models.layers` (`to_2tuple`,
`trunc_normal_`; a FutureWarning on timm >= 1.0). Its `train.py` / `test.py` import
`pytorch_msssim`; the shared worker never imports them.
