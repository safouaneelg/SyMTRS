# Third-party pins

The four shadow-removal upstreams are git submodules. The commit each one is pinned to is
recorded by git itself (`.gitmodules` holds the URL, the gitlink holds the commit), so this file
no longer carries the hashes: a second copy here could disagree with the one that is checked
out. `git submodule status third_party` prints them, with a leading `+` when a checkout has
drifted from the recorded commit.

    git submodule update --init third_party      # or: python scripts/fetch_third_party.py
    python scripts/fetch_third_party.py --check  # verify the checkouts match the pins

| Repo | URL | Pinned on | Commit date |
|---|---|---|---|
| ShadowFormer | https://github.com/guolanqing/ShadowFormer | f721581 | 2023-08-05 |
| Restormer | https://github.com/swz30/Restormer | 68dc6ac | 2025-10-23 |
| Uformer | https://github.com/ZhendongWang6/Uformer | 65fc970 | 2025-03-21 |
| ShadowMaskFormer | https://github.com/lizhh268/ShadowMaskFormer | 9c505e1 | 2025-05-08 |

(The short hashes are for reading only; the authoritative pin is the gitlink.)

Only the network class is taken from each checkout (`symtrs/model/shadow/archs.py` names the
file and class); none of their training or test scripts is run, and none of the four is
modified. The upstream requirements pin torch 1.7/1.8; we run them on the main `.venv` (torch
cu128) because none of the four builds a CUDA extension, and no patch has been needed on
torch 2.11. A submodule cannot carry local edits: if one is ever needed, fork the repository and
point the submodule at the fork, or vendor it the way `pytorch-CycleGAN-and-pix2pix/` is.

**ShadowMaskFormer has no licence file** in its repository. A submodule copies none of its
code into this repository (only the URL and the commit), and only its network class is imported
at run time. It also publishes **no weights**, so it is always trained from scratch on SyMTRS
(`shadow.init_weights` stays empty); its comparison with ShadowFormer, which starts from ISTD+
weights, is not like-for-like. It needs only torch and `timm.models.layers` (`to_2tuple`,
`trunc_normal_`; a FutureWarning on timm >= 1.0). Its `train.py` / `test.py` import
`pytorch_msssim`; the shared worker never imports them.
