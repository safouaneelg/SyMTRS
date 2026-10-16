"""Which upstream class each shadow-removal model is, and the arguments it is built with.

Torch-free on purpose: the worker imports the class, but everything that merely *describes*
it - `symtrs models`, the dry run, the tests that read the upstream constructor signature
from source - must work without torch or the upstream trees.

The constructor arguments are the ones each paper's own shadow-removal entry point uses
(ShadowFormer's `utils.get_arch`, Uformer's `Uformer_B`, Restormer's defaults), recorded
here once so a run and a checkpoint agree on the network. `img_size` is not listed: it is
the training patch, `train.patch_size`, and is injected by the worker for the classes that
take it (their shifted-window masks are sized from it).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping

REPO_ROOT = Path(__file__).resolve().parents[3]
THIRD_PARTY = REPO_ROOT / "third_party"


@dataclass(frozen=True)
class Arch:
    key: str
    #: Directory under third_party/ (the upstream checkout, see third_party/PINS.md).
    repo: str
    #: Python file that defines the class, relative to the checkout.
    source: str
    cls: str
    #: ShadowFormer's forward is `net(image, mask)`; the other two see the image only.
    takes_mask: bool
    #: Whether the constructor takes `img_size` (the training patch).
    takes_img_size: bool
    kwargs: Mapping[str, Any] = field(default_factory=dict)

    @property
    def source_path(self) -> Path:
        return THIRD_PARTY / self.repo / self.source

    @property
    def repo_path(self) -> Path:
        return THIRD_PARTY / self.repo


ARCHS: Dict[str, Arch] = {
    "shadowformer": Arch(
        "shadowformer", "ShadowFormer", "model.py", "ShadowFormer",
        takes_mask=True, takes_img_size=True,
        kwargs={"embed_dim": 32, "win_size": 10, "token_projection": "linear", "token_mlp": "leff"},
    ),
    "uformer": Arch(
        "uformer", "Uformer", "model.py", "Uformer",
        takes_mask=False, takes_img_size=True,
        kwargs={
            "embed_dim": 32, "win_size": 8, "token_projection": "linear", "token_mlp": "leff",
            "depths": [1, 2, 8, 8, 2, 8, 8, 2, 1], "modulator": True,
        },
    ),
    "restormer": Arch(
        "restormer", "Restormer", "basicsr/models/archs/restormer_arch.py", "Restormer",
        takes_mask=False, takes_img_size=False,
        # BiasFree is the real_denoising release's setting (the class default is WithBias):
        # it is what lets shadow.init_weights load all 494 tensors instead of 406.
        kwargs={"LayerNorm_type": "BiasFree"},
    ),
}


def arch_for(key: str) -> Arch:
    try:
        return ARCHS[key]
    except KeyError:
        raise ValueError(
            f"{key!r} is not a shadow-removal model; expected one of {sorted(ARCHS)}. "
            "`symtrs models --task shadow_removal` lists them."
        ) from None


def upstream_missing(key: str) -> str:
    """Empty string when the checkout is usable, else the sentence that says how to fix it."""
    arch = arch_for(key)
    if arch.source_path.is_file():
        return ""
    return (
        f"the {arch.repo} checkout is missing ({arch.source_path}). The three upstream repos "
        "are cloned at pinned revisions, not committed: run "
        "`python scripts/fetch_third_party.py` (add `--check` to verify)."
    )
