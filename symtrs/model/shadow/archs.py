"""Which upstream class each shadow-removal model is, and the arguments it is built with.

Torch-free on purpose: the worker imports the class, but everything that merely *describes*
it - `symtrs models`, the dry run, the tests that read the upstream constructor signature
from source - must work without torch or the upstream trees.

The constructor arguments are the ones each paper's own shadow-removal entry point uses
(ShadowFormer's `utils.get_arch`, Uformer's `Uformer_B`, Restormer's defaults,
ShadowMaskFormer's `shadowmaskformer_b()` factory), recorded here once so a run and a
checkpoint agree on the network. `img_size` is not listed: it is
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
    #: ShadowFormer's and ShadowMaskFormer's forward is `net(image, mask)`; the others see the
    #: image only.
    takes_mask: bool
    #: Whether the constructor takes `img_size` (the training patch).
    takes_img_size: bool
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    #: What the authors' *released* weights were trained for, i.e. what `shadow.init_weights`
    #: starts from when it points at them. Only ShadowFormer ships shadow-removal weights;
    #: the other two start from denoising, which is a different (and for this benchmark
    #: weaker) prior, so a results table must state it beside every row.
    init_source_task: str = ""
    #: The value range the *network* is written for. The worker keeps images and masks in
    #: [0, 1] (mask: shadow = 1) and converts at the network boundary only, so that the
    #: loss, the validation PSNR and the PNG writer are the same code for every model.
    #: "unit": the network takes and returns [0, 1]. "signed": it takes image and mask in
    #: [-1, 1] (`x * 2 - 1`, shadow = +1) and returns [-1, 1], unclamped (ShadowMaskFormer's
    #: own train/test scripts do exactly this, then `.clamp_(-1, 1) * 0.5 + 0.5`).
    input_range: str = "unit"
    #: False when the authors published no weights at all: the model is always trained from
    #: random initialisation (`shadow.init_weights` stays empty), which makes its row not
    #: like-for-like with a model that starts from a checkpoint.
    released_weights: bool = True

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
        init_source_task="shadow removal (ISTD+)",
    ),
    "uformer": Arch(
        "uformer", "Uformer", "model.py", "Uformer",
        takes_mask=False, takes_img_size=True,
        kwargs={
            "embed_dim": 32, "win_size": 8, "token_projection": "linear", "token_mlp": "leff",
            "depths": [1, 2, 8, 8, 2, 8, 8, 2, 1], "modulator": True,
        },
        init_source_task="denoising (SIDD); no shadow-removal weights released",
    ),
    "restormer": Arch(
        "restormer", "Restormer", "basicsr/models/archs/restormer_arch.py", "Restormer",
        takes_mask=False, takes_img_size=False,
        # BiasFree is the real_denoising release's setting (the class default is WithBias):
        # it is what lets shadow.init_weights load all 494 tensors instead of 406.
        kwargs={"LayerNorm_type": "BiasFree"},
        init_source_task="real image denoising (SIDD); no shadow-removal weights released",
    ),
    "shadowmaskformer": Arch(
        "shadowmaskformer", "ShadowMaskFormer", "models/shadowmaskformer.py", "ShadowMaskFormer",
        takes_mask=True, takes_img_size=False, input_range="signed", released_weights=False,
        # The "-b" variant, i.e. upstream's `shadowmaskformer_b()` factory and the default of
        # its train.py (2.28 M parameters). Spelled out instead of calling the factory so the
        # test that checks every recorded name against the constructor covers it. The network
        # pads to a multiple of 4 and crops back by itself, so there is no `img_size`.
        kwargs={
            "in_chans": 3, "out_chans": 4, "window_size": 10,
            "embed_dims": [24, 48, 96, 48, 24], "mlp_ratios": [2.0, 4.0, 4.0, 2.0, 2.0],
            "depths": [16, 16, 14, 8, 8], "num_heads": [2, 4, 6, 1, 1],
            "attn_ratio": [0.25, 0.5, 0.75, 0, 0], "conv_type": ["DWConv"] * 5,
        },
        init_source_task="none: no weights released; trained from scratch",
    ),
}

#: The ranges `Arch.input_range` can take.
INPUT_RANGES = ("unit", "signed")


def to_network_range(arch: Arch, image, mask):
    """Worker range ([0, 1] image, {0, 1} mask) -> what the network was written for."""
    if arch.input_range == "signed":
        return image * 2 - 1, mask * 2 - 1
    return image, mask


def from_network_range(arch: Arch, output):
    """The network's output -> [0, 1] (not clamped: the caller clamps, as it always did)."""
    return output * 0.5 + 0.5 if arch.input_range == "signed" else output


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
