#!/usr/bin/env python3
"""Emit the full `configs/` grid: every (task, model, resolution) cell as a YAML file.

    python3 configs/generate_grid.py            # write the files
    python3 configs/generate_grid.py --check    # verify the tree matches the tables

The grid is large - 11 depth models x 4 resolutions, 6 segmentation models x 4, 4 domain
adaptation runs, 4 super-resolution models x 4 scales = 88 files - and hand-maintaining
that many near-identical YAMLs is how they drift: one file keeps a stale learning rate,
another keeps an `align` that does not match its model's protocol, and the results table
quietly mixes two protocols.

So the *tables* below are the source of truth, one entry per model, and this script
expands them. The generated files are committed and are what you run; this is how they
are regenerated when a model is added or a hyper-parameter changes across the board.

What the tables encode, per model:

  * the depth protocol (`relative` -> SILog loss + `align: scale`; `metric` -> BErHu loss
    + `align: none`). Getting this wrong is the single most damaging mistake available in
    this benchmark: a relative model scored without alignment looks broken, and a metric
    model scored with it looks suspiciously good.
  * the memory posture at native resolution, which is what the per-resolution batch sizes
    and `grad_accum` are. The *effective* batch is batch_size x grad_accum and is meant to
    stay roughly constant down the resolution axis, so a 256 run is a smaller-image run
    and not also an accidentally different optimisation.
  * the model-specific caveats worth stating next to the number they affect.

The per-model prose lives on the native (2048) file. The lower-resolution files carry the
resolution note and a pointer back, so the same three paragraphs are not repeated 4 times
where only one sentence changes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent

RESOLUTIONS = (2048, 1024, 512, 256)
SCALE_SUFFIX = {2048: "x1", 1024: "x2", 512: "x4", 256: "x8"}
SCALE_FACTOR = {2048: 1, 1024: 2, 512: 4, 256: 8}

#: The same paragraph in every file, because the resolution axis is the same question
#: everywhere and a reader who opens one file should not have to open three more.
RESOLUTION_NOTE = {
    2048: (
        "resolution: 2048 reads train/rgb_day and train/depth - the native folders. Nothing\n"
        "is downsampled and nothing is upsampled: this is the data as it was generated."
    ),
    1024: (
        "resolution: 1024 reads train/rgb_day_lr/x2 and train/depth_lr/x2. Input and ground\n"
        "truth are both 1024, so this is the same task on smaller imagery, not a model being\n"
        "asked to upsample. The 2048 and 1024 runs are two rows of one table, not two stages\n"
        "of a pipeline."
    ),
    512: (
        "resolution: 512 reads train/rgb_day_lr/x4 and train/depth_lr/x4. Input and ground\n"
        "truth are both 512, so this is the same task on smaller imagery, not a model being\n"
        "asked to upsample. The 2048 and 512 runs are two rows of one table, not two stages\n"
        "of a pipeline."
    ),
    256: (
        "resolution: 256 reads train/rgb_day_lr/x8 and train/depth_lr/x8. Input and ground\n"
        "truth are both 256 - the smallest cell of the grid, and the one that says how much\n"
        "of a model's score is resolution and how much is the model. The 2048 and 256 runs\n"
        "are two rows of one table, not two stages of a pipeline."
    ),
}

SEG_RESOLUTION_NOTE = {
    2048: (
        "resolution: 2048 reads train/rgb_day and train/semantic - the native folders. The\n"
        "model is scored at 2048 against 2048 masks."
    ),
    1024: (
        "resolution: 1024 reads train/rgb_day_lr/x2 and train/semantic_lr/x2. Input and\n"
        "ground truth are both 1024, and the model is scored at 1024 - this is the same\n"
        "segmentation task on smaller imagery, not super-resolution. Compare it to the 2048\n"
        "run to see what the lost detail costs, which is the question the resolution axis\n"
        "exists to answer."
    ),
    512: (
        "resolution: 512 reads train/rgb_day_lr/x4 and train/semantic_lr/x4. Input and ground\n"
        "truth are both 512, and the model is scored at 512. At this scale small objects are\n"
        "a handful of pixels, which is exactly what the rare-class metrics in eval.metrics\n"
        "are there to expose."
    ),
    256: (
        "resolution: 256 reads train/rgb_day_lr/x8 and train/semantic_lr/x8. Input and ground\n"
        "truth are both 256. This is the floor of the grid: a class that survives here is a\n"
        "class the model genuinely recognises rather than one it resolves."
    ),
}

# --------------------------------------------------------------------------- depth


class DepthModel:
    """One depth entry: the protocol plus the memory posture at each resolution."""

    def __init__(
        self,
        key: str,
        title: str,
        protocol: str,  # "relative" | "metric"
        batch: Dict[int, int],
        lr: float,
        grad_accum: int,
        patch: Dict[int, int],
        freeze: str = "",
        note: str = "",
        model_kwargs: Optional[Dict[str, object]] = None,
    ) -> None:
        self.key = key
        self.title = title
        self.protocol = protocol
        self.batch = batch
        self.lr = lr
        self.grad_accum = grad_accum
        self.patch = patch
        self.freeze = freeze
        self.note = note
        self.model_kwargs = model_kwargs or {}

    @property
    def align(self) -> str:
        return "scale" if self.protocol == "relative" else "none"

    @property
    def loss(self) -> str:
        return "silog" if self.protocol == "relative" else "berhu"

    @property
    def effective_batch(self) -> int:
        return self.batch[2048] * self.grad_accum


#: The 11 depth entries. `batch` is stated at every resolution because the whole point of
#: the grid is that the four cells are separate runs; the values roughly double each time
#: the pixel count quarters, which keeps the effective batch and the memory per step flat.
DEPTH_MODELS: List[DepthModel] = [
    DepthModel(
        "midas_small",
        "MiDaS small",
        "relative",
        batch={2048: 8, 1024: 16, 512: 32, 256: 64},
        lr=2.0e-5,
        grad_accum=1,
        patch={2048: 384, 1024: 0},
        note=(
            "The cheapest cell in the grid: EfficientNet-lite3 backbone, a DPT-lite head, and\n"
            "a 256px training resolution inside MiDaS itself. A full four-resolution sweep\n"
            "finishes in the time DPT-BEiT-Large takes for one cell, which makes it the model\n"
            "to use while the dataset is still being regenerated.\n"
            "\n"
            "It is relative-depth, so it is scored with align: scale and trained with the\n"
            "scale-invariant SILog loss MiDaS was trained with."
        ),
    ),
    DepthModel(
        "midas_dpt_hybrid",
        "MiDaS DPT-Hybrid",
        "relative",
        batch={2048: 4, 1024: 8, 512: 16, 256: 32},
        lr=1.0e-5,
        grad_accum=1,
        patch={2048: 384, 1024: 0},
        note=(
            "The standard published baseline, and the model the MiDaS numbers in the paper's\n"
            "comparison table are quoted from. Relative-depth: it predicts disparity, and a\n"
            "per-image affine fit is needed before any metric means anything, which is what\n"
            "eval.align does. Training uses SILog, the loss MiDaS itself was trained with, so\n"
            "the fine-tune starts from a matched objective instead of fighting a metric loss\n"
            "the pretrained head has never seen.\n"
            "\n"
            "At 2048 a full batch of 4 is ~200MB of input per step and the DPT decoder's\n"
            "attention is quadratic in the token count on top of that, so training crops to\n"
            "384px patches. Below 2048 the whole image fits and patch_size drops to 0."
        ),
    ),
    DepthModel(
        "midas_dpt_large",
        "MiDaS DPT-Large",
        "relative",
        batch={2048: 2, 1024: 4, 512: 8, 256: 16},
        lr=1.0e-5,
        grad_accum=4,
        patch={2048: 384, 1024: 512},
        freeze="encoder",
        note=(
            "The same head as DPT-Hybrid on a ViT-Large encoder: ~3x the parameters and the\n"
            "slowest of the MiDaS family. The encoder is frozen here, which is the posture the\n"
            "big encoders in this pack share - the ViT was pretrained on a billion images and\n"
            "20 epochs on a few thousand aerial tiles will not improve it, while the DPT head\n"
            "is what has to adapt to a nadir view. Set `freeze: \"\"` if you want to fine-tune\n"
            "the whole thing; expect to lower train.lr if you do."
        ),
    ),
    DepthModel(
        "midas_dpt_swin2_tiny",
        "MiDaS DPT-SwinV2-Tiny",
        "relative",
        batch={2048: 4, 1024: 8, 512: 16, 256: 32},
        lr=2.0e-5,
        grad_accum=1,
        patch={2048: 384, 1024: 0},
        note=(
            "SwinV2-Tiny with the DPT head. Window attention rather than global attention, so\n"
            "it is the MiDaS variant whose cost grows most gently with resolution - which is\n"
            "why it is worth running the 2048 cell of this one even if you skip the others.\n"
            "Still relative-depth: align: scale, SILog loss."
        ),
    ),
    DepthModel(
        "midas_dpt_beit_large",
        "MiDaS DPT-BEiT-Large",
        "relative",
        batch={2048: 1, 1024: 2, 512: 4, 256: 8},
        lr=5.0e-6,
        grad_accum=8,
        patch={2048: 384, 1024: 512},
        freeze="encoder",
        note=(
            "The largest MiDaS variant, and the widest memory budget in the grid: batch 1 with\n"
            "8 accumulation steps at 2048. The encoder is frozen for the same reason as\n"
            "DPT-Large, and more so - it is the most expensive forward pass here and the least\n"
            "likely to change usefully in 20 epochs.\n"
            "\n"
            "If the accumulate-then-step arithmetic is not obvious: batch_size x grad_accum is\n"
            "the effective batch, so this runs the same 8-image step as the hybrid does."
        ),
    ),
    DepthModel(
        "zoedepth_n",
        "ZoeDepth-N",
        "relative",
        batch={2048: 2, 1024: 4, 512: 8, 256: 16},
        lr=5.0e-6,
        grad_accum=4,
        patch={2048: 512, 1024: 0},
        freeze="encoder",
        note=(
            "ZoeDepth-N is the relative-depth member of the ZoeDepth family: a MiDaS backbone\n"
            "with a relative head, so it is scored with align: scale and trained with SILog.\n"
            "Swapping this config's model to zoedepth_k changes both of those, which is the\n"
            "whole reason the depth adapters carry a `protocol` attribute.\n"
            "\n"
            "ZoeDepth resizes its input to a 384px minimum side internally, so a 2048px crop\n"
            "costs memory it does not spend on detail; 512 is the crop that reaches the network\n"
            "at the scale it was trained for. The encoder is frozen because that resize is the\n"
            "part that already works."
        ),
    ),
    DepthModel(
        "zoedepth_k",
        "ZoeDepth-K",
        "metric",
        batch={2048: 2, 1024: 4, 512: 8, 256: 16},
        lr=5.0e-6,
        grad_accum=4,
        patch={2048: 512, 1024: 0},
        freeze="encoder",
        note=(
            "The *metric* baseline: it predicts metres directly and needs no alignment, so\n"
            "eval.align is none and the BErHu loss is used. Compare it against a relative model\n"
            "only after checking that both were scored under their own protocol.\n"
            "\n"
            "This is the cell most sensitive to data.depth_scale: if the dataset stores depth in\n"
            "millimetres or as uint16 centimetres, a model predicting metres is wrong by a\n"
            "constant factor and every metric is bad in a way that looks like a broken model.\n"
            "Check `symtrs inspect` reports the same unit for the native and the _lr depth\n"
            "folders before reading a single number from this file."
        ),
    ),
    DepthModel(
        "zoedepth_nk",
        "ZoeDepth-NK",
        "metric",
        batch={2048: 2, 1024: 4, 512: 8, 256: 16},
        lr=5.0e-6,
        grad_accum=4,
        patch={2048: 512, 1024: 0},
        freeze="encoder",
        note=(
            "Metric, like ZoeDepth-K, with an extra domain classifier on the encoder features.\n"
            "The classifier is what `model_kwargs.domain_id` selects: it tells the model which\n"
            "of its training domains the input resembles, and it is the one knob here that has\n"
            "no correct answer in advance. Leave it unset and the model uses its own head; set\n"
            "it only if you are deliberately testing the domain conditioning."
        ),
        model_kwargs={"domain_id": None},
    ),
    DepthModel(
        "da3_large",
        "Depth Anything 3 Large (relative)",
        "relative",
        batch={2048: 2, 1024: 4, 512: 8, 256: 16},
        lr=1.0e-5,
        grad_accum=2,
        patch={2048: 512, 1024: 0},
        note=(
            "Depth Anything 3's un-metriced checkpoint: relative depth, so align: scale and\n"
            "SILog. It loads through `transformers` rather than torch.hub:\n"
            "\n"
            "    pip install -e '.[da3]'\n"
            "\n"
            "Which protocol is in force is decided by the checkpoint, not by a flag here:\n"
            "`da3_large` names the relative checkpoint and `da3_metric_large` the metric one,\n"
            "and the resolution is recorded in the checkpoint metadata either way, so a mixed-up\n"
            "comparison is visible in the .pt rather than only in the results table."
        ),
    ),
    DepthModel(
        "da3_metric_large",
        "Depth Anything 3 Large (metric)",
        "metric",
        batch={2048: 4, 1024: 8, 512: 16, 256: 32},
        lr=1.0e-5,
        grad_accum=2,
        patch={2048: 512, 1024: 0},
        note=(
            "The metric head: metres, no alignment, BErHu. Same loader caveats as da3_large.\n"
            "One extra: the DA3 Hugging Face repos do not all ship an image-processor config,\n"
            "so `AutoImageProcessor.from_pretrained` can refuse. That raises with the model id\n"
            "and the command to run - `symtrs doctor --model da3_metric_large --build` - rather\n"
            "than silently substituting some other normalisation."
        ),
    ),
    DepthModel(
        "depth_pro",
        "Apple Depth Pro",
        "metric",
        batch={2048: 1, 1024: 2, 512: 4, 256: 8},
        lr=5.0e-6,
        grad_accum=8,
        patch={2048: 512, 1024: 512},
        freeze="encoder",
        note=(
            "Metric depth from Apple: ~1.9GB of weights, a ViT-L encoder that wants a 1536px\n"
            "input, and the heaviest forward pass in the pack. The encoder is frozen for that\n"
            "reason - 300M+ parameters, and the head is what adapts to a new sensor.\n"
            "\n"
            "The one thing about this model that is not plug-and-play for aerial imagery is the\n"
            "focal length, and it is worth getting right before a long run:\n"
            "\n"
            "  * With `f_px: null` (the default below) Depth Pro estimates the field of view\n"
            "    with its own head. That head was trained on internet photography, so a nadir\n"
            "    aerial frame can be given a plausible-looking but wrong FOV - and because the\n"
            "    model predicts metres, a wrong FOV is a wrong *scale* over the whole scene.\n"
            "    Depth ordering looks perfect and RMSE is bad.\n"
            "  * Set `f_px` to the SyMTRS camera's true focal length in pixels and the metric\n"
            "    prediction is anchored. That is the honest way to run this baseline.\n"
            "\n"
            "Depth Pro windows its own inference internally at 1536px, so scoring at 2048 in\n"
            "one pass does not mean the model saw 2048 pixels at once."
        ),
        model_kwargs={"f_px": None},
    ),
]

# -------------------------------------------------------------------- segmentation


class SegModel:
    def __init__(
        self,
        key: str,
        title: str,
        batch: Dict[int, int],
        lr: float,
        grad_accum: int,
        loss: str,
        loss_kwargs: Optional[Dict[str, object]] = None,
        patch: Optional[Dict[int, int]] = None,
        note: str = "",
    ) -> None:
        self.key = key
        self.title = title
        self.batch = batch
        self.lr = lr
        self.grad_accum = grad_accum
        self.loss = loss
        self.loss_kwargs = loss_kwargs or {}
        self.patch = patch or {2048: 512, 1024: 512, 512: 0, 256: 0}
        self.note = note


SEG_MODELS: List[SegModel] = [
    SegModel(
        "unet_small",
        "Small from-scratch U-Net",
        batch={2048: 8, 1024: 16, 512: 32, 256: 64},
        lr=3.0e-4,
        grad_accum=1,
        loss="ce_dice",
        note=(
            "The cheapest segmentation cell: a compact U-Net trained from scratch, with no\n"
            "pretrained encoder at all. That makes it the lower bound in the table - everything\n"
            "else here starts from ImageNet or COCO weights - and the one model whose learning\n"
            "rate is an order of magnitude above the rest, because there are no pretrained\n"
            "features for a small lr to protect.\n"
            "\n"
            "It is also the model that shows what the dataset alone supports: if the fine-tuned\n"
            "U-Net reaches a miou close to DeepLabV3's, the pretrained backbones are not earning\n"
            "their memory."
        ),
    ),
    SegModel(
        "deeplabv3_r50",
        "DeepLabV3 (ResNet-50)",
        batch={2048: 4, 1024: 8, 512: 16, 256: 32},
        lr=1.0e-4,
        grad_accum=1,
        loss="ce_aux",
        loss_kwargs={"aux_weight": 0.4},
        note=(
            "The cheap half of the deep segmentation family: torchvision DeepLabV3 on a\n"
            "ResNet-50 backbone pretrained on COCO+VOC, with the whole backbone trainable.\n"
            "It fits on a single mid-range GPU at these batch sizes.\n"
            "\n"
            "torchvision's DeepLabV3 emits an auxiliary head off the ResNet features and our\n"
            "loss folds its term in automatically. That is what \"ce_aux\" names: cross-entropy\n"
            "on both heads, weighted 0.4 to the auxiliary one (the weight from the DeepLabV3\n"
            "paper). \"ce\" would ignore the auxiliary head entirely.\n"
            "\n"
            "One trap worth stating: the atrous head contains BatchNorm, and torchvision's\n"
            "implementation cannot be run with a batch of 1 in training mode. train.batch_size\n"
            "must stay at 2 or more - at 256px the batch below is 32, but a `--batch-size 1` on\n"
            "the command line for a smoke run will be refused with that explanation rather than\n"
            "a shape error from inside BatchNorm."
        ),
    ),
    SegModel(
        "deeplabv3_r101",
        "DeepLabV3 (ResNet-101)",
        batch={2048: 2, 1024: 4, 512: 8, 256: 16},
        lr=8.0e-5,
        grad_accum=2,
        loss="ce_aux",
        loss_kwargs={"aux_weight": 0.4},
        note=(
            "ResNet-101 instead of ResNet-50: deeper, ~2x the parameters, and the cell to run\n"
            "if the ResNet-50 result sits below the SegFormer ones and you want to know whether\n"
            "the gap is the architecture or the encoder depth. Same auxiliary-head loss and the\n"
            "same batch-size-2 minimum as deeplabv3_r50."
        ),
    ),
    SegModel(
        "segformer_b0",
        "SegFormer (MiT-B0)",
        batch={2048: 8, 1024: 16, 512: 32, 256: 64},
        lr=6.0e-5,
        grad_accum=1,
        loss="ce_dice",
        note=(
            "The smallest MiT encoder with SegFormer's decode head. Its encoder is loaded from\n"
            "timm's ImageNet weights, so `freeze` is empty: those weights were learned on\n"
            "natural-image classification, not on aerial segmentation, and SegFormer's own\n"
            "paper fine-tunes the whole network. Freezing here would trade away most of what\n"
            "the model has to offer."
        ),
    ),
    SegModel(
        "segformer_b1",
        "SegFormer (MiT-B1)",
        batch={2048: 4, 1024: 8, 512: 16, 256: 32},
        lr=6.0e-5,
        grad_accum=2,
        loss="ce_dice",
        note=(
            "The middle MiT encoder. Between this and MiT-B2 you get a clean read on how much\n"
            "of SegFormer's score is encoder scale, since the decode head is identical across\n"
            "the three."
        ),
    ),
    SegModel(
        "segformer_b2",
        "SegFormer (MiT-B2)",
        batch={2048: 4, 1024: 8, 512: 16, 256: 32},
        lr=6.0e-5,
        grad_accum=2,
        loss="ce_dice",
        note=(
            "The largest SegFormer in the pack and the strongest of the torchvision+timm\n"
            "baselines. Fully convolutional with a cheap decode head, so 512px crops are\n"
            "affordable and still contain several whole objects at this ground sampling\n"
            "distance.\n"
            "\n"
            "ce_dice rather than plain cross-entropy: with six classes over aerial imagery the\n"
            "rare ones (vehicle, water) can be a fraction of a percent of the pixels, and\n"
            "cross-entropy alone is free to ignore them. The Dice term is what pulls them up.\n"
            "`loss_kwargs.class_weights` is the other way to do it, but Dice needs no per-class\n"
            "statistics estimated first."
        ),
    ),
]

#: Every segmentation file repeats this, because the numbers it names are recorded in the
#: checkpoint and a wrong one silently invalidates a score.
SEG_CLASS_NOTE = (
    "Six classes: the dataset's semantic/ masks are graylevel PNGs, one byte per pixel, and\n"
    "255 is the ignore index for pixels that belong to no class. Both numbers are written\n"
    "into the checkpoint, so a model trained here cannot be silently scored against a\n"
    "differently-labelled mask set later. If the mask generator ever writes 0 for \"no class\"\n"
    "instead of 255, change data.ignore_index in the config and not the metric - a metric\n"
    "that guesses which pixels to ignore is worse than one that does not."
)

# ------------------------------------------------------------------ domain adapt


class DomainModel:
    def __init__(self, key: str, upstream_model: str, dataset_mode: str, title: str, note: str) -> None:
        self.key = key
        self.upstream_model = upstream_model
        self.dataset_mode = dataset_mode
        self.title = title
        self.note = note


DOMAIN_MODELS: List[DomainModel] = [
    DomainModel(
        "cyclegan",
        "cycle_gan",
        "unaligned",
        "CycleGAN",
        (
            "CycleGAN is *unpaired*: it learns the two mappings from two sets of images with no\n"
            "correspondence between them, which is what the day and night folders are - the same\n"
            "scenes re-rendered at a different time of day, but not pixel-aligned pairs by\n"
            "construction. That is also why the score is FID/KID rather than an accuracy: a\n"
            "translation is judged by how close its image distribution is to the real one."
        ),
    ),
    DomainModel(
        "pix2pix",
        "pix2pix",
        "aligned",
        "pix2pix",
        (
            "pix2pix is *paired* and conditional: it needs an A image and the B image of the same\n"
            "scene, matched by filename. SyMTRS day and night frames share their stems, so the\n"
            "matching exists - but it is a stronger assumption than CycleGAN makes, and the gap\n"
            "between the two models on this dataset is a measurement of how well that assumption\n"
            "holds. If the night rendering was produced independently of the day frame rather\n"
            "than from the same geometry, pix2pix is being asked for something the data does not\n"
            "contain and its FID should be read as a lower bound on what paired supervision could\n"
            "buy, not as a fair comparison."
        ),
    ),
]

DOMAIN_DIRECTIONS = ("day2night", "night2day")

# --------------------------------------------------------------- super-resolution


class SuperResModel:
    def __init__(
        self,
        key: str,
        title: str,
        batch: int,
        lr: float,
        epochs: int,
        model_kwargs: Optional[Dict[str, object]] = None,
        note: str = "",
    ) -> None:
        self.key = key
        self.title = title
        self.batch = batch
        self.lr = lr
        self.epochs = epochs
        self.model_kwargs = model_kwargs or {}
        self.note = note


SUPERRES_MODELS: List[SuperResModel] = [
    SuperResModel(
        "srcnn",
        "SRCNN",
        batch=16,
        lr=1.0e-4,
        epochs=40,
        model_kwargs={"use_y_channel": True},
        note=(
            "Three convolutional layers on a bicubic-upsampled input: the cheapest baseline\n"
            "there is, and deliberately so - it establishes what a purely local, shallow mapping\n"
            "recovers before any of the deep models get credit for it.\n"
            "\n"
            "`use_y_channel: true` is SRCNN's own convention: the network sees the luma channel\n"
            "only and the chroma is carried over from the bicubic upsample. That is how the\n"
            "published SRCNN numbers are produced, and it also makes this run about a third the\n"
            "cost of the colour version."
        ),
    ),
    SuperResModel(
        "autoencoder",
        "autoencoder refiner",
        batch=8,
        lr=1.0e-4,
        epochs=40,
        note=(
            "An encoder-decoder that refines the bicubic upsample rather than predicting the\n"
            "high-resolution image from nothing. Same input convention as SRCNN, more capacity,\n"
            "no adversarial term - so if this does not beat SRCNN, the extra depth is not paying\n"
            "for itself and the GAN below has to justify its instability."
        ),
    ),
    SuperResModel(
        "srgan",
        "SRGAN",
        batch=4,
        lr=1.0e-4,
        epochs=60,
        model_kwargs={"adv_weight": 1.0e-3},
        note=(
            "The only model in this task with a discriminator, and the only one whose loss is\n"
            "adversarial. That has a consequence for how its numbers should be read: an\n"
            "adversarial term improves perceived sharpness and *can* lower PSNR and SSIM while\n"
            "producing images a human prefers. Report all three metrics for this row, and do not\n"
            "quietly drop it from the table if it loses on PSNR - the disagreement is the finding.\n"
            "\n"
            "`adv_weight` is the adversarial term's weight against the pixel loss. 1e-3 is the\n"
            "SRGAN paper's order of magnitude; above ~1e-2 the run usually collapses into\n"
            "artefacts, which shows up as a falling PSNR with a rising discriminator accuracy in\n"
            "metrics.csv. 60 epochs rather than 40 because GAN training needs longer to settle."
        ),
    ),
    SuperResModel(
        "swinir",
        "SwinIR",
        batch=4,
        lr=2.0e-4,
        epochs=40,
        model_kwargs={
            "swinir_embed_dim": 96,
            "swinir_depths": [6, 6, 6, 6],
            "swinir_num_heads": [6, 6, 6, 6],
            "swinir_window": 7,
            "swinir_mlp_ratio": 4.0,
            "swinir_upsampler": "pixelshuffle",
            "swinir_checkpoint": False,
        },
        note=(
            "Window-attention transformer: the heaviest SR baseline and, unlike the other\n"
            "three, one whose cost grows with the *input* size rather than being fixed by the\n"
            "network. That is what train.patch_size is for - the other three models train on the\n"
            "whole image.\n"
            "\n"
            "The architecture keys in model_kwargs (embed_dim, depths, num_heads, window,\n"
            "mlp_ratio, upsampler) are fixed at build time and have to match the checkpoint, so\n"
            "they are recorded in the run's config instead of being left to the script's\n"
            "defaults. Set `swinir_checkpoint: true` to trade ~30% of the speed for a much\n"
            "smaller activation footprint, which is the lever to reach for if x8 runs out of\n"
            "memory. train.amp is also passed for this model only; the script implements AMP for\n"
            "SwinIR and for nothing else."
        ),
    ),
]

#: The scale ladder of this task. `resolution` names the *input* scale exactly as it does
#: for depth and segmentation - 1024 means the `_lr/x2` folder - and the upscale factor is
#: derived from it rather than configured, so a file cannot say resolution: 512 and mean x2.
SR_RESOLUTION_NOTE = {
    2048: (
        "resolution: 2048 means scale x1: the LR input and the HR target are the *same* folder\n"
        "(rgb_day), and the model is an image-to-image refiner rather than an upsampler. It is\n"
        "a legitimate control row - it isolates how much of a x4 result is restoration rather\n"
        "than resolution, which is the question worth asking when the low-resolution folders\n"
        "were produced by downsampling the same captures - but it is not super-resolution, and\n"
        "the run's own summary says so. Nothing is upsampled here."
    ),
    1024: (
        "resolution: 1024 means scale x2: the LR input is rgb_day_lr/x2 and the HR target is\n"
        "native rgb_day. The factor is derived from the resolution, never configured beside it,\n"
        "so `resolution: 1024` cannot mean x4 in one file and x2 in another."
    ),
    512: (
        "resolution: 512 means scale x4: the LR input is rgb_day_lr/x4 and the HR target is\n"
        "native rgb_day. The factor is derived from the resolution, never configured beside it."
    ),
    256: (
        "resolution: 256 means scale x8: the LR input is rgb_day_lr/x8 and the HR target is\n"
        "native rgb_day. This is the hardest cell of the ladder and the one where a model that\n"
        "merely smooths and one that reconstructs separate most clearly - watch SSIM and the\n"
        "sample panels, not just PSNR, which rewards blur."
    ),
}


# ------------------------------------------------------------------------ writers



def _banner(parts: List[str]) -> str:
    """Comment a block: every line becomes `# ...`, blank lines become a bare `#`.

    Elements may themselves be multi-line - the model notes are written as prose
    paragraphs with hard wraps - so each one is split before it is prefixed. Otherwise
    only the first line of a paragraph would be commented and the rest would land in the
    YAML stream as syntax errors.
    """
    flat: List[str] = []
    for part in parts:
        flat.extend(part.split("\n"))
    return "\n".join(("# " + line).rstrip() for line in flat)


def _render(header: List[str], body: List[str]) -> str:
    """A config file: the prose header as comments, then the YAML it describes."""
    return _banner(header) + "\n\n" + "\n".join(body) + "\n"


def depth_yaml(model: DepthModel, resolution: int) -> str:
    where = f"configs/depth/{resolution}/{model.key}_{resolution}.yaml"
    header = [
        f"{model.title} at {resolution}px ({SCALE_SUFFIX[resolution]} of native).",
        "",
        f"    symtrs train {where} --data-root /data/SyMTRS",
        f"    symtrs eval  {where} --data-root /data/SyMTRS --eval-split test",
        "",
        RESOLUTION_NOTE[resolution],
        "",
    ]
    if resolution == 2048:
        header += model.note.split("\n") + [""]
        header += [
            "The other three cells of this row are the same file at 1024, 512 and 256, in the",
            f"sibling folders. `symtrs sweep --task depth --models {model.key}` walks them.",
            "",
        ]
    else:
        header += [
            f"{model.title} is described in full at configs/depth/2048/{model.key}_2048.yaml -",
            "read that one for the protocol and the memory posture. What changes here is only",
            "the numbers on the resolution axis.",
            "",
            f"This model is {model.protocol}-depth at every resolution, so the loss stays",
            f"{model.loss} and eval.align stays {model.align}. A row that switches protocol",
            "between resolutions is not comparable to itself.",
            "",
        ]

    body = [
        "task: depth",
        f"model: {model.key}",
        f"resolution: {resolution}",
        "seed: 42",
        "",
    ]
    if model.model_kwargs:
        body.append("model_kwargs:")
        for key, value in model.model_kwargs.items():
            body.append(f"  {key}: {_scalar(value)}")
        body.append("")
    body += [
        "data:",
        "  root: /data/SyMTRS",
        "  split: train",
        "  val_split: val",
        "  test_split: test",
        "  # Metres are what the metric models predict. Check the raw encoding before trusting",
        "  # it: 0.01 for millimetres stored as uint16, 1.0 for metres.",
        "  depth_scale: 1.0",
        "  depth_channel: 0",
        "  depth_backend: auto",
        "  num_workers: 8",
        "  pin_memory: true",
        "",
        "train:",
        "  epochs: 20",
        f"  batch_size: {model.batch[resolution]}",
        f"  lr: {_float(model.lr)}",
        "  weight_decay: 0.01",
        "  optim: adamw",
        "  sched: cosine",
        "  warmup_epochs: 1",
        "  min_lr_ratio: 0.01",
    ]
    patch = model.patch.get(resolution, 0)
    if patch:
        body += [
            f"  # Random {patch}px crops: a full {resolution}px batch is more memory than this",
            "  # encoder can hold, and the crop still shows a representative slice of the scene",
            "  # at this ground sampling distance.",
            f"  patch_size: {patch}",
        ]
    else:
        body += [
            "  # The whole image fits at this resolution, so no cropping is needed.",
            "  patch_size: 0",
        ]
    body += [
        "  val_patch_size: 0",
        "  amp: true",
        "  gpus: 1",
        "  grad_clip: 1.0",
        f"  grad_accum: {model.grad_accum}",
    ]
    if model.grad_accum > 1:
        effective = model.batch[resolution] * model.grad_accum
        body += [
            f"  # Effective batch = {model.batch[resolution]} x {model.grad_accum} = {effective},",
            "  # which is what the learning rate was chosen against.",
        ]
    body += ["  loss: " + model.loss]
    body.append(f"  freeze: {model.freeze}" if model.freeze else '  freeze: ""')
    body += [
        "  log_every: 50",
        "  save_samples_every: 1",
        "  device: cuda",
        "",
        "eval:",
        "  # The first name here that the depth scorer can minimise decides which checkpoint",
        "  # weights/best.pt holds. The rest are reported but do not select it.",
        "  metrics: [rmse, mae, abs_rel, sq_rel, d1, d2, d3, silog]",
        "  batch_size: 1",
        "  amp: true",
        "  # 0 runs the whole image in one forward pass, which is what the published numbers",
        "  # use. If you hit OOM, a sweep over 512/128 tiles scores the same metres.",
        "  tile_size: 0",
        "  tile_overlap: 128",
        f"  align: {model.align}",
        "  num_visualizations: 4",
    ]
    return _render(header, body)


def segmentation_yaml(model: SegModel, resolution: int) -> str:
    where = f"configs/segmentation/{resolution}/{model.key}_{resolution}.yaml"
    header = [
        f"{model.title} at {resolution}px.",
        "",
        f"    symtrs train {where} --data-root /data/SyMTRS",
        f"    symtrs eval  {where} --data-root /data/SyMTRS --eval-split test",
        "",
        SEG_RESOLUTION_NOTE[resolution],
        "",
        SEG_CLASS_NOTE,
        "",
    ]
    if resolution == 2048:
        header += model.note.split("\n") + [""]
        header += [
            "The other three cells of this row are the same file at 1024, 512 and 256, in the",
            f"sibling folders. `symtrs sweep --task segmentation --models {model.key}` walks",
            "them.",
            "",
        ]
    else:
        header += [
            f"{model.title} is described in full at",
            f"configs/segmentation/2048/{model.key}_2048.yaml - read that one for why it uses",
            "this loss and this learning rate. What changes here is only the resolution axis.",
            "",
        ]

    body = [
        "task: segmentation",
        f"model: {model.key}",
        f"resolution: {resolution}",
        "seed: 42",
        "",
        "data:",
        "  root: /data/SyMTRS",
        "  split: train",
        "  val_split: val",
        "  test_split: test",
        "  num_classes: 6",
        "  # Pixels labelled 255 are excluded from both the loss and the confusion matrix.",
        "  ignore_index: 255",
        "  num_workers: 8",
        "  pin_memory: true",
        "",
        "train:",
        "  epochs: 40",
        f"  batch_size: {model.batch[resolution]}",
        f"  lr: {_float(model.lr)}",
        "  weight_decay: 0.01",
        "  optim: adamw",
        "  sched: cosine",
        "  warmup_epochs: 2",
        "  min_lr_ratio: 0.01",
        f"  patch_size: {model.patch.get(resolution, 0)}",
        "  val_patch_size: 0",
        "  amp: true",
        "  gpus: 1",
        "  grad_clip: 1.0",
        f"  grad_accum: {model.grad_accum}",
        f"  loss: {model.loss}",
    ]
    if model.loss_kwargs:
        body.append("  loss_kwargs:")
        for key, value in model.loss_kwargs.items():
            body.append(f"    {key}: {_scalar(value)}")
    body += [
        '  freeze: ""',
        "  log_every: 50",
        "  save_samples_every: 1",
        "  device: cuda",
        "",
        "eval:",
        "  metrics: [miou, pixel_acc, mean_acc, mean_dice, fwiou]",
        "  batch_size: 1",
        "  amp: true",
        "  tile_size: 0",
        "  tile_overlap: 128",
        '  align: ""',
        "  num_visualizations: 4",
    ]
    return _render(header, body)


def domain_yaml(model: DomainModel, direction: str) -> str:
    source, target = direction.split("2")
    where = f"configs/domain_adapt/{model.key}_{direction}.yaml"
    header = [
        f"{model.title} {source} -> {target}, at native resolution.",
        "",
        f"    symtrs train {where} --data-root /data/SyMTRS",
        f"    symtrs eval  {where} --data-root /data/SyMTRS --eval-split test",
        "",
        f"Both directions of this pair are in this folder: {model.key}_day2night.yaml and",
        f"{model.key}_night2day.yaml. They are separate runs with separate checkpoints, and",
        "the direction changes the upstream experiment name too, so the two cannot overwrite",
        "each other. Report both or neither - a quality claim in one direction says nothing",
        "about the other, since the two domains are not equally hard.",
        "",
    ]
    header += model.note.split("\n") + [""]
    header += [
        "This is not a SyMTRS model and the run does not train one. The network is upstream",
        "junyanz/pytorch-CycleGAN-and-pix2pix, built, optimised and saved entirely inside that",
        "project, which is vendored in this repository under pytorch-CycleGAN-and-pix2pix/ at",
        "a pinned revision and launched as a subprocess. Nothing is cloned or downloaded at",
        "run time.",
        "",
        "What SyMTRS adds on top is the input staging and the scoring:",
        "",
        "  * Before each phase, the split's day and night folders are symlinked into",
        "    <run_dir>/staging/<split>/{train,val,test}{A,B}. That is not decoration.",
        "    Upstream's UnalignedDataset re-splits the data by --train_ratio whenever phase",
        "    folders are absent, which would ignore the SyMTRS train/val/test split entirely.",
        "  * After training, test.py generates the translations and the result is scored with",
        "    FID and KID against the real target domain.",
        "",
        "`symtrs setup-adapt --check` verifies the vendored tree in one call;",
        "`symtrs setup-adapt --fetch` diffs it against upstream at the pinned revision.",
        "",
        "Two dependencies are not optional here: upstream's train.py and test.py both import",
        "dominate and wandb (through util/visualizer.py) before they parse an option, so the",
        "run cannot start without them. `pip install -e '.[adapt]'` installs both, and",
        "`symtrs doctor` reports them.",
        "",
    ]
    body = [
        "task: domain_adapt",
        f"model: {model.key}",
        "resolution: 2048",
        "seed: 42",
        "",
        "model_kwargs:",
        f"  direction: {direction}",
        f"  # {model.dataset_mode} is derived from the model key automatically; it is stated",
        "  # here so the pairing assumption is visible in the file that depends on it.",
        f"  dataset_mode: {model.dataset_mode}",
        '  # gpu_ids: "1"      # which GPUs upstream may see, as CUDA_VISIBLE_DEVICES.',
        "  # Upstream picks its own device inside the process and has no --gpu_ids flag,",
        "  # so this is the lever, and train.device: cpu hides every GPU.",
        "  # Anything upstream's train.py / test.py accepts can go in extra_args. A typo is",
        "  # reported by upstream's own argument parser, which is the parser that matters.",
        "  extra_args: []",
        "  # name: a_custom_experiment_name",
        "",
        "data:",
        "  root: /data/SyMTRS",
        "  split: train",
        "  val_split: val",
        "  test_split: test",
        "  num_workers: 8",
        "  pin_memory: true",
        "",
        "train:",
        "  # The *total* number of epochs, as for every other task here. Upstream takes two",
        "  # flags (--n_epochs and --n_epochs_decay) and defaults the second to 100, so the",
        "  # wrapper splits this budget following CycleGAN's own flat-then-decay shape: two",
        "  # thirds flat, one third decaying to zero. Set model_kwargs.decay_epochs to override",
        "  # the split, or put --n_epochs_decay in extra_args to take the schedule over.",
        "  epochs: 40",
        "  batch_size: 1",
        "  lr: 2.0e-4",
        "  amp: true",
        "  gpus: 1",
        "  device: cuda",
        "",
        "eval:",
        "  # FID and KID are computed outside the tensor pipeline, over the folders upstream",
        "  # wrote to, so nothing here selects them; eval.batch_size is the batch used for the",
        "  # Inception feature extraction.",
        "  metrics: []",
        "  batch_size: 8",
        "  amp: true",
        '  align: ""',
        "  num_visualizations: 4",
        "",
        "# Extra arguments for the upstream script, on top of model_kwargs.extra_args. Kept",
        "# separate so a CLI `--set model_kwargs.extra_args=...` does not clobber them.",
        "upstream_args: []",
    ]
    return _render(header, body)


def superresolution_yaml(model: SuperResModel, resolution: int) -> str:
    scale = SCALE_FACTOR[resolution]
    where = f"configs/superresolution/{model.key}_x{scale}.yaml"
    header = [
        f"{model.title} at x{scale} (input {resolution}px -> target 2048px).",
        "",
        f"    symtrs train {where} --data-root /data/SyMTRS",
        f"    symtrs eval  {where} --data-root /data/SyMTRS --eval-split test",
        "",
        SR_RESOLUTION_NOTE[resolution],
        "",
    ]
    if scale == 1:
        header += model.note.split("\n") + [""]
        header += [
            f"The real super-resolution cells of this row are {model.key}_x2.yaml,",
            f"{model.key}_x4.yaml and {model.key}_x8.yaml in this folder.",
            "",
        ]
    else:
        header += [
            f"{model.title} is described in full at configs/superresolution/",
            f"{model.key}_x1.yaml - read that one for the loss, the batch size and the",
            "model-specific caveats. Only the scale changes here.",
            "",
        ]
        if model.key == "swinir":
            header += [
                "This is the one model whose training cost grows with the input size, so",
                "train.patch_size matters more as the scale rises: the crop is in LR pixels,",
                f"so a 64px crop is a {64 * scale}px region of the target. Raise it if memory",
                "allows, and set model_kwargs.swinir_checkpoint: true at x8 if it does not.",
                "",
            ]
    header += [
        "The training code for this task predates the `symtrs` package and now lives inside",
        "it: `symtrs/legacy/superres/train.py` owns the model set, the data loading and the",
        "metrics, launched as `python -m symtrs.legacy.superres.train`. `symtrs` translates",
        "this config into that command line and reads test_metrics.json back.",
        "`symtrs train <this file> --dry-run` prints it without launching anything, which is",
        "the quickest way to see which two folders a scale will read.",
        "",
    ]

    body = [
        "task: superresolution",
        f"model: {model.key}",
        f"resolution: {resolution}",
        "seed: 42",
        "",
    ]
    if model.model_kwargs:
        body.append("model_kwargs:")
        for key, value in model.model_kwargs.items():
            body.append(f"  {key}: {_scalar(value)}")
        if model.key == "srcnn":
            body.append("  # Set use_y_channel false to train on RGB instead of luma; the")
            body.append("  # published SRCNN numbers are Y-channel, so changing it also changes")
            body.append("  # what you are comparing against.")
        body.append("")
    body += [
        "data:",
        "  root: /data/SyMTRS",
        "  split: train",
        "  val_split: val",
        "  test_split: test",
        "  num_workers: 8",
        "  pin_memory: true",
        "",
        "train:",
        f"  epochs: {model.epochs}",
        f"  batch_size: {model.batch}",
        f"  lr: {_float(model.lr)}",
        "  save_samples_every: 1",
        "  gpus: 1",
        "  device: cuda",
        "  # A path here resumes a previous run; `symtrs eval` scores the run directory's",
        "  # weights/best.pt without training at all.",
        '  resume: ""',
    ]
    if model.key == "swinir":
        body += [
            "  # LR-space crops, SwinIR only: the other three models train on the whole image.",
            "  patch_size: 64",
            "  val_patch_size: 0",
            "  amp: true",
        ]
    body += [
        "",
        "eval:",
        "  # The super-resolution scorer always reports PSNR, SSIM and MSE; `metrics` is the",
        "  # reported order, and the first name here is what `symtrs tables` prints first.",
        "  metrics: [psnr, ssim, mse]",
        "  batch_size: 1",
        "  amp: true",
        "  num_visualizations: 4",
    ]
    return _render(header, body)


def _float(value: float) -> str:
    """`1.0e-5` rather than `1e-05`: the same value, and it reads like the other configs."""
    text = f"{value:.1e}"
    mantissa, exponent = text.split("e")
    if "." not in mantissa:
        mantissa += ".0"
    return f"{mantissa}e{int(exponent)}"


def _scalar(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_scalar(item) for item in value) + "]"
    if isinstance(value, float):
        return _float(value)
    return str(value)


def build_tree() -> Dict[Path, str]:
    """path -> file contents, for every file this script owns."""
    files: Dict[Path, str] = {}
    for model in DEPTH_MODELS:
        for resolution in RESOLUTIONS:
            path = HERE / "depth" / str(resolution) / f"{model.key}_{resolution}.yaml"
            files[path] = depth_yaml(model, resolution)
    for model in SEG_MODELS:
        for resolution in RESOLUTIONS:
            path = HERE / "segmentation" / str(resolution) / f"{model.key}_{resolution}.yaml"
            files[path] = segmentation_yaml(model, resolution)
    for model in DOMAIN_MODELS:
        for direction in DOMAIN_DIRECTIONS:
            path = HERE / "domain_adapt" / f"{model.key}_{direction}.yaml"
            files[path] = domain_yaml(model, direction)
    for model in SUPERRES_MODELS:
        for resolution in RESOLUTIONS:
            scale = SCALE_FACTOR[resolution]
            path = HERE / "superresolution" / f"{model.key}_x{scale}.yaml"
            files[path] = superresolution_yaml(model, resolution)
    return files


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true",
                        help="report files that are missing or differ, and write nothing")
    args = parser.parse_args(argv)

    files = build_tree()
    stale: List[Path] = []

    for path, text in sorted(files.items()):
        if path.is_file() and path.read_text(encoding="utf-8") == text:
            continue
        if args.check:
            stale.append(path)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    if args.check:
        if stale:
            print(f"{len(stale)} file(s) out of date; run "
                  f"`python3 configs/generate_grid.py` to rewrite them:")
            for path in stale:
                print(f"  {path.relative_to(HERE.parent)}")
            return 1
        print(f"configs/ is up to date ({len(files)} generated files)")
        return 0

    def count(task: str) -> int:
        # Every task owns a folder under configs/; the resolution subfolders are one level
        # deeper for the four-resolution tasks and absent for the other two.
        return sum(1 for path in files if task in path.relative_to(HERE).parts[:-1])

    depth = count("depth")
    seg = count("segmentation")
    domain = count("domain_adapt")
    sr = count("superresolution")
    print(f"wrote {len(files)} configs")
    print(f"  depth           {depth:3}  ({len(DEPTH_MODELS)} models x {len(RESOLUTIONS)} resolutions)")
    print(f"  segmentation    {seg:3}  ({len(SEG_MODELS)} models x {len(RESOLUTIONS)} resolutions)")
    print(f"  domain_adapt    {domain:3}  ({len(DOMAIN_MODELS)} models x {len(DOMAIN_DIRECTIONS)} directions)")
    print(f"  superresolution {sr:3}  ({len(SUPERRES_MODELS)} models x {len(RESOLUTIONS)} scales)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
