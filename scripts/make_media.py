#!/usr/bin/env python3
"""Generate the README and project-page media from a real SyMTRS root.

Every picture here is built from the dataset's `test/` split and from checkpoints that
already exist under `runs/`, so a reader sees scenes no model in this repository trained
on. Nothing is hand-edited: re-run this and the same files come back.

    python scripts/make_media.py --data-root /mnt/ssd1/SyMTRS
    python scripts/make_media.py --data-root /mnt/ssd1/SyMTRS --only hero,grid

Writes under `docs/assets/media/`:

  hero.gif, hero.webm          the five modalities of one scene, cross-faded
  slider-*.gif / .webm         A/B wipes: day/night, shadow, depth, segmentation, SR
  grid-tasks.png               5 tasks x (input, model, ground truth)
  pairs/*.jpg                  the same A/B pairs as stills, for the web sliders
  media.json                   which scenes, split and checkpoints were used

The A/B clips are wipes rather than cross-fades on purpose: a cross-fade of a day and a
night frame reads as a brightness change, while a divider that sweeps across the same
frame lets a reader see that the *geometry* is identical and only the domain changed.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]

#: A dark, high-contrast treatment. The project page is dark; README figures that carry
#: their own light background look pasted-on beside it, so the panels are dark too.
BG = (10, 13, 17)
FG = (231, 237, 243)
MUTED = (139, 152, 166)
ACCENT = (94, 230, 168)

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans" + ("-Bold" if bold else "")
    return ImageFont.truetype(str(FONT_DIR / f"{name}.ttf"), size)


# --------------------------------------------------------------------------- scenes

#: The five classes of `data.num_classes`, background included. Brighter than the
#: training preview palette (`symtrs.common.plots`) because these panels are read at a
#: glance, not compared pixel by pixel; the class *order* is the dataset's.
CLASS_RGB = np.array(
    [
        [26, 30, 36],      # background
        [226, 84, 60],     # building
        [76, 175, 102],    # green
        [126, 136, 150],   # road
        [242, 193, 78],    # car
    ],
    dtype=np.uint8,
)

#: One scene per story, all from `test/`. Chosen by eye for what the task is about:
#: an intersection (all five classes, cast shadows), a building canyon (depth range), a
#: parking lot (cars), a motorway interchange (fine lane markings and night lights).
SCENES = {
    "hero": "rgb_day_2048_loc439_rot120deg_h00_x+049937_y-032621_z+007060.png",
    "daynight": "rgb_day_2048_loc461_rot274deg_h02_x+049266_y-005273_z+013063.png",
    "shadow": "rgb_day_2048_loc439_rot256deg_h02_x+049937_y-032621_z+013060.png",
    "depth": "rgb_day_2048_loc470_rot319deg_h02_x+056284_y+031034_z+015570.png",
    "segmentation": "rgb_day_2048_loc439_rot108deg_h02_x+049937_y-032621_z+013060.png",
    "superresolution": "rgb_day_2048_loc461_rot274deg_h02_x+049266_y-005273_z+013063.png",
}

#: Checkpoints the prediction panels are drawn with, named so a caption can state which
#: model produced them.
SEG_CKPT = "runs/campaign2/segformer_b2_2048/weights/best.pt"
DEPTH_CKPT = "runs/campaign2/midas_dpt_hybrid_2048/weights/best.pt"
SR_RUN = "runs/campaign2/swinir_x8"
SHADOW_RUN = "runs/campaign2/restormer"
CYCLEGAN_RUN = "runs/campaign2/cyclegan_day2night"


def key_of(day_name: str) -> str:
    """`rgb_day_2048_loc...png` -> `2048_loc...png`, the stem every folder shares."""
    return re.sub(r"^rgb_day_", "", day_name)


def plain_key(day_name: str) -> str:
    """The stem upstream and the shadow runner write: no `rgb_day_` and no `2048_`."""
    return key_of(day_name).split("_", 1)[1][:-4]


def scene_paths(root: Path, split: str, day_name: str) -> Dict[str, Path]:
    key = key_of(day_name)
    return {
        "day": root / split / "rgb_day" / day_name,
        "night": root / split / "rgb_night" / f"rgb_night_{key}",
        "noshadow": root / split / "rgb_noshadow" / f"rgb_noshadow_{key}",
        "depth": root / split / "depth" / f"depth_day_{key[:-4]}.exr",
        "semantic": root / split / "semantic" / f"seg_day_{key}",
    }


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def square(image: Image.Image) -> Image.Image:
    side = min(image.size)
    left = (image.width - side) // 2
    top = (image.height - side) // 2
    return image.crop((left, top, left + side, top + side))


def crop_vertical(image: Image.Image, top: float, bottom: float) -> Image.Image:
    """The 16:9 band of a square frame, as a fraction, for the wide hero crop."""
    height = image.height
    return image.crop((0, int(top * height), image.width, int(bottom * height)))


def fit(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    """Cover-crop to `size` (never letterbox: a black band reads as a broken asset)."""
    sw, sh = size
    scale = max(sw / image.width, sh / image.height)
    resized = image.resize(
        (max(sw, round(image.width * scale)), max(sh, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    left = (resized.width - sw) // 2
    top = (resized.height - sh) // 2
    return resized.crop((left, top, left + sw, top + sh))


def detailed_window(image: Image.Image, side: int) -> Image.Image:
    """The `side` x `side` window with the most edge energy.

    The middle of a frame is often flat road; a super-resolution panel that shows flat road
    shows nothing. Picking the window with the most gradient is a deterministic stand-in
    for "choose a crop by eye", and it survives a new dataset without an edit here.
    """
    small = 256
    gray = np.asarray(image.convert("L").resize((small, small), Image.Resampling.LANCZOS),
                      np.float32)
    energy = np.zeros((small, small), np.float32)
    energy[:, :-1] += np.abs(np.diff(gray, axis=1))
    energy[:-1, :] += np.abs(np.diff(gray, axis=0))
    integral = np.pad(energy.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    win = max(1, round(side / image.width * small))
    totals = (integral[win:, win:] - integral[:-win, win:]
              - integral[win:, :-win] + integral[:-win, :-win])
    top, left = np.unravel_index(int(np.argmax(totals)), totals.shape)
    scale = image.width / small
    x, y = int(left * scale), int(top * scale)
    return image.crop((x, y, x + side, y + side))


def colorize_depth(depth: np.ndarray, vmin: float | None = None,
                   vmax: float | None = None) -> np.ndarray:
    """Turbo-colour a depth map, on the ground truth's scale by default.

    A model panel colourised on its own quantiles and a ground-truth panel on theirs put the
    same colour at two different depths; the depth row of the grid passes one shared scale.
    """
    import matplotlib

    matplotlib.use("Agg")
    values = depth[np.isfinite(depth) & (depth > 0)]
    if vmin is None:
        vmin = float(np.quantile(values, 0.02)) if values.size else 0.0
    if vmax is None:
        vmax = float(np.quantile(values, 0.98)) if values.size else 1.0
    normalized = np.clip((depth - vmin) / max(float(vmax) - float(vmin), 1e-6), 0.0, 1.0)
    return (matplotlib.colormaps["turbo"](normalized)[..., :3] * 255).round().astype(np.uint8)


#: The dataset's EXR stores depth scaled by 100, exactly as `data.depth_scale: 0.01` says
#: in every config: a raw read reports 2,260-7,790 for a scene whose metric depth is
#: 22-78 m. Reading it straight and aligning a disparity against it clamps the whole map to
#: `1 / max_depth` and draws one flat colour, which is how this was found.
DEPTH_SCALE = 0.01


def read_depth(path: Path) -> np.ndarray:
    from symtrs.data.exr import read_depth_file

    return read_depth_file(path).astype(np.float32) * DEPTH_SCALE


def semantic_rgb(mask: Image.Image) -> Image.Image:
    values = np.asarray(mask.convert("L"))
    index = values.astype(np.int32) // 63
    out = np.zeros(values.shape + (3,), np.uint8)
    valid = (values != 255) & (index < len(CLASS_RGB))
    out[valid] = CLASS_RGB[index[valid]]
    return Image.fromarray(out)


# --------------------------------------------------------------------------- models


def seg_prediction(ckpt: Path, image: Image.Image) -> Image.Image:
    from symtrs.model import autoloader

    classes = np.asarray(autoloader(str(ckpt)).predict(np.asarray(image)))
    while classes.ndim > 2:
        classes = classes[0]
    classes = classes.astype(np.int32)
    out = np.zeros(classes.shape + (3,), np.uint8)
    valid = (classes >= 0) & (classes < len(CLASS_RGB))
    out[valid] = CLASS_RGB[classes[valid]]
    return Image.fromarray(out)


def depth_prediction(ckpt: Path, image: Image.Image, target: np.ndarray) -> np.ndarray:
    """MiDaS raw disparity, aligned to the ground truth with the MiDaS protocol.

    `SymTRSModel.predict` is not used here: it returns `1 / disparity` ("what `predict`
    returns for qualitative use", `DepthBase.to_metres`), and a scale-and-shift fit in depth
    space against values that already contain a `1 / x` leaves one or two near-zero
    disparities dominating the least squares — the whole map then collapses to a single
    colour. The raw output comes from `forward_native`, the same call the evaluation pass
    aligns, so this panel and the results table describe the same map.
    """
    import torch

    from symtrs.engine.loop import forward_native
    from symtrs.metrics.depth import align_prediction
    from symtrs.model import autoloader
    from symtrs.model.wrapper import to_tensor

    model = autoloader(str(ckpt))
    tensor, _ = to_tensor(image)
    with torch.no_grad():
        native = forward_native(model.net, tensor.to(model.device)).cpu().float()

    gt = torch.from_numpy(target)[None, None]
    mask = torch.isfinite(gt) & (gt > 0) & torch.isfinite(native) & (native > 0)
    aligned, _ = align_prediction(native, gt, mask, mode="scale_shift_disparity")
    return aligned[0, 0].numpy()


def sr_prediction(run: Path, lr: Image.Image) -> Image.Image:
    """SwinIR x8 output for one LR crop, built exactly as `evaluate_on_test` builds it.

    The architecture and the preprocessing come from the legacy trainer rather than a
    second copy here, so the panel cannot drift from what the checkpoint was trained for.
    """
    import argparse as _argparse

    import torch
    import yaml

    from symtrs.legacy.superres.models import SwinIR
    from symtrs.legacy.superres.train import super_resolve

    kwargs = yaml.safe_load((run / "config.yaml").read_text()).get("model_kwargs", {})
    args = _argparse.Namespace(
        model="swinir",
        scale=8,
        use_y_channel=False,
        amp=False,
        patch_size=64,
        eval_tile_size=512,
        eval_tile_overlap=64,
        swinir_embed_dim=kwargs.get("swinir_embed_dim", 96),
        swinir_depths=kwargs.get("swinir_depths", [6, 6, 6, 6]),
        swinir_num_heads=kwargs.get("swinir_num_heads", [6, 6, 6, 6]),
        swinir_window=kwargs.get("swinir_window", 7),
        swinir_mlp_ratio=kwargs.get("swinir_mlp_ratio", 4.0),
        swinir_upsampler=kwargs.get("swinir_upsampler", "pixelshuffle"),
        swinir_checkpoint=False,
    )
    # `swinir_img_size` in the trainer rounds the 64 px training crop up to a whole number
    # of 7 px windows; the same value keeps the checkpoint's mask buffers consistent. At
    # inference the mask is recomputed on the fly, which requires the LR crop to be a
    # multiple of the window too (63 = 9 x 7), so `super_resolve` is given 63 px inputs.
    model = SwinIR(
        img_size=70,
        in_chans=3,
        embed_dim=args.swinir_embed_dim,
        depths=args.swinir_depths,
        num_heads=args.swinir_num_heads,
        window_size=args.swinir_window,
        mlp_ratio=args.swinir_mlp_ratio,
        upscale=args.scale,
        img_range=1.0,
        upsampler=args.swinir_upsampler,
        use_checkpoint=False,
    )
    model.load_state_dict(torch.load(run / "weights" / "best.pt", map_location="cpu")["model"])
    model.eval()

    tensor = torch.from_numpy(np.asarray(lr, np.float32) / 255.0).permute(2, 0, 1)[None]
    with torch.no_grad():
        out = super_resolve(args, model, tensor)
    out = out[0].clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray((out * 255).round().astype(np.uint8))


def bicubic(lr: Image.Image, size: Tuple[int, int]) -> Image.Image:
    return lr.resize(size, Image.Resampling.BICUBIC)


# --------------------------------------------------------------------------- chrome


def text_size(draw: ImageDraw.ImageDraw, text: str, f: ImageFont.FreeTypeFont) -> Tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=f)
    return box[2] - box[0], box[3] - box[1]


def chip(draw: ImageDraw.ImageDraw, xy, text, *, f, fill=(0, 0, 0, 165), color=FG) -> int:
    """A translucent label chip; returns its width so callers can right-align one."""
    x, y = xy
    pad_x, pad_y = 12, 7
    tw, th = text_size(draw, text, f)
    width, height = tw + 2 * pad_x, th + 2 * pad_y
    draw.rounded_rectangle((x, y, x + width, y + height), radius=8, fill=fill)
    draw.text((x + pad_x, y + pad_y - 1), text, font=f, fill=color)
    return width


def bottom_scrim(image: Image.Image, height: int = 175, strength: int = 235) -> Image.Image:
    """A dark gradient at the bottom, so white captions stay readable over any scene.

    Near-linear rather than a steep curve: the tag line sits high in the band, and over a
    sunlit pavement a curve dark enough at the title left the accent-green tag unreadable.
    """
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for i in range(height):
        alpha = int(strength * (i / height) ** 1.05)
        y = image.height - height + i
        draw.line([(0, y), (image.width, y)], fill=(0, 0, 0, alpha))
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def draw_caption(image: Image.Image, tag: str, title: str, subtitle: str = "",
                 progress: Tuple[int, int] = (0, 0)) -> Image.Image:
    image = bottom_scrim(image)
    draw = ImageDraw.Draw(image)
    f_tag, f_title = font(14, bold=True), font(27, bold=True)
    x, y = 28, image.height - 104
    draw.text((x, y), tag.upper(), font=f_tag, fill=ACCENT)
    draw.text((x, y + 22), title, font=f_title, fill=FG)
    if subtitle:
        # The same caption is rendered at 960 px for the video and at 480 px for the README
        # GIF; a fixed 15 px subtitle that fits the wide copy runs off the narrow one.
        size = 15
        while size > 11 and text_size(draw, subtitle, font(size))[0] > image.width - 2 * x:
            size -= 1
        draw.text((x, y + 59), subtitle, font=font(size), fill=MUTED)
    if progress[1]:
        # Five ticks, the current one filled: the reader knows where the loop is.
        current, total = progress
        tick_w, gap = 18, 5
        total_w = total * tick_w + (total - 1) * gap
        start = image.width - 28 - total_w
        for i in range(total):
            colour = ACCENT if i == current else (90, 100, 112)
            draw.rounded_rectangle(
                (start + i * (tick_w + gap), image.height - 26,
                 start + i * (tick_w + gap) + tick_w, image.height - 22),
                radius=2, fill=colour,
            )
    return image


def divider_frame(base: Image.Image, other: Image.Image, x: float) -> Image.Image:
    """`base` left of `x`, `other` right of it, with a handle on the seam."""
    frame = base.copy()
    xi = int(round(max(0, min(base.width, x))))
    if xi < base.width:
        frame.paste(other.crop((xi, 0, other.width, other.height)), (xi, 0))
    draw = ImageDraw.Draw(frame)
    draw.line([(xi, 0), (xi, base.height)], fill=(255, 255, 255), width=2)
    draw.line([(xi + 2, 0), (xi + 2, base.height)], fill=(0, 0, 0), width=1)
    cy, r = base.height // 2, 17
    draw.ellipse([xi - r, cy - r, xi + r, cy + r], fill=BG, outline=(255, 255, 255), width=2)
    draw.line([(xi - 7, cy), (xi - 2, cy)], fill=FG, width=2)
    draw.line([(xi + 2, cy), (xi + 7, cy)], fill=FG, width=2)
    draw.polygon([(xi - 9, cy), (xi - 4, cy - 4), (xi - 4, cy + 4)], fill=FG)
    draw.polygon([(xi + 9, cy), (xi + 4, cy - 4), (xi + 4, cy + 4)], fill=FG)
    return frame


def slider_frames(
    a: Image.Image,
    b: Image.Image,
    tag: str,
    title: str,
    label_a: str,
    label_b: str,
    subtitle: str = "",
    frames: int = 60,
    low: float = 0.12,
    high: float = 0.88,
) -> List[Image.Image]:
    """A-B wipe that sweeps out and back, ping-pong, so the loop has no visible cut."""
    f_label = font(14, bold=True)
    out: List[Image.Image] = []
    for i in range(frames):
        # One full out-and-back, sine-eased at both ends.
        t = (1 - np.cos(2 * np.pi * i / frames)) / 2
        frame = divider_frame(a, b, (low + (high - low) * t) * a.width)
        draw = ImageDraw.Draw(frame, "RGBA")
        chip(draw, (18, 18), label_a, f=f_label)
        width = text_size(draw, label_b, f_label)[0] + 24
        chip(draw, (a.width - 18 - width, 18), label_b, f=f_label)
        out.append(draw_caption(frame, tag, title, subtitle))
    return out


# --------------------------------------------------------------------------- writers


def write_gif(
    frames: Sequence[Image.Image],
    path: Path,
    *,
    size: Tuple[int, int],
    fps: int = 6,
    max_frames: int = 24,
    colors: int = 64,
) -> None:
    """A small, still-recognisable GIF for the README.

    A photographic 720 px frame is ~2.5 MB on its own in GIF, so the README copy is
    downscaled, subsampled and quantised with no dithering; the website plays the full
    `.webm` beside it. The 64-colour palette is chosen per frame, and no dithering is the
    single biggest size lever: Floyd-Steinberg noise more than doubles the file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    picks = np.linspace(0, len(frames) - 1, min(max_frames, len(frames))).round().astype(int)
    paletted = []
    for index in picks:
        frame = frames[int(index)].resize(size, Image.Resampling.LANCZOS)
        paletted.append(frame.quantize(
            colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE
        ))
    paletted[0].save(
        path, save_all=True, append_images=paletted[1:], duration=max(20, round(1000 / fps)),
        loop=0, optimize=True, disposal=2,
    )


def write_webm(frames: Sequence[Image.Image], path: Path, fps: int,
               size: Tuple[int, int]) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"VP90"), fps, (width, height))
    if not writer.isOpened():
        raise SystemExit(f"cannot open a VP9 writer for {path}")
    for frame in frames:
        if frame.size != size:
            frame = frame.resize(size, Image.Resampling.LANCZOS)
        writer.write(np.asarray(frame.convert("RGB"))[:, :, ::-1])
    writer.release()


# --------------------------------------------------------------------------- assets


def build_hero(root: Path) -> List[Image.Image]:
    paths = scene_paths(root, "test", SCENES["hero"])
    depth = read_depth(paths["depth"])
    semantic = Image.open(paths["semantic"])
    top, bottom = 0.18, 0.82
    modalities = [
        ("rgb_day · 2048 px", "RGB · day", "aligned 2048 x 2048 aerial capture",
         crop_vertical(load_rgb(paths["day"]), top, bottom)),
        ("rgb_night", "RGB · night", "same camera and geometry, different illumination",
         crop_vertical(load_rgb(paths["night"]), top, bottom)),
        ("rgb_noshadow", "RGB · shadow-free", "the direct sun removed: cast shadows gone",
         crop_vertical(load_rgb(paths["noshadow"]), top, bottom)),
        ("depth", "Depth · metric", "per-pixel metric depth in metres (EXR)",
         crop_vertical(Image.fromarray(colorize_depth(depth)), top, bottom)),
        ("semantic", "Semantic · 5 classes", "background, building, green, road, car",
         crop_vertical(semantic_rgb(semantic), top, bottom)),
    ]

    size = (960, 540)
    hold, blend = 16, 8
    frames: List[Image.Image] = []
    for index, (tag, title, subtitle, image) in enumerate(modalities):
        composed = fit(image, size)
        for _ in range(hold):
            frames.append(draw_caption(composed.copy(), tag, title, subtitle,
                                       progress=(index, len(modalities))))
        nxt = modalities[(index + 1) % len(modalities)]
        composed_next = fit(nxt[3], size)
        for step in range(blend):
            alpha = (step + 1) / (blend + 1)
            mixed = Image.blend(composed, composed_next, alpha)
            current = index if alpha < 0.5 else (index + 1) % len(modalities)
            frames.append(draw_caption(
                mixed, tag,
                title if alpha < 0.5 else nxt[1],
                subtitle if alpha < 0.5 else nxt[2],
                progress=(current, len(modalities)),
            ))
    return frames


def build_sliders(root: Path) -> Dict[str, Dict[str, Any]]:
    clips: Dict[str, Dict[str, Any]] = {}

    # Day <-> night: the geometry is identical, the domain is not.
    paths = scene_paths(root, "test", SCENES["daynight"])
    day, night = square(load_rgb(paths["day"])), square(load_rgb(paths["night"]))
    clips["daynight"] = _clip(
        day, night, "domain adaptation", "Day / night",
        "RGB · day", "RGB · night",
        "aligned pairs · 2048 px · the split lives in the dataset",
    )

    # Day <-> shadow-free. The dataset ships no shadow mask; the trainer derives one, and
    # the caption says so rather than implying it is annotated.
    paths = scene_paths(root, "test", SCENES["shadow"])
    day, noshadow = square(load_rgb(paths["day"])), square(load_rgb(paths["noshadow"]))
    clips["shadow"] = _clip(
        day, noshadow, "shadow removal", "Shadow / shadow-free",
        "RGB · day", "RGB · shadow-free",
        "ground-truth render · the model's mask is derived at train time",
    )

    # RGB <-> metric depth ground truth.
    paths = scene_paths(root, "test", SCENES["depth"])
    day = square(load_rgb(paths["day"]))
    depth = colorize_depth(read_depth(paths["depth"]))
    clips["depth"] = _clip(
        day, Image.fromarray(depth), "monocular depth", "RGB / metric depth",
        "RGB", "depth · metres",
        "UE5 ground truth · per-pixel metric depth, no stereo or LiDAR",
    )

    # RGB <-> the semantic ground truth.
    paths = scene_paths(root, "test", SCENES["segmentation"])
    day = square(load_rgb(paths["day"]))
    mask = semantic_rgb(square(Image.open(paths["semantic"])))
    clips["segmentation"] = _clip(
        day, mask, "semantic segmentation", "RGB / classes",
        "RGB", "5 classes",
        "background · building · green · road · car · ignore = 255",
    )

    # Bicubic x8 <-> SwinIR x8 from a 63 px LR input (9 x the 7 px window: SwinIR cannot
    # process a mask for any other size). 63 x 8 = 504 HR pixels.
    paths = scene_paths(root, "test", SCENES["superresolution"])
    hr = detailed_window(load_rgb(paths["day"]), 504)
    lr = hr.resize((63, 63), Image.Resampling.BICUBIC)
    sr = sr_prediction(REPO / SR_RUN, lr)
    clips["superresolution"] = _clip(
        bicubic(lr, sr.size), sr, "super-resolution", "Bicubic x8 / SwinIR x8",
        "bicubic x8", "SwinIR x8",
        "same 63 px input · 504 px output",
    )
    return clips


def _clip(a: Image.Image, b: Image.Image, tag: str, title: str,
          label_a: str, label_b: str, subtitle: str = "") -> Dict[str, Any]:
    """Keep the pair and the wording; frames are rendered for each output size.

    A frame rendered at 2048 and then downscaled to 440 puts 8 px captions in the README,
    so each output renders its own frames at its own size.
    """
    return {"a": a, "b": b, "tag": tag, "title": title,
            "label_a": label_a, "label_b": label_b, "subtitle": subtitle}


def build_grid(root: Path) -> Image.Image:
    """5 tasks x 3 columns: what goes in, what the model says, what is true."""
    cell, gap, margin, label_w = 300, 10, 28, 26
    rows = [
        ("depth", "Depth", ["RGB", "MiDaS DPT-Hybrid", "ground truth"]),
        ("segmentation", "Segmentation", ["RGB", "SegFormer-B2", "ground truth"]),
        ("daynight", "Day / night", ["RGB · day", "CycleGAN", "RGB · night"]),
        ("shadow", "Shadow removal", ["RGB · day", "Restormer", "shadow-free"]),
        ("superresolution", "Super-resolution x8", ["bicubic x8", "SwinIR x8", "2048 px"]),
    ]
    width = margin * 2 + label_w + 3 * cell + 2 * gap
    height = 44 + len(rows) * (cell + 40 + gap) + margin
    sheet = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(sheet)
    f_head, f_row, f_small = font(15, bold=True), font(18, bold=True), font(12)

    for col, name in enumerate(["input", "model", "ground truth"]):
        x = margin + label_w + col * (cell + gap)
        draw.text((x, 14), name.upper(), font=f_head, fill=MUTED)

    for r, (scene, title, labels) in enumerate(rows):
        y = 44 + r * (cell + 40 + gap)
        draw.text((margin, y + 6), title, font=f_row, fill=FG)
        draw.line([(margin, y + 30), (width - margin, y + 30)], fill=(42, 48, 55), width=1)
        for c, image in enumerate(_grid_cells(root, scene)):
            x = margin + label_w + c * (cell + gap)
            sheet.paste(fit(image, (cell, cell)), (x, y + 38))
            draw.rectangle([x, y + 38, x + cell - 1, y + 38 + cell - 1], outline=(38, 44, 50))
            draw.text((x, y + 38 + cell + 5), labels[c], font=f_small, fill=MUTED)
    return sheet


def _grid_cells(root: Path, scene: str) -> List[Image.Image]:
    paths = scene_paths(root, "test", SCENES[scene])
    if scene == "depth":
        day = square(load_rgb(paths["day"]))
        target = read_depth(paths["depth"])
        pred = depth_prediction(REPO / DEPTH_CKPT, day, target)
        values = target[np.isfinite(target) & (target > 0)]
        vmin = float(np.quantile(values, 0.02))
        vmax = float(np.quantile(values, 0.98))
        return [day, Image.fromarray(colorize_depth(pred, vmin, vmax)),
                Image.fromarray(colorize_depth(target, vmin, vmax))]
    if scene == "segmentation":
        day = square(load_rgb(paths["day"]))
        return [day, seg_prediction(REPO / SEG_CKPT, day),
                semantic_rgb(square(Image.open(paths["semantic"])))]
    if scene == "daynight":
        key = plain_key(SCENES[scene])
        day = square(load_rgb(paths["day"]))
        generated = square(load_rgb(
            REPO / CYCLEGAN_RUN / "results" / "symtrs_cyclegan_day2night" / "test_latest"
            / "images" / f"{key}_fake_B.png"
        ))
        return [day, generated, square(load_rgb(paths["night"]))]
    if scene == "shadow":
        key = plain_key(SCENES[scene])
        day = square(load_rgb(paths["day"]))
        pred = load_rgb(REPO / SHADOW_RUN / "predictions" / "test" / f"{key}.png")
        return [day, pred, square(load_rgb(paths["noshadow"]))]
    if scene == "superresolution":
        hr = detailed_window(load_rgb(paths["day"]), 504)
        lr = hr.resize((63, 63), Image.Resampling.BICUBIC)
        return [bicubic(lr, hr.size), sr_prediction(REPO / SR_RUN, lr), hr]
    raise KeyError(scene)


def write_pairs(clips: Dict[str, Dict[str, Any]], out: Path) -> None:
    """Stills for the web sliders, at 1100 px so a retina screen still has detail."""
    (out / "pairs").mkdir(parents=True, exist_ok=True)
    for name, clip in clips.items():
        for side in ("a", "b"):
            fit(clip[side], (1100, 1100)).save(
                out / "pairs" / f"{name}-{side}.jpg", quality=86, optimize=True
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/ssd1/SyMTRS"))
    parser.add_argument("--out", type=Path, default=REPO / "docs" / "assets" / "media")
    parser.add_argument("--only", default="hero,sliders,grid,pairs",
                        help="comma-separated subset of: hero,sliders,grid,pairs")
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()
    only = {part.strip() for part in args.only.split(",") if part.strip()}
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "data_root": str(args.data_root),
        "split": "test",
        "scenes": SCENES,
        "checkpoints": {"seg": SEG_CKPT, "depth": DEPTH_CKPT, "sr": SR_RUN,
                        "shadow": SHADOW_RUN, "cyclegan": CYCLEGAN_RUN},
    }

    clips: Dict[str, Dict[str, Any]] = {}
    if "hero" in only:
        frames = build_hero(args.data_root)
        write_gif(frames, out / "hero.gif", size=(720, 405), fps=6, max_frames=30)
        write_webm(frames, out / "hero.webm", fps=args.fps, size=(960, 540))
        print(f"[hero] {len(frames)} frames")
    if "sliders" in only:
        clips = build_sliders(args.data_root)
        for name, clip in clips.items():
            web = slider_frames(
                fit(clip["a"], (720, 720)), fit(clip["b"], (720, 720)),
                clip["tag"], clip["title"], clip["label_a"], clip["label_b"],
                clip["subtitle"], frames=60,
            )
            write_webm(web, out / f"slider-{name}.webm", fps=args.fps, size=(720, 720))
            gif = slider_frames(
                fit(clip["a"], (480, 480)), fit(clip["b"], (480, 480)),
                clip["tag"], clip["title"], clip["label_a"], clip["label_b"],
                clip["subtitle"], frames=16,
            )
            write_gif(gif, out / f"slider-{name}.gif", size=(480, 480),
                      fps=6, max_frames=16)
            print(f"[slider] {name}: gif {len(gif)} frames, webm {len(web)} frames")
    if "grid" in only:
        grid = build_grid(args.data_root)
        out.mkdir(parents=True, exist_ok=True)
        grid.save(out / "grid-tasks.png", optimize=True)
        print(f"[grid] {grid.size}")
    if "pairs" in only:
        if not clips:
            clips = build_sliders(args.data_root)
        write_pairs(clips, out)
        print(f"[pairs] {len(clips) * 2} stills")
    (out / "media.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
