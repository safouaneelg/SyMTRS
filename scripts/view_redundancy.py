#!/usr/bin/env python3
"""Appearance redundancy of the views: the half of R1.2 that geometry cannot answer.

`scripts/footprint_overlap.py` measures how much *ground* two views share. That is the right
measure for the split (R1.3), and it is one half of the diversity question: two views can cover
the same square and still show different pixels, and two views of different squares can look
almost the same if they face the same row of identical facades. Reviewer R1.2 asks the second
half -- are 20,000 views 20,000 samples? -- and the honest answer is an appearance measurement:

    for every view, how similar is the most similar *other* view?

The encoder is DINOv3 ViT-L/16 trained on 493M satellite images (``sat493m``), because the
question is about aerial imagery: it is a frozen, self-supervised backbone with no SyMTRS
training and no labels, so the answer cannot be an artefact of this benchmark's own objective.
It is fetched through ``timm`` and downloaded once into the torch hub cache; if it is not
available the script stops with the command that installs it rather than falling back silently
to a different encoder, because the number means a different thing with a different backbone.

Two numbers come out, and the second is the one the reviewer can use:

  * ``nn_same_location`` -- the share of views whose nearest neighbour in feature space is a
    view of the *same* location (another yaw or altitude). This is the "diversity is pose, not
    scene content" statement, measured rather than asserted;
  * ``sim_different_location`` -- the distribution of each view's best match among views of
    other locations. Its tail is the redundancy: a view whose best foreign match is as similar
    as a same-location match adds little that its neighbour does not already provide.

Nothing here trains anything. The run is one pass over the 256 px day level, so it costs a few
minutes on one GPU and nothing at all on CPU for a sample:

    python scripts/view_redundancy.py --data-root /mnt/ssd1/SyMTRS --device cuda:0
    python scripts/view_redundancy.py --data-root /mnt/ssd1/SyMTRS --sample 400 --device cpu

Reads ``<root>/<split>/rgb_day_lr/x8``; writes ``runs/campaign2/view_redundancy.json``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from symtrs.data.splits import subdir_for

NAME = re.compile(
    r"^rgb_day_(?P<resolution>\d+)_loc(?P<location>\d+)_rot(?P<rotation>\d+)deg_"
    r"h(?P<altitude>\d+)_x(?P<x>[+-]\d+)_y(?P<y>[+-]\d+)_z(?P<z>[+-]\d+)$"
)
SPLITS = ("train", "val", "test")
ENCODER = "vit_large_patch16_dinov3.sat493m"
OUT = Path("runs/campaign2/view_redundancy.json")
#: Remote-sensing CLIP checkpoints, which open_clip can load but timm cannot. Keys are the
#: ``--encoder clip:<key>`` names; values are (open_clip architecture, HF repo, file).
CLIP_PRESETS = {
    "remoteclip-vit-l14": ("ViT-L-14", "chendelong/RemoteCLIP", "RemoteCLIP-ViT-L-14.pt"),
    "remoteclip-vit-b32": ("ViT-B-32", "chendelong/RemoteCLIP", "RemoteCLIP-ViT-B-32.pt"),
    "georsclip-vit-l14": ("ViT-L-14", "Zilun/GeoRSCLIP", "ckpt/RS5M_ViT-L-14.pt"),
}
#: Ranks at which the "nearest re-render of the same location" curve is reported. The figure
#: draws this curve; the points are fixed here so the JSON is the only thing the figure reads.
CURVE_K = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000)


def views(root: Path, split: str, resolution: int, sample: int | None, rng) -> list[dict]:
    """Every (or a seeded sample of) day view of one split, with the pose decoded from its name.

    A sample is drawn by *location*, never by view: the ten rotations and four altitudes of a
    location are what "same location" means here, so thinning views at random would remove
    most of every view's siblings and make the same-location rate an artefact of the sample.
    """
    directory = root / split / subdir_for("day", resolution)
    if not directory.is_dir():
        raise SystemExit(
            f"{directory} is not a directory. Is {root} a SyMTRS root? "
            f"Run `symtrs doctor --data-root {root}` to see which levels exist."
        )
    by_location: dict[int, list[dict]] = {}
    for path in sorted(directory.iterdir()):
        match = NAME.match(path.stem)
        if match is None:
            continue
        view = {
            "path": path,
            "split": split,
            "location": int(match["location"]),
            "rotation": int(match["rotation"]),
            "altitude": int(match["altitude"]),
        }
        by_location.setdefault(view["location"], []).append(view)
    if not by_location:
        raise SystemExit(f"{directory} holds no rgb_day_* files")
    locations = sorted(by_location)
    if sample is not None and len(locations) > sample:
        picked = rng.choice(len(locations), size=sample, replace=False)
        locations = [locations[i] for i in sorted(picked)]
    return [view for location in locations for view in by_location[location]]


def embed_clip(items: list[dict], preset: str, device: str, batch_size: int) -> np.ndarray:
    """Features from a CLIP image encoder, including the remote-sensing checkpoints.

    RemoteCLIP and GeoRSCLIP are open_clip checkpoints from Hugging Face rather than timm
    models: the image tower is the part we use, and its text tower is not exercised. The
    projection head (512 or 768 dims) is what makes the two towers comparable, so it is the
    right output here -- it is the space CLIP was trained to align, not the pre-projection
    ViT features.
    """
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    try:
        import open_clip
    except ModuleNotFoundError as error:  # pragma: no cover - environment dependent
        raise SystemExit(
            "CLIP backbones need open_clip: `uv pip install --python .venv/bin/python "
            "--no-deps open_clip_torch ftfy regex wcwidth`."
        ) from error
    from huggingface_hub import hf_hub_download

    if preset not in CLIP_PRESETS:
        raise SystemExit(
            f"unknown CLIP preset {preset!r}; choose one of {', '.join(sorted(CLIP_PRESETS))}"
        )
    architecture, repo, filename = CLIP_PRESETS[preset]
    model, _, preprocess = open_clip.create_model_and_transforms(
        architecture, pretrained=None, device=device
    )
    path = hf_hub_download(repo, filename)
    open_clip.load_checkpoint(model, path)
    model.eval().to(device)

    class Images(Dataset):
        def __len__(self) -> int:
            return len(items)

        def __getitem__(self, index: int):
            return preprocess(Image.open(items[index]["path"]).convert("RGB"))

    loader = DataLoader(
        Images(), batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=False
    )
    chunks = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                features = model.encode_image(batch)
            chunks.append(features.float().cpu().numpy())
    matrix = np.concatenate(chunks)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def embed(items: list[dict], encoder: str, device: str, batch_size: int) -> np.ndarray:
    """L2-normalised encoder features, one row per item."""
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    if encoder.startswith("clip:"):
        return embed_clip(items, encoder.split(":", 1)[1], device, batch_size)

    try:
        import timm
    except ModuleNotFoundError as error:  # pragma: no cover - environment dependent
        raise SystemExit(
            "this analysis needs timm for the DINOv3 backbone: "
            "`pip install -e '.[timm]'` (or `pip install timm`)."
        ) from error

    class Images(Dataset):
        def __len__(self) -> int:
            return len(items)

        def __getitem__(self, index: int):
            image = Image.open(items[index]["path"]).convert("RGB")
            image = image.resize((size, size), Image.BICUBIC)
            array = np.asarray(image, dtype=np.float32) / 255.0
            return torch.from_numpy(((array - mean) / std).transpose(2, 0, 1))

    if encoder == "pixel":
        # A descriptor with no model and no training behind it: each view as a 32x32 RGB
        # thumbnail. It exists so the redundancy result cannot be attributed to one
        # backbone's "opinion" -- if a 1 KB thumbnail finds the same-location sibling as
        # reliably as a 300 M-parameter encoder, the structure is in the images.
        size = 32
        mean, std = np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32)
        loader = DataLoader(Images(), batch_size=batch_size, num_workers=8, shuffle=False)
        chunks = []
        for batch in loader:
            chunks.append(batch.flatten(1).numpy())
        matrix = np.concatenate(chunks)
        return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)

    model = timm.create_model(encoder, pretrained=True, num_classes=0)
    model.eval().to(device)
    config = model.pretrained_cfg
    size = config.get("input_size", (3, 224, 224))[-1]
    mean = np.array(config["mean"], dtype=np.float32)
    std = np.array(config["std"], dtype=np.float32)

    loader = DataLoader(
        Images(), batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=False
    )
    chunks = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                features = model.forward_features(batch)
            if features.ndim == 4:  # a convolutional map (B, C, H, W): pool it as the head would
                features = model.forward_head(features, pre_logits=True)
            elif features.ndim == 3:  # (B, tokens, C): take the class token
                features = features[:, 0]
            chunks.append(features.float().cpu().numpy())
    matrix = np.concatenate(chunks)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def top_matches(features: np.ndarray, block: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Best and best-other similarity for each row, and the index of the best match."""
    n = len(features)
    best = np.full(n, -np.inf, dtype=np.float32)
    best_index = np.zeros(n, dtype=np.int64)
    for start in range(0, n, block):
        stop = min(start + block, n)
        similarity = features[start:stop] @ features.T
        rows = np.arange(stop - start)
        similarity[rows, np.arange(start, stop)] = -np.inf  # self
        best[start:stop] = similarity.max(axis=1)
        best_index[start:stop] = similarity.argmax(axis=1)
    return best, best_index


def same_location_ranks(features: np.ndarray, location: np.ndarray, block: int = 512) -> np.ndarray:
    """For each view, the rank (1 = nearest) of its nearest *same-location* sibling.

    Absolute cosine values are a poor instrument on these features: a large shared component
    puts every pair at about 0.93, so a view and its own re-render are barely closer than a
    view and a building across the city. The rank is invariant to that offset and asks the
    question the reviewer is really asking -- is location identity recoverable from the image?
    """
    n = len(features)
    ranks = np.empty(n, dtype=np.int32)
    for start in range(0, n, block):
        stop = min(start + block, n)
        similarity = features[start:stop] @ features.T
        for row in range(stop - start):
            index = start + row
            line = similarity[row]
            line[index] = -np.inf  # self
            siblings = line[location == location[index]]
            if siblings.size <= 1:  # only the view itself: nothing to rank against
                ranks[index] = n
                continue
            line[index] = -np.inf
            best_sibling = line[location == location[index]].max()
            ranks[index] = int((line > best_sibling).sum())
    return ranks


def positions(root: Path, resolution: int) -> dict[tuple[str, int, int, int], tuple[float, float]]:
    """(split, location, rotation, altitude) -> (x, y) in metres, read from the file names.

    Cosine similarity alone does not say *why* two views look alike. The distance to the
    nearest look-alike does: a view whose best foreign match is 90 m away is the same street
    seen from a neighbouring camera station, and that is the redundancy statement a reader can
    act on, in metres, rather than a feature-space number whose scale depends on the encoder.
    """
    found: dict[tuple[str, int, int, int], tuple[float, float]] = {}
    for split in SPLITS:
        directory = root / split / subdir_for("day", resolution)
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            match = NAME.match(path.stem)
            if match is None:
                continue
            key = (split, int(match["location"]), int(match["rotation"]), int(match["altitude"]))
            found[key] = (int(match["x"]) / 100.0, int(match["y"]) / 100.0)
    return found


def analyse(features: np.ndarray, location: np.ndarray, rotation: np.ndarray,
            altitude: np.ndarray, split: np.ndarray, xy: np.ndarray, label: str) -> dict:
    """Every statistic the script reports, computed in one feature space."""
    best, best_index = top_matches(features)
    same_location = location[best_index] == location
    # Same location, different pose: the ten rotations and four altitudes of one location.
    pose_changed = same_location & (
        (rotation[best_index] != rotation) | (altitude[best_index] != altitude)
    )

    foreign = []
    foreign_index = []
    for start in range(0, len(features), 2048):
        stop = min(start + 2048, len(features))
        similarity = features[start:stop] @ features.T
        for row in range(stop - start):
            index = start + row
            line = similarity[row]
            line[index] = -np.inf
            line[location == location[index]] = -np.inf
        foreign.append(similarity.max(axis=1))
        foreign_index.append(similarity.argmax(axis=1))
    foreign = np.concatenate(foreign)
    foreign_index = np.concatenate(foreign_index)
    # How far away the nearest look-alike from another location is, in metres.
    foreign_distance = np.sqrt(((xy[foreign_index] - xy) ** 2).sum(axis=1))

    ranks = same_location_ranks(features, location)

    def summary(values: np.ndarray) -> dict[str, float]:
        return {
            "min": float(np.min(values)),
            "p05": float(np.percentile(values, 5)),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
        }

    result = {
        f"{label}_nn_same_location": float(same_location.mean()),
        f"{label}_nn_same_location_other_pose": float(pose_changed.mean()),
        f"{label}_nn_similarity_same_location": summary(best[same_location]),
        f"{label}_nn_similarity_different_location": summary(best[~same_location]),
        f"{label}_best_foreign_similarity": summary(foreign),
        f"{label}_best_foreign_distance_m": summary(foreign_distance),
        f"{label}_best_foreign_above_same_location_median": float(
            (foreign >= np.median(best[same_location])).mean()
        ),
        f"{label}_sibling_rank_median": float(np.median(ranks)),
        f"{label}_sibling_rank_mean": float(np.mean(ranks)),
        f"{label}_sibling_rank_at_1": float((ranks < 1).mean()),
        f"{label}_sibling_rank_at_5": float((ranks < 5).mean()),
        f"{label}_sibling_rank_at_10": float((ranks < 10).mean()),
        f"{label}_sibling_rank_curve": {
            "k": list(CURVE_K),
            "share": [float((ranks < k).mean()) for k in CURVE_K],
        },
    }
    for name in SPLITS:
        mask = split == name
        result[f"{name}_{label}_sibling_rank_median"] = float(np.median(ranks[mask]))
        result[f"{name}_{label}_sibling_rank_at_10"] = float((ranks[mask] < 10).mean())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/ssd1/SyMTRS"))
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--encoder", default=ENCODER,
                        help="a timm model name, or 'pixel' for a 32x32 thumbnail with no model")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sample", type=int, default=None,
                        help="cap the locations per split (seeded, all their views kept); "
                             "omit for the whole release")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--features", type=Path, default=None,
                        help="where the frozen embeddings are cached, so re-analysis is free")
    parser.add_argument("--reuse", action="store_true",
                        help="reuse the cached embeddings instead of running the encoder again")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    # One cache and one result file per encoder: several backbones are meant to be run side by
    # side, and a shared default path would have them overwrite each other's embeddings.
    slug = args.encoder.replace(".", "_").replace("/", "_").replace(":", "-")
    if args.features is None:
        args.features = Path(f"runs/campaign2/view_redundancy_features_{slug}.npz")
    if args.out is None:
        args.out = Path(f"runs/campaign2/view_redundancy_{slug}.json")

    if args.device.startswith("cpu") and args.sample is None and args.encoder != "pixel":
        raise SystemExit(
            "a CPU pass over the whole release takes hours; pass --sample, "
            "e.g. --sample 10 --device cpu (10 locations per split is 1,200 views), "
            "or run on a GPU with --device cuda:0"
        )

    rng = np.random.default_rng(args.seed)
    if args.reuse and args.features.is_file():
        cache = np.load(args.features)
        features = cache["features"].astype(np.float32)
        location, rotation, altitude = cache["location"], cache["rotation"], cache["altitude"]
        split = cache["split"]
        features /= np.linalg.norm(features, axis=1, keepdims=True)
        per_split = {name: int((split == name).sum()) for name in SPLITS}
        print(f"reused {features.shape[0]} cached embeddings from {args.features}")
    else:
        items: list[dict] = []
        per_split = {}
        for name in SPLITS:
            found = views(args.data_root, name, args.resolution, args.sample, rng)
            per_split[name] = len(found)
            items.extend(found)
        print("views:", ", ".join(f"{k} {v}" for k, v in per_split.items()))
        features = embed(items, args.encoder, args.device, args.batch_size)
        print(f"embedded {features.shape[0]} views into {features.shape[1]} dimensions")
        location = np.array([item["location"] for item in items])
        rotation = np.array([item["rotation"] for item in items])
        altitude = np.array([item["altitude"] for item in items])
        split = np.array([item["split"] for item in items])
        args.features.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.features,
            features=features.astype(np.float16), location=location,
            rotation=rotation, altitude=altitude, split=split,
        )
        print(f"cached embeddings in {args.features}")

    # Camera positions in the same order as the features, reconstructed from the file names so
    # that a cached embedding file can be re-analysed without re-running the encoder.
    table = positions(args.data_root, args.resolution)
    try:
        xy = np.array(
            [table[(str(split[i]), int(location[i]), int(rotation[i]), int(altitude[i]))]
             for i in range(len(location))],
            dtype=np.float32,
        )
    except KeyError as error:
        raise SystemExit(
            f"no file name matches view {error.args[0]} under {args.data_root}: the cached "
            f"embeddings were made from a different root or resolution. Rerun without --reuse."
        ) from error

    # The mean feature is shared by every view (one city, one renderer), and it puts all pairs
    # at about 0.93. Centring removes it so the ranking reflects content. Both spaces are
    # reported: the raw one is what a downstream user of these features would see, the centred
    # one is the discriminating part.
    centred = features - features.mean(axis=0)
    centred /= np.linalg.norm(centred, axis=1, keepdims=True)

    result = {
        "encoder": args.encoder,
        "resolution": args.resolution,
        "views": per_split,
        **analyse(features, location, rotation, altitude, split, xy, "raw"),
        **analyse(centred, location, rotation, altitude, split, xy, "centred"),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")

    for label in ("raw", "centred"):
        print(f"[{label}] nearest neighbour is the same location: "
              f"{100 * result[f'{label}_nn_same_location']:.1f} %")
        print(f"[{label}] nearest same-location sibling at rank 1 / 5 / 10: "
              f"{100 * result[f'{label}_sibling_rank_at_1']:.1f} / "
              f"{100 * result[f'{label}_sibling_rank_at_5']:.1f} / "
              f"{100 * result[f'{label}_sibling_rank_at_10']:.1f} % "
              f"(median rank {result[f'{label}_sibling_rank_median']:.0f})")
        print(f"[{label}] best foreign match >= same-location median: "
              f"{100 * result[f'{label}_best_foreign_above_same_location_median']:.1f} %")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
