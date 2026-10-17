"""Score the "do nothing" domain-adaptation reference: the untranslated input as the output.

A KID of 0.25 for day-to-night means nothing on its own: a day image is already at some
feature distance from the real night images, and a translator has only done something if it
lands closer than that. This script computes that distance for the four directions
(`DIRECTION_ROLES` in `symtrs/model/domain/upstream.py`) between

    the INPUT-domain test images, as they are (untranslated)   and
    the real TARGET-domain test images,

plus a split-half noise floor per target domain (KID between the first and the second half of
the real target images, halves chosen by sorted pairing key; and, as a second bracket, between
the even and odd positions), which says how much KID two samples of the *same* domain of this
size already differ by.

Like for like with `symtrs.model.domain.runner.evaluate_adaptation`, which scores a translator:

  * the metric code is the runner's: `symtrs.metrics.domain.extract_features` /
    `distribution_metrics` (Inception-V3 pooled features, antialiased resize to 299, KID with
    the default full-set estimate and FID), the runner's `_image_paths` / `_folder_loader`,
    one shared extractor, and the same checks that both sides have the same number of images;
  * the TARGET side is exactly the runner's real side: the native 2048 px frames,
    `extract_features(..., resize_to=(256, 256))`, i.e. `match_resolution` (torch bilinear,
    antialiased) to the generated size. 256 is what upstream's `test.py` writes
    (`load_size = crop_size = 256`) and what the validation runs record as
    `feature_resize_h/w`; it is an option here (`--size`), not a guess;
  * the INPUT side stands in for a generated image, which is a 256 px PNG that upstream made
    by resizing the 2048 frame itself (`transforms.Resize([256, 256], BICUBIC)` on a PIL
    image = PIL bicubic, antialiased, uint8). The headline number therefore resizes the input
    the same way (PIL BICUBIC to `--size`, nothing else) and feeds it to Inception unchanged;
  * `kid_symmetric_resize` / `fid_symmetric_resize` give the same comparison with the input
    resized by `match_resolution` like the target. If the two differ much, the filter and not
    the domain is moving the number; the headline stays the upstream-path one because that is
    the path a translator's output has been through.

What is deliberately not mirrored: upstream's network (nothing is translated), the [-1, 1]
normalisation round trip (an identity up to uint8 quantisation), and the shadow-removal
paired metrics (see `scripts/shadow_identity_baseline.py` for those).

Reads the dataset, writes one JSON file outside it, and runs on the CPU by default so it never
competes with a training job.

    .venv/Scripts/python.exe scripts/domain_identity_baseline.py --data-root "<root>" \\
        [--out runs/validation/_baselines/domain_identity.json] [--split test] [--size 256]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DESCRIPTION = (
    "input (untranslated) vs target; a translator must beat this KID to have done anything"
)


# ------------------------------------------------------------------ pure parts (no torch)


def default_key(path: Path) -> str:
    """The pairing key of a dataset file: the runner's own, `normalize_stem(stem)`."""
    from symtrs.data.io import normalize_stem

    return normalize_stem(path.stem)


def keyed(paths: Sequence[Path], key: Callable[[Path], str] = None) -> Dict[str, Path]:
    """`{key: path}`; two files with one key is an error, not a silent last-one-wins."""
    key = key or default_key
    out: Dict[str, Path] = {}
    for path in paths:
        k = key(path)
        if k in out:
            raise ValueError(
                f"{path.name} and {out[k].name} share the pairing key {k!r}; the pairing "
                "would silently drop one of them. Check the folder for a duplicated export."
            )
        out[k] = path
    return out


def pair_by_key(
    inputs: Sequence[Path], targets: Sequence[Path], key: Callable[[Path], str] = None
) -> List[Tuple[str, Path, Path]]:
    """`(key, input, target)` for every key present on both sides, in sorted key order.

    The runner refuses to score when the generated and real counts differ (a partial test pass
    would compare two different scene sets), so the reference does the same on the
    intersection and says what it dropped, rather than comparing unequal scene sets quietly.
    """
    a, b = keyed(inputs, key), keyed(targets, key)
    shared = sorted(set(a) & set(b))
    if not shared:
        raise ValueError(
            f"no input/target pair shares a key ({len(a)} input, {len(b)} target files); "
            "run `symtrs doctor --data-root <root>` for a coverage report"
        )
    dropped = len(a) + len(b) - 2 * len(shared)
    if dropped:
        print(f"[symtrs] WARNING: {dropped} file(s) have no counterpart and are not scored")
    return [(k, a[k], b[k]) for k in shared]


def split_halves(
    items: Sequence[Path], key: Callable[[Path], str] = None
) -> Tuple[List[Path], List[Path]]:
    """First and second half of `items` ordered by key: deterministic, no RNG.

    `n // 2` go to the first half and the rest (one more when n is odd) to the second. KID's
    estimator is unbiased at any size, so unequal halves are fine; fewer than 4 images cannot
    give the 2 per side KID needs.
    """
    key = key or default_key
    ordered = sorted(items, key=key)
    if len(ordered) < 4:
        raise ValueError(f"a split-half noise floor needs at least 4 images, got {len(ordered)}")
    half = len(ordered) // 2
    return ordered[:half], ordered[half:]


def split_interleaved(
    items: Sequence[Path], key: Callable[[Path], str] = None
) -> Tuple[List[Path], List[Path]]:
    """Even and odd positions of `items` ordered by key: the other deterministic split.

    Contiguous halves of the SyMTRS test split are two different locations (loc010 / loc011),
    so first-vs-second half measures a scene shift as much as sampling noise; interleaving
    mixes the locations but pairs near-duplicate views (neighbouring heights of one pose), so
    it under-states the spread. The two floors bracket what "same domain" can look like here.
    """
    key = key or default_key
    ordered = sorted(items, key=key)
    if len(ordered) < 4:
        raise ValueError(f"a split-half noise floor needs at least 4 images, got {len(ordered)}")
    return ordered[0::2], ordered[1::2]


def roles_for(direction: str) -> Tuple[str, str]:
    """`(input role, target role)` of a direction, from `DIRECTION_ROLES` (never retyped)."""
    from symtrs.model.domain import upstream

    a_role, b_role, _ = upstream.direction_spec(direction)
    target = upstream.target_role(direction)
    return (a_role if target == b_role else b_role), target


# ------------------------------------------------------------------------- the scoring


class _ResizedFolder:
    """Images as upstream's `test.py` reads them: PIL bicubic to `size` x `size`, uint8."""

    def __init__(self, paths: Sequence[Path], size: int) -> None:
        self.paths, self.size = list(paths), int(size)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        import numpy as np
        import torch
        from PIL import Image

        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB").resize((self.size, self.size), Image.BICUBIC)
            array = np.asarray(image, dtype=np.uint8).copy()
        return {"image": torch.from_numpy(array).permute(2, 0, 1).float() / 255.0}


def _features(paths: Sequence[Path], size: int, upstream_path: bool, extractor):
    from torch.utils.data import DataLoader

    from symtrs.metrics.domain import extract_features
    from symtrs.model.domain.runner import _folder_loader

    if upstream_path:
        loader = DataLoader(_ResizedFolder(paths, size), batch_size=4, num_workers=0)
        return extract_features(loader, extractor=extractor, progress=False)
    return extract_features(
        _folder_loader(list(paths), 4, 0), extractor=extractor, resize_to=(size, size),
        progress=False,
    )


def run(data_root: str, split: str, size: int, device: str, directions: Sequence[str]) -> dict:
    import torch

    from symtrs.metrics.domain import InceptionFeatures, compute_kid, distribution_metrics
    from symtrs.model.domain.runner import _image_paths

    extractor = InceptionFeatures(torch.device(device))
    split_dir = Path(data_root) / split

    def folder(role: str) -> List[Path]:
        directory = split_dir / f"rgb_{role}"
        paths = _image_paths(directory) if directory.is_dir() else []
        if not paths:
            raise SystemExit(
                f"no images in {directory}; check --data-root with "
                f"`symtrs doctor --data-root {data_root}`"
            )
        return sorted(paths, key=default_key)

    cache: Dict[Tuple[str, bool], Tuple[List[str], object]] = {}

    def feats(role: str, upstream_path: bool):
        if (role, upstream_path) not in cache:
            paths = folder(role)
            print(f"[symtrs] {role}: {len(paths)} images, "
                  f"{'upstream PIL-bicubic' if upstream_path else 'match_resolution'} path")
            cache[(role, upstream_path)] = (
                [default_key(p) for p in paths], _features(paths, size, upstream_path, extractor)
            )
        keys, array = cache[(role, upstream_path)]
        return keys, array

    result: dict = {
        "model": "identity (untranslated input as output)",
        "description": DESCRIPTION,
        "split": split,
        "feature_resize": [size, size],
        "directions": {},
        "noise_floor": {},
    }
    for direction in directions:
        source, target = roles_for(direction)
        pairs = pair_by_key(folder(source), folder(target))
        keep = [k for k, _, _ in pairs]
        src_keys, src_up = feats(source, True)
        _, src_sym = feats(source, False)
        tgt_keys, tgt = feats(target, False)
        take = lambda keys, array: array[[keys.index(k) for k in keep]]  # noqa: E731
        metrics = distribution_metrics(take(tgt_keys, tgt), take(src_keys, src_up))
        symmetric = distribution_metrics(take(tgt_keys, tgt), take(src_keys, src_sym))
        result["directions"][direction] = {
            "input": source, "target": target,
            "kid": metrics["kid"], "kid_std": metrics["kid_std"], "fid": metrics["fid"],
            "fid_rank_deficient": metrics["fid_rank_deficient"],
            "n_real": metrics["n_real"], "n_fake": metrics["n_fake"],
            "kid_symmetric_resize": symmetric["kid"], "fid_symmetric_resize": symmetric["fid"],
        }
        print(f"{direction}: kid {metrics['kid']:.4f} (symmetric resize "
              f"{symmetric['kid']:.4f})  fid {metrics['fid']:.1f}  n={int(metrics['n_fake'])}")

    for target in sorted({roles_for(d)[1] for d in directions}):
        paths = folder(target)
        first, second = split_halves(paths)
        tgt_keys, tgt = feats(target, False)
        rows = lambda part: tgt[[tgt_keys.index(default_key(p)) for p in part]]  # noqa: E731
        floor = compute_kid(rows(first), rows(second))
        even, odd = split_interleaved(paths)
        interleaved = compute_kid(rows(even), rows(odd))
        result["noise_floor"][target] = {
            "kid": floor["kid"], "kid_std": floor["kid_std"], "n_first": len(first),
            "n_second": len(second),
            "split": "sorted pairing key: first n//2 vs the rest (on the test split: two "
                     "different locations)",
            "kid_interleaved": interleaved["kid"],
            "split_interleaved": "sorted pairing key: even vs odd positions (mixes locations, "
                                 "pairs near-duplicate views)",
        }
        print(f"noise floor {target}: kid {floor['kid']:.4f} halves, "
              f"{interleaved['kid']:.4f} interleaved  ({len(first)} vs {len(second)})")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--size", type=int, default=256, help="generated image size of upstream's test.py")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="runs/validation/_baselines/domain_identity.json")
    args = ap.parse_args()

    from symtrs.common.jsonio import write_json
    from symtrs.model.domain.runner import DIRECTIONS

    result = run(args.data_root, args.split, args.size, args.device, DIRECTIONS)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, result)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
