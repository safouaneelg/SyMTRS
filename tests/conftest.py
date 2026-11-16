"""Shared fixtures.

Two kinds of fixture live here, split by what they need:

  * A **layout fixture** that only creates directories. Everything about the resolution
    axis, the split names and the `_lr/x2` suffixes is pure pathlib, so those tests run
    anywhere, including on a machine with no torch - which is the situation you are in
    when you are checking that a freshly generated dataset has the right shape.

  * A **data fixture** that writes actual `.npy` depth maps and label arrays and builds a
    `RunConfig` over them. Those tests need torch and skip when it is absent.

`requires_torch` is a skip marker rather than a hard import at module scope on purpose: a
collection error in a file that cannot import torch would hide the torch-free tests in the
same run, and those are the ones that catch a mis-generated dataset.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: Resolutions and the directory suffix each one reads, mirroring the dataset layout.
RESOLUTION_SUFFIX = {2048: None, 1024: "x2", 512: "x4", 256: "x8"}


def _have(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


HAVE_TORCH = _have("torch")
HAVE_NUMPY = _have("numpy")
HAVE_PIL = _have("PIL")

requires_torch = pytest.mark.skipif(not HAVE_TORCH, reason="torch is not installed")
requires_numpy = pytest.mark.skipif(not HAVE_NUMPY, reason="numpy is not installed")
requires_pil = pytest.mark.skipif(not HAVE_PIL, reason="Pillow is not installed")

#: Anything that builds a model downloads weights. Opt in with `-m network` explicitly.
network = pytest.mark.network


@pytest.fixture
def layout(tmp_path: Path) -> Path:
    """An empty but correctly shaped SyMTRS dataset root.

    Directories only - no images, no depth, no labels. This is what `symtrs inspect`
    and a dry run look at, and it is what the resolution-mapping tests assert against.
    """
    root = tmp_path / "SyMTRS"
    for split in ("train", "val", "test"):
        split_root = root / split
        (split_root / "rgb_day").mkdir(parents=True)
        (split_root / "rgb_night").mkdir(parents=True)
        (split_root / "rgb_noshadow").mkdir(parents=True)
        (split_root / "depth").mkdir(parents=True)
        (split_root / "semantic").mkdir(parents=True)
        for suffix in ("x2", "x4", "x8"):
            (split_root / "rgb_day_lr" / suffix).mkdir(parents=True)
            (split_root / "depth_lr" / suffix).mkdir(parents=True)
            (split_root / "semantic_lr" / suffix).mkdir(parents=True)
    return root


@pytest.fixture
def partial_layout(tmp_path: Path) -> Path:
    """A dataset that is missing everything below native resolution.

    The shape a dataset has while it is still being generated, which is exactly when you
    want `symtrs inspect` to tell you what exists instead of raising.
    """
    root = tmp_path / "SyMTRS_partial"
    for split in ("train", "val", "test"):
        split_root = root / split
        (split_root / "rgb_day").mkdir(parents=True)
        (split_root / "rgb_night").mkdir(parents=True)
        (split_root / "depth").mkdir(parents=True)
        (split_root / "semantic").mkdir(parents=True)
        (split_root / "rgb_day_lr" / "x2").mkdir(parents=True)
    return root


# --------------------------------------------------------------------------- hub


_HUB_KEYS = {
    "train": (
        "loc000_rot103deg_h01_x-049013_y+035369_z+012230",
        "loc001_rot210deg_h05_x+001000_y-002000_z+015000",
    ),
    "val": ("loc010_rot045deg_h02_x+010000_y+020000_z+010000",),
    "test": ("loc020_rot300deg_h03_x-030000_y-010000_z+020000",),
}

#: The fourteen files of a view, as (modality, light, resolution).
_HUB_VIEW_FILES = (
    ("rgb", "day", 2048), ("rgb", "day", 1024), ("rgb", "day", 512), ("rgb", "day", 256),
    ("rgb", "night", 2048), ("rgb", "noshadow", 2048),
    ("depth", "day", 2048), ("depth", "day", 1024), ("depth", "day", 512), ("depth", "day", 256),
    ("seg", "day", 2048), ("seg", "day", 1024), ("seg", "day", 512), ("seg", "day", 256),
)


def _write_jsonl_gz(path: Path, rows) -> int:
    import gzip
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path.stat().st_size


@pytest.fixture
def hub_mirror(tmp_path: Path):
    """The multi-repo release on disk: one directory per HF project, catalog in metadata/.

    Every view carries the fourteen files, so the install tests exercise the real name
    grammar and the flat dataset paths. Groups share `rgb_day_2048` across projects on purpose: the dedupe
    is the point of the launcher and this fixture is where it is pinned.
    """
    import hashlib
    import json
    from types import SimpleNamespace

    from symtrs.data.splits import subdir_for
    from symtrs.hub.fetch import BackendSet, mirror_backend_factory
    from symtrs.hub.manifest import load_manifest_file
    from symtrs.hub.repos import PLACEHOLDER_REPOS

    root = tmp_path / "hub"
    sizes = {}

    def write(repo_key: str, relative: str, data: bytes) -> None:
        path = root / repo_key / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        sizes[(repo_key, relative)] = len(data)

    def size(repo_key: str, relative: str) -> int:
        return sizes[(repo_key, relative)]

    def view_relative(split: str, modality: str, light: str, resolution: int, key: str) -> str:
        """The current flat dataset layout: <split>/rgb_day, <split>/depth_lr/x2, ..."""
        if modality == "rgb":
            role = light
        elif modality in ("seg", "semantic"):
            role = "semantic"
        else:
            role = "depth"
        extension = "exr" if modality == "depth" else "png"
        name = f"{modality}_{light}_{resolution}_{key}.{extension}"
        return f"{split}/{subdir_for(role, resolution)}/{name}"

    # The image store: one copy of each view's fourteen files, all in the views project,
    # at exactly the paths the current package reads. The launcher does not reshape this.
    for split, keys in _HUB_KEYS.items():
        for key in keys:
            for modality, light, resolution in _HUB_VIEW_FILES:
                relative = view_relative(split, modality, light, resolution, key)
                write("views", relative, relative.encode())

    for split in _HUB_KEYS:
        write("metadata", "metadata/views.csv", b"view_key,split\n" + b"\n".join(
            key.encode() for keys in _HUB_KEYS.values() for key in keys
        ) + b"\n")
        write("metadata", "metadata/camera.yaml", b"horizontal_fov_deg: 60\n")
        write("metadata", "metadata/classes.yaml", b"num_classes: 5\n")
        write("metadata", f"metadata/splits/{split}.txt",
              ("\n".join(_HUB_KEYS[split]) + "\n").encode())

    def view_rows(split: str, modalities) -> list:
        rows = []
        for key in _HUB_KEYS[split]:
            files = []
            for modality, light, resolution in modalities:
                relative = view_relative(split, modality, light, resolution, key)
                files.append([relative, size("views", relative)])
            rows.append({"key": key, "split": split, "files": files})
        return rows

    groups = []

    def add_group(group_id, task, variant, repo, modalities, *, resolution=None, native=False):
        table = f"tables/{task}/{{split}}/{variant}/table.jsonl.gz"
        per_split_bytes = {}
        per_split_views = {}
        per_split_table = {}
        for split in _HUB_KEYS:
            rel = table.format(split=split)
            rows = view_rows(split, modalities)
            per_split_table[split] = _write_jsonl_gz(root / repo / rel, rows)
            per_split_bytes[split] = sum(item for row in rows for _, item in row["files"])
            per_split_views[split] = len(rows)
        groups.append({
            "id": group_id,
            "repo": repo,
            "task": task,
            "variant": variant,
            "resolution": resolution,
            "native": native,
            "label": f"{task} · {variant}",
            "modalities": [f"{m}_{light}" for m, light, _ in modalities],
            "splits": list(_HUB_KEYS),
            "table": table,
            "requires": ["metadata"] if group_id != "metadata" else [],
            "bytes": per_split_bytes,
            "views": per_split_views,
            "table_bytes": per_split_table,
        })

    # metadata: a real group in the metadata project, and the default that rides along.
    metadata_table = "tables/metadata/{split}.jsonl.gz"
    metadata_bytes, metadata_tables = {}, {}
    for split in _HUB_KEYS:
        rel = metadata_table.format(split=split)
        rows = [{
            "key": "metadata",
            "split": split,
            "files": [
                ["metadata/views.csv", size("metadata", "metadata/views.csv")],
                ["metadata/camera.yaml", size("metadata", "metadata/camera.yaml")],
                ["metadata/classes.yaml", size("metadata", "metadata/classes.yaml")],
                [f"metadata/splits/{split}.txt", size("metadata", f"metadata/splits/{split}.txt")],
            ],
        }]
        metadata_tables[split] = _write_jsonl_gz(root / "metadata" / rel, rows)
        metadata_bytes[split] = sum(item for _, item in rows[0]["files"])
    groups.append({
        "id": "metadata", "repo": "metadata", "task": "metadata", "variant": "all",
        "resolution": None, "native": True, "label": "metadata", "modalities": [],
        "splits": list(_HUB_KEYS), "table": metadata_table, "requires": [],
        "bytes": metadata_bytes,
        "views": {split: len(keys) for split, keys in _HUB_KEYS.items()},
        "table_bytes": metadata_tables,
    })

    add_group("depth/2048", "depth", "2048", "depth",
              [("rgb", "day", 2048), ("depth", "day", 2048)], resolution=2048, native=True)
    add_group("segmentation/2048", "segmentation", "2048", "segmentation",
              [("rgb", "day", 2048), ("seg", "day", 2048)], resolution=2048, native=True)
    add_group("domain_adapt/day-night", "domain_adapt", "day-night", "night_translation",
              [("rgb", "day", 2048), ("rgb", "night", 2048)], resolution=2048, native=True)
    add_group("shadow_removal/2048", "shadow_removal", "2048", "shadow_removal",
              [("rgb", "day", 2048), ("rgb", "noshadow", 2048)], resolution=2048, native=True)
    add_group("superresolution/x2", "superresolution", "x2", "sr",
              [("rgb", "day", 256), ("rgb", "day", 2048)])

    models = []
    for model_id, task, repo, resolutions, extra, protocol, output_space in (
        ("midas_small", "depth", "models-depth", (2048, 1024), "timm", "relative", "disparity"),
        ("segformer_b2", "segmentation", "models-segmentation", (2048,), "transformers", "", ""),
    ):
        files = []
        for resolution in resolutions:
            relative = f"{task}/{model_id}/{resolution}/weights/best.pt"
            data = f"{model_id}/{resolution}".encode() * 64
            write(repo, relative, data)
            files.append({
                "resolution": resolution,
                "path": relative,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            })
        models.append({
            "id": model_id, "task": task, "repo": repo, "registry_key": model_id,
            "description": f"fixture {model_id}", "protocol": protocol,
            "output_space": output_space, "requires_extra": extra, "files": files,
        })

    repo_keys = (
        "metadata", "views", "depth", "sr", "segmentation", "night_translation",
        "shadow_removal", "models-depth", "models-segmentation",
    )
    repos = [
        {"key": key, "repo": PLACEHOLDER_REPOS[key]["repo"],
         "type": PLACEHOLDER_REPOS[key]["type"], "revision": "v1"}
        for key in repo_keys
    ]

    total_files = sum(1 for path in root.rglob("*") if path.is_file())
    total_bytes = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    manifest = {
        "schema": "symtrs-hub/2",
        "release": {
            "name": "SyMTRS", "revision": "v1", "commit": "a" * 40, "layout": "flat",
            "views": sum(len(keys) for keys in _HUB_KEYS.values()),
            "files": total_files,
            "bytes": total_bytes,
        },
        "repos": repos,
        "file_repos": {
            "metadata/": "metadata",
            "train/": "views", "val/": "views", "test/": "views",
        },
        "default_groups": ["metadata"],
        "groups": groups,
        "models": models,
        "presets": [
            {"id": "depth-2048", "label": "depth at native resolution",
             "groups": ["metadata", "depth/2048"], "splits": ["train", "val"],
             "models": ["midas_small"]},
            {"id": "seg-2048", "label": "segmentation at native resolution",
             "groups": ["metadata", "segmentation/2048"], "splits": ["train"],
             "models": ["segformer_b2"]},
        ],
    }
    (root / "metadata" / "symtrs-hub.json").write_text(json.dumps(manifest, indent=2))
    (root / "metadata" / "README.md").write_text("fixture release\n")

    loaded = load_manifest_file(root / "metadata" / "symtrs-hub.json")
    backends = BackendSet(loaded.repos, mirror_backend_factory(root))
    return SimpleNamespace(root=root, manifest=loaded, sizes=sizes, keys=_HUB_KEYS,
                           backends=backends)
