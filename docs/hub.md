# The release map and `symtrs get`

The release is split across several Hugging Face projects: one per task vertical, one for
the shared image store, and one metadata project that carries the catalog. `symtrs` with no
subcommand opens the launcher; `symtrs get` is the same selection without a terminal.

The launcher is a **downloader, not a formatter**. It copies every file to the repo-relative
path the manifest and tables name. It never renames, links, converts or re-lays-out the
dataset; the release format is the maintainer's, and downloading it is a copy. Today that
format is the existing flat `<split>/rgb_day` tree; a future release may publish the
task-first tree from the hierarchy figure, and the launcher will copy that just the same.

## Repository map (placeholders)

These are the names the release is planned under. The real URLs live in `symtrs-hub.json`;
change them there, not in code.

| key | project | holds | default |
| --- | --- | --- | --- |
| `metadata` | `https://huggingface.co/datasets/safouaneelg/SyMTRS-metadata` | `symtrs-hub.json`, `metadata/views.csv`, `camera.yaml`, `classes.yaml`, splits | **auto** |
| `views` | `https://huggingface.co/datasets/safouaneelg/SyMTRS-views` | the only copy of every image, at its published dataset path | on demand |
| `depth` | `https://huggingface.co/datasets/safouaneelg/SyMTRS-depth` | depth task tables | vertical |
| `sr` | `https://huggingface.co/datasets/safouaneelg/SyMTRS-sr` | super-resolution task tables | vertical |
| `segmentation` | `https://huggingface.co/datasets/safouaneelg/SyMTRS-segmentation` | segmentation task tables | vertical |
| `night_translation` | `https://huggingface.co/datasets/safouaneelg/SyMTRS-night_translation` | day⇄night task tables | vertical |
| `shadow_removal` | `https://huggingface.co/datasets/safouaneelg/SyMTRS-shadow_removal` | shadow-removal task tables | vertical |
| `models-depth` | `https://huggingface.co/safouaneelg/SyMTRS-models-depth` | depth checkpoints | vertical |
| `models-sr` | `https://huggingface.co/safouaneelg/SyMTRS-models-sr` | super-resolution checkpoints | vertical |
| `models-segmentation` | `https://huggingface.co/safouaneelg/SyMTRS-models-segmentation` | segmentation checkpoints | vertical |
| `models-night_translation` | `https://huggingface.co/safouaneelg/SyMTRS-models-night_translation` | translation checkpoints | vertical |
| `models-shadow_removal` | `https://huggingface.co/safouaneelg/SyMTRS-models-shadow_removal` | shadow-removal checkpoints | vertical |

The **catalog** is `metadata/symtrs-hub.json` in the metadata project. `metadata` is the one
default group: selecting any dataset vertical adds it, because a task table without the
camera/class metadata cannot be interpreted. Models-only pulls do not create a dataset root.

## Commands

```
symtrs                                  # launcher: pick groups, splits, models; review; download
symtrs get --list                       # repos, groups, models, sizes, what is installed
symtrs get --plan --task depth --resolutions 2048 --splits train,val
symtrs get --preset depth-2048 --data-root /data/SyMTRS --yes
symtrs get --models midas_dpt_hybrid@1024 --models-dir runs/
symtrs get --verify [--repair]
```

`--plan` fetches only the tables needed to price the selection. `--yes` is required in a
non-interactive session; without a TTY the command prints the catalogue rather than hanging.

Selecting `depth/2048` reads the depth project's table, routes `train/...` rows to the view
project and `metadata/...` to the metadata project, and writes one plan. Selecting
`depth/2048` and `segmentation/2048` fetches shared `train/rgb_day` once; `--plan` shows the
bytes that saves.

## `symtrs-hub.json`

```json
{
  "schema": "symtrs-hub/2",
  "release": {
    "name": "SyMTRS", "revision": "v1.0", "commit": "<sha>",
    "layout": "flat", "views": 20000, "files": 280000, "bytes": 858993459200
  },
  "repos": [
    {"key": "metadata", "repo": "safouaneelg/SyMTRS-metadata", "type": "dataset", "revision": "v1.0"},
    {"key": "views", "repo": "safouaneelg/SyMTRS-views", "type": "dataset", "revision": "v1.0"},
    {"key": "depth", "repo": "safouaneelg/SyMTRS-depth", "type": "dataset", "revision": "v1.0"},
    {"key": "models-depth", "repo": "safouaneelg/SyMTRS-models-depth", "type": "model", "revision": "v1.0"}
  ],
  "file_repos": {
    "metadata/": "metadata",
    "train/": "views", "val/": "views", "test/": "views"
  },
  "default_groups": ["metadata"],
  "groups": [
    {
      "id": "depth/2048", "repo": "depth",
      "task": "depth", "variant": "2048",
      "resolution": 2048, "native": true, "label": "depth · 2048",
      "modalities": ["rgb_day", "depth_day"],
      "splits": ["train", "val", "test"],
      "table": "tables/depth/{split}/2048/table.jsonl.gz",
      "requires": ["metadata"],
      "bytes":       {"train": 0, "val": 0, "test": 0},
      "views":       {"train": 0, "val": 0, "test": 0},
      "table_bytes": {"train": 0, "val": 0, "test": 0}
    }
  ],
  "models": [
    {
      "id": "midas_dpt_hybrid", "task": "depth", "registry_key": "midas_dpt_hybrid",
      "repo": "models-depth",
      "protocol": "relative", "output_space": "disparity", "requires_extra": "timm",
      "files": [
        {"resolution": 2048, "path": "depth/midas_dpt_hybrid/2048/weights/best.pt",
         "bytes": 0, "sha256": "<64 hex>"}
      ]
    }
  ],
  "presets": [
    {"id": "depth-2048", "label": "depth at native resolution",
     "groups": ["metadata", "depth/2048"], "splits": ["train", "val"],
     "models": ["midas_dpt_hybrid"]}
  ]
}
```

Rules the loader enforces, each with an error naming the fix:

- `schema` must be `symtrs-hub/2`; another version says to update `symtrs`.
- `release.layout` is `flat` today (the existing `<split>/rgb_day` tree) or `task-first`.
  It is declarative: the launcher copies paths either way and never converts between them.
- every repo key is unique, has type `dataset` or `model`, and the map includes the
  catalog repo `metadata`.
- `file_repos` prefixes end with `/` and name declared repo keys. A path matching no
  prefix belongs to the repo that declared it (so a table stays with its vertical).
- every group's `repo` is a declared **dataset** repo; every model's `repo` is a declared
  **model** repo.
- `table` contains exactly one `{split}` placeholder.
- every split a group publishes has an entry in `bytes`, `views` and `table_bytes`.
- `requires` and `default_groups` name published groups.
- every model path is repo-relative, uses `/`, and has no `..`.
- `sha256`, when present, is 64 hexadecimal characters.
- preset group/model ids exist.

## Task tables

A table line is one view and the files that task reads from it, at their published dataset
paths:

```json
{"key": "loc000_rot103deg_h01_x-049013_y+035369_z+012230", "split": "train",
 "files": [["train/rgb_day/rgb_day_2048_loc000_….png", 12884901],
           ["train/depth/depth_day_2048_loc000_….exr", 8388608]]}
```

Each vertical repo holds `tables/<task>/<split>/<variant>/table.jsonl.gz`. `<variant>` is
`2048|1024|512|256` for depth, segmentation and shadow removal, `x2|x4|x8` for
super-resolution, and `day-night|shadow` for translation.

## Model repos

Checkpoints are published in the run layout the package already reads:

```
<task>/<registry_key>/<resolution>/weights/best.pt
<task>/<registry_key>/<resolution>/summary.json
<task>/<registry_key>/<resolution>/config.yaml
```

After `symtrs get --models … --models-dir runs/`, `symtrs tables runs` and
`autoloader(runs/…/weights/best.pt)` work with no conversion.

## What lands on disk

Every selected file is copied to `data_root/<published path>` or
`models_dir/<published path>`, unchanged. Nothing else is written except the lockfile:

```
/data/SyMTRS/
  train/rgb_day/…  train/depth/…  train/semantic/…   copied from SyMTRS-views
  metadata/…                                         copied from SyMTRS-metadata
  tables/…                                           copied from the vertical's repo
  .symtrs/install.json                               release, repo map, installed groups
```

The lockfile is what makes a partial install legible. `symtrs get --verify` re-reads the
installed tables, checks every file's size, and (for models) its sha256. A destination
holding two releases is refused unless `--allow-mixed-revision` is passed.

## Offline and mirrors

A mirror holds the same multi-repo layout under one directory, named by repo key:

```
mirror/
  metadata/symtrs-hub.json
  metadata/metadata/views.csv
  views/train/rgb_day/…
  depth/tables/depth/train/2048/table.jsonl.gz
  models-depth/depth/<key>/<res>/weights/best.pt
```

- `--mirror DIR` reads the catalog from `DIR/metadata/symtrs-hub.json` and files from each
  `DIR/<repo key>/`.
- `--manifest <path>` reads the manifest from a file; files still come from the Hub.
- `--offline` forbids the Hub and therefore requires `--mirror`.

## Publishing checklist (maintainer)

1. Push the image store to `SyMTRS-views` once, at the dataset paths the package reads.
2. Write one table per `(task, split, variant)` into that vertical's repo and push it there.
3. Push `metadata/` (built by `scripts/build_metadata.py`) to `SyMTRS-metadata`, including
   `symtrs-hub.json` with the `repos` map above.
4. Push each model vertical with sha256 per checkpoint.
5. Tag every repo with the same release tag and point each `repos[].revision` at it, never
   at `main`.

The publisher scripts are not part of the launcher core yet; the manifest and table formats
above are the contract they must produce. `symtrs/hub/repos.py` holds the placeholder map
they should start from.
