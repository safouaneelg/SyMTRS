"""The dataset/model hub front-end.

`symtrs` with no subcommand opens the interactive launcher; `symtrs get` is the same
selection without a terminal. Both build a `Selection`, resolve it against the published
manifest and either print the plan or install it.

Neither `huggingface_hub` nor `textual` is imported at package import time. The reproducible
core (`manifest`, `plan`, `state`) works with nothing beyond the package's own
dependencies, which is why it can be tested offline and why a training-only install never
pays for a downloader.
"""

from __future__ import annotations

from symtrs.hub.errors import BackendError, DiskSpaceError, HubError, ManifestError, SelectionError
from symtrs.hub.fetch import (
    DEFAULT_REPO,
    DEFAULT_REVISION,
    Backend,
    BackendSet,
    HuggingFaceBackend,
    LocalBackend,
    huggingface_backend_factory,
    mirror_backend_factory,
)
from symtrs.hub.install import InstallReport, human_bytes, install
from symtrs.hub.manifest import Manifest, ReleaseInfo, Repo, load_manifest_bytes, load_manifest_file
from symtrs.hub.plan import Plan, Selection, inspect_destination, parse_model_spec, resolve, select_tasks
from symtrs.hub.state import Lock, read_lock, verify_dataset, verify_models, write_lock

__all__ = [
    "Backend",
    "BackendSet",
    "BackendError",
    "DEFAULT_REPO",
    "DEFAULT_REVISION",
    "DiskSpaceError",
    "HubError",
    "HuggingFaceBackend",
    "InstallReport",
    "LocalBackend",
    "Lock",
    "Manifest",
    "ManifestError",
    "Plan",
    "ReleaseInfo",
    "Repo",
    "Selection",
    "SelectionError",
    "human_bytes",
    "huggingface_backend_factory",
    "inspect_destination",
    "install",
    "mirror_backend_factory",
    "load_manifest_bytes",
    "load_manifest_file",
    "parse_model_spec",
    "read_lock",
    "resolve",
    "select_tasks",
    "verify_dataset",
    "verify_models",
    "write_lock",
]
