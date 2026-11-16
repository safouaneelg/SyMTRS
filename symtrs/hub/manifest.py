"""The published release's manifest: schema, validation and the objects the launcher uses.

The release is spread across several Hugging Face projects - one per task vertical, one for
the image store, one for the metadata that carries this manifest, one per model vertical.
The manifest is the map: it names every repo, says which repo each task's tables live in,
and routes file paths to repos by prefix. It is fetched once and cached; the tables it
points at are fetched only for the groups someone selected.

Everything here is strict about shape and loose about extras: an unknown top-level key is
kept in `raw` so a newer release can add one without breaking an older package, but a
missing field or an unknown schema version stops with a message naming both sides. A
launcher that guessed would download the wrong file and report a number nobody can defend.
"""

from __future__ import annotations

import hashlib
import json
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from symtrs.data.splits import RESOLUTIONS, SPLITS

from symtrs.hub.errors import ManifestError
from symtrs.hub.repos import CATALOG_REPO_KEY, repo_url

#: The only schema this package reads. A release that reorganises the manifest bumps this.
SCHEMA = "symtrs-hub/2"

#: The release layouts the launcher accepts. It never re-shapes a release: files are copied
#: to the paths the manifest and tables name, so whichever layout the maintainer publishes
#: is the layout that lands on disk. `flat` is the existing `<split>/rgb_day` tree;
#: `task-first` is the figure's `views/` + `tasks/` tree.
SUPPORTED_LAYOUTS = ("flat", "task-first")

REPO_TYPES = ("dataset", "model")


@dataclass(frozen=True)
class Repo:
    """One Hugging Face project in the release map."""

    key: str
    repo: str
    type: str = "dataset"
    revision: str = "main"

    @property
    def url(self) -> str:
        return repo_url(self.repo, self.type)

    def describe(self) -> str:
        return f"{self.key}  {self.url}"


@dataclass(frozen=True)
class ReleaseInfo:
    """The release as a whole: which revision, how much data, and the layout."""

    name: str
    revision: str
    commit: str = ""
    layout: str = "task-first"
    views: int = 0
    files: int = 0
    bytes: int = 0

    @property
    def pinned(self) -> bool:
        return bool(self.commit)


@dataclass(frozen=True)
class Group:
    """One task variant (a resolution, a scale, or a translation direction) as published.

    `repo` is the key of the dataset repo that holds this group's table. Files a table lists
    are routed by `Manifest.file_repos` (prefix -> repo key), falling back to this repo.
    `bytes`, `views` and `table_bytes` are the per-split totals the manifest advertises, so
    `--list` and a plan can show a size without downloading a table.
    """

    id: str
    task: str
    variant: str
    label: str
    repo: str
    modalities: Tuple[str, ...]
    splits: Tuple[str, ...]
    table: str
    requires: Tuple[str, ...]
    bytes: Mapping[str, int]
    views: Mapping[str, int]
    table_bytes: Mapping[str, int] = field(default_factory=dict)
    resolution: Optional[int] = None
    native: bool = False

    def table_for(self, split: str) -> str:
        if split not in self.splits:
            raise ManifestError(
                f"group {self.id!r} has no {split!r} split; it publishes {', '.join(self.splits)}"
            )
        return self.table.format(split=split)

    def size_for(self, splits: Tuple[str, ...]) -> int:
        return sum(self.bytes.get(split, 0) + self.table_bytes.get(split, 0) for split in splits)

    def count_for(self, splits: Tuple[str, ...]) -> int:
        return sum(self.views.get(split, 0) for split in splits)

    def matches_resolution(self, resolution: int) -> bool:
        return self.resolution == resolution

    def matches_variant(self, variant: str) -> bool:
        return self.variant == variant


@dataclass(frozen=True)
class ModelFile:
    """One checkpoint file, addressed by its path inside the model repo."""

    resolution: int
    path: str
    bytes: int
    sha256: str = ""


@dataclass(frozen=True)
class Model:
    """One registry key's published checkpoints, in one model-repo vertical."""

    id: str
    task: str
    registry_key: str
    repo: str
    description: str = ""
    protocol: str = ""
    output_space: str = ""
    requires_extra: str = ""
    files: Tuple[ModelFile, ...] = field(default_factory=tuple)

    @property
    def resolutions(self) -> Tuple[int, ...]:
        return tuple(sorted({f.resolution for f in self.files}, reverse=True))

    def files_for(self, resolutions: Tuple[int, ...] = ()) -> Tuple[ModelFile, ...]:
        if not resolutions:
            return self.files
        wanted = set(resolutions)
        return tuple(f for f in self.files if f.resolution in wanted)

    def size_for(self, resolutions: Tuple[int, ...] = ()) -> int:
        return sum(f.bytes for f in self.files_for(resolutions))


@dataclass(frozen=True)
class Preset:
    """A named selection, kept in the manifest so it can change without a package release."""

    id: str
    label: str
    groups: Tuple[str, ...]
    splits: Tuple[str, ...]
    models: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Manifest:
    """The validated release manifest, plus the digest of the bytes it was read from."""

    schema: str
    release: ReleaseInfo
    repos: Mapping[str, Repo]
    file_repos: Mapping[str, str]
    default_groups: Tuple[str, ...]
    groups: Mapping[str, Group]
    models: Mapping[str, Model]
    presets: Mapping[str, Preset]
    digest: str
    source: str = ""
    raw: Mapping[str, Any] = field(default_factory=dict)

    @property
    def catalog(self) -> Optional[Repo]:
        return self.repos.get(CATALOG_REPO_KEY)

    def repo(self, key: str) -> Repo:
        try:
            return self.repos[key]
        except KeyError:
            raise ManifestError(
                f"the manifest references repo {key!r}, which it does not declare; "
                f"declared: {', '.join(self.repos)}"
            ) from None

    def repo_for_path(self, path: str, fallback: str) -> str:
        """Which repo key holds `path`, from the release's prefix map.

        The map is explicit in the manifest because the same logical tree is split across
        projects: `views/` lives in the view store, `metadata/` in the metadata project,
        and a task's table lives beside that task. A path that matches no prefix belongs
        to the repo that declared it.
        """
        best = None
        for prefix, key in self.file_repos.items():
            if prefix and path.startswith(prefix):
                if best is None or len(prefix) > len(best[0]):
                    best = (prefix, key)
        return best[1] if best is not None else fallback

    def group_ids_by_task(self) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for group in self.groups.values():
            out.setdefault(group.task, []).append(group.id)
        return out

    def tasks(self) -> Tuple[str, ...]:
        return tuple(sorted(self.group_ids_by_task()))


def _where(source: str, path: str) -> str:
    return f"{source}: {path}" if source else path


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ManifestError(f"{where} must be an object, got {type(value).__name__}")
    return value


def _string(value: Any, where: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ManifestError(f"{where} must be a string, got {type(value).__name__}")
    if not value and not allow_empty:
        raise ManifestError(f"{where} must not be empty")
    return value


def _integer(value: Any, where: str, *, minimum: int = 0) -> int:
    # bool is an int in Python, and a JSON `true` where a byte count belongs is a bug.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManifestError(f"{where} must be an integer, got {type(value).__name__}")
    if value < minimum:
        raise ManifestError(f"{where} must be >= {minimum}, got {value}")
    return value


def _boolean(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise ManifestError(f"{where} must be true or false, got {type(value).__name__}")
    return value


def _string_list(value: Any, where: str, *, unique: bool = True) -> Tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ManifestError(f"{where} must be a list, got {type(value).__name__}")
    out = tuple(_string(item, f"{where}[{index}]") for index, item in enumerate(value))
    if unique and len(set(out)) != len(out):
        raise ManifestError(f"{where} contains a duplicate entry")
    return out


def safe_relative_path(path: str, where: str) -> str:
    """A repo-relative POSIX path, refusing anything that could escape the destination.

    The table is published data; the launcher writes it under a user-chosen root. A path
    like `../../.ssh/config` in a table would otherwise be honored by `dest / path`.
    """
    if path.startswith("/") or path.startswith("\\"):
        raise ManifestError(f"{where} must be relative, got {path!r}")
    if "\\" in path:
        raise ManifestError(f"{where} must use '/' separators, got {path!r}")
    parts = path.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ManifestError(f"{where} contains an empty, '.' or '..' component: {path!r}")
    return path


def _table_template(value: Any, where: str) -> str:
    template = _string(value, where)
    fields = [field_name for _, field_name, _, _ in string.Formatter().parse(template) if field_name]
    if fields != ["split"]:
        raise ManifestError(
            f"{where} must contain exactly one '{{split}}' placeholder and nothing else, "
            f"got {template!r}; the launcher fetches one table per split"
        )
    if "{split}" not in template:
        raise ManifestError(f"{where} must contain a '{{split}}' placeholder, got {template!r}")
    return template


def _splits(value: Any, where: str) -> Tuple[str, ...]:
    splits = _string_list(value, where)
    if not splits:
        raise ManifestError(f"{where} must name at least one split")
    unknown = [s for s in splits if s not in SPLITS]
    if unknown:
        raise ManifestError(
            f"{where} names unknown split(s) {', '.join(map(repr, unknown))}; "
            f"this package reads {', '.join(SPLITS)}"
        )
    return splits


def _per_split_counts(value: Any, splits: Tuple[str, ...], where: str) -> Mapping[str, int]:
    mapping = _mapping(value, where)
    keys, missing = set(mapping), set(splits) - set(mapping)
    if missing:
        raise ManifestError(
            f"{where} has no entry for split(s) {', '.join(sorted(missing))}; "
            f"every split the group publishes needs one"
        )
    extra = keys - set(splits)
    if extra:
        raise ManifestError(
            f"{where} describes split(s) {', '.join(sorted(extra))} that the group does not publish"
        )
    return {split: _integer(mapping[split], f"{where}[{split!r}]") for split in splits}


def _parse_release(value: Any, source: str) -> ReleaseInfo:
    where = _where(source, "release")
    data = _mapping(value, where)
    layout = _string(data.get("layout", "task-first"), f"{where}.layout")
    if layout not in SUPPORTED_LAYOUTS:
        raise ManifestError(
            f"{where}.layout is {layout!r}; this package reads {', '.join(SUPPORTED_LAYOUTS)}. "
            f"Update symtrs (pip install -U symtrs) or use an older manifest."
        )
    return ReleaseInfo(
        name=_string(data.get("name", "SyMTRS"), f"{where}.name"),
        revision=_string(data.get("revision"), f"{where}.revision"),
        commit=_string(data.get("commit", ""), f"{where}.commit", allow_empty=True),
        layout=layout,
        views=_integer(data.get("views", 0), f"{where}.views"),
        files=_integer(data.get("files", 0), f"{where}.files"),
        bytes=_integer(data.get("bytes", 0), f"{where}.bytes"),
    )


def _parse_repos(value: Any, source: str, release: ReleaseInfo) -> Dict[str, Repo]:
    where = _where(source, "repos")
    if not isinstance(value, list) or not value:
        raise ManifestError(f"{where} must be a non-empty list")
    repos: Dict[str, Repo] = {}
    for index, item in enumerate(value):
        at = f"{where}[{index}]"
        data = _mapping(item, at)
        key = _string(data.get("key"), f"{at}.key")
        repo_type = _string(data.get("type", "dataset"), f"{at}.type")
        if repo_type not in REPO_TYPES:
            raise ManifestError(
                f"{at}.type is {repo_type!r}; expected one of {', '.join(REPO_TYPES)}"
            )
        if key in repos:
            raise ManifestError(f"{where}: repo key {key!r} appears twice")
        repos[key] = Repo(
            key=key,
            repo=_string(data.get("repo"), f"{at}.repo"),
            type=repo_type,
            revision=_string(data.get("revision", release.revision), f"{at}.revision"),
        )
    if CATALOG_REPO_KEY not in repos:
        raise ManifestError(
            f"{where} must include the catalog repo {CATALOG_REPO_KEY!r}, "
            f"which carries {CATALOG_REPO_KEY}/symtrs-hub.json"
        )
    return repos


def _parse_file_repos(value: Any, source: str, repos: Mapping[str, Repo]) -> Dict[str, str]:
    where = _where(source, "file_repos")
    data = _mapping(value, where)
    out: Dict[str, str] = {}
    for prefix, key in data.items():
        prefix = _string(prefix, f"{where} key")
        if not prefix.endswith("/"):
            raise ManifestError(f"{where} prefix {prefix!r} must end with '/'")
        key = _string(key, f"{where}[{prefix!r}]")
        if key not in repos:
            raise ManifestError(
                f"{where}[{prefix!r}] names repo {key!r}, which the manifest does not declare"
            )
        out[prefix] = key
    return out


def _parse_group(value: Any, index: int, source: str, repos: Mapping[str, Repo]) -> Group:
    where = _where(source, f"groups[{index}]")
    data = _mapping(value, where)
    group_id = _string(data.get("id"), f"{where}.id")
    task = _string(data.get("task"), f"{where}.task")
    variant = _string(data.get("variant"), f"{where}.variant")
    repo = _string(data.get("repo"), f"{where}.repo")
    if repo not in repos:
        raise ManifestError(f"{where}.repo names {repo!r}, which the manifest does not declare")
    if repos[repo].type != "dataset":
        raise ManifestError(f"{where}.repo {repo!r} is a model repo; a task table needs a dataset repo")
    splits = _splits(data.get("splits"), f"{where}.splits")
    resolution_value = data.get("resolution")
    resolution = None if resolution_value is None else _integer(resolution_value, f"{where}.resolution", minimum=1)
    if resolution is not None and resolution not in RESOLUTIONS:
        raise ManifestError(
            f"{where}.resolution is {resolution}; this package names {RESOLUTIONS}"
        )
    native = _boolean(data.get("native", False), f"{where}.native")
    # The table is how the launcher knows which files a group needs; it is required even
    # for a release that publishes a flat tree, because the task repos hold the tables
    # rather than a second copy of the images.
    table = _table_template(data.get("table"), f"{where}.table")
    requires = _string_list(data.get("requires", []), f"{where}.requires")
    return Group(
        id=group_id,
        task=task,
        variant=variant,
        label=_string(data.get("label", f"{task} · {variant}"), f"{where}.label"),
        repo=repo,
        # Duplicates are legitimate here: super-resolution reads rgb_day at two
        # resolutions (the low-resolution input and the native target).
        modalities=_string_list(data.get("modalities", []), f"{where}.modalities", unique=False),
        splits=splits,
        table=table,
        requires=requires,
        bytes=_per_split_counts(data.get("bytes", {}), splits, f"{where}.bytes"),
        views=_per_split_counts(data.get("views", {}), splits, f"{where}.views"),
        table_bytes=_per_split_counts(data.get("table_bytes", {}), splits, f"{where}.table_bytes"),
        resolution=resolution,
        native=native,
    )


def _parse_model_file(value: Any, where: str) -> ModelFile:
    data = _mapping(value, where)
    sha256 = _string(data.get("sha256", ""), f"{where}.sha256", allow_empty=True)
    if sha256 and (len(sha256) != 64 or any(ch not in "0123456789abcdef" for ch in sha256.lower())):
        raise ManifestError(f"{where}.sha256 must be 64 hexadecimal characters, got {sha256!r}")
    return ModelFile(
        resolution=_integer(data.get("resolution"), f"{where}.resolution", minimum=1),
        path=safe_relative_path(_string(data.get("path"), f"{where}.path"), f"{where}.path"),
        bytes=_integer(data.get("bytes"), f"{where}.bytes"),
        sha256=sha256.lower(),
    )


def _parse_model(value: Any, index: int, source: str, repos: Mapping[str, Repo]) -> Model:
    where = _where(source, f"models[{index}]")
    data = _mapping(value, where)
    repo = _string(data.get("repo"), f"{where}.repo")
    if repo not in repos:
        raise ManifestError(f"{where}.repo names {repo!r}, which the manifest does not declare")
    if repos[repo].type != "model":
        raise ManifestError(f"{where}.repo {repo!r} is a dataset repo; a checkpoint needs a model repo")
    files_value = data.get("files")
    if not isinstance(files_value, list) or not files_value:
        raise ManifestError(f"{where}.files must be a non-empty list")
    files = tuple(
        _parse_model_file(item, f"{where}.files[{file_index}]")
        for file_index, item in enumerate(files_value)
    )
    paths = [f.path for f in files]
    if len(set(paths)) != len(paths):
        raise ManifestError(f"{where}.files contains the same path twice")
    return Model(
        id=_string(data.get("id"), f"{where}.id"),
        task=_string(data.get("task"), f"{where}.task"),
        registry_key=_string(data.get("registry_key", data.get("id")), f"{where}.registry_key"),
        repo=repo,
        description=_string(data.get("description", ""), f"{where}.description", allow_empty=True),
        protocol=_string(data.get("protocol", ""), f"{where}.protocol", allow_empty=True),
        output_space=_string(data.get("output_space", ""), f"{where}.output_space", allow_empty=True),
        requires_extra=_string(data.get("requires_extra", ""), f"{where}.requires_extra", allow_empty=True),
        files=files,
    )


def _parse_preset(value: Any, index: int, source: str) -> Preset:
    where = _where(source, f"presets[{index}]")
    data = _mapping(value, where)
    # Presets are named in the manifest so the launcher can accept `--preset depth-2048`
    # without a package release. They are plain selections; no table is fetched here.
    return Preset(
        id=_string(data.get("id"), f"{where}.id"),
        label=_string(data.get("label", data.get("id")), f"{where}.label"),
        groups=_string_list(data.get("groups", []), f"{where}.groups"),
        splits=_splits(data.get("splits", list(SPLITS)), f"{where}.splits"),
        models=_string_list(data.get("models", []), f"{where}.models"),
    )


def load_manifest_bytes(data: bytes, source: str = "<manifest>") -> Manifest:
    """Parse and validate the manifest bytes."""
    try:
        raw = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"{source} is not UTF-8 JSON: {exc}") from exc
    root = _mapping(raw, source)

    schema = _string(root.get("schema"), f"{source}: schema")
    if schema != SCHEMA:
        raise ManifestError(
            f"{source} has schema {schema!r}; this package reads {SCHEMA!r}. "
            f"Update symtrs (pip install -U symtrs) or point --manifest at a matching release."
        )

    release = _parse_release(root.get("release"), source)
    repos = _parse_repos(root.get("repos"), source, release)
    file_repos = _parse_file_repos(root.get("file_repos", {}), source, repos)

    groups_value = root.get("groups", [])
    if not isinstance(groups_value, list) or not groups_value:
        raise ManifestError(f"{source}: groups must be a non-empty list")
    groups: Dict[str, Group] = {}
    for index, value in enumerate(groups_value):
        group = _parse_group(value, index, source, repos)
        if group.id in groups:
            raise ManifestError(f"{source}: group id {group.id!r} appears twice")
        groups[group.id] = group

    for group in groups.values():
        for required in group.requires:
            if required not in groups:
                raise ManifestError(
                    f"{source}: group {group.id!r} requires {required!r}, which is not in the manifest"
                )

    default_groups = _string_list(root.get("default_groups", []), f"{source}: default_groups")
    for group_id in default_groups:
        if group_id not in groups:
            raise ManifestError(
                f"{source}: default_groups names {group_id!r}, which is not in the manifest"
            )

    models_value = root.get("models", [])
    if not isinstance(models_value, list):
        raise ManifestError(f"{source}: models must be a list")
    models: Dict[str, Model] = {}
    for index, value in enumerate(models_value):
        model = _parse_model(value, index, source, repos)
        if model.id in models:
            raise ManifestError(f"{source}: model id {model.id!r} appears twice")
        models[model.id] = model

    presets_value = root.get("presets", [])
    if not isinstance(presets_value, list):
        raise ManifestError(f"{source}: presets must be a list")
    presets: Dict[str, Preset] = {}
    for index, value in enumerate(presets_value):
        preset = _parse_preset(value, index, source)
        if preset.id in presets:
            raise ManifestError(f"{source}: preset id {preset.id!r} appears twice")
        unknown = [g for g in preset.groups if g not in groups]
        if unknown:
            raise ManifestError(
                f"{source}: preset {preset.id!r} names unknown group(s) {', '.join(map(repr, unknown))}"
            )
        unknown_models = [m for m in preset.models if m not in models]
        if unknown_models:
            raise ManifestError(
                f"{source}: preset {preset.id!r} names unknown model(s) "
                f"{', '.join(map(repr, unknown_models))}"
            )
        presets[preset.id] = preset

    digest = hashlib.sha256(data).hexdigest()
    return Manifest(
        schema=schema,
        release=release,
        repos=repos,
        file_repos=file_repos,
        default_groups=default_groups,
        groups=groups,
        models=models,
        presets=presets,
        digest=digest,
        source=source,
        raw=root,
    )


def load_manifest_file(path: Path) -> Manifest:
    """Read a manifest from disk, naming the path in every error."""
    path = Path(path)
    try:
        data = path.read_bytes()
    except FileNotFoundError as exc:
        raise ManifestError(
            f"no manifest at {path}\n"
            f"  point --manifest at the release's symtrs-hub.json, or use --mirror <dir>"
        ) from exc
    return load_manifest_bytes(data, source=str(path))


__all__ = [
    "Repo",
    "SCHEMA",
    "SUPPORTED_LAYOUTS",
    "Group",
    "Manifest",
    "ManifestError",
    "Model",
    "ModelFile",
    "Preset",
    "ReleaseInfo",
    "load_manifest_bytes",
    "load_manifest_file",
    "safe_relative_path",
]
