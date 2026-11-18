"""Turn a selection into an exact download plan.

This module is the launcher's arithmetic and it is deliberately pure: it reads a manifest,
reads task tables through an injected `table_source`, and returns the union of files with
their sizes. It never touches the network and never writes a file. That is what lets the
same code answer `--plan` (print it), `--yes` (download it), the TUI's footer (price it)
and the tests (assert the dedupe) without four implementations of the same rule.

The rule that matters: a view's `rgb_day_2048` file is shared by depth and segmentation, so
selecting both downloads it once. The plan records every group that references a file and
reports the bytes saved, because "pull only what you need" is only true if the launcher
does not pull the same pixels twice.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from symtrs.data.splits import SPLITS

from symtrs.hub.errors import ManifestError, SelectionError
from symtrs.hub.manifest import Group, Manifest, Model, ModelFile, safe_relative_path

#: `(repo key, path) -> rows of {"files": [[path, bytes], ...]}`. `fetch.BackendSet`
#: supplies it; tests supply a dict. The repo key is first because a task table lives in
#: its vertical's project, not in the view store that holds the images.
TableSource = Callable[[str, str], Iterable[Mapping[str, Any]]]

#: Ask for spare room beyond the bytes we know about: a partial file, filesystem block
#: rounding and a partially written file all need a little, and failing at 99% of
#: an 800 GB download is the worst possible time to discover there is no headroom.
HEADROOM = 1.05


@dataclass(frozen=True)
class FileRef:
    """One file in the plan. Dataset refs carry every group that references them."""

    path: str
    bytes: int
    repo: str = ""
    groups: Tuple[str, ...] = ()
    model: str = ""
    resolution: Optional[int] = None
    sha256: str = ""

    @property
    def is_model(self) -> bool:
        return bool(self.model)


@dataclass(frozen=True)
class GroupPlan:
    """One selected group, split by split.

    `bytes` counts every reference, so a shared `rgb_day` appears in both groups' rows; the
    plan's `dataset_bytes` is the deduplicated total. Both numbers are shown so the saving
    is visible rather than implied.
    """

    id: str
    task: str
    variant: str
    splits: Tuple[str, ...]
    by_split: Mapping[str, Tuple[str, ...]]
    files: int
    bytes: int


@dataclass(frozen=True)
class ModelPlan:
    id: str
    task: str
    resolutions: Tuple[int, ...]
    files: int
    bytes: int


@dataclass(frozen=True)
class Plan:
    manifest: Manifest
    dataset: Tuple[FileRef, ...]
    models: Tuple[FileRef, ...]
    groups: Tuple[GroupPlan, ...]
    model_plans: Tuple[ModelPlan, ...]

    @property
    def dataset_bytes(self) -> int:
        return sum(ref.bytes for ref in self.dataset)

    @property
    def model_bytes(self) -> int:
        return sum(ref.bytes for ref in self.models)

    @property
    def total_bytes(self) -> int:
        return self.dataset_bytes + self.model_bytes

    @property
    def referenced_dataset_bytes(self) -> int:
        return sum(group.bytes for group in self.groups)

    @property
    def shared_bytes(self) -> int:
        """Bytes of files referenced by more than one selected group."""
        return sum(ref.bytes for ref in self.dataset if len(ref.groups) > 1)

    @property
    def saved_bytes(self) -> int:
        """Duplicate references avoided: referenced bytes minus unique bytes."""
        return self.referenced_dataset_bytes - self.dataset_bytes

    @property
    def empty(self) -> bool:
        return not self.dataset and not self.models

    @property
    def used_repos(self) -> Tuple[str, ...]:
        """Repo keys the plan reads, dataset first, in first-use order."""
        seen: List[str] = []
        for ref in (*self.dataset, *self.models):
            if ref.repo and ref.repo not in seen:
                seen.append(ref.repo)
        return tuple(seen)


@dataclass
class Selection:
    """What someone asked for: groups with splits, models with resolutions.

    An empty resolution tuple means every published resolution of that model, which is why
    the distinction between `()` and `(2048,)` is kept rather than collapsed to a set.
    """

    groups: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    models: Dict[str, Tuple[int, ...]] = field(default_factory=dict)

    def add_group(self, group_id: str, splits: Sequence[str]) -> None:
        current = set(self.groups.get(group_id, ()))
        current.update(splits)
        self.groups[group_id] = tuple(split for split in SPLITS if split in current)

    def add_model(self, model_id: str, resolutions: Sequence[int] = ()) -> None:
        if resolutions:
            self.models[model_id] = tuple(sorted(set(resolutions), reverse=True))
        else:
            self.models.setdefault(model_id, ())

    @property
    def empty(self) -> bool:
        return not self.groups and not self.models


def _close(text: str, choices: Iterable[str]) -> str:
    import difflib

    matches = difflib.get_close_matches(text, list(choices), n=3, cutoff=0.6)
    return f" Did you mean {', '.join(map(repr, matches))}?" if matches else ""


def select_tasks(
    manifest: Manifest,
    tasks: Sequence[str],
    *,
    splits: Sequence[str] = (),
    resolutions: Sequence[int] = (),
    variants: Sequence[str] = (),
) -> Selection:
    """Every group of `tasks`, filtered by resolution and variant.

    A requested split or resolution that no matching group publishes is an error, not a
    silent narrowing: `--splits test` that installs train and val would make a benchmark
    number incomparable without saying so.
    """
    split_filter = tuple(splits) or ()
    resolution_filter = set(resolutions)
    variant_filter = set(variants)
    selection = Selection()
    matched = 0
    for group in manifest.groups.values():
        if group.task not in tasks:
            continue
        if resolution_filter and group.resolution not in resolution_filter:
            continue
        if variant_filter and group.variant not in variant_filter:
            continue
        matched += 1
        wanted = tuple(s for s in group.splits if not split_filter or s in split_filter)
        missing = sorted(set(split_filter) - set(group.splits))
        if missing:
            raise SelectionError(
                f"group {group.id!r} does not publish split(s) {', '.join(missing)}; "
                f"it has {', '.join(group.splits)}"
            )
        if not wanted:
            raise SelectionError(
                f"no split selected for group {group.id!r}; it has {', '.join(group.splits)}"
            )
        selection.add_group(group.id, wanted)
    if matched == 0:
        filters = []
        if resolution_filter:
            filters.append(f"resolution(s) {', '.join(map(str, sorted(resolution_filter)))}")
        if variant_filter:
            filters.append(f"variant(s) {', '.join(sorted(variant_filter))}")
        suffix = " with " + " and ".join(filters) if filters else ""
        raise SelectionError(
            f"no published group for task(s) {', '.join(tasks)}{suffix}. "
            f"Published tasks: {', '.join(manifest.tasks())}. Run `symtrs get --list`."
        )
    return selection


def parse_model_spec(manifest: Manifest, items: Sequence[str]) -> Dict[str, Tuple[int, ...]]:
    """`["midas_small@1024", "segformer_b2"]` -> `{id: (1024,), id: ()}`.

    An empty tuple means "every resolution the model publishes"; `@all` says the same
    thing explicitly. A resolution the model does not publish is refused with the list,
    because a typo that quietly installs the wrong level is the mistake this benchmark
    cannot afford.
    """
    out: Dict[str, Tuple[int, ...]] = {}
    for item in items:
        model_id, _, resolution_text = item.partition("@")
        model_id = model_id.strip()
        if model_id == "all":
            for known in manifest.models:
                out.setdefault(known, ())
            continue
        model = manifest.models.get(model_id)
        if model is None:
            raise SelectionError(
                f"no published model {model_id!r}.{_close(model_id, manifest.models)} "
                f"Run `symtrs get --list`."
            )
        if not resolution_text or resolution_text.strip().lower() == "all":
            out.setdefault(model_id, ())
            continue
        try:
            resolutions = tuple(int(part) for part in resolution_text.split("+") if part.strip())
        except ValueError:
            raise SelectionError(
                f"{item!r}: the part after '@' must be a resolution (1024), several joined "
                f"with '+', or 'all', not {resolution_text!r}"
            ) from None
        unknown = sorted(set(resolutions) - set(model.resolutions))
        if unknown:
            raise SelectionError(
                f"model {model_id!r} publishes resolution(s) "
                f"{', '.join(map(str, model.resolutions))}, not {', '.join(map(str, unknown))}"
            )
        out[model_id] = tuple(sorted(set(out.get(model_id, ())) | set(resolutions), reverse=True))
    return out


def _expand_requires(manifest: Manifest, groups: Mapping[str, Tuple[str, ...]]) -> Dict[str, Tuple[str, ...]]:
    """Add groups named in `requires`, with the same splits as the group that needs them.

    A depth table without `metadata/` is a table whose columns cannot be interpreted; the
    requirement is in the manifest, so the launcher obeys it instead of leaving a user to
    discover a missing camera.yaml later.
    """
    expanded = {group_id: tuple(splits) for group_id, splits in groups.items()}
    queue = list(expanded.items())
    while queue:
        group_id, splits = queue.pop(0)
        for required in manifest.groups[group_id].requires:
            current = set(expanded.get(required, ()))
            merged = tuple(split for split in SPLITS if split in current | set(splits))
            if set(merged) != current:
                expanded[required] = merged
                queue.append((required, merged))
    return expanded


def _rows_for(table_source: TableSource, group: Group, split: str) -> List[Tuple[str, int]]:
    path = group.table_for(split)
    try:
        rows = table_source(group.repo, path)
        out: List[Tuple[str, int]] = []
        for row_index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise ManifestError(f"{path}: row {row_index} must be an object")
            files = row.get("files")
            if not isinstance(files, (list, tuple)) or not files:
                raise ManifestError(f"{path}: row {row_index} has no files list")
            for file_index, entry in enumerate(files):
                if isinstance(entry, Mapping):
                    file_path = entry.get("path")
                    size = entry.get("bytes")
                elif isinstance(entry, (list, tuple)) and len(entry) == 2:
                    file_path, size = entry
                else:
                    raise ManifestError(
                        f"{path}: row {row_index}.files[{file_index}] must be [path, bytes] "
                        f"or {{\"path\": ..., \"bytes\": ...}}"
                    )
                if not isinstance(file_path, str):
                    raise ManifestError(f"{path}: row {row_index} has a non-string file path")
                if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                    raise ManifestError(f"{path}: row {row_index} has a bad byte count {size!r}")
                out.append((safe_relative_path(file_path, f"{path}: row {row_index}"), size))
        return out
    except ManifestError:
        raise
    except OSError as exc:
        raise ManifestError(f"cannot read table {path}: {exc}") from exc
    except json.JSONDecodeError as exc:  # a backend may stream JSON lines itself
        raise ManifestError(f"{path} is not valid JSON lines: {exc}") from exc


def resolve(manifest: Manifest, selection: Selection, table_source: TableSource) -> Plan:
    """Resolve a selection to the exact union of files, deduplicated by path."""
    if selection.empty:
        raise SelectionError(
            "nothing selected. Name groups or models, or run `symtrs get --list` to see the release."
        )

    unknown = [group_id for group_id in selection.groups if group_id not in manifest.groups]
    if unknown:
        raise SelectionError(
            f"no published group {unknown[0]!r}.{_close(unknown[0], manifest.groups)} "
            f"Run `symtrs get --list`."
        )

    requested = {group_id: tuple(splits) for group_id, splits in selection.groups.items()}
    # A release can nominate groups that ride along with any vertical (metadata). They are
    # added only when a vertical is selected: a models-only pull must not create a dataset
    # root because the manifest happened to name a default.
    if requested:
        union = tuple(s for s in SPLITS if any(s in splits for splits in requested.values()))
        for group_id in manifest.default_groups:
            if group_id in requested:
                continue
            group = manifest.groups[group_id]
            available = tuple(split for split in union if split in group.splits)
            if not available:
                raise SelectionError(
                    f"default group {group_id!r} does not publish any of the selected split(s) "
                    f"{', '.join(union)}; it has {', '.join(group.splits)}"
                )
            requested[group_id] = available

    groups = _expand_requires(manifest, requested)
    for group_id, splits in groups.items():
        published = set(manifest.groups[group_id].splits)
        missing = [split for split in splits if split not in published]
        if missing:
            raise SelectionError(
                f"group {group_id!r} does not publish split(s) {', '.join(missing)}; "
                f"it has {', '.join(manifest.groups[group_id].splits)}"
            )

    # Order is the manifest's order, not the selection's: two people selecting the same
    # groups in a different order must get byte-identical plans.
    ordered = [group_id for group_id in manifest.groups if group_id in groups]

    refs: Dict[str, Dict[str, Any]] = {}

    def add_ref(path: str, size: int, group_id: str, repo: str) -> None:
        record = refs.get(path)
        if record is None:
            refs[path] = {
                "bytes": size,
                "repo": repo,
                "groups": [group_id],
                "resolution": None,
                "sha256": "",
            }
            return
        if record["bytes"] != size:
            raise ManifestError(
                f"the release disagrees with itself about {path!r}: "
                f"{record['bytes']} bytes in one table, {size} in another"
            )
        if record["repo"] != repo:
            raise ManifestError(
                f"the release publishes {path!r} in both {record['repo']!r} and {repo!r}; "
                f"the launcher cannot know which copy is the data"
            )
        if group_id not in record["groups"]:
            record["groups"].append(group_id)

    group_plans: List[GroupPlan] = []
    for group_id in ordered:
        group = manifest.groups[group_id]
        by_split: Dict[str, Tuple[str, ...]] = {}
        referenced = 0
        for split in groups[group_id]:
            # The table itself is part of what makes a task usable offline, so it is
            # installed with the data and counted in the plan. Its size is advertised by
            # the manifest, because reading it to find out would download it twice.
            table_path = group.table_for(split)
            paths: List[str] = [table_path]
            table_size = group.table_bytes.get(split, 0)
            add_ref(table_path, table_size, group_id, group.repo)
            referenced += table_size
            for path, size in _rows_for(table_source, group, split):
                add_ref(path, size, group_id, manifest.repo_for_path(path, group.repo))
                paths.append(path)
                referenced += size
            by_split[split] = tuple(paths)
        group_plans.append(
            GroupPlan(
                id=group.id,
                task=group.task,
                variant=group.variant,
                splits=tuple(groups[group_id]),
                by_split=by_split,
                files=sum(len(paths) for paths in by_split.values()),
                bytes=referenced,
            )
        )

    dataset = tuple(
        FileRef(
            path=path,
            bytes=int(record["bytes"]),
            repo=str(record["repo"]),
            groups=tuple(record["groups"]),
            resolution=record["resolution"],
            sha256=record["sha256"],
        )
        for path, record in refs.items()
    )

    model_refs: Dict[str, FileRef] = {}
    model_plans: List[ModelPlan] = []
    for model_id in selection.models:
        model = manifest.models.get(model_id)
        if model is None:
            raise SelectionError(
                f"no published model {model_id!r}.{_close(model_id, manifest.models)} "
                f"Run `symtrs get --list`."
            )
        wanted = selection.models[model_id] or model.resolutions
        files = model.files_for(wanted)
        missing = sorted(set(wanted) - set(model.resolutions))
        if missing:
            raise SelectionError(
                f"model {model_id!r} publishes resolution(s) "
                f"{', '.join(map(str, model.resolutions))}, not {', '.join(map(str, missing))}"
            )
        for model_file in files:
            _add_model_ref(model_refs, model, model_file)
        model_plans.append(
            ModelPlan(
                id=model.id,
                task=model.task,
                resolutions=tuple(sorted(set(wanted), reverse=True)),
                files=len(files),
                bytes=sum(f.bytes for f in files),
            )
        )

    return Plan(
        manifest=manifest,
        dataset=dataset,
        models=tuple(model_refs.values()),
        groups=tuple(group_plans),
        model_plans=tuple(model_plans),
    )


def _add_model_ref(refs: Dict[str, FileRef], model: Model, model_file: ModelFile) -> None:
    path = safe_relative_path(model_file.path, f"model {model.id!r} file path")
    existing = refs.get(path)
    if existing is not None:
        if existing.bytes != model_file.bytes:
            raise ManifestError(
                f"two models publish {path!r} with different sizes "
                f"({existing.model!r} and {model.id!r}); the launcher would overwrite one"
            )
        return
    refs[path] = FileRef(
        path=path,
        bytes=model_file.bytes,
        repo=model.repo,
        model=model.id,
        resolution=model_file.resolution,
        sha256=model_file.sha256,
    )


@dataclass(frozen=True)
class DestinationReport:
    """What already exists at one destination, and whether the plan fits."""

    root: Path
    present: Tuple[str, ...]
    missing: Tuple[str, ...]
    conflicts: Tuple[str, ...]
    missing_bytes: int
    present_bytes: int
    free_bytes: Optional[int]

    @property
    def fits(self) -> bool:
        if self.free_bytes is None:
            return True
        return self.missing_bytes * HEADROOM <= self.free_bytes

    @property
    def required_bytes(self) -> int:
        return int(self.missing_bytes * HEADROOM)


def _free_bytes(root: Path) -> Optional[int]:
    from shutil import disk_usage

    for candidate in (root, *root.parents):
        if candidate.exists():
            try:
                return disk_usage(candidate).free
            except OSError:
                return None
    return None


def inspect_destination(refs: Sequence[FileRef], root: Path) -> DestinationReport:
    """Compare the plan with the destination. Pure filesystem reads, no writes."""
    root = Path(root)
    present: List[str] = []
    missing: List[str] = []
    conflicts: List[str] = []
    present_bytes = 0
    missing_bytes = 0
    for ref in refs:
        path = root / ref.path
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            missing.append(ref.path)
            missing_bytes += ref.bytes
            continue
        except OSError:
            conflicts.append(ref.path)
            missing_bytes += ref.bytes
            continue
        if size == ref.bytes:
            present.append(ref.path)
            present_bytes += ref.bytes
        else:
            conflicts.append(ref.path)
            missing_bytes += ref.bytes
    return DestinationReport(
        root=root,
        present=tuple(present),
        missing=tuple(missing),
        conflicts=tuple(conflicts),
        missing_bytes=missing_bytes,
        present_bytes=present_bytes,
        free_bytes=_free_bytes(root),
    )


def plan_as_dict(plan: Plan, data_root: Optional[Path] = None, models_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Machine-readable plan for `--json` and for the tests that pin the arithmetic."""
    out: Dict[str, Any] = {
        "manifest": plan.manifest.source,
        "digest": plan.manifest.digest,
        "revision": plan.manifest.release.revision,
        "commit": plan.manifest.release.commit,
        "repos": list(plan.used_repos),
        "dataset": {
            "files": len(plan.dataset),
            "bytes": plan.dataset_bytes,
            "shared_bytes": plan.shared_bytes,
            "groups": [
                {
                    "id": group.id,
                    "repo": plan.manifest.groups[group.id].repo,
                    "splits": list(group.splits),
                    "files": group.files,
                    "bytes": group.bytes,
                }
                for group in plan.groups
            ],
            "paths": [ref.path for ref in plan.dataset],
        },
        "models": {
            "files": len(plan.models),
            "bytes": plan.model_bytes,
            "entries": [
                {
                    "id": entry.id,
                    "repo": plan.manifest.models[entry.id].repo,
                    "resolutions": list(entry.resolutions),
                    "files": entry.files,
                    "bytes": entry.bytes,
                }
                for entry in plan.model_plans
            ],
            "paths": [ref.path for ref in plan.models],
        },
    }
    if data_root is not None:
        report = inspect_destination(plan.dataset, data_root)
        out["dataset"]["destination"] = str(data_root)
        out["dataset"]["present"] = len(report.present)
        out["dataset"]["missing"] = len(report.missing)
        out["dataset"]["conflicts"] = list(report.conflicts)
        out["dataset"]["free_bytes"] = report.free_bytes
        out["dataset"]["fits"] = report.fits
    if models_dir is not None:
        report = inspect_destination(plan.models, models_dir)
        out["models"]["destination"] = str(models_dir)
        out["models"]["present"] = len(report.present)
        out["models"]["missing"] = len(report.missing)
        out["models"]["conflicts"] = list(report.conflicts)
        out["models"]["free_bytes"] = report.free_bytes
        out["models"]["fits"] = report.fits
    return out


__all__ = [
    "DestinationReport",
    "FileRef",
    "GroupPlan",
    "HEADROOM",
    "ModelPlan",
    "Plan",
    "Selection",
    "TableSource",
    "inspect_destination",
    "parse_model_spec",
    "plan_as_dict",
    "resolve",
    "select_tasks",
]
