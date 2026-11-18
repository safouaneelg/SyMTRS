"""The install lockfile: what a destination root actually contains.

A partial download is the normal state of a dataset root - someone pulls depth/2048 for a
depth paper and nothing else - and the package has to be able to say what is there, from
which release, and whether it is intact. That is what `.symtrs/install.json` records. It
is written under the destination, never inside `views/` or `tasks/`, so a dataset root
remains exactly the published tree plus one small directory.

The lock deliberately does not list every dataset file: a table already does that, and a
40,000-line JSON array would be a second copy to keep in step. It records the table path
and the per-split totals; `verify` re-reads the table. Models are few enough to list by
path and size, which makes `--verify` possible without the manifest too.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from symtrs.hub.errors import HubError
from symtrs.hub.manifest import Group, Manifest, Model

LOCK_DIR = ".symtrs"
LOCK_NAME = "install.json"
LOCK_SCHEMA = 1


@dataclass
class Lock:
    """One destination's record. `kind` is `dataset` or `models`."""

    kind: str
    schema: int = LOCK_SCHEMA
    revision: str = ""
    commit: str = ""
    manifest_sha256: str = ""
    mixed: bool = False
    previous_revisions: List[str] = field(default_factory=list)
    #: repo key -> {"repo", "type", "revision"} for every project this install touched
    repos: Dict[str, Dict[str, str]] = field(default_factory=dict)
    #: group id -> split -> {"table": path, "files": n, "bytes": n}
    groups: Dict[str, Dict[str, Dict[str, Any]]] = field(default_factory=dict)
    #: model id -> {"repo", "revision", "resolutions": [...], "files": {path: bytes}}
    models: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    updated: str = ""

    @property
    def revision_label(self) -> str:
        if not self.revision:
            return "unknown"
        return f"{self.revision} (mixed)" if self.mixed else self.revision

    @property
    def empty(self) -> bool:
        return not self.groups and not self.models

    def installed_splits(self, group_id: str) -> Tuple[str, ...]:
        return tuple(sorted(self.groups.get(group_id, {})))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "kind": self.kind,
            "revision": self.revision,
            "commit": self.commit,
            "manifest_sha256": self.manifest_sha256,
            "mixed": self.mixed,
            "previous_revisions": list(self.previous_revisions),
            "repos": self.repos,
            "groups": self.groups,
            "models": self.models,
            "updated": self.updated,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], path: Path) -> "Lock":
        if data.get("schema") != LOCK_SCHEMA:
            raise HubError(
                f"{path} has lock schema {data.get('schema')!r}; this package writes {LOCK_SCHEMA}. "
                f"Move it aside (the files it describes are still usable) and run "
                f"`symtrs get --verify` after reinstalling."
            )
        kind = data.get("kind")
        if kind not in ("dataset", "models"):
            raise HubError(f"{path}: lock kind must be 'dataset' or 'models', got {kind!r}")
        return cls(
            kind=kind,
            schema=LOCK_SCHEMA,
            revision=str(data.get("revision", "")),
            commit=str(data.get("commit", "")),
            manifest_sha256=str(data.get("manifest_sha256", "")),
            mixed=bool(data.get("mixed", False)),
            previous_revisions=[str(item) for item in data.get("previous_revisions", [])],
            repos=dict(data.get("repos", {})),
            groups=dict(data.get("groups", {})),
            models=dict(data.get("models", {})),
            updated=str(data.get("updated", "")),
        )


def lock_path(root: Path) -> Path:
    return Path(root) / LOCK_DIR / LOCK_NAME


def read_lock(root: Path) -> Optional[Lock]:
    """The lock for `root`, or None when nothing has been installed there yet."""
    path = lock_path(root)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise HubError(
            f"cannot read {path}: {exc}\n"
            f"  remove it and run `symtrs get --verify`; the installed files are unaffected"
        ) from exc
    return Lock.from_dict(data, path)


def write_lock(root: Path, lock: Lock) -> Path:
    """Write the lock atomically so an interrupted install cannot leave half a JSON file."""
    path = lock_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock.updated = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(lock.to_dict(), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
    return path


def note_manifest(root: Path, lock: Lock, manifest: Manifest, *, allow_mixed: bool = False) -> None:
    """Record which release this install came from, refusing a silent mixture by default.

    Mixing two releases in one root changes what a resolution or a split means between
    files, and the resulting benchmark number cannot be defended. The flag exists for the
    person who has a reason and is willing to say so; it is not a convenience.
    """
    revision = manifest.release.revision
    if lock.revision and lock.revision != revision:
        if not allow_mixed:
            raise HubError(
                f"{Path(root)} already holds release {lock.revision!r}, "
                f"but this manifest is {revision!r}.\n"
                f"  use a fresh --data-root/--models-dir, or pass --allow-mixed-revision "
                f"and record why"
            )
        lock.mixed = True
        if lock.revision not in lock.previous_revisions:
            lock.previous_revisions.append(lock.revision)
    lock.revision = revision
    lock.commit = manifest.release.commit
    lock.manifest_sha256 = manifest.digest
    lock.repos = {
        key: {"repo": repo.repo, "type": repo.type, "revision": repo.revision}
        for key, repo in manifest.repos.items()
    }


def record_group(
    lock: Lock,
    group: Group,
    split: str,
    table: str,
    files: int,
    bytes_: int,
    table_bytes: int = 0,
) -> None:
    lock.groups.setdefault(group.id, {})[split] = {
        "table": table,
        "table_bytes": int(table_bytes),
        "files": files,
        "bytes": bytes_,
    }


def record_model(
    lock: Lock,
    model: Model,
    files: Mapping[str, int],
    resolutions: Iterable[int] = (),
    revision: str = "",
) -> None:
    entry = lock.models.setdefault(model.id, {})
    recorded = set(entry.get("resolutions", [])) | set(resolutions)
    entry.update(
        {
            "repo": model.repo,
            "revision": revision,
            "resolutions": sorted(recorded),
            "files": {path: int(size) for path, size in files.items()},
        }
    )


def _iter_table(path: Path) -> Iterable[Mapping[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise HubError(f"{path}:{line_number} is not valid JSON: {exc}") from exc
            if not isinstance(row, Mapping):
                raise HubError(f"{path}:{line_number} must be a JSON object")
            yield row


@dataclass(frozen=True)
class VerifyIssue:
    """One file that is missing, the wrong size, or does not match its recorded hash."""

    path: Path
    kind: str
    expected: str = ""
    actual: str = ""

    def describe(self) -> str:
        if self.kind == "missing":
            return f"missing    {self.path}"
        if self.kind == "size":
            return f"size       {self.path}  expected {self.expected}, found {self.actual}"
        if self.kind == "sha256":
            return f"checksum   {self.path}  expected {self.expected[:12]}…, found {self.actual[:12]}…"
        return f"{self.kind:<10} {self.path}  {self.expected} {self.actual}".rstrip()


@dataclass
class VerifyReport:
    root: Path
    kind: str
    groups_checked: int = 0
    models_checked: int = 0
    files_checked: int = 0
    bytes_checked: int = 0
    issues: List[VerifyIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues

    def add(self, issue: VerifyIssue) -> None:
        self.issues.append(issue)


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def verify_dataset(root: Path, lock: Lock) -> VerifyReport:
    """Check every installed group against the table it was installed from."""
    root = Path(root)
    report = VerifyReport(root=root, kind="dataset")
    for group_id in sorted(lock.groups):
        for split in sorted(lock.groups[group_id]):
            entry = lock.groups[group_id][split]
            table = root / str(entry.get("table", ""))
            if not entry.get("table") or not table.is_file():
                report.add(VerifyIssue(table, "missing", f"table for {group_id}/{split}"))
                continue
            expected_table = int(entry.get("table_bytes", 0))
            actual_table = table.stat().st_size
            if expected_table and actual_table != expected_table:
                report.add(VerifyIssue(table, "size", str(expected_table), str(actual_table)))
                # Do not parse a truncated archive: it would either raise or, worse, stop
                # mid-file and report the missing rows as verified.
                continue
            report.groups_checked += 1
            try:
                rows = list(_iter_table(table))
            except (OSError, EOFError, HubError) as exc:
                report.add(VerifyIssue(table, "table", str(exc)))
                continue
            for row in rows:
                files = row.get("files") or []
                for item in files:
                    if isinstance(item, Mapping):
                        relative, size = item.get("path"), item.get("bytes")
                    else:
                        relative, size = item
                    path = root / str(relative)
                    try:
                        actual = path.stat().st_size
                    except OSError:
                        report.add(VerifyIssue(path, "missing"))
                        continue
                    report.files_checked += 1
                    report.bytes_checked += int(size)
                    if actual != int(size):
                        report.add(VerifyIssue(path, "size", str(size), str(actual)))
    return report


def verify_models(root: Path, lock: Lock, manifest: Optional[Manifest] = None) -> VerifyReport:
    """Check model files by size and, when a hash is recorded, by sha256.

    The hash is the only way to catch a truncated or corrupted checkpoint whose size still
    matches, so `--verify` pays the read. Installation does not: it already knows the file
    came from the backend in one piece.
    """
    root = Path(root)
    report = VerifyReport(root=root, kind="models")
    hashes: Dict[str, str] = {}
    if manifest is not None:
        for model in manifest.models.values():
            for model_file in model.files:
                if model_file.sha256:
                    hashes[model_file.path] = model_file.sha256
    for model_id in sorted(lock.models):
        entry = lock.models[model_id]
        files = entry.get("files", {})
        if not isinstance(files, Mapping):
            continue
        report.models_checked += 1
        for relative, expected_size in files.items():
            path = root / str(relative)
            try:
                actual_size = path.stat().st_size
            except OSError:
                report.add(VerifyIssue(path, "missing"))
                continue
            report.files_checked += 1
            report.bytes_checked += int(expected_size)
            if actual_size != int(expected_size):
                report.add(VerifyIssue(path, "size", str(expected_size), str(actual_size)))
                continue
            expected_hash = hashes.get(str(relative))
            if expected_hash:
                actual_hash = file_sha256(path)
                if actual_hash != expected_hash:
                    report.add(VerifyIssue(path, "sha256", expected_hash, actual_hash))
    return report


def repair_paths(report: VerifyReport) -> List[Path]:
    """Files `--repair` must delete before fetching again.

    A missing file needs no deletion; a file with the wrong size or hash does, because the
    downloader skips a path whose size already matches (the common resume rule).
    """
    return [issue.path for issue in report.issues if issue.kind in ("size", "sha256")]


__all__ = [
    "LOCK_DIR",
    "LOCK_NAME",
    "LOCK_SCHEMA",
    "Lock",
    "VerifyIssue",
    "VerifyReport",
    "file_sha256",
    "lock_path",
    "note_manifest",
    "read_lock",
    "record_group",
    "record_model",
    "repair_paths",
    "verify_dataset",
    "verify_models",
    "write_lock",
]
