"""Execute a plan: download the published files, record the lockfile.

The installer is the only module that writes into a destination. It keeps three promises:

  * **Resume rather than restart.** A file whose size already matches is skipped; a partial
    download from an interrupted run is replaced, not appended to.
  * **Say what happened.** A cancelled or failed run still records the groups that are
    complete, so `--verify` and the next run start from the truth.
  * **Never half-claim.** A group is marked installed only when every file its table lists
    exists at the advertised size.

Downloads run in a thread pool because the cost here is the network, not the CPU. Hugging
Face's downloader is thread-safe and deduplicates metadata calls, and a local mirror is a
set of independent copies. Progress callbacks are invoked from the calling thread, which is
what lets the TUI update widgets without reaching into worker threads.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from symtrs.hub.errors import DiskSpaceError, HubError
from symtrs.hub.fetch import Backend, BackendSet
from symtrs.hub.plan import FileRef, Plan, inspect_destination
from symtrs.hub.state import (
    Lock,
    note_manifest,
    read_lock,
    record_group,
    record_model,
    write_lock,
)

#: `(files_done, files_total, bytes_done, bytes_total, current_path)`.
Progress = Callable[[int, int, int, int, str], None]


@dataclass
class InstallReport:
    dataset_files: int = 0
    dataset_bytes: int = 0
    model_files: int = 0
    model_bytes: int = 0
    skipped_files: int = 0
    skipped_bytes: int = 0
    groups_installed: int = 0
    models_installed: int = 0
    cancelled: bool = False
    seconds: float = 0.0
    data_root: Optional[Path] = None
    models_dir: Optional[Path] = None

    @property
    def downloaded_files(self) -> int:
        return self.dataset_files + self.model_files

    @property
    def downloaded_bytes(self) -> int:
        return self.dataset_bytes + self.model_bytes


def human_bytes(value: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    number = float(value)
    for unit in units:
        if number < 1024 or unit == units[-1]:
            return f"{int(number)} B" if unit == "B" else f"{number:.1f} {unit}"
        number /= 1024
    return f"{value} B"


def _needs_download(ref: FileRef, root: Path) -> Tuple[bool, int]:
    """`(download, existing_bytes)` for one file."""
    path = root / ref.path
    try:
        size = path.stat().st_size
    except OSError:
        return True, 0
    if size == ref.bytes:
        return False, size
    return True, size


def _remove_mismatched(refs: Sequence[FileRef], root: Path) -> None:
    """Delete files whose size is not the advertised one, before a forced/repair fetch.

    A backend resumes by size; without the delete, `--force` on a truncated file would call
    the backend and get the same truncated file back as "already present".
    """
    report = inspect_destination(refs, root)
    for path_text in report.conflicts:
        path = root / path_text
        if path.is_dir():
            raise HubError(f"cannot replace {path}: it is a directory, not a file")
        path.unlink(missing_ok=True)


def _check_space(refs: Sequence[FileRef], root: Path) -> None:
    report = inspect_destination(refs, root)
    if not report.fits:
        raise DiskSpaceError(
            f"not enough space at {root}: the missing files need "
            f"{human_bytes(report.required_bytes)} (including 5% headroom) "
            f"but only {human_bytes(report.free_bytes or 0)} is free.\n"
            f"  choose a smaller selection, or another --data-root / --models-dir"
        )


def _download_refs(
    pairs: Sequence[Tuple[Path, FileRef, Backend]],
    *,
    jobs: int,
    progress: Optional[Progress],
    cancel: Optional[threading.Event],
) -> bool:
    """Download `(root, ref, backend)` triples. Returns True when cancelled early."""
    if not pairs:
        return False
    total_files = len(pairs)
    total_bytes = sum(ref.bytes for _, ref, _ in pairs)
    done_files = 0
    done_bytes = 0
    cancelled = False
    executor = ThreadPoolExecutor(max_workers=max(1, jobs))
    futures = {}
    try:
        for root, ref, backend in pairs:
            futures[executor.submit(backend.download, ref.path, root)] = ref
        for future in as_completed(futures):
            if cancel is not None and cancel.is_set():
                cancelled = True
                break
            ref = futures[future]
            written = future.result()
            done_files += 1
            done_bytes += int(written) if written else ref.bytes
            if progress is not None:
                progress(done_files, total_files, done_bytes, total_bytes, ref.path)
    finally:
        executor.shutdown(wait=not cancelled, cancel_futures=cancelled)
    return cancelled


def _mark_complete(
    plan: Plan,
    lock: Lock,
    data_root: Optional[Path],
    models_dir: Optional[Path],
) -> Tuple[int, int]:
    """Record only groups/models whose every file is present at the right size."""
    sizes: Dict[str, int] = {ref.path: ref.bytes for ref in plan.dataset}
    groups = 0
    if data_root is not None:
        for group_plan in plan.groups:
            group = plan.manifest.groups[group_plan.id]
            for split in group_plan.splits:
                paths = group_plan.by_split.get(split, ())
                if not paths:
                    continue
                if not all(
                    (data_root / path).is_file()
                    and (data_root / path).stat().st_size == sizes[path]
                    for path in paths
                ):
                    continue
                record_group(
                    lock,
                    group,
                    split,
                    group.table_for(split),
                    files=len(paths),
                    bytes_=sum(sizes[path] for path in paths),
                    table_bytes=group.table_bytes.get(split, 0),
                )
                groups += 1
    models = 0
    if models_dir is not None:
        for model_plan in plan.model_plans:
            model = plan.manifest.models[model_plan.id]
            files = model.files_for(model_plan.resolutions)
            if not files:
                continue
            if not all(
                (models_dir / f.path).is_file()
                and (models_dir / f.path).stat().st_size == f.bytes
                for f in files
            ):
                continue
            record_model(
                lock,
                model,
                {f.path: f.bytes for f in files},
                resolutions=model_plan.resolutions,
                revision=plan.manifest.repos[model.repo].revision,
            )
            models += 1
    return groups, models


def _finalise(
    plan: Plan,
    report: InstallReport,
    dataset_root: Optional[Path],
    model_root: Optional[Path],
    allow_mixed: bool,
) -> None:
    """Write the lock even after an interruption."""
    if dataset_root is not None:
        lock = read_lock(dataset_root) or Lock(kind="dataset")
        note_manifest(dataset_root, lock, plan.manifest, allow_mixed=allow_mixed)
        report.groups_installed, _ = _mark_complete(plan, lock, dataset_root, None)
        write_lock(dataset_root, lock)
    if model_root is not None:
        lock = read_lock(model_root) or Lock(kind="models")
        note_manifest(model_root, lock, plan.manifest, allow_mixed=allow_mixed)
        _, report.models_installed = _mark_complete(plan, lock, None, model_root)
        write_lock(model_root, lock)


def install(
    plan: Plan,
    backends: BackendSet,
    *,
    data_root: Optional[Path] = None,
    models_dir: Optional[Path] = None,
    jobs: int = 4,
    force: bool = False,
    repair: bool = False,
    progress: Optional[Progress] = None,
    cancel: Optional[threading.Event] = None,
    allow_mixed: bool = False,
) -> InstallReport:
    """Download a plan, one repo at a time, into dataset and model roots."""
    if plan.dataset and data_root is None:
        raise HubError("this plan includes dataset files but no --data-root was given")
    if plan.models and models_dir is None:
        raise HubError("this plan includes models but no --models-dir (--dest) was given")

    dataset_root = None if data_root is None else Path(data_root).resolve()
    model_root = None if models_dir is None else Path(models_dir).resolve()
    if dataset_root is not None and model_root is not None and dataset_root == model_root:
        raise HubError(
            "--data-root and --models-dir must be different directories: "
            "they hold different lockfiles (dataset and models)"
        )

    for root, expected in ((dataset_root, "dataset"), (model_root, "models")):
        if root is None:
            continue
        existing = read_lock(root)
        if existing is not None and existing.kind != expected:
            raise HubError(
                f"{root} holds a {existing.kind} install, not {expected}.\n"
                f"  use separate --data-root and --models-dir destinations"
            )

    started = time.time()
    report = InstallReport(data_root=dataset_root, models_dir=model_root)

    for refs, root, label in (
        (plan.dataset, dataset_root, "dataset"),
        (plan.models, model_root, "models"),
    ):
        if root is None or not refs:
            continue
        _check_space(refs, root)
        conflicts = inspect_destination(refs, root).conflicts
        if conflicts and not (force or repair):
            raise HubError(
                f"{root} already has {len(conflicts)} {label} file(s) at the planned paths "
                f"with a different size, e.g. {conflicts[0]}.\n"
                f"  pass --force to overwrite, use --repair to refetch after --verify, "
                f"or use a fresh destination"
            )
        if conflicts:
            _remove_mismatched(refs, root)

    pairs: List[Tuple[Path, FileRef, Backend]] = []
    for refs, root in ((plan.dataset, dataset_root), (plan.models, model_root)):
        if root is None:
            continue
        for ref in refs:
            needed, existing = _needs_download(ref, root)
            if needed:
                pairs.append((root, ref, backends.for_key(ref.repo)))
            else:
                report.skipped_files += 1
                report.skipped_bytes += existing

    cancelled = False
    try:
        cancelled = _download_refs(pairs, jobs=jobs, progress=progress, cancel=cancel)
    finally:
        # Count from disk: a cancelled or failed run must not report a file it never got.
        for root, ref, _ in pairs:
            target = root / ref.path
            if target.is_file() and target.stat().st_size == ref.bytes:
                if root == dataset_root:
                    report.dataset_files += 1
                    report.dataset_bytes += ref.bytes
                else:
                    report.model_files += 1
                    report.model_bytes += ref.bytes
        if dataset_root is not None or model_root is not None:
            _finalise(plan, report, dataset_root, model_root, allow_mixed)
    report.cancelled = cancelled
    report.seconds = time.time() - started
    return report


__all__ = ["InstallReport", "Progress", "human_bytes", "install"]
