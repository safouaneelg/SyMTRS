"""Plain-text rendering for `--list`, `--plan` and `--verify`.

No colour, no cursor movement, no width assumptions beyond a terminal's ordinary line
length. These are the outputs that get pasted into a README, a queue log or a paper's
reproduction appendix, and they are what a non-interactive run prints, so they must be
readable in `less`, in a CI log, and with `TERM=dumb`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from symtrs.hub.install import InstallReport, human_bytes
from symtrs.hub.manifest import Manifest
from symtrs.hub.plan import DestinationReport, Plan
from symtrs.hub.state import Lock, VerifyReport


def _table(headers: Sequence[str], rows: Iterable[Sequence[str]], right: Sequence[int] = ()) -> str:
    rows = [list(row) for row in rows]
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    lines = []
    header = "  ".join(
        cell.rjust(widths[index]) if index in right else cell.ljust(widths[index])
        for index, cell in enumerate(headers)
    )
    lines.append(header)
    for row in rows:
        lines.append(
            "  ".join(
                cell.rjust(widths[index]) if index in right else cell.ljust(widths[index])
                for index, cell in enumerate(row)
            )
        )
    return "\n".join(lines)


def _installed_label(lock: Optional[Lock], group_id: str) -> str:
    if lock is None:
        return "—"
    splits = lock.installed_splits(group_id)
    return "+".join(splits) if splits else "—"


def render_catalog(
    manifest: Manifest,
    *,
    lock: Optional[Lock] = None,
    data_root: Optional[Path] = None,
    models_lock: Optional[Lock] = None,
    models_dir: Optional[Path] = None,
) -> str:
    release = manifest.release
    commit = f"  commit {release.commit[:12]}" if release.commit else ""
    lines = [
        f"release    {release.name}@{release.revision}{commit}",
        f"           {release.views:,} views · {release.files:,} files · {human_bytes(release.bytes)}"
        f" · {release.layout} layout",
    ]
    if data_root is not None:
        installed = "not installed" if lock is None else f"release {lock.revision_label}"
        lines.append(f"dataset    {data_root}  ({installed})")
    if models_dir is not None:
        installed = "not installed" if models_lock is None else f"release {models_lock.revision_label}"
        lines.append(f"models     {models_dir}  ({installed})")
    lines.append("")

    repo_rows: List[Sequence[str]] = []
    for key, repo in manifest.repos.items():
        tags = []
        if key == "metadata":
            tags.append("catalog")
        if key in manifest.default_groups:
            tags.append("auto")
        repo_rows.append((key, repo.url, repo.type, ",".join(tags) or "—"))
    lines.append(_table(("repo", "project", "type", "role"), repo_rows))
    lines.append("")

    group_rows: List[Sequence[str]] = []
    for group in manifest.groups.values():
        group_rows.append(
            (
                group.id,
                group.repo,
                group.task,
                group.variant,
                ",".join(group.splits),
                f"{group.count_for(group.splits):,}",
                human_bytes(group.size_for(group.splits)),
                _installed_label(lock, group.id),
            )
        )
    lines.append(_table(
        ("group", "repo", "task", "variant", "splits", "views", "size", "installed"),
        group_rows,
        right=(5, 6),
    ))

    if manifest.models:
        model_rows: List[Sequence[str]] = []
        for model in manifest.models.values():
            badge = f"[{model.requires_extra}]" if model.requires_extra else "—"
            protocol = f"{model.protocol}/{model.output_space}" if model.protocol else ""
            model_rows.append(
                (
                    model.id,
                    model.repo,
                    model.task,
                    protocol,
                    ",".join(str(res) for res in model.resolutions),
                    human_bytes(model.size_for()),
                    badge,
                )
            )
        lines.append("")
        lines.append(_table(
            ("model", "repo", "task", "protocol", "resolutions", "size", "needs"),
            model_rows,
            right=(4, 5),
        ))

    if manifest.presets:
        lines.append("")
        lines.append("presets: " + ", ".join(manifest.presets))
    lines.append("")
    lines.append("choose with:  symtrs get --task depth --resolutions 2048 --splits train,val --yes")
    lines.append("              symtrs get --preset <id> --yes   ·   symtrs get   (interactive)")
    return "\n".join(lines)


def _destination_lines(report: DestinationReport, label: str) -> List[str]:
    lines = [
        f"  {label} files      "
        f"{len(report.present) + len(report.missing) + len(report.conflicts):,}"
        f"  present {len(report.present):,}  missing {len(report.missing):,}"
        f"  conflicts {len(report.conflicts):,}",
        f"  {label} to fetch   {human_bytes(report.missing_bytes)}"
        + (f"  (needs {human_bytes(report.required_bytes)} with headroom)" if report.missing_bytes else ""),
    ]
    if report.free_bytes is not None:
        lines.append(
            f"  {label} free       {human_bytes(report.free_bytes)}"
            + ("" if report.fits else "  ← NOT ENOUGH")
        )
    if report.conflicts:
        lines.append(f"  {label} conflict   {report.conflicts[0]}")
        if len(report.conflicts) > 1:
            lines.append(f"  {' ' * len(label)}            and {len(report.conflicts) - 1} more")
    return lines


def render_plan(
    plan: Plan,
    *,
    data_report: Optional[DestinationReport] = None,
    models_report: Optional[DestinationReport] = None,
    sample: int = 3,
) -> str:
    lines = [
        f"release    {plan.manifest.release.name}@{plan.manifest.release.revision}"
        + (f"  commit {plan.manifest.release.commit[:12]}" if plan.manifest.release.commit else ""),
    ]
    if plan.groups:
        lines.append(f"dataset    {len(plan.groups)} group(s)")
        for group in plan.groups:
            lines.append(
                f"  {group.id:<28} {plan.manifest.groups[group.id].repo:<16} {','.join(group.splits):<16} "
                f"{group.files:>8,} files  {human_bytes(group.bytes):>10}"
            )
        lines.append(
            f"  total      {len(plan.dataset):,} files  {human_bytes(plan.dataset_bytes)}"
            + (f"   ({human_bytes(plan.saved_bytes)} shared, counted once)" if plan.saved_bytes else "")
        )
        if data_report is not None:
            lines.extend(_destination_lines(data_report, "dataset   "))
        for ref in plan.dataset[:sample]:
            lines.append(f"  e.g.       {ref.path}")
        if len(plan.dataset) > sample:
            lines.append(f"             and {len(plan.dataset) - sample:,} more")
    if plan.model_plans:
        lines.append(f"models     {len(plan.model_plans)} model(s)")
        for entry in plan.model_plans:
            resolutions = ",".join(str(res) for res in entry.resolutions) or "all"
            lines.append(
                f"  {entry.id:<28} {plan.manifest.models[entry.id].repo:<16} {resolutions:<16} "
                f"{entry.files:>8,} files  {human_bytes(entry.bytes):>10}"
            )
        if models_report is not None:
            lines.extend(_destination_lines(models_report, "models    "))
    if plan.used_repos:
        lines.append("repos      " + ", ".join(plan.used_repos))
    lines.append(f"total      {human_bytes(plan.total_bytes)}")
    return "\n".join(lines)


def render_verify(report: VerifyReport, *, label: str = "") -> str:
    title = label or str(report.root)
    if report.ok:
        return (
            f"{title}: {report.groups_checked} group(s), {report.models_checked} model(s), "
            f"{report.files_checked:,} files, {human_bytes(report.bytes_checked)} — OK"
        )
    lines = [
        f"{title}: {len(report.issues)} problem(s) in {report.files_checked:,} checked files",
    ]
    for issue in report.issues[:40]:
        lines.append("  " + issue.describe())
    if len(report.issues) > 40:
        lines.append(f"  … and {len(report.issues) - 40} more")
    lines.append("repair with:  symtrs get --verify --repair")
    return "\n".join(lines)


def render_install(report: InstallReport) -> str:
    if report.cancelled:
        head = "cancelled"
    elif report.downloaded_files:
        head = "done"
    else:
        head = "already complete"
    lines = [
        f"{head}: {report.downloaded_files:,} file(s), {human_bytes(report.downloaded_bytes)} "
        f"in {report.seconds:.1f}s"
    ]
    if report.skipped_files:
        lines.append(
            f"  skipped  {report.skipped_files:,} file(s), "
            f"{human_bytes(report.skipped_bytes)} already present"
        )
    if report.data_root is not None:
        lines.append(f"  dataset  {report.data_root}  ({report.groups_installed} group-split(s) recorded)")
    if report.models_dir is not None:
        lines.append(f"  models   {report.models_dir}  ({report.models_installed} model(s) recorded)")
    lines.append("  verify with:  symtrs get --verify")
    return "\n".join(lines)


def catalog_as_dict(manifest: Manifest, lock: Optional[Lock] = None) -> dict:
    """Machine-readable `--list --json`."""
    return {
        "release": {
            "name": manifest.release.name,
            "revision": manifest.release.revision,
            "commit": manifest.release.commit,
            "views": manifest.release.views,
            "files": manifest.release.files,
            "bytes": manifest.release.bytes,
            "digest": manifest.digest,
        },
        "repos": [
            {"key": repo.key, "repo": repo.repo, "url": repo.url, "type": repo.type,
             "revision": repo.revision, "default": repo.key in manifest.default_groups}
            for repo in manifest.repos.values()
        ],
        "groups": [
            {
                "id": group.id,
                "repo": group.repo,
                "task": group.task,
                "variant": group.variant,
                "resolution": group.resolution,
                "native": group.native,
                "modalities": list(group.modalities),
                "splits": list(group.splits),
                "views": dict(group.views),
                "bytes": dict(group.bytes),
                "requires": list(group.requires),
                "installed_splits": list(lock.installed_splits(group.id)) if lock else [],
            }
            for group in manifest.groups.values()
        ],
        "models": [
            {
                "id": model.id,
                "repo": model.repo,
                "task": model.task,
                "registry_key": model.registry_key,
                "protocol": model.protocol,
                "output_space": model.output_space,
                "requires_extra": model.requires_extra,
                "resolutions": list(model.resolutions),
                "bytes": model.size_for(),
            }
            for model in manifest.models.values()
        ],
        "presets": [
            {
                "id": preset.id,
                "label": preset.label,
                "groups": list(preset.groups),
                "splits": list(preset.splits),
                "models": list(preset.models),
            }
            for preset in manifest.presets.values()
        ],
    }


__all__ = ["catalog_as_dict", "render_catalog", "render_install", "render_plan", "render_verify"]

