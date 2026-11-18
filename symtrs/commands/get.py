"""`symtrs get` - browse and download the release, and the entry point behind bare `symtrs`.

Three ways in, one selection model:

    symtrs get                         interactive launcher (also bare `symtrs`)
    symtrs get --list                  what is published, and what this root already has
    symtrs get --task depth --resolutions 2048 --splits train,val --yes

`--plan` resolves a selection to the exact files and bytes without downloading; `--yes`
executes it. Nothing here interprets a model or a metric - it moves the published bytes to
the paths the rest of the package already reads.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from symtrs.data.splits import SPLITS

from symtrs.hub.errors import HubError, SelectionError
from symtrs.hub.fetch import (
    BackendSet,
    HuggingFaceBackend,
    LocalBackend,
    huggingface_backend_factory,
    mirror_backend_factory,
)
from symtrs.hub.install import install
from symtrs.hub.manifest import Manifest, load_manifest_bytes, load_manifest_file
from symtrs.hub.plan import (
    Selection,
    inspect_destination,
    parse_model_spec,
    plan_as_dict,
    resolve,
    select_tasks,
)
from symtrs.hub.render import (
    catalog_as_dict,
    render_catalog,
    render_install,
    render_plan,
    render_verify,
)
from symtrs.hub.repos import CATALOG_REPO_KEY, DEFAULT_CATALOG_REPO
from symtrs.hub.state import (
    Lock,
    read_lock,
    repair_paths,
    verify_dataset,
    verify_models,
)
from symtrs.hub.viewmodel import LauncherState


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "get",
        help="browse and download dataset groups and trained models",
        description=(
            "Download only the SyMTRS groups and models you need. With no selection this "
            "opens the interactive launcher (the same thing bare `symtrs` does)."
        ),
    )
    mode = parser.add_argument_group("what to do")
    mode.add_argument("--list", action="store_true", help="list the published release and what is installed")
    mode.add_argument("--plan", action="store_true", help="print the exact files and bytes, download nothing")
    mode.add_argument("--json", action="store_true", help="machine-readable --list/--plan output")
    mode.add_argument("--verify", action="store_true", help="check the files already installed")
    mode.add_argument("--repair", action="store_true", help="with --verify: refetch missing or damaged files")
    mode.add_argument("--yes", "-y", action="store_true", help="download without asking (required without a TTY)")
    mode.add_argument("--no-tui", action="store_true", help="never open the interactive launcher")

    choose = parser.add_argument_group("selection")
    choose.add_argument("--all", action="store_true", help="every group, every split, every model")
    choose.add_argument("--preset", action="append", default=[], metavar="ID",
                        help="a named selection published in the manifest")
    choose.add_argument("--task", action="append", default=[], metavar="TASK",
                        help="every group of a task: depth, segmentation, domain_adapt, ...")
    choose.add_argument("--group", action="append", default=[], metavar="ID",
                        help="a group id from --list, e.g. depth/2048")
    choose.add_argument("--resolutions", default=None, metavar="2048,1024",
                        help="filter --task by resolution")
    choose.add_argument("--variants", default=None, metavar="x2,day-night",
                        help="filter --task by release variant (super-resolution scales, translation directions)")
    choose.add_argument("--splits", default=None, metavar="train,val",
                        help="splits to take from each selected group (default: all it publishes)")
    choose.add_argument("--models", action="append", default=[], metavar="KEY[@RES]",
                        help="a model id, optionally @resolution; repeat or comma-separate ids")

    where = parser.add_argument_group("where")
    where.add_argument("--data-root", dest="data_root", default=None,
                       help="dataset destination (default: $SYMTRS_DATA or ./SyMTRS)")
    where.add_argument("--models-dir", "--dest", dest="models_dir", default=None,
                       help="model destination (default: $SYMTRS_MODELS or ./runs)")

    source = parser.add_argument_group("source")
    source.add_argument("--mirror", default=None, metavar="DIR",
                        help="read the release from a local directory instead of the Hub; "
                             "it holds one subdirectory per repo key, with the catalog in "
                             "DIR/metadata/symtrs-hub.json")
    source.add_argument("--manifest", default=None, metavar="PATH",
                        help="read the manifest from a local file, then fetch files from its repo")
    source.add_argument("--repo", default=None,
                        help=f"catalog repo (default: {DEFAULT_CATALOG_REPO})")
    source.add_argument("--revision", default=None,
                        help="catalog revision (default: main; the manifest pins the data repos)")
    source.add_argument("--offline", action="store_true", help="forbid the Hub; requires --mirror")
    source.add_argument("--jobs", type=int, default=4, help="parallel downloads (default: 4)")
    source.add_argument("--force", action="store_true", help="overwrite files whose size disagrees")
    source.add_argument("--allow-mixed-revision", dest="allow_mixed_revision", action="store_true",
                        help="allow two releases in one destination (records the mixture)")

    parser.set_defaults(handler=_run)


# --------------------------------------------------------------------- defaults


def _env_path(name: str, fallback: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else fallback


def data_root_for(args: argparse.Namespace) -> Path:
    return Path(args.data_root).expanduser() if args.data_root else _env_path("SYMTRS_DATA", Path.cwd() / "SyMTRS")


def models_dir_for(args: argparse.Namespace) -> Path:
    return Path(args.models_dir).expanduser() if args.models_dir else _env_path("SYMTRS_MODELS", Path.cwd() / "runs")


def _interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


# --------------------------------------------------------------------- source


def open_catalog_backend(args: argparse.Namespace):
    """The backend that carries `symtrs-hub.json` (the metadata project)."""
    if args.mirror:
        return LocalBackend(
            Path(args.mirror).expanduser() / CATALOG_REPO_KEY,
            repo=args.repo or "local mirror",
            revision=args.revision or "local",
        )
    if args.offline:
        raise HubError(
            "--offline forbids the Hub, so the release has to be on disk.\n"
            "  pass --mirror <dir> (one subdirectory per repo key, catalog in metadata/)"
        )
    return HuggingFaceBackend(
        repo=args.repo or DEFAULT_CATALOG_REPO,
        repo_type="dataset",
        revision=args.revision or "main",
    )


def load_manifest_for(args: argparse.Namespace, catalog_backend) -> Manifest:
    if args.manifest:
        return load_manifest_file(Path(args.manifest).expanduser())
    return load_manifest_bytes(catalog_backend.read_manifest_bytes(), source=catalog_backend.describe())


def build_backends(manifest: Manifest, args: argparse.Namespace) -> BackendSet:
    """One lazily-opened backend per repo the manifest declares.

    The catalog repo is only read for the manifest; the data comes from the view store,
    the task verticals and the model verticals named in `manifest.repos`. A mirror is the
    same layout under a directory; the Hub uses each repo's declared revision.
    """
    if args.mirror:
        factory = mirror_backend_factory(Path(args.mirror).expanduser())
    else:
        factory = huggingface_backend_factory()
    return BackendSet(manifest.repos, factory)


# --------------------------------------------------------------------- selection


def _csv(value: Optional[str]) -> List[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def _splits(args: argparse.Namespace) -> List[str]:
    splits = _csv(args.splits)
    unknown = [split for split in splits if split not in SPLITS]
    if unknown:
        raise SelectionError(
            f"unknown split(s) {', '.join(map(repr, unknown))}; this package reads {', '.join(SPLITS)}"
        )
    return splits


def _resolutions(args: argparse.Namespace) -> List[int]:
    try:
        return [int(item) for item in _csv(args.resolutions)]
    except ValueError:
        raise SelectionError(
            f"--resolutions expects comma-separated integers, got {args.resolutions!r}"
        ) from None


def _merge(into: Selection, other: Selection) -> None:
    for group_id, splits in other.groups.items():
        into.add_group(group_id, splits)
    for model_id, resolutions in other.models.items():
        into.add_model(model_id, resolutions)


def _add_group(manifest: Manifest, selection: Selection, group_id: str, splits: Sequence[str]) -> None:
    group = manifest.groups.get(group_id)
    if group is None:
        import difflib

        close = difflib.get_close_matches(group_id, list(manifest.groups), n=3, cutoff=0.6)
        hint = f" Did you mean {', '.join(map(repr, close))}?" if close else ""
        raise SelectionError(
            f"no published group {group_id!r}.{hint} Run `symtrs get --list`."
        )
    chosen = splits or group.splits
    missing = sorted(set(chosen) - set(group.splits))
    if missing:
        raise SelectionError(
            f"group {group_id!r} does not publish split(s) {', '.join(missing)}; "
            f"it has {', '.join(group.splits)}"
        )
    selection.add_group(group_id, chosen)


def build_selection(manifest: Manifest, args: argparse.Namespace) -> Selection:
    selection = Selection()
    if args.all:
        for group in manifest.groups.values():
            _add_group(manifest, selection, group.id, _splits(args))
        for model in manifest.models:
            selection.add_model(model)
    for preset_id in args.preset:
        preset = manifest.presets.get(preset_id)
        if preset is None:
            raise SelectionError(
                f"no preset {preset_id!r}; published presets: "
                f"{', '.join(manifest.presets) or 'none'}"
            )
        for group_id in preset.groups:
            _add_group(manifest, selection, group_id, preset.splits)
        for model_id in preset.models:
            selection.add_model(model_id)
    if args.task:
        _merge(
            selection,
            select_tasks(
                manifest,
                args.task,
                splits=_splits(args),
                resolutions=_resolutions(args),
                variants=_csv(args.variants),
            ),
        )
    for group_id in args.group:
        _add_group(manifest, selection, group_id, _splits(args))
    model_items = [item for value in args.models for item in _csv(value)]
    for model_id, resolutions in parse_model_spec(manifest, model_items).items():
        selection.add_model(model_id, resolutions)
    return selection


def _selection_from_locks(manifest: Manifest, data_lock: Optional[Lock], models_lock: Optional[Lock]) -> Selection:
    selection = Selection()
    for group_id, splits in (data_lock.groups if data_lock else {}).items():
        if group_id not in manifest.groups:
            print(f"[symtrs] skipping {group_id}: it is not in this release")
            continue
        selection.add_group(group_id, list(splits))
    for model_id, entry in (models_lock.models if models_lock else {}).items():
        if model_id not in manifest.models:
            print(f"[symtrs] skipping model {model_id}: it is not in this release")
            continue
        selection.add_model(model_id, entry.get("resolutions", []))
    return selection


# --------------------------------------------------------------------- modes


def _list(args, manifest, data_root, models_dir, data_lock, models_lock) -> int:
    if args.json:
        out: Dict[str, Any] = {"dataset": catalog_as_dict(manifest, data_lock)}
        out["models_installed"] = list((models_lock.models if models_lock else {}))
        out["data_root"] = str(data_root)
        out["models_dir"] = str(models_dir)
        print(json.dumps(out, indent=2))
        return 0
    print(render_catalog(
        manifest,
        lock=data_lock,
        data_root=data_root,
        models_lock=models_lock,
        models_dir=models_dir,
    ))
    return 0


def _verify(args, manifest, data_root, models_dir, backends: BackendSet) -> int:
    data_lock = read_lock(data_root) if data_root else None
    models_lock = read_lock(models_dir) if models_dir else None
    if data_lock is not None and data_lock.kind != "dataset":
        raise HubError(
            f"{data_root} holds a {data_lock.kind} install; --data-root must point at a "
            f"dataset destination"
        )
    if models_lock is not None and models_lock.kind != "models":
        raise HubError(
            f"{models_dir} holds a {models_lock.kind} install; --models-dir must point at a "
            f"model destination"
        )
    if data_lock is None and models_lock is None:
        raise HubError(
            f"nothing is installed at {data_root} or {models_dir}.\n"
            f"  install something first:  symtrs get --list   then --preset <id> --yes"
        )
    reports = []
    if data_lock is not None:
        reports.append(verify_dataset(data_root, data_lock))
    if models_lock is not None:
        reports.append(verify_models(models_dir, models_lock, manifest))
    for report, label in zip(reports, (str(data_root), str(models_dir))):
        print(render_verify(report, label=label))
    ok = all(report.ok for report in reports)
    if not args.repair:
        return 0 if ok else 1

    # Repair: delete the files that are the wrong size/hash (a missing file needs no
    # delete), then resolve the recorded selection again and fetch what is absent.
    to_remove = [path for report in reports for path in repair_paths(report)]
    for path in to_remove:
        if path.is_dir():
            raise HubError(f"cannot repair {path}: it is a directory")
        path.unlink(missing_ok=True)
    selection = _selection_from_locks(manifest, data_lock, models_lock)
    if selection.empty:
        print("[symtrs] nothing in the lockfiles belongs to this release; nothing to repair")
        return 1
    print(f"[symtrs] repairing {len(to_remove)} damaged file(s)")
    plan = resolve(manifest, selection, backends.table)
    report = install(
        plan,
        backends,
        data_root=data_root if plan.dataset else None,
        models_dir=models_dir if plan.models else None,
        jobs=args.jobs,
        force=True,
        allow_mixed=args.allow_mixed_revision,
    )
    print(render_install(report))
    # Re-check rather than assuming the fetch fixed it.
    final = []
    if data_lock is not None:
        final.append(verify_dataset(data_root, read_lock(data_root) or data_lock))
    if models_lock is not None:
        final.append(verify_models(models_dir, read_lock(models_dir) or models_lock, manifest))
    for report_, label in zip(final, (str(data_root), str(models_dir))):
        print(render_verify(report_, label=label))
    return 0 if all(item.ok for item in final) else 1


def _run(args: argparse.Namespace) -> int:
    if args.repair and not args.verify:
        raise HubError("--repair is used with --verify: symtrs get --verify --repair")

    catalog_backend = open_catalog_backend(args)
    manifest = load_manifest_for(args, catalog_backend)
    data_root = data_root_for(args)
    models_dir = models_dir_for(args)
    data_lock = read_lock(data_root) if data_root else None
    models_lock = read_lock(models_dir) if models_dir else None
    backends = build_backends(manifest, args)

    if args.verify:
        return _verify(args, manifest, data_root, models_dir, backends)
    if args.list:
        return _list(args, manifest, data_root, models_dir, data_lock, models_lock)

    selection = build_selection(manifest, args)
    if selection.empty:
        if not args.no_tui and _interactive():
            from symtrs.hub.tui import run_launcher

            state = LauncherState(
                manifest=manifest,
                data_root=data_root,
                models_dir=models_dir,
                    )
            run_launcher(state, backends, jobs=args.jobs)
            return 0
        # Not a terminal: a catalogue is the useful answer, not a hung prompt.
        print(render_catalog(manifest, lock=data_lock, data_root=data_root,
                             models_lock=models_lock, models_dir=models_dir))
        print()
        print("[symtrs] no selection: add --task/--group/--models or --preset, then --yes")
        return 0

    plan = resolve(manifest, selection, backends.table)
    data_report = inspect_destination(plan.dataset, data_root) if plan.dataset else None
    models_report = inspect_destination(plan.models, models_dir) if plan.models else None
    text = render_plan(plan, data_report=data_report, models_report=models_report)

    if args.json:
        print(json.dumps(
            plan_as_dict(
                plan,
                data_root if plan.dataset else None,
                models_dir if plan.models else None,
            ),
            indent=2,
        ))
        return 0
    if args.plan:
        print(text)
        return 0

    print(text)
    if not args.yes:
        if not _interactive():
            raise HubError(
                "refusing to download without --yes in a non-interactive session.\n"
                "  add --yes to the command, or run `symtrs get` in a terminal"
            )
        answer = input("\ndownload this plan? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("[symtrs] nothing downloaded")
            return 1

    report = install(
        plan,
        backends,
        data_root=data_root if plan.dataset else None,
        models_dir=models_dir if plan.models else None,
        jobs=args.jobs,
        force=args.force,
        allow_mixed=args.allow_mixed_revision,
    )
    print(render_install(report))
    return 0 if not report.cancelled else 130


__all__ = ["add_parser", "build_backends", "build_selection", "data_root_for", "models_dir_for",
           "open_catalog_backend"]
