"""The interactive launcher's state, independent of the terminal.

The TUI is a view over this object. Keeping selection, split overrides and model
resolution sets here means the rules can be tested without a terminal, and it means the
interactive path and `--task/--splits/--models` share one meaning: an empty resolution
tuple is every published resolution, a group with no split override uses all its splits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

from symtrs.hub.errors import SelectionError
from symtrs.hub.manifest import Manifest
from symtrs.hub.plan import Plan, Selection, TableSource, resolve


@dataclass(frozen=True)
class GroupRow:
    id: str
    repo: str
    task: str
    variant: str
    label: str
    selected: bool
    splits: Tuple[str, ...]
    views: int
    bytes: int
    native: bool


@dataclass(frozen=True)
class ModelRow:
    id: str
    repo: str
    task: str
    resolution: int
    selected: bool
    protocol: str
    requires_extra: str
    bytes: int


@dataclass
class LauncherState:
    manifest: Manifest
    data_root: Path
    models_dir: Path
    selected_groups: Set[str] = field(default_factory=set)
    #: group id -> splits; absent means every split the group publishes
    splits: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    #: model id -> resolutions; an empty tuple means every published resolution
    selected_models: Dict[str, Tuple[int, ...]] = field(default_factory=dict)

    # ---------------------------------------------------------------- groups

    def splits_for(self, group_id: str) -> Tuple[str, ...]:
        return self.splits.get(group_id, self.manifest.groups[group_id].splits)

    def toggle_group(self, group_id: str) -> None:
        if group_id in self.selected_groups:
            self.selected_groups.discard(group_id)
        else:
            self.selected_groups.add(group_id)

    def set_splits(self, group_id: str, splits: Iterable[str]) -> None:
        group = self.manifest.groups[group_id]
        wanted = tuple(split for split in group.splits if split in set(splits))
        if not wanted:
            raise SelectionError(
                f"group {group_id!r} needs at least one split; it has {', '.join(group.splits)}"
            )
        self.splits[group_id] = wanted

    def select_all_groups(self, selected: bool = True) -> None:
        if selected:
            self.selected_groups.update(self.manifest.groups)
        else:
            self.selected_groups.clear()

    def group_rows(self) -> List[GroupRow]:
        rows = []
        for group in self.manifest.groups.values():
            splits = self.splits_for(group.id)
            rows.append(
                GroupRow(
                    id=group.id,
                    repo=group.repo,
                    task=group.task,
                    variant=group.variant,
                    label=group.label,
                    selected=group.id in self.selected_groups,
                    splits=splits,
                    views=group.count_for(splits),
                    bytes=group.size_for(splits),
                    native=group.native,
                )
            )
        return rows

    # ---------------------------------------------------------------- models

    def model_selected(self, model_id: str, resolution: int) -> bool:
        if model_id not in self.selected_models:
            return False
        chosen = self.selected_models[model_id]
        return not chosen or resolution in chosen

    def toggle_model_resolution(self, model_id: str, resolution: int) -> None:
        model = self.manifest.models[model_id]
        all_resolutions = set(model.resolutions)
        if model_id not in self.selected_models:
            chosen = {resolution}
        else:
            current = self.selected_models[model_id]
            chosen = set(current) if current else set(all_resolutions)
            if resolution in chosen:
                chosen.discard(resolution)
            else:
                chosen.add(resolution)
        if not chosen:
            self.selected_models.pop(model_id, None)
        elif chosen == all_resolutions:
            # `()` means every resolution; keeping the set would silently pin today's list
            # if the release later adds a level.
            self.selected_models[model_id] = ()
        else:
            self.selected_models[model_id] = tuple(sorted(chosen, reverse=True))

    def select_all_models(self, selected: bool = True) -> None:
        if selected:
            for model_id in self.manifest.models:
                self.selected_models[model_id] = ()
        else:
            self.selected_models.clear()

    def model_rows(self) -> List[ModelRow]:
        rows = []
        for model in self.manifest.models.values():
            for model_file in sorted(model.files, key=lambda f: f.resolution, reverse=True):
                rows.append(
                    ModelRow(
                        id=model.id,
                        repo=model.repo,
                        task=model.task,
                        resolution=model_file.resolution,
                        selected=self.model_selected(model.id, model_file.resolution),
                        protocol=f"{model.protocol}/{model.output_space}" if model.protocol else "",
                        requires_extra=model.requires_extra,
                        bytes=model_file.bytes,
                    )
                )
        return rows

    # ---------------------------------------------------------------- presets and plans

    def apply_preset(self, preset_id: str) -> None:
        preset = self.manifest.presets.get(preset_id)
        if preset is None:
            raise SelectionError(
                f"no preset {preset_id!r}; published presets: "
                f"{', '.join(self.manifest.presets) or 'none'}"
            )
        self.selected_groups = set(preset.groups)
        self.splits = {group_id: tuple(preset.splits) for group_id in preset.groups}
        self.selected_models = {model_id: () for model_id in preset.models}

    def clear(self) -> None:
        self.selected_groups.clear()
        self.splits.clear()
        self.selected_models.clear()

    def selection(self) -> Selection:
        selection = Selection()
        for group_id in self.selected_groups:
            selection.add_group(group_id, self.splits_for(group_id))
        for model_id, resolutions in self.selected_models.items():
            selection.add_model(model_id, resolutions)
        return selection

    def plan(self, table_source: TableSource) -> Plan:
        return resolve(self.manifest, self.selection(), table_source)

    def summary_lines(self) -> List[str]:
        """The right-hand panel: what is selected, before any table is fetched."""
        lines: List[str] = []
        if self.selected_groups:
            groups_bytes = sum(
                self.manifest.groups[group_id].size_for(self.splits_for(group_id))
                for group_id in sorted(self.selected_groups)
            )
            lines.append(f"groups   {len(self.selected_groups)}")
            for group_id in sorted(self.selected_groups):
                group = self.manifest.groups[group_id]
                lines.append(
                    f"  {group_id}  {group.repo}  ({','.join(self.splits_for(group_id))})"
                )
            lines.append(f"  ≈ {groups_bytes / 1024 ** 3:.1f} GB (before dedupe)")
            auto = [g for g in self.manifest.default_groups if g not in self.selected_groups]
            if auto:
                lines.append("auto     " + ", ".join(auto))
        else:
            lines.append("groups   none")
        if self.selected_models:
            lines.append(f"models   {len(self.selected_models)}")
            for model_id in sorted(self.selected_models):
                chosen = self.selected_models[model_id]
                label = "all resolutions" if not chosen else ",".join(map(str, chosen))
                lines.append(f"  {model_id}  ({label})")
        else:
            lines.append("models   none")
        lines.append("")
        lines.append("the plan is exact once the tables are fetched")
        return lines


__all__ = ["GroupRow", "LauncherState", "ModelRow"]
