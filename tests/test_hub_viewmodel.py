"""The launcher's state: selection rules without a terminal.

Textual redraws whatever this object says. If the rules live here, they are testable on a
machine with no terminal and no optional dependency, which is the same reason the planner
is pure.
"""

from __future__ import annotations

import pytest

from symtrs.hub.errors import SelectionError
from symtrs.hub.viewmodel import LauncherState


def _state(hub_mirror, tmp_path) -> LauncherState:
    return LauncherState(
        manifest=hub_mirror.manifest,
        data_root=tmp_path / "data",
        models_dir=tmp_path / "runs",
    )


def test_group_toggle_and_split_override(hub_mirror, tmp_path):
    state = _state(hub_mirror, tmp_path)
    state.toggle_group("depth/2048")
    assert state.selection().groups == {"depth/2048": ("train", "val", "test")}

    state.set_splits("depth/2048", ["train", "val"])
    assert state.splits_for("depth/2048") == ("train", "val")
    with pytest.raises(SelectionError, match="at least one split"):
        state.set_splits("depth/2048", [])

    state.toggle_group("depth/2048")
    assert state.selection().empty


def test_model_resolution_toggle_collapses_to_all(hub_mirror, tmp_path):
    state = _state(hub_mirror, tmp_path)
    assert not state.model_selected("midas_small", 1024)

    state.toggle_model_resolution("midas_small", 1024)
    assert state.selected_models["midas_small"] == (1024,)
    assert state.model_selected("midas_small", 1024)
    assert not state.model_selected("midas_small", 2048)

    state.toggle_model_resolution("midas_small", 2048)
    assert state.selected_models["midas_small"] == ()  # every published resolution
    assert state.model_selected("midas_small", 1024)

    state.toggle_model_resolution("midas_small", 2048)
    assert state.selected_models["midas_small"] == (1024,)
    state.toggle_model_resolution("midas_small", 1024)
    assert "midas_small" not in state.selected_models


def test_preset_replaces_the_selection(hub_mirror, tmp_path):
    state = _state(hub_mirror, tmp_path)
    state.toggle_group("segmentation/2048")
    state.apply_preset("depth-2048")
    assert state.selected_groups == {"metadata", "depth/2048"}
    assert state.splits_for("depth/2048") == ("train", "val")
    assert set(state.selected_models) == {"midas_small"}
    assert state.selected_models["midas_small"] == ()


def test_state_plans_the_same_way_as_the_flags(hub_mirror, tmp_path):
    state = _state(hub_mirror, tmp_path)
    state.toggle_group("depth/2048")
    state.set_splits("depth/2048", ["train"])
    state.toggle_model_resolution("midas_small", 1024)
    plan = state.plan(hub_mirror.backends.table)
    assert {group.id for group in plan.groups} == {"depth/2048", "metadata"}
    assert [ref.path for ref in plan.models] == ["depth/midas_small/1024/weights/best.pt"]


def test_summary_names_the_selection(hub_mirror, tmp_path):
    state = _state(hub_mirror, tmp_path)
    assert "groups   none" in "\n".join(state.summary_lines())
    state.toggle_group("depth/2048")
    text = "\n".join(state.summary_lines())
    assert "depth/2048" in text and "train,val,test" in text
