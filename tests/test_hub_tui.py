"""A smoke test for the Textual app.

Skipped where `textual` is not installed, because the launcher is an optional extra. The
state rules are covered without it in `test_hub_viewmodel.py`; this test is only the wiring
between the terminal and that state.
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("textual")

from symtrs.hub.tui import make_app
from symtrs.hub.viewmodel import LauncherState


def test_the_launcher_toggles_groups_and_models(hub_mirror, tmp_path):
    state = LauncherState(
        manifest=hub_mirror.manifest,
        data_root=tmp_path / "data",
        models_dir=tmp_path / "runs",
    )
    app = make_app(state, hub_mirror.backends)

    async def drive() -> None:
        async with app.run_test() as pilot:
            await pilot.pause()
            assert state.selected_groups == set()
            # The first row is the metadata group; space toggles the focused table's row.
            await pilot.press("space")
            assert state.selected_groups == {"metadata"}

            await pilot.press("a")
            assert state.selected_groups == set(state.manifest.groups)
            assert set(state.selected_models) == set(state.manifest.models)

            await pilot.press("c")
            assert state.selected_groups == set()
            assert state.selected_models == {}

            await pilot.press("m")
            await pilot.press("space")
            assert state.selected_models, "focusing models and toggling a row selects one"

    asyncio.run(drive())


def test_review_prices_the_selection_from_the_tables(hub_mirror, tmp_path):
    state = LauncherState(
        manifest=hub_mirror.manifest,
        data_root=tmp_path / "data",
        models_dir=tmp_path / "runs",
    )
    app = make_app(state, hub_mirror.backends)

    async def drive() -> None:
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("space")       # the metadata group
            await pilot.press("r")           # fetch tables and open the review screen
            await pilot.pause(0.5)
            assert app.screen.__class__.__name__ == "ReviewScreen"
            await pilot.press("escape")
            await pilot.pause()
            assert app.screen.__class__.__name__ != "ReviewScreen"

    asyncio.run(drive())


def test_download_from_the_review_screen(hub_mirror, tmp_path):
    state = LauncherState(
        manifest=hub_mirror.manifest,
        data_root=tmp_path / "data",
        models_dir=tmp_path / "runs",
    )
    app = make_app(state, hub_mirror.backends, jobs=1)

    async def drive() -> None:
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("space")       # metadata
            await pilot.press("a")           # then every group and model
            await pilot.press("r")
            await pilot.pause(0.5)
            await pilot.press("enter")
            await pilot.pause(1.0)
            assert (state.data_root / "train").is_dir()
            assert (state.data_root / "train" / "rgb_day").is_dir()
            assert (state.data_root / ".symtrs" / "install.json").is_file()
            assert (state.models_dir / "depth" / "midas_small" / "1024" / "weights" / "best.pt").is_file()

    asyncio.run(drive())


def test_pressing_p_applies_a_preset(hub_mirror, tmp_path):
    state = LauncherState(
        manifest=hub_mirror.manifest,
        data_root=tmp_path / "data",
        models_dir=tmp_path / "runs",
    )
    app = make_app(state, hub_mirror.backends)

    async def drive() -> None:
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("p")
            await pilot.pause()
            assert app.screen.__class__.__name__ == "PresetScreen"
            await pilot.press("enter")       # the first preset
            await pilot.pause()
            assert state.selected_groups == {"metadata", "depth/2048"}
            assert state.splits_for("depth/2048") == ("train", "val")

    asyncio.run(drive())
