"""The interactive launcher: `symtrs` with no subcommand.

A thin view over `LauncherState`. Everything priceable or testable lives in the state and
the planner; this module only wires widgets, keys and one worker thread per long operation.
Textual is imported when the launcher starts, never at package import, because a user who
runs `symtrs train` must not pay for a TUI dependency they did not install.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import List, Optional, Tuple

from symtrs.hub.errors import HubError
from symtrs.hub.fetch import BackendSet
from symtrs.hub.install import InstallReport, Progress, human_bytes, install
from symtrs.hub.plan import Plan, inspect_destination
from symtrs.hub.render import render_install, render_plan
from symtrs.hub.viewmodel import LauncherState


def _textual():
    try:
        import textual
    except ImportError as exc:
        raise HubError(
            "the interactive launcher needs textual, which is not installed.\n"
            "  install it with:  pip install -e '.[launcher]'\n"
            "  or use the flag-driven path:  symtrs get --list   (then --plan / --yes)"
        ) from exc
    return textual


BANNER = r"""
  ███████╗██╗   ██╗███╗   ███╗████████╗██████╗ ███████╗
  ██╔════╝╚██╗ ██╔╝████╗ ████║╚══██╔══╝██╔══██╗██╔════╝
  ███████╗ ╚████╔╝ ██╔████╔██║   ██║   ██████╔╝███████╗
  ╚════██║  ╚██╔╝  ██║╚██╔╝██║   ██║   ██╔══██╗╚════██║
  ███████║   ██║   ██║ ╚═╝ ██║   ██║   ██║  ██║███████║
  ╚══════╝   ╚═╝   ╚═╝     ╚═╝   ╚═╝   ╚═╝  ╚═╝╚══════╝
"""

BANNER_SMALL = r"""
  █▀ █ █ █▄█ ▀█▀ █▀█ █▀
  ▄█ █▄█ █ █  █  █▀▄ ▄█
"""

TAGLINE = "one capture · five tasks · pull only what you need"


def make_app(state: LauncherState, backends: BackendSet, *, jobs: int = 4):
    """Build the Textual app. Raises HubError when textual is absent, with the install command."""
    _textual()
    from textual import work
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical
    from textual.screen import ModalScreen, Screen
    from textual.widgets import (
        Button,
        Checkbox,
        DataTable,
        Footer,
        Header,
        Input,
        Label,
        OptionList,
        ProgressBar,
        RichLog,
        Static,
    )
    from textual.widgets.option_list import Option

    class SplitsScreen(ModalScreen[Optional[Tuple[str, ...]]]):
        BINDINGS = [Binding("escape", "cancel", "cancel")]

        def __init__(self, group_id: str, splits: Tuple[str, ...], selected: Tuple[str, ...]) -> None:
            super().__init__()
            self.group_id = group_id
            self.splits = splits
            self.selected = set(selected)

        def compose(self) -> ComposeResult:
            with Vertical(id="dialog"):
                yield Label(f"splits for {self.group_id}")
                for split in self.splits:
                    yield Checkbox(split, value=split in self.selected, id=f"split-{split}")
                with Horizontal(id="dialog-buttons"):
                    yield Button("Apply", variant="primary", id="apply")
                    yield Button("Cancel", id="cancel")

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "apply":
                chosen = tuple(
                    split
                    for split in self.splits
                    if self.query_one(f"#split-{split}", Checkbox).value
                )
                if not chosen:
                    self.notify("pick at least one split", severity="warning")
                    return
                self.dismiss(chosen)
            else:
                self.dismiss(None)

        def action_cancel(self) -> None:
            self.dismiss(None)

    class PresetScreen(ModalScreen[Optional[str]]):
        BINDINGS = [Binding("escape", "cancel", "cancel")]

        def __init__(self, state: LauncherState) -> None:
            super().__init__()
            self.state = state

        def compose(self) -> ComposeResult:
            with Vertical(id="dialog"):
                yield Label("presets")
                options = OptionList(
                    *[
                        Option(f"{preset.label}  —  {len(preset.groups)} group(s)", id=preset.id)
                        for preset in self.state.manifest.presets.values()
                    ],
                    id="presets",
                )
                yield options

        def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
            self.dismiss(event.option.id)

        def action_cancel(self) -> None:
            self.dismiss(None)

    class ReviewScreen(ModalScreen[bool]):
        BINDINGS = [Binding("escape", "back", "back")]

        def __init__(self, plan: Plan, text: str, conflicts: bool, fits: bool) -> None:
            super().__init__()
            self.plan = plan
            self.text = text
            self.conflicts = conflicts
            self.fits = fits

        def compose(self) -> ComposeResult:
            with Vertical(id="review-dialog"):
                yield Label("review the plan")
                yield Static(self.text, id="review-text")
                with Horizontal(id="dialog-buttons"):
                    yield Button("Download", variant="primary", id="download")
                    yield Button("Back", id="back")

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "download":
                if not self.fits:
                    self.notify("not enough disk space for this plan", severity="error")
                    return
                self.dismiss(True)
            else:
                self.dismiss(False)

        def action_back(self) -> None:
            self.dismiss(False)

    class DownloadScreen(Screen[None]):
        BINDINGS = [Binding("escape", "cancel_or_close", "cancel / close")]

        def __init__(self, plan: Plan, state: LauncherState, backends: BackendSet, jobs: int) -> None:
            super().__init__()
            self.plan = plan
            self.state = state
            self.backends = backends
            self.jobs = jobs
            self.cancel = threading.Event()
            self.report: Optional[InstallReport] = None
            self.error: Optional[Exception] = None
            self.finished = False

        def compose(self) -> ComposeResult:
            yield Header()
            with Vertical(id="download"):
                yield Label("downloading — esc cancels", id="download-title")
                yield ProgressBar(total=1, show_eta=True, id="progress")
                yield Static("starting…", id="status")
                yield RichLog(id="log", highlight=False)
                with Horizontal(id="dialog-buttons"):
                    yield Button("Cancel", id="cancel")
                    yield Button("Close", id="close", disabled=True)
            yield Footer()

        def on_mount(self) -> None:
            self._start()

        def _start(self) -> None:
            self._download(self.plan)

        def _progress(self, done: int, total: int, bytes_done: int, bytes_total: int, path: str) -> None:
            # Called on the worker thread on purpose; marshal back to the UI thread.
            self.app.call_from_thread(self._show_progress, done, total, bytes_done, bytes_total, path)

        def _show_progress(self, done: int, total: int, bytes_done: int, bytes_total: int, path: str) -> None:
            bar = self.query_one("#progress", ProgressBar)
            bar.update(total=max(total, 1), progress=done)
            short = path if len(path) <= 64 else "…" + path[-63:]
            self.query_one("#status", Static).update(
                f"{done}/{total} files · {human_bytes(bytes_done)}/{human_bytes(bytes_total)} · {short}"
            )
            self.query_one("#log", RichLog).write(short)

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "cancel":
                self.cancel.set()
                self.query_one("#status", Static).update("cancelling after in-flight files…")
            elif event.button.id == "close":
                self.app.pop_screen()

        def action_cancel_or_close(self) -> None:
            if self.finished:
                self.app.pop_screen()
            else:
                self.cancel.set()

        @staticmethod
        def _install(plan: Plan, state: LauncherState, backends: BackendSet, jobs: int,
                     cancel: threading.Event, progress: Progress) -> InstallReport:
            return install(
                plan,
                backends,
                data_root=state.data_root if plan.dataset else None,
                models_dir=state.models_dir if plan.models else None,
                jobs=jobs,
                progress=progress,
                cancel=cancel,
            )

        @work(thread=True, exclusive=True)
        def _download(self, plan: Plan) -> None:
            try:
                report = self._install(plan, self.state, self.backends, self.jobs,
                                       self.cancel, self._progress)
            except Exception as exc:  # shown in the screen; an install failure is not a crash
                self.app.call_from_thread(self._finished, None, exc)
            else:
                self.app.call_from_thread(self._finished, report, None)

        def _finished(self, report: Optional[InstallReport], error: Optional[Exception]) -> None:
            self.finished = True
            self.report = report
            self.error = error
            self.query_one("#close", Button).disabled = False
            self.query_one("#cancel", Button).disabled = True
            if error is not None:
                self.query_one("#download-title", Label).update("failed — see the message below")
                self.query_one("#status", Static).update(str(error))
                self.query_one("#log", RichLog).write(str(error))
                self.notify(str(error), severity="error", timeout=12)
                return
            assert report is not None
            self.query_one("#download-title", Label).update("done" if not report.cancelled else "cancelled")
            self.query_one("#status", Static).update(render_install(report))
            self.query_one("#log", RichLog).write(render_install(report))
            self.query_one("#progress", ProgressBar).update(
                total=max(report.downloaded_files + report.skipped_files, 1),
                progress=report.downloaded_files + report.skipped_files,
            )

    class Launcher(App[None]):
        TITLE = "SyMTRS launcher"
        SUB_TITLE = "pull only what you need"
        BINDINGS = [
            Binding("space", "toggle", "toggle"),
            Binding("s", "splits", "splits"),
            Binding("p", "preset", "preset"),
            Binding("a", "select_all", "select all"),
            Binding("c", "clear", "clear"),
            Binding("m", "focus_models", "models"),
            Binding("r", "review", "review"),
            Binding("q", "quit", "quit"),
        ]
        CSS = """
        #banner { height: auto; color: $accent; text-style: bold; }
        #tagline { height: 1; color: $text-muted; padding: 0 2 1 2; }
        #main { height: 1fr; }
        #left { width: 2fr; border-right: solid $surface; }
        #right { width: 1fr; padding: 0 1; }
        #groups { height: 3fr; }
        #models { height: 2fr; }
        #summary { height: 1fr; padding: 1 0; }
        .field { margin-top: 1; }
        #dialog, #review-dialog { width: 90; height: auto; padding: 1 2; background: $surface; border: thick $primary; }
        #dialog-buttons { height: auto; align-horizontal: right; }
        #dialog Button { margin-left: 1; }
        #review-text { height: auto; max-height: 30; }
        #download { padding: 1 2; height: 1fr; }
        #log { height: 1fr; }
        """

        def __init__(self, state: LauncherState, backends: BackendSet, jobs: int) -> None:
            super().__init__()
            self.state = state
            self.backends = backends
            self.jobs = jobs
            self._group_ids: List[str] = []
            self._model_rows: List[Tuple[str, int]] = []
            self._resolving = False

        def compose(self) -> ComposeResult:
            yield Static(BANNER, id="banner")
            yield Static(TAGLINE, id="tagline")
            with Horizontal(id="main"):
                with Vertical(id="left"):
                    yield Label("dataset")
                    yield DataTable(id="groups", cursor_type="row", zebra_stripes=True)
                    yield Label("models")
                    yield DataTable(id="models", cursor_type="row", zebra_stripes=True)
                with Vertical(id="right"):
                    yield Label("selected")
                    yield Static(id="summary")
                    yield Label("data root", classes="field")
                    yield Input(value=str(self.state.data_root), id="data-root")
                    yield Label("models dir", classes="field")
                    yield Input(value=str(self.state.models_dir), id="models-dir")
                    yield Button("Review & download", variant="primary", id="review")
            yield Footer()

        def on_mount(self) -> None:
            groups = self.query_one("#groups", DataTable)
            groups.add_columns("sel", "group", "repo", "task", "variant", "splits", "views", "size")
            models = self.query_one("#models", DataTable)
            models.add_columns("sel", "model", "repo", "task", "res", "protocol", "needs", "size")
            self._refresh_groups()
            self._refresh_models()
            groups.focus()
            self._refresh_summary()
            self._fit_banner()

        def _fit_banner(self) -> None:
            roomy = self.size.height >= 28 and self.size.width >= 66
            self.query_one("#banner", Static).update(BANNER if roomy else BANNER_SMALL)

        def on_resize(self, _event) -> None:
            self._fit_banner()

        # ------------------------------------------------------------ rendering

        def _refresh_groups(self) -> None:
            table = self.query_one("#groups", DataTable)
            row = table.cursor_row
            table.clear()
            self._group_ids = []
            for item in self.state.group_rows():
                mark = "[x]" if item.selected else "[ ]"
                table.add_row(
                    mark,
                    item.id,
                    item.repo,
                    item.task,
                    item.variant,
                    ",".join(item.splits),
                    f"{item.views:,}",
                    human_bytes(item.bytes),
                )
                self._group_ids.append(item.id)
            if self._group_ids:
                table.move_cursor(row=min(max(row, 0), len(self._group_ids) - 1))

        def _refresh_models(self) -> None:
            table = self.query_one("#models", DataTable)
            row = table.cursor_row
            table.clear()
            self._model_rows = []
            for item in self.state.model_rows():
                mark = "[x]" if item.selected else "[ ]"
                table.add_row(
                    mark,
                    item.id,
                    item.repo,
                    item.task,
                    str(item.resolution),
                    item.protocol or "—",
                    item.requires_extra or "—",
                    human_bytes(item.bytes),
                )
                self._model_rows.append((item.id, item.resolution))
            if self._model_rows:
                table.move_cursor(row=min(max(row, 0), len(self._model_rows) - 1))

        def _refresh_summary(self) -> None:
            self.query_one("#summary", Static).update("\n".join(self.state.summary_lines()))

        def _refresh_all(self) -> None:
            self._refresh_groups()
            self._refresh_models()
            self._refresh_summary()

        # ------------------------------------------------------------ actions

        def action_toggle(self) -> None:
            focused = self.focused
            if focused is None:
                return
            if focused.id == "groups" and self._group_ids:
                table = self.query_one("#groups", DataTable)
                self.state.toggle_group(self._group_ids[table.cursor_row])
            elif focused.id == "models" and self._model_rows:
                table = self.query_one("#models", DataTable)
                model_id, resolution = self._model_rows[table.cursor_row]
                self.state.toggle_model_resolution(model_id, resolution)
            self._refresh_all()

        def action_splits(self) -> None:
            if not self._group_ids:
                return
            table = self.query_one("#groups", DataTable)
            group_id = self._group_ids[table.cursor_row]
            group = self.state.manifest.groups[group_id]
            selected = self.state.splits_for(group_id)
            self.push_screen(
                SplitsScreen(group_id, group.splits, selected),
                callback=lambda result: self._splits_done(group_id, result),
            )

        def _splits_done(self, group_id: str, result: Optional[Tuple[str, ...]]) -> None:
            if result is None:
                return
            self.state.set_splits(group_id, result)
            if group_id not in self.state.selected_groups:
                self.state.selected_groups.add(group_id)
            self._refresh_all()

        def action_preset(self) -> None:
            if not self.state.manifest.presets:
                self.notify("this release publishes no presets", severity="warning")
                return
            self.push_screen(
                PresetScreen(self.state),
                callback=self._preset_done,
            )

        def _preset_done(self, preset_id: Optional[str]) -> None:
            if preset_id is None:
                return
            self.state.apply_preset(preset_id)
            self._refresh_all()

        def action_select_all(self) -> None:
            self.state.select_all_groups(True)
            self.state.select_all_models(True)
            self._refresh_all()

        def action_clear(self) -> None:
            self.state.clear()
            self._refresh_all()

        def action_focus_models(self) -> None:
            self.query_one("#models", DataTable).focus()

        def action_review(self) -> None:
            selection = self.state.selection()
            if selection.empty:
                self.notify("select at least one group or model (space toggles)", severity="warning")
                return
            self._refresh_destinations()
            if not self._resolving:
                self._resolving = True
                self.notify("fetching tables to price the plan…")
                self._resolve_worker()

        def _refresh_destinations(self) -> None:
            data = self.query_one("#data-root", Input).value.strip()
            models = self.query_one("#models-dir", Input).value.strip()
            if data:
                self.state.data_root = Path(data).expanduser()
            if models:
                self.state.models_dir = Path(models).expanduser()

        def on_input_changed(self, event: Input.Changed) -> None:
            if event.input.id == "data-root" and event.value.strip():
                self.state.data_root = Path(event.value.strip()).expanduser()
            elif event.input.id == "models-dir" and event.value.strip():
                self.state.models_dir = Path(event.value.strip()).expanduser()

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "review":
                self.action_review()

        @staticmethod
        def _plan_and_text(plan: Plan, state: LauncherState) -> Tuple[str, bool, bool]:
            data_report = inspect_destination(plan.dataset, state.data_root) if plan.dataset else None
            models_report = inspect_destination(plan.models, state.models_dir) if plan.models else None
            text = render_plan(plan, data_report=data_report, models_report=models_report)
            conflicts = bool(
                (data_report is not None and data_report.conflicts)
                or (models_report is not None and models_report.conflicts)
            )
            fits = (data_report is None or data_report.fits) and (
                models_report is None or models_report.fits
            )
            if conflicts:
                text += "\n\nconflicts exist; the download will refuse unless --force is used"
            return text, conflicts, fits

        @work(thread=True, exclusive=True)
        def _resolve_worker(self) -> None:
            state = self.state
            try:
                plan = state.plan(self.backends.table)
                text, conflicts, fits = self._plan_and_text(plan, state)
            except Exception as exc:
                self.app.call_from_thread(self._plan_failed, exc)
            else:
                self.app.call_from_thread(self._plan_ready, plan, text, conflicts, fits)

        def _plan_failed(self, exc: Exception) -> None:
            self._resolving = False
            self.notify(str(exc), severity="error", timeout=12)

        def _plan_ready(self, plan: Plan, text: str, conflicts: bool, fits: bool) -> None:
            self._resolving = False
            self.push_screen(
                ReviewScreen(plan, text, conflicts, fits),
                callback=lambda go: self._review_done(plan, go),
            )

        def _review_done(self, plan: Plan, go: Optional[bool]) -> None:
            if not go:
                return
            self.push_screen(DownloadScreen(plan, self.state, self.backends, self.jobs))

    return Launcher(state, backends, jobs)


def run_launcher(state: LauncherState, backends: BackendSet, *, jobs: int = 4) -> None:
    """Start the TUI."""
    make_app(state, backends, jobs=jobs).run()


__all__ = ["make_app", "run_launcher"]
