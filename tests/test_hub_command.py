"""`symtrs get` from the command line, and bare `symtrs` in a non-terminal.

The command is the scriptable path: no TUI, no prompt, exit codes that a queue can read.
The tests run it through `symtrs.cli.main` so the argument parser and the one-line error
reporting are covered too.
"""

from __future__ import annotations

import json

from symtrs.cli import main


def _get(hub_mirror, *args: str) -> list:
    return ["get", "--mirror", str(hub_mirror.root), *args]


def test_list_shows_the_release_and_the_sizes(hub_mirror, capsys):
    assert main(_get(hub_mirror, "--list")) == 0
    out = capsys.readouterr().out
    assert "release" in out
    assert "depth/2048" in out
    assert "midas_small" in out
    assert "presets:" in out


def test_list_json_is_parseable(hub_mirror, capsys):
    assert main(_get(hub_mirror, "--list", "--json")) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["dataset"]["release"]["revision"] == "v1"
    assert any(group["id"] == "depth/2048" for group in data["dataset"]["groups"])


def test_plan_downloads_nothing(hub_mirror, tmp_path, capsys):
    data = tmp_path / "data"
    models = tmp_path / "runs"
    code = main(_get(
        hub_mirror, "--task", "depth", "--resolutions", "2048", "--splits", "train",
        "--plan", "--data-root", str(data), "--models-dir", str(models),
    ))
    assert code == 0
    out = capsys.readouterr().out
    assert "depth/2048" in out
    assert "total" in out
    assert not data.exists()
    assert not models.exists()


def test_plan_json_counts_the_files(hub_mirror, tmp_path, capsys):
    data = tmp_path / "data"
    code = main(_get(
        hub_mirror, "--task", "depth", "--resolutions", "2048", "--splits", "train",
        "--plan", "--json", "--data-root", str(data), "--models-dir", str(tmp_path / "runs"),
    ))
    assert code == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["dataset"]["files"] > 0
    assert plan["dataset"]["bytes"] > 0
    assert plan["dataset"]["destination"] == str(data)


def test_download_refuses_without_yes_when_not_a_terminal(hub_mirror, tmp_path, capsys):
    data = tmp_path / "data"
    code = main(_get(
        hub_mirror, "--task", "depth", "--resolutions", "2048", "--splits", "train",
        "--data-root", str(data), "--models-dir", str(tmp_path / "runs"),
    ))
    assert code == 2
    err = capsys.readouterr().err
    assert "--yes" in err
    assert not data.exists()


def test_yes_installs_and_a_second_run_is_idempotent(hub_mirror, tmp_path, capsys):
    data = tmp_path / "data"
    models = tmp_path / "runs"
    common = _get(
        hub_mirror, "--task", "depth", "--resolutions", "2048", "--splits", "train",
        "--yes", "--data-root", str(data), "--models-dir", str(models),
    )
    assert main(common) == 0
    assert (data / "train" / "rgb_day").is_dir()
    capsys.readouterr()
    assert main(common) == 0
    out = capsys.readouterr().out
    assert "skipped" in out or "already complete" in out


def test_models_only_does_not_create_the_dataset_root(hub_mirror, tmp_path, capsys):
    data = tmp_path / "data"
    models = tmp_path / "runs"
    assert main(_get(
        hub_mirror, "--models", "midas_small@1024", "--yes",
        "--data-root", str(data), "--models-dir", str(models),
    )) == 0
    assert (models / "depth" / "midas_small" / "1024" / "weights" / "best.pt").is_file()
    assert not (models / "depth" / "midas_small" / "2048").exists()
    assert not data.exists()


def test_unknown_group_exits_two_with_the_list_hint(hub_mirror, tmp_path, capsys):
    code = main(_get(hub_mirror, "--group", "depth/9999", "--plan",
                     "--data-root", str(tmp_path / "data")))
    assert code == 2
    assert "symtrs get --list" in capsys.readouterr().err


def test_verify_reports_a_damaged_file_and_repair_restores_it(hub_mirror, tmp_path, capsys):
    data = tmp_path / "data"
    models = tmp_path / "runs"
    common = _get(
        hub_mirror, "--task", "depth", "--resolutions", "2048", "--splits", "train",
        "--yes", "--data-root", str(data), "--models-dir", str(models),
    )
    assert main(common) == 0
    victim = next((data / "train" / "rgb_day").rglob("rgb_day_2048_*.png"))
    victim.write_bytes(b"x")
    capsys.readouterr()

    verify = _get(hub_mirror, "--verify", "--data-root", str(data), "--models-dir", str(models))
    assert main(verify) == 1
    assert "size" in capsys.readouterr().out

    assert main(verify + ["--repair"]) == 0
    out = capsys.readouterr().out
    assert "— OK" in out


def test_repair_without_verify_is_refused(hub_mirror, tmp_path, capsys):
    code = main(_get(hub_mirror, "--repair", "--data-root", str(tmp_path / "data")))
    assert code == 2
    assert "--verify" in capsys.readouterr().err


def test_bare_symtrs_in_a_non_terminal_prints_help(monkeypatch, capsys):
    monkeypatch.setattr("symtrs.cli._interactive", lambda: False)
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "usage: symtrs" in out
    assert "get" in out


def test_bare_symtrs_in_a_terminal_routes_to_get(monkeypatch):
    monkeypatch.setattr("symtrs.cli._interactive", lambda: True)
    monkeypatch.setattr("symtrs.cli._launcher_installed", lambda: True)
    seen = {}

    def fake_dispatch(args):
        seen["command"] = args.command
        return 0

    monkeypatch.setattr("symtrs.cli.dispatch", fake_dispatch)
    assert main([]) == 0
    assert seen["command"] == "get"


def test_bare_symtrs_without_the_extra_prints_help_and_the_install_line(monkeypatch, capsys):
    monkeypatch.setattr("symtrs.cli._interactive", lambda: True)
    monkeypatch.setattr("symtrs.cli._launcher_installed", lambda: False)
    assert main([]) == 0
    captured = capsys.readouterr()
    assert "usage: symtrs" in captured.out
    assert "pip install -e '.[launcher]'" in captured.err
