"""Tests for `weco start claude` pre-flight checks and dispatch."""

from __future__ import annotations

import argparse

import pytest
from rich.console import Console

from weco.commands.start import cli
from weco.commands.start import tui_bridge


def test_require_claude_cli_exits_when_not_installed(monkeypatch):
    monkeypatch.setattr(cli.shutil, "which", lambda _name: None)
    with pytest.raises(SystemExit) as exc:
        cli._require_claude_cli(Console())
    assert exc.value.code == 1


def test_require_claude_cli_passes_when_installed(monkeypatch):
    monkeypatch.setattr(cli.shutil, "which", lambda _name: "/usr/local/bin/claude")
    cli._require_claude_cli(Console())  # no exit


def _claude_args(**overrides):
    base = dict(
        start_command="claude", allow_tools=False, effort=None, billing="claude", headless=False, prompt=None, claude_args=[]
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _stub_runners(monkeypatch):
    calls = {}
    monkeypatch.setattr(cli, "load_weco_api_key", lambda: "weco-key")
    monkeypatch.setattr(cli.shutil, "which", lambda _name: "/usr/local/bin/claude")

    def _record(key):
        def runner(**kw):
            calls[key] = kw
            return 0

        return runner

    monkeypatch.setattr(tui_bridge, "run_tui_bridge", _record("tui"))
    monkeypatch.setattr(tui_bridge, "run_headless_bridge", _record("headless"))
    return calls


def test_dispatch_defaults_to_tui_runner(monkeypatch):
    calls = _stub_runners(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        cli.handle_start_command(_claude_args(), Console())
    assert exc.value.code == 0
    assert "tui" in calls and "headless" not in calls


def test_headless_flag_routes_to_headless_runner_with_seed(monkeypatch):
    calls = _stub_runners(monkeypatch)
    with pytest.raises(SystemExit):
        cli.handle_start_command(_claude_args(headless=True, prompt="optimize this"), Console())
    assert "headless" in calls and "tui" not in calls
    assert calls["headless"]["seed_prompt"] == "optimize this"


def test_allow_tools_appends_skip_permissions(monkeypatch):
    calls = _stub_runners(monkeypatch)
    with pytest.raises(SystemExit):
        cli.handle_start_command(_claude_args(allow_tools=True), Console())
    assert "--dangerously-skip-permissions" in calls["tui"]["claude_args"]
