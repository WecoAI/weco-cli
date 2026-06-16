"""Tests for `weco start claude` pre-flight checks."""

from __future__ import annotations

import pytest
from rich.console import Console

from weco.commands.start import cli


def test_require_claude_cli_exits_when_not_installed(monkeypatch):
    monkeypatch.setattr(cli.shutil, "which", lambda _name: None)
    with pytest.raises(SystemExit) as exc:
        cli._require_claude_cli(Console())
    assert exc.value.code == 1


def test_require_claude_cli_passes_when_installed(monkeypatch):
    monkeypatch.setattr(cli.shutil, "which", lambda _name: "/usr/local/bin/claude")
    cli._require_claude_cli(Console())  # no exit
