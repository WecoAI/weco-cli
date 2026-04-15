"""Tests for setup parser and target selection."""

import argparse
import io

import pytest
from rich.console import Console

from weco.cli import configure_setup_parser
from weco.commands.setup import handle_setup_command, prompt_tool_selection
from weco.commands.setup.targets import ALL_SETUP_OPTION_NAME, SETUP_TARGET_NAMES


def build_setup_parser() -> argparse.ArgumentParser:
    """Create an isolated parser for setup command tests."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    setup_parser = subparsers.add_parser("setup")
    configure_setup_parser(setup_parser)
    return parser


def build_console() -> Console:
    """Create a quiet console for tests."""
    return Console(file=io.StringIO(), force_terminal=False, color_system=None)


def test_setup_parser_accepts_all_supported_tools():
    """The setup parser should expose each supported tool plus the all shortcut."""
    parser = build_setup_parser()

    supported_tools = (*SETUP_TARGET_NAMES, ALL_SETUP_OPTION_NAME)
    for tool in supported_tools:
        args = parser.parse_args(["setup", tool])
        assert args.command == "setup"
        assert args.tool == tool


def test_prompt_tool_selection_defaults_to_all(monkeypatch):
    """Pressing enter at the prompt should select all setup targets."""
    captured = {}

    def fake_ask(*_args, **kwargs):
        captured["default"] = kwargs.get("default")
        return kwargs["default"]

    monkeypatch.setattr("weco.commands.setup.Prompt.ask", fake_ask)

    selected = prompt_tool_selection(console=build_console())

    assert captured["default"] == str(len(SETUP_TARGET_NAMES) + 1)
    assert selected == list(SETUP_TARGET_NAMES)


def test_handle_setup_command_all_runs_all_handlers(monkeypatch):
    """The all shortcut should invoke every supported setup handler."""
    called_tools = []

    def fake_run_setup(tool, console, local_path, ctx):
        called_tools.append((tool, local_path, ctx))

    monkeypatch.setattr("weco.commands.setup.run_setup_for_tool", fake_run_setup)
    args = argparse.Namespace(tool=ALL_SETUP_OPTION_NAME, local=None)

    handle_setup_command(args, console=build_console())

    assert [tool for tool, _, _ in called_tools] == list(SETUP_TARGET_NAMES)


@pytest.mark.parametrize("tool", SETUP_TARGET_NAMES)
def test_handle_setup_command_runs_single_selected_handler(monkeypatch, tool):
    """Named setup targets should dispatch to a single handler."""
    called_tools = []

    def fake_run_setup(selected_tool, console, local_path, ctx):
        called_tools.append((selected_tool, local_path, ctx))

    monkeypatch.setattr("weco.commands.setup.run_setup_for_tool", fake_run_setup)
    args = argparse.Namespace(tool=tool, local=None)

    handle_setup_command(args, console=build_console())

    assert [selected_tool for selected_tool, _, _ in called_tools] == [tool]
