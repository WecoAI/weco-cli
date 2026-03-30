"""``weco setup`` — install Weco skills for AI coding tools."""

import pathlib
import sys
import time

from rich.console import Console
from rich.prompt import Prompt

from ...core.events import (
    send_event,
    create_event_context,
    SkillInstallStartedEvent,
    SkillInstallCompletedEvent,
    SkillInstallFailedEvent,
)
from ...core.files import copy_file, SafetyError
from .errors import SetupError, DownloadError
from .install import install_skill
from .paths import (
    WECO_SKILL_DIR,
    WECO_CLAUDE_SNIPPET_PATH,
    WECO_CLAUDE_MD_PATH,
    CURSOR_WECO_SKILL_DIR,
)


def setup_claude_code(console: Console, local_path: pathlib.Path | None = None) -> None:
    """Set up Weco skill for Claude Code."""
    console.print("[bold blue]Setting up Weco for Claude Code...[/]\n")

    install_skill(WECO_SKILL_DIR, console, local_path)

    copy_file(WECO_CLAUDE_SNIPPET_PATH, WECO_CLAUDE_MD_PATH)
    console.print("[green]CLAUDE.md installed to skill directory.[/]")

    console.print("\n[bold green]Setup complete![/]")
    if local_path:
        console.print(f"[dim]Skill copied from: {local_path}[/]")
    console.print(f"[dim]Skill installed at: {WECO_SKILL_DIR}[/]")


def setup_cursor(console: Console, local_path: pathlib.Path | None = None) -> None:
    """Set up Weco rules for Cursor."""
    console.print("[bold blue]Setting up Weco for Cursor...[/]\n")

    install_skill(CURSOR_WECO_SKILL_DIR, console, local_path)

    console.print("\n[bold green]Setup complete![/]")
    if local_path:
        console.print(f"[dim]Skill copied from: {local_path}[/]")
    console.print(f"[dim]Skill installed at: {CURSOR_WECO_SKILL_DIR}[/]")


SETUP_HANDLERS = {"claude-code": setup_claude_code, "cursor": setup_cursor}


def prompt_tool_selection(console: Console) -> list[str]:
    """Prompt the user to select which tool(s) to set up.

    Returns:
        List of tool names to set up.
    """
    tool_names = list(SETUP_HANDLERS.keys())
    all_option = len(tool_names) + 1

    console.print("\n[bold cyan]Available tools to set up:[/]\n")
    for i, name in enumerate(tool_names, 1):
        console.print(f"  {i}. {name}")
    console.print(f"  {all_option}. All of the above")

    valid_choices = [str(i) for i in range(1, all_option + 1)]
    choice = Prompt.ask("\n[bold]Select an option[/]", choices=valid_choices, show_choices=True)

    idx = int(choice)
    if idx == all_option:
        return tool_names
    return [tool_names[idx - 1]]


def _run_setup_for_tool(tool: str, console: Console, local_path: pathlib.Path | None, ctx) -> None:
    """Run setup for a single tool with event tracking and error handling."""
    source = "local" if local_path else "download"

    send_event(SkillInstallStartedEvent(tool=tool, source=source), ctx)

    start_time = time.time()

    try:
        handler = SETUP_HANDLERS[tool]
        handler(console, local_path=local_path)

        duration_ms = int((time.time() - start_time) * 1000)
        send_event(SkillInstallCompletedEvent(tool=tool, source=source, duration_ms=duration_ms), ctx)

    except DownloadError as e:
        send_event(SkillInstallFailedEvent(tool=tool, source=source, error_type="download_error", stage="download"), ctx)
        console.print(f"[bold red]Error:[/] {e}")
        sys.exit(1)
    except SafetyError as e:
        send_event(SkillInstallFailedEvent(tool=tool, source=source, error_type="safety_error", stage="setup"), ctx)
        console.print(f"[bold red]Safety Error:[/] {e}")
        sys.exit(1)
    except (SetupError, FileNotFoundError, OSError, ValueError) as e:
        error_type = type(e).__name__
        send_event(SkillInstallFailedEvent(tool=tool, source=source, error_type=error_type, stage="setup"), ctx)
        console.print(f"[bold red]Error:[/] {e}")
        sys.exit(1)


def handle_setup_command(args, console: Console) -> None:
    """Handle the ``weco setup`` command."""
    ctx = create_event_context()

    if args.tool is None:
        selected_tools = prompt_tool_selection(console)
    else:
        handler = SETUP_HANDLERS.get(args.tool)
        if handler is None:
            available_tools = ", ".join(SETUP_HANDLERS)
            console.print(f"[bold red]Error:[/] Unknown tool: {args.tool}")
            console.print(f"Available tools: {available_tools}")
            sys.exit(1)
        selected_tools = [args.tool]

    local_path = None
    if hasattr(args, "local") and args.local:
        local_path = pathlib.Path(args.local).expanduser().resolve()

    for tool in selected_tools:
        _run_setup_for_tool(tool, console, local_path, ctx)
