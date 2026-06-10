"""Setup commands for integrating Weco with various AI tools."""

import pathlib
import sys
import tempfile
import time

from rich.console import Console
from rich.prompt import Prompt

from ...events import (
    create_event_context,
    send_event,
    SkillInstallCompletedEvent,
    SkillInstallFailedEvent,
    SkillInstallStartedEvent,
)
from ...utils import DownloadError
from .install import SafetyError, SetupError, download_skill_archive, install_target
from .targets import ALL_SETUP_OPTION_LABEL, ALL_SETUP_OPTION_NAME, SETUP_TARGET_BY_NAME, SETUP_TARGET_NAMES, SETUP_TARGETS


class _SkillSource:
    """Resolves the skill source directory on first use, downloading once if needed.

    Use as a context manager so any downloaded tempdir is cleaned up on exit.
    The resolved path is reused across every target so ``weco setup all``
    downloads exactly once.
    """

    def __init__(self, local_path: pathlib.Path | None, console: Console):
        self._local_path = local_path
        self._console = console
        self._tmp_dir: tempfile.TemporaryDirectory | None = None
        self._downloaded_path: pathlib.Path | None = None

    def __enter__(self) -> "_SkillSource":
        return self

    def __exit__(self, *exc_info) -> None:
        if self._tmp_dir is not None:
            self._tmp_dir.cleanup()
            self._tmp_dir = None

    @property
    def kind(self) -> str:
        return "local" if self._local_path else "download"

    def path(self) -> pathlib.Path:
        if self._local_path is not None:
            return self._local_path
        if self._downloaded_path is None:
            self._tmp_dir = tempfile.TemporaryDirectory()
            dest = pathlib.Path(self._tmp_dir.name) / "skill"
            download_skill_archive(dest, self._console)
            self._downloaded_path = dest
        return self._downloaded_path


def prompt_tool_selection(console: Console) -> list[str]:
    """Prompt the user to select which tool(s) to set up."""
    tool_names = list(SETUP_TARGET_NAMES)
    all_option = len(tool_names) + 1

    console.print("\n[bold cyan]Available tools to set up:[/]\n")
    for i, target in enumerate(SETUP_TARGETS, 1):
        console.print(f"  {i}. {target.label} [dim]({target.name})[/]")
    console.print(f"  {all_option}. {ALL_SETUP_OPTION_LABEL} [dim](default)[/]")

    valid_choices = [str(i) for i in range(1, all_option + 1)]
    choice = Prompt.ask("\n[bold]Select an option[/]", choices=valid_choices, default=str(all_option), show_choices=True)

    idx = int(choice)
    if idx == all_option:
        return tool_names
    return [tool_names[idx - 1]]


def run_setup_for_tool(tool: str, console: Console, source: _SkillSource, ctx) -> None:
    """Run setup for a single tool with event tracking and error handling."""
    send_event(SkillInstallStartedEvent(tool=tool, source=source.kind), ctx)
    start_time = time.time()

    try:
        source_path = source.path()
        install_target(SETUP_TARGET_BY_NAME[tool], console, source_path)
    except DownloadError as e:
        send_event(SkillInstallFailedEvent(tool=tool, source=source.kind, error_type="download_error", stage="download"), ctx)
        console.print(f"\n[bold red]Error:[/] {e}")
        sys.exit(1)
    except SafetyError as e:
        send_event(SkillInstallFailedEvent(tool=tool, source=source.kind, error_type="safety_error", stage="setup"), ctx)
        console.print(f"\n[bold red]Safety Error:[/] {e}")
        sys.exit(1)
    except (SetupError, FileNotFoundError, OSError, ValueError) as e:
        send_event(SkillInstallFailedEvent(tool=tool, source=source.kind, error_type=type(e).__name__, stage="setup"), ctx)
        console.print(f"\n[bold red]Error:[/] {e}")
        sys.exit(1)

    duration_ms = int((time.time() - start_time) * 1000)
    send_event(SkillInstallCompletedEvent(tool=tool, source=source.kind, duration_ms=duration_ms), ctx)


def handle_setup_command(args, console: Console) -> None:
    """Handle the ``weco setup`` command."""
    ctx = create_event_context()

    if args.tool is None:
        selected_tools = prompt_tool_selection(console)
    elif args.tool == ALL_SETUP_OPTION_NAME:
        selected_tools = list(SETUP_TARGET_NAMES)
    elif args.tool in SETUP_TARGET_BY_NAME:
        selected_tools = [args.tool]
    else:
        available = ", ".join((*SETUP_TARGET_NAMES, ALL_SETUP_OPTION_NAME))
        console.print(f"[bold red]Error:[/] Unknown tool: {args.tool}")
        console.print(f"Available tools: {available}")
        sys.exit(1)

    local_path = None
    if getattr(args, "local", None):
        local_path = pathlib.Path(args.local).expanduser().resolve()
        console.print(f"[bold cyan]Using local skill source:[/] {local_path}\n")

    with _SkillSource(local_path, console) as source:
        for tool in selected_tools:
            run_setup_for_tool(tool, console, source, ctx)

    console.print("\n[bold green]Setup complete.[/]")
