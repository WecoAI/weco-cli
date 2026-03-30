"""CLI entry point: parser construction, environment setup, and dispatch."""

import argparse
import os
import sys

from rich.console import Console
from rich.traceback import install

from . import commands
from .backends import load_backend
from .parsers import (
    configure_credits_parser,
    configure_observe_parser,
    configure_resume_parser,
    configure_run_parser,
    configure_setup_parser,
    configure_share_parser,
)

install(show_locals=os.environ.get("WECO_TRACEBACK_SHOW_LOCALS", "").lower() in {"1", "true", "yes"})
console = Console()

# Backward-compatible exports used by tests and external callers.
parse_api_keys = commands.parse_api_keys
_load_backend = load_backend


def main() -> None:
    """Main function for the Weco CLI."""
    try:
        _main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/]")
        sys.exit(130)


def _main() -> None:
    """Build the parser, parse args, and dispatch."""
    parser = argparse.ArgumentParser(
        description="[bold cyan]Weco CLI[/]\nEnhance your code with AI-driven optimization.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--via-skill", action="store_true", help=argparse.SUPPRESS)

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Register each top-level command with its configure function and handler
    register = [
        ("run", "Run and manage optimizations", configure_run_parser, commands.cmd_run),
        ("resume", "Resume an interrupted optimization run", configure_resume_parser, commands.cmd_resume),
        ("credits", "Manage your Weco credits", configure_credits_parser, commands.cmd_credits),
        ("share", "Create a public share link for a run", configure_share_parser, commands.cmd_share),
        ("setup", "Set up Weco for use with AI tools", configure_setup_parser, commands.cmd_setup),
        ("observe", "Track external optimization runs", configure_observe_parser, commands.cmd_observe),
        ("login", "Log in to Weco and save your API key.", None, commands.cmd_login),
        ("logout", "Log out from Weco and clear saved API key.", None, commands.cmd_logout),
    ]

    for name, help_text, configure_fn, handler_fn in register:
        fmt = argparse.RawTextHelpFormatter if name in ("run", "resume") else None
        kwargs = {"help": help_text, "allow_abbrev": False}
        if fmt:
            kwargs["formatter_class"] = fmt
        command_parser = subparsers.add_parser(name, **kwargs)
        if configure_fn:
            configure_fn(command_parser)
        command_parser.set_defaults(func=handler_fn)

    args = parser.parse_args()

    # Initialise environment
    from ..core.env import WecoEnv

    env = WecoEnv(via_skill=getattr(args, "via_skill", False))
    if args.command != "setup":
        env.check_for_updates()

    # Telemetry
    from ..core.events import CLIInvokedEvent, send_event

    send_event(
        CLIInvokedEvent(
            command=args.command or "help",
            installed_skills=[{"tool": s.tool, "version": s.version} for s in env.installed_skills],
        ),
        env.event_context,
    )

    # Dispatch
    handler = getattr(args, "func", None)
    if handler is None:
        parser.print_help()
        sys.exit(1)
    handler(args, console=console)
