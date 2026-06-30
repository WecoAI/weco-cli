"""`weco start claude` — run Claude Code with a bidirectional bridge to the dashboard."""

from __future__ import annotations

import argparse
import shutil
import sys

from rich.console import Console

from weco.config import load_weco_api_key

from .sdk_config import VALID_EFFORTS


def configure_start_parser(start_parser: argparse.ArgumentParser) -> None:
    sub = start_parser.add_subparsers(dest="start_command", help="What to start")

    claude_parser = sub.add_parser(
        "claude",
        help="Launch Claude Code bridged to the Weco dashboard (bidirectional)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    claude_parser.add_argument(
        "--allow-tools",
        action="store_true",
        help=(
            "Auto-approve all Claude Code tool calls (passes --dangerously-skip-permissions). "
            "Bash, Write, Edit etc. run without prompting. Use only when you trust the agent."
        ),
    )
    claude_parser.add_argument(
        "--effort",
        type=str,
        choices=list(VALID_EFFORTS),
        default=None,
        help=(
            "Thinking effort level. Raises the thinking-token budget so Claude returns "
            "`thinking` content blocks alongside its response — these stream into the "
            "dashboard's Reasoning section. Omit to inherit Claude's own default (no extra "
            "thinking)."
        ),
    )
    claude_parser.add_argument(
        "--headless",
        action="store_true",
        help=(
            "Run with no local TUI — stream only to the dashboard, printing key "
            "lifecycle lines to the console. Use when launching in the background "
            "(no terminal to draw into), e.g. an agent spawning a bridged session. "
            "Pair with --allow-tools (no local approval modal) and --prompt to seed "
            "the first turn; the dashboard is the interactive surface."
        ),
    )
    claude_parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default=None,
        help=(
            "Seed the first turn with this prompt instead of waiting for input. "
            "Required to make --headless do anything immediately; also works in the TUI."
        ),
    )
    claude_parser.add_argument(
        "--billing",
        type=str,
        choices=["claude", "weco"],
        default="claude",
        help=(
            "Where LLM calls are billed. 'claude' (default) uses your local Claude auth "
            "(OAuth from `claude login`, or `ANTHROPIC_API_KEY`) and your usage counts "
            "against whatever credit pool Anthropic's billing applies to those credentials. "
            "'weco' routes through Weco's LLM proxy and deducts from your Weco credit wallet "
            "(works for any auth scheme). Choose 'weco' if you'd rather pay through Weco."
        ),
    )
    claude_parser.add_argument(
        "claude_args",
        nargs=argparse.REMAINDER,
        help="Arguments to forward to claude (prefix with -- to separate from weco flags)",
    )


def handle_start_command(args: argparse.Namespace, console: Console) -> None:
    sub = getattr(args, "start_command", None)
    if sub == "claude":
        _handle_claude(args, console)
    else:
        console.print("[red]Unknown start subcommand.[/]")
        console.print("Usage: [bold]weco start claude[/]")
        sys.exit(2)


def _require_api_key(console: Console) -> str:
    api_key = load_weco_api_key()
    if not api_key:
        console.print("[red]Not logged in.[/] Run [bold]weco login[/] first.")
        sys.exit(1)
    return api_key


def _require_claude_cli(console: Console) -> None:
    """Fail fast (before creating a session or launching the TUI) if Claude
    Code isn't installed — `weco start claude` drives it under the hood."""
    if shutil.which("claude"):
        return
    console.print(
        "[red]Claude Code CLI not found.[/] [bold]weco start claude[/] runs Claude Code under the hood.\n"
        "Install it, then re-run: https://code.claude.com/docs/en/quickstart"
    )
    sys.exit(1)


def _strip_arg_separator(args: list[str]) -> list[str]:
    if args and args[0] == "--":
        return args[1:]
    return args


def _handle_claude(args: argparse.Namespace, console: Console) -> None:
    api_key = _require_api_key(console)
    _require_claude_cli(console)

    forwarded = _strip_arg_separator(list(getattr(args, "claude_args", []) or []))
    if getattr(args, "allow_tools", False) and not any(a == "--dangerously-skip-permissions" for a in forwarded):
        forwarded.append("--dangerously-skip-permissions")

    effort = getattr(args, "effort", None)
    billing = getattr(args, "billing", "claude")
    headless = getattr(args, "headless", False)
    seed_prompt = getattr(args, "prompt", None)

    from weco import __base_url__
    from .tui_bridge import run_headless_bridge, run_tui_bridge

    runner = run_headless_bridge if headless else run_tui_bridge
    exit_code = runner(
        claude_args=forwarded,
        api_key=api_key,
        console=console,
        billing=billing,
        weco_api_base=__base_url__,
        effort=effort,
        seed_prompt=seed_prompt,
    )
    sys.exit(exit_code)
