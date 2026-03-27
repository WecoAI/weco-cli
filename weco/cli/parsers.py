import argparse

from . import commands
from .backends import EVAL_BACKENDS, register_backend_args
from ..constants import DEFAULT_MODELS
from ..observe.cli import configure_observe_parser


def add_source_args(parser: argparse.ArgumentParser, *, required: bool = False) -> None:
    """Add the mutually-exclusive --source / --sources pair to *parser*."""
    group = parser.add_mutually_exclusive_group(required=required)
    group.add_argument("-s", "--source", type=str, help="Path to a single source file")
    group.add_argument("--sources", nargs="+", type=str, help="Paths to multiple source files")


def collect_source_paths(args: argparse.Namespace) -> list[str] | None:
    """Return a list of source paths from parsed args, or None if neither flag was given."""
    if getattr(args, "sources", None):
        return args.sources
    if getattr(args, "source", None):
        return [args.source]
    return None


def add_api_key_args(parser: argparse.ArgumentParser) -> None:
    """Add the --api-key flag used by `run` and `resume`."""
    default_api_keys = " ".join([f"{provider}=xxx" for provider, _ in DEFAULT_MODELS])
    supported_providers = ", ".join([provider for provider, _ in DEFAULT_MODELS])
    default_models_for_providers = "\n".join([f"- {provider}: {model}" for provider, model in DEFAULT_MODELS])

    parser.add_argument(
        "--api-key",
        nargs="+",
        type=str,
        default=None,
        help=f"""Provide one or more API keys for supported LLM providers.

Use the format 'provider=KEY', separated by spaces to specify multiple keys.

Example:
    --api-key {default_api_keys}

Supported provider names: {supported_providers}.

Default models for providers:
{default_models_for_providers}
""",
    )


def add_output_arg(parser: argparse.ArgumentParser) -> None:
    """Add the --output flag used by several commands."""
    parser.add_argument(
        "--output",
        type=str,
        choices=["rich", "plain"],
        default="rich",
        help="Output mode: 'rich' for interactive terminal UI (default), 'plain' for machine-readable text output.",
    )


def configure_run_parser(run_parser: argparse.ArgumentParser) -> None:
    """Configure the `weco run` parser — optimization flags + subcommands for existing runs."""

    # --- Flags for starting a new optimization (`weco run --source ...`) ---
    add_source_args(run_parser)
    run_parser.add_argument(
        "-c",
        "--eval-command",
        type=str,
        default=None,
        help="Command to run for evaluation. Required unless --eval-backend is used.",
    )
    run_parser.add_argument(
        "-m",
        "--metric",
        type=str,
        default=None,
        help="Metric to optimize (e.g. 'accuracy', 'loss') printed by the eval command.",
    )
    run_parser.add_argument(
        "-g",
        "--goal",
        type=str,
        choices=["maximize", "max", "minimize", "min"],
        default=None,
        help="'maximize'/'max' or 'minimize'/'min'.",
    )
    run_parser.add_argument("-n", "--steps", type=int, default=100, help="Number of steps. Defaults to 100.")
    run_parser.add_argument(
        "-M",
        "--model",
        type=str,
        default=None,
        help="Model to use. Defaults to o4-mini. See https://docs.weco.ai/cli/supported-models",
    )
    run_parser.add_argument("-l", "--log-dir", type=str, default=".runs", help="Directory for logs. Defaults to `.runs`.")
    run_parser.add_argument(
        "-i",
        "--additional-instructions",
        default=None,
        type=str,
        help="Extra guidance (text or file path).",
    )
    run_parser.add_argument("--eval-timeout", type=int, default=None, help="Timeout in seconds per evaluation.")
    run_parser.add_argument("--save-logs", action="store_true", help="Save execution outputs per step.")
    run_parser.add_argument("--apply-change", action="store_true", help="Auto-apply the best solution without prompting.")
    run_parser.add_argument("--require-review", action="store_true", help="Require approval before each evaluation.")
    add_api_key_args(run_parser)
    add_output_arg(run_parser)

    # Eval backend integration
    run_parser.add_argument(
        "--eval-backend",
        type=str,
        choices=["shell"] + list(EVAL_BACKENDS),
        default="shell",
        help="Evaluation backend. 'shell' (default) runs --eval-command directly.",
    )
    register_backend_args(run_parser)

    # --- Subcommands for inspecting/interacting with existing runs ---
    configure_run_subcommands(run_parser)


def configure_run_subcommands(run_parser: argparse.ArgumentParser) -> None:
    """Add subcommands under `weco run` for inspecting and managing existing runs."""
    subs = run_parser.add_subparsers(dest="run_subcommand")

    # weco run status <run-id>
    p = subs.add_parser("status", help="Show run status and progress (JSON)")
    p.add_argument("run_id", type=str, help="Run UUID")
    p.set_defaults(func=commands.cmd_run_status)

    # weco run results <run-id>
    p = subs.add_parser("results", help="Show results sorted by metric")
    p.add_argument("run_id", type=str, help="Run UUID")
    p.add_argument("--top", type=int, default=None, help="Show only the top N results")
    p.add_argument("--format", type=str, choices=["json", "table", "csv"], default="json", help="Output format")
    p.add_argument("--plot", action="store_true", help="Show ASCII metric trajectory")
    p.add_argument("--include-code", action="store_true", help="Include full source code")
    p.set_defaults(func=commands.cmd_run_results)

    # weco run show <run-id> --step N
    p = subs.add_parser("show", help="Show details for a specific step")
    p.add_argument("run_id", type=str, help="Run UUID")
    p.add_argument("--step", type=str, required=True, help="Step number or 'best'")
    p.set_defaults(func=commands.cmd_run_show)

    # weco run diff <run-id> --step N
    p = subs.add_parser("diff", help="Show code diff between steps")
    p.add_argument("run_id", type=str, help="Run UUID")
    p.add_argument("--step", type=str, required=True, help="Step number or 'best'")
    p.add_argument("--against", type=str, default="baseline", help="'baseline' (default), 'parent', or step number")
    p.set_defaults(func=commands.cmd_run_diff)

    # weco run stop <run-id>
    p = subs.add_parser("stop", help="Terminate a running optimization")
    p.add_argument("run_id", type=str, help="Run UUID")
    p.set_defaults(func=commands.cmd_run_stop)

    # weco run instruct <run-id> <instructions>
    p = subs.add_parser("instruct", help="Update additional instructions for an active run")
    p.add_argument("run_id", type=str, help="Run UUID")
    p.add_argument("instructions", type=str, help="New instructions (text or path to file)")
    p.set_defaults(func=commands.cmd_run_instruct)

    # weco run review <run-id>
    p = subs.add_parser("review", help="Show pending approval nodes (review mode)")
    p.add_argument("run_id", type=str, help="Run UUID")
    p.set_defaults(func=commands.cmd_run_review)

    # weco run revise <run-id> --node <id> --source <file>
    p = subs.add_parser("revise", help="Replace a pending node's code with a new revision")
    p.add_argument("run_id", type=str, help="Run UUID")
    p.add_argument("--node", type=str, required=True, help="Node ID to revise")
    add_source_args(p, required=True)
    p.set_defaults(func=commands.cmd_run_revise)

    # weco run submit <run-id> --node <id>
    p = subs.add_parser("submit", help="Submit a pending node for evaluation (review mode)")
    p.add_argument("run_id", type=str, help="Run UUID")
    p.add_argument("--node", type=str, required=True, help="Node ID to submit")
    add_source_args(p)  # optional — creates revision before submitting
    p.add_argument(
        "-c", "--eval-command", type=str, default=None,
        help="Override the eval command (use when the stored command doesn't work in this environment)",
    )
    p.set_defaults(func=commands.cmd_run_submit)


def configure_resume_parser(resume_parser: argparse.ArgumentParser) -> None:
    """Configure the `weco resume` parser."""
    resume_parser.add_argument("run_id", type=str, help="UUID of the run to resume")
    resume_parser.add_argument("--apply-change", action="store_true", help="Auto-apply the best solution without prompting.")
    add_api_key_args(resume_parser)
    add_output_arg(resume_parser)


def configure_credits_parser(credits_parser: argparse.ArgumentParser) -> None:
    """Configure the `weco credits` parser and its subcommands."""
    subs = credits_parser.add_subparsers(dest="credits_command", help="Credit management commands")

    subs.add_parser("balance", help="Check your current credit balance")

    def parse_credit_amount(value: str) -> float:
        try:
            amount = float(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("Amount must be a number.") from exc
        return round(amount, 2)

    topup = subs.add_parser("topup", help="Purchase additional credits")
    topup.add_argument(
        "amount",
        nargs="?",
        type=parse_credit_amount,
        default=parse_credit_amount("10"),
        metavar="CREDITS",
        help="Credits to purchase (minimum 2, default 10)",
    )

    cost = subs.add_parser("cost", help="Check credit spend for a run")
    cost.add_argument("run_id", type=str, help="Run UUID")

    autotopup = subs.add_parser("autotopup", help="Configure automatic top-up")
    autotopup.add_argument("--enable", action="store_true", help="Enable automatic top-up")
    autotopup.add_argument("--disable", action="store_true", help="Disable automatic top-up")
    autotopup.add_argument("--threshold", type=float, default=4.0, help="Balance threshold (default: 4.0)")
    autotopup.add_argument("--amount", type=float, default=50.0, help="Top-up amount (default: 50.0)")


def configure_share_parser(share_parser: argparse.ArgumentParser) -> None:
    """Configure the `weco share` parser."""
    share_parser.add_argument("run_id", type=str, help="UUID of the run to share")
    add_output_arg(share_parser)


def configure_setup_parser(setup_parser: argparse.ArgumentParser) -> None:
    """Configure the `weco setup` parser and its subcommands."""
    subs = setup_parser.add_subparsers(dest="tool", help="AI tool to set up")

    for tool_name, tool_help in [("claude-code", "Set up Weco skill for Claude Code"), ("cursor", "Set up Weco rules for Cursor")]:
        p = subs.add_parser(tool_name, help=tool_help)
        p.add_argument("--local", type=str, metavar="PATH", help="Use a local weco-skill directory (development)")


__all__ = [
    "collect_source_paths",
    "configure_credits_parser",
    "configure_observe_parser",
    "configure_resume_parser",
    "configure_run_parser",
    "configure_setup_parser",
    "configure_share_parser",
]
