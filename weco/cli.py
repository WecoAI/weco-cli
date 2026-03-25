import argparse
import importlib
import sys
from rich.console import Console
from rich.traceback import install

from .constants import DEFAULT_MODELS
from .observe.cli import configure_observe_parser

install(show_locals=True)
console = Console()

# Eval backend registry. Each entry maps a backend name to its module path.
# To add a new backend, create weco/integrations/<name>/backend.py with
# register_args(), validate_args(), and build_eval_command(), then add it here.
_EVAL_BACKENDS = {"langsmith": "weco.integrations.langsmith.backend", "langfuse": "weco.integrations.langfuse.backend"}


def _load_backend(name: str):
    """Lazily import an eval backend module by name."""
    return importlib.import_module(_EVAL_BACKENDS[name])


def parse_api_keys(api_key_args: list[str] | None) -> dict[str, str]:
    """Parse API key arguments from CLI into a dictionary.

    Args:
        api_key_args: List of strings in format 'provider=key' (e.g., ['openai=sk-xxx', 'anthropic=sk-ant-yyy'])

    Returns:
        Dictionary mapping provider names to API keys. Returns empty dict if no keys provided.

    Raises:
        ValueError: If any argument is not in the correct format.
    """
    if not api_key_args:
        return {}

    api_keys = {}
    for arg in api_key_args:
        try:
            provider, key = (s.strip() for s in arg.split("=", 1))
        except Exception:
            raise ValueError(f"Invalid API key format: '{arg}'. Expected format: 'provider=key'")

        if not provider or not key:
            raise ValueError(f"Invalid API key format: '{arg}'. Provider and key must be non-empty.")

        api_keys[provider.lower()] = key

    return api_keys


# ---------------------------------------------------------------------------
# Helpers shared by `--source` / `--sources` flags
# ---------------------------------------------------------------------------


def _add_source_args(parser: argparse.ArgumentParser, *, required: bool = False) -> None:
    """Add the mutually-exclusive --source / --sources pair to *parser*."""
    group = parser.add_mutually_exclusive_group(required=required)
    group.add_argument("-s", "--source", type=str, help="Path to a single source file")
    group.add_argument("--sources", nargs="+", type=str, help="Paths to multiple source files")


def _collect_source_paths(args: argparse.Namespace) -> list[str] | None:
    """Return a list of source paths from parsed args, or None if neither flag was given."""
    if getattr(args, "sources", None):
        return args.sources
    if getattr(args, "source", None):
        return [args.source]
    return None


# ---------------------------------------------------------------------------
# Parser configuration — one `configure_*` function per top-level command
# ---------------------------------------------------------------------------


def _add_api_key_args(parser: argparse.ArgumentParser) -> None:
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


def _add_output_arg(parser: argparse.ArgumentParser) -> None:
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
    _add_source_args(run_parser)
    run_parser.add_argument(
        "-c", "--eval-command", type=str, default=None,
        help="Command to run for evaluation. Required unless --eval-backend is used.",
    )
    run_parser.add_argument(
        "-m", "--metric", type=str, default=None,
        help="Metric to optimize (e.g. 'accuracy', 'loss') printed by the eval command.",
    )
    run_parser.add_argument(
        "-g", "--goal", type=str, choices=["maximize", "max", "minimize", "min"], default=None,
        help="'maximize'/'max' or 'minimize'/'min'.",
    )
    run_parser.add_argument("-n", "--steps", type=int, default=100, help="Number of steps. Defaults to 100.")
    run_parser.add_argument(
        "-M", "--model", type=str, default=None,
        help="Model to use. Defaults to o4-mini. See https://docs.weco.ai/cli/supported-models",
    )
    run_parser.add_argument("-l", "--log-dir", type=str, default=".runs", help="Directory for logs. Defaults to `.runs`.")
    run_parser.add_argument(
        "-i", "--additional-instructions", default=None, type=str,
        help="Extra guidance (text or file path).",
    )
    run_parser.add_argument("--eval-timeout", type=int, default=None, help="Timeout in seconds per evaluation.")
    run_parser.add_argument("--save-logs", action="store_true", help="Save execution outputs per step.")
    run_parser.add_argument("--apply-change", action="store_true", help="Auto-apply the best solution without prompting.")
    run_parser.add_argument("--require-review", action="store_true", help="Require approval before each evaluation.")
    _add_api_key_args(run_parser)
    _add_output_arg(run_parser)

    # Eval backend integration
    run_parser.add_argument(
        "--eval-backend", type=str, choices=["shell"] + list(_EVAL_BACKENDS), default="shell",
        help="Evaluation backend. 'shell' (default) runs --eval-command directly.",
    )
    for backend_name in _EVAL_BACKENDS:
        _load_backend(backend_name).register_args(run_parser)

    # --- Subcommands for inspecting/interacting with existing runs ---
    _configure_run_subcommands(run_parser)


def _configure_run_subcommands(run_parser: argparse.ArgumentParser) -> None:
    """Add subcommands under `weco run` for inspecting and managing existing runs."""
    subs = run_parser.add_subparsers(dest="run_subcommand")

    # weco run status <run-id>
    p = subs.add_parser("status", help="Show run status and progress (JSON)")
    p.add_argument("run_id", type=str, help="Run UUID")

    # weco run results <run-id>
    p = subs.add_parser("results", help="Show results sorted by metric")
    p.add_argument("run_id", type=str, help="Run UUID")
    p.add_argument("--top", type=int, default=None, help="Show only the top N results")
    p.add_argument("--format", type=str, choices=["json", "table", "csv"], default="json", help="Output format")
    p.add_argument("--plot", action="store_true", help="Show ASCII metric trajectory")
    p.add_argument("--include-code", action="store_true", help="Include full source code")

    # weco run show <run-id> --step N
    p = subs.add_parser("show", help="Show details for a specific step")
    p.add_argument("run_id", type=str, help="Run UUID")
    p.add_argument("--step", type=str, required=True, help="Step number or 'best'")

    # weco run diff <run-id> --step N
    p = subs.add_parser("diff", help="Show code diff between steps")
    p.add_argument("run_id", type=str, help="Run UUID")
    p.add_argument("--step", type=str, required=True, help="Step number or 'best'")
    p.add_argument("--against", type=str, default="baseline", help="'baseline' (default), 'parent', or step number")

    # weco run stop <run-id>
    p = subs.add_parser("stop", help="Terminate a running optimization")
    p.add_argument("run_id", type=str, help="Run UUID")

    # weco run review <run-id>
    p = subs.add_parser("review", help="Show pending approval nodes (review mode)")
    p.add_argument("run_id", type=str, help="Run UUID")

    # weco run revise <run-id> --node <id> --source <file>
    p = subs.add_parser("revise", help="Replace a pending node's code with a new revision")
    p.add_argument("run_id", type=str, help="Run UUID")
    p.add_argument("--node", type=str, required=True, help="Node ID to revise")
    _add_source_args(p, required=True)

    # weco run submit <run-id> --node <id>
    p = subs.add_parser("submit", help="Submit a pending node for evaluation (review mode)")
    p.add_argument("run_id", type=str, help="Run UUID")
    p.add_argument("--node", type=str, required=True, help="Node ID to submit")
    _add_source_args(p)  # optional — creates revision before submitting


def configure_resume_parser(resume_parser: argparse.ArgumentParser) -> None:
    """Configure the `weco resume` parser."""
    resume_parser.add_argument("run_id", type=str, help="UUID of the run to resume")
    resume_parser.add_argument("--apply-change", action="store_true", help="Auto-apply the best solution without prompting.")
    _add_api_key_args(resume_parser)
    _add_output_arg(resume_parser)


def configure_credits_parser(credits_parser: argparse.ArgumentParser) -> None:
    """Configure the `weco credits` parser and its subcommands."""
    subs = credits_parser.add_subparsers(dest="credits_command", help="Credit management commands")

    subs.add_parser("balance", help="Check your current credit balance")

    def _parse_credit_amount(value: str) -> float:
        try:
            amount = float(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("Amount must be a number.") from exc
        return round(amount, 2)

    topup = subs.add_parser("topup", help="Purchase additional credits")
    topup.add_argument(
        "amount", nargs="?", type=_parse_credit_amount, default=_parse_credit_amount("10"),
        metavar="CREDITS", help="Credits to purchase (minimum 2, default 10)",
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
    _add_output_arg(share_parser)


def configure_setup_parser(setup_parser: argparse.ArgumentParser) -> None:
    """Configure the `weco setup` parser and its subcommands."""
    subs = setup_parser.add_subparsers(dest="tool", help="AI tool to set up")

    for tool_name, tool_help in [("claude-code", "Set up Weco skill for Claude Code"),
                                  ("cursor", "Set up Weco rules for Cursor")]:
        p = subs.add_parser(tool_name, help=tool_help)
        p.add_argument("--local", type=str, metavar="PATH", help="Use a local weco-skill directory (development)")


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def _cmd_login(args: argparse.Namespace) -> None:
    from .auth import perform_login
    from .config import load_weco_api_key

    if load_weco_api_key():
        console.print("[bold green]You are already logged in.[/]")
        console.print("[dim]Use 'weco logout' to log out first if you want to switch accounts.[/]")
        sys.exit(0)

    sys.exit(0 if perform_login(console) else 1)


def _cmd_logout(_args: argparse.Namespace) -> None:
    from .config import clear_api_key

    clear_api_key()


def _cmd_run(args: argparse.Namespace) -> None:
    """Dispatch `weco run` — start a new optimization or route to a subcommand."""
    sub = getattr(args, "run_subcommand", None)
    if sub is not None:
        _cmd_run_subcommand(args, sub)
        return

    # No subcommand — start a new optimization
    from .optimizer import optimize
    from .utils import get_default_model, UnrecognizedAPIKeysError, DefaultModelNotFoundError
    from .validation import validate_sources, validate_log_directory, ValidationError, print_validation_error

    ctx = _get_event_context()

    # Resolve eval backend
    backend_name = getattr(args, "eval_backend", "shell")
    if backend_name != "shell":
        backend = _load_backend(backend_name)
        backend.validate_args(args)
        args.eval_command = backend.build_eval_command(args)
    elif not args.eval_command:
        console.print("[bold red]Error: --eval-command is required (or use --eval-backend <backend>)[/]")
        sys.exit(1)

    # Validate required args
    missing = []
    if not args.source and not args.sources:
        missing.append("--source / --sources")
    if not args.metric:
        missing.append("--metric")
    if not args.goal:
        missing.append("--goal")
    if missing:
        console.print(f"[bold red]Error: missing required arguments: {', '.join(missing)}[/]")
        sys.exit(1)

    source_arg = args.sources if args.sources is not None else [args.source]

    try:
        validate_sources(source_arg)
        validate_log_directory(args.log_dir)
    except ValidationError as e:
        print_validation_error(e, console)
        sys.exit(1)

    try:
        api_keys = parse_api_keys(args.api_key)
    except ValueError as e:
        console.print(f"[bold red]Error parsing API keys: {e}[/]")
        sys.exit(1)

    model = args.model
    if not model:
        try:
            model = get_default_model(api_keys=api_keys)
        except (UnrecognizedAPIKeysError, DefaultModelNotFoundError) as e:
            console.print(f"[bold red]Error: {e}[/]")
            sys.exit(1)
        if api_keys:
            console.print(f"[bold yellow]Custom API keys provided. Using default model: {model} for the run.[/]")

    from .credits import check_promotional_credits

    model = check_promotional_credits(model, api_keys, console)

    from .events import send_event, RunStartAttemptedEvent

    send_event(
        RunStartAttemptedEvent(
            output_mode=args.output, require_review=args.require_review,
            save_logs=args.save_logs, steps=args.steps, model=model,
        ),
        ctx,
    )

    success = optimize(
        source=source_arg, eval_command=args.eval_command, metric=args.metric,
        goal=args.goal, model=model, steps=args.steps, log_dir=args.log_dir,
        additional_instructions=args.additional_instructions, eval_timeout=args.eval_timeout,
        save_logs=args.save_logs, api_keys=api_keys, apply_change=args.apply_change,
        require_review=args.require_review, output_mode=args.output,
    )
    sys.exit(0 if success else 1)


def _cmd_run_subcommand(args: argparse.Namespace, sub: str) -> None:
    """Dispatch a `weco run <subcommand>` to the appropriate handler."""
    from .commands import (
        handle_status_command, handle_results_command, handle_show_command,
        handle_diff_command, handle_stop_command, handle_review_command,
        handle_revise_command, handle_submit_command,
    )

    handlers = {
        "status": lambda: handle_status_command(run_id=args.run_id, console=console),
        "results": lambda: handle_results_command(
            run_id=args.run_id, top=args.top, format=args.format,
            plot=args.plot, include_code=args.include_code, console=console,
        ),
        "show": lambda: handle_show_command(run_id=args.run_id, step=args.step, console=console),
        "diff": lambda: handle_diff_command(run_id=args.run_id, step=args.step, against=args.against, console=console),
        "stop": lambda: handle_stop_command(run_id=args.run_id, console=console),
        "review": lambda: handle_review_command(run_id=args.run_id, console=console),
        "revise": lambda: handle_revise_command(
            run_id=args.run_id, node_id=args.node,
            source_paths=_collect_source_paths(args), console=console,
        ),
        "submit": lambda: handle_submit_command(
            run_id=args.run_id, node_id=args.node,
            source_paths=_collect_source_paths(args), console=console,
        ),
    }

    handler = handlers.get(sub)
    if handler is None:
        console.print(f"[bold red]Unknown run subcommand: {sub}[/]")
        sys.exit(1)
    handler()


def _cmd_resume(args: argparse.Namespace) -> None:
    from .optimizer import resume_optimization

    try:
        api_keys = parse_api_keys(args.api_key)
    except ValueError as e:
        console.print(f"[bold red]Error parsing API keys: {e}[/]")
        sys.exit(1)

    success = resume_optimization(
        run_id=args.run_id, api_keys=api_keys, apply_change=args.apply_change, output_mode=args.output,
    )
    sys.exit(0 if success else 1)


def _cmd_credits(args: argparse.Namespace) -> None:
    from .credits import handle_credits_command

    handle_credits_command(args, console)


def _cmd_share(args: argparse.Namespace) -> None:
    from .share import handle_share_command

    handle_share_command(run_id=args.run_id, output_mode=args.output, console=console)


def _cmd_setup(args: argparse.Namespace) -> None:
    from .setup import handle_setup_command

    handle_setup_command(args, console)


def _cmd_observe(args: argparse.Namespace) -> None:
    from .observe.cli import execute_observe_command

    execute_observe_command(args)


# ---------------------------------------------------------------------------
# Event helpers
# ---------------------------------------------------------------------------


def _get_event_context():
    from .events import get_event_context

    return get_event_context()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


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
    _register = [
        ("run",     "Run and manage optimizations",               configure_run_parser,     _cmd_run),
        ("resume",  "Resume an interrupted optimization run",     configure_resume_parser,  _cmd_resume),
        ("credits", "Manage your Weco credits",                   configure_credits_parser, _cmd_credits),
        ("share",   "Create a public share link for a run",       configure_share_parser,   _cmd_share),
        ("setup",   "Set up Weco for use with AI tools",          configure_setup_parser,   _cmd_setup),
        ("observe", "Track external optimization runs",           configure_observe_parser, _cmd_observe),
        ("login",   "Log in to Weco and save your API key.",      None,                     _cmd_login),
        ("logout",  "Log out from Weco and clear saved API key.", None,                     _cmd_logout),
    ]

    for name, help_text, configure_fn, handler_fn in _register:
        fmt = argparse.RawTextHelpFormatter if name in ("run", "resume") else None
        kwargs = {"help": help_text, "allow_abbrev": False}
        if fmt:
            kwargs["formatter_class"] = fmt
        p = subparsers.add_parser(name, **kwargs)
        if configure_fn:
            configure_fn(p)
        p.set_defaults(func=handler_fn)

    args = parser.parse_args()

    # Initialise environment
    from .env import WecoEnv

    env = WecoEnv(via_skill=getattr(args, "via_skill", False))
    if args.command != "setup":
        env.check_for_updates()

    # Telemetry
    from .events import send_event, CLIInvokedEvent

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
    handler(args)
