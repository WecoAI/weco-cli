import argparse
import sys
from rich.console import Console
from rich.traceback import install

from .auth import perform_login
from .config import clear_api_key, load_weco_api_key
from .constants import DEFAULT_MODELS
from .events import (
    send_event,
    create_event_context,
    get_event_context,
    set_event_context,
    CLIInvokedEvent,
    RunStartAttemptedEvent,
)
from .utils import check_for_cli_updates, get_default_model, UnrecognizedAPIKeysError, DefaultModelNotFoundError
from .validation import validate_source_file, validate_log_directory, ValidationError, print_validation_error


install(show_locals=True)
console = Console()


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


# Function to define and return the run_parser (or configure it on a passed subparser object)
# This helps keep main() cleaner and centralizes run command arg definitions.
def configure_run_parser(run_parser: argparse.ArgumentParser) -> None:
    run_parser.add_argument(
        "-s",
        "--source",
        type=str,
        required=True,
        help="Path to the source code file that will be optimized (e.g., `optimize.py`)",
    )
    run_parser.add_argument(
        "-c",
        "--eval-command",
        type=str,
        required=True,
        help="Command to run for evaluation (e.g. 'python eval.py --arg1=val1').",
    )
    run_parser.add_argument(
        "-m",
        "--metric",
        type=str,
        required=True,
        help="Metric to optimize (e.g. 'accuracy', 'loss', 'f1_score') that is printed to the terminal by the eval command.",
    )
    run_parser.add_argument(
        "-g",
        "--goal",
        type=str,
        choices=["maximize", "max", "minimize", "min"],
        required=True,
        help="Specify 'maximize'/'max' to maximize the metric or 'minimize'/'min' to minimize it.",
    )
    run_parser.add_argument("-n", "--steps", type=int, default=100, help="Number of steps to run. Defaults to 100.")
    run_parser.add_argument(
        "-M",
        "--model",
        type=str,
        default=None,
        help="Model to use for optimization. Defaults to `o4-mini`. See full list at https://docs.weco.ai/cli/supported-models",
    )
    run_parser.add_argument(
        "-l", "--log-dir", type=str, default=".runs", help="Directory to store logs and results. Defaults to `.runs`."
    )
    run_parser.add_argument(
        "-i",
        "--additional-instructions",
        default=None,
        type=str,
        help="Description of additional instruction or path to a file containing additional instructions. Defaults to None.",
    )
    run_parser.add_argument(
        "--eval-timeout",
        type=int,
        default=None,
        help="Timeout in seconds for each evaluation. No timeout by default. Example: --eval-timeout 3600",
    )
    run_parser.add_argument(
        "--save-logs",
        action="store_true",
        help="Save execution output to .runs/<run-id>/outputs/step_<n>.out.txt with JSONL index",
    )
    run_parser.add_argument(
        "--apply-change",
        action="store_true",
        help="Automatically apply the best solution to the source file without prompting",
    )
    run_parser.add_argument(
        "--require-review",
        action="store_true",
        help="Require manual review and approval of each proposed change before execution",
    )

    default_api_keys = " ".join([f"{provider}=xxx" for provider, _ in DEFAULT_MODELS])
    supported_providers = ", ".join([provider for provider, _ in DEFAULT_MODELS])
    default_models_for_providers = "\n".join([f"- {provider}: {model}" for provider, model in DEFAULT_MODELS])
    run_parser.add_argument(
        "--api-key",
        nargs="+",
        type=str,
        default=None,
        help=f"""Provide one or more API keys for supported LLM providers. Specify a model with the --model flag.
Weco will use the default model for the provider if no model is specified.

Use the format 'provider=KEY', separated by spaces to specify multiple keys.

Example:
    --api-key {default_api_keys}

Supported provider names: {supported_providers}.

Default models for providers:
{default_models_for_providers}
""",
    )
    run_parser.add_argument(
        "--output",
        type=str,
        choices=["rich", "plain"],
        default="rich",
        help="Output mode: 'rich' for interactive terminal UI (default), 'plain' for machine-readable text output suitable for LLM agents.",
    )


def configure_credits_parser(credits_parser: argparse.ArgumentParser) -> None:
    """Configure the credits command parser and all its subcommands."""
    credits_subparsers = credits_parser.add_subparsers(dest="credits_command", help="Credit management commands")

    # Credits balance command
    _ = credits_subparsers.add_parser("balance", help="Check your current credit balance")

    # Coerce CLI input into a float with two decimal precision for the API payload.
    def _parse_credit_amount(value: str) -> float:
        try:
            amount = float(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("Amount must be a number.") from exc

        return round(amount, 2)

    # Credits topup command
    topup_parser = credits_subparsers.add_parser("topup", help="Purchase additional credits")
    topup_parser.add_argument(
        "amount",
        nargs="?",
        type=_parse_credit_amount,
        default=_parse_credit_amount("10"),
        metavar="CREDITS",
        help="Amount of credits to purchase (minimum 2, defaults to 10)",
    )

    # Credits cost command
    cost_parser = credits_subparsers.add_parser("cost", help="Check credit spend for a run")
    cost_parser.add_argument(
        "run_id", type=str, help="The run ID to check credit spend for (e.g., '0002e071-1b67-411f-a514-36947f0c4b31')"
    )

    # Credits autotopup command
    autotopup_parser = credits_subparsers.add_parser("autotopup", help="Configure automatic top-up")
    autotopup_parser.add_argument("--enable", action="store_true", help="Enable automatic top-up")
    autotopup_parser.add_argument("--disable", action="store_true", help="Disable automatic top-up")
    autotopup_parser.add_argument(
        "--threshold", type=float, default=4.0, help="Balance threshold to trigger auto top-up (default: 4.0 credits)"
    )
    autotopup_parser.add_argument(
        "--amount", type=float, default=50.0, help="Amount to top up when threshold is reached (default: 50.0 credits)"
    )


def _add_setup_source_args(parser: argparse.ArgumentParser) -> None:
    """Add common source arguments to a setup subparser."""
    parser.add_argument(
        "--local", type=str, metavar="PATH", help="Use a local weco-skill directory instead of downloading (for development)"
    )


def configure_setup_parser(setup_parser: argparse.ArgumentParser) -> None:
    """Configure the setup command parser and its subcommands."""
    setup_subparsers = setup_parser.add_subparsers(dest="tool", help="AI tool to set up")

    claude_parser = setup_subparsers.add_parser("claude-code", help="Set up Weco skill for Claude Code")
    _add_setup_source_args(claude_parser)

    cursor_parser = setup_subparsers.add_parser("cursor", help="Set up Weco rules for Cursor")
    _add_setup_source_args(cursor_parser)


def configure_resume_parser(resume_parser: argparse.ArgumentParser) -> None:
    """Configure arguments for the resume command."""
    resume_parser.add_argument(
        "run_id", type=str, help="The UUID of the run to resume (e.g., '0002e071-1b67-411f-a514-36947f0c4b31')"
    )
    resume_parser.add_argument(
        "--apply-change",
        action="store_true",
        help="Automatically apply the best solution to the source file without prompting",
    )

    default_api_keys = " ".join([f"{provider}=xxx" for provider, _ in DEFAULT_MODELS])
    supported_providers = ", ".join([provider for provider, _ in DEFAULT_MODELS])

    resume_parser.add_argument(
        "--api-key",
        nargs="+",
        type=str,
        default=None,
        help=f"""Provide one or more API keys for supported LLM providers.
Weco will use the model associated with the run you are resuming.

Use the format 'provider=KEY', separated by spaces to specify multiple keys.

Example:
    --api-key {default_api_keys}

Supported provider names: {supported_providers}.
""",
    )
    resume_parser.add_argument(
        "--output",
        type=str,
        choices=["rich", "plain"],
        default="rich",
        help="Output mode: 'rich' for interactive terminal UI (default), 'plain' for machine-readable text output suitable for LLM agents.",
    )


def execute_run_command(args: argparse.Namespace) -> None:
    """Execute the 'weco run' command with all its logic."""
    from .optimizer import optimize

    ctx = get_event_context()

    # Early validation — fail fast with helpful errors
    try:
        validate_source_file(args.source)
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

    # Send run attempt event before starting (helps measure dropoff before server)
    send_event(
        RunStartAttemptedEvent(
            output_mode=args.output,
            require_review=args.require_review,
            save_logs=args.save_logs,
            steps=args.steps,
            model=model,
        ),
        ctx,
    )

    success = optimize(
        source=args.source,
        eval_command=args.eval_command,
        metric=args.metric,
        goal=args.goal,
        model=model,
        steps=args.steps,
        log_dir=args.log_dir,
        additional_instructions=args.additional_instructions,
        eval_timeout=args.eval_timeout,
        save_logs=args.save_logs,
        api_keys=api_keys,
        apply_change=args.apply_change,
        require_review=args.require_review,
        output_mode=args.output,
    )

    exit_code = 0 if success else 1
    sys.exit(exit_code)


def execute_resume_command(args: argparse.Namespace) -> None:
    """Execute the 'weco resume' command with all its logic."""
    from .optimizer import resume_optimization

    try:
        api_keys = parse_api_keys(args.api_key)
    except ValueError as e:
        console.print(f"[bold red]Error parsing API keys: {e}[/]")
        sys.exit(1)

    success = resume_optimization(
        run_id=args.run_id, api_keys=api_keys, apply_change=args.apply_change, output_mode=args.output
    )

    sys.exit(0 if success else 1)


def main() -> None:
    """Main function for the Weco CLI."""
    try:
        _main()
    except KeyboardInterrupt:
        # Clean exit on Ctrl+C without traceback
        console.print("\n[yellow]Interrupted.[/]")
        sys.exit(130)  # Standard exit code for SIGINT


def _main() -> None:
    """Internal main function containing the CLI logic."""
    check_for_cli_updates()

    parser = argparse.ArgumentParser(
        description="[bold cyan]Weco CLI[/]\nEnhance your code with AI-driven optimization.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Global flags
    parser.add_argument(
        "--via-skill",
        action="store_true",
        help=argparse.SUPPRESS,  # Hidden flag for AI agents invoking via skill
    )

    subparsers = parser.add_subparsers(
        dest="command", help="Available commands"
    )  # Removed required=True for now to handle chatbot case easily

    # --- Run Command Parser Setup ---
    run_parser = subparsers.add_parser(
        "run", help="Run code optimization", formatter_class=argparse.RawTextHelpFormatter, allow_abbrev=False
    )
    configure_run_parser(run_parser)  # Use the helper to add arguments

    # --- Login Command Parser Setup ---
    _ = subparsers.add_parser("login", help="Log in to Weco and save your API key.")

    # --- Logout Command Parser Setup ---
    _ = subparsers.add_parser("logout", help="Log out from Weco and clear saved API key.")

    # --- Credits Command Parser Setup ---
    credits_parser = subparsers.add_parser("credits", help="Manage your Weco credits")
    configure_credits_parser(credits_parser)  # Use the helper to add subcommands and arguments

    # --- Resume Command Parser Setup ---
    resume_parser = subparsers.add_parser(
        "resume",
        help="Resume an interrupted optimization run",
        formatter_class=argparse.RawTextHelpFormatter,
        allow_abbrev=False,
    )
    configure_resume_parser(resume_parser)

    # --- Setup Command Parser Setup ---
    setup_parser = subparsers.add_parser("setup", help="Set up Weco for use with AI tools")
    configure_setup_parser(setup_parser)

    args = parser.parse_args()

    # Create event context with via_skill flag
    via_skill = getattr(args, "via_skill", False)
    ctx = create_event_context(via_skill=via_skill)
    set_event_context(ctx)

    # Send CLI invocation event
    send_event(CLIInvokedEvent(command=args.command or "help"), ctx)

    if args.command == "login":
        # Check if already logged in
        existing_key = load_weco_api_key()
        if existing_key:
            console.print("[bold green]You are already logged in.[/]")
            console.print("[dim]Use 'weco logout' to log out first if you want to switch accounts.[/]")
            sys.exit(0)

        # Perform the login flow
        if perform_login(console):
            sys.exit(0)
        else:
            sys.exit(1)
    elif args.command == "logout":
        clear_api_key()
        sys.exit(0)
    elif args.command == "run":
        execute_run_command(args)
    elif args.command == "credits":
        from .credits import handle_credits_command

        handle_credits_command(args, console)
        sys.exit(0)
    elif args.command == "resume":
        execute_resume_command(args)
    elif args.command == "setup":
        from .setup import handle_setup_command

        handle_setup_command(args, console)
        sys.exit(0)
    else:
        # This case should be hit if 'weco' is run alone and chatbot logic didn't catch it,
        # or if an invalid command is provided.
        parser.print_help()  # Default action if no command given and not chatbot.
        sys.exit(1)
