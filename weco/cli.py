import argparse
import sys
import pathlib
from rich.console import Console
from rich.traceback import install

from .auth import clear_api_key
from .utils import check_for_cli_updates

install(show_locals=True)
console = Console()


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
        help="Model to use for optimization. Defaults to `o4-mini` when `OPENAI_API_KEY` is set, `claude-sonnet-4-0` when `ANTHROPIC_API_KEY` is set, and `gemini-2.5-pro` when `GEMINI_API_KEY` is set. When multiple keys are set, the priority is `OPENAI_API_KEY` > `ANTHROPIC_API_KEY` > `GEMINI_API_KEY`.",
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


def execute_run_command(args: argparse.Namespace) -> None:
    """Execute the 'weco run' command with all its logic."""
    from .optimizer import execute_optimization

    success = execute_optimization(
        source=args.source,
        eval_command=args.eval_command,
        metric=args.metric,
        goal=args.goal,
        steps=args.steps,
        model=args.model,
        log_dir=args.log_dir,
        additional_instructions=args.additional_instructions,
        console=console,
        eval_timeout=args.eval_timeout,
        save_logs=args.save_logs,
    )
    exit_code = 0 if success else 1
    sys.exit(exit_code)


def main() -> None:
    """Main function for the Weco CLI."""
    check_for_cli_updates()

    parser = argparse.ArgumentParser(
        description="[bold cyan]Weco CLI[/]\nEnhance your code with AI-driven optimization.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Add global model argument
    parser.add_argument(
        "-M",
        "--model",
        type=str,
        default=None,
        help="Model to use for optimization. Defaults to `o4-mini` when `OPENAI_API_KEY` is set, `claude-sonnet-4-0` when `ANTHROPIC_API_KEY` is set, and `gemini-2.5-pro` when `GEMINI_API_KEY` is set. When multiple keys are set, the priority is `OPENAI_API_KEY` > `ANTHROPIC_API_KEY` > `GEMINI_API_KEY`.",
    )

    subparsers = parser.add_subparsers(
        dest="command", help="Available commands"
    )  # Removed required=True for now to handle chatbot case easily

    # --- Run Command Parser Setup ---
    run_parser = subparsers.add_parser(
        "run", help="Run code optimization", formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False
    )
    configure_run_parser(run_parser)  # Use the helper to add arguments

    # --- Logout Command Parser Setup ---
    _ = subparsers.add_parser("logout", help="Log out from Weco and clear saved API key.")

    # Check if we should run the chatbot
    # This logic needs to be robust. If 'run' or 'logout' is present, or -h/--help, don't run chatbot.
    # Otherwise, if it's just 'weco' or 'weco <path>' (with optional --model), run chatbot.

    def should_run_chatbot(args_list):
        """Determine if we should run chatbot by filtering out model arguments."""
        filtered = []
        i = 0
        while i < len(args_list):
            if args_list[i] in ["-M", "--model"]:
                # Skip the model argument and its value (if it exists)
                i += 1  # Skip the model flag
                if i < len(args_list):  # Skip the model value if it exists
                    i += 1
            elif args_list[i].startswith("--model="):
                i += 1  # Skip --model=value format
            else:
                filtered.append(args_list[i])
                i += 1

        # Apply existing chatbot detection logic to filtered args
        return len(filtered) == 0 or (len(filtered) == 1 and not filtered[0].startswith("-"))

    # Check for known commands by looking at the first non-option argument
    def get_first_non_option_arg():
        for arg in sys.argv[1:]:
            if not arg.startswith("-"):
                return arg
        return None

    first_non_option = get_first_non_option_arg()
    is_known_command = first_non_option in ["run", "logout"]
    is_help_command = len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help"]  # Check for global help

    should_run_chatbot_result = should_run_chatbot(sys.argv[1:])
    should_run_chatbot_flag = not is_known_command and not is_help_command and should_run_chatbot_result

    if should_run_chatbot_flag:
        from .chatbot import run_onboarding_chatbot  # Moved import inside

        # Create a simple parser just for extracting the model argument
        model_parser = argparse.ArgumentParser(add_help=False)
        model_parser.add_argument("-M", "--model", type=str, default=None)

        # Parse args to extract model
        args, unknown = model_parser.parse_known_args()

        # Determine project path from remaining arguments
        filtered_args = []
        i = 1
        while i < len(sys.argv):
            if sys.argv[i] in ["-M", "--model"]:
                # Skip the model argument and its value (if it exists)
                i += 1  # Skip the model flag
                if i < len(sys.argv):  # Skip the model value if it exists
                    i += 1
            elif sys.argv[i].startswith("--model="):
                i += 1  # Skip --model=value format
            else:
                filtered_args.append(sys.argv[i])
                i += 1

        project_path = pathlib.Path(filtered_args[0]) if filtered_args else pathlib.Path.cwd()
        if not project_path.is_dir():
            console.print(
                f"[bold red]Error:[/] The path '{project_path}' is not a valid directory. Please provide a valid directory path."
            )
            sys.exit(1)

        # Pass the run_parser and model to the chatbot
        run_onboarding_chatbot(project_path, console, run_parser, model=args.model)
        sys.exit(0)

    # If not running chatbot, proceed with normal arg parsing
    # If we reached here, a command (run, logout) or help is expected.
    args = parser.parse_args()

    if args.command == "logout":
        clear_api_key()
        sys.exit(0)
    elif args.command == "run":
        execute_run_command(args)
    else:
        # This case should be hit if 'weco' is run alone and chatbot logic didn't catch it,
        # or if an invalid command is provided.
        parser.print_help()  # Default action if no command given and not chatbot.
        sys.exit(1)
