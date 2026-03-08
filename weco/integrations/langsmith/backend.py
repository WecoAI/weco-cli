"""LangSmith eval backend for the Weco CLI.

Provides the three functions required by the eval backend interface:
  - register_args(parser)     -- adds --langsmith-* flags
  - validate_args(args)       -- checks required flags, fills defaults
  - build_eval_command(args)  -- builds `python -m weco.integrations.langsmith ...`
"""

import argparse
import sys

from rich.console import Console

console = Console()


def register_args(parser: argparse.ArgumentParser) -> None:
    """Add LangSmith-specific arguments to the run parser."""
    parser.add_argument(
        "--langsmith-dataset", type=str, default=None, help="LangSmith dataset name or ID (requires --eval-backend langsmith)."
    )
    parser.add_argument(
        "--langsmith-target",
        type=str,
        default=None,
        help="Target function as module:function, e.g. 'agent:run_chain' (requires --eval-backend langsmith).",
    )
    parser.add_argument(
        "--langsmith-evaluators",
        nargs="+",
        type=str,
        default=None,
        help="LangSmith evaluator names (requires --eval-backend langsmith). Can be built-in names or module:function specs.",
    )
    parser.add_argument(
        "--langsmith-experiment-prefix",
        type=str,
        default=None,
        help="Prefix for LangSmith experiment names. Groups experiments in the LangSmith UI for comparison.",
    )
    parser.add_argument(
        "--langsmith-summary",
        type=str,
        choices=["mean", "median", "min", "max"],
        default="mean",
        help="How to aggregate per-example scores when using LangSmith (default: mean).",
    )
    parser.add_argument(
        "--langsmith-max-examples",
        type=int,
        default=None,
        help="Evaluate only N examples from the LangSmith dataset (faster iteration).",
    )
    parser.add_argument(
        "--langsmith-max-concurrency",
        type=int,
        default=None,
        help="Number of parallel evaluation threads for LangSmith evaluation.",
    )
    parser.add_argument(
        "--langsmith-target-adapter",
        type=str,
        choices=["raw", "langchain", "single-input"],
        default="raw",
        help="How to adapt the target function for LangSmith's evaluate(). "
        "'raw': pass through (default), 'langchain': call .invoke(), "
        "'single-input': extract a single string input.",
    )
    parser.add_argument(
        "--langsmith-dashboard-evaluators",
        nargs="+",
        type=str,
        default=None,
        help="Names of dashboard-bound evaluators configured in the LangSmith UI. "
        "Enables polling for their async scores after evaluation completes.",
    )
    parser.add_argument(
        "--langsmith-metric-function",
        type=str,
        default=None,
        help="Custom function to compute the final metric from evaluator scores, "
        "as module:function (e.g. 'scoring:combine'). The function receives a dict "
        "of {evaluator_name: aggregated_score} and returns a single float.",
    )
    parser.add_argument(
        "--langsmith-dashboard-evaluator-timeout",
        type=int,
        default=900,
        help="Seconds to poll for dashboard evaluator scores (default: 900). "
        "Only used when --langsmith-dashboard-evaluators is set. "
        "Polls every 10s, stops early when scores stabilize.",
    )


def validate_args(args: argparse.Namespace) -> None:
    """Validate LangSmith arguments. Exits on error. May mutate args for defaults."""
    missing_required = not args.langsmith_dataset or not args.langsmith_target

    if missing_required:
        if sys.stdin.isatty():
            from .wizard import run_wizard

            run_wizard(args)
        else:
            if not args.langsmith_dataset:
                console.print("[bold red]Error: --langsmith-dataset is required with --eval-backend langsmith[/]")
            if not args.langsmith_target:
                console.print("[bold red]Error: --langsmith-target is required with --eval-backend langsmith[/]")
            sys.exit(1)

    has_dashboard = bool(args.langsmith_dashboard_evaluators)

    # Default evaluators
    if not args.langsmith_evaluators:
        if has_dashboard:
            # Dashboard-only mode: no code evaluators needed
            args.langsmith_evaluators = []
        else:
            # Default to using the metric name as the evaluator
            args.langsmith_evaluators = [args.metric]

    if not args.langsmith_evaluators and not has_dashboard:
        console.print("[bold red]Error: provide --langsmith-evaluators or --langsmith-dashboard-evaluators[/]")
        sys.exit(1)


def build_eval_command(args: argparse.Namespace) -> str:
    """Build the eval command string for LangSmith backend."""
    parts = ["python", "-m", "weco.integrations.langsmith"]
    parts.extend(["--dataset", args.langsmith_dataset])
    parts.extend(["--target", args.langsmith_target])
    if args.langsmith_evaluators:
        parts.append("--evaluators")
        parts.extend(args.langsmith_evaluators)
    parts.extend(["--metric", args.metric])
    if args.langsmith_summary != "mean":
        parts.extend(["--summary", args.langsmith_summary])
    if args.langsmith_experiment_prefix:
        parts.extend(["--experiment-prefix", args.langsmith_experiment_prefix])
    if args.langsmith_max_concurrency:
        parts.extend(["--max-concurrency", str(args.langsmith_max_concurrency)])
    if args.langsmith_max_examples:
        parts.extend(["--max-examples", str(args.langsmith_max_examples)])
    if args.langsmith_target_adapter != "raw":
        parts.extend(["--target-adapter", args.langsmith_target_adapter])
    if args.langsmith_metric_function:
        parts.extend(["--metric-function", args.langsmith_metric_function])
    if args.langsmith_dashboard_evaluators:
        parts.append("--dashboard-evaluators")
        parts.extend(args.langsmith_dashboard_evaluators)
        if args.langsmith_dashboard_evaluator_timeout and args.langsmith_dashboard_evaluator_timeout > 0:
            parts.extend(["--dashboard-evaluator-timeout", str(args.langsmith_dashboard_evaluator_timeout)])
    return " ".join(parts)
