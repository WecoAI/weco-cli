"""LangFuse eval backend for the Weco CLI.

Provides the three functions required by the eval backend interface:
  - register_args(parser)     -- adds --langfuse-* flags
  - validate_args(args)       -- checks required flags, fills defaults
  - build_eval_command(args)  -- builds `python -m weco.integrations.langfuse ...`
"""

import argparse
import shlex
import sys
from pathlib import Path

from rich.console import Console

console = Console()


def register_args(parser: argparse.ArgumentParser) -> None:
    """Add LangFuse-specific arguments to the run parser."""
    parser.add_argument(
        "--langfuse-dataset", type=str, default=None, help="LangFuse dataset name (requires --eval-backend langfuse)."
    )
    parser.add_argument(
        "--langfuse-target",
        type=str,
        default=None,
        help="Target function as module:function, e.g. 'agent:run_chain' (requires --eval-backend langfuse).",
    )
    parser.add_argument(
        "--langfuse-evaluators",
        nargs="+",
        type=str,
        default=None,
        help="LangFuse evaluator names (requires --eval-backend langfuse). Can be module:function specs.",
    )
    parser.add_argument(
        "--langfuse-experiment-name",
        type=str,
        default=None,
        help="Experiment name in the LangFuse UI. Groups experiment runs for comparison.",
    )
    parser.add_argument(
        "--langfuse-summary",
        type=str,
        choices=["mean", "median", "min", "max"],
        default="mean",
        help="How to aggregate per-example scores when using LangFuse (default: mean).",
    )
    parser.add_argument(
        "--langfuse-max-concurrency",
        type=int,
        default=None,
        help="Number of parallel evaluation threads for LangFuse evaluation.",
    )
    parser.add_argument(
        "--langfuse-metric-function",
        type=str,
        default=None,
        help="Custom function to compute the final metric from evaluator scores, "
        "as module:function (e.g. 'scoring:combine'). The function receives a dict "
        "of {evaluator_name: aggregated_score} and returns a single float.",
    )
    parser.add_argument(
        "--langfuse-managed-evaluators",
        nargs="+",
        type=str,
        default=None,
        help="Names of LLM-as-a-Judge evaluators configured in the LangFuse UI. "
        "Enables polling for their server-side scores after evaluation completes.",
    )
    parser.add_argument(
        "--langfuse-managed-evaluator-timeout",
        type=int,
        default=900,
        help="Seconds to poll for managed evaluator scores (default: 900). "
        "Only used when --langfuse-managed-evaluators is set. "
        "Polls every 10s, stops early when scores stabilize.",
    )


def validate_args(args: argparse.Namespace) -> None:
    """Validate LangFuse arguments. Exits on error. May mutate args for defaults."""
    try:
        import langfuse  # noqa: F401
    except ImportError:
        console.print("[bold red]Error: langfuse package not installed.[/]")
        console.print("Install with: [bold]pip install 'weco\\[langfuse]'[/]")
        sys.exit(1)

    missing_required = not args.langfuse_dataset or not args.langfuse_target

    if missing_required:
        if sys.stdin.isatty():
            from .wizard import run_wizard

            run_wizard(args)
        else:
            if not args.langfuse_dataset:
                console.print("[bold red]Error: --langfuse-dataset is required with --eval-backend langfuse[/]")
            if not args.langfuse_target:
                console.print("[bold red]Error: --langfuse-target is required with --eval-backend langfuse[/]")
            sys.exit(1)

    has_managed = bool(args.langfuse_managed_evaluators)

    # Default evaluators
    if not args.langfuse_evaluators:
        if has_managed:
            # Managed-only mode: no local evaluators needed
            args.langfuse_evaluators = []
        else:
            # Default to using the metric name as the evaluator
            args.langfuse_evaluators = [args.metric]

    if not args.langfuse_evaluators and not has_managed:
        console.print("[bold red]Error: provide --langfuse-evaluators or --langfuse-managed-evaluators[/]")
        sys.exit(1)


def build_eval_command(args: argparse.Namespace) -> str:
    """Build the eval command string for LangFuse backend.

    All user-provided values are shell-quoted to prevent injection, since the
    resulting string is executed via ``shell=True`` in ``run_evaluation()``.
    """
    q = shlex.quote
    bridge = str(Path(__file__).resolve().parent / "bridge.py")
    parts = ["python", q(bridge)]
    parts.extend(["--dataset", q(args.langfuse_dataset)])
    parts.extend(["--target", q(args.langfuse_target)])
    if args.langfuse_evaluators:
        parts.append("--evaluators")
        parts.extend(q(e) for e in args.langfuse_evaluators)
    parts.extend(["--metric", q(args.metric)])
    if args.langfuse_summary != "mean":
        parts.extend(["--summary", q(args.langfuse_summary)])
    if args.langfuse_experiment_name:
        parts.extend(["--experiment-name", q(args.langfuse_experiment_name)])
    if args.langfuse_max_concurrency:
        parts.extend(["--max-concurrency", str(int(args.langfuse_max_concurrency))])
    if args.langfuse_metric_function:
        parts.extend(["--metric-function", q(args.langfuse_metric_function)])
    if args.langfuse_managed_evaluators:
        parts.append("--managed-evaluators")
        parts.extend(q(e) for e in args.langfuse_managed_evaluators)
        if args.langfuse_managed_evaluator_timeout and args.langfuse_managed_evaluator_timeout > 0:
            parts.extend(["--managed-evaluator-timeout", str(int(args.langfuse_managed_evaluator_timeout))])
    return " ".join(parts)
