"""Browser-based setup wizard for the LangSmith eval backend."""

import argparse
import os
import sys
import threading
from pathlib import Path

from rich.console import Console

from weco.core.browser import open_browser

from .server import WizardHandler, WizardServer

console = Console()


def run_wizard(args: argparse.Namespace) -> None:
    """Launch browser-based setup wizard. Blocks until user submits config.

    Spins up a local HTTP server serving a single-page wizard UI that guides
    the user through LangSmith configuration (API key, dataset, target, evaluators).
    Mutates ``args`` in place with the submitted configuration.
    """
    # Shared state between server thread and main thread
    done_event = threading.Event()
    config_result: dict = {}

    initial_state = {
        "api_key_set": bool(os.environ.get("LANGCHAIN_API_KEY")),
        # Core params
        "metric": getattr(args, "metric", ""),
        "goal": getattr(args, "goal", ""),
        "source": getattr(args, "source", None),
        "sources": getattr(args, "sources", None),
        "steps": getattr(args, "steps", 100),
        "model": getattr(args, "model", None),
        "log_dir": getattr(args, "log_dir", ".runs"),
        "additional_instructions": getattr(args, "additional_instructions", None),
        "eval_timeout": getattr(args, "eval_timeout", None),
        "save_logs": getattr(args, "save_logs", False),
        "apply_change": getattr(args, "apply_change", False),
        "require_review": getattr(args, "require_review", False),
        # LangSmith params
        "langsmith_summary": getattr(args, "langsmith_summary", "mean"),
        "langsmith_experiment_prefix": getattr(args, "langsmith_experiment_prefix", None),
        "langsmith_max_examples": getattr(args, "langsmith_max_examples", None),
        "langsmith_max_concurrency": getattr(args, "langsmith_max_concurrency", None),
        "langsmith_dashboard_evaluator_timeout": getattr(args, "langsmith_dashboard_evaluator_timeout", 900),
        "langsmith_splits": getattr(args, "langsmith_splits", None),
    }

    html_path = Path(__file__).parent / "page.html"

    server = WizardServer(
        ("127.0.0.1", 0),
        WizardHandler,
        done_event=done_event,
        config_result=config_result,
        initial_state=initial_state,
        html_path=html_path,
    )
    port = server.server_address[1]
    url = f"http://127.0.0.1:{port}"

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    console.print("\n[bold cyan]LangSmith Setup Wizard[/]")
    console.print(f"[dim]Open in your browser:[/] {url}")

    if not open_browser(url):
        console.print("[yellow]Could not open browser automatically. Please open the URL above.[/]")

    console.print("[dim]Waiting for configuration... (Ctrl+C to cancel)[/]\n")

    try:
        done_event.wait()
    except KeyboardInterrupt:
        console.print("\n[yellow]Wizard cancelled.[/]")
        server.shutdown()
        sys.exit(1)

    server.shutdown()
    server_thread.join(timeout=2)

    if not config_result:
        console.print("[bold red]No configuration received from wizard.[/]")
        sys.exit(1)

    # Mutate args with wizard choices

    # Source files
    source_files = config_result.get("source_files")
    if source_files:
        if len(source_files) == 1:
            args.source = source_files[0]
            args.sources = None
        else:
            args.source = None
            args.sources = source_files

    # LangSmith required args
    for required in ("dataset", "target"):
        if required not in config_result:
            console.print(f"[bold red]Wizard config missing required field: {required}[/]")
            sys.exit(1)

    args.langsmith_dataset = config_result["dataset"]
    args.langsmith_target = config_result["target"]
    args.langsmith_target_adapter = config_result.get("adapter", "raw")

    code_evaluators = config_result.get("code_evaluators", [])
    if code_evaluators:
        args.langsmith_evaluators = code_evaluators

    dashboard_evaluators = config_result.get("dashboard_evaluators", [])
    if dashboard_evaluators:
        args.langsmith_dashboard_evaluators = dashboard_evaluators

    # Core run params
    if config_result.get("metric"):
        args.metric = config_result["metric"]
    if config_result.get("goal"):
        args.goal = config_result["goal"]
    if config_result.get("steps") is not None:
        args.steps = config_result["steps"]
    if config_result.get("model") is not None:
        args.model = config_result["model"]
    if config_result.get("log_dir") is not None:
        args.log_dir = config_result["log_dir"]
    if config_result.get("additional_instructions") is not None:
        args.additional_instructions = config_result["additional_instructions"]
    if config_result.get("eval_timeout") is not None:
        args.eval_timeout = config_result["eval_timeout"]

    # Boolean flags (use "in" not .get() since False is a valid value)
    if "save_logs" in config_result:
        args.save_logs = config_result["save_logs"]
    if "apply_change" in config_result:
        args.apply_change = config_result["apply_change"]
    if "require_review" in config_result:
        args.require_review = config_result["require_review"]

    # LangSmith optional params
    if config_result.get("langsmith_summary") is not None:
        args.langsmith_summary = config_result["langsmith_summary"]
    if config_result.get("langsmith_experiment_prefix") is not None:
        args.langsmith_experiment_prefix = config_result["langsmith_experiment_prefix"]
    if config_result.get("langsmith_max_examples") is not None:
        args.langsmith_max_examples = config_result["langsmith_max_examples"]
    if config_result.get("langsmith_max_concurrency") is not None:
        args.langsmith_max_concurrency = config_result["langsmith_max_concurrency"]
    if config_result.get("langsmith_dashboard_evaluator_timeout") is not None:
        args.langsmith_dashboard_evaluator_timeout = config_result["langsmith_dashboard_evaluator_timeout"]
    if config_result.get("splits"):
        args.langsmith_splits = config_result["splits"]
    if config_result.get("metric_function"):
        args.langsmith_metric_function = config_result["metric_function"]

    console.print("[bold green]Configuration received![/]\n")
