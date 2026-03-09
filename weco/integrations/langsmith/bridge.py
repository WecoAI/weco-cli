"""LangSmith evaluation bridge for Weco.

Runs a target function against a LangSmith dataset with specified evaluators,
and prints the aggregated metric values in Weco's expected format (key: value).

Usage as eval command:
    weco run --source agent.py \\
        --eval-command "python -m weco.integrations.langsmith \\
            --dataset my-dataset \\
            --target my_module:my_function \\
            --evaluators accuracy relevance" \\
        --metric accuracy --goal maximize

The bridge:
1. Dynamically imports the target function from the user's source module
2. Loads the LangSmith dataset
3. Runs client.evaluate() with the specified evaluators
4. Aggregates results into metric values
5. Prints all metrics as "key: value" lines to stdout
"""

import argparse
import importlib
import os
import re
import sys
import time
from typing import Callable


def _sanitize_error(msg: str) -> str:
    """Remove potential API keys or tokens from error messages."""
    return re.sub(r"(ls[a-z]*-[A-Za-z0-9_-]{20,})", "***", str(msg))


def import_target(spec: str) -> Callable:
    """Import a callable from a 'module:function' specification.

    Supports dotted module paths (e.g. 'my_app.agent:run_chain').
    Adds cwd to sys.path so local modules resolve.

    Works correctly with Weco's file-swap approach because each evaluation
    runs as a fresh subprocess — the modified source file on disk is what
    gets imported. No cache-busting needed.
    """
    if ":" not in spec:
        raise ValueError(f"Target must be 'module:function' (e.g. 'agent:run_chain'), got '{spec}'")

    module_path, func_name = spec.rsplit(":", 1)

    if not module_path or not func_name:
        raise ValueError(f"Both module and function must be non-empty in '{spec}'")

    # Ensure cwd is importable (Weco runs eval from project root)
    if "" not in sys.path:
        sys.path.insert(0, "")

    module = importlib.import_module(module_path)
    func = getattr(module, func_name)

    if not callable(func):
        raise TypeError(f"'{spec}' resolved to {type(func).__name__}, expected a callable")

    return func


def _try_langsmith_builtin(name: str):
    """Try to find a built-in LangSmith evaluator by name. Returns None if not found."""
    try:
        from langsmith import evaluation as ls_eval

        evaluator = getattr(ls_eval, name, None)
        if evaluator and callable(evaluator):
            return evaluator
    except ImportError:
        pass
    return None


def resolve_evaluators(names: list) -> list:
    """Resolve evaluator names to callable evaluator functions.

    Resolution order for each name:
    1. If contains ':', treat as module:function spec
    2. Try as a built-in LangSmith evaluator
    3. Try importing from evaluators:<name> (conventional local file)
    """
    resolved = []
    for name in names:
        if ":" in name:
            resolved.append(import_target(name))
        else:
            builtin = _try_langsmith_builtin(name)
            if builtin is not None:
                resolved.append(builtin)
            else:
                try:
                    resolved.append(import_target(f"evaluators:{name}"))
                except (ImportError, AttributeError) as e:
                    raise ValueError(
                        f"Could not resolve evaluator '{name}'. Not a LangSmith built-in, and not found in evaluators.py: {e}"
                    ) from e
    return resolved


def _aggregate(values: list, mode: str) -> float:
    """Aggregate a list of numeric scores using the specified mode."""
    if not values:
        return 0.0
    if mode == "mean":
        return sum(values) / len(values)
    elif mode == "median":
        sorted_vals = sorted(values)
        mid = len(sorted_vals) // 2
        if len(sorted_vals) % 2 == 0:
            return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2
        return sorted_vals[mid]
    elif mode == "min":
        return min(values)
    elif mode == "max":
        return max(values)
    return sum(values) / len(values)


def _adapt_target(target: Callable, adapter: str) -> Callable:
    """Wrap the target function to match LangSmith's evaluate() expected signature.

    LangSmith's evaluate() passes each dataset example's inputs dict to the target.
    The target should return a dict of outputs.
    """
    if adapter == "raw":
        return target
    elif adapter == "langchain":

        def langchain_wrapper(inputs: dict) -> dict:
            result = target.invoke(inputs)
            if isinstance(result, dict):
                return result
            return {"output": result}

        return langchain_wrapper
    elif adapter == "single-input":

        def single_input_wrapper(inputs: dict) -> dict:
            text = inputs.get("input") or inputs.get("question") or inputs.get("text") or inputs.get("query", "")
            result = target(text)
            if isinstance(result, dict):
                return result
            return {"output": result}

        return single_input_wrapper

    return target


def _poll_dashboard_scores(
    client, run_ids: list, known_keys: set, expected_keys: set, timeout: int = 900, poll_interval: int = 10
) -> dict:
    """Poll LangSmith for scores from dashboard-bound evaluators.

    After evaluate() completes, dashboard-bound evaluators run asynchronously
    server-side. This function polls until every expected evaluator has
    produced a score for every run, or the timeout is reached.

    Args:
        client: LangSmith Client instance.
        run_ids: List of run IDs to poll feedback for.
        known_keys: Set of metric keys already captured from code-passed evaluators.
        expected_keys: Set of dashboard evaluator keys we expect to receive.
        timeout: Max seconds to wait for dashboard evaluator results.
        poll_interval: Seconds between polls.

    Returns:
        Dict mapping new metric keys to lists of scores.
    """
    if not run_ids:
        return {}

    num_runs = len(run_ids)
    new_scores = {}
    deadline = time.time() + timeout

    while time.time() < deadline:
        new_scores = {}
        feedback_list = list(client.list_feedback(run_ids=run_ids))

        for fb in feedback_list:
            if fb.key not in known_keys and fb.score is not None:
                new_scores.setdefault(fb.key, []).append(float(fb.score))

        # Stop when every expected evaluator has scored every run
        if all(len(new_scores.get(key, [])) >= num_runs for key in expected_keys):
            break

        time.sleep(poll_interval)

    return new_scores


def run_langsmith_eval(
    dataset_name: str,
    target: Callable,
    evaluator_names: list,
    metric_name: str,
    experiment_prefix: str = None,
    summary_mode: str = "mean",
    max_concurrency: int = None,
    max_examples: int = None,
    splits: list = None,
    dashboard_evaluators: list = None,
    dashboard_evaluator_timeout: int = 0,
    metric_function: Callable = None,
) -> dict:
    """Run LangSmith evaluation and return aggregated scores for all metrics.

    Args:
        dataset_name: LangSmith dataset name or ID.
        target: Callable that takes dict inputs, returns dict outputs.
        evaluator_names: List of evaluator names to resolve and apply.
        metric_name: Primary metric name (for diagnostics; all metrics are returned).
        experiment_prefix: Groups experiments in LangSmith UI for comparison.
        summary_mode: How to aggregate per-example scores (mean/median/min/max).
        max_concurrency: Number of parallel evaluation threads.
        max_examples: Limit evaluation to N examples from the dataset.
        splits: Filter to specific dataset splits (e.g. ['train', 'test']).
        dashboard_evaluators: Names of expected dashboard-bound evaluators.
            When set, enables polling with a default timeout of 900s.
        dashboard_evaluator_timeout: Seconds to poll for dashboard-bound evaluator scores
            after the main evaluation completes. 0 = don't poll (default).
        metric_function: Optional callable that receives {evaluator: aggregated_score}
            and returns a single float. Result is stored under metric_name.

    Returns:
        Dict mapping metric names to aggregated float scores.
    """
    try:
        from langsmith import Client
    except ImportError:
        print("ERROR: langsmith package not installed.")
        print("Install with: pip install weco[langsmith]")
        sys.exit(1)

    client = Client()
    evaluators = resolve_evaluators(evaluator_names)

    # When splits are specified, filter examples by split
    if splits:
        data = client.list_examples(dataset_name=dataset_name, splits=splits)
    else:
        data = dataset_name

    # target is positional-only in Client.evaluate(), so pass it separately
    eval_kwargs = {"data": data, "evaluators": evaluators, "experiment_prefix": experiment_prefix or f"weco-{dataset_name}"}
    if max_concurrency is not None:
        eval_kwargs["max_concurrency"] = max_concurrency

    results = client.evaluate(target, **eval_kwargs)

    # Collect all metric scores and run IDs across examples
    all_scores = {}
    run_ids = []
    example_count = 0

    for result in results:
        example_count += 1
        if max_examples and example_count > max_examples:
            break

        # Capture run ID for dashboard evaluator polling
        run = result.get("run")
        if run is not None and hasattr(run, "id"):
            run_ids.append(run.id)

        eval_results = result.get("evaluation_results", {})
        for eval_result in eval_results.get("results", []):
            key = eval_result.key
            if eval_result.score is not None:
                all_scores.setdefault(key, []).append(float(eval_result.score))

    # Optionally poll for dashboard-bound evaluator scores
    expected_dashboard = set(dashboard_evaluators or [])
    if expected_dashboard and dashboard_evaluator_timeout > 0:
        known_keys = set(all_scores.keys())
        dashboard_scores = _poll_dashboard_scores(
            client, run_ids, known_keys, expected_keys=expected_dashboard, timeout=dashboard_evaluator_timeout
        )
        found_keys = set(dashboard_scores.keys())
        missing = expected_dashboard - found_keys - known_keys
        if missing:
            print(
                f"WARNING: Dashboard evaluators not found within {dashboard_evaluator_timeout}s: "
                f"{', '.join(sorted(missing))}. Continuing with available scores.",
                file=sys.stderr,
            )
        elif expected_dashboard and not dashboard_scores:
            print(
                f"WARNING: No dashboard evaluator scores found within {dashboard_evaluator_timeout}s. "
                "Continuing with code-passed evaluator scores only.",
                file=sys.stderr,
            )
        for key, values in dashboard_scores.items():
            all_scores.setdefault(key, []).extend(values)

    # Aggregate each metric
    aggregated = {}
    for key, values in all_scores.items():
        aggregated[key] = _aggregate(values, summary_mode)

    # Apply custom metric function if provided
    if metric_function is not None:
        aggregated[metric_name] = metric_function(aggregated)

    return aggregated


def main():
    parser = argparse.ArgumentParser(
        description="LangSmith evaluation bridge for Weco",
        epilog="Runs a target function against a LangSmith dataset and prints metrics in Weco's expected format.",
    )
    parser.add_argument("--dataset", required=True, help="LangSmith dataset name or ID")
    parser.add_argument("--target", required=True, help="Target function as module:function (e.g., agent:run_chain)")
    parser.add_argument(
        "--evaluators",
        nargs="+",
        default=[],
        help="Evaluator names: built-in LangSmith names, or module:function specs. "
        "Optional when using --dashboard-evaluator-timeout with dashboard-bound evaluators.",
    )
    parser.add_argument("--metric", required=True, help="Primary metric name for Weco to optimize")
    parser.add_argument(
        "--summary",
        default="mean",
        choices=["mean", "median", "min", "max"],
        help="How to aggregate per-example scores (default: mean)",
    )
    parser.add_argument(
        "--experiment-prefix", default=None, help="Prefix for LangSmith experiment names (groups experiments in UI)"
    )
    parser.add_argument("--max-concurrency", type=int, default=None, help="Number of parallel evaluation threads")
    parser.add_argument(
        "--max-examples", type=int, default=None, help="Evaluate only N examples from the dataset (faster iteration)"
    )
    parser.add_argument(
        "--target-adapter",
        default="raw",
        choices=["raw", "langchain", "single-input"],
        help="How to adapt the target function signature for LangSmith's evaluate(). "
        "'raw': pass through (default), 'langchain': call .invoke(), "
        "'single-input': extract a single string input from the dict",
    )
    parser.add_argument(
        "--dashboard-evaluators",
        nargs="+",
        default=[],
        help="Names of dashboard-bound evaluators configured in the LangSmith UI. "
        "Enables polling for their async scores after evaluation completes.",
    )
    parser.add_argument(
        "--dashboard-evaluator-timeout",
        type=int,
        default=900,
        help="Seconds to poll for dashboard evaluator scores (default: 900). "
        "Only used when --dashboard-evaluators is set. Polls every 10s.",
    )
    parser.add_argument(
        "--splits", nargs="+", default=None, help="Evaluate only examples in these dataset splits (e.g. 'train', 'test')."
    )
    parser.add_argument(
        "--metric-function",
        default=None,
        help="Custom aggregation function as module:function. Receives dict of "
        "{evaluator: aggregated_score}, returns a single float used as the metric value.",
    )

    args = parser.parse_args()

    # Validate environment
    if not os.environ.get("LANGCHAIN_API_KEY"):
        print("ERROR: LANGCHAIN_API_KEY environment variable not set.")
        print("Get your API key at https://smith.langchain.com/settings")
        sys.exit(1)

    # Import target function — may fail if Weco generated a bad code variant.
    # The error output is useful feedback for the optimization backend.
    try:
        target = import_target(args.target)
    except (ImportError, SyntaxError, AttributeError) as e:
        print(f"Import error: {_sanitize_error(e)}")
        print("The generated code variant has issues preventing evaluation.")
        sys.exit(1)

    # Apply adapter
    target = _adapt_target(target, args.target_adapter)

    # Import custom metric function if specified
    metric_fn = None
    if args.metric_function:
        try:
            metric_fn = import_target(args.metric_function)
        except (ImportError, SyntaxError, AttributeError) as e:
            print(f"Metric function import error: {_sanitize_error(e)}")
            sys.exit(1)

    # Run evaluation
    try:
        metrics = run_langsmith_eval(
            dataset_name=args.dataset,
            target=target,
            evaluator_names=args.evaluators,
            metric_name=args.metric,
            experiment_prefix=args.experiment_prefix,
            summary_mode=args.summary,
            max_concurrency=args.max_concurrency,
            max_examples=args.max_examples,
            splits=args.splits,
            dashboard_evaluators=args.dashboard_evaluators,
            dashboard_evaluator_timeout=args.dashboard_evaluator_timeout or 0,
            metric_function=metric_fn,
        )
    except Exception as e:
        print(f"LangSmith evaluation error: {_sanitize_error(e)}")
        sys.exit(1)

    if not metrics:
        if args.dashboard_evaluators:
            print("WARNING: No metrics returned. Dashboard evaluators may need more time.")
            print(f"Try increasing --dashboard-evaluator-timeout (currently {args.dashboard_evaluator_timeout}s).")
        else:
            print(f"WARNING: No metrics returned for evaluators {args.evaluators}.")
            print("Check that evaluator names are correct and the dataset is non-empty.")
        sys.exit(1)

    # Print ALL metrics — Weco optimizes --metric but benefits from seeing everything
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
