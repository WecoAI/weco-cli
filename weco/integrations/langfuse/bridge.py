"""LangFuse evaluation bridge for Weco.

Runs a target function against a LangFuse dataset with specified evaluators,
and prints the aggregated metric values in Weco's expected format (key: value).

Usage as eval command:
    weco run --source agent.py \\
        --eval-command "python -m weco.integrations.langfuse \\
            --dataset my-dataset \\
            --target my_module:my_function \\
            --evaluators accuracy relevance" \\
        --metric accuracy --goal maximize

The bridge:
1. Dynamically imports the target function from the user's source module
2. Loads the LangFuse dataset
3. Runs dataset.run_experiment() with the specified evaluators
4. Optionally polls for managed (LLM-as-a-Judge) evaluator scores
5. Aggregates results into metric values
6. Prints all metrics as "key: value" lines to stdout
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
    return re.sub(r"((?:sk|pk)-lf-[A-Za-z0-9_-]{20,})", "***", str(msg))


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


def resolve_evaluators(names: list) -> list:
    """Resolve evaluator names to callable evaluator functions.

    Resolution order for each name:
    1. If contains ':', treat as module:function spec
    2. Try importing from evaluators:<name> (conventional local file)
    """
    resolved = []
    for name in names:
        if ":" in name:
            resolved.append(import_target(name))
        else:
            try:
                resolved.append(import_target(f"evaluators:{name}"))
            except (ImportError, AttributeError) as e:
                raise ValueError(f"Could not resolve evaluator '{name}'. Not found in evaluators.py: {e}") from e
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


def _wrap_target(target: Callable) -> Callable:
    """Wrap user's target function for LangFuse's task signature.

    LangFuse's run_experiment() passes each dataset item to the task as a
    keyword argument. The item has an `input` attribute (for hosted datasets)
    or is a plain dict (for local data). The user's target expects a plain
    dict of inputs and returns a dict of outputs.
    """

    def task_wrapper(*, item, **kwargs):
        # Hosted DatasetItem has .input attribute; local items are dicts
        if hasattr(item, "input"):
            inputs = item.input
        elif isinstance(item, dict):
            inputs = item.get("input", item)
        else:
            inputs = item
        return target(inputs)

    return task_wrapper


def _normalize_evaluator(evaluator: Callable) -> Callable:
    """Wrap an evaluator to ensure it returns Evaluation objects.

    Accepts evaluators that return either:
    - A LangFuse Evaluation object (passed through)
    - A dict with 'name'/'key' and 'value'/'score' keys (converted)
    """

    def wrapper(*, input, output, expected_output=None, metadata=None, **kwargs):
        result = evaluator(input=input, output=output, expected_output=expected_output, metadata=metadata, **kwargs)

        if isinstance(result, dict):
            from langfuse import Evaluation

            return Evaluation(
                name=result.get("key", result.get("name", evaluator.__name__)),
                value=result.get("score", result.get("value", 0.0)),
                comment=result.get("comment"),
            )
        return result

    return wrapper


def _poll_managed_scores(
    client, trace_ids: list, known_keys: set, expected_keys: set, timeout: int = 900, poll_interval: int = 10
) -> dict:
    """Poll LangFuse for scores from managed (LLM-as-a-Judge) evaluators.

    After run_experiment() completes, managed evaluators configured in the
    LangFuse UI run asynchronously server-side on the experiment traces.
    This function polls until every expected evaluator has produced a score
    for every trace, or the timeout is reached.

    Args:
        client: LangFuse client instance.
        trace_ids: List of trace IDs to poll scores for.
        known_keys: Set of metric keys already captured from local evaluators.
        expected_keys: Set of managed evaluator keys we expect to receive.
        timeout: Max seconds to wait for managed evaluator results.
        poll_interval: Seconds between polls.

    Returns:
        Dict mapping new metric keys to lists of scores.
    """
    if not trace_ids:
        return {}

    num_traces = len(trace_ids)
    new_scores = {}
    deadline = time.time() + timeout

    while time.time() < deadline:
        new_scores = {}

        try:
            # Use the scores API to fetch scores for our trace IDs
            for trace_id in trace_ids:
                response = client.api.scores.get_many(trace_id=trace_id)
                for score in response.data:
                    key = getattr(score, "name", None)
                    value = getattr(score, "value", None)
                    if key and key not in known_keys and value is not None:
                        new_scores.setdefault(key, []).append(float(value))
        except Exception:
            pass

        # Stop when every expected evaluator has scored every trace
        if all(len(new_scores.get(key, [])) >= num_traces for key in expected_keys):
            break

        time.sleep(poll_interval)

    return new_scores


def run_langfuse_eval(
    dataset_name: str,
    target: Callable,
    evaluator_names: list,
    metric_name: str,
    experiment_name: str = None,
    summary_mode: str = "mean",
    max_concurrency: int = None,
    managed_evaluators: list = None,
    managed_evaluator_timeout: int = 0,
    metric_function: Callable = None,
) -> dict:
    """Run LangFuse evaluation and return aggregated scores for all metrics.

    Args:
        dataset_name: LangFuse dataset name.
        target: Callable that takes dict inputs, returns dict outputs.
        evaluator_names: List of evaluator names to resolve and apply locally.
        metric_name: Primary metric name (for diagnostics; all metrics are returned).
        experiment_name: Name for the experiment in LangFuse UI.
        summary_mode: How to aggregate per-example scores (mean/median/min/max).
        max_concurrency: Number of parallel evaluation threads.
        managed_evaluators: Names of managed (LLM-as-a-Judge) evaluators
            configured in the LangFuse UI. When set, enables polling.
        managed_evaluator_timeout: Seconds to poll for managed evaluator scores
            after the main evaluation completes. 0 = don't poll (default).
        metric_function: Optional callable that receives {evaluator: aggregated_score}
            and returns a single float. Result is stored under metric_name.

    Returns:
        Dict mapping metric names to aggregated float scores.
    """
    try:
        from langfuse import Evaluation, Langfuse  # noqa: F401
    except ImportError:
        print("ERROR: langfuse package not installed.")
        print("Install with: pip install 'weco[langfuse]'")
        sys.exit(1)

    client = Langfuse()

    # Resolve and normalize local evaluators
    evaluators = resolve_evaluators(evaluator_names) if evaluator_names else []
    normalized_evaluators = [_normalize_evaluator(e) for e in evaluators]

    # Wrap target for LangFuse task signature
    wrapped_target = _wrap_target(target)

    # Get dataset
    dataset = client.get_dataset(dataset_name)

    # Build experiment kwargs
    exp_name = experiment_name or f"weco-{dataset_name}"
    exp_kwargs = {"name": exp_name, "task": wrapped_target, "evaluators": normalized_evaluators}
    if max_concurrency is not None:
        exp_kwargs["max_concurrency"] = max_concurrency

    # Run experiment
    result = dataset.run_experiment(**exp_kwargs)

    # Extract item-level scores from local evaluators
    all_scores = {}
    trace_ids = []

    for item_result in result.item_results:
        # Collect trace IDs for managed evaluator polling
        if item_result.trace_id:
            trace_ids.append(item_result.trace_id)

        for evaluation in item_result.evaluations:
            if evaluation.value is not None:
                try:
                    score_val = float(evaluation.value)
                except (TypeError, ValueError):
                    continue
                all_scores.setdefault(evaluation.name, []).append(score_val)

    # Flush to ensure all traces are sent before polling
    client.flush()

    # Optionally poll for managed evaluator scores
    expected_managed = set(managed_evaluators or [])
    if expected_managed and managed_evaluator_timeout > 0:
        known_keys = set(all_scores.keys())
        managed_scores = _poll_managed_scores(
            client, trace_ids, known_keys, expected_keys=expected_managed, timeout=managed_evaluator_timeout
        )
        found_keys = set(managed_scores.keys())
        missing = expected_managed - found_keys - known_keys
        if missing:
            print(
                f"WARNING: Managed evaluators not found within {managed_evaluator_timeout}s: "
                f"{', '.join(sorted(missing))}. Continuing with available scores.",
                file=sys.stderr,
            )
        elif expected_managed and not managed_scores:
            print(
                f"WARNING: No managed evaluator scores found within {managed_evaluator_timeout}s. "
                "Continuing with local evaluator scores only.",
                file=sys.stderr,
            )
        for key, values in managed_scores.items():
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
        description="LangFuse evaluation bridge for Weco",
        epilog="Runs a target function against a LangFuse dataset and prints metrics in Weco's expected format.",
    )
    parser.add_argument("--dataset", required=True, help="LangFuse dataset name")
    parser.add_argument("--target", required=True, help="Target function as module:function (e.g., agent:run_chain)")
    parser.add_argument(
        "--evaluators",
        nargs="+",
        default=[],
        help="Evaluator names: module:function specs or names from evaluators.py. "
        "Optional when using --managed-evaluator-timeout with managed evaluators.",
    )
    parser.add_argument("--metric", required=True, help="Primary metric name for Weco to optimize")
    parser.add_argument(
        "--summary",
        default="mean",
        choices=["mean", "median", "min", "max"],
        help="How to aggregate per-example scores (default: mean)",
    )
    parser.add_argument(
        "--experiment-name", default=None, help="Experiment name in the LangFuse UI (groups experiments for comparison)"
    )
    parser.add_argument("--max-concurrency", type=int, default=None, help="Number of parallel evaluation threads")
    parser.add_argument(
        "--managed-evaluators",
        nargs="+",
        default=[],
        help="Names of managed (LLM-as-a-Judge) evaluators configured in the LangFuse UI. "
        "Enables polling for their async scores after evaluation completes.",
    )
    parser.add_argument(
        "--managed-evaluator-timeout",
        type=int,
        default=900,
        help="Seconds to poll for managed evaluator scores (default: 900). "
        "Only used when --managed-evaluators is set. Polls every 10s.",
    )
    parser.add_argument(
        "--metric-function",
        default=None,
        help="Custom aggregation function as module:function. Receives dict of "
        "{evaluator: aggregated_score}, returns a single float used as the metric value.",
    )

    args = parser.parse_args()

    # Validate environment
    if not os.environ.get("LANGFUSE_SECRET_KEY"):
        print("ERROR: LANGFUSE_SECRET_KEY environment variable not set.")
        print("Get your API keys at https://cloud.langfuse.com")
        sys.exit(1)
    if not os.environ.get("LANGFUSE_PUBLIC_KEY"):
        print("ERROR: LANGFUSE_PUBLIC_KEY environment variable not set.")
        print("Get your API keys at https://cloud.langfuse.com")
        sys.exit(1)

    # Import target function — may fail if Weco generated a bad code variant.
    # The error output is useful feedback for the optimization backend.
    try:
        target = import_target(args.target)
    except (ImportError, SyntaxError, AttributeError) as e:
        print(f"Import error: {_sanitize_error(e)}")
        print("The generated code variant has issues preventing evaluation.")
        sys.exit(1)

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
        metrics = run_langfuse_eval(
            dataset_name=args.dataset,
            target=target,
            evaluator_names=args.evaluators,
            metric_name=args.metric,
            experiment_name=args.experiment_name,
            summary_mode=args.summary,
            max_concurrency=args.max_concurrency,
            managed_evaluators=args.managed_evaluators,
            managed_evaluator_timeout=args.managed_evaluator_timeout or 0,
            metric_function=metric_fn,
        )
    except Exception as e:
        print(f"LangFuse evaluation error: {_sanitize_error(e)}")
        sys.exit(1)

    if not metrics:
        if args.managed_evaluators:
            print("WARNING: No metrics returned. Managed evaluators may need more time.")
            print(f"Try increasing --managed-evaluator-timeout (currently {args.managed_evaluator_timeout}s).")
        else:
            print(f"WARNING: No metrics returned for evaluators {args.evaluators}.")
            print("Check that evaluator names are correct and the dataset is non-empty.")
        sys.exit(1)

    # Print ALL metrics — Weco optimizes --metric but benefits from seeing everything
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
