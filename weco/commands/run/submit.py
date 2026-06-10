"""``weco run submit <run-id> --node <id> [--source <file>] [--eval-command <cmd>]``"""

import json
import pathlib
import sys
from typing import Optional

from rich.console import Console

from .. import make_client, fetch_run, fetch_nodes, read_source_code


def _validate_relative_code_paths(paths: list[str]) -> None:
    """Validate revision code paths are safe relative paths."""
    for raw_path in paths:
        if not isinstance(raw_path, str) or not raw_path.strip():
            print(json.dumps({"error": "Invalid empty file path in revision code"}))
            sys.exit(1)
        parsed = pathlib.Path(raw_path)
        if parsed.is_absolute() or ".." in parsed.parts:
            print(json.dumps({"error": f"Unsafe file path in revision code: {raw_path}"}))
            sys.exit(1)


def handle(
    run_id: str, node_id: str, source_paths: Optional[list[str]], eval_command_override: Optional[str], console: Console
) -> None:
    """Submit a pending node for local evaluation and report results.

    If ``--source`` is provided, creates a revision first (revise + submit
    in one step).  ``--eval-command`` overrides the stored eval command.
    """
    from ...utils import run_evaluation_with_files_swap

    client = make_client(console)

    # If source files provided, create a revision first
    if source_paths:
        has_explicit = any("=" in arg for arg in source_paths)
        if has_explicit:
            code = read_source_code(source_paths)
        else:
            _meta, baseline_nodes = fetch_nodes(client, run_id, step=0)
            run_code_keys = list(baseline_nodes[0]["code"].keys()) if baseline_nodes and baseline_nodes[0].get("code") else []
            code = read_source_code(source_paths, run_code_keys=run_code_keys)

        revision_result = client.create_revision(node_id, code)
        if revision_result is None:
            print(json.dumps({"error": f"Failed to create revision for node {node_id}"}))
            sys.exit(1)

    # Submit node for evaluation — creates an execution task
    submit_result = client.submit_node(node_id)
    if submit_result is None:
        print(
            json.dumps(
                {
                    "error": f"Failed to submit node {node_id} for evaluation. Is the run in review mode and the node pending approval?"
                }
            )
        )
        sys.exit(1)

    task_id = submit_result.get("task_id")
    if not task_id:
        print(json.dumps({"error": "No task_id returned from submit"}))
        sys.exit(1)

    # Claim the task
    claimed = client.claim_task(task_id)
    if claimed is None:
        print(json.dumps({"error": f"Failed to claim task {task_id}"}))
        sys.exit(1)

    # Get the code from the claimed revision
    revision = claimed.get("revision", {})
    file_map = revision.get("code", {})
    if isinstance(file_map, str):
        print(json.dumps({"error": "Unexpected single-file code format from revision"}))
        sys.exit(1)
    _validate_relative_code_paths(list(file_map.keys()))

    # Fetch run config to get eval command
    data = fetch_run(client, run_id, include_history=False)
    objective = data.get("objective") or {}
    eval_command = eval_command_override or objective.get("evaluation_command", "")
    eval_timeout = data.get("eval_timeout")

    if not eval_command:
        print(json.dumps({"error": "No evaluation command found for this run. Use --eval-command to specify one."}))
        sys.exit(1)

    # Read original source files for restoration after eval
    originals = {}
    for rel_path in file_map:
        p = pathlib.Path(rel_path)
        originals[rel_path] = p.read_text() if p.exists() else ""

    # Run evaluation
    print(json.dumps({"status": "evaluating", "eval_command": eval_command}))
    term_out = run_evaluation_with_files_swap(
        file_map=file_map, originals=originals, eval_command=eval_command, timeout=eval_timeout
    )

    # Submit result
    result = client.suggest(run_id, execution_output=term_out, task_id=task_id)

    if result is None:
        print(json.dumps({"error": "Failed to submit execution result"}))
        sys.exit(1)

    metric_value = result.get("previous_solution_metric_value")
    eval_failed = metric_value is None

    output = {
        "status": "eval_failed" if eval_failed else "submitted",
        "node_id": node_id,
        "task_id": task_id,
        "metric": metric_value,
        "run_completed": result.get("is_done", False),
        "execution_output": term_out[:2000] if term_out else "",
    }

    print(json.dumps(output, indent=2))
