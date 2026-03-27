"""``weco run submit <run-id> --node <id> [--source <file>] [--eval-command <cmd>]``"""

import json
import pathlib
import sys
from typing import Optional

from rich.console import Console

from .. import make_client, fetch_run, read_source_files, validate_relative_code_paths


def handle(
    run_id: str, node_id: str, source_paths: Optional[list[str]], eval_command_override: Optional[str], console: Console
) -> None:
    from ...core.evaluation import run_evaluation_with_files_swap

    client = make_client(console)

    # If source files provided, create a revision first
    if source_paths:
        code = read_source_files(source_paths)
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
    validate_relative_code_paths(list(file_map.keys()))

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
        if p.exists():
            originals[rel_path] = p.read_text()
        else:
            originals[rel_path] = ""

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
