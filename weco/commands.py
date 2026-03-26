"""Agent-facing CLI commands for inspecting and interacting with Weco runs."""

import difflib
import json
import sys
from typing import Optional

from rich.console import Console

from .api import (
    get_optimization_run_status,
    report_termination,
    create_node_revision,
    submit_node_for_evaluation,
    claim_execution_task,
    submit_execution_result,
)
from .auth import handle_authentication


def _authenticate(console: Console) -> dict:
    """Authenticate and return auth headers, or exit."""
    _, auth_headers = handle_authentication(console)
    if not auth_headers:
        sys.exit(1)
    return auth_headers


def _fetch_run(console: Console, run_id: str, auth_headers: dict, include_history: bool = True) -> dict:
    """Fetch run data from API, or exit on error."""
    try:
        return get_optimization_run_status(
            console=console, run_id=run_id, include_history=include_history, auth_headers=auth_headers
        )
    except Exception:
        sys.exit(1)


def _sort_nodes_by_metric(nodes: list[dict], maximize: bool) -> list[dict]:
    """Sort nodes by metric value, best first. Nodes without metrics go last."""
    scored = [n for n in nodes if n.get("metric_value") is not None]
    scored.sort(key=lambda n: n["metric_value"], reverse=maximize)
    return scored


def _get_node_id(node: dict) -> str:
    """Extract the node ID from an API node dict.

    The API returns the node ID as `solution_id` (legacy) or `id`.
    """
    return node.get("solution_id") or node.get("id", "")


def _find_node_by_step(nodes: list[dict], step: int) -> Optional[dict]:
    """Find a node by step number."""
    for node in nodes:
        if node.get("step") == step:
            return node
    return None


def _find_best_node(nodes: list[dict], maximize: bool) -> Optional[dict]:
    """Find the best-scoring node."""
    scored = _sort_nodes_by_metric(nodes, maximize)
    return scored[0] if scored else None


def _resolve_step(nodes: list[dict], step_arg: str, maximize: bool) -> dict:
    """Resolve a --step argument ('best' or integer) to a node, or exit."""
    if step_arg == "best":
        node = _find_best_node(nodes, maximize)
        if not node:
            print(json.dumps({"error": "No scored nodes found"}))
            sys.exit(1)
        return node

    try:
        step = int(step_arg)
    except ValueError:
        print(json.dumps({"error": f"Invalid step: {step_arg}. Use an integer or 'best'"}))
        sys.exit(1)

    node = _find_node_by_step(nodes, step)
    if not node:
        print(json.dumps({"error": f"No node found at step {step}"}))
        sys.exit(1)
    return node


def _sparkline(values: list[float], width: int = 30) -> str:
    """Generate an ASCII sparkline from a list of values."""
    if not values:
        return ""
    blocks = " ▁▂▃▄▅▆▇█"
    lo, hi = min(values), max(values)
    spread = hi - lo if hi != lo else 1
    # Sample values to fit width
    if len(values) > width:
        step = len(values) / width
        sampled = [values[int(i * step)] for i in range(width)]
    else:
        sampled = values
    return "".join(blocks[min(int((v - lo) / spread * (len(blocks) - 1)), len(blocks) - 1)] for v in sampled)


# --- Command handlers ---


def handle_status_command(run_id: str, console: Console) -> None:
    """Handle `weco status <run-id>`."""
    auth_headers = _authenticate(console)
    data = _fetch_run(console, run_id, auth_headers)

    nodes = data.get("nodes") or []
    best_result = data.get("best_result") or {}
    objective = data.get("objective") or {}
    optimizer = data.get("optimizer") or {}
    maximize = bool(objective.get("maximize", True))

    # Find pending approval nodes
    pending_nodes = [
        {"node_id": _get_node_id(n), "step": n.get("step"), "plan": n.get("plan", "")}
        for n in nodes
        if n.get("status") == "pending_approval"
    ]

    output = {
        "run_id": run_id,
        "status": data.get("status"),
        "name": data.get("run_name"),
        "current_step": data.get("current_step"),
        "total_steps": optimizer.get("steps"),
        "best_metric": best_result.get("metric_value"),
        "best_step": best_result.get("step"),
        "metric_name": objective.get("metric_name"),
        "goal": "maximize" if maximize else "minimize",
        "model": (optimizer.get("code_generator") or {}).get("model"),
        "require_review": data.get("require_review", False),
        "pending_approval_nodes": pending_nodes,
    }

    print(json.dumps(output, indent=2))


def handle_results_command(
    run_id: str,
    top: Optional[int],
    format: str,
    plot: bool,
    include_code: bool,
    console: Console,
) -> None:
    """Handle `weco results <run-id>`."""
    auth_headers = _authenticate(console)
    data = _fetch_run(console, run_id, auth_headers)

    nodes = data.get("nodes") or []
    objective = data.get("objective") or {}
    maximize = bool(objective.get("maximize", True))
    metric_name = objective.get("metric_name", "metric")

    sorted_nodes = _sort_nodes_by_metric(nodes, maximize)
    if top is not None:
        sorted_nodes = sorted_nodes[:top]

    if format == "json":
        results = []
        for n in sorted_nodes:
            entry = {
                "step": n.get("step"),
                "metric": n.get("metric_value"),
                "plan": n.get("plan", ""),
                "node_id": _get_node_id(n),
            }
            if include_code:
                entry["code"] = n.get("code", {})
            results.append(entry)
        print(json.dumps(results, indent=2))

    elif format == "table":
        # Header
        header = f"{'Step':>6}  {'Metric':>12}  Plan"
        print(header)
        print("-" * len(header))
        for n in sorted_nodes:
            step = n.get("step", "?")
            metric = n.get("metric_value")
            metric_str = f"{metric:.6g}" if metric is not None else "N/A"
            plan = (n.get("plan") or "")[:80]
            print(f"{step:>6}  {metric_str:>12}  {plan}")

    elif format == "csv":
        print(f"step,{metric_name},plan,node_id")
        for n in sorted_nodes:
            step = n.get("step", "")
            metric = n.get("metric_value", "")
            plan = (n.get("plan") or "").replace('"', '""')
            nid = _get_node_id(n)
            print(f'{step},{metric},"{plan}",{nid}')

    if plot:
        # Build trajectory in step order
        all_nodes_by_step = sorted(
            [n for n in (data.get("nodes") or []) if n.get("metric_value") is not None],
            key=lambda n: n.get("step", 0),
        )
        if all_nodes_by_step:
            values = [n["metric_value"] for n in all_nodes_by_step]
            best_node = _find_best_node(data.get("nodes") or [], maximize)
            best_val = best_node["metric_value"] if best_node else values[-1]
            best_step = best_node.get("step", "?") if best_node else "?"
            first_val = values[0]
            spark = _sparkline(values)
            total = len(all_nodes_by_step)
            print(f"\nSteps 0-{total}: {first_val:.4g} {spark} → {best_val:.4g} (best at step {best_step})")


def handle_show_command(run_id: str, step: str, console: Console) -> None:
    """Handle `weco show <run-id> --step N`."""
    auth_headers = _authenticate(console)
    data = _fetch_run(console, run_id, auth_headers)

    nodes = data.get("nodes") or []
    objective = data.get("objective") or {}
    maximize = bool(objective.get("maximize", True))

    node = _resolve_step(nodes, step, maximize)

    # Find parent step number
    parent_id = node.get("parent_id")
    parent_step = None
    if parent_id:
        for n in nodes:
            if _get_node_id(n) == parent_id:
                parent_step = n.get("step")
                break

    output = {
        "step": node.get("step"),
        "metric": node.get("metric_value"),
        "plan": node.get("plan", ""),
        "code": node.get("code", {}),
        "parent_step": parent_step,
        "node_id": _get_node_id(node),
        "status": node.get("status"),
        "is_buggy": node.get("is_buggy"),
    }

    print(json.dumps(output, indent=2))


def handle_diff_command(run_id: str, step: str, against: str, console: Console) -> None:
    """Handle `weco diff <run-id> --step N [--against baseline|parent|M]`."""
    auth_headers = _authenticate(console)
    data = _fetch_run(console, run_id, auth_headers)

    nodes = data.get("nodes") or []
    objective = data.get("objective") or {}
    maximize = bool(objective.get("maximize", True))

    target_node = _resolve_step(nodes, step, maximize)
    target_code = target_node.get("code") or {}

    # Resolve the comparison node
    if against == "baseline":
        base_node = _find_node_by_step(nodes, 0)
        if not base_node:
            print(json.dumps({"error": "No baseline node (step 0) found"}))
            sys.exit(1)
        base_label = "step 0 (baseline)"
    elif against == "parent":
        parent_id = target_node.get("parent_id")
        if not parent_id:
            print(json.dumps({"error": f"Node at step {target_node.get('step')} has no parent"}))
            sys.exit(1)
        base_node = None
        for n in nodes:
            if _get_node_id(n) == parent_id:
                base_node = n
                break
        if not base_node:
            print(json.dumps({"error": f"Parent node {parent_id} not found"}))
            sys.exit(1)
        base_label = f"step {base_node.get('step')} (parent)"
    else:
        try:
            base_step = int(against)
        except ValueError:
            print(json.dumps({"error": f"Invalid --against value: {against}. Use 'baseline', 'parent', or a step number"}))
            sys.exit(1)
        base_node = _find_node_by_step(nodes, base_step)
        if not base_node:
            print(json.dumps({"error": f"No node found at step {base_step}"}))
            sys.exit(1)
        base_label = f"step {base_step}"

    base_code = base_node.get("code") or {}
    target_label = f"step {target_node.get('step')}"

    # Generate diffs for each file
    all_files = sorted(set(list(base_code.keys()) + list(target_code.keys())))
    has_diff = False

    for filepath in all_files:
        old = base_code.get(filepath, "")
        new = target_code.get(filepath, "")
        if old == new:
            continue

        old_lines = old.splitlines(keepends=True)
        new_lines = new.splitlines(keepends=True)

        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"{filepath} ({base_label})",
            tofile=f"{filepath} ({target_label})",
        )
        diff_text = "".join(diff)
        if diff_text:
            has_diff = True
            print(diff_text)

    if not has_diff:
        print(f"No differences between {base_label} and {target_label}.")


def handle_stop_command(run_id: str, console: Console) -> None:
    """Handle `weco stop <run-id>`."""
    auth_headers = _authenticate(console)

    # Fetch current status first
    data = _fetch_run(console, run_id, auth_headers, include_history=False)
    current_status = data.get("status")

    if current_status in ("completed", "stopped", "terminated"):
        print(json.dumps({
            "run_id": run_id,
            "status": current_status,
            "message": f"Run is already {current_status}",
        }, indent=2))
        return

    # Terminate
    success = report_termination(
        run_id=run_id,
        status_update="terminated",
        reason="user_terminated_cli_stop",
        details="Terminated via `weco stop` command",
        auth_headers=auth_headers,
    )

    if not success:
        print(json.dumps({"error": "Failed to terminate run"}))
        sys.exit(1)

    # Fetch updated status for best metric info
    try:
        data = _fetch_run(console, run_id, auth_headers, include_history=False)
    except SystemExit:
        data = {}

    best_result = data.get("best_result") or {}

    output = {
        "run_id": run_id,
        "status": "terminated",
        "best_metric": best_result.get("metric_value"),
        "best_step": best_result.get("step"),
        "message": "Run terminated. Solution tree preserved. Resume with: weco resume " + run_id,
    }

    print(json.dumps(output, indent=2))


# --- Review mode commands ---


def _read_source_files(source_paths: list[str]) -> dict[str, str]:
    """Read source files into a dict mapping path -> content."""
    import pathlib

    code = {}
    for path in source_paths:
        p = pathlib.Path(path)
        if not p.exists():
            print(json.dumps({"error": f"Source file not found: {path}"}))
            sys.exit(1)
        code[str(p)] = p.read_text()
    return code


def handle_review_command(run_id: str, console: Console) -> None:
    """Handle `weco review <run-id>` — show pending approval nodes."""
    auth_headers = _authenticate(console)
    data = _fetch_run(console, run_id, auth_headers)

    require_review = data.get("require_review", False)
    nodes = data.get("nodes") or []

    # Filter to pending_approval nodes
    pending = []
    for n in nodes:
        if n.get("status") == "pending_approval":
            pending.append({
                "node_id": _get_node_id(n),
                "step": n.get("step"),
                "plan": n.get("plan", ""),
                "code": n.get("code", {}),
            })

    output = {
        "run_id": run_id,
        "require_review": require_review,
        "pending_nodes": pending,
    }

    print(json.dumps(output, indent=2))


def handle_revise_command(
    run_id: str,
    node_id: str,
    source_paths: list[str],
    console: Console,
) -> None:
    """Handle `weco revise <run-id> --node <id> --source <file>` — create a new revision."""
    auth_headers = _authenticate(console)

    code = _read_source_files(source_paths)

    result = create_node_revision(node_id=node_id, code=code, auth_headers=auth_headers)
    if result is None:
        print(json.dumps({"error": f"Failed to create revision for node {node_id}"}))
        sys.exit(1)

    output = {
        "node_id": node_id,
        "revision_id": result.get("id"),
        "plan": result.get("plan", ""),
        "created_by": result.get("created_by"),
        "message": f"Revision created. Submit with: weco run submit {run_id} --node {node_id}",
    }

    print(json.dumps(output, indent=2))


def handle_submit_command(
    run_id: str,
    node_id: str,
    source_paths: Optional[list[str]],
    console: Console,
) -> None:
    """Handle `weco submit <run-id> --node <id> [--source <file>]`.

    If --source is provided, creates a revision first (revise + submit in one step).
    Then submits the node for evaluation, claims the task, evaluates locally, and reports results.
    """
    from .utils import run_evaluation_with_files_swap

    auth_headers = _authenticate(console)

    # If source files provided, create a revision first
    if source_paths:
        code = _read_source_files(source_paths)
        revision_result = create_node_revision(node_id=node_id, code=code, auth_headers=auth_headers)
        if revision_result is None:
            print(json.dumps({"error": f"Failed to create revision for node {node_id}"}))
            sys.exit(1)

    # Submit node for evaluation — creates an execution task
    submit_result = submit_node_for_evaluation(node_id=node_id, auth_headers=auth_headers)
    if submit_result is None:
        print(json.dumps({"error": f"Failed to submit node {node_id} for evaluation. Is the run in review mode and the node pending approval?"}))
        sys.exit(1)

    task_id = submit_result.get("task_id")
    if not task_id:
        print(json.dumps({"error": "No task_id returned from submit"}))
        sys.exit(1)

    # Claim the task
    claimed = claim_execution_task(task_id=task_id, auth_headers=auth_headers)
    if claimed is None:
        print(json.dumps({"error": f"Failed to claim task {task_id}"}))
        sys.exit(1)

    # Get the code from the claimed revision
    revision = claimed.get("revision", {})
    file_map = revision.get("code", {})
    if isinstance(file_map, str):
        print(json.dumps({"error": "Unexpected single-file code format from revision"}))
        sys.exit(1)

    # Fetch run config to get eval command and source paths for restoration
    data = _fetch_run(console, run_id, auth_headers, include_history=False)
    objective = data.get("objective") or {}
    eval_command = objective.get("evaluation_command", "")
    eval_timeout = data.get("eval_timeout")

    if not eval_command:
        print(json.dumps({"error": "No evaluation command found for this run"}))
        sys.exit(1)

    # Read original source files for restoration after eval
    import pathlib

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
    result = submit_execution_result(
        run_id=run_id,
        task_id=task_id,
        execution_output=term_out,
        auth_headers=auth_headers,
    )

    if result is None:
        print(json.dumps({"error": "Failed to submit execution result"}))
        sys.exit(1)

    metric_value = result.get("previous_solution_metric_value")

    output = {
        "status": "submitted",
        "node_id": node_id,
        "task_id": task_id,
        "metric": metric_value,
        "is_done": result.get("is_done", False),
        "execution_output": term_out[:2000] if term_out else "",
    }

    print(json.dumps(output, indent=2))
