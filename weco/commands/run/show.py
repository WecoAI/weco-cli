"""``weco run show <run-id> --step N``"""

import json
import sys

from rich.console import Console

from .. import make_client, fetch_nodes, resolve_lineage_id, fetch_lineage


def _lineage_best_node(client, run_id: str) -> tuple[dict | None, str | None]:
    """Resolve the lineage-global best node — the same node `derive --from-step
    best` branches from. Returns ``(node, best_run_id)`` with the node hydrated
    via its owning run so it carries code/parent_step, or ``(None, None)``."""
    lineage_id = resolve_lineage_id(client, run_id)
    lineage = fetch_lineage(client, lineage_id)
    best = lineage.get("best")
    if not best:
        return None, None
    best_run_id = best.get("run_id")
    _meta, nodes = fetch_nodes(client, best_run_id, step=best.get("step"))
    return (nodes[0] if nodes else None), best_run_id


def handle(run_id: str, step: str, console: Console) -> None:
    """Show details for a specific step/node."""
    client = make_client(console)

    best_run_id = run_id
    if step == "best":
        # Lineage-global best, consistent with `derive --from-step best`.
        node, best_run_id = _lineage_best_node(client, run_id)
        if not node:
            print(json.dumps({"error": "No scored nodes found in lineage"}))
            sys.exit(1)
    elif step == "run-best":
        _run_meta, nodes = fetch_nodes(client, run_id, sort="metric", top=1)
        if not nodes:
            print(json.dumps({"error": "No scored nodes found"}))
            sys.exit(1)
        node = nodes[0]
    else:
        try:
            step_num = int(step)
        except ValueError:
            print(json.dumps({"error": f"Invalid step: {step}. Use an integer, 'best', or 'run-best'"}))
            sys.exit(1)
        _run_meta, nodes = fetch_nodes(client, run_id, step=step_num)
        if not nodes:
            print(json.dumps({"error": f"No node found at step {step_num}"}))
            sys.exit(1)
        node = nodes[0]

    output = {
        "step": node.get("step"),
        "metric": node.get("metric_value"),
        "plan": node.get("plan", ""),
        "code": node.get("code", {}),
        "parent_step": node.get("parent_step"),
        "node_id": node.get("node_id"),
        "status": node.get("status"),
        "is_buggy": node.get("is_buggy"),
    }
    # When resolved across the lineage, surface which run holds the best node so
    # the agent isn't misled into thinking it belongs to the queried run.
    if step == "best" and best_run_id and best_run_id != run_id:
        output["run_id"] = best_run_id

    print(json.dumps(output, indent=2))
