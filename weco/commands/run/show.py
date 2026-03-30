"""``weco run show <run-id> --step N``"""

import json
import sys

from rich.console import Console

from .. import make_client, fetch_nodes


def handle(run_id: str, step: str, console: Console) -> None:
    """Show details for a specific step/node."""
    client = make_client(console)

    if step == "best":
        _run_meta, nodes = fetch_nodes(client, run_id, sort="metric", top=1)
        if not nodes:
            print(json.dumps({"error": "No scored nodes found"}))
            sys.exit(1)
        node = nodes[0]
    else:
        try:
            step_num = int(step)
        except ValueError:
            print(json.dumps({"error": f"Invalid step: {step}. Use an integer or 'best'"}))
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

    print(json.dumps(output, indent=2))
