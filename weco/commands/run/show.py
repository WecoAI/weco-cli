"""``weco run show <run-id> --step N``"""

import json

from rich.console import Console

from .. import make_client, fetch_run, resolve_step, get_node_id


def handle(run_id: str, step: str, console: Console) -> None:
    client = make_client(console)
    data = fetch_run(client, run_id)

    nodes = data.get("nodes") or []
    objective = data.get("objective") or {}
    maximize = bool(objective.get("maximize", True))

    node = resolve_step(nodes, step, maximize)

    # Find parent step number
    parent_id = node.get("parent_id")
    parent_step = None
    if parent_id:
        for n in nodes:
            if get_node_id(n) == parent_id:
                parent_step = n.get("step")
                break

    output = {
        "step": node.get("step"),
        "metric": node.get("metric_value"),
        "plan": node.get("plan", ""),
        "code": node.get("code", {}),
        "parent_step": parent_step,
        "node_id": get_node_id(node),
        "status": node.get("status"),
        "is_buggy": node.get("is_buggy"),
    }

    print(json.dumps(output, indent=2))
