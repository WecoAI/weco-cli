"""``weco run review <run-id>``"""

import json

from rich.console import Console

from .. import make_client, fetch_run, get_node_id


def handle(run_id: str, console: Console) -> None:
    client = make_client(console)
    data = fetch_run(client, run_id)

    nodes = data.get("nodes") or []

    pending = []
    for n in nodes:
        if n.get("status") in ("pending_approval", "pending_evaluation"):
            pending.append(
                {"node_id": get_node_id(n), "step": n.get("step"), "plan": n.get("plan", ""), "code": n.get("code", {})}
            )

    output = {"run_id": run_id, "require_review": data.get("require_review", False), "pending_nodes": pending}

    print(json.dumps(output, indent=2))
