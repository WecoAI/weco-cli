"""``weco run review <run-id>``"""

import json

from rich.console import Console

from .. import make_client, fetch_nodes


def handle(run_id: str, console: Console) -> None:
    """Show nodes awaiting action (pending approval or evaluation)."""
    client = make_client(console)

    # Single call: run metadata + pending nodes with code
    run_meta, pending_raw = fetch_nodes(client, run_id, status="pending_approval,pending_evaluation")

    pending = [
        {"node_id": n.get("node_id"), "step": n.get("step"), "plan": n.get("plan", ""), "code": n.get("code", {})}
        for n in pending_raw
    ]

    output = {"run_id": run_id, "require_review": run_meta.get("require_review", False), "pending_nodes": pending}

    print(json.dumps(output, indent=2))
