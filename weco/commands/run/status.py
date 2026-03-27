"""``weco run status <run-id>``"""

import json

from rich.console import Console

from .. import make_client, fetch_run, get_node_id


def handle(run_id: str, console: Console) -> None:
    client = make_client(console)
    data = fetch_run(client, run_id)

    nodes = data.get("nodes") or []
    best_result = data.get("best_result") or {}
    objective = data.get("objective") or {}
    optimizer = data.get("optimizer") or {}
    maximize = bool(objective.get("maximize", True))

    pending_nodes = [
        {"node_id": get_node_id(n), "step": n.get("step"), "plan": n.get("plan", "")}
        for n in nodes
        if n.get("status") in ("pending_approval", "pending_evaluation")
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
        "pending_nodes": pending_nodes,
    }

    print(json.dumps(output, indent=2))
