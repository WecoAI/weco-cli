"""``weco run status <run-id>``"""

import json

from rich.console import Console

from .. import make_client, fetch_nodes, fetch_run


def handle(run_id: str, console: Console) -> None:
    """Show run status and progress as JSON."""
    client = make_client(console)

    # Single call: run metadata + pending nodes (no code)
    run_meta, pending_raw = fetch_nodes(client, run_id, status="pending_approval,pending_evaluation", include_code=False)

    pending_nodes = [{"node_id": n.get("node_id"), "step": n.get("step"), "plan": n.get("plan", "")} for n in pending_raw]

    output = {
        "run_id": run_id,
        "status": run_meta.get("status"),
        "name": run_meta.get("name"),
        "current_step": run_meta.get("current_step"),
        "total_steps": run_meta.get("steps"),
        "best_metric": run_meta.get("best_metric"),
        "best_step": run_meta.get("best_step"),
        "metric_name": run_meta.get("metric_name"),
        "goal": "maximize" if run_meta.get("maximize") else "minimize",
        "model": run_meta.get("model"),
        "require_review": run_meta.get("require_review", False),
        "pending_nodes": pending_nodes,
    }

    # Add lineage info if present (requires full run status fetch)
    try:
        full_status = fetch_run(client, run_id, include_history=False)
        if full_status.get("lineage_id"):
            output["lineage_id"] = full_status["lineage_id"]
        if full_status.get("derived_from"):
            output["derived_from"] = full_status["derived_from"]
    except SystemExit:
        pass  # Non-critical — lineage info is supplementary

    print(json.dumps(output, indent=2))
