"""``weco run status <run-id>``"""

import json

from rich.console import Console

from .. import make_client, fetch_nodes, fetch_run, resolve_lineage_id, fetch_lineage, fetch_lineage_nodes

# Statuses that mean the run/lineage is still working — used to decide whether
# to surface the agent operating contract.
_ACTIVE_STATUSES = ("running", "stopping", "pending_approval", "pending_evaluation")

# Just-in-time operating contract for an agent driving this run. Surfaced on the
# canonical progress check (not buried in a doc read once at the start) so it
# lands at the decision point every monitoring cycle: the run is self-driving,
# so stay responsive to the user and act via derive/stop.
_AGENT_GUIDANCE = (
    "This run is self-driving — do NOT tail logs or use any blocking/streaming watch; "
    "just poll this command (it returns immediately). Between polls you remain available "
    "to the user: if they send a message, acknowledge and act on it before polling again — "
    'steer with `weco run derive <run-id> --from-step best -i "<direction>"` (works mid-run, '
    "no need to stop first), or abort with `weco run stop <run-id>`. For the full lineage "
    "picture across all derived runs (global best, tree), use `weco run overview <run-id>`."
)


def _lineage_status(client, run_id: str) -> dict:
    """Lineage-wide aggregate status across every derived run in the lineage."""
    lineage_id = resolve_lineage_id(client, run_id)
    lineage = fetch_lineage(client, lineage_id)
    pending = fetch_lineage_nodes(client, lineage_id, status="pending_approval,pending_evaluation")
    pending_nodes = [
        {"node_id": n.get("id"), "run_id": n.get("run_id"), "global_step": n.get("global_step"), "step": n.get("step")}
        for n in pending
    ]
    best = lineage.get("best") or {}
    output = {
        "lineage_id": lineage.get("id", lineage_id),
        "scope": "lineage",
        "status": lineage.get("status"),
        "name": lineage.get("name"),
        "current_step": lineage.get("current_step"),
        "total_steps": lineage.get("total_steps"),
        "best_metric": lineage.get("best_metric"),
        "best_step": best.get("step"),
        "best_run_id": best.get("run_id"),
        "metric_name": lineage.get("metric_name"),
        "goal": "maximize" if lineage.get("maximize") else "minimize",
        "member_count": lineage.get("member_count"),
        "active_member_count": lineage.get("active_member_count"),
        "require_review": lineage.get("require_review", False),
        "pending_nodes": pending_nodes,
    }
    if lineage.get("status") in _ACTIVE_STATUSES:
        output["agent_guidance"] = _AGENT_GUIDANCE
    return output


def handle(run_id: str, lineage: bool, console: Console) -> None:
    """Show run status and progress as JSON."""
    client = make_client(console)

    if lineage:
        print(json.dumps(_lineage_status(client, run_id), indent=2))
        return

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

    if run_meta.get("status") in _ACTIVE_STATUSES:
        output["agent_guidance"] = _AGENT_GUIDANCE

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
