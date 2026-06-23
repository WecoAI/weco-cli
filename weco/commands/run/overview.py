"""``weco run overview <run-id>`` — lineage-wide run picture (dashboard parity).

The other read commands (``status``, ``results``, ``show``, ``diff``) are
single-run scoped. Once a run has been ``derive``d, the related runs form a
*lineage* — a tree of root + derived branches — and a per-run view can't show
the global best, the derived branches, or where each branched. This command
resolves a run to its lineage and returns the whole picture in one call,
matching what the dashboard run page shows.
"""

import json

from rich.console import Console

from .. import make_client, resolve_lineage_id, fetch_lineage, fetch_lineage_nodes
from .results import _sparkline


def handle(run_id: str, include_code: bool, plot: bool, console: Console) -> None:
    """Show the full lineage overview for the run as JSON."""
    client = make_client(console)

    lineage_id = resolve_lineage_id(client, run_id)
    lineage = fetch_lineage(client, lineage_id)
    nodes = fetch_lineage_nodes(client, lineage_id, include_details=include_code)

    best = lineage.get("best")
    members = []
    for m in lineage.get("members", []):
        members.append(
            {
                "run_id": m.get("id"),
                "name": m.get("name"),
                "status": m.get("status"),
                "sub_run_index": m.get("sub_run_index"),
                "best_metric": m.get("best_metric"),
                "best_step": m.get("best_step"),
                "current_step": m.get("current_step"),
                "steps": m.get("steps"),
                "derived_from": m.get("derived_from"),
                "additional_instructions": m.get("additional_instructions"),
                "children": m.get("children", []),
            }
        )

    node_list = []
    for n in nodes:
        entry = {
            "global_step": n.get("global_step"),
            "run_id": n.get("run_id"),
            "step": n.get("step"),
            "node_id": n.get("id"),
            "parent_id": n.get("parent_id"),
            "metric": n.get("metric_value"),
            "status": n.get("status"),
            "is_buggy": n.get("is_buggy"),
            "summary_title": n.get("summary_title"),
        }
        if include_code:
            entry["plan"] = n.get("plan", "")
            entry["code"] = n.get("code", {})
        node_list.append(entry)

    output = {
        "lineage_id": lineage.get("id", lineage_id),
        "root_run_id": lineage.get("root_run_id"),
        "name": lineage.get("name"),
        "metric_name": lineage.get("metric_name"),
        "goal": "maximize" if lineage.get("maximize") else "minimize",
        "status": lineage.get("status"),
        "best_metric": lineage.get("best_metric"),
        "current_step": lineage.get("current_step"),
        "total_steps": lineage.get("total_steps"),
        "member_count": lineage.get("member_count"),
        "active_member_count": lineage.get("active_member_count"),
        "best": best,
        "members": members,
        "nodes": node_list,
    }

    print(json.dumps(output, indent=2))

    if plot:
        # Lineage-wide trajectory in global-step order, scored nodes only.
        scored = sorted(
            (n for n in nodes if n.get("metric_value") is not None),
            key=lambda n: n.get("global_step") if n.get("global_step") is not None else 0,
        )
        if scored:
            values = [n["metric_value"] for n in scored]
            best_val = lineage.get("best_metric", values[0])
            best_step = best.get("step") if best else "?"
            best_run = best.get("run_id") if best else "?"
            spark = _sparkline(values)
            print(
                f"\nLineage steps 0-{len(scored)}: {values[0]:.4g} {spark} -> "
                f"{best_val:.4g} (best at step {best_step} in run {best_run})"
            )
