"""``weco run results <run-id>``"""

import json
from typing import Optional

from rich.console import Console

from .. import make_client, fetch_run, sort_nodes_by_metric, find_best_node, get_node_id


def _sparkline(values: list[float], width: int = 30) -> str:
    """Generate an ASCII sparkline from a list of values."""
    if not values:
        return ""
    blocks = " ▁▂▃▄▅▆▇█"
    lo, hi = min(values), max(values)
    spread = hi - lo if hi != lo else 1
    if len(values) > width:
        step = len(values) / width
        sampled = [values[int(i * step)] for i in range(width)]
    else:
        sampled = values
    return "".join(blocks[min(int((v - lo) / spread * (len(blocks) - 1)), len(blocks) - 1)] for v in sampled)


def handle(run_id: str, top: Optional[int], format: str, plot: bool, include_code: bool, console: Console) -> None:
    client = make_client(console)
    data = fetch_run(client, run_id)

    nodes = data.get("nodes") or []
    objective = data.get("objective") or {}
    maximize = bool(objective.get("maximize", True))
    metric_name = objective.get("metric_name", "metric")

    sorted_nodes = sort_nodes_by_metric(nodes, maximize)
    if top is not None:
        sorted_nodes = sorted_nodes[:top]

    if format == "json":
        results = []
        for n in sorted_nodes:
            entry = {
                "step": n.get("step"),
                "metric": n.get("metric_value"),
                "plan": n.get("plan", ""),
                "node_id": get_node_id(n),
            }
            if include_code:
                entry["code"] = n.get("code", {})
            results.append(entry)
        print(json.dumps(results, indent=2))

    elif format == "table":
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
            nid = get_node_id(n)
            print(f'{step},{metric},"{plan}",{nid}')

    if plot:
        all_nodes_by_step = sorted(
            [n for n in (data.get("nodes") or []) if n.get("metric_value") is not None], key=lambda n: n.get("step", 0)
        )
        if all_nodes_by_step:
            values = [n["metric_value"] for n in all_nodes_by_step]
            best_node = find_best_node(data.get("nodes") or [], maximize)
            best_val = best_node["metric_value"] if best_node else values[-1]
            best_step = best_node.get("step", "?") if best_node else "?"
            first_val = values[0]
            spark = _sparkline(values)
            total = len(all_nodes_by_step)
            print(f"\nSteps 0-{total}: {first_val:.4g} {spark} → {best_val:.4g} (best at step {best_step})")
