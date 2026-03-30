"""``weco run results <run-id>``"""

import json
from typing import Optional

from rich.console import Console

from .. import make_client, fetch_nodes


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
    """Show run results sorted by metric."""
    client = make_client(console)

    if top is not None and top <= 0:
        print(json.dumps([]))
        return

    run_meta, nodes = fetch_nodes(client, run_id, top=top, sort="metric", include_code=include_code)
    metric_name = run_meta.get("metric_name", "metric")

    if format == "json":
        results = []
        for n in nodes:
            entry = {
                "step": n.get("step"),
                "metric": n.get("metric_value"),
                "plan": n.get("plan", ""),
                "node_id": n.get("node_id"),
            }
            if include_code:
                entry["code"] = n.get("code", {})
            results.append(entry)
        print(json.dumps(results, indent=2))

    elif format == "table":
        header = f"{'Step':>6}  {'Metric':>12}  Plan"
        print(header)
        print("-" * len(header))
        for n in nodes:
            step = n.get("step", "?")
            metric = n.get("metric_value")
            metric_str = f"{metric:.6g}" if metric is not None else "N/A"
            plan = (n.get("plan") or "")[:80]
            print(f"{step:>6}  {metric_str:>12}  {plan}")

    elif format == "csv":
        print(f"step,{metric_name},plan,node_id")
        for n in nodes:
            step = n.get("step", "")
            metric = n.get("metric_value", "")
            plan = (n.get("plan") or "").replace('"', '""')
            nid = n.get("node_id", "")
            print(f'{step},{metric},"{plan}",{nid}')

    if plot:
        # For trajectory we need all nodes in step order (not just top N)
        _run_meta, all_nodes = fetch_nodes(client, run_id, sort="metric", include_code=False)
        all_by_step = sorted([n for n in all_nodes if n.get("metric_value") is not None], key=lambda n: n.get("step", 0))
        if all_by_step:
            values = [n["metric_value"] for n in all_by_step]
            best_val = run_meta.get("best_metric", values[0])
            best_step = run_meta.get("best_step", "?")
            first_val = values[0]
            spark = _sparkline(values)
            total = len(all_by_step)
            print(f"\nSteps 0-{total}: {first_val:.4g} {spark} -> {best_val:.4g} (best at step {best_step})")
