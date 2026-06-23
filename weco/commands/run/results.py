"""``weco run results <run-id>``"""

import json
from typing import Optional

from rich.console import Console

from .. import make_client, fetch_nodes, resolve_lineage_id, fetch_lineage, fetch_lineage_nodes


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


def _lineage_results(client, run_id: str, top: Optional[int], include_code: bool) -> tuple[dict, list[dict]]:
    """Fetch lineage-wide nodes ranked by metric, normalised to the per-run node shape.

    Each node also carries ``global_step`` and ``run_id`` so the lineage view can
    show where in the tree a result came from.
    """
    lineage_id = resolve_lineage_id(client, run_id)
    lineage = fetch_lineage(client, lineage_id)
    raw = fetch_lineage_nodes(client, lineage_id, include_details=include_code)

    scored = [n for n in raw if n.get("metric_value") is not None]
    scored.sort(key=lambda n: n["metric_value"], reverse=bool(lineage.get("maximize")))
    if top is not None:
        scored = scored[:top]

    nodes = [
        {
            "step": n.get("step"),
            "global_step": n.get("global_step"),
            "run_id": n.get("run_id"),
            "metric_value": n.get("metric_value"),
            "plan": n.get("plan", ""),
            "node_id": n.get("id"),
            "code": n.get("code", {}),
        }
        for n in scored
    ]
    run_meta = {
        "metric_name": lineage.get("metric_name", "metric"),
        "best_metric": lineage.get("best_metric"),
        "best_step": (lineage.get("best") or {}).get("step"),
    }
    return run_meta, nodes


def handle(
    run_id: str, top: Optional[int], format: str, plot: bool, include_code: bool, lineage: bool, console: Console
) -> None:
    """Show run results sorted by metric."""
    client = make_client(console)

    if top is not None and top <= 0:
        print(json.dumps([]))
        return

    if lineage:
        run_meta, nodes = _lineage_results(client, run_id, top, include_code)
    else:
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
            if lineage:
                entry["global_step"] = n.get("global_step")
                entry["run_id"] = n.get("run_id")
            if include_code:
                entry["code"] = n.get("code", {})
            results.append(entry)
        print(json.dumps(results, indent=2))

    elif format == "table":
        if lineage:
            header = f"{'Global':>6}  {'Run:Step':>14}  {'Metric':>12}  Plan"
        else:
            header = f"{'Step':>6}  {'Metric':>12}  Plan"
        print(header)
        print("-" * len(header))
        for n in nodes:
            metric = n.get("metric_value")
            metric_str = f"{metric:.6g}" if metric is not None else "N/A"
            plan = (n.get("plan") or "")[:80]
            if lineage:
                gstep = n.get("global_step")
                gstr = str(gstep) if gstep is not None else "?"
                coord = f"{(n.get('run_id') or '')[:8]}:{n.get('step', '?')}"
                print(f"{gstr:>6}  {coord:>14}  {metric_str:>12}  {plan}")
            else:
                print(f"{n.get('step', '?'):>6}  {metric_str:>12}  {plan}")

    elif format == "csv":
        if lineage:
            print(f"global_step,run_id,step,{metric_name},plan,node_id")
            for n in nodes:
                plan = (n.get("plan") or "").replace('"', '""')
                print(
                    f"{n.get('global_step', '')},{n.get('run_id', '')},{n.get('step', '')},"
                    f'{n.get("metric_value", "")},"{plan}",{n.get("node_id", "")}'
                )
        else:
            print(f"step,{metric_name},plan,node_id")
            for n in nodes:
                plan = (n.get("plan") or "").replace('"', '""')
                print(f'{n.get("step", "")},{n.get("metric_value", "")},"{plan}",{n.get("node_id", "")}')

    if plot:
        if lineage:
            # Already have the lineage-wide set; order by global step for trajectory.
            all_by_step = sorted(
                [n for n in nodes if n.get("metric_value") is not None],
                key=lambda n: n.get("global_step") if n.get("global_step") is not None else 0,
            )
        else:
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
            label = "Lineage steps" if lineage else "Steps"
            print(f"\n{label} 0-{total}: {first_val:.4g} {spark} -> {best_val:.4g} (best at step {best_step})")
