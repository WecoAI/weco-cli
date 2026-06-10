"""``weco run diff <run-id> --step N [--against ...]``"""

import difflib
import json
import sys

from rich.console import Console

from .. import make_client, fetch_nodes


def _get_node_by_step(client, run_id: str, step: int, include_code: bool = True) -> dict | None:
    """Fetch a single node by step number."""
    _meta, nodes = fetch_nodes(client, run_id, step=step, include_code=include_code)
    return nodes[0] if nodes else None


def _get_best_node(client, run_id: str) -> dict | None:
    """Fetch the best-scoring node."""
    _meta, nodes = fetch_nodes(client, run_id, sort="metric", top=1)
    return nodes[0] if nodes else None


def handle(run_id: str, step: str, against: str, console: Console) -> None:
    """Show code diff between steps."""
    client = make_client(console)

    # Resolve target node
    if step == "best":
        target_node = _get_best_node(client, run_id)
        if not target_node:
            print(json.dumps({"error": "No scored nodes found"}))
            sys.exit(1)
    else:
        try:
            step_num = int(step)
        except ValueError:
            print(json.dumps({"error": f"Invalid step: {step}. Use an integer or 'best'"}))
            sys.exit(1)
        target_node = _get_node_by_step(client, run_id, step_num)
        if not target_node:
            print(json.dumps({"error": f"No node found at step {step_num}"}))
            sys.exit(1)

    target_code = target_node.get("code") or {}

    # Resolve base node
    if against == "baseline":
        base_node = _get_node_by_step(client, run_id, 0)
        if not base_node:
            print(json.dumps({"error": "No baseline node (step 0) found"}))
            sys.exit(1)
        base_label = "step 0 (baseline)"
    elif against == "parent":
        parent_step = target_node.get("parent_step")
        if parent_step is None:
            print(json.dumps({"error": f"Node at step {target_node.get('step')} has no parent"}))
            sys.exit(1)
        base_node = _get_node_by_step(client, run_id, parent_step)
        if not base_node:
            print(json.dumps({"error": f"Parent node at step {parent_step} not found"}))
            sys.exit(1)
        base_label = f"step {parent_step} (parent)"
    else:
        try:
            base_step = int(against)
        except ValueError:
            print(json.dumps({"error": f"Invalid --against value: {against}. Use 'baseline', 'parent', or a step number"}))
            sys.exit(1)
        base_node = _get_node_by_step(client, run_id, base_step)
        if not base_node:
            print(json.dumps({"error": f"No node found at step {base_step}"}))
            sys.exit(1)
        base_label = f"step {base_step}"

    base_code = base_node.get("code") or {}
    target_label = f"step {target_node.get('step')}"

    all_files = sorted(set(list(base_code.keys()) + list(target_code.keys())))
    has_diff = False

    for filepath in all_files:
        old = base_code.get(filepath, "")
        new = target_code.get(filepath, "")
        if old == new:
            continue

        diff = difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"{filepath} ({base_label})",
            tofile=f"{filepath} ({target_label})",
        )
        diff_text = "".join(diff)
        if diff_text:
            has_diff = True
            print(diff_text)

    if not has_diff:
        print(f"No differences between {base_label} and {target_label}.")
