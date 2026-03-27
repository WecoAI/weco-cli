"""``weco run diff <run-id> --step N [--against ...]``"""

import difflib
import json
import sys

from rich.console import Console

from .. import make_client, fetch_run, resolve_step, find_node_by_step, get_node_id


def handle(run_id: str, step: str, against: str, console: Console) -> None:
    client = make_client(console)
    data = fetch_run(client, run_id)

    nodes = data.get("nodes") or []
    objective = data.get("objective") or {}
    maximize = bool(objective.get("maximize", True))

    target_node = resolve_step(nodes, step, maximize)
    target_code = target_node.get("code") or {}

    # Resolve the comparison node
    if against == "baseline":
        base_node = find_node_by_step(nodes, 0)
        if not base_node:
            print(json.dumps({"error": "No baseline node (step 0) found"}))
            sys.exit(1)
        base_label = "step 0 (baseline)"
    elif against == "parent":
        parent_id = target_node.get("parent_id")
        if not parent_id:
            print(json.dumps({"error": f"Node at step {target_node.get('step')} has no parent"}))
            sys.exit(1)
        base_node = None
        for n in nodes:
            if get_node_id(n) == parent_id:
                base_node = n
                break
        if not base_node:
            print(json.dumps({"error": f"Parent node {parent_id} not found"}))
            sys.exit(1)
        base_label = f"step {base_node.get('step')} (parent)"
    else:
        try:
            base_step = int(against)
        except ValueError:
            print(json.dumps({"error": f"Invalid --against value: {against}. Use 'baseline', 'parent', or a step number"}))
            sys.exit(1)
        base_node = find_node_by_step(nodes, base_step)
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
