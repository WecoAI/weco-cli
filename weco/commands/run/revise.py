"""``weco run revise <run-id> --node <id> --source <file>``"""

import json
import sys

from rich.console import Console

from .. import make_client, fetch_nodes, read_source_code


def handle(run_id: str, node_id: str, source_paths: list[str], console: Console) -> None:
    """Replace a pending node's code with a new revision."""
    client = make_client(console)

    # Check if any arg uses explicit mapping (target=local)
    has_explicit = any("=" in arg for arg in source_paths)

    if has_explicit:
        # All mappings are explicit — no need to fetch baseline keys
        code = read_source_code(source_paths)
    else:
        # Fetch baseline to get the run's original code keys for positional mapping
        _meta, baseline_nodes = fetch_nodes(client, run_id, step=0)
        run_code_keys = list(baseline_nodes[0]["code"].keys()) if baseline_nodes and baseline_nodes[0].get("code") else []
        code = read_source_code(source_paths, run_code_keys=run_code_keys)

    result = client.create_revision(node_id, code)
    if result is None:
        print(json.dumps({"error": f"Failed to create revision for node {node_id}"}))
        sys.exit(1)

    output = {
        "node_id": node_id,
        "revision_id": result.get("id"),
        "plan": result.get("plan", ""),
        "created_by": result.get("created_by"),
        "message": f"Revision created. Submit with: weco run submit {run_id} --node {node_id}",
    }

    print(json.dumps(output, indent=2))
