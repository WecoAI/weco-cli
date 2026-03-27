"""``weco run revise <run-id> --node <id> --source <file>``"""

import json
import sys

from rich.console import Console

from .. import make_client, read_source_files


def handle(run_id: str, node_id: str, source_paths: list[str], console: Console) -> None:
    client = make_client(console)

    code = read_source_files(source_paths)

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
