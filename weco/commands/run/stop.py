"""``weco run stop <run-id>``"""

import json
import sys

from rich.console import Console

from .. import make_client, fetch_run


def handle(run_id: str, console: Console) -> None:
    """Terminate a running optimization.  Tree is preserved for ``weco resume``."""
    client = make_client(console)

    data = fetch_run(client, run_id, include_history=False)
    current_status = data.get("status")

    if current_status in ("completed", "stopped", "terminated"):
        print(
            json.dumps({"run_id": run_id, "status": current_status, "message": f"Run is already {current_status}"}, indent=2)
        )
        return

    success = client.terminate(run_id, reason="user_terminated_cli_stop", details="Terminated via `weco run stop` command")

    if not success:
        print(json.dumps({"error": "Failed to terminate run"}))
        sys.exit(1)

    try:
        data = fetch_run(client, run_id, include_history=False)
    except SystemExit:
        data = {}

    best_result = data.get("best_result") or {}

    output = {
        "run_id": run_id,
        "status": "terminated",
        "best_metric": best_result.get("metric_value"),
        "best_step": best_result.get("step"),
        "message": "Run terminated. Solution tree preserved. Resume with: weco resume " + run_id,
    }

    print(json.dumps(output, indent=2))
