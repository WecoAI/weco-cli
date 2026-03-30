"""Shared helpers for ``weco run`` subcommand handlers."""

import json
import pathlib
import sys

from rich.console import Console

from ..core.api import WecoClient
from ..auth import handle_authentication


def make_client(console: Console) -> WecoClient:
    """Authenticate and return a ``WecoClient``, or exit on failure."""
    _, auth_headers = handle_authentication(console)
    if not auth_headers:
        sys.exit(1)
    return WecoClient(auth_headers)


def fetch_run(client: WecoClient, run_id: str, include_history: bool = True) -> dict:
    """Fetch full run data via ``GET /runs/{run_id}``, or exit on error."""
    try:
        return client.get_run_status(run_id, include_history=include_history)
    except Exception as e:
        print(json.dumps({"error": f"Failed to fetch run {run_id}: {e}"}))
        sys.exit(1)


def fetch_nodes(
    client: WecoClient,
    run_id: str,
    *,
    step: int | None = None,
    status: str | None = None,
    top: int | None = None,
    sort: str | None = None,
    include_code: bool = True,
) -> tuple[dict, list[dict]]:
    """Fetch nodes via ``GET /runs/{run_id}/nodes``, or exit on error.

    Returns:
        A ``(run_metadata, nodes)`` tuple where ``run_metadata`` contains
        lightweight run info (status, metric_name, best_metric, etc.) and
        ``nodes`` is the filtered/sorted list of node dicts.
    """
    try:
        data = client.list_nodes(run_id, step=step, status=status, top=top, sort=sort, include_code=include_code)
        return data.get("run", {}), data.get("nodes", [])
    except Exception as e:
        print(json.dumps({"error": f"Failed to fetch nodes for run {run_id}: {e}"}))
        sys.exit(1)


def read_source_code(source_args: list[str], run_code_keys: list[str] | None = None) -> dict[str, str]:
    """Read source files and map them to run code keys.

    Each entry in *source_args* is either:

    * ``target_path=local_path`` — explicit mapping (store content of
      *local_path* under the key *target_path*)
    * ``local_path`` — no mapping; falls back to positional matching
      against *run_code_keys* if available, otherwise uses the local path.

    Args:
        source_args: Raw ``--source`` / ``--sources`` values from argparse.
        run_code_keys: The run's original source file keys (from baseline
            node).  Used for positional fallback when no explicit mapping.

    Returns:
        Dict mapping target path → file content.
    """
    explicit: dict[str, str] = {}
    unmapped_paths: list[str] = []
    unmapped_contents: list[str] = []

    for arg in source_args:
        if "=" in arg:
            target, local = arg.split("=", 1)
            target, local = target.strip(), local.strip()
            if not target or not local:
                print(json.dumps({"error": f"Invalid source mapping: '{arg}'. Use target_path=local_path"}))
                sys.exit(1)
            p = pathlib.Path(local)
            if not p.exists():
                print(json.dumps({"error": f"Source file not found: {local}"}))
                sys.exit(1)
            explicit[target] = p.read_text()
        else:
            p = pathlib.Path(arg)
            if not p.exists():
                print(json.dumps({"error": f"Source file not found: {arg}"}))
                sys.exit(1)
            unmapped_paths.append(arg)
            unmapped_contents.append(p.read_text())

    # Resolve unmapped files
    if unmapped_contents:
        if run_code_keys and len(unmapped_contents) == len(run_code_keys):
            # Positional match against the run's source keys
            for key, content in zip(run_code_keys, unmapped_contents):
                explicit[key] = content
        else:
            # Use local paths as-is
            for path, content in zip(unmapped_paths, unmapped_contents):
                explicit[str(pathlib.Path(path))] = content

    return explicit
