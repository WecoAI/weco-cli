"""Shared helpers for command implementations."""

import json
import pathlib
import sys
from typing import Optional

from rich.console import Console

from ..core.api import WecoClient
from ..core.auth import handle_authentication


def make_client(console: Console) -> WecoClient:
    """Authenticate and return a WecoClient, or exit."""
    _, auth_headers = handle_authentication(console)
    if not auth_headers:
        sys.exit(1)
    return WecoClient(auth_headers)


def fetch_run(client: WecoClient, run_id: str, include_history: bool = True) -> dict:
    """Fetch run data from API, or exit on error."""
    try:
        return client.get_run_status(run_id, include_history=include_history)
    except Exception:
        sys.exit(1)


def get_node_id(node: dict) -> str:
    """Extract the node ID from an API node dict.

    The API returns the node ID as ``solution_id`` (legacy) or ``id``.
    """
    return node.get("solution_id") or node.get("id", "")


def sort_nodes_by_metric(nodes: list[dict], maximize: bool) -> list[dict]:
    """Sort nodes by metric value, best first. Nodes without metrics are excluded."""
    scored = [n for n in nodes if n.get("metric_value") is not None]
    scored.sort(key=lambda n: n["metric_value"], reverse=maximize)
    return scored


def find_node_by_step(nodes: list[dict], step: int) -> Optional[dict]:
    """Find a node by step number."""
    for node in nodes:
        if node.get("step") == step:
            return node
    return None


def find_best_node(nodes: list[dict], maximize: bool) -> Optional[dict]:
    """Find the best-scoring node."""
    scored = sort_nodes_by_metric(nodes, maximize)
    return scored[0] if scored else None


def resolve_step(nodes: list[dict], step_arg: str, maximize: bool) -> dict:
    """Resolve a ``--step`` argument ('best' or integer) to a node, or exit."""
    if step_arg == "best":
        node = find_best_node(nodes, maximize)
        if not node:
            print(json.dumps({"error": "No scored nodes found"}))
            sys.exit(1)
        return node

    try:
        step = int(step_arg)
    except ValueError:
        print(json.dumps({"error": f"Invalid step: {step_arg}. Use an integer or 'best'"}))
        sys.exit(1)

    node = find_node_by_step(nodes, step)
    if not node:
        print(json.dumps({"error": f"No node found at step {step}"}))
        sys.exit(1)
    return node


def read_source_files(source_paths: list[str]) -> dict[str, str]:
    """Read source files into a dict mapping path -> content."""
    code = {}
    for path in source_paths:
        p = pathlib.Path(path)
        if not p.exists():
            print(json.dumps({"error": f"Source file not found: {path}"}))
            sys.exit(1)
        code[str(p)] = p.read_text()
    return code


def validate_relative_code_paths(paths: list[str]) -> None:
    """Validate revision code paths are safe relative paths."""
    for raw_path in paths:
        if not isinstance(raw_path, str) or not raw_path.strip():
            print(json.dumps({"error": "Invalid empty file path in revision code"}))
            sys.exit(1)

        parsed = pathlib.Path(raw_path)
        if parsed.is_absolute() or ".." in parsed.parts:
            print(json.dumps({"error": f"Unsafe file path in revision code: {raw_path}"}))
            sys.exit(1)
