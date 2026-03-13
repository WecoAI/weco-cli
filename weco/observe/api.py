"""HTTP client for external run API endpoints.

All functions are synchronous (using requests) and never raise exceptions.
Errors are returned as None so the caller can warn without crashing.
"""

import warnings
from typing import Any

import requests

from weco import __base_url__


def create_run(
    *,
    source_code: dict[str, str],
    metric_name: str,
    maximize: bool,
    name: str | None = None,
    additional_instructions: str | None = None,
    metadata: dict[str, Any] | None = None,
    auth_headers: dict[str, str],
) -> dict | None:
    """Create an external run. Returns response dict or None on failure."""
    try:
        payload: dict[str, Any] = {"source_code": source_code, "metric_name": metric_name, "maximize": maximize}
        if name is not None:
            payload["name"] = name
        if additional_instructions is not None:
            payload["additional_instructions"] = additional_instructions
        if metadata:
            payload["metadata"] = metadata

        response = requests.post(f"{__base_url__}/external/runs", json=payload, headers=auth_headers, timeout=(5, 30))
        response.raise_for_status()
        return response.json()
    except Exception as e:
        warnings.warn(f"weco observe: failed to create run: {e}", stacklevel=2)
        return None


def log_step(
    *,
    run_id: str,
    step: int,
    status: str = "completed",
    description: str | None = None,
    metrics: dict[str, float] | None = None,
    code: dict[str, str] | None = None,
    parent_step: int | None = None,
    metadata: dict[str, Any] | None = None,
    auth_headers: dict[str, str],
) -> dict | None:
    """Log a step for an external run. Returns response dict or None on failure."""
    try:
        payload: dict[str, Any] = {"step": step, "status": status}
        if description is not None:
            payload["description"] = description
        if metrics:
            payload["metrics"] = metrics
        if code is not None:
            payload["code"] = code
        if parent_step is not None:
            payload["parent_step"] = parent_step
        if metadata:
            payload["metadata"] = metadata

        response = requests.post(
            f"{__base_url__}/external/runs/{run_id}/steps", json=payload, headers=auth_headers, timeout=(5, 30)
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        warnings.warn(f"weco observe: failed to log step {step}: {e}", stacklevel=2)
        return None
