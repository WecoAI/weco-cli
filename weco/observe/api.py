"""HTTP client for external run API endpoints.

All functions are synchronous (using requests) and raise typed exceptions on
failure — a contract change from earlier releases, where errors were swallowed
and returned as ``None``:

- ``CallerError``: the request is wrong in a way only the caller can fix
  (not logged in, 4xx). Retrying the identical call cannot succeed.
- ``TransientError``: a weco-side or network blip (5xx, 408/429, timeout,
  connection drop, malformed response body), raised only after bounded
  retries. Callers embedded in a loop they must not crash (the CLI's ``log``,
  the SDK) catch this and carry on.

Failures that fit neither pattern are treated as transient: misclassifying a
real error as transient costs one dropped data point, while misclassifying a
blip as fatal can kill the loop being tracked.
"""

import random
import time
from typing import Any

import requests

from weco import __base_url__
from weco.core.api import api_error_message


class ObserveError(Exception):
    """Base class for observe API failures."""


class CallerError(ObserveError):
    """The request is wrong in a way only the caller can fix (auth, 4xx)."""


class TransientError(ObserveError):
    """A weco-side or network blip that persisted through retries."""


# 408/429 and every 5xx are worth retrying; the remaining 4xx are the caller's.
_RETRYABLE_STATUS = {408, 429}
_TIMEOUT = (5, 30)  # (connect, read); read stays generous for large source uploads
_BACKOFF_BASE = 1.0  # seconds, doubled per attempt, with jitter
_RETRY_AFTER_CAP = 10.0  # a Retry-After header must not stall the caller's loop


def _is_retryable_status(status: int) -> bool:
    return status >= 500 or status in _RETRYABLE_STATUS


def _retry_delay(response: requests.Response | None, attempt: int) -> float:
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return min(float(retry_after), _RETRY_AFTER_CAP)
            except ValueError:
                pass
    return _BACKOFF_BASE * (2**attempt) * random.uniform(0.5, 1.5)


def _status_description(response: requests.Response) -> str:
    """Describe a retryable error status, surfacing the API's `detail` when present."""
    detail = None
    try:
        body = response.json()
        if isinstance(body, dict):
            detail = body.get("detail")
    except Exception:
        pass
    base = f"HTTP {response.status_code} from weco"
    return f"{base} ({detail})" if detail else base


def _post(url: str, *, payload: dict[str, Any], headers: dict[str, str], attempts: int, what: str) -> dict:
    """POST with bounded retries on retryable failures.

    Retrying after an ambiguous failure (e.g. a timeout where the write may
    have landed) is safe only because the API upserts steps by (run_id, step).
    """
    failure: Exception | None = None
    for attempt in range(attempts):
        response = None
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=_TIMEOUT)
            if _is_retryable_status(response.status_code):
                failure = TransientError(f"{what}: {_status_description(response)}")
            else:
                response.raise_for_status()
                # A 2xx whose body isn't a JSON object (ValueError below, or a
                # proxy/LB interstitial serving JSON of the wrong shape) is as
                # weco-side as a 5xx — it falls through to transient.
                body = response.json()
                if isinstance(body, dict):
                    return body
                failure = TransientError(f"{what}: unexpected response body from weco")
        except requests.HTTPError as e:
            raise CallerError(f"{what}: {api_error_message(e)}") from e
        except Exception as e:
            failure = e
        if attempt + 1 < attempts:
            time.sleep(_retry_delay(response, attempt))
    if isinstance(failure, TransientError):
        raise failure
    raise TransientError(f"{what}: {failure}") from failure


def create_run(
    *,
    source_code: dict[str, str],
    metric_name: str,
    maximize: bool,
    name: str | None = None,
    additional_instructions: str | None = None,
    metadata: dict[str, Any] | None = None,
    auth_headers: dict[str, str],
) -> dict:
    """Create an external run. Returns the response dict; raises ObserveError on failure."""
    payload: dict[str, Any] = {"source_code": source_code, "metric_name": metric_name, "maximize": maximize}
    if name is not None:
        payload["name"] = name
    if additional_instructions is not None:
        payload["additional_instructions"] = additional_instructions
    if metadata:
        payload["metadata"] = metadata

    # Runs once, before the caller's loop starts, so extra attempts are cheap.
    return _post(
        f"{__base_url__}/external/runs", payload=payload, headers=auth_headers, attempts=3, what="failed to create run"
    )


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
) -> dict:
    """Log a step for an external run. Returns the response dict; raises ObserveError on failure."""
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

    # Runs inside the tracked loop: one retry keeps the added latency bounded.
    return _post(
        f"{__base_url__}/external/runs/{run_id}/steps",
        payload=payload,
        headers=auth_headers,
        attempts=2,
        what=f"failed to log step {step}",
    )
