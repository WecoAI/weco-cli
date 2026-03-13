"""WecoObserver — Python SDK for tracking external optimization runs.

All public methods are designed to never raise exceptions. Errors are
reported via warnings.warn() so the user's optimization loop is never
interrupted by observability failures.
"""

import warnings
from typing import Any

from weco.config import load_weco_api_key

from . import api


class ObserveRun:
    """Handle for an active external run. Returned by WecoObserver.create_run()."""

    def __init__(self, run_id: str, auth_headers: dict[str, str]):
        self._run_id = run_id
        self._auth_headers = auth_headers

    @property
    def run_id(self) -> str:
        return self._run_id

    def log_step(
        self,
        step: int,
        *,
        status: str = "completed",
        description: str | None = None,
        metrics: dict[str, float] | None = None,
        code: dict[str, str] | None = None,
        parent_step: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log a step to this run. Never raises."""
        api.log_step(
            run_id=self._run_id,
            step=step,
            status=status,
            description=description,
            metrics=metrics,
            code=code,
            parent_step=parent_step,
            metadata=metadata,
            auth_headers=self._auth_headers,
        )


class WecoObserver:
    """Client for creating and managing external optimization runs.

    Usage:
        obs = WecoObserver()
        run = obs.create_run(
            source_code={"train.py": open("train.py").read()},
            primary_metric="val_bpb",
            maximize=False,
        )
        run.log_step(step=1, metrics={"val_bpb": 1.03})
    """

    def __init__(self, api_key: str | None = None):
        """Initialize the observer.

        Args:
            api_key: Weco API key. If not provided, reads from config/env.
        """
        if api_key is None:
            api_key = load_weco_api_key()
        if not api_key:
            warnings.warn("weco observe: no API key found. Run `weco login` first.", stacklevel=2)
        self._auth_headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def create_run(
        self,
        *,
        source_code: dict[str, str],
        primary_metric: str,
        maximize: bool = False,
        name: str | None = None,
        additional_instructions: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ObserveRun | None:
        """Create a new external run.

        Returns an ObserveRun handle, or None if creation failed.
        """
        result = api.create_run(
            source_code=source_code,
            metric_name=primary_metric,
            maximize=maximize,
            name=name,
            additional_instructions=additional_instructions,
            metadata=metadata,
            auth_headers=self._auth_headers,
        )
        if result is None:
            return None
        return ObserveRun(run_id=result["run_id"], auth_headers=self._auth_headers)
