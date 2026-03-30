"""``weco observe init`` and ``weco observe log``

Also contains the Python SDK classes (``WecoObserver``, ``ObserveRun``)
for programmatic use.

All CLI handlers follow the fire-and-forget pattern: they print warnings to
stderr on failure but always exit 0 so they never crash an agent's loop.
"""

import argparse
import json
import sys
import warnings
from typing import Any

from rich.console import Console

from ..core.api import WecoClient
from ..core.auth import handle_authentication
from ..core.browser import open_browser
from ..core.config import load_weco_api_key
from ..core.events import send_event, ObserveInitEvent, ObserveLogEvent
from .. import __dashboard_url__


# ---------------------------------------------------------------------------
# Python SDK
# ---------------------------------------------------------------------------


class ObserveRun:
    """Handle for an active external run. Returned by ``WecoObserver.create_run()``."""

    def __init__(self, run_id: str, client: WecoClient):
        self._run_id = run_id
        self._client = client

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
        self._client.log_external_step(
            self._run_id,
            step=step,
            status=status,
            description=description,
            metrics=metrics,
            code=code,
            parent_step=parent_step,
            metadata=metadata,
        )


class WecoObserver:
    """Client for creating and managing external optimization runs.

    Usage::

        obs = WecoObserver()
        run = obs.create_run(source_code={"train.py": open("train.py").read()}, primary_metric="val_bpb", maximize=False)
        run.log_step(step=1, metrics={"val_bpb": 1.03})
    """

    def __init__(self, api_key: str | None = None):
        if api_key is None:
            api_key = load_weco_api_key()
        if not api_key:
            warnings.warn("weco observe: no API key found. Run `weco login` first.", stacklevel=2)
        auth_headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = WecoClient(auth_headers)

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
        """Create a new external run. Returns an ``ObserveRun`` handle, or None on failure."""
        result = self._client.create_external_run(
            source_code=source_code,
            metric_name=primary_metric,
            maximize=maximize,
            name=name,
            additional_instructions=additional_instructions,
            metadata=metadata,
        )
        if result is None:
            return None
        return ObserveRun(run_id=result["run_id"], client=self._client)


# ---------------------------------------------------------------------------
# CLI handlers
# ---------------------------------------------------------------------------


def _read_code_files(paths: list[str]) -> dict[str, str]:
    """Read source code files from disk."""
    source_code = {}
    for path in paths:
        try:
            with open(path) as f:
                source_code[path] = f.read()
        except FileNotFoundError:
            warnings.warn(f"weco observe: file not found: {path}", stacklevel=2)
        except Exception as e:
            warnings.warn(f"weco observe: error reading {path}: {e}", stacklevel=2)
    return source_code


def handle(args: argparse.Namespace, console: Console) -> None:
    """Execute an observe subcommand. Always exits 0."""
    del console  # unused

    if not args.observe_command:
        print("Usage: weco observe {init,log}", file=sys.stderr)
        sys.exit(0)

    try:
        _, auth_headers = handle_authentication(None)
        if not auth_headers:
            print("weco observe: not logged in. Run `weco login` first.", file=sys.stderr)
            sys.exit(0)
    except Exception as e:
        print(f"weco observe: authentication failed: {e}", file=sys.stderr)
        sys.exit(0)

    client = WecoClient(auth_headers)

    if args.observe_command == "init":
        _handle_init(args, client)
    elif args.observe_command == "log":
        _handle_log(args, client)


def _handle_init(args: argparse.Namespace, client: WecoClient) -> None:
    source_arg = args.sources if args.sources is not None else [args.source]
    source_code = _read_code_files(source_arg)
    if not source_code:
        print("weco observe: no source files could be read", file=sys.stderr)
        sys.exit(0)

    maximize = args.goal in ("maximize", "max")

    send_event(
        ObserveInitEvent(metric=args.metric, goal="maximize" if maximize else "minimize", source_count=len(source_code))
    )

    result = client.create_external_run(
        source_code=source_code,
        metric_name=args.metric,
        maximize=maximize,
        name=args.name,
        additional_instructions=args.additional_instructions,
    )

    if result and result.get("run_id"):
        run_id = result["run_id"]
        print(run_id)
        open_browser(f"{__dashboard_url__}/runs/{run_id}")
    else:
        print("weco observe: failed to create run", file=sys.stderr)


def _handle_log(args: argparse.Namespace, client: WecoClient) -> None:
    metrics = {}
    if args.metrics:
        try:
            metrics = json.loads(args.metrics)
        except json.JSONDecodeError as e:
            print(f"weco observe: invalid metrics JSON: {e}", file=sys.stderr)
            sys.exit(0)

    code = None
    source_arg = args.sources if args.sources is not None else ([args.source] if args.source else None)
    if source_arg:
        code = _read_code_files(source_arg)

    send_event(ObserveLogEvent(status=args.status))

    client.log_external_step(
        args.run_id,
        step=args.step,
        status=args.status,
        description=args.description,
        metrics=metrics,
        code=code,
        parent_step=args.parent_step,
    )
