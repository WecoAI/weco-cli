"""Backward-compatible free functions wrapping ``core.api.WecoClient``.

All existing callers (``optimizer.py``, ``share.py``, ``credits.py``,
``auth.py``) continue to work without changes.  New code should import
``WecoClient`` from ``weco.core.api`` directly.
"""

from typing import Any, Dict, Optional, Tuple, Union

import requests
from rich.console import Console

from weco import __base_url__

# Re-export everything from core.api so ``from weco.api import X`` keeps working.
from .core.api import (  # noqa: F401
    WecoClient,
    RunSummary,
    ExecutionTasksResult,
    handle_api_error,
    _truncate_output,
)

# Keep truncate_output importable under its old name for callers like utils.py tests.
truncate_output = _truncate_output


# ---------------------------------------------------------------------------
# Legacy free functions — delegate to WecoClient
# ---------------------------------------------------------------------------


def start_optimization_run(
    console: Console,
    source_code: dict[str, str],
    source_path: str | None,
    evaluation_command: str,
    metric_name: str,
    maximize: bool,
    steps: int,
    code_generator_config: Dict[str, Any],
    evaluator_config: Dict[str, Any],
    search_policy_config: Dict[str, Any],
    additional_instructions: str = None,
    eval_timeout: Optional[int] = None,
    save_logs: bool = False,
    log_dir: str = ".runs",
    auth_headers: dict = {},
    timeout: Union[int, Tuple[int, int]] = (10, 3650),
    api_keys: Optional[Dict[str, str]] = None,
    require_review: bool = False,
    installation_id: Optional[str] = None,
    invocation_id: Optional[str] = None,
    invoked_via: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Start the optimization run."""
    with console.status("[bold green]Starting Optimization..."):
        try:
            return WecoClient(auth_headers).start_run(
                source_code=source_code,
                source_path=source_path,
                evaluation_command=evaluation_command,
                metric_name=metric_name,
                maximize=maximize,
                steps=steps,
                code_generator_config=code_generator_config,
                evaluator_config=evaluator_config,
                search_policy_config=search_policy_config,
                additional_instructions=additional_instructions,
                eval_timeout=eval_timeout,
                save_logs=save_logs,
                log_dir=log_dir,
                api_keys=api_keys,
                require_review=require_review,
                installation_id=installation_id,
                invocation_id=invocation_id,
                invoked_via=invoked_via,
            )
        except requests.exceptions.HTTPError as e:
            handle_api_error(e, console)
            return None
        except Exception as e:
            console.print(f"[bold red]Error starting run: {e}[/]")
            return None


def resume_optimization_run(
    console: Console,
    run_id: str,
    auth_headers: dict = {},
    api_keys: Optional[Dict[str, str]] = None,
    timeout: Union[int, Tuple[int, int]] = (5, 10),
) -> Optional[Dict[str, Any]]:
    """Request the backend to resume an interrupted run."""
    with console.status("[bold green]Resuming run..."):
        try:
            return WecoClient(auth_headers).resume_run(run_id, api_keys=api_keys)
        except requests.exceptions.HTTPError as e:
            handle_api_error(e, console)
            return None
        except Exception as e:
            console.print(f"[bold red]Error resuming run: {e}[/]")
            return None


def evaluate_feedback_then_suggest_next_solution(
    console: Console,
    run_id: str,
    step: int,
    execution_output: str,
    auth_headers: dict = {},
    timeout: Union[int, Tuple[int, int]] = (10, 3650),
    api_keys: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Evaluate the feedback and suggest the next solution."""
    client = WecoClient(auth_headers)
    try:
        return client.suggest(run_id, execution_output=execution_output, step=step, api_keys=api_keys)
    except requests.exceptions.ReadTimeout:
        # Original behavior: re-raise without printing — caller handles resume.
        raise
    except requests.exceptions.ConnectionError as e:
        # Original behavior: show structured error via handle_api_error.
        handle_api_error(e, console)
        raise
    except requests.exceptions.HTTPError as e:
        handle_api_error(e, console)
        raise
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/]")
        raise


def get_optimization_run_status(
    console: Console,
    run_id: str,
    include_history: bool = False,
    auth_headers: dict = {},
    timeout: Union[int, Tuple[int, int]] = (5, 10),
) -> Dict[str, Any]:
    """Get the current status of the optimization run."""
    client = WecoClient(auth_headers)
    try:
        return client.get_run_status(run_id, include_history=include_history)
    except requests.exceptions.HTTPError as e:
        handle_api_error(e, console)
        raise
    except Exception as e:
        console.print(f"[bold red]Error getting run status: {e}[/]")
        raise


def send_heartbeat(run_id: str, auth_headers: dict = {}, timeout: Union[int, Tuple[int, int]] = (5, 10)) -> bool:
    """Send a heartbeat signal to the backend."""
    return WecoClient(auth_headers).heartbeat(run_id)


def report_termination(
    run_id: str,
    status_update: str,
    reason: str,
    details: Optional[str] = None,
    auth_headers: dict = {},
    timeout: Union[int, Tuple[int, int]] = (5, 10),
) -> bool:
    """Report the termination reason to the backend."""
    return WecoClient(auth_headers).terminate(run_id, status_update=status_update, reason=reason, details=details)


# --- Execution Queue API Functions ---


def get_execution_tasks(
    run_id: str, auth_headers: dict = {}, timeout: Union[int, Tuple[int, int]] = (5, 30)
) -> Optional[ExecutionTasksResult]:
    """Poll for ready execution tasks."""
    return WecoClient(auth_headers).get_execution_tasks(run_id)


def claim_execution_task(
    task_id: str, auth_headers: dict = {}, timeout: Union[int, Tuple[int, int]] = (5, 30)
) -> Optional[Dict[str, Any]]:
    """Claim an execution task."""
    return WecoClient(auth_headers).claim_task(task_id)


def submit_execution_result(
    run_id: str,
    task_id: str,
    execution_output: str,
    auth_headers: dict = {},
    timeout: Union[int, Tuple[int, int]] = (10, 3650),
    api_keys: Optional[Dict[str, str]] = None,
) -> Optional[Dict[str, Any]]:
    """Submit execution result for a task."""
    client = WecoClient(auth_headers)
    try:
        return client.suggest(run_id, execution_output=execution_output, task_id=task_id, api_keys=api_keys)
    except Exception:
        return None


# --- Share API Functions ---


def create_share_link(
    console: Console, run_id: str, auth_headers: dict = {}, timeout: Union[int, Tuple[int, int]] = (5, 30)
) -> Optional[str]:
    """Create a public share link for a run."""
    try:
        resp = requests.post(f"{__base_url__}/runs/{run_id}/share", headers=auth_headers, timeout=timeout)
        resp.raise_for_status()
        result = resp.json()
        return result.get("share_id")
    except requests.exceptions.HTTPError as e:
        handle_api_error(e, console)
        return None
    except Exception as e:
        console.print(f"[bold red]Error creating share link: {e}[/]")
        return None
