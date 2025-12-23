import sys
from typing import Dict, Any, Optional, Union, Tuple
import requests
from rich.console import Console

from weco import __pkg_version__, __base_url__
from .utils import truncate_output


def handle_api_error(e: requests.exceptions.HTTPError, console: Console) -> None:
    """Extract and display error messages from API responses in a structured format."""
    status = getattr(e.response, "status_code", None)
    try:
        payload = e.response.json()
        detail = payload.get("detail", payload)
    except (ValueError, AttributeError):
        detail = getattr(e.response, "text", "") or f"HTTP {status} Error"

    def _render(detail_obj: Any) -> None:
        if isinstance(detail_obj, str):
            console.print(f"[bold red]{detail_obj}[/]")
        elif isinstance(detail_obj, dict):
            # Try common message keys in order of preference
            message_keys = ("message", "error", "msg", "detail")
            message = next((detail_obj.get(key) for key in message_keys if detail_obj.get(key)), None)
            suggestion = detail_obj.get("suggestion")
            if message:
                console.print(f"[bold red]{message}[/]")
            else:
                console.print(f"[bold red]HTTP {status} Error[/]")
            if suggestion:
                console.print(f"[yellow]{suggestion}[/]")
            extras = {
                k: v
                for k, v in detail_obj.items()
                if k not in {"message", "error", "msg", "detail", "suggestion"} and v not in (None, "")
            }
            for key, value in extras.items():
                console.print(f"[dim]{key}: {value}[/]")
        elif isinstance(detail_obj, list) and detail_obj:
            _render(detail_obj[0])
            for extra in detail_obj[1:]:
                console.print(f"[yellow]{extra}[/]")
        else:
            console.print(f"[bold red]{detail_obj or f'HTTP {status} Error'}[/]")

    _render(detail)


def _recover_suggest_after_transport_error(
    console: Console, run_id: str, step: int, auth_headers: dict
) -> Optional[Dict[str, Any]]:
    """
    Try to reconstruct the /suggest response after a transport error (ReadTimeout/502/RemoteDisconnected)
    by fetching run status and using the latest nodes.

    Args:
        console: The console object to use for logging.
        run_id: The ID of the run to recover.
        step: The step of the solution to recover.
        auth_headers: The authentication headers to use for the request.

    Returns:
        The recovered response if the run is in a valid state, otherwise None.
    """
    run_status_recovery_response = get_optimization_run_status(
        console=console, run_id=run_id, include_history=True, auth_headers=auth_headers
    )
    current_step = run_status_recovery_response.get("current_step")
    current_status = run_status_recovery_response.get("status")
    # The run should be "running" and the current step should correspond to the solution step we are attempting to generate
    is_valid_run_state = current_status is not None and current_status == "running"
    is_valid_step = current_step is not None and current_step == step
    if is_valid_run_state and is_valid_step:
        nodes = run_status_recovery_response.get("nodes") or []
        # We need at least 2 nodes to reconstruct the expected response i.e., the last two nodes
        if len(nodes) >= 2:
            nodes_sorted_ascending = sorted(nodes, key=lambda n: n["step"])
            latest_node = nodes_sorted_ascending[-1]
            penultimate_node = nodes_sorted_ascending[-2]
            # If the server finished generating the next candidate, it should be exactly this step
            if latest_node and latest_node["step"] == step:
                # Try to reconstruct the expected response from the /suggest endpoint using the run status info
                return {
                    "run_id": run_id,
                    "previous_solution_metric_value": penultimate_node.get("metric_value"),
                    "solution_id": latest_node.get("solution_id"),
                    "code": latest_node.get("code"),
                    "plan": latest_node.get("plan"),
                    "is_done": False,
                }
    return None


def start_optimization_run(
    console: Console,
    source_code: str,
    source_path: str,
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
) -> Optional[Dict[str, Any]]:
    """Start the optimization run."""
    with console.status("[bold green]Starting Optimization..."):
        try:
            request_json = {
                "source_code": source_code,
                "source_path": source_path,
                "additional_instructions": additional_instructions,
                "objective": {"evaluation_command": evaluation_command, "metric_name": metric_name, "maximize": maximize},
                "optimizer": {
                    "steps": steps,
                    "code_generator": code_generator_config,
                    "evaluator": evaluator_config,
                    "search_policy": search_policy_config,
                },
                "eval_timeout": eval_timeout,
                "save_logs": save_logs,
                "log_dir": log_dir,
                "metadata": {"client_name": "cli", "client_version": __pkg_version__},
            }
            if api_keys:
                request_json["api_keys"] = api_keys

            response = requests.post(f"{__base_url__}/runs/", json=request_json, headers=auth_headers, timeout=timeout)
            response.raise_for_status()
            result = response.json()
            # Handle None values for code and plan fields
            if result.get("plan") is None:
                result["plan"] = ""
            if result.get("code") is None:
                result["code"] = ""
            return result
        except requests.exceptions.HTTPError as e:
            handle_api_error(e, console)
            return None
        except Exception as e:
            console.print(f"[bold red]Error starting run: {e}[/]")
            return None


def resume_optimization_run(
    console: Console, run_id: str, auth_headers: dict = {}, timeout: Union[int, Tuple[int, int]] = (5, 10)
) -> Optional[Dict[str, Any]]:
    """Request the backend to resume an interrupted run."""
    with console.status("[bold green]Resuming run..."):
        try:
            request_json = {"metadata": {"client_name": "cli", "client_version": __pkg_version__}}

            response = requests.post(
                f"{__base_url__}/runs/{run_id}/resume", json=request_json, headers=auth_headers, timeout=timeout
            )
            response.raise_for_status()
            result = response.json()
            return result
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
    try:
        # Truncate the execution output before sending to backend
        truncated_output = truncate_output(execution_output)

        request_json = {"execution_output": truncated_output, "metadata": {}}
        if api_keys:
            request_json["api_keys"] = api_keys

        response = requests.post(
            f"{__base_url__}/runs/{run_id}/suggest", json=request_json, headers=auth_headers, timeout=timeout
        )
        response.raise_for_status()
        result = response.json()
        # Handle None values for code and plan fields
        if result.get("plan") is None:
            result["plan"] = ""
        if result.get("code") is None:
            result["code"] = ""
        return result
    except requests.exceptions.ReadTimeout as e:
        # ReadTimeout can mean either:
        # 1) the server truly didn't finish before the client's read timeout, or
        # 2) the server finished but an intermediary (proxy/LB) dropped the response.
        # We only try to recover case (2): fetch run status to confirm the step completed and reconstruct the response.
        recovered = _recover_suggest_after_transport_error(
            console=console, run_id=run_id, step=step, auth_headers=auth_headers
        )
        if recovered is not None:
            return recovered
        # If we cannot confirm completion, bubble up the timeout so the caller can resume later.
        raise requests.exceptions.ReadTimeout(e)
    except requests.exceptions.HTTPError as e:
        # Treat only 502 Bad Gateway as a transient transport/gateway issue (akin to a dropped response).
        # For 502, attempt the status-based recovery method used for ReadTimeout errors; otherwise render the HTTP error normally.
        if (resp := getattr(e, "response", None)) is not None and resp.status_code == 502:
            recovered = _recover_suggest_after_transport_error(
                console=console, run_id=run_id, step=step, auth_headers=auth_headers
            )
            if recovered is not None:
                return recovered
        # Surface non-502 HTTP errors to the user.
        handle_api_error(e, console)
        raise
    except requests.exceptions.ConnectionError as e:
        # Covers connection resets with no HTTP response (e.g., RemoteDisconnected).
        # Treat as a potential "response lost after completion": try status-based recovery first similar to how ReadTimeout errors are handled.
        recovered = _recover_suggest_after_transport_error(
            console=console, run_id=run_id, step=step, auth_headers=auth_headers
        )
        if recovered is not None:
            return recovered
        # Surface the connection error to the user.
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
    try:
        response = requests.get(
            f"{__base_url__}/runs/{run_id}", params={"include_history": include_history}, headers=auth_headers, timeout=timeout
        )
        response.raise_for_status()
        result = response.json()
        # Handle None values for code and plan fields in best_result and nodes
        if result.get("best_result"):
            if result["best_result"].get("code") is None:
                result["best_result"]["code"] = ""
            if result["best_result"].get("plan") is None:
                result["best_result"]["plan"] = ""
        # Handle None values for code and plan fields in nodes array
        if result.get("nodes"):
            for i, node in enumerate(result["nodes"]):
                if node.get("plan") is None:
                    result["nodes"][i]["plan"] = ""
                if node.get("code") is None:
                    result["nodes"][i]["code"] = ""
        return result
    except requests.exceptions.HTTPError as e:
        handle_api_error(e, console)
        raise
    except Exception as e:
        console.print(f"[bold red]Error getting run status: {e}[/]")
        raise


def send_heartbeat(run_id: str, auth_headers: dict = {}, timeout: Union[int, Tuple[int, int]] = (5, 10)) -> bool:
    """Send a heartbeat signal to the backend."""
    try:
        response = requests.put(f"{__base_url__}/runs/{run_id}/heartbeat", headers=auth_headers, timeout=timeout)
        response.raise_for_status()
        return True
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 409:
            print(f"Polling ignore: Run {run_id} is not running.", file=sys.stderr)
        else:
            print(f"Polling failed for run {run_id}: HTTP {e.response.status_code}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Error sending heartbeat for run {run_id}: {e}", file=sys.stderr)
        return False


def report_termination(
    run_id: str,
    status_update: str,
    reason: str,
    details: Optional[str] = None,
    auth_headers: dict = {},
    timeout: Union[int, Tuple[int, int]] = (5, 10),
) -> bool:
    """Report the termination reason to the backend."""
    try:
        response = requests.post(
            f"{__base_url__}/runs/{run_id}/terminate",
            json={"status_update": status_update, "termination_reason": reason, "termination_details": details},
            headers=auth_headers,
            timeout=timeout,
        )
        response.raise_for_status()
        return True
    except Exception as e:
        print(f"Warning: Failed to report termination to backend for run {run_id}: {e}", file=sys.stderr)
        return False
