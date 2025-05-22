from typing import Dict, Any, Optional
import rich
import requests
from weco import __pkg_version__, __base_url__
import sys
from rich.console import Console


def handle_api_error(e: requests.exceptions.HTTPError, console: rich.console.Console) -> None:
    """Extract and display error messages from API responses in a structured format. Does not exit."""
    try:
        error_data = e.response.json()
        # Handle structured detail from FastAPI (which can be a dict or string)
        detail = error_data.get("detail", e.response.text)
        if isinstance(detail, dict) and "message" in detail:
            detail_message = detail["message"]
        elif isinstance(detail, str):
            detail_message = detail
        else:  # Fallback for unexpected detail structure
            detail_message = e.response.text
    except (ValueError, KeyError):  # Handle cases where response is not JSON or detail key is missing
        detail_message = f"HTTP {e.response.status_code} Error: {e.response.text}"

    console.print(f"[bold red]Error: {detail_message}[/]")
    # Removed sys.exit(1) - let caller handle control flow.


def start_optimization_run(
    console: rich.console.Console,
    source_code: str,
    evaluation_command: str,
    metric_name: str,
    maximize: bool,
    steps: int,
    code_generator_config: Dict[str, Any],
    evaluator_config: Dict[str, Any],
    search_policy_config: Dict[str, Any],
    additional_instructions: str = None,
    api_keys: Dict[str, Any] = {},
    auth_headers: dict = {},
    timeout: int = 800,
) -> Dict[str, Any]:
    """Start the optimization run."""
    with console.status("[bold green]Starting Optimization..."):
        try:
            response = requests.post(
                f"{__base_url__}/runs",
                json={
                    "source_code": source_code,
                    "additional_instructions": additional_instructions,
                    "objective": {"evaluation_command": evaluation_command, "metric_name": metric_name, "maximize": maximize},
                    "optimizer": {
                        "steps": steps,
                        "code_generator": code_generator_config,
                        "evaluator": evaluator_config,
                        "search_policy": search_policy_config,
                    },
                    "metadata": {"client_name": "cli", "client_version": __pkg_version__, **api_keys},
                },
                headers=auth_headers,  # Add headers
                timeout=timeout,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            handle_api_error(e, console)
            sys.exit(1)
        except requests.exceptions.RequestException as e:
            console.print(f"[bold red]Network Error starting run: {e}[/]")
            sys.exit(1)


def evaluate_feedback_then_suggest_next_solution(
    run_id: str,
    execution_output: str,
    additional_instructions: str = None,
    api_keys: Dict[str, Any] = {},
    auth_headers: dict = {},
    timeout: int = 800,
) -> Dict[str, Any]:
    """Evaluate the feedback and suggest the next solution."""
    try:
        response = requests.post(
            f"{__base_url__}/runs/{run_id}/suggest",
            json={
                "execution_output": execution_output,
                "additional_instructions": additional_instructions,
                "metadata": {**api_keys},
            },
            headers=auth_headers,  # Add headers
            timeout=timeout,
        )
        # Check for 402 explicitly BEFORE raise_for_status
        if response.status_code == 402:
            try:
                # Return the JSON error payload for the CLI to inspect
                return response.json()
            except ValueError:  # Not JSON, though our backend should send JSON
                return {
                    "error": "api_key_required_unknown_format",
                    "status_code": response.status_code,
                    "detail": response.text,
                }

        response.raise_for_status()  # For other errors (4xx, 5xx)
        return response.json()
    except requests.exceptions.HTTPError as e:
        handle_api_error(e, Console())  # Print the error
        raise  # Re-raise HTTPError to be caught by cli.py's main handler
    except requests.exceptions.RequestException as e:
        console = Console()  # Create a console instance if not passed
        console.print(f"[bold red]Network Error during suggest: {e}[/]")
        raise  # Re-raise RequestException to be caught by cli.py's main handler


def get_optimization_run_status(
    run_id: str, include_history: bool = False, auth_headers: dict = {}, timeout: int = 800
) -> Dict[str, Any]:
    """Get the current status of the optimization run."""
    try:
        response = requests.get(
            f"{__base_url__}/runs/{run_id}", params={"include_history": include_history}, headers=auth_headers, timeout=timeout
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        handle_api_error(e, Console())  # Use default console
        raise  # Re-raise
    except requests.exceptions.RequestException as e:
        print(f"Network Error getting status: {e}")
        raise  # Re-raise


def send_heartbeat(run_id: str, auth_headers: dict = {}, timeout: int = 10) -> bool:
    """Send a heartbeat signal to the backend."""
    try:
        response = requests.put(f"{__base_url__}/runs/{run_id}/heartbeat", headers=auth_headers, timeout=timeout)
        response.raise_for_status()
        return True
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 409:
            print(f"Heartbeat ignored: Run {run_id} is not running.", file=sys.stderr)
        else:
            print(f"Heartbeat failed for run {run_id}: HTTP {e.response.status_code}", file=sys.stderr)
        return False
    except requests.exceptions.RequestException as e:
        print(f"Heartbeat network error for run {run_id}: {e}", file=sys.stderr)
        return False


def report_termination(
    run_id: str, status_update: str, reason: str, details: Optional[str] = None, auth_headers: dict = {}, timeout: int = 30
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
    except requests.exceptions.RequestException as e:
        print(f"Warning: Failed to report termination to backend for run {run_id}: {e}", file=sys.stderr)
        return False
