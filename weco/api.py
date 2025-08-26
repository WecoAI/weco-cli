import sys
from typing import Dict, Any, Optional, Union, Tuple, List
import requests
from rich.console import Console

from weco import __pkg_version__, __base_url__
from .constants import DEFAULT_API_TIMEOUT


def handle_api_error(e: requests.exceptions.HTTPError, console: Console) -> None:
    """Extract and display error messages from API responses in a structured format."""
    try:
        detail = e.response.json()["detail"]
    except (ValueError, KeyError):  # Handle cases where response is not JSON or detail key is missing
        detail = f"HTTP {e.response.status_code} Error: {e.response.text}"
    console.print(f"[bold red]{detail}[/]")


def start_optimization_run(
    console: Console,
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
    source_path: Optional[str] = None,
    eval_timeout: Optional[int] = None,
    save_logs: bool = False,
    log_dir: str = ".runs",
    auth_headers: dict = {},
    timeout: Union[int, Tuple[int, int]] = DEFAULT_API_TIMEOUT,
) -> Optional[Dict[str, Any]]:
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
                    "metadata": {
                        "client_name": "cli",
                        "client_version": __pkg_version__,
                        "source_path": source_path,
                        "eval_timeout": eval_timeout,
                        "save_logs": save_logs,
                        "log_dir": log_dir,
                        **api_keys,
                    },
                },
                headers=auth_headers,
                timeout=timeout,
            )
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


def evaluate_feedback_then_suggest_next_solution(
    console: Console,
    run_id: str,
    execution_output: str,
    additional_instructions: str = None,
    api_keys: Dict[str, Any] = {},
    auth_headers: dict = {},
    timeout: Union[int, Tuple[int, int]] = DEFAULT_API_TIMEOUT,
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
            headers=auth_headers,
            timeout=timeout,
        )
        response.raise_for_status()
        result = response.json()
        # Handle None values for code and plan fields
        if result.get("plan") is None:
            result["plan"] = ""
        if result.get("code") is None:
            result["code"] = ""

        return result
    except requests.exceptions.HTTPError as e:
        # Allow caller to handle suggest errors, maybe retry or terminate
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
    timeout: Union[int, Tuple[int, int]] = DEFAULT_API_TIMEOUT,
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


def send_heartbeat(run_id: str, auth_headers: dict = {}, timeout: Union[int, Tuple[int, int]] = (10, 10)) -> bool:
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
    timeout: Union[int, Tuple[int, int]] = (10, 30),
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


# --- Chatbot API Functions ---
def _determine_model_and_api_key() -> tuple[str, dict[str, str]]:
    """Determine the model and API key to use based on available environment variables.

    Uses the shared model selection logic to maintain consistency.
    Returns (model_name, api_key_dict)
    """
    from .utils import read_api_keys_from_env, determine_default_model

    llm_api_keys = read_api_keys_from_env()
    model = determine_default_model(llm_api_keys)

    # Create API key dictionary with only the key for the selected model
    if model == "o4-mini":
        api_key_dict = {"OPENAI_API_KEY": llm_api_keys["OPENAI_API_KEY"]}
    elif model == "claude-sonnet-4-0":
        api_key_dict = {"ANTHROPIC_API_KEY": llm_api_keys["ANTHROPIC_API_KEY"]}
    elif model == "gemini-2.5-pro":
        api_key_dict = {"GEMINI_API_KEY": llm_api_keys["GEMINI_API_KEY"]}
    else:
        # This should never happen if determine_default_model works correctly
        raise ValueError(f"Unknown default model choice: {model}")

    return model, api_key_dict


def get_optimization_suggestions_from_codebase(
    console: Console,
    gitingest_summary: str,
    gitingest_tree: str,
    gitingest_content_str: str,
    auth_headers: dict = {},
    timeout: Union[int, Tuple[int, int]] = DEFAULT_API_TIMEOUT,
) -> Optional[List[Dict[str, Any]]]:
    """Analyze codebase and get optimization suggestions using the model-agnostic backend API."""
    try:
        model, api_key_dict = _determine_model_and_api_key()
        response = requests.post(
            f"{__base_url__}/onboard/analyze-codebase",
            json={
                "gitingest_summary": gitingest_summary,
                "gitingest_tree": gitingest_tree,
                "gitingest_content": gitingest_content_str,
                "model": model,
                "metadata": api_key_dict,
            },
            headers=auth_headers,
            timeout=timeout,
        )
        response.raise_for_status()
        result = response.json()
        return [option for option in result.get("options", [])]

    except requests.exceptions.HTTPError as e:
        handle_api_error(e, console)
        return None
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/]")
        return None


def generate_evaluation_script_and_metrics(
    console: Console,
    target_file: str,
    description: str,
    gitingest_content_str: str,
    auth_headers: dict = {},
    timeout: Union[int, Tuple[int, int]] = DEFAULT_API_TIMEOUT,
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Generate evaluation script and determine metrics using the model-agnostic backend API."""
    try:
        model, api_key_dict = _determine_model_and_api_key()
        response = requests.post(
            f"{__base_url__}/onboard/generate-script",
            json={
                "target_file": target_file,
                "description": description,
                "gitingest_content": gitingest_content_str,
                "model": model,
                "metadata": api_key_dict,
            },
            headers=auth_headers,
            timeout=timeout,
        )
        response.raise_for_status()
        result = response.json()
        return result.get("script_content"), result.get("metric_name"), result.get("goal"), result.get("reasoning")
    except requests.exceptions.HTTPError as e:
        handle_api_error(e, console)
        return None, None, None, None
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/]")
        return None, None, None, None


def analyze_evaluation_environment(
    console: Console,
    target_file: str,
    description: str,
    gitingest_summary: str,
    gitingest_tree: str,
    gitingest_content_str: str,
    auth_headers: dict = {},
    timeout: Union[int, Tuple[int, int]] = DEFAULT_API_TIMEOUT,
) -> Optional[Dict[str, Any]]:
    """Analyze existing evaluation scripts and environment using the model-agnostic backend API."""
    try:
        model, api_key_dict = _determine_model_and_api_key()
        response = requests.post(
            f"{__base_url__}/onboard/analyze-environment",
            json={
                "target_file": target_file,
                "description": description,
                "gitingest_summary": gitingest_summary,
                "gitingest_tree": gitingest_tree,
                "gitingest_content": gitingest_content_str,
                "model": model,
                "metadata": api_key_dict,
            },
            headers=auth_headers,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.HTTPError as e:
        handle_api_error(e, console)
        return None
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/]")
        return None


def analyze_script_execution_requirements(
    console: Console,
    script_content: str,
    script_path: str,
    target_file: str,
    auth_headers: dict = {},
    timeout: Union[int, Tuple[int, int]] = DEFAULT_API_TIMEOUT,
) -> Optional[str]:
    """Analyze script to determine proper execution command using the model-agnostic backend API."""
    try:
        model, api_key_dict = _determine_model_and_api_key()
        response = requests.post(
            f"{__base_url__}/onboard/analyze-script",
            json={
                "script_content": script_content,
                "script_path": script_path,
                "target_file": target_file,
                "model": model,
                "metadata": api_key_dict,
            },
            headers=auth_headers,
            timeout=timeout,
        )
        response.raise_for_status()
        result = response.json()
        return result.get("command", f"python {script_path}")

    except requests.exceptions.HTTPError as e:
        handle_api_error(e, console)
        return f"python {script_path}"
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/]")
        return f"python {script_path}"


def resume_optimization_run(
    console: Console,
    run_id: str,
    api_keys: Dict[str, Any] = {},
    auth_headers: dict = {},
    timeout: Union[int, Tuple[int, int]] = DEFAULT_API_TIMEOUT,
) -> Optional[Dict[str, Any]]:
    """Resume an optimization run from the last completed step."""
    with console.status(f"[bold green]Resuming run {run_id}..."):
        try:
            response = requests.post(
                f"{__base_url__}/runs/{run_id}/resume",
                json={"metadata": {"client_name": "cli", "client_version": __pkg_version__, **api_keys}},
                headers=auth_headers,
                timeout=timeout,
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


def extend_optimization_run(
    console: Console,
    run_id: str,
    additional_steps: int,
    api_keys: Dict[str, Any] = {},
    auth_headers: dict = {},
    timeout: Union[int, Tuple[int, int]] = DEFAULT_API_TIMEOUT,
) -> Optional[Dict[str, Any]]:
    """Extend a completed optimization run with additional steps."""
    with console.status(f"[bold green]Extending run {run_id} with {additional_steps} additional steps..."):
        try:
            response = requests.post(
                f"{__base_url__}/runs/{run_id}/extend",
                json={
                    "additional_steps": additional_steps,
                    "metadata": {"client_name": "cli", "client_version": __pkg_version__, **api_keys},
                },
                headers=auth_headers,
                timeout=timeout,
            )
            response.raise_for_status()
            result = response.json()
            return result
        except requests.exceptions.HTTPError as e:
            handle_api_error(e, console)
            return None
        except Exception as e:
            console.print(f"[bold red]Error extending run: {e}[/]")
            return None
