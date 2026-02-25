import math
import pathlib
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Optional

from rich.console import Console
from rich.prompt import Confirm

from . import __dashboard_url__
from .api import (
    claim_execution_task,
    get_execution_tasks,
    get_optimization_run_status,
    report_termination,
    resume_optimization_run,
    send_heartbeat,
    start_optimization_run,
    submit_execution_result,
)
from .artifacts import RunArtifacts
from .auth import handle_authentication
from .events import get_event_context
from .browser import open_browser
from .ui import OptimizationUI, LiveOptimizationUI, PlainOptimizationUI
from .utils import read_additional_instructions, read_from_path, write_to_path, run_evaluation_with_files_swap


@dataclass
class OptimizationResult:
    """Result from a queue-based optimization loop."""

    success: bool
    final_step: int
    status: str  # "completed", "terminated", "error"
    reason: str  # e.g. "completed_successfully", "user_terminated_sigint"
    details: Optional[str] = None


# --- Heartbeat Sender Class ---
class HeartbeatSender(threading.Thread):
    def __init__(self, run_id: str, auth_headers: dict, stop_event: threading.Event, interval: int = 30):
        super().__init__(daemon=True)  # Daemon thread exits when main thread exits
        self.run_id = run_id
        self.auth_headers = auth_headers
        self.interval = interval
        self.stop_event = stop_event

    def run(self):
        try:
            while not self.stop_event.is_set():
                if not send_heartbeat(self.run_id, self.auth_headers):
                    # send_heartbeat logs to stderr when it returns False.
                    pass

                if self.stop_event.is_set():
                    # Check once more before waiting for faster shutdown.
                    break

                # Wait for the next interval, or exit early on stop signal.
                self.stop_event.wait(self.interval)

        except Exception as e:
            # Avoid silent heartbeat thread failures during long runs.
            print(f"[ERROR HeartbeatSender] Unexpected error in heartbeat thread for run {self.run_id}: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)


def _run_optimization_loop(
    ui: OptimizationUI,
    run_id: str,
    auth_headers: dict,
    source_code: dict[str, str],
    eval_command: str,
    eval_timeout: Optional[int],
    artifacts: RunArtifacts,
    save_logs: bool,
    start_step: int = 0,
    poll_interval: float = 2.0,
    max_poll_attempts: int = 300,
    api_keys: Optional[dict] = None,
) -> OptimizationResult:
    """
    Shared queue-based execution loop for optimize and resume.

    Polls for execution tasks, executes locally, and submits results.
    This function handles the core optimization loop and returns a result
    object describing the outcome.

    Args:
        ui: UI handler for displaying progress and events.
        run_id: The optimization run ID.
        auth_headers: Authentication headers.
        source_code: Original source code content keyed by file path.
        eval_command: Evaluation command to run.
        eval_timeout: Timeout for evaluation in seconds.
        artifacts: RunArtifacts instance for writing step/output artifacts.
        save_logs: Whether to save execution logs.
        start_step: Initial step number (0 for new runs, current_step for resume).
        poll_interval: Seconds between polling attempts.
        max_poll_attempts: Max polls before timeout (~10 min with 2s interval).
        api_keys: Optional API keys for LLM providers.

    Returns:
        OptimizationResult with success status and termination info.
    """
    step = start_step

    try:
        while True:
            # Check if run has been stopped via dashboard
            try:
                status_response = get_optimization_run_status(
                    console=None, run_id=run_id, include_history=False, auth_headers=auth_headers
                )
                if status_response.get("status") == "stopping":
                    ui.on_stop_requested()
                    return OptimizationResult(
                        success=False,
                        final_step=step,
                        status="terminated",
                        reason="user_requested_stop",
                        details="Run stopped by user request via dashboard.",
                    )
            except Exception as e:
                ui.on_warning(f"Unable to check run status: {e}")

            # Poll for ready tasks
            ui.on_polling(step)
            tasks_result = None
            poll_attempts = 0

            while not tasks_result or not tasks_result.tasks:
                tasks_result = get_execution_tasks(run_id, auth_headers)

                # Check if run was stopped (from run summary in response)
                if tasks_result and tasks_result.run:
                    run_status = tasks_result.run.status
                    if run_status in ("stopping", "stopped", "terminated", "error", "completed"):
                        ui.on_stop_requested()
                        return OptimizationResult(
                            success=False,
                            final_step=step,
                            status="terminated",
                            reason="user_requested_stop",
                            details=f"Run status is '{run_status}'.",
                        )

                if not tasks_result or not tasks_result.tasks:
                    poll_attempts += 1
                    if poll_attempts >= max_poll_attempts:
                        ui.on_error("Timeout waiting for execution tasks")
                        return OptimizationResult(
                            success=False,
                            final_step=step,
                            status="error",
                            reason="timeout_waiting_for_tasks",
                            details="Timeout waiting for execution tasks",
                        )
                    time.sleep(poll_interval)

            task = tasks_result.tasks[0]
            task_id = task["id"]

            # Claim the task
            claimed = claim_execution_task(task_id, auth_headers)
            if claimed is None:
                ui.on_warning(f"Task {task_id} already claimed, retrying...")
                continue

            code = claimed["revision"]["code"]
            plan = claimed["revision"]["plan"]

            ui.on_executing(step)
            ui.on_task_claimed(task_id, plan)

            file_map = code

            artifacts.save_step_code(step, file_map)
            term_out = run_evaluation_with_files_swap(
                file_map=file_map, originals=source_code, eval_command=eval_command, timeout=eval_timeout
            )

            if save_logs:
                artifacts.save_execution_output(step=step, output=term_out)

            ui.on_output(term_out)

            # Submit result
            ui.on_submitting()
            result = submit_execution_result(
                run_id=run_id, task_id=task_id, execution_output=term_out, auth_headers=auth_headers, api_keys=api_keys
            )

            if result is None:
                ui.on_error("Failed to submit result")
                return OptimizationResult(
                    success=False,
                    final_step=step,
                    status="error",
                    reason="submit_failed",
                    details="Failed to submit execution result",
                )

            is_done = result.get("is_done", False)
            prev_metric = result.get("previous_solution_metric_value")

            if prev_metric is not None:
                ui.on_metric(step, prev_metric)

            step += 1

            if is_done:
                ui.on_complete(step)
                return OptimizationResult(success=True, final_step=step, status="completed", reason="completed_successfully")

    except KeyboardInterrupt:
        ui.on_interrupted()
        return OptimizationResult(success=False, final_step=step, status="terminated", reason="user_terminated_sigint")
    except Exception as e:
        ui.on_error(f"Error: {e}")
        return OptimizationResult(success=False, final_step=step, status="error", reason="unknown", details=str(e))


def _offer_apply_best_solution(
    console: Console,
    run_id: str,
    source_code: dict[str, str],
    artifacts: RunArtifacts,
    auth_headers: dict,
    apply_change: bool = False,
) -> None:
    """
    Fetch the best solution from the backend and offer to apply it.

    Args:
        console: Rich console for output.
        run_id: The optimization run ID.
        source_code: Original source code content keyed by file path.
        artifacts: RunArtifacts instance for saving best solution.
        auth_headers: Authentication headers.
        apply_change: If True, apply automatically without prompting.
    """
    try:
        # Fetch final status to get best solution
        status = get_optimization_run_status(console=console, run_id=run_id, include_history=False, auth_headers=auth_headers)
        best_result = status.get("best_result")

        if best_result is None:
            console.print("\n[yellow]No solution found. No changes to apply.[/]\n")
            return

        best_code = best_result.get("code")
        best_metric = best_result.get("metric_value")

        if not best_code:
            console.print("\n[green]Best solution is the same as original. No changes to apply.[/]\n")
            return
        best_file_map = best_code

        if best_file_map == source_code:
            console.print("\n[green]Best solution is the same as original. No changes to apply.[/]\n")
            return

        # Save best solution to logs
        best_dir = artifacts.save_best_code(best_file_map)

        # Show summary
        console.print("\n[bold green]Optimization complete![/]")
        if best_metric is not None:
            console.print(f"[green]Best metric value: {best_metric}[/]")

        console.print(f"[dim]Files modified: {', '.join(sorted(best_file_map.keys()))}[/]")

        # Ask user or auto-apply
        if apply_change:
            should_apply = True
        else:
            should_apply = Confirm.ask("Would you like to apply the best solution to your source files?", default=True)

        if should_apply:
            for rel_path, content in best_file_map.items():
                fp = pathlib.Path(rel_path)
                fp.parent.mkdir(parents=True, exist_ok=True)
                write_to_path(fp=fp, content=content)
            console.print(f"[green]Best solution applied to {len(best_file_map)} files[/]\n")
        else:
            console.print(f"[dim]Best solution saved to {best_dir}[/]\n")

    except Exception as e:
        console.print(f"[yellow]Could not fetch best solution: {e}[/]")


def resume_optimization(
    run_id: str,
    api_keys: Optional[dict] = None,
    poll_interval: float = 2.0,
    apply_change: bool = False,
    output_mode: str = "rich",
) -> bool:
    """
    Resume an interrupted run using the queue-based optimization loop.

    Polls for execution tasks, executes locally, and submits results.
    Uses the execution queue flow instead of the legacy direct flow.

    Args:
        run_id: The UUID of the run to resume.
        api_keys: Optional API keys for LLM providers.
        poll_interval: Seconds between polling attempts.
        apply_change: If True, automatically apply best solution; if False, prompt user.
        output_mode: "rich" for interactive terminal UI, "plain" for machine-readable output.

    Returns:
        True if optimization completed successfully, False otherwise.
    """
    console = Console(force_terminal=output_mode == "rich")

    # Authenticate
    weco_api_key, auth_headers = handle_authentication(console)
    if weco_api_key is None:
        return False

    # Fetch status first for validation and to display confirmation info
    try:
        status = get_optimization_run_status(console=console, run_id=run_id, include_history=True, auth_headers=auth_headers)
    except Exception as e:
        console.print(f"[bold red]Error fetching run status: {e}[/]")
        return False

    run_status_val = status.get("status")
    if run_status_val not in ("error", "terminated"):
        console.print(
            f"[yellow]Run {run_id} cannot be resumed (status: {run_status_val}). "
            f"Only 'error' or 'terminated' runs can be resumed.[/]"
        )
        return False

    objective = status.get("objective", {})
    metric_name = objective.get("metric_name", "metric")
    maximize = bool(objective.get("maximize", True))
    eval_command = objective.get("evaluation_command", "")

    optimizer = status.get("optimizer", {})
    total_steps = optimizer.get("steps", 0)
    current_step = int(status.get("current_step", 0))
    steps_remaining = int(total_steps) - current_step

    model_name = (
        (optimizer.get("code_generator") or {}).get("model") or (optimizer.get("evaluator") or {}).get("model") or "unknown"
    )

    # Display confirmation info
    console.print("[cyan]Resume Run Confirmation[/]")
    console.print(f"  Run ID: {run_id}")
    console.print(f"  Run Name: {status.get('run_name', 'N/A')}")
    console.print(f"  Status: {run_status_val}")
    console.print(f"  Objective: {metric_name} ({'maximize' if maximize else 'minimize'})")
    console.print(f"  Model: {model_name}")
    console.print(f"  Eval Command: {eval_command}")
    console.print(f"  Total Steps: {total_steps} | Current Step: {current_step} | Steps Remaining: {steps_remaining}")
    console.print(f"  Last Updated: {status.get('updated_at', 'N/A')}")

    unchanged = Confirm.ask(
        "Have you kept the source files and evaluation command unchanged since the original run?", default=True
    )
    if not unchanged:
        console.print("[yellow]Resume cancelled. Please start a new run if the environment changed.[/]")
        return False

    # Call backend to prepare resume (this sets status to 'running')
    resume_resp = resume_optimization_run(console=console, run_id=run_id, auth_headers=auth_headers)
    if resume_resp is None:
        return False

    log_dir = resume_resp.get("log_dir", ".runs")
    save_logs = bool(resume_resp.get("save_logs", False))
    eval_timeout = resume_resp.get("eval_timeout")

    # Read original source code and normalize to a file map.
    resume_source_code = resume_resp.get("source_code")
    if not isinstance(resume_source_code, dict):
        console.print("[bold red]Cannot resume run: source_code is not in the expected dict format.[/]")
        return False
    source_code: dict[str, str] = {}
    for rel_path, fallback_content in resume_source_code.items():
        fp = pathlib.Path(rel_path)
        source_code[rel_path] = read_from_path(fp=fp, is_json=False) if fp.exists() else fallback_content

    dashboard_url = f"{__dashboard_url__}/runs/{run_id}"
    run_name = resume_resp.get("run_name", run_id)

    # Open dashboard in the user's browser
    open_browser(dashboard_url)

    # Setup artifacts manager
    artifacts = RunArtifacts(log_dir=log_dir, run_id=run_id)

    # Start heartbeat thread
    stop_heartbeat_event = threading.Event()
    heartbeat_thread = HeartbeatSender(run_id, auth_headers, stop_heartbeat_event)
    heartbeat_thread.start()

    # Extract best solution info from resume response (if available)
    best_metric_value = resume_resp.get("best_metric_value")
    best_step = resume_resp.get("best_step")

    result: Optional[OptimizationResult] = None
    try:
        # Select UI implementation based on output mode
        if output_mode == "plain":
            ui_instance = PlainOptimizationUI(
                run_id, run_name, total_steps, dashboard_url, model=model_name, metric_name=metric_name
            )
        else:
            ui_instance = LiveOptimizationUI(
                console, run_id, run_name, total_steps, dashboard_url, model=model_name, metric_name=metric_name
            )

        with ui_instance as ui:
            # Populate UI with best solution from previous run if available
            if best_metric_value is not None and best_step is not None:
                ui.on_metric(best_step, best_metric_value)

            result = _run_optimization_loop(
                ui=ui,
                run_id=run_id,
                auth_headers=auth_headers,
                source_code=source_code,
                eval_command=eval_command,
                eval_timeout=eval_timeout,
                artifacts=artifacts,
                save_logs=save_logs,
                start_step=current_step,
                poll_interval=poll_interval,
                api_keys=api_keys,
            )

        # Stop heartbeat immediately after loop completes
        stop_heartbeat_event.set()
        heartbeat_thread.join(timeout=2)

        # Show resume message if interrupted
        if result.status == "terminated":
            if output_mode == "plain":
                print(f"\nTo resume this run, use: weco resume {run_id}\n", flush=True)
            else:
                console.print(f"\n[cyan]To resume this run, use:[/] [bold]weco resume {run_id}[/]\n")

        # Offer to apply best solution
        _offer_apply_best_solution(
            console=console,
            run_id=run_id,
            source_code=source_code,
            artifacts=artifacts,
            auth_headers=auth_headers,
            apply_change=apply_change,
        )

        return result.success
    finally:
        # Ensure heartbeat is stopped (in case of early exit/exception)
        stop_heartbeat_event.set()
        heartbeat_thread.join(timeout=2)

        # Report termination to backend
        if result is not None:
            try:
                report_termination(
                    run_id=run_id,
                    status_update=result.status,
                    reason=result.reason,
                    details=result.details,
                    auth_headers=auth_headers,
                )
            except Exception:
                pass  # Best effort


def optimize(
    source: "str | list[str]",
    eval_command: str,
    metric: str,
    goal: str = "maximize",
    model: str = "o4-mini",
    steps: int = 5,
    additional_instructions: Optional[str] = None,
    eval_timeout: Optional[int] = None,
    save_logs: bool = False,
    log_dir: str = ".runs",
    api_keys: Optional[dict] = None,
    poll_interval: float = 2.0,
    apply_change: bool = False,
    require_review: bool = False,
    output_mode: str = "rich",
) -> bool:
    """
    Simplified queue-based optimization loop.

    Polls for execution tasks, executes locally, and submits results.
    Uses the new execution queue flow instead of the legacy direct flow.

    Args:
        source: Path to a single source file (str) or list of file paths.
        eval_command: Command to run for evaluation.
        metric: Name of the metric to optimize.
        goal: "maximize" or "minimize".
        model: LLM model to use.
        steps: Number of optimization steps.
        additional_instructions: Optional instructions for the optimizer.
        eval_timeout: Timeout for evaluation command in seconds.
        save_logs: Whether to save execution logs.
        log_dir: Directory for logs.
        api_keys: Optional API keys for LLM providers.
        poll_interval: Seconds between polling attempts.
        apply_change: If True, automatically apply best solution; if False, prompt user.
        output_mode: "rich" for interactive terminal UI, "plain" for machine-readable output.

    Returns:
        True if optimization completed successfully, False otherwise.
    """
    console = Console(force_terminal=output_mode == "rich")

    # Authenticate
    weco_api_key, auth_headers = handle_authentication(console)
    if weco_api_key is None:
        # Authentication failed or user declined
        return False

    # Process parameters
    maximize = goal.lower() in ["maximize", "max"]

    source_paths = source if isinstance(source, list) else [source]
    source_code: dict[str, str] = {}
    for s in source_paths:
        fp = pathlib.Path(s)
        source_code[str(fp)] = read_from_path(fp=fp, is_json=False)

    # Always send as multi-file payload, even for a single source file.
    source_path_for_api: Optional[str] = None

    code_generator_config = {"model": model}
    evaluator_config = {"model": model, "include_analysis": True}
    search_policy_config = {
        "num_drafts": max(1, math.ceil(0.15 * steps)),
        "debug_prob": 0.5,
        "max_debug_depth": max(1, math.ceil(0.1 * steps)),
    }
    processed_instructions = read_additional_instructions(additional_instructions)

    # Get event context for tracking
    event_ctx = get_event_context()

    # Start the run
    run_response = start_optimization_run(
        console=console,
        source_code=source_code,
        source_path=source_path_for_api,
        evaluation_command=eval_command,
        metric_name=metric,
        maximize=maximize,
        steps=steps,
        code_generator_config=code_generator_config,
        evaluator_config=evaluator_config,
        search_policy_config=search_policy_config,
        additional_instructions=processed_instructions,
        eval_timeout=eval_timeout,
        save_logs=save_logs,
        log_dir=log_dir,
        auth_headers=auth_headers,
        api_keys=api_keys,
        require_review=require_review,
        installation_id=event_ctx.installation_id,
        invocation_id=event_ctx.invocation_id,
        invoked_via=event_ctx.invoked_via,
    )

    if run_response is None:
        return False

    run_id = run_response["run_id"]
    run_name = run_response["run_name"]
    dashboard_url = f"{__dashboard_url__}/runs/{run_id}"

    # Open dashboard in the user's browser
    open_browser(dashboard_url)

    # Setup artifacts manager
    artifacts = RunArtifacts(log_dir=log_dir, run_id=run_id)

    # Start heartbeat thread
    stop_heartbeat_event = threading.Event()
    heartbeat_thread = HeartbeatSender(run_id, auth_headers, stop_heartbeat_event)
    heartbeat_thread.start()

    result: Optional[OptimizationResult] = None
    try:
        # Select UI implementation based on output mode
        if output_mode == "plain":
            ui_instance = PlainOptimizationUI(run_id, run_name, steps, dashboard_url, model=model, metric_name=metric)
        else:
            ui_instance = LiveOptimizationUI(console, run_id, run_name, steps, dashboard_url, model=model, metric_name=metric)

        with ui_instance as ui:
            result = _run_optimization_loop(
                ui=ui,
                run_id=run_id,
                auth_headers=auth_headers,
                source_code=source_code,
                eval_command=eval_command,
                eval_timeout=eval_timeout,
                artifacts=artifacts,
                save_logs=save_logs,
                start_step=0,
                poll_interval=poll_interval,
                api_keys=api_keys,
            )

        # Stop heartbeat immediately after loop completes
        stop_heartbeat_event.set()
        heartbeat_thread.join(timeout=2)

        # Show resume message if interrupted
        if result.status == "terminated":
            if output_mode == "plain":
                print(f"\nTo resume this run, use: weco resume {run_id}\n", flush=True)
            else:
                console.print(f"\n[cyan]To resume this run, use:[/] [bold]weco resume {run_id}[/]\n")

        # Offer to apply best solution
        _offer_apply_best_solution(
            console=console,
            run_id=run_id,
            source_code=source_code,
            artifacts=artifacts,
            auth_headers=auth_headers,
            apply_change=apply_change,
        )

        return result.success
    finally:
        # Ensure heartbeat is stopped (in case of early exit/exception)
        stop_heartbeat_event.set()
        heartbeat_thread.join(timeout=2)

        # Report termination to backend
        if result is not None:
            try:
                report_termination(
                    run_id=run_id,
                    status_update=result.status,
                    reason=result.reason,
                    details=result.details,
                    auth_headers=auth_headers,
                )
            except Exception:
                pass  # Best effort
