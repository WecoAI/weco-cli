import pathlib
import math
import requests
import threading
import signal
import sys
import traceback
import json
from datetime import datetime
from typing import Optional
from rich.console import Console
from rich.prompt import Confirm
from .api import (
    start_optimization_run,
    evaluate_feedback_then_suggest_next_solution,
    get_optimization_run_status,
    send_heartbeat,
    report_termination,
    resume_optimization_run,
)
from .auth import handle_authentication
from .panels import Node
from .utils import read_additional_instructions, read_from_path, write_to_path, run_evaluation_with_file_swap
from .output import create_output_handler, OutputHandler


def save_execution_output(runs_dir: pathlib.Path, step: int, output: str) -> None:
    """
    Save execution output using hybrid approach:
    1. Per-step raw files under outputs/step_<n>.out.txt
    2. Centralized JSONL index in exec_output.jsonl

    Args:
        runs_dir: Path to the run directory (.runs/<run_id>)
        step: Current step number
        output: The execution output to save
    """
    timestamp = datetime.now().isoformat()

    # Create outputs directory if it doesn't exist
    outputs_dir = runs_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Save per-step raw output file
    step_file = outputs_dir / f"step_{step}.out.txt"
    with open(step_file, "w", encoding="utf-8") as f:
        f.write(output)

    # Append to centralized JSONL index
    jsonl_file = runs_dir / "exec_output.jsonl"
    output_file_path = step_file.relative_to(runs_dir).as_posix()
    entry = {"step": step, "timestamp": timestamp, "output_file": output_file_path, "output_length": len(output)}
    with open(jsonl_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


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
                    # send_heartbeat itself prints errors to stderr if it returns False
                    # No explicit HeartbeatSender log needed here unless more detail is desired for a False return
                    pass

                if self.stop_event.is_set():  # Check before waiting for responsiveness
                    break

                self.stop_event.wait(self.interval)  # Wait for interval or stop signal

        except Exception as e:
            # Catch any unexpected error in the loop to prevent silent thread death
            print(f"[ERROR HeartbeatSender] Unexpected error in heartbeat thread for run {self.run_id}: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            # The loop will break due to the exception, and thread will terminate via finally.


def get_best_node_from_status(status_response: dict) -> Optional[Node]:
    """Extract the best node from a status response as a panels.Node instance."""
    if status_response.get("best_result") is not None:
        return Node(
            id=status_response["best_result"]["solution_id"],
            parent_id=status_response["best_result"]["parent_id"],
            code=status_response["best_result"]["code"],
            metric=status_response["best_result"]["metric_value"],
            is_buggy=status_response["best_result"]["is_buggy"],
        )
    return None


def get_node_from_status(status_response: dict, solution_id: str) -> Node:
    """Find the node with the given solution_id from a status response; raise if not found."""
    nodes = status_response.get("nodes") or []
    for node_data in nodes:
        if node_data.get("solution_id") == solution_id:
            return Node(
                id=node_data["solution_id"],
                parent_id=node_data["parent_id"],
                code=node_data["code"],
                metric=node_data["metric_value"],
                is_buggy=node_data["is_buggy"],
            )
    raise ValueError(
        "Current solution node not found in the optimization status response. This may indicate a synchronization issue with the backend."
    )


def execute_optimization(
    source: str,
    eval_command: str,
    metric: str,
    goal: str,  # "maximize" or "minimize"
    steps: int = 100,
    model: Optional[str] = None,
    log_dir: str = ".runs",
    additional_instructions: Optional[str] = None,
    console: Optional[Console] = None,
    eval_timeout: Optional[int] = None,
    save_logs: bool = False,
    apply_change: bool = False,
    output_mode: str = "rich",
) -> bool:
    """
    Execute the core optimization logic.

    Returns:
        bool: True if optimization completed successfully, False otherwise
    """
    if console is None:
        console = Console()

    # Global variables for this optimization run
    heartbeat_thread = None
    stop_heartbeat_event = threading.Event()
    current_run_id_for_heartbeat = None
    current_auth_headers_for_heartbeat = {}
    output_handler: Optional[OutputHandler] = None

    best_solution_code = None
    original_source_code = None

    # --- Signal Handler for this optimization run ---
    def signal_handler(signum, frame):
        nonlocal output_handler

        # Stop the live display if active
        if output_handler:
            live_ref = output_handler.get_live_ref()
            if live_ref is not None:
                live_ref.stop()

        signal_name = signal.Signals(signum).name
        console.print(f"\n[bold yellow]Termination signal ({signal_name}) received. Shutting down...[/]\n")

        # Stop heartbeat thread
        stop_heartbeat_event.set()
        if heartbeat_thread and heartbeat_thread.is_alive():
            heartbeat_thread.join(timeout=2)

        # Report termination (best effort)
        if current_run_id_for_heartbeat:
            report_termination(
                run_id=current_run_id_for_heartbeat,
                status_update="terminated",
                reason=f"user_terminated_{signal_name.lower()}",
                details=f"Process terminated by signal {signal_name} ({signum}).",
                auth_headers=current_auth_headers_for_heartbeat,
            )
            console.print(f"[cyan]To resume this run, use:[/] [bold cyan]weco resume {current_run_id_for_heartbeat}[/]\n")

        sys.exit(0)

    # Set up signal handlers for this run
    original_sigint_handler = signal.signal(signal.SIGINT, signal_handler)
    original_sigterm_handler = signal.signal(signal.SIGTERM, signal_handler)

    run_id = None
    optimization_completed_normally = False
    user_stop_requested_flag = False

    try:
        # --- Login/Authentication Handling (now mandatory) ---
        weco_api_key, auth_headers = handle_authentication(console)
        if weco_api_key is None:
            return False

        current_auth_headers_for_heartbeat = auth_headers

        # --- Process Parameters ---
        maximize = goal.lower() in ["maximize", "max"]

        if model is None:
            model = "o4-mini"

        code_generator_config = {"model": model}
        evaluator_config = {"model": model, "include_analysis": True}
        search_policy_config = {
            "num_drafts": max(1, math.ceil(0.15 * steps)),
            "debug_prob": 0.5,
            "max_debug_depth": max(1, math.ceil(0.1 * steps)),
        }
        processed_additional_instructions = read_additional_instructions(additional_instructions=additional_instructions)
        source_fp = pathlib.Path(source)
        source_code = read_from_path(fp=source_fp, is_json=False)
        original_source_code = source_code

        # --- Start Optimization Run ---
        run_response = start_optimization_run(
            console=console,
            source_code=source_code,
            source_path=str(source_fp),
            evaluation_command=eval_command,
            metric_name=metric,
            maximize=maximize,
            steps=steps,
            code_generator_config=code_generator_config,
            evaluator_config=evaluator_config,
            search_policy_config=search_policy_config,
            additional_instructions=processed_additional_instructions,
            eval_timeout=eval_timeout,
            save_logs=save_logs,
            log_dir=log_dir,
            auth_headers=auth_headers,
        )
        if run_response is None:
            return False

        run_id = run_response["run_id"]
        run_name = run_response["run_name"]
        current_run_id_for_heartbeat = run_id

        # --- Start Heartbeat Thread ---
        stop_heartbeat_event.clear()
        heartbeat_thread = HeartbeatSender(run_id, auth_headers, stop_heartbeat_event)
        heartbeat_thread.start()

        # --- Create Output Handler ---
        from .__init__ import __dashboard_url__

        output_handler = create_output_handler(
            output_mode=output_mode,
            console=console,
            metric_name=metric,
            maximize=maximize,
            total_steps=steps,
            model=model,
            runs_dir=log_dir,
            source_fp=source_fp,
        )

        # --- Main Optimization Loop ---
        with output_handler:
            # Define the runs directory
            runs_dir = pathlib.Path(log_dir) / run_id
            runs_dir.mkdir(parents=True, exist_ok=True)

            # Initialize logging structure if save_logs is enabled
            if save_logs:
                jsonl_file = runs_dir / "exec_output.jsonl"
                metadata = {
                    "type": "metadata",
                    "run_id": run_id,
                    "run_name": run_name,
                    "started": datetime.now().isoformat(),
                    "eval_command": eval_command,
                    "metric": metric,
                    "goal": "maximize" if maximize else "minimize",
                    "total_steps": steps,
                }
                with open(jsonl_file, "w", encoding="utf-8") as f:
                    f.write(json.dumps(metadata) + "\n")

            # Notify handler of run start
            output_handler.on_run_started(
                run_id=run_id,
                run_name=run_name,
                dashboard_url=f"{__dashboard_url__}/runs/{run_id}",
                model=model,
                runs_dir=log_dir,
                plan=run_response["plan"],
                initial_code=run_response["code"],
                initial_solution_id=run_response["solution_id"],
            )

            # Write the initial code string to the logs
            write_to_path(fp=runs_dir / f"step_0{source_fp.suffix}", content=run_response["code"])

            # Baseline evaluation - print message before running local evaluation
            output_handler.on_baseline_evaluating()

            # Run evaluation on the initial solution (file swap ensures original is restored)
            term_out = run_evaluation_with_file_swap(
                file_path=source_fp,
                new_content=run_response["code"],
                original_content=source_code,
                eval_command=eval_command,
                timeout=eval_timeout,
            )

            # Save logs if requested
            if save_logs:
                save_execution_output(runs_dir, step=0, output=term_out)

            # Track previous best for detecting new bests
            previous_best_metric = None
            baseline_reported = False
            # Track previous solution to report its result when the API evaluates it
            previous_solution_id = run_response["solution_id"]

            # Starting from step 1 to steps (inclusive) because the baseline solution is step 0, so we want to optimize for steps worth of steps
            for step in range(1, steps + 1):
                # Check for stop request
                if run_id:
                    try:
                        current_status_response = get_optimization_run_status(
                            console=console, run_id=run_id, include_history=False, auth_headers=auth_headers
                        )
                        current_run_status_val = current_status_response.get("status")
                        if current_run_status_val == "stopping":
                            output_handler.on_stop_requested()
                            user_stop_requested_flag = True
                            break
                    except requests.exceptions.RequestException as e:
                        output_handler.on_warning(f"Unable to check run status: {e}. Continuing optimization...")
                    except Exception as e:
                        output_handler.on_warning(f"Error checking run status: {e}. Continuing optimization...")

                output_handler.on_step_starting(step=step, previous_best_metric=previous_best_metric)

                # Send feedback and get next suggestion
                eval_and_next_solution_response = evaluate_feedback_then_suggest_next_solution(
                    console=console, step=step, run_id=run_id, execution_output=term_out, auth_headers=auth_headers
                )
                # Save next solution (.runs/<run-id>/step_<step>.<extension>)
                write_to_path(fp=runs_dir / f"step_{step}{source_fp.suffix}", content=eval_and_next_solution_response["code"])

                status_response = get_optimization_run_status(
                    console=console, run_id=run_id, include_history=True, auth_headers=auth_headers
                )

                # Report baseline result after first API call evaluates it
                if not baseline_reported:
                    baseline_metric = status_response.get("best_result", {}).get("metric_value")
                    output_handler.on_baseline_completed(eval_output=term_out, best_metric=baseline_metric)
                    if baseline_metric is not None:
                        previous_best_metric = baseline_metric
                    baseline_reported = True
                else:
                    # Report the previous step's result (step - 1) now that API has evaluated it
                    try:
                        prev_node = get_node_from_status(status_response=status_response, solution_id=previous_solution_id)
                        prev_metric = prev_node.metric
                        is_new_best = prev_metric is not None and prev_metric != previous_best_metric and (
                            previous_best_metric is None
                            or (maximize and prev_metric > previous_best_metric)
                            or (not maximize and prev_metric < previous_best_metric)
                        )
                        output_handler.on_step_result(step=step - 1, metric=prev_metric, is_new_best=is_new_best)
                        if is_new_best:
                            previous_best_metric = prev_metric
                    except ValueError:
                        # Previous node not found - report as buggy
                        output_handler.on_step_result(step=step - 1, metric=None, is_new_best=False)

                # Update previous_solution_id for next iteration
                previous_solution_id = eval_and_next_solution_response["solution_id"]

                nodes_list_from_status = status_response.get("nodes") or []
                best_solution_node = get_best_node_from_status(status_response=status_response)
                current_solution_node = get_node_from_status(
                    status_response=status_response, solution_id=eval_and_next_solution_response["solution_id"]
                )

                # Set best solution and save optimization results
                try:
                    best_solution_code = best_solution_node.code
                except AttributeError:
                    # Can happen if the code was buggy
                    best_solution_code = read_from_path(fp=runs_dir / f"step_0{source_fp.suffix}", is_json=False)

                # Save best solution to .runs/<run-id>/best.<extension>
                write_to_path(fp=runs_dir / f"best{source_fp.suffix}", content=best_solution_code)

                # Notify handler of step generation complete
                output_handler.on_step_generated(
                    step=step,
                    code=eval_and_next_solution_response["code"],
                    plan=eval_and_next_solution_response["plan"],
                    nodes=nodes_list_from_status,
                    current_node=current_solution_node,
                    best_node=best_solution_node,
                    solution_id=eval_and_next_solution_response["solution_id"],
                )

                # Run evaluation and restore original code after
                term_out = run_evaluation_with_file_swap(
                    file_path=source_fp,
                    new_content=eval_and_next_solution_response["code"],
                    original_content=source_code,
                    eval_command=eval_command,
                    timeout=eval_timeout,
                )

                # Save logs if requested
                if save_logs:
                    save_execution_output(runs_dir, step=step, output=term_out)

                output_handler.on_step_completed(step=step, eval_output=term_out)

            if not user_stop_requested_flag:
                # Evaluate the final solution that's been generated
                eval_and_next_solution_response = evaluate_feedback_then_suggest_next_solution(
                    console=console, step=steps, run_id=run_id, execution_output=term_out, auth_headers=auth_headers
                )
                status_response = get_optimization_run_status(
                    console=console, run_id=run_id, include_history=True, auth_headers=auth_headers
                )

                nodes_list_from_status_final = status_response.get("nodes") or []
                best_solution_node = get_best_node_from_status(status_response=status_response)

                # Report the final step's result now that API has evaluated it
                try:
                    final_node = get_node_from_status(status_response=status_response, solution_id=previous_solution_id)
                    final_metric = final_node.metric
                    is_new_best = final_metric is not None and final_metric != previous_best_metric and (
                        previous_best_metric is None
                        or (maximize and final_metric > previous_best_metric)
                        or (not maximize and final_metric < previous_best_metric)
                    )
                    output_handler.on_step_result(step=steps, metric=final_metric, is_new_best=is_new_best)
                except ValueError:
                    # Final node not found - use best result as fallback
                    if best_solution_node:
                        output_handler.on_step_result(step=steps, metric=best_solution_node.metric, is_new_best=False)
                    else:
                        output_handler.on_step_result(step=steps, metric=None, is_new_best=False)
                best_solution_code = best_solution_node.code if best_solution_node else None
                best_metric_value = status_response.get("best_result", {}).get("metric_value")

                # Save best solution to .runs/<run-id>/best.<extension>
                if best_solution_code:
                    write_to_path(fp=runs_dir / f"best{source_fp.suffix}", content=best_solution_code)

                output_handler.on_run_completed(
                    best_node=best_solution_node, best_metric_value=best_metric_value, nodes=nodes_list_from_status_final
                )

                # Mark as completed normally for the finally block
                optimization_completed_normally = True

    except Exception as e:
        # Catch errors during the main optimization loop or setup
        try:
            error_message = e.response.json()["detail"]
        except Exception:
            error_message = str(e)
        output_handler.on_error(message=error_message, run_id=run_id)
        # Ensure optimization_completed_normally is False
        optimization_completed_normally = False
    finally:
        # Restore original signal handlers
        signal.signal(signal.SIGINT, original_sigint_handler)
        signal.signal(signal.SIGTERM, original_sigterm_handler)

        # Stop heartbeat thread
        stop_heartbeat_event.set()
        if heartbeat_thread and heartbeat_thread.is_alive():
            heartbeat_thread.join(timeout=2)

        # Report final status if run exists
        if run_id:
            if optimization_completed_normally:
                status, reason, details = "completed", "completed_successfully", None
            elif user_stop_requested_flag:
                status, reason, details = "terminated", "user_requested_stop", "Run stopped by user request via dashboard."
            else:
                status, reason = "error", "error_cli_internal"
                details = locals().get("error_details") or (
                    traceback.format_exc()
                    if "e" in locals() and isinstance(locals()["e"], Exception)
                    else "CLI terminated unexpectedly without a specific exception captured."
                )

            if best_solution_code and best_solution_code != original_source_code:
                # Determine whether to apply: automatically if --apply-change is set, otherwise ask user
                should_apply = apply_change or Confirm.ask(
                    "Would you like to apply the best solution to the source file?", default=True
                )
                if should_apply:
                    write_to_path(fp=source_fp, content=best_solution_code)
                    console.print("\n[green]Best solution applied to the source file.[/]\n")
            else:
                console.print("\n[green]A better solution was not found. No changes to apply.[/]\n")

            report_termination(
                run_id=run_id,
                status_update=status,
                reason=reason,
                details=details,
                auth_headers=current_auth_headers_for_heartbeat,
            )

        # Handle exit
        if user_stop_requested_flag:
            console.print("[yellow]Run terminated by user request.[/]")
            console.print(f"\n[cyan]To resume this run, use:[/] [bold cyan]weco resume {run_id}[/]\n")

    return optimization_completed_normally or user_stop_requested_flag


def resume_optimization(
    run_id: str, console: Optional[Console] = None, apply_change: bool = False, output_mode: str = "rich"
) -> bool:
    """Resume an interrupted run from the most recent node and continue optimization."""
    if console is None:
        console = Console()

    # Globals for this optimization run
    heartbeat_thread = None
    stop_heartbeat_event = threading.Event()
    current_run_id_for_heartbeat = None
    current_auth_headers_for_heartbeat = {}
    output_handler: Optional[OutputHandler] = None

    best_solution_code = None
    original_source_code = None

    # Signal handler for this optimization run
    def signal_handler(signum, frame):
        nonlocal output_handler

        # Stop the live display if active
        if output_handler:
            live_ref = output_handler.get_live_ref()
            if live_ref is not None:
                live_ref.stop()

        signal_name = signal.Signals(signum).name
        console.print(f"\n[bold yellow]Termination signal ({signal_name}) received. Shutting down...[/]\n")
        stop_heartbeat_event.set()
        if heartbeat_thread and heartbeat_thread.is_alive():
            heartbeat_thread.join(timeout=2)
        if current_run_id_for_heartbeat:
            report_termination(
                run_id=current_run_id_for_heartbeat,
                status_update="terminated",
                reason=f"user_terminated_{signal_name.lower()}",
                details=f"Process terminated by signal {signal_name} ({signum}).",
                auth_headers=current_auth_headers_for_heartbeat,
            )
            console.print(f"\n[cyan]To resume this run, use:[/] [bold cyan]weco resume {current_run_id_for_heartbeat}[/]\n")
        sys.exit(0)

    # Set up signal handlers for this run
    original_sigint_handler = signal.signal(signal.SIGINT, signal_handler)
    original_sigterm_handler = signal.signal(signal.SIGTERM, signal_handler)

    optimization_completed_normally = False
    user_stop_requested_flag = False

    try:
        # --- Login/Authentication Handling (now mandatory) ---
        weco_api_key, auth_headers = handle_authentication(console)
        if weco_api_key is None:
            # Authentication failed or user declined
            return False

        current_auth_headers_for_heartbeat = auth_headers

        # Fetch status first for validation and to display confirmation info
        try:
            status = get_optimization_run_status(
                console=console, run_id=run_id, include_history=True, auth_headers=auth_headers
            )
        except Exception as e:
            console.print(f"[bold red]Error fetching run status:[/] {e}")
            return False

        run_status_val = status.get("status")
        if run_status_val not in ("error", "terminated"):
            console.print(
                f"[yellow]Run {run_id} cannot be resumed (status: {run_status_val}). Only 'error' or 'terminated' runs can be resumed.[/]"
            )
            return False

        objective = status.get("objective", {})
        metric_name = objective.get("metric_name", "metric")
        maximize = bool(objective.get("maximize", True))

        optimizer = status.get("optimizer", {})

        model_name = (
            (optimizer.get("code_generator") or {}).get("model")
            or (optimizer.get("evaluator") or {}).get("model")
            or "unknown"
        )
        total_steps = optimizer.get("steps")
        current_step = int(status["current_step"])
        steps_remaining = int(total_steps) - int(current_step)

        console.print("[cyan]Resume Run Confirmation[/]")
        console.print(f"  [bold]Run ID:[/] {run_id}")
        console.print(f"  [bold]Run Name:[/] {status.get('run_name', 'N/A')}")
        console.print(f"  [bold]Status:[/] {run_status_val}")
        console.print(f"  [bold]Objective:[/] {metric_name} ({'maximize' if maximize else 'minimize'})")
        console.print(f"  [bold]Model:[/] {model_name}")
        console.print(f"  [bold]Eval Command:[/] {objective.get('evaluation_command', 'N/A')}")
        console.print(
            f"  [bold]Total Steps:[/] {total_steps} | [bold]Resume Step:[/] {current_step} | [bold]Steps Remaining:[/] {steps_remaining}"
        )
        console.print(f"  [bold]Last Updated:[/] {status.get('updated_at', 'N/A')}")

        unchanged = Confirm.ask(
            "Have you kept the source file and evaluation command unchanged since the original run?", default=True
        )
        if not unchanged:
            console.print("[yellow]Resume cancelled. Please start a new run if the environment changed.[/]")
            return False

        # Call backend to prepare resume
        resume_resp = resume_optimization_run(console=console, run_id=run_id, auth_headers=auth_headers)
        if resume_resp is None:
            return False

        eval_command = resume_resp["evaluation_command"]
        source_path = resume_resp.get("source_path")

        # Use backend-saved values
        log_dir = resume_resp.get("log_dir", ".runs")
        save_logs = bool(resume_resp.get("save_logs", False))
        eval_timeout = resume_resp.get("eval_timeout")

        # Read the original source code from the file before we start modifying it
        source_fp = pathlib.Path(source_path)
        source_fp.parent.mkdir(parents=True, exist_ok=True)
        # Store the original content to restore after each evaluation
        original_source_code = read_from_path(fp=source_fp, is_json=False) if source_fp.exists() else ""
        # The code to restore is the code from the last step of the previous run
        code_to_restore = resume_resp.get("code") or resume_resp.get("source_code") or ""

        # Compute best and current nodes
        nodes_list_from_status = status.get("nodes") or []
        best_solution_node = get_best_node_from_status(status_response=status)
        current_solution_node = get_node_from_status(status_response=status, solution_id=resume_resp.get("solution_id"))

        # Ensure runs dir exists
        runs_dir = pathlib.Path(log_dir) / resume_resp["run_id"]
        runs_dir.mkdir(parents=True, exist_ok=True)
        # Persist last step's code into logs as step_<current_step>
        write_to_path(fp=runs_dir / f"step_{current_step}{source_fp.suffix}", content=code_to_restore)

        # Initialize best solution code
        try:
            best_solution_code = best_solution_node.code
        except AttributeError:
            # Edge case: best solution node is not available.
            # This can happen if the user has cancelled the run before even running the baseline solution
            pass  # Leave best solution code as None

        # Start Heartbeat Thread
        stop_heartbeat_event.clear()
        heartbeat_thread = HeartbeatSender(resume_resp["run_id"], auth_headers, stop_heartbeat_event)
        heartbeat_thread.start()
        current_run_id_for_heartbeat = resume_resp["run_id"]

        # Create Output Handler
        from .__init__ import __dashboard_url__

        output_handler = create_output_handler(
            output_mode=output_mode,
            console=console,
            metric_name=metric_name,
            maximize=maximize,
            total_steps=total_steps,
            model=model_name,
            runs_dir=log_dir,
            source_fp=source_fp,
        )

        # Main Optimization Loop
        with output_handler:
            # Notify handler of run start (resume)
            output_handler.on_run_started(
                run_id=resume_resp["run_id"],
                run_name=resume_resp.get("run_name", ""),
                dashboard_url=f"{__dashboard_url__}/runs/{resume_resp['run_id']}",
                model=model_name,
                runs_dir=log_dir,
                plan=resume_resp.get("plan", ""),
                initial_code=code_to_restore,
                initial_solution_id=resume_resp.get("solution_id"),
            )

            # Use backend-provided execution output only (no fallback)
            term_out = resume_resp.get("execution_output") or ""
            is_baseline_step = current_step == 0

            # If missing output, evaluate once before first suggest
            if term_out is None or len(term_out.strip()) == 0:
                if is_baseline_step:
                    output_handler.on_baseline_evaluating()

                term_out = run_evaluation_with_file_swap(
                    file_path=source_fp,
                    new_content=code_to_restore,
                    original_content=original_source_code,
                    eval_command=eval_command,
                    timeout=eval_timeout,
                )

            if save_logs:
                save_execution_output(runs_dir, step=current_step, output=term_out)

            # Track previous best for detecting new bests
            previous_best_metric = None
            # Track previous solution to report its result when the API evaluates it
            previous_solution_id = resume_resp.get("solution_id")
            # Track if this is the first iteration (to handle baseline vs step results)
            first_iteration = True

            # Continue optimization: steps current_step+1..total_steps
            for step in range(current_step + 1, total_steps + 1):
                output_handler.on_step_starting(step=step, previous_best_metric=previous_best_metric)

                # Check for stop request
                try:
                    current_status_response = get_optimization_run_status(
                        console=console, run_id=resume_resp["run_id"], include_history=False, auth_headers=auth_headers
                    )
                    if current_status_response.get("status") == "stopping":
                        output_handler.on_stop_requested()
                        user_stop_requested_flag = True
                        break
                except requests.exceptions.RequestException as e:
                    output_handler.on_warning(f"Unable to check run status: {e}. Continuing optimization...")
                except Exception as e:
                    output_handler.on_warning(f"Error checking run status: {e}. Continuing optimization...")

                # Suggest next
                eval_and_next_solution_response = evaluate_feedback_then_suggest_next_solution(
                    console=console,
                    step=step,
                    run_id=resume_resp["run_id"],
                    execution_output=term_out,
                    auth_headers=auth_headers,
                )

                # Save next solution to logs
                write_to_path(fp=runs_dir / f"step_{step}{source_fp.suffix}", content=eval_and_next_solution_response["code"])

                # Refresh status with history
                status_response = get_optimization_run_status(
                    console=console, run_id=resume_resp["run_id"], include_history=True, auth_headers=auth_headers
                )

                # Report the previous step's result now that API has evaluated it
                if first_iteration:
                    # For resume, the first iteration evaluates the step we resumed from
                    # Report it as baseline if it was step 0, otherwise as a step result
                    if current_step == 0:
                        baseline_metric = status_response.get("best_result", {}).get("metric_value")
                        output_handler.on_baseline_completed(eval_output=term_out, best_metric=baseline_metric)
                        if baseline_metric is not None:
                            previous_best_metric = baseline_metric
                    else:
                        try:
                            prev_node = get_node_from_status(status_response=status_response, solution_id=previous_solution_id)
                            prev_metric = prev_node.metric
                            if prev_metric is not None:
                                # First iteration of resume - treat as potentially new best
                                output_handler.on_step_result(step=current_step, metric=prev_metric, is_new_best=True)
                                previous_best_metric = prev_metric
                        except ValueError:
                            pass
                    first_iteration = False
                else:
                    # Report the previous step's result (step - 1)
                    try:
                        prev_node = get_node_from_status(status_response=status_response, solution_id=previous_solution_id)
                        prev_metric = prev_node.metric
                        is_new_best = prev_metric is not None and prev_metric != previous_best_metric and (
                            previous_best_metric is None
                            or (maximize and prev_metric > previous_best_metric)
                            or (not maximize and prev_metric < previous_best_metric)
                        )
                        output_handler.on_step_result(step=step - 1, metric=prev_metric, is_new_best=is_new_best)
                        if is_new_best:
                            previous_best_metric = prev_metric
                    except ValueError:
                        # Previous node not found - report as buggy
                        output_handler.on_step_result(step=step - 1, metric=None, is_new_best=False)

                # Update previous_solution_id for next iteration
                previous_solution_id = eval_and_next_solution_response["solution_id"]

                nodes_list = status_response.get("nodes") or []
                best_solution_node = get_best_node_from_status(status_response=status_response)
                current_solution_node = get_node_from_status(
                    status_response=status_response, solution_id=eval_and_next_solution_response["solution_id"]
                )

                # Set best solution and save optimization results
                try:
                    best_solution_code = best_solution_node.code
                except AttributeError:
                    # Can happen if the code was buggy
                    best_solution_code = read_from_path(fp=runs_dir / f"step_0{source_fp.suffix}", is_json=False)

                # Save best solution to .runs/<run-id>/best.<extension>
                write_to_path(fp=runs_dir / f"best{source_fp.suffix}", content=best_solution_code)

                # Notify handler of step generation complete
                output_handler.on_step_generated(
                    step=step,
                    code=eval_and_next_solution_response["code"],
                    plan=eval_and_next_solution_response.get("plan", ""),
                    nodes=nodes_list,
                    current_node=current_solution_node,
                    best_node=best_solution_node,
                    solution_id=eval_and_next_solution_response["solution_id"],
                )

                # Evaluate this new solution and restore original code after
                term_out = run_evaluation_with_file_swap(
                    file_path=source_fp,
                    new_content=eval_and_next_solution_response["code"],
                    original_content=original_source_code,
                    eval_command=eval_command,
                    timeout=eval_timeout,
                )
                if save_logs:
                    save_execution_output(runs_dir, step=step, output=term_out)

                output_handler.on_step_completed(step=step, eval_output=term_out)

            # Final flush if not stopped
            if not user_stop_requested_flag:
                eval_and_next_solution_response = evaluate_feedback_then_suggest_next_solution(
                    console=console,
                    step=total_steps,
                    run_id=resume_resp["run_id"],
                    execution_output=term_out,
                    auth_headers=auth_headers,
                )
                status_response = get_optimization_run_status(
                    console=console, run_id=resume_resp["run_id"], include_history=True, auth_headers=auth_headers
                )

                nodes_final = status_response.get("nodes") or []
                best_solution_node = get_best_node_from_status(status_response=status_response)

                # Report the final step's result now that API has evaluated it
                try:
                    final_node = get_node_from_status(status_response=status_response, solution_id=previous_solution_id)
                    final_metric = final_node.metric
                    is_new_best = final_metric is not None and final_metric != previous_best_metric and (
                        previous_best_metric is None
                        or (maximize and final_metric > previous_best_metric)
                        or (not maximize and final_metric < previous_best_metric)
                    )
                    output_handler.on_step_result(step=total_steps, metric=final_metric, is_new_best=is_new_best)
                except ValueError:
                    # Final node not found - use best result as fallback
                    if best_solution_node:
                        output_handler.on_step_result(step=total_steps, metric=best_solution_node.metric, is_new_best=False)
                    else:
                        output_handler.on_step_result(step=total_steps, metric=None, is_new_best=False)
                best_solution_code = best_solution_node.code if best_solution_node else None
                best_metric_value = status_response.get("best_result", {}).get("metric_value")

                # Save best solution to .runs/<run-id>/best.<extension>
                if best_solution_code:
                    write_to_path(fp=runs_dir / f"best{source_fp.suffix}", content=best_solution_code)

                output_handler.on_run_completed(
                    best_node=best_solution_node, best_metric_value=best_metric_value, nodes=nodes_final
                )

                optimization_completed_normally = True

    except Exception as e:
        try:
            error_message = e.response.json()["detail"]
        except Exception:
            error_message = str(e)
        if "output_handler" in locals():
            output_handler.on_error(message=error_message, run_id=run_id)
        else:
            console.print(f"\n[bold red]Error:[/] {error_message}")
            console.print(f"\n[cyan]To resume this run, use:[/] [bold cyan]weco resume {run_id}[/]\n")
        optimization_completed_normally = False
    finally:
        signal.signal(signal.SIGINT, original_sigint_handler)
        signal.signal(signal.SIGTERM, original_sigterm_handler)
        stop_heartbeat_event.set()
        if heartbeat_thread and heartbeat_thread.is_alive():
            heartbeat_thread.join(timeout=2)

        try:
            run_id = resume_resp.get("run_id")
        except Exception:
            run_id = None

        # Report final status if run exists
        if run_id:
            if optimization_completed_normally:
                status, reason, details = "completed", "completed_successfully", None
            elif user_stop_requested_flag:
                status, reason, details = "terminated", "user_requested_stop", "Run stopped by user request via dashboard."
            else:
                status, reason = "error", "error_cli_internal"
                details = locals().get("error_details") or (
                    traceback.format_exc()
                    if "e" in locals() and isinstance(locals()["e"], Exception)
                    else "CLI terminated unexpectedly without a specific exception captured."
                )

            if best_solution_code and best_solution_code != original_source_code:
                should_apply = apply_change or Confirm.ask(
                    "Would you like to apply the best solution to the source file?", default=True
                )
                if should_apply:
                    write_to_path(fp=source_fp, content=best_solution_code)
                    console.print("\n[green]Best solution applied to the source file.[/]\n")
            else:
                console.print("\n[green]A better solution was not found. No changes to apply.[/]\n")

            report_termination(
                run_id=run_id,
                status_update=status,
                reason=reason,
                details=details,
                auth_headers=current_auth_headers_for_heartbeat,
            )
        if user_stop_requested_flag:
            console.print("[yellow]Run terminated by user request.[/]")
            console.print(f"\n[cyan]To resume this run, use:[/] [bold cyan]weco resume {run_id}[/]\n")
    return optimization_completed_normally or user_stop_requested_flag
