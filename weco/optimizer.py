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
from rich.live import Live
from rich.panel import Panel
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
from .panels import (
    SummaryPanel,
    Node,
    MetricTreePanel,
    EvaluationOutputPanel,
    SolutionPanels,
    create_optimization_layout,
    create_end_optimization_layout,
)
from .utils import read_additional_instructions, read_from_path, write_to_path, run_evaluation, smooth_update


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

    # --- Signal Handler for this optimization run ---
    def signal_handler(signum, frame):
        signal_name = signal.Signals(signum).name
        console.print(f"\n[bold yellow]Termination signal ({signal_name}) received. Shutting down...[/]")

        # Stop heartbeat thread
        stop_heartbeat_event.set()
        if heartbeat_thread and heartbeat_thread.is_alive():
            heartbeat_thread.join(timeout=2)  # Give it a moment to stop

        # Report termination (best effort)
        if current_run_id_for_heartbeat:
            report_termination(
                run_id=current_run_id_for_heartbeat,
                status_update="terminated",
                reason=f"user_terminated_{signal_name.lower()}",
                details=f"Process terminated by signal {signal_name} ({signum}).",
                auth_headers=current_auth_headers_for_heartbeat,
            )
            console.print(f"\n[cyan]To resume this run, use:[/] [bold cyan]weco resume {current_run_id_for_heartbeat}[/]")

        # Exit gracefully
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
            # Authentication failed or user declined
            return False

        current_auth_headers_for_heartbeat = auth_headers

        # --- Process Parameters ---
        maximize = goal.lower() in ["maximize", "max"]

        # Determine the model to use
        if model is None:
            # Default to o4-mini with credit-based billing
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

        # --- Panel Initialization ---
        summary_panel = SummaryPanel(maximize=maximize, metric_name=metric, total_steps=steps, model=model, runs_dir=log_dir)
        solution_panels = SolutionPanels(metric_name=metric, source_fp=source_fp)
        eval_output_panel = EvaluationOutputPanel()
        tree_panel = MetricTreePanel(maximize=maximize)
        layout = create_optimization_layout()
        end_optimization_layout = create_end_optimization_layout()

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
        # Indicate the endpoint failed to return a response and the optimization was unsuccessful
        if run_response is None:
            return False

        run_id = run_response["run_id"]
        run_name = run_response["run_name"]
        current_run_id_for_heartbeat = run_id

        # --- Start Heartbeat Thread ---
        stop_heartbeat_event.clear()
        heartbeat_thread = HeartbeatSender(run_id, auth_headers, stop_heartbeat_event)
        heartbeat_thread.start()

        # --- Live Update Loop ---
        refresh_rate = 4
        with Live(layout, refresh_per_second=refresh_rate) as live:
            # Define the runs directory (.runs/<run-id>) to store logs and results
            runs_dir = pathlib.Path(log_dir) / run_id
            runs_dir.mkdir(parents=True, exist_ok=True)

            # Initialize logging structure if save_logs is enabled
            if save_logs:
                # Initialize JSONL index with metadata
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
            # Write the initial code string to the logs
            write_to_path(fp=runs_dir / f"step_0{source_fp.suffix}", content=run_response["code"])
            # Write the initial code string to the source file path
            write_to_path(fp=source_fp, content=run_response["code"])

            # Update the panels with the initial solution
            # Add run id and run name now that we have it
            summary_panel.set_run_id(run_id=run_id)
            summary_panel.set_run_name(run_name=run_name)
            # Set the step of the progress bar
            summary_panel.set_step(step=0)
            summary_panel.update_thinking(thinking=run_response["plan"])
            # Build the metric tree
            tree_panel.build_metric_tree(
                nodes=[
                    {
                        "solution_id": run_response["solution_id"],
                        "parent_id": None,
                        "code": run_response["code"],
                        "step": 0,
                        "metric_value": None,
                        "is_buggy": None,
                    }
                ]
            )
            # Set the current solution as unevaluated since we haven't run the evaluation function and fed it back to the model yet
            tree_panel.set_unevaluated_node(node_id=run_response["solution_id"])
            # Update the solution panels with the initial solution and get the panel displays
            solution_panels.update(
                current_node=Node(
                    id=run_response["solution_id"], parent_id=None, code=run_response["code"], metric=None, is_buggy=None
                ),
                best_node=None,
            )
            current_solution_panel, best_solution_panel = solution_panels.get_display(current_step=0)
            # Update the live layout with the initial solution panels
            smooth_update(
                live=live,
                layout=layout,
                sections_to_update=[
                    ("summary", summary_panel.get_display()),
                    ("tree", tree_panel.get_display(is_done=False)),
                    ("current_solution", current_solution_panel),
                    ("best_solution", best_solution_panel),
                    ("eval_output", eval_output_panel.get_display()),
                ],
                transition_delay=0.1,
            )

            # Run evaluation on the initial solution
            term_out = run_evaluation(eval_command=eval_command, timeout=eval_timeout)
            # Save logs if requested
            if save_logs:
                save_execution_output(runs_dir, step=0, output=term_out)
            # Update the evaluation output panel
            eval_output_panel.update(output=term_out)
            smooth_update(
                live=live,
                layout=layout,
                sections_to_update=[("eval_output", eval_output_panel.get_display())],
                transition_delay=0.1,
            )

            # Starting from step 1 to steps (inclusive) because the baseline solution is step 0, so we want to optimize for steps worth of steps
            for step in range(1, steps + 1):
                # Re-read instructions from the original source (file path or string) BEFORE each suggest call
                current_additional_instructions = read_additional_instructions(additional_instructions=additional_instructions)
                if run_id:
                    try:
                        current_status_response = get_optimization_run_status(
                            console=console, run_id=run_id, include_history=False, auth_headers=auth_headers
                        )
                        current_run_status_val = current_status_response.get("status")
                        if current_run_status_val == "stopping":
                            console.print("\n[bold yellow]Stop request received. Terminating run gracefully...[/]")
                            user_stop_requested_flag = True
                            break
                    except requests.exceptions.RequestException as e:
                        console.print(f"\n[bold red]Warning: Unable to check run status: {e}. Continuing optimization...[/]")
                    except Exception as e:
                        console.print(f"\n[bold red]Warning: Error checking run status: {e}. Continuing optimization...[/]")

                # Send feedback and get next suggestion
                eval_and_next_solution_response = evaluate_feedback_then_suggest_next_solution(
                    console=console,
                    step=step,
                    run_id=run_id,
                    execution_output=term_out,
                    additional_instructions=current_additional_instructions,
                    auth_headers=auth_headers,
                )
                # Save next solution (.runs/<run-id>/step_<step>.<extension>)
                write_to_path(fp=runs_dir / f"step_{step}{source_fp.suffix}", content=eval_and_next_solution_response["code"])
                # Write the next solution to the source file
                write_to_path(fp=source_fp, content=eval_and_next_solution_response["code"])
                status_response = get_optimization_run_status(
                    console=console, run_id=run_id, include_history=True, auth_headers=auth_headers
                )
                # Update the step of the progress bar, plan and metric tree
                summary_panel.set_step(step=step)
                summary_panel.update_thinking(thinking=eval_and_next_solution_response["plan"])

                nodes_list_from_status = status_response.get("nodes")
                tree_panel.build_metric_tree(nodes=nodes_list_from_status if nodes_list_from_status is not None else [])
                tree_panel.set_unevaluated_node(node_id=eval_and_next_solution_response["solution_id"])

                # Update the solution panels with the next solution and best solution (and score)
                # Figure out if we have a best solution so far
                best_solution_node = get_best_node_from_status(status_response=status_response)
                current_solution_node = get_node_from_status(
                    status_response=status_response, solution_id=eval_and_next_solution_response["solution_id"]
                )

                # Update the solution panels with the current and best solution
                solution_panels.update(current_node=current_solution_node, best_node=best_solution_node)
                current_solution_panel, best_solution_panel = solution_panels.get_display(current_step=step)
                # Clear evaluation output since we are running a evaluation on a new solution
                eval_output_panel.clear()
                smooth_update(
                    live=live,
                    layout=layout,
                    sections_to_update=[
                        ("summary", summary_panel.get_display()),
                        ("tree", tree_panel.get_display(is_done=False)),
                        ("current_solution", current_solution_panel),
                        ("best_solution", best_solution_panel),
                        ("eval_output", eval_output_panel.get_display()),
                    ],
                    transition_delay=0.08,  # Slightly longer delay for more noticeable transitions
                )
                term_out = run_evaluation(eval_command=eval_command, timeout=eval_timeout)
                # Save logs if requested
                if save_logs:
                    save_execution_output(runs_dir, step=step, output=term_out)
                eval_output_panel.update(output=term_out)
                smooth_update(
                    live=live,
                    layout=layout,
                    sections_to_update=[("eval_output", eval_output_panel.get_display())],
                    transition_delay=0.1,
                )

            if not user_stop_requested_flag:
                # Re-read instructions from the original source (file path or string) BEFORE each suggest call
                current_additional_instructions = read_additional_instructions(additional_instructions=additional_instructions)
                # Evaluate the final solution thats been generated
                eval_and_next_solution_response = evaluate_feedback_then_suggest_next_solution(
                    console=console,
                    step=steps,
                    run_id=run_id,
                    execution_output=term_out,
                    additional_instructions=current_additional_instructions,
                    auth_headers=auth_headers,
                )
                summary_panel.set_step(step=steps)
                status_response = get_optimization_run_status(
                    console=console, run_id=run_id, include_history=True, auth_headers=auth_headers
                )
                # No need to update the plan panel since we have finished the optimization
                # Get the optimization run status for
                # the best solution, its score, and the history to plot the tree
                nodes_list_from_status_final = status_response.get("nodes")
                tree_panel.build_metric_tree(
                    nodes=nodes_list_from_status_final if nodes_list_from_status_final is not None else []
                )
                # No need to set any solution to unevaluated since we have finished the optimization
                # and all solutions have been evaluated
                # No neeed to update the current solution panel since we have finished the optimization
                # We only need to update the best solution panel
                # Figure out if we have a best solution so far
                best_solution_node = get_best_node_from_status(status_response=status_response)
                solution_panels.update(current_node=None, best_node=best_solution_node)
                _, best_solution_panel = solution_panels.get_display(current_step=steps)
                # Update the end optimization layout
                final_message = (
                    f"{summary_panel.metric_name.capitalize()} {'maximized' if summary_panel.maximize else 'minimized'}! Best solution {summary_panel.metric_name.lower()} = [green]{status_response['best_result']['metric_value']}[/] 🏆"
                    if best_solution_node is not None and best_solution_node.metric is not None
                    else "[red] No valid solution found.[/]"
                )
                end_optimization_layout["summary"].update(summary_panel.get_display(final_message=final_message))
                end_optimization_layout["tree"].update(tree_panel.get_display(is_done=True))
                end_optimization_layout["best_solution"].update(best_solution_panel)

                # Save optimization results
                # If the best solution does not exist or is has not been measured at the end of the optimization
                # save the original solution as the best solution
                if best_solution_node is not None:
                    best_solution_content = best_solution_node.code
                else:
                    best_solution_content = read_from_path(fp=runs_dir / f"step_0{source_fp.suffix}", is_json=False)

                # Save best solution to .runs/<run-id>/best.<extension>
                write_to_path(fp=runs_dir / f"best{source_fp.suffix}", content=best_solution_content)
                # write the best solution to the source file
                write_to_path(fp=source_fp, content=best_solution_content)
                # Mark as completed normally for the finally block
                optimization_completed_normally = True
                live.update(end_optimization_layout)

    except Exception as e:
        # Catch errors during the main optimization loop or setup
        try:
            error_message = e.response.json()["detail"]
        except Exception:
            error_message = str(e)
        console.print(Panel(f"[bold red]Error: {error_message}", title="[bold red]Optimization Error", border_style="red"))
        console.print(f"\n[cyan]To resume this run, use:[/] [bold cyan]weco resume {run_id}[/]")
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
            console.print(f"\n[cyan]To resume this run, use:[/] [bold cyan]weco resume {run_id}[/]")

    return optimization_completed_normally or user_stop_requested_flag


def resume_optimization(run_id: str, console: Optional[Console] = None) -> bool:
    """Resume an interrupted run from the most recent node and continue optimization."""
    if console is None:
        console = Console()

    # Globals for this optimization run
    heartbeat_thread = None
    stop_heartbeat_event = threading.Event()
    current_run_id_for_heartbeat = None
    current_auth_headers_for_heartbeat = {}

    # Signal handler for this optimization run
    def signal_handler(signum, frame):
        signal_name = signal.Signals(signum).name
        console.print(f"\n[bold yellow]Termination signal ({signal_name}) received. Shutting down...[/]")
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
            console.print(f"\n[cyan]To resume this run, use:[/] [bold cyan]weco resume {current_run_id_for_heartbeat}[/]")
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
            console.print(
                Panel(f"[bold red]Error fetching run status: {e}", title="[bold red]Resume Error", border_style="red")
            )
            return False

        run_status_val = status.get("status")
        if run_status_val not in ("error", "terminated"):
            console.print(
                Panel(
                    f"Run {run_id} cannot be resumed (status: {run_status_val}). Only 'error' or 'terminated' runs can be resumed.",
                    title="[bold yellow]Resume Not Allowed",
                    border_style="yellow",
                )
            )
            return False

        objective = status.get("objective", {})
        metric_name = objective.get("metric_name", "metric")
        maximize = bool(objective.get("maximize", True))

        optimizer = status.get("optimizer", {})

        console.print("[cyan]Resume Run Confirmation[/]")
        console.print(f"  Run ID: {run_id}")
        console.print(f"  Run Name: {status.get('run_name', 'N/A')}")
        console.print(f"  Status: {run_status_val}")
        # Objective and model
        console.print(f"  Objective: {metric_name} ({'maximize' if maximize else 'minimize'})")
        model_name = (
            (optimizer.get("code_generator") or {}).get("model")
            or (optimizer.get("evaluator") or {}).get("model")
            or "unknown"
        )
        console.print(f"  Model: {model_name}")
        console.print(f"  Eval Command: {objective.get('evaluation_command', 'N/A')}")
        # Steps summary
        total_steps = optimizer.get("steps")
        current_step = int(status["current_step"])
        steps_remaining = int(total_steps) - int(current_step)
        console.print(f"  Total Steps: {total_steps} | Resume Step: {current_step} | Steps Remaining: {steps_remaining}")
        console.print(f"  Last Updated: {status.get('updated_at', 'N/A')}")
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
        additional_instructions = resume_resp.get("additional_instructions")

        # Write last solution code to source path
        source_fp = pathlib.Path(source_path)
        source_fp.parent.mkdir(parents=True, exist_ok=True)
        code_to_restore = resume_resp.get("code") or resume_resp.get("source_code") or ""
        write_to_path(fp=source_fp, content=code_to_restore)

        # Prepare UI panels
        summary_panel = SummaryPanel(
            maximize=maximize, metric_name=metric_name, total_steps=total_steps, model=model_name, runs_dir=log_dir
        )
        summary_panel.set_run_id(run_id=resume_resp["run_id"])
        if resume_resp.get("run_name"):
            summary_panel.set_run_name(resume_resp.get("run_name"))
        summary_panel.set_step(step=current_step)
        summary_panel.update_thinking(resume_resp.get("plan"))

        solution_panels = SolutionPanels(metric_name=metric_name, source_fp=source_fp)
        eval_output_panel = EvaluationOutputPanel()
        tree_panel = MetricTreePanel(maximize=maximize)
        layout = create_optimization_layout()
        end_optimization_layout = create_end_optimization_layout()

        # Build tree from nodes returned by status (history)
        nodes_list_from_status = status.get("nodes") or []
        tree_panel.build_metric_tree(nodes=nodes_list_from_status)

        # Compute best and current nodes
        best_solution_node = get_best_node_from_status(status_response=status)
        current_solution_node = get_node_from_status(status_response=status, solution_id=resume_resp.get("solution_id"))

        # Ensure runs dir exists
        runs_dir = pathlib.Path(log_dir) / resume_resp["run_id"]
        runs_dir.mkdir(parents=True, exist_ok=True)
        # Persist last step's code into logs as step_<current_step>
        write_to_path(fp=runs_dir / f"step_{current_step}{source_fp.suffix}", content=code_to_restore)

        # Start Heartbeat Thread
        stop_heartbeat_event.clear()
        heartbeat_thread = HeartbeatSender(resume_resp["run_id"], auth_headers, stop_heartbeat_event)
        heartbeat_thread.start()
        current_run_id_for_heartbeat = resume_resp["run_id"]

        # Seed solution panels with current and best nodes
        solution_panels.update(current_node=current_solution_node, best_node=best_solution_node)

        # --- Live UI ---
        refresh_rate = 4
        with Live(layout, refresh_per_second=refresh_rate) as live:
            # Initial panels
            current_solution_panel, best_solution_panel = solution_panels.get_display(current_step=current_step)
            # Use backend-provided execution output only (no fallback)
            term_out = resume_resp.get("execution_output") or ""
            eval_output_panel.update(output=term_out)

            # Update the initial panels
            smooth_update(
                live=live,
                layout=layout,
                sections_to_update=[
                    ("summary", summary_panel.get_display()),
                    ("tree", tree_panel.get_display(is_done=False)),
                    ("current_solution", current_solution_panel),
                    ("best_solution", best_solution_panel),
                    ("eval_output", eval_output_panel.get_display()),
                ],
                transition_delay=0.1,
            )

            # If missing output, evaluate once before first suggest
            if term_out is None or len(term_out.strip()) == 0:
                term_out = run_evaluation(eval_command=eval_command, timeout=eval_timeout)
                eval_output_panel.update(output=term_out)
                # Update the evaluation output panel
                smooth_update(
                    live=live,
                    layout=layout,
                    sections_to_update=[("eval_output", eval_output_panel.get_display())],
                    transition_delay=0.1,
                )

            if save_logs:
                save_execution_output(runs_dir, step=current_step, output=term_out)

            # Continue optimization: steps current_step+1..total_steps
            for step in range(current_step + 1, total_steps + 1):
                # Stop polling
                try:
                    current_status_response = get_optimization_run_status(
                        console=console, run_id=resume_resp["run_id"], include_history=False, auth_headers=auth_headers
                    )
                    if current_status_response.get("status") == "stopping":
                        console.print("\n[bold yellow]Stop request received. Terminating run gracefully...[/]")
                        user_stop_requested_flag = True
                        break
                except requests.exceptions.RequestException as e:
                    console.print(f"\n[bold red]Warning: Unable to check run status: {e}. Continuing optimization...[/]")
                except Exception as e:
                    console.print(f"\n[bold red]Warning: Error checking run status: {e}. Continuing optimization...[/]")

                # Suggest next
                eval_and_next_solution_response = evaluate_feedback_then_suggest_next_solution(
                    console=console,
                    step=step,
                    run_id=resume_resp["run_id"],
                    execution_output=term_out,
                    additional_instructions=additional_instructions,
                    auth_headers=auth_headers,
                )

                # Save next solution file(s)
                write_to_path(fp=runs_dir / f"step_{step}{source_fp.suffix}", content=eval_and_next_solution_response["code"])
                write_to_path(fp=source_fp, content=eval_and_next_solution_response["code"])

                # Refresh status with history and update panels
                status_response = get_optimization_run_status(
                    console=console, run_id=resume_resp["run_id"], include_history=True, auth_headers=auth_headers
                )
                summary_panel.set_step(step=step)
                summary_panel.update_thinking(thinking=eval_and_next_solution_response.get("plan", ""))
                nodes_list = status_response.get("nodes") or []
                tree_panel.build_metric_tree(nodes=nodes_list)
                tree_panel.set_unevaluated_node(node_id=eval_and_next_solution_response["solution_id"])
                best_solution_node = get_best_node_from_status(status_response=status_response)
                current_solution_node = get_node_from_status(
                    status_response=status_response, solution_id=eval_and_next_solution_response["solution_id"]
                )
                solution_panels.update(current_node=current_solution_node, best_node=best_solution_node)
                current_solution_panel, best_solution_panel = solution_panels.get_display(current_step=step)
                eval_output_panel.clear()
                smooth_update(
                    live=live,
                    layout=layout,
                    sections_to_update=[
                        ("summary", summary_panel.get_display()),
                        ("tree", tree_panel.get_display(is_done=False)),
                        ("current_solution", current_solution_panel),
                        ("best_solution", best_solution_panel),
                        ("eval_output", eval_output_panel.get_display()),
                    ],
                    transition_delay=0.08,
                )

                # Evaluate this new solution
                term_out = run_evaluation(eval_command=eval_command, timeout=eval_timeout)
                if save_logs:
                    save_execution_output(runs_dir, step=step, output=term_out)
                eval_output_panel.update(output=term_out)
                smooth_update(
                    live=live,
                    layout=layout,
                    sections_to_update=[("eval_output", eval_output_panel.get_display())],
                    transition_delay=0.1,
                )

            # Final flush if not stopped
            if not user_stop_requested_flag:
                eval_and_next_solution_response = evaluate_feedback_then_suggest_next_solution(
                    console=console,
                    step=total_steps,
                    run_id=resume_resp["run_id"],
                    execution_output=term_out,
                    additional_instructions=additional_instructions,
                    auth_headers=auth_headers,
                )
                summary_panel.set_step(step=total_steps)
                status_response = get_optimization_run_status(
                    console=console, run_id=resume_resp["run_id"], include_history=True, auth_headers=auth_headers
                )
                nodes_final = status_response.get("nodes") or []
                tree_panel.build_metric_tree(nodes=nodes_final)
                # Best solution panel and final message
                best_solution_node = get_best_node_from_status(status_response=status_response)
                solution_panels.update(current_node=None, best_node=best_solution_node)
                _, best_solution_panel = solution_panels.get_display(current_step=total_steps)
                final_message = (
                    f"{summary_panel.metric_name.capitalize()} {'maximized' if summary_panel.maximize else 'minimized'}! Best solution {summary_panel.metric_name.lower()} = [green]{status_response['best_result']['metric_value']}[/] 🏆"
                    if best_solution_node is not None and best_solution_node.metric is not None
                    else "[red] No valid solution found.[/]"
                )
                end_optimization_layout["summary"].update(summary_panel.get_display(final_message=final_message))
                end_optimization_layout["tree"].update(tree_panel.get_display(is_done=True))
                end_optimization_layout["best_solution"].update(best_solution_panel)

                # Save best
                if best_solution_node is not None:
                    best_solution_content = best_solution_node.code
                else:
                    best_solution_content = read_from_path(fp=runs_dir / f"step_0{source_fp.suffix}", is_json=False)

                write_to_path(fp=runs_dir / f"best{source_fp.suffix}", content=best_solution_content)
                write_to_path(fp=source_fp, content=best_solution_content)
                optimization_completed_normally = True
                live.update(end_optimization_layout)

    except Exception as e:
        try:
            error_message = e.response.json()["detail"]
        except Exception:
            error_message = str(e)
        console.print(Panel(f"[bold red]Error: {error_message}", title="[bold red]Optimization Error", border_style="red"))
        console.print(f"\n[cyan]To resume this run, use:[/] [bold cyan]weco resume {run_id}[/]")
        optimization_completed_normally = False
    finally:
        signal.signal(signal.SIGINT, original_sigint_handler)
        signal.signal(signal.SIGTERM, original_sigterm_handler)
        stop_heartbeat_event.set()
        if heartbeat_thread and heartbeat_thread.is_alive():
            heartbeat_thread.join(timeout=2)

        run_id = resume_resp.get("run_id")
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
            report_termination(
                run_id=run_id,
                status_update=status,
                reason=reason,
                details=details,
                auth_headers=current_auth_headers_for_heartbeat,
            )
        if user_stop_requested_flag:
            console.print("[yellow]Run terminated by user request.[/]")
            console.print(f"\n[cyan]To resume this run, use:[/] [bold cyan]weco resume {run_id}[/]")
    return optimization_completed_normally or user_stop_requested_flag
