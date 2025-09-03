import pathlib
import math
import threading
import signal
import sys
import traceback
import json
from datetime import datetime
from typing import Optional, Union, Tuple
from rich.console import Console
from rich.live import Live
from rich.panel import Panel

from .api import (
    start_optimization_run,
    evaluate_feedback_then_suggest_next_solution,
    get_optimization_run_status,
    send_heartbeat,
    report_termination,
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
from .utils import (
    read_api_keys_from_env,
    read_additional_instructions,
    read_from_path,
    write_to_path,
    run_evaluation,
    smooth_update,
    format_number,
)
from .constants import DEFAULT_API_TIMEOUT


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
    entry = {"step": step, "timestamp": timestamp, "output_file": f"outputs/step_{step}.out.txt", "output_length": len(output)}
    with open(jsonl_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def initialize_or_append_logs(
    runs_dir: pathlib.Path,
    run_id: str,
    run_name: str,
    context: str,
    eval_command: str,
    metric_name: str,
    maximize: bool,
    total_steps: int,
    **extra_fields,
) -> None:
    """
    Initialize or append to the execution logs JSONL file.

    Args:
        runs_dir: Path to the run directory (.runs/<run_id>)
        run_id: The run ID
        run_name: The run name
        context: Context of the operation ("run", "resume", "extend")
        eval_command: The evaluation command
        metric_name: The metric name
        maximize: Whether maximizing or minimizing
        total_steps: Total number of steps
        **extra_fields: Additional fields to include in the metadata
    """
    jsonl_file = runs_dir / "exec_output.jsonl"

    # Check if this is a fresh run or continuation
    if context == "run" or not jsonl_file.exists():
        # Initialize new JSONL file with metadata
        metadata = {
            "type": "metadata",
            "run_id": run_id,
            "run_name": run_name,
            "started": datetime.now().isoformat(),
            "eval_command": eval_command,
            "metric": metric_name,
            "goal": "maximize" if maximize else "minimize",
            "total_steps": total_steps,
            **extra_fields,
        }
        with open(jsonl_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(metadata) + "\n")
    else:
        # Append context marker for resume/extend
        context_marker = {"type": context, f"{context}_at": datetime.now().isoformat(), **extra_fields}
        with open(jsonl_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(context_marker) + "\n")


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


# ============================================================================
# Helper Functions for Common Optimization Patterns
# ============================================================================


def initialize_panels(
    maximize: bool,
    metric_name: str,
    total_steps: int,
    model: str,
    log_dir: str,
    source_fp: pathlib.Path,
    run_id: Optional[str] = None,
    run_name: Optional[str] = None,
) -> tuple:
    """
    Initialize all panels used in optimization.

    Returns:
        tuple: (summary_panel, solution_panels, eval_output_panel, tree_panel)
    """
    summary_panel = SummaryPanel(
        maximize=maximize,
        metric_name=metric_name,
        total_steps=total_steps,
        model=model,
        runs_dir=log_dir,
        run_id=run_id,
        run_name=run_name,
    )
    solution_panels = SolutionPanels(metric_name=metric_name, source_fp=source_fp)
    eval_output_panel = EvaluationOutputPanel()
    tree_panel = MetricTreePanel(maximize=maximize)

    return summary_panel, solution_panels, eval_output_panel, tree_panel


def update_live_display(
    live,
    layout,
    summary_panel,
    tree_panel,
    solution_panels,
    eval_output_panel,
    current_step: int,
    is_done: bool = False,
    transition_delay: float = 0.08,
) -> None:
    """Update the live display with all panel contents."""
    current_solution_panel, best_solution_panel = solution_panels.get_display(current_step=current_step)
    smooth_update(
        live=live,
        layout=layout,
        sections_to_update=[
            ("summary", summary_panel.get_display()),
            ("tree", tree_panel.get_display(is_done=is_done)),
            ("current_solution", current_solution_panel),
            ("best_solution", best_solution_panel),
            ("eval_output", eval_output_panel.get_display()),
        ],
        transition_delay=transition_delay,
    )


def create_node_from_status(node_data: dict) -> Node:
    """Create a Node object from API status response data."""
    return Node(
        id=node_data["solution_id"],
        parent_id=node_data.get("parent_id"),
        code=node_data.get("code"),
        metric=node_data.get("metric_value"),
        is_buggy=node_data.get("is_buggy"),
    )


def find_node_in_status(status_response: dict, solution_id: str) -> Optional[Node]:
    """Find and create a Node from status response by solution ID."""
    if not status_response.get("nodes"):
        return None

    for node_data in status_response["nodes"]:
        if node_data["solution_id"] == solution_id:
            return create_node_from_status(node_data)
    return None


def get_best_node_from_status(status_response: dict) -> Optional[Node]:
    """Extract the best solution node from status response."""
    if not status_response.get("best_result"):
        return None

    best_result = status_response["best_result"]
    return Node(
        id=best_result["solution_id"],
        parent_id=best_result.get("parent_id"),
        code=best_result.get("code"),
        metric=best_result.get("metric_value"),
        is_buggy=best_result.get("is_buggy"),
    )


def run_and_log_evaluation(
    eval_command: str,
    eval_timeout: Optional[int],
    save_logs: bool,
    runs_dir: pathlib.Path,
    step: int,
    eval_output_panel: EvaluationOutputPanel,
) -> str:
    """Run evaluation, optionally save logs, and update output panel."""
    execution_output = run_evaluation(eval_command, timeout=eval_timeout)

    if save_logs:
        save_execution_output(runs_dir, step=step, output=execution_output)

    eval_output_panel.update(execution_output)
    return execution_output


def write_solution_files(code: str, source_fp: pathlib.Path, runs_dir: pathlib.Path, step: int) -> None:
    """Write solution code to both source file and log directory."""
    write_to_path(fp=source_fp, content=code)
    write_to_path(fp=runs_dir / f"step_{step}{source_fp.suffix}", content=code)


def run_optimization_loop(
    live,
    layout,
    console: Console,
    run_id: str,
    start_step: int,
    total_steps: int,
    eval_command: str,
    eval_timeout: Optional[int],
    save_logs: bool,
    runs_dir: pathlib.Path,
    source_fp: pathlib.Path,
    summary_panel,
    solution_panels,
    eval_output_panel,
    tree_panel,
    api_keys: dict,
    auth_headers: dict,
    stop_heartbeat_event: threading.Event,
    initial_execution_output: str = "",
    additional_instructions: Optional[str] = None,
    api_timeout: Union[int, Tuple[int, int]] = DEFAULT_API_TIMEOUT,
) -> tuple[bool, bool]:
    """
    Shared optimization loop logic for execute, resume, and extend operations.

    Returns:
        tuple: (optimization_completed_normally, user_stop_requested_flag)
    """
    optimization_completed_normally = False
    user_stop_requested_flag = False
    execution_output = ""  # Initialize for the loop

    for step in range(start_step, total_steps + 1):
        # Check for user stop request first (before updating progress)
        try:
            run_status = get_optimization_run_status(
                console, run_id, include_history=False, timeout=api_timeout, auth_headers=auth_headers
            )
            if run_status and run_status.get("status") == "stopping":
                user_stop_requested_flag = True
                console.print("\n[yellow]User requested stop via dashboard. Stopping optimization...[/]")
                stop_heartbeat_event.set()
                break
        except Exception as e:
            console.print(f"\n[bold red]Warning: Error checking run status: {e}. Continuing optimization...[/]")

        # Handle execution output for the first step
        # For execute_optimization: initial_execution_output contains the evaluation from step 0
        # For resume/extend: initial_execution_output may contain cached output or need evaluation
        if step == start_step and initial_execution_output and initial_execution_output.strip():
            # Use the provided initial execution output
            execution_output = initial_execution_output
        elif step == start_step and (not initial_execution_output or not initial_execution_output.strip()):
            # No valid initial output provided - need to evaluate the previous step's solution
            eval_output_panel.clear()
            execution_output = run_and_log_evaluation(
                eval_command=eval_command,
                eval_timeout=eval_timeout,
                save_logs=save_logs,
                runs_dir=runs_dir,
                step=step - 1,  # Evaluate the previous step's solution that was just restored
                eval_output_panel=eval_output_panel,
            )

        # Get next solution
        response = evaluate_feedback_then_suggest_next_solution(
            console=console,
            run_id=run_id,
            execution_output=execution_output,
            additional_instructions=additional_instructions,
            api_keys=api_keys,
            auth_headers=auth_headers,
            timeout=api_timeout,
        )

        if not response:
            console.print("[bold red]Failed to get next solution. Stopping optimization.[/]")
            break

        # Update panels with new solution
        if response.get("code"):
            write_solution_files(code=response["code"], source_fp=source_fp, runs_dir=runs_dir, step=step)

        # Refresh the entire tree from the status to avoid synchronization issues
        status_response = get_optimization_run_status(
            console=console, run_id=run_id, include_history=True, timeout=api_timeout, auth_headers=auth_headers
        )

        # Use the actual current step from the API response for consistency
        if status_response:
            current_step_from_api = status_response.get("current_step", step)
            # Update progress bar with API step to ensure consistency
            summary_panel.set_step(current_step_from_api)
        else:
            # Fallback to loop step if API response is unavailable
            summary_panel.set_step(step)

        # Rebuild the metric tree with all nodes
        if status_response and status_response.get("nodes"):
            tree_panel.build_metric_tree(nodes=status_response["nodes"])

            # Mark the current node as unevaluated if it's a new one
            if response.get("solution_id"):
                try:
                    tree_panel.set_unevaluated_node(node_id=response["solution_id"])
                except Exception:
                    pass  # Node might not exist yet

            # Update solution panels with current and best nodes
            current_solution_node = find_node_in_status(status_response, response.get("solution_id"))
            best_solution_node = get_best_node_from_status(status_response)

            if current_solution_node:
                solution_panels.update(current_node=current_solution_node, best_node=best_solution_node)
            elif response.get("code"):
                # Fallback if node not found in status
                current_node = Node(
                    id=response.get("solution_id", f"temp_{step}"),
                    parent_id=response.get("parent_id"),
                    code=response["code"],
                    metric=None,
                    is_buggy=None,
                )
                solution_panels.update(current_node=current_node, best_node=best_solution_node)

        # Update token usage and thinking
        if response.get("usage"):
            summary_panel.update_token_counts(usage=response["usage"])
        if response.get("plan"):
            summary_panel.update_thinking(thinking=response["plan"])

        # Update the display
        update_live_display(
            live=live,
            layout=layout,
            summary_panel=summary_panel,
            tree_panel=tree_panel,
            solution_panels=solution_panels,
            eval_output_panel=eval_output_panel,
            current_step=step,
            is_done=False,
            transition_delay=0.08,
        )

        # Check if optimization is done
        if response.get("is_done"):
            optimization_completed_normally = True
            stop_heartbeat_event.set()
            break

        # Handle evaluation after solution
        # Always evaluate after getting solution (this is the pattern for all calls)
        # Clear evaluation output since we are running evaluation on a new solution
        eval_output_panel.clear()
        update_live_display(
            live=live,
            layout=layout,
            summary_panel=summary_panel,
            tree_panel=tree_panel,
            solution_panels=solution_panels,
            eval_output_panel=eval_output_panel,
            current_step=step,
            is_done=False,
            transition_delay=0.08,
        )
        execution_output = run_and_log_evaluation(
            eval_command=eval_command,
            eval_timeout=eval_timeout,
            save_logs=save_logs,
            runs_dir=runs_dir,
            step=step,
            eval_output_panel=eval_output_panel,
        )
        smooth_update(
            live=live,
            layout=layout,
            sections_to_update=[("eval_output", eval_output_panel.get_display())],
            transition_delay=0.1,
        )

    # Handle final evaluation
    if not user_stop_requested_flag:
        # Still need to evaluate the final solution as the last call to `evaluate_feedback_then_suggest_next_solution`
        # evaluated solution for `step=total_steps - 1` and generated the solution for `step=total_steps`.
        response = evaluate_feedback_then_suggest_next_solution(
            console=console,
            run_id=run_id,
            execution_output=execution_output,
            additional_instructions=additional_instructions,
            api_keys=api_keys,
            timeout=api_timeout,
            auth_headers=auth_headers,
        )
        summary_panel.update_token_counts(usage=response["usage"])
        status_response = get_optimization_run_status(
            console=console, run_id=run_id, include_history=True, timeout=api_timeout, auth_headers=auth_headers
        )

        # Update panels with final status using consistent step from API
        if status_response:
            final_step = status_response.get("current_step", total_steps)
            summary_panel.set_step(step=final_step)
            if status_response.get("nodes"):
                tree_panel.build_metric_tree(nodes=status_response["nodes"])
        else:
            # Fallback if no API response
            summary_panel.set_step(step=total_steps)

        optimization_completed_normally = True

    return optimization_completed_normally, user_stop_requested_flag


def prime_live_layout(
    layout, summary_panel, tree_panel, solution_panels, eval_output_panel, current_step: int, is_done: bool = False
):
    """
    Helper function to hydrate the Live layout with current panel states.

    Args:
        layout: The layout dictionary to update
        summary_panel: The summary panel instance
        tree_panel: The metric tree panel instance
        solution_panels: The solution panels instance
        eval_output_panel: The evaluation output panel instance
        current_step: The current optimization step
        is_done: Whether optimization is complete
    """
    layout["summary"].update(summary_panel.get_display())
    layout["tree"].update(tree_panel.get_display(is_done=is_done))
    current_solution_panel, best_solution_panel = solution_panels.get_display(current_step=current_step)
    layout["current_solution"].update(current_solution_panel)
    layout["best_solution"].update(best_solution_panel)
    layout["eval_output"].update(eval_output_panel.get_display())


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
                timeout=3,
            )
            # Suggest resume command
            console.print(
                f"\n[bold cyan]To resume this run, use:[/] [bold green]weco resume {current_run_id_for_heartbeat}[/]"
            )
        else:
            # If run_id not available yet, show generic message
            console.print(f"\n[bold cyan]Run interrupted. Check {log_dir}/ directory for run ID to resume.[/]")

        # Exit gracefully
        sys.exit(0)

    # Set up signal handlers for this run
    original_sigint_handler = signal.signal(signal.SIGINT, signal_handler)
    original_sigterm_handler = signal.signal(signal.SIGTERM, signal_handler)

    run_id = None
    optimization_completed_normally = False
    user_stop_requested_flag = False

    try:
        llm_api_keys = read_api_keys_from_env()

        # --- Login/Authentication Handling ---
        weco_api_key, auth_headers = handle_authentication(console, llm_api_keys)
        if weco_api_key is None and not llm_api_keys:
            # Authentication failed and no LLM keys available
            return False

        current_auth_headers_for_heartbeat = auth_headers

        # --- Process Parameters ---
        maximize = goal.lower() in ["maximize", "max"]

        # Determine the model to use
        if model is None:
            from .utils import determine_default_model

            model = determine_default_model(llm_api_keys)

        code_generator_config = {"model": model}
        evaluator_config = {"model": model, "include_analysis": True}
        search_policy_config = {
            "num_drafts": max(1, math.ceil(0.15 * steps)),
            "debug_prob": 0.5,
            "max_debug_depth": max(1, math.ceil(0.1 * steps)),
        }
        api_timeout = DEFAULT_API_TIMEOUT
        processed_additional_instructions = read_additional_instructions(additional_instructions=additional_instructions)
        source_fp = pathlib.Path(source)
        source_code = read_from_path(fp=source_fp, is_json=False)

        # --- Panel Initialization ---
        summary_panel, solution_panels, eval_output_panel, tree_panel = initialize_panels(
            maximize=maximize, metric_name=metric, total_steps=steps, model=model, log_dir=log_dir, source_fp=source_fp
        )
        layout = create_optimization_layout()
        end_optimization_layout = create_end_optimization_layout()

        # --- Start Optimization Run ---
        run_response = start_optimization_run(
            console=console,
            source_code=source_code,
            evaluation_command=eval_command,
            metric_name=metric,
            maximize=maximize,
            steps=steps,
            code_generator_config=code_generator_config,
            evaluator_config=evaluator_config,
            search_policy_config=search_policy_config,
            additional_instructions=processed_additional_instructions,
            api_keys=llm_api_keys,
            source_path=str(source) if source else None,  # Ensure it's a string for JSON serialization
            eval_timeout=eval_timeout,  # Store the evaluation timeout (None means no limit)
            save_logs=save_logs,  # Store the save_logs preference
            log_dir=log_dir,  # Store the log directory path
            auth_headers=auth_headers,
            timeout=api_timeout,
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
                initialize_or_append_logs(runs_dir, run_id, run_name, "run", eval_command, metric, maximize, steps)
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
            # Update the token counts
            summary_panel.update_token_counts(usage=run_response["usage"])
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
            # Update the live layout with the initial solution panels
            update_live_display(
                live=live,
                layout=layout,
                summary_panel=summary_panel,
                tree_panel=tree_panel,
                solution_panels=solution_panels,
                eval_output_panel=eval_output_panel,
                current_step=0,
                is_done=False,
                transition_delay=0.1,
            )

            # Run evaluation on the initial solution
            term_out = run_and_log_evaluation(
                eval_command=eval_command,
                eval_timeout=eval_timeout,
                save_logs=save_logs,
                runs_dir=runs_dir,
                step=0,
                eval_output_panel=eval_output_panel,
            )
            smooth_update(
                live=live,
                layout=layout,
                sections_to_update=[("eval_output", eval_output_panel.get_display())],
                transition_delay=0.1,
            )

            # Use shared optimization loop for steps 1 to steps (inclusive)
            optimization_completed_normally, user_stop_requested_flag = run_optimization_loop(
                live=live,
                layout=layout,
                console=console,
                run_id=run_id,
                start_step=1,
                total_steps=steps,
                eval_command=eval_command,
                eval_timeout=eval_timeout,
                save_logs=save_logs,
                runs_dir=runs_dir,
                source_fp=source_fp,
                summary_panel=summary_panel,
                solution_panels=solution_panels,
                eval_output_panel=eval_output_panel,
                tree_panel=tree_panel,
                api_keys=llm_api_keys,
                auth_headers=auth_headers,
                stop_heartbeat_event=stop_heartbeat_event,
                initial_execution_output=term_out,
                additional_instructions=processed_additional_instructions,
                api_timeout=api_timeout,
            )

            # Handle end-of-optimization display and file saving
            if optimization_completed_normally:
                # Get final status for display
                status_response = get_optimization_run_status(
                    console=console, run_id=run_id, include_history=True, timeout=api_timeout, auth_headers=auth_headers
                )

                # Update final display
                # Update tree panel and solution panels for final display
                nodes_list_from_status_final = status_response.get("nodes")
                tree_panel.build_metric_tree(
                    nodes=nodes_list_from_status_final if nodes_list_from_status_final is not None else []
                )

                best_solution_node = get_best_node_from_status(status_response)
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
                    best_solution_code = best_solution_node.code
                    best_solution_score = best_solution_node.metric
                else:
                    best_solution_code = None
                    best_solution_score = None

                if best_solution_code is None or best_solution_score is None:
                    best_solution_content = f"# Weco could not find a better solution\n\n{read_from_path(fp=runs_dir / f'step_0{source_fp.suffix}', is_json=False)}"
                else:
                    # Format score for the comment
                    best_score_str = (
                        format_number(best_solution_score)
                        if best_solution_score is not None and isinstance(best_solution_score, (int, float))
                        else "N/A"
                    )
                    best_solution_content = (
                        f"# Best solution from Weco with a score of {best_score_str}\n\n{best_solution_code}"
                    )
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
        # Ensure optimization_completed_normally is False
        optimization_completed_normally = False
        # Suggest resume command if we have a run_id
        if run_id:
            console.print(f"\n[bold cyan]To resume this run, use:[/] [bold green]weco resume {run_id}[/]")
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

            report_termination(run_id, status, reason, details, current_auth_headers_for_heartbeat)

        # Handle exit messages
        if optimization_completed_normally:
            # Run completed successfully - show extend option
            if run_id:
                console.print(
                    f"\n[bold cyan]To extend this run with more steps, use:[/] [bold green]weco extend {run_id} <additional_steps>[/]"
                )
        elif user_stop_requested_flag:
            # Run was terminated by user - show resume option
            console.print("[yellow]Run terminated by user request.[/]")
            if run_id:
                console.print(f"\n[bold cyan]To resume this run, use:[/] [bold green]weco resume {run_id}[/]")

    return optimization_completed_normally or user_stop_requested_flag


def resume_optimization(
    run_id: str, skip_validation: bool = False, log_dir: str = ".runs", console: Optional[Console] = None
) -> bool:
    """
    Resume an interrupted optimization run from the last completed step.

    Args:
        run_id: The ID of the run to resume
        skip_validation: Whether to skip environment validation checks
        log_dir: Directory to store logs and results
        console: Rich console for output

    Returns:
        bool: True if optimization completed successfully, False otherwise
    """
    if console is None:
        console = Console()

    from .api import resume_optimization_run, get_optimization_run_status

    # Read authentication and API keys
    api_keys = read_api_keys_from_env()
    api_key, auth_headers = handle_authentication(console, api_keys)

    # First, check the run status
    run_status = get_optimization_run_status(console, run_id, include_history=False, auth_headers=auth_headers)
    if not run_status:
        console.print(f"[bold red]Failed to get run status for ID: {run_id}[/]")
        console.print("[yellow]Possible reasons:[/]")
        console.print("  • The run ID may be incorrect")
        console.print("  • The run may not exist or has been deleted")
        console.print("  • There may be a temporary server issue")
        console.print("\n[cyan]Please verify the run ID and try again.[/]")
        return False

    current_status = run_status.get("status")

    # Check if run is completed
    if current_status == "completed":
        console.print("[bold red]Run is already completed. Use 'weco extend' command to add more steps.[/]")
        return False

    # Check if run is already running
    if current_status == "running":
        console.print("[bold red]Run is already running. Please wait for it to complete or stop it first.[/]")
        return False

    # Environment validation (unless skipped) - moved before API call
    if not skip_validation:
        console.print("\n[bold cyan]Resume Validation[/]")
        console.print(f"Run ID: {run_id}")
        console.print(f"Status: {current_status}")

        # Validation prompts
        console.print("\n[bold yellow]Please confirm:[/]")
        console.print("1. Your evaluation script hasn't been modified since the run started")
        console.print("2. Your test environment is the same (dependencies, data files, etc.)")
        console.print("3. You haven't modified any of the generated solutions")

        if console.input("\n[bold]Continue with resume? \\[y]es/No (default=No): [/]").lower().strip() not in ["y", "yes"]:
            console.print("[yellow]Resume cancelled by user.[/]")
            return False

    # Now call resume endpoint (this changes DB status to "running")
    resume_info = resume_optimization_run(console=console, run_id=run_id, api_keys=api_keys, auth_headers=auth_headers)
    if not resume_info:
        console.print("[bold red]Failed to resume run. Please check the run ID and try again.[/]")
        return False

    # Extract resume information
    last_step = resume_info["last_completed_step"]
    total_steps = resume_info["total_steps"]
    evaluation_command = resume_info["evaluation_command"]
    source_code = resume_info["source_code"]
    last_node = resume_info["last_solution"]  # API returns last_solution but it's actually the last node
    run_name = resume_info.get("run_name", run_id)
    source_path_from_api = resume_info.get("source_path")  # Get source_path from API
    eval_timeout = resume_info.get("eval_timeout")  # Get eval_timeout from API
    save_logs = resume_info.get("save_logs", False)  # Get save_logs from API (inherit from original run)
    # Use the provided log_dir parameter instead of getting from API

    if save_logs:
        console.print("[dim]Local logging enabled (from original run)[/]")

    # Show detailed resume information now that we have it
    console.print("\n[bold green]Resume Details:[/]")
    console.print(f"[cyan]Run Name:[/] {run_name}")
    console.print(f"[cyan]Last completed step:[/] {last_step}/{total_steps}")
    console.print(f"[cyan]Will resume from step:[/] {last_step + 1}")
    console.print(f"[cyan]Evaluation command:[/] {evaluation_command}")

    # Note if the last solution was buggy
    last_was_buggy = last_node.get("is_buggy", False)
    if last_was_buggy:
        console.print(f"[yellow]Note: Step {last_step} resulted in a bug. Continuing optimization.[/]")

    # Get metric info from run_status
    objective = run_status.get("objective", {})
    metric_name = objective.get("metric_name", "metric")
    maximize = objective.get("maximize", True)

    # Get optimizer config for model info
    optimizer_config = run_status.get("optimizer", {})
    model = optimizer_config.get("code_generator", {}).get("model")
    if not model:
        # Use helper function to determine default model based on API keys
        from .utils import determine_default_model

        model = determine_default_model(api_keys)

    # Log directory was already retrieved from resume_info above
    run_log_dir = pathlib.Path(log_dir) / run_id

    # Ensure log directory exists
    run_log_dir.mkdir(parents=True, exist_ok=True)

    # Write the last solution to the appropriate file (always overwrite to ensure it's current)
    last_solution_path = run_log_dir / f"step_{last_step}.py"
    if last_node.get("code"):
        write_to_path(last_solution_path, last_node["code"])
    else:
        # If the last node doesn't have code, we can't resume
        console.print("[bold red]Error: Last solution node doesn't have code. Cannot resume this run.[/]")
        return False

    # Use source path from API if available, otherwise ask the user
    if source_path_from_api and pathlib.Path(source_path_from_api).exists():
        source_path = source_path_from_api
        console.print(f"[cyan]Using source file from original run: {source_path}[/]")
    else:
        # Try to find the source file automatically by looking for common patterns
        # First check if there's a file matching the metric name pattern
        possible_files = []
        for pattern in ["train.py", "main.py", "solution.py", "*.py"]:
            files = list(pathlib.Path(".").glob(pattern))
            possible_files.extend(files)

        # Remove duplicates and filter to actual files
        possible_files = list(set(f for f in possible_files if f.is_file()))

        if len(possible_files) == 1:
            # Only one Python file found, use it
            source_path = str(possible_files[0])
            console.print(f"[cyan]Found source file: {source_path}[/]")
        else:
            # Ask user for source file path
            if not source_path_from_api:
                console.print(
                    "\n[yellow]Source path not found in run data (run may have been created with an older version).[/]"
                )
            console.print("[yellow]Please specify the source file to optimize.[/]")
            source_path = console.input("[bold]Enter the path to the source file to optimize: [/]").strip()
            if not pathlib.Path(source_path).exists():
                console.print(f"[bold red]Source file not found: {source_path}[/]")
                return False

    # Write last solution to source file
    if last_node.get("code"):
        write_to_path(pathlib.Path(source_path), last_node["code"])
        console.print(f"[green]✓[/] Restored last completed solution (step {last_step}) to {source_path}")
    else:
        # Fallback to original source code if no solution available
        write_to_path(pathlib.Path(source_path), source_code)
        console.print(f"[yellow]No last solution found, restored original source code to {source_path}[/]")

    # Initialize/append logging structure if save_logs is enabled
    if save_logs:
        initialize_or_append_logs(
            run_log_dir,
            run_id,
            run_name,
            "resume",
            evaluation_command,
            metric_name,
            maximize,
            total_steps,
            resumed_from_step=last_step,
        )
        console.print("[dim]Continuing with local execution logging[/]")

    # Display resume information
    console.print(f"\n[bold green]Resuming optimization from step {last_step + 1}/{total_steps}[/]")
    console.print(f"[dim]Will run steps {last_step + 1} through {total_steps}[/]")

    # Continue optimization from the next step
    console.print("\n" + "=" * 50)
    console.print("[bold cyan]Continuing Optimization[/]")
    console.print("=" * 50 + "\n")

    # Set up signal handlers and heartbeat (similar to execute_optimization)
    heartbeat_thread = None
    stop_heartbeat_event = threading.Event()

    def signal_handler(signum, frame):
        signal_name = signal.Signals(signum).name
        console.print(f"\n[yellow]Received {signal_name} signal. Cleaning up...[/]")
        if heartbeat_thread and heartbeat_thread.is_alive():
            stop_heartbeat_event.set()
            heartbeat_thread.join(timeout=2)
        report_termination(run_id, "terminated", f"user_terminated_{signal_name.lower()}", None, auth_headers)
        # Show resume message
        console.print(f"\n[bold cyan]To resume this run, use:[/] [bold green]weco resume {run_id}[/]")
        sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start heartbeat thread with a small delay to ensure DB status update propagates
    # This prevents the heartbeat from failing immediately after resume/extend
    import time

    time.sleep(2)  # Give the backend time to update status
    heartbeat_thread = HeartbeatSender(run_id, auth_headers, stop_heartbeat_event)
    heartbeat_thread.start()

    # Initialize panels for display
    source_fp = pathlib.Path(source_path)
    summary_panel, solution_panels, eval_output_panel, tree_panel = initialize_panels(
        maximize=maximize,
        metric_name=metric_name,
        total_steps=total_steps,
        model=model,
        log_dir=log_dir,
        source_fp=source_fp,
        run_id=run_id,
        run_name=run_name,
    )
    # Set the dashboard URL and run_name since we already have them
    summary_panel.set_dashboard_url(run_id)
    if run_name:
        summary_panel.set_run_name(run_name)
    # Set the initial progress to the step we're about to work on
    summary_panel.set_step(last_step + 1)

    # Load previous history if available
    run_status = get_optimization_run_status(console, run_id, include_history=True, auth_headers=auth_headers)
    if run_status and "nodes" in run_status and run_status["nodes"]:
        # Build the metric tree with all previous nodes (only if not empty)
        tree_panel.build_metric_tree(nodes=run_status["nodes"])

    # Initialize solution panels with current and best solutions
    current_node = None
    best_node = None

    # Set the current node from the resumed state
    if last_node:
        current_node = Node(
            id=last_node.get("solution_id", ""),
            parent_id=last_node.get("parent_id"),
            code=last_node.get("code"),
            metric=last_node.get("metric_value"),
            is_buggy=last_node.get("is_buggy"),
        )

    # Set the best solution if available
    if run_status and run_status.get("best_result"):
        best_result = run_status["best_result"]
        best_node = Node(
            id=best_result["solution_id"],
            parent_id=best_result.get("parent_id"),
            code=best_result.get("code"),
            metric=best_result.get("metric_value"),
            is_buggy=best_result.get("is_buggy", False),
        )

    # Update solution panels with both current and best
    solution_panels.update(current_node=current_node, best_node=best_node)

    optimization_completed_normally = False
    user_stop_requested_flag = False

    # Create the layouts for display
    layout = create_optimization_layout()
    end_optimization_layout = create_end_optimization_layout()

    # Initialize layout with panel content using helper
    prime_live_layout(
        layout=layout,
        summary_panel=summary_panel,
        tree_panel=tree_panel,
        solution_panels=solution_panels,
        eval_output_panel=eval_output_panel,
        current_step=last_step,
        is_done=False,
    )

    try:
        with Live(layout, console=console, refresh_per_second=4) as live:
            # Check if we need to evaluate the last solution first
            # The last node may or may not have execution_output already
            if last_node and last_node.get("execution_output"):
                initial_execution_output = last_node.get("execution_output")
                console.print("[dim]Using cached execution output from last solution[/]")
            else:
                # Need to evaluate the last solution since it wasn't evaluated yet
                console.print("[dim]Evaluating last solution before continuing...[/]")
                eval_output_panel.clear()
                initial_execution_output = run_and_log_evaluation(
                    eval_command=evaluation_command,
                    eval_timeout=eval_timeout,
                    save_logs=save_logs,
                    runs_dir=run_log_dir,
                    step=last_step,
                    eval_output_panel=eval_output_panel,
                )
                # Update display to show evaluation output
                smooth_update(
                    live=live,
                    layout=layout,
                    sections_to_update=[("eval_output", eval_output_panel.get_display())],
                    transition_delay=0.1,
                )

            # Continue from the next step using shared optimization loop
            optimization_completed_normally, user_stop_requested_flag = run_optimization_loop(
                live=live,
                layout=layout,
                console=console,
                run_id=run_id,
                start_step=last_step + 1,
                total_steps=total_steps,
                eval_command=evaluation_command,
                eval_timeout=eval_timeout,
                save_logs=save_logs,
                runs_dir=run_log_dir,
                source_fp=pathlib.Path(source_path),
                summary_panel=summary_panel,
                solution_panels=solution_panels,
                eval_output_panel=eval_output_panel,
                tree_panel=tree_panel,
                api_keys=api_keys,
                auth_headers=auth_headers,
                stop_heartbeat_event=stop_heartbeat_event,
                initial_execution_output=initial_execution_output,
                additional_instructions=None,
            )

            # Get the final step value for later use
            step = total_steps

            # If we completed all steps but API didn't mark as done, make explicit completion call
            if not optimization_completed_normally and step == total_steps:
                try:
                    # Mark run as completed since we finished all steps
                    report_termination(run_id, "completed", "completed_successfully", None, auth_headers)
                    console.print(f"[dim]Marked run as completed (step {step}/{total_steps})[/]")
                    optimization_completed_normally = True
                except Exception as e:
                    console.print(f"[dim yellow]Warning: Could not update run status to completed: {e}[/]")

                # Display final results
                run_status = get_optimization_run_status(console, run_id, include_history=False, auth_headers=auth_headers)
                if run_status and run_status.get("best_result"):
                    best = run_status["best_result"]
                    if best.get("code"):
                        best_node = Node(
                            id=best.get("solution_id", ""),
                            parent_id=best.get("parent_id"),
                            code=best["code"],
                            metric=best.get("metric_value"),
                            is_buggy=best.get("is_buggy", False),
                        )
                        solution_panels.update(current_node=solution_panels.current_node, best_node=best_node)

                        # Format score for the comment
                        best_score_str = (
                            format_number(best.get("metric_value"))
                            if best.get("metric_value") is not None and isinstance(best.get("metric_value"), (int, float))
                            else "N/A"
                        )
                        best_solution_content = f"# Best solution from Weco with a score of {best_score_str}\n\n{best['code']}"
                        # Save best solution to .runs/<run-id>/best.<extension>
                        write_to_path(run_log_dir / f"best{pathlib.Path(source_path).suffix}", best_solution_content)
                        # write the best solution to the source file
                        write_to_path(pathlib.Path(source_path), best_solution_content)

                        # Final display with end optimization layout
                        _, best_solution_panel = solution_panels.get_display(current_step=step)
                        final_message = (
                            f"{metric_name.capitalize()} {'maximized' if maximize else 'minimized'}! Best solution {metric_name.lower()} = [green]{best.get('metric_value')}[/] 🏆"
                            if best.get("metric_value") is not None
                            else "[red] No valid solution found.[/]"
                        )
                        end_optimization_layout["summary"].update(summary_panel.get_display(final_message=final_message))
                        end_optimization_layout["tree"].update(tree_panel.get_display(is_done=True))
                        end_optimization_layout["best_solution"].update(best_solution_panel)
                        live.update(end_optimization_layout)

                optimization_completed_normally = True

    except Exception as e:
        console.print(f"\n[bold red]Error during optimization: {e}[/]")
        traceback.print_exc()

    finally:
        # Stop heartbeat
        if heartbeat_thread and heartbeat_thread.is_alive():
            stop_heartbeat_event.set()
            heartbeat_thread.join(timeout=2)

        # Report termination status only if not completed normally
        # (API already marks as completed when is_done=True)
        if not optimization_completed_normally:
            if user_stop_requested_flag:
                status, reason, details = "terminated", "user_requested_stop", "Run stopped by user request via dashboard."
            else:
                status, reason = "error", "error_cli_internal"
                details = "Resume failed due to an error"

            report_termination(run_id, status, reason, details, auth_headers)

        # Show completion message for resume
        if optimization_completed_normally:
            console.print(
                f"\n[bold cyan]To extend this run with more steps, use:[/] [bold green]weco extend {run_id} <additional_steps>[/]"
            )

    return optimization_completed_normally or user_stop_requested_flag


def extend_optimization(
    run_id: str,
    additional_steps: int,
    skip_validation: bool = False,
    log_dir: str = ".runs",
    console: Optional[Console] = None,
) -> bool:
    """
    Extend a completed optimization run with additional steps.

    Args:
        run_id: The ID of the completed run to extend
        additional_steps: Number of additional steps to add
        skip_validation: Whether to skip environment validation checks
        log_dir: Directory to store logs and results
        console: Rich console for output

    Returns:
        bool: True if optimization completed successfully, False otherwise
    """
    if console is None:
        console = Console()

    from .api import extend_optimization_run, get_optimization_run_status

    # Read authentication and API keys
    api_keys = read_api_keys_from_env()
    api_key, auth_headers = handle_authentication(console, api_keys)

    # First, check the run status
    run_status = get_optimization_run_status(console, run_id, include_history=False, auth_headers=auth_headers)
    if not run_status:
        console.print(f"[bold red]Failed to get run status for ID: {run_id}[/]")
        console.print("[yellow]Possible reasons:[/]")
        console.print("  • The run ID may be incorrect (should be a UUID like '0002e071-1b67-411f-a514-36947f0c4b31')")
        console.print("  • The run may not exist or has been deleted")
        console.print("  • There may be a temporary server issue")
        console.print("\n[cyan]Tip: You can find run IDs in the .runs/ directory or from previous run outputs.[/]")
        console.print("[cyan]Usage: weco extend <run-id> <additional-steps>[/]")
        console.print("[cyan]Example: weco extend 0002e071-1b67-411f-a514-36947f0c4b31 50[/]")
        return False

    current_status = run_status.get("status")
    # Get the total steps from optimizer config, not completed_steps which may be 0
    original_steps = run_status.get("optimizer", {}).get("steps", 0)
    current_step = run_status.get("current_step", 0)

    # Check if run is completed - trust database status as single source of truth
    if current_status != "completed":
        console.print(f"[bold red]Run is not completed (status: {current_status}).[/]")
        if current_status in ["interrupted", "terminated", "error"]:
            console.print(
                f"[cyan]This run was interrupted after {current_step} steps. Use 'weco resume {run_id}' to continue it.[/]"
            )
        else:
            console.print(f"[cyan]Current status: {current_status}. Only completed runs can be extended.[/]")
        return False

    # Get basic info for validation (before changing DB status)
    objective = run_status.get("objective", {})
    metric_name = objective.get("metric_name", "metric")
    maximize = objective.get("maximize", True)
    evaluation_command = run_status.get("evaluation_command", "")

    # Show extension preview and get confirmation first
    console.print("\n[bold green]Extension Preview:[/]")
    console.print(f"[cyan]Run ID:[/] {run_id}")
    console.print(f"[cyan]Original Steps:[/] {original_steps}")
    console.print(f"[cyan]Additional Steps:[/] {additional_steps}")
    console.print(f"[cyan]New Total Steps:[/] {original_steps + additional_steps}")
    console.print(f"[cyan]Metric:[/] {'Maximizing' if maximize else 'Minimizing'} {metric_name}")
    console.print(f"[cyan]Evaluation Command:[/] {evaluation_command}")

    # Environment validation (unless skipped) - moved before API call
    if not skip_validation:
        console.print("\n[bold cyan]Extension Validation[/]")
        console.print("This will continue optimization from where the original run completed.")

        # Validation prompts
        console.print("\n[bold yellow]Please confirm:[/]")
        console.print("1. Your evaluation script hasn't been modified since the original run")
        console.print("2. Your test environment is the same (dependencies, data files, etc.)")
        console.print("3. The extension parameters are correct for your optimization goals")

        if console.input(f"\n[bold]Continue extending run {run_id}? \\[y]es/No (default=No): [/]").lower().strip() not in [
            "y",
            "yes",
        ]:
            console.print("[yellow]Extension cancelled by user.[/]")
            return False

    # Now call extend endpoint (this changes DB status to "running")
    console.print(
        f"[cyan]Extending completed run (originally {original_steps} steps) with {additional_steps} additional steps...[/]"
    )
    extend_info = extend_optimization_run(
        console=console, run_id=run_id, additional_steps=additional_steps, api_keys=api_keys, auth_headers=auth_headers
    )
    if not extend_info:
        console.print("[bold red]Failed to extend run. Please try again.[/]")
        console.print("[cyan]Tip: Make sure the run is fully completed and not currently being processed.[/]")
        return False

    # Extract extend information
    last_step = extend_info["previous_steps"]  # The completed steps
    total_steps = last_step + additional_steps  # Calculate new total steps
    evaluation_command = extend_info["evaluation_command"]
    # For extend, we use the last node as the starting point
    last_node = extend_info.get("last_solution")  # API returns last_solution but it's actually the last node
    run_name = extend_info.get("run_name", run_id)
    source_path_from_api = extend_info.get("source_path")  # Get source_path from API
    eval_timeout = extend_info.get("eval_timeout")  # Get eval_timeout from API
    save_logs = extend_info.get("save_logs", False)  # Get save_logs from API (inherit from original run)
    # Use the provided log_dir parameter instead of getting from API

    if save_logs:
        console.print("[dim]Local logging enabled (from original run)[/]")

    # Show detailed extend information now that we have it
    console.print("\n[bold green]Extension Details:[/]")
    console.print(f"[cyan]Run Name:[/] {run_name}")
    console.print(f"[cyan]Previous steps completed:[/] {last_step}")
    console.print(f"[cyan]Additional steps:[/] {additional_steps}")
    console.print(f"[cyan]New total steps:[/] {total_steps}")
    console.print(f"[cyan]Will continue from step:[/] {last_step + 1}")
    console.print(f"[cyan]Evaluation command:[/] {evaluation_command}")
    if last_node:
        console.print(f"[cyan]Last step metric:[/] {last_node.get('metric_value', 'N/A')}")

    # Get metric info from run_status
    objective = run_status.get("objective", {})
    metric_name = objective.get("metric_name", "metric")
    maximize = objective.get("maximize", True)

    # Get optimizer config for model info
    optimizer_config = run_status.get("optimizer", {})
    model = optimizer_config.get("code_generator", {}).get("model")
    if not model:
        # Use helper function to determine default model based on API keys
        from .utils import determine_default_model

        model = determine_default_model(api_keys)

    # Log directory was already retrieved from extend_info above
    run_log_dir = pathlib.Path(log_dir) / run_id
    run_log_dir.mkdir(parents=True, exist_ok=True)

    # Determine the source file path
    if source_path_from_api and pathlib.Path(source_path_from_api).exists():
        source_path = source_path_from_api
        console.print(f"[cyan]Using source file from original run: {source_path}[/]")
    else:
        # Try to find the source file
        possible_files = []
        for pattern in ["train.py", "main.py", "solution.py", "*.py"]:
            files = list(pathlib.Path(".").glob(pattern))
            possible_files.extend(files)

        possible_files = list(set(f for f in possible_files if f.is_file()))

        if len(possible_files) == 1:
            source_path = str(possible_files[0])
            console.print(f"[cyan]Found source file: {source_path}[/]")
        else:
            console.print("[yellow]Please specify the source file to continue optimizing.[/]")
            source_path = console.input("[bold]Enter the path to the source file: [/]").strip()
            if not pathlib.Path(source_path).exists():
                console.print(f"[bold red]Source file not found: {source_path}[/]")
                return False

    # Write the last completed solution (starting point) to the source file
    # We continue from where we left off (last completed step)
    if not last_node or not last_node.get("code"):
        # If the last node doesn't have code, we can't extend
        console.print("[bold red]Error: Last solution node doesn't have code. Cannot extend this run.[/]")
        return False

    write_to_path(pathlib.Path(source_path), last_node["code"])
    write_to_path(run_log_dir / f"step_{last_step}.py", last_node["code"])
    console.print(
        f"\n[green]Extending from last completed step {last_step} (metric: {last_node.get('metric_value', 'N/A')}):[/] {source_path}"
    )

    console.print(f"[cyan]Extending from step {last_step + 1} to {total_steps}[/]\n")

    # Initialize/append logging structure if save_logs is enabled
    if save_logs:
        initialize_or_append_logs(
            run_log_dir,
            run_id,
            run_name,
            "extend",
            evaluation_command,
            metric_name,
            maximize,
            total_steps,
            extended_from_step=last_step,
            additional_steps=additional_steps,
        )
        console.print("[dim]Continuing with local execution logging for extension[/]")

    # Set up signal handlers and heartbeat
    heartbeat_thread = None
    stop_heartbeat_event = threading.Event()

    def signal_handler(signum, frame):
        signal_name = signal.Signals(signum).name
        console.print(f"\n[yellow]Received {signal_name} signal. Cleaning up...[/]")
        if heartbeat_thread and heartbeat_thread.is_alive():
            stop_heartbeat_event.set()
            heartbeat_thread.join(timeout=2)
        report_termination(run_id, "terminated", f"user_terminated_{signal_name.lower()}", None, auth_headers)
        # Show resume message
        console.print(f"\n[bold cyan]To resume this run, use:[/] [bold green]weco resume {run_id}[/]")
        sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start heartbeat thread with a small delay to ensure DB status update propagates
    # This prevents the heartbeat from failing immediately after resume/extend
    import time

    time.sleep(2)  # Give the backend time to update status
    heartbeat_thread = HeartbeatSender(run_id, auth_headers, stop_heartbeat_event)
    heartbeat_thread.start()

    # Initialize panels for display
    source_fp = pathlib.Path(source_path)
    summary_panel, solution_panels, eval_output_panel, tree_panel = initialize_panels(
        maximize=maximize,
        metric_name=metric_name,
        total_steps=total_steps,
        model=model,
        log_dir=log_dir,
        source_fp=source_fp,
        run_id=run_id,
        run_name=run_name,
    )
    summary_panel.set_dashboard_url(run_id)
    if run_name:
        summary_panel.set_run_name(run_name)
    summary_panel.set_step(last_step + 1)  # Show the step we're about to work on

    # Load previous history
    run_status = get_optimization_run_status(console, run_id, include_history=True, auth_headers=auth_headers)
    if run_status and "nodes" in run_status and run_status["nodes"]:
        tree_panel.build_metric_tree(nodes=run_status["nodes"])

        # Add a placeholder node for the next step to be worked on
        # Since we're extending, we always have more steps to do
        next_step_id = f"next_step_{last_step + 1}"
        parent_id = last_node.get("solution_id") if last_node and last_node.get("solution_id") else None
        tree_panel.build_metric_tree(
            nodes=run_status["nodes"]
            + [
                {
                    "solution_id": next_step_id,
                    "parent_id": parent_id,
                    "step": last_step + 1,
                    "code": None,
                    "metric_value": None,
                    "is_buggy": None,
                }
            ]
        )
        # Mark the next step as unevaluated to show "evaluating..." indicator
        tree_panel.set_unevaluated_node(node_id=next_step_id)

    # Initialize with last solution
    last_node_obj = None
    if last_node:
        last_node_obj = Node(
            id=last_node.get("solution_id", ""),
            parent_id=last_node.get("parent_id"),
            code=last_node.get("code"),
            metric=last_node.get("metric_value"),
            is_buggy=last_node.get("is_buggy", False),
        )
        solution_panels.update(current_node=last_node_obj, best_node=last_node_obj)

    optimization_completed_normally = False
    user_stop_requested_flag = False

    # Create the layouts for display
    layout = create_optimization_layout()
    end_optimization_layout = create_end_optimization_layout()

    # Initialize layout with panel content using helper
    prime_live_layout(
        layout=layout,
        summary_panel=summary_panel,
        tree_panel=tree_panel,
        solution_panels=solution_panels,
        eval_output_panel=eval_output_panel,
        current_step=last_step,
        is_done=False,
    )

    try:
        with Live(layout, console=console, refresh_per_second=4) as live:
            # Check if we need to evaluate the last solution first
            # If the last_node has execution_output, use it; otherwise evaluate it
            if last_node and last_node.get("execution_output"):
                console.print("[cyan]Using cached execution output from last completed solution...[/]")
                execution_output = last_node.get("execution_output")
                eval_output_panel.update(execution_output)
            else:
                console.print("[cyan]Evaluating last completed solution...[/]")
                eval_output_panel.clear()
                execution_output = run_and_log_evaluation(
                    eval_command=evaluation_command,
                    eval_timeout=eval_timeout,
                    save_logs=save_logs,
                    runs_dir=run_log_dir,
                    step=last_step,
                    eval_output_panel=eval_output_panel,
                )

            # Continue from last_step + 1 to total_steps using shared optimization loop
            optimization_completed_normally, user_stop_requested_flag = run_optimization_loop(
                live=live,
                layout=layout,
                console=console,
                run_id=run_id,
                start_step=last_step + 1,
                total_steps=total_steps,
                eval_command=evaluation_command,
                eval_timeout=eval_timeout,
                save_logs=save_logs,
                runs_dir=run_log_dir,
                source_fp=pathlib.Path(source_path),
                summary_panel=summary_panel,
                solution_panels=solution_panels,
                eval_output_panel=eval_output_panel,
                tree_panel=tree_panel,
                api_keys=api_keys,
                auth_headers=auth_headers,
                stop_heartbeat_event=stop_heartbeat_event,
                initial_execution_output=execution_output,
                additional_instructions=None,
            )

            # Get the final step value for later use
            step = total_steps

            # Final handling
            if optimization_completed_normally or step == total_steps:
                run_status = get_optimization_run_status(console, run_id, include_history=False, auth_headers=auth_headers)
                if run_status and run_status.get("best_result"):
                    best = run_status["best_result"]
                    if best.get("code"):
                        # Format score for the comment
                        best_score_str = (
                            format_number(best.get("metric_value"))
                            if best.get("metric_value") is not None and isinstance(best.get("metric_value"), (int, float))
                            else "N/A"
                        )
                        best_solution_content = f"# Best solution from Weco with a score of {best_score_str}\n\n{best['code']}"
                        # Save best solution to .runs/<run-id>/best.<extension>
                        write_to_path(run_log_dir / f"best{pathlib.Path(source_path).suffix}", best_solution_content)
                        # write the best solution to the source file
                        write_to_path(pathlib.Path(source_path), best_solution_content)

                        # Final display with end optimization layout
                        _, best_solution_panel = solution_panels.get_display(current_step=step)
                        final_message = (
                            f"{metric_name.capitalize()} {'maximized' if maximize else 'minimized'}! Best solution {metric_name.lower()} = [green]{best.get('metric_value')}[/] 🏆"
                            if best.get("metric_value") is not None
                            else "[red] No valid solution found.[/]"
                        )
                        end_optimization_layout["summary"].update(summary_panel.get_display(final_message=final_message))
                        end_optimization_layout["tree"].update(tree_panel.get_display(is_done=True))
                        end_optimization_layout["best_solution"].update(best_solution_panel)
                        live.update(end_optimization_layout)

                optimization_completed_normally = True

    except Exception as e:
        console.print(f"\n[bold red]Error during extension: {e}[/]")
        traceback.print_exc()

    finally:
        # Stop heartbeat
        if heartbeat_thread and heartbeat_thread.is_alive():
            stop_heartbeat_event.set()
            heartbeat_thread.join(timeout=2)

        # Report termination status only if not completed normally
        if not optimization_completed_normally:
            if user_stop_requested_flag:
                status, reason, details = "terminated", "user_requested_stop", "Run stopped by user request via dashboard."
            else:
                status, reason = "error", "error_cli_internal"
                details = "Extension failed due to an error"

            report_termination(run_id, status, reason, details, auth_headers)

        # Show completion message for extend
        if optimization_completed_normally:
            console.print(
                f"\n[bold cyan]To extend this run with more steps, use:[/] [bold green]weco extend {run_id} <additional_steps>[/]"
            )

    return optimization_completed_normally or user_stop_requested_flag
