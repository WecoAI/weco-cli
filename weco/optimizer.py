import pathlib
import math
import requests
import threading
import signal
import sys
import traceback
from typing import Optional
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
    PlanPanel,
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
            print(f"[ERROR HeartbeatSender] Unhandled exception in run loop for run {self.run_id}: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            # The loop will break due to the exception, and thread will terminate via finally.


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
        timeout = 800
        processed_additional_instructions = read_additional_instructions(additional_instructions=additional_instructions)
        source_fp = pathlib.Path(source)
        source_code = read_from_path(fp=source_fp, is_json=False)

        # --- Panel Initialization ---
        summary_panel = SummaryPanel(maximize=maximize, metric_name=metric, total_steps=steps, model=model, runs_dir=log_dir)
        plan_panel = PlanPanel()
        solution_panels = SolutionPanels(metric_name=metric, source_fp=source_fp)
        eval_output_panel = EvaluationOutputPanel()
        tree_panel = MetricTreePanel(maximize=maximize)
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
            auth_headers=auth_headers,
            timeout=timeout,
        )
        run_id = run_response["run_id"]
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
            # Write the initial code string to the logs
            write_to_path(fp=runs_dir / f"step_0{source_fp.suffix}", content=run_response["code"])
            # Write the initial code string to the source file path
            write_to_path(fp=source_fp, content=run_response["code"])

            # Update the panels with the initial solution
            summary_panel.set_run_id(run_id=run_id)  # Add run id now that we have it
            # Set the step of the progress bar
            summary_panel.set_step(step=0)
            # Update the token counts
            summary_panel.update_token_counts(usage=run_response["usage"])
            plan_panel.update(plan=run_response["plan"])
            # Build the metric tree
            tree_panel.build_metric_tree(
                nodes=[
                    {
                        "solution_id": run_response["solution_id"],
                        "parent_id": None,
                        "code": run_response["code"],
                        "step": 0,
                        "metric_value": None,
                        "is_buggy": False,
                    }
                ]
            )
            # Set the current solution as unevaluated since we haven't run the evaluation function and fed it back to the model yet
            tree_panel.set_unevaluated_node(node_id=run_response["solution_id"])
            # Update the solution panels with the initial solution and get the panel displays
            solution_panels.update(
                current_node=Node(
                    id=run_response["solution_id"], parent_id=None, code=run_response["code"], metric=None, is_buggy=False
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
                    ("plan", plan_panel.get_display()),
                    ("tree", tree_panel.get_display(is_done=False)),
                    ("current_solution", current_solution_panel),
                    ("best_solution", best_solution_panel),
                    ("eval_output", eval_output_panel.get_display()),
                ],
                transition_delay=0.1,
            )

            # Run evaluation on the initial solution
            term_out = run_evaluation(eval_command=eval_command)
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
                            run_id=run_id, include_history=False, timeout=30, auth_headers=auth_headers
                        )
                        current_run_status_val = current_status_response.get("status")
                        if current_run_status_val == "stopping":
                            console.print("\n[bold yellow]Stop request received. Terminating run gracefully...[/]")
                            user_stop_requested_flag = True
                            break
                    except requests.exceptions.RequestException as e:
                        console.print(f"\n[bold red]Warning: Could not check run status: {e}. Continuing optimization...[/]")
                    except Exception as e:
                        console.print(f"\n[bold red]Warning: Error checking run status: {e}. Continuing optimization...[/]")

                # Send feedback and get next suggestion
                eval_and_next_solution_response = evaluate_feedback_then_suggest_next_solution(
                    run_id=run_id,
                    execution_output=term_out,
                    additional_instructions=current_additional_instructions,
                    api_keys=llm_api_keys,
                    auth_headers=auth_headers,
                    timeout=timeout,
                )
                # Save next solution (.runs/<run-id>/step_<step>.<extension>)
                write_to_path(fp=runs_dir / f"step_{step}{source_fp.suffix}", content=eval_and_next_solution_response["code"])
                # Write the next solution to the source file
                write_to_path(fp=source_fp, content=eval_and_next_solution_response["code"])
                status_response = get_optimization_run_status(
                    run_id=run_id, include_history=True, timeout=timeout, auth_headers=auth_headers
                )
                # Update the step of the progress bar, token counts, plan and metric tree
                summary_panel.set_step(step=step)
                summary_panel.update_token_counts(usage=eval_and_next_solution_response["usage"])
                plan_panel.update(plan=eval_and_next_solution_response["plan"])

                nodes_list_from_status = status_response.get("nodes")
                tree_panel.build_metric_tree(nodes=nodes_list_from_status if nodes_list_from_status is not None else [])
                tree_panel.set_unevaluated_node(node_id=eval_and_next_solution_response["solution_id"])

                # Update the solution panels with the next solution and best solution (and score)
                # Figure out if we have a best solution so far
                if status_response["best_result"] is not None:
                    best_solution_node = Node(
                        id=status_response["best_result"]["solution_id"],
                        parent_id=status_response["best_result"]["parent_id"],
                        code=status_response["best_result"]["code"],
                        metric=status_response["best_result"]["metric_value"],
                        is_buggy=status_response["best_result"]["is_buggy"],
                    )
                else:
                    best_solution_node = None

                current_solution_node = None
                if status_response.get("nodes"):
                    for node_data in status_response["nodes"]:
                        if node_data["solution_id"] == eval_and_next_solution_response["solution_id"]:
                            current_solution_node = Node(
                                id=node_data["solution_id"],
                                parent_id=node_data["parent_id"],
                                code=node_data["code"],
                                metric=node_data["metric_value"],
                                is_buggy=node_data["is_buggy"],
                            )
                if current_solution_node is None:
                    raise ValueError("Current solution node not found in nodes list from status response")

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
                        ("plan", plan_panel.get_display()),
                        ("tree", tree_panel.get_display(is_done=False)),
                        ("current_solution", current_solution_panel),
                        ("best_solution", best_solution_panel),
                        ("eval_output", eval_output_panel.get_display()),
                    ],
                    transition_delay=0.08,  # Slightly longer delay for more noticeable transitions
                )
                term_out = run_evaluation(eval_command=eval_command)
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
                    run_id=run_id,
                    execution_output=term_out,
                    additional_instructions=current_additional_instructions,
                    api_keys=llm_api_keys,
                    timeout=timeout,
                    auth_headers=auth_headers,
                )
                summary_panel.set_step(step=steps)
                summary_panel.update_token_counts(usage=eval_and_next_solution_response["usage"])
                status_response = get_optimization_run_status(
                    run_id=run_id, include_history=True, timeout=timeout, auth_headers=auth_headers
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
                if status_response["best_result"] is not None:
                    best_solution_node = Node(
                        id=status_response["best_result"]["solution_id"],
                        parent_id=status_response["best_result"]["parent_id"],
                        code=status_response["best_result"]["code"],
                        metric=status_response["best_result"]["metric_value"],
                        is_buggy=status_response["best_result"]["is_buggy"],
                    )
                else:
                    best_solution_node = None
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

        # Handle exit
        if user_stop_requested_flag:
            console.print("[yellow]Run terminated by user request.[/]")

    return optimization_completed_normally or user_stop_requested_flag
