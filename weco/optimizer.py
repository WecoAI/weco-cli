import math
import os
import pathlib
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

from requests.exceptions import ConnectionError as RequestsConnectionError, HTTPError, ReadTimeout
from rich.console import Console
from rich.prompt import Confirm

from . import __dashboard_url__
from .api import (
    claim_execution_task,
    format_api_error,
    get_execution_tasks,
    get_lineage_execution_tasks,
    get_optimization_run_status,
    report_termination,
    resume_optimization_run,
    send_heartbeat,
    send_lineage_heartbeat,
    start_optimization_run,
    submit_execution_result,
)
from .core.api import WecoClient
from .artifacts import RunArtifacts
from .consumer_lock import try_acquire, release
from .auth import handle_authentication
from .events import get_event_context
from .browser import open_browser
from .ui import OptimizationUI, LiveOptimizationUI, PlainOptimizationUI
from .utils import read_additional_instructions, read_from_path, write_to_path, run_evaluation_with_files_swap


def _should_auto_open_browser(no_open: bool) -> bool:
    """Whether the CLI should pop the run's dashboard URL in a new tab.

    Suppressed when:
      * `--no-open` was passed explicitly, or
      * the CLI is running inside a `weco start <agent>` session (signalled
        by `WECO_CC_SESSION_ID` in the env). In that case the attached
        dashboard receives the new run via the session's Realtime channel
        and surfaces it with an in-page toast instead — popping a separate
        tab on top of an already-open dashboard would be noise.
    """
    if no_open:
        return False
    if os.environ.get("WECO_CC_SESSION_ID"):
        return False
    return True


@dataclass
class OptimizationResult:
    """Result from a queue-based optimization loop."""

    success: bool
    final_step: int
    status: str  # "completed", "terminated", "error"
    reason: str  # e.g. "completed_successfully", "user_terminated_sigint"
    details: Optional[str] = None


@dataclass
class AutoResumePolicy:
    """Policy for auto-resuming a run after transient errors."""

    enabled: bool = True
    max_attempts: int = 5
    backoff_initial_s: float = 5.0
    backoff_max_s: float = 60.0
    backoff_factor: float = 2.0


# Reasons produced by run_optimization_loop that indicate a retryable failure.
# 5xx bursts imply layer-2 recovery already tried and gave up; waiting and
# resuming the run is the right response. 4xx (auth, validation, insufficient
# credits) and user-driven terminations must propagate.
_TRANSIENT_REASONS = frozenset({"transient_network_error", "http_502", "http_503", "http_504"})


def _is_transient(result: OptimizationResult) -> bool:
    return result.reason in _TRANSIENT_REASONS


class _PollAction(Enum):
    """What a single lineage-queue poll tells the consumer to do next."""

    PROCESS = "process"  # ready work exists — claim and evaluate it
    WAIT = "wait"  # uncertain, or runs active but no ready task — keep polling
    DONE = "done"  # confirmed: read succeeded, no ready work, no active runs


def _classify_lineage_poll(read_ok: bool, has_ready: bool, active_run_count: Optional[int]) -> _PollAction:
    """Decide what one lineage-queue poll means. Pure — no I/O, no state.

    The single rule that prevents orphaning: only a *confirmed* read showing no
    ready work AND no active runs is ``DONE``. A failed read, or an active run
    with no ready task yet (between candidates, or freshly derived), is ``WAIT``
    — never mistaken for "the lineage is finished".

    ``active_run_count`` is authoritative only when ``has_ready`` is False (the
    backend omits it when tasks are present, since the consumer processes those
    regardless). ``None`` there means the backend didn't report it (e.g. version
    skew) — treated as ``WAIT`` rather than guessing ``DONE``.
    """
    if not read_ok:
        return _PollAction.WAIT
    if has_ready:
        return _PollAction.PROCESS
    if active_run_count is None or active_run_count > 0:
        return _PollAction.WAIT
    return _PollAction.DONE


class _SilentResumeOutcome(Enum):
    """Result of a background (no-console-output) resume attempt.

    Distinguishes the three cases the auto-resume wrapper must act on
    differently: the run was flipped back to ``running`` and has work to do
    (``RESUMED``); the backend's resume repair pass found no runnable work and
    already finalized the run ``completed`` (``ALREADY_COMPLETE``, signalled by
    ``is_done`` in the resume response); or the resume call failed and should be
    retried (``FAILED``).
    """

    RESUMED = "resumed"
    ALREADY_COMPLETE = "already_complete"
    FAILED = "failed"


def _silent_resume(run_id: str, auth_headers: dict, api_keys: Optional[dict]) -> _SilentResumeOutcome:
    """Flip a run back to 'running' without emitting any console output.

    Reads ``is_done`` from the resume response defensively (absent → ``False``),
    so an older backend that never sends the field is treated as a normal
    ``RESUMED`` and legacy behavior is preserved.
    """
    try:
        resp = WecoClient(auth_headers).resume_run(run_id, api_keys=api_keys)
    except Exception:
        return _SilentResumeOutcome.FAILED
    if isinstance(resp, dict) and resp.get("is_done", False):
        return _SilentResumeOutcome.ALREADY_COMPLETE
    return _SilentResumeOutcome.RESUMED


def _run_loop_with_auto_resume(
    loop_factory: Callable[[int], OptimizationResult],
    *,
    ui: "OptimizationUI",
    run_id: str,
    auth_headers: dict,
    api_keys: Optional[dict],
    policy: AutoResumePolicy,
    initial_start_step: int,
) -> OptimizationResult:
    """Invoke the optimization loop; on transient failure, resume and re-enter.

    ``loop_factory(start_step)`` runs one attempt of ``run_optimization_loop``.
    If the attempt exits with a transient reason and auto-resume is enabled,
    this sleeps with exponential backoff, calls the backend resume endpoint,
    and re-enters. Non-transient outcomes (completed, user interrupt, HTTP 4xx)
    are returned verbatim.
    """
    start_step = initial_start_step
    attempts_used = 0

    while True:
        result = loop_factory(start_step)

        if not policy.enabled or not _is_transient(result):
            return result

        resumed = False
        while attempts_used < policy.max_attempts:
            attempts_used += 1
            backoff = min(policy.backoff_initial_s * (policy.backoff_factor ** (attempts_used - 1)), policy.backoff_max_s)
            ui.on_reconnecting(attempts_used, policy.max_attempts, backoff)
            time.sleep(backoff)

            outcome = _silent_resume(run_id, auth_headers, api_keys)
            if outcome is _SilentResumeOutcome.ALREADY_COMPLETE:
                # The backend's resume repair pass finalized the run 'completed'
                # (no runnable work left). Re-entering the loop would only poll,
                # observe run.status == 'completed', and misreport a stop. Treat
                # it as terminal success — the same exit a normal is_done takes.
                ui.on_reconnected()
                ui.on_complete(result.final_step)
                return OptimizationResult(
                    success=True, final_step=result.final_step, status="completed", reason="completed_successfully"
                )
            if outcome is _SilentResumeOutcome.RESUMED:
                ui.on_reconnected()
                start_step = result.final_step
                resumed = True
                break

        if not resumed:
            ui.on_error(
                f"Auto-resume exhausted after {policy.max_attempts} attempt(s). "
                f"Use 'weco resume {run_id}' to continue manually."
            )
            return result


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


class LineageHeartbeatSender(threading.Thread):
    """Heartbeat every ``running`` member of a lineage on a fixed cadence.

    One call (``POST /lineages/{id}/heartbeat``) keeps all running members alive
    — including derived runs queued behind the one currently being evaluated.
    The single lineage consumer owns liveness for the whole lineage, so a member
    waiting its turn isn't reaped by the heartbeat-timeout cron.
    """

    def __init__(self, lineage_id: str, auth_headers: dict, stop_event: threading.Event, interval: int = 30):
        super().__init__(daemon=True)
        self.lineage_id = lineage_id
        self.auth_headers = auth_headers
        self.interval = interval
        self.stop_event = stop_event

    def run(self):
        try:
            while not self.stop_event.is_set():
                send_lineage_heartbeat(self.lineage_id, self.auth_headers)
                if self.stop_event.is_set():
                    break
                self.stop_event.wait(self.interval)
        except Exception as e:
            print(f"[ERROR LineageHeartbeatSender] Unexpected error for lineage {self.lineage_id}: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)


def run_optimization_loop(
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
    submit_timeout: Optional[int] = None,
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
        submit_timeout: Optional read-timeout override (seconds) for the
            ``/suggest`` call made when submitting a step's result. ``None``
            preserves the existing ~61-minute default.

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

            # Submit result. HTTP errors (insufficient credits, candidate generation
            # failures, etc.) propagate and are handled centrally below so the real
            # backend detail reaches the user and the run's termination record.
            ui.on_submitting()
            submit_timeout_tuple = (10, submit_timeout) if submit_timeout is not None else None
            result = submit_execution_result(
                run_id=run_id,
                task_id=task_id,
                execution_output=term_out,
                auth_headers=auth_headers,
                api_keys=api_keys,
                timeout=submit_timeout_tuple,
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
    except (RequestsConnectionError, ReadTimeout) as e:
        # Tagged separately so the outer auto-resume wrapper can distinguish
        # transport failures from unrecoverable errors.
        ui.on_warning(f"Network error during optimization: {e}")
        return OptimizationResult(
            success=False, final_step=step, status="error", reason="transient_network_error", details=str(e)
        )
    except HTTPError as e:
        # Surface structured API error details (insufficient credits, auth failures, candidate
        # generation failures, etc.) through the UI rather than a generic exception string.
        error_message = format_api_error(e)
        ui.on_error(error_message)
        status_code = getattr(e.response, "status_code", None)
        return OptimizationResult(
            success=False,
            final_step=step,
            status="error",
            reason=f"http_{status_code}" if status_code else "http_error",
            details=error_message,
        )
    except Exception as e:
        ui.on_error(f"Error: {e}")
        return OptimizationResult(success=False, final_step=step, status="error", reason="unknown", details=str(e))


class _TaggedPlainUI(PlainOptimizationUI):
    """Plain UI whose every line is prefixed with the run name, so a single
    lineage consumer's interleaved per-run output stays readable."""

    def _print(self, message: str) -> None:
        print(f"[{self.run_name}] {message}", flush=True)


def _build_run_state(run_id: str, auth_headers: dict, log_dir: str, dashboard_base: str) -> dict:
    """Lazily construct per-run UI + artifacts for a run discovered in the queue.

    Eval config (command/timeout/save_logs) is lineage-invariant and supplied to
    the loop once; here we only fetch the per-run *display* fields (name, model,
    step target, metric) for the run's UI header.
    """
    run_name = run_id
    metric_name = "metric"
    maximize = True
    total_steps = 0
    model = ""
    current_step = 0
    lineage_id = run_id
    try:
        status = get_optimization_run_status(console=None, run_id=run_id, include_history=False, auth_headers=auth_headers)
        run_name = status.get("run_name", run_id)
        objective = status.get("objective", {}) or {}
        metric_name = objective.get("metric_name", "metric")
        maximize = bool(objective.get("maximize", True))
        optimizer = status.get("optimizer", {}) or {}
        total_steps = optimizer.get("steps", 0) or 0
        model = (optimizer.get("code_generator") or {}).get("model") or ""
        current_step = int(status.get("current_step", 0) or 0)
        lineage_id = status.get("lineage_id") or run_id
    except Exception:
        pass  # Display-only; fall back to ids if status is unavailable.

    ui = _TaggedPlainUI(
        run_id,
        run_name,
        total_steps,
        f"{dashboard_base}/runs/{lineage_id}",
        model=model,
        metric_name=metric_name,
        maximize=maximize,
    )
    ui.on_init()
    return {"ui": ui, "artifacts": RunArtifacts(log_dir=log_dir, run_id=run_id), "step": max(current_step, 1), "done": False}


def run_lineage_loop(
    lineage_id: str,
    auth_headers: dict,
    originals: dict[str, str],
    eval_command: str,
    eval_timeout: Optional[int],
    save_logs: bool,
    log_dir: str,
    dashboard_base: str,
    *,
    api_keys: Optional[dict] = None,
    submit_timeout: Optional[int] = None,
    poll_interval: float = 2.0,
    max_idle_polls: int = 300,
) -> bool:
    """Single consumer that drains every active run in a lineage, one eval at a time.

    The collision-safe generalization of :func:`run_optimization_loop`: instead of
    one run, it polls the lineage-wide ready queue, claims one task at a time,
    evaluates it via the shared working tree (serially — never two at once), and
    submits. It keeps every running member alive with a single lineage heartbeat,
    and exits only when no member is running and the queue is empty.

    The caller MUST hold the working-tree consumer lock for the duration. Eval
    config (``eval_command``/``eval_timeout``/``save_logs``/``originals``) is
    lineage-invariant; only per-run display state is fetched lazily.

    Returns ``True`` unless an account-level failure (auth / insufficient credits)
    aborted the consumer.
    """
    states: dict[str, dict] = {}
    fatal = False
    # Consecutive non-PROCESS polls since work last flowed. Bounds the wait when
    # the backend is wedged (active runs that never produce a task) or the queue
    # read keeps failing — the only ways the loop can spin without progress.
    idle_polls = 0

    stop_event = threading.Event()
    heartbeat_thread = LineageHeartbeatSender(lineage_id, auth_headers, stop_event)
    heartbeat_thread.start()

    print(f"[lineage {lineage_id}] consumer started; draining ready evaluations across the lineage.", flush=True)

    try:
        while True:
            # One authoritative read per poll: ready tasks across the lineage,
            # plus active_run_count (how many members are still running/stopping)
            # when the queue is empty. _classify_lineage_poll turns it into an
            # action — never declaring "done" on an unconfirmed read.
            tasks_result = get_lineage_execution_tasks(lineage_id, auth_headers)
            read_ok = tasks_result is not None  # None == read failed, NOT "empty"
            ready = (tasks_result.tasks if tasks_result else []) or []
            active_run_count = tasks_result.active_run_count if tasks_result else None

            action = _classify_lineage_poll(read_ok, bool(ready), active_run_count)

            if action is _PollAction.DONE:
                break
            if action is _PollAction.WAIT:
                idle_polls += 1
                if idle_polls >= max_idle_polls:
                    print(f"[lineage {lineage_id}] no progress after {max_idle_polls} polls; stopping consumer.", flush=True)
                    break
                time.sleep(poll_interval)
                continue
            idle_polls = 0

            task = ready[0]
            run_id = task["run_id"]
            task_run = task.get("run") or {}
            if task_run.get("status") not in (None, "running"):
                # Stale task for a run that's winding down; let the backend settle it.
                time.sleep(poll_interval)
                continue

            st = states.get(run_id) or states.setdefault(
                run_id, _build_run_state(run_id, auth_headers, log_dir, dashboard_base)
            )
            ui = st["ui"]
            artifacts = st["artifacts"]
            step = st["step"]

            claimed = claim_execution_task(task["id"], auth_headers)
            if claimed is None:
                # No competing consumer should exist (we hold the lock); the task
                # was likely cancelled as its run wound down. Skip it.
                continue

            code = claimed["revision"]["code"]
            plan = claimed["revision"].get("plan")
            ui.on_executing(step)
            ui.on_task_claimed(task["id"], plan)

            artifacts.save_step_code(step, code)
            term_out = run_evaluation_with_files_swap(
                file_map=code, originals=originals, eval_command=eval_command, timeout=eval_timeout
            )
            if save_logs:
                artifacts.save_execution_output(step=step, output=term_out)
            ui.on_output(term_out)

            ui.on_submitting()
            submit_timeout_tuple = (10, submit_timeout) if submit_timeout is not None else None
            try:
                result = submit_execution_result(
                    run_id=run_id,
                    task_id=task["id"],
                    execution_output=term_out,
                    auth_headers=auth_headers,
                    api_keys=api_keys,
                    timeout=submit_timeout_tuple,
                )
            except HTTPError as e:
                msg = format_api_error(e)
                ui.on_error(msg)
                status_code = getattr(e.response, "status_code", None)
                if status_code in (401, 402):
                    # Account-level (auth / insufficient credits): affects every
                    # member — stop the whole consumer.
                    fatal = True
                    break
                # Per-run failure (e.g. 409: the run was finalized under us while
                # winding down). Retire this run and keep draining the others.
                st["done"] = True
                st["step"] = step + 1
                continue
            except (RequestsConnectionError, ReadTimeout) as e:
                ui.on_warning(f"Network error during evaluation submit: {e}")
                time.sleep(poll_interval)
                continue

            is_done = result.get("is_done", False)
            prev_metric = result.get("previous_solution_metric_value")
            if prev_metric is not None:
                ui.on_metric(step, prev_metric)
            st["step"] = step + 1
            if is_done:
                ui.on_complete(st["step"])
                st["done"] = True

    except KeyboardInterrupt:
        for st in states.values():
            st["ui"].on_interrupted()
    finally:
        stop_event.set()
        heartbeat_thread.join(timeout=2)

    print(f"[lineage {lineage_id}] consumer finished.", flush=True)
    return not fatal


def _drain_lineage_remainder(
    lineage_id: str,
    auth_headers: dict,
    originals: dict[str, str],
    eval_command: str,
    eval_timeout: Optional[int],
    save_logs: bool,
    log_dir: str,
    *,
    api_keys: Optional[dict] = None,
    submit_timeout: Optional[int] = None,
) -> None:
    """After a single-run loop ends, drain any still-active siblings in its lineage.

    Covers the case where runs were derived from this one while it was running:
    the seed run's loop exits (completed, or stopped by the derive), and this
    same process — still holding the working-tree consumer lock — keeps
    evaluating the rest of the lineage. No-op when the lineage is quiet.
    """
    # One probe of the lineage queue: skip the drain only when we're certain the
    # lineage is quiescent (read succeeded, no ready work, no active runs).
    probe = get_lineage_execution_tasks(lineage_id, auth_headers)
    has_ready = bool(probe.tasks) if probe else False
    active_run_count = probe.active_run_count if probe else None
    if _classify_lineage_poll(probe is not None, has_ready, active_run_count) is _PollAction.DONE:
        return
    run_lineage_loop(
        lineage_id=lineage_id,
        auth_headers=auth_headers,
        originals=originals,
        eval_command=eval_command,
        eval_timeout=eval_timeout,
        save_logs=save_logs,
        log_dir=log_dir,
        dashboard_base=__dashboard_url__,
        api_keys=api_keys,
        submit_timeout=submit_timeout,
    )


def offer_apply_best_solution(
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


def _daemonize_to_log(run_id: str) -> Optional[pathlib.Path]:
    """Fork + setsid + redirect FDs so the eval loop survives our agent's process tree.

    POSIX-only. On platforms without ``os.fork`` (Windows), returns ``None`` and the
    caller falls through to foreground execution. On success, the parent process
    exits via ``os._exit(0)`` (skipping atexit handlers so it doesn't disturb global
    state) — only the child returns from this function. The child's stdin is closed,
    stdout/stderr are redirected to ``/tmp/weco-run-<run-id>.log``.

    Call AFTER the create-run/auth POSTs (so the parent has a run_id to print) and
    BEFORE starting any threads or async loops (which can't be safely forked).
    """
    if not hasattr(os, "fork"):
        return None

    log_path = pathlib.Path(f"/tmp/weco-run-{run_id}.log")

    # Flush before fork so buffered output isn't duplicated by both processes.
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass

    try:
        pid = os.fork()
    except OSError:
        return None

    if pid > 0:
        # Parent: exit immediately, skipping Python finalizers. The child carries on.
        os._exit(0)

    # --- Child only past this point ---

    # Detach from controlling terminal: become session + process-group leader.
    try:
        os.setsid()
    except OSError:
        pass

    # Replace stdin with /dev/null so any blocking reads return EOF.
    try:
        devnull_fd = os.open(os.devnull, os.O_RDONLY)
        os.dup2(devnull_fd, sys.stdin.fileno())
        os.close(devnull_fd)
    except OSError:
        pass

    # Redirect stdout/stderr to the log file. Anyone tailing it sees the full eval log.
    try:
        log_fd = os.open(str(log_path), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        os.dup2(log_fd, sys.stdout.fileno())
        os.dup2(log_fd, sys.stderr.fileno())
        os.close(log_fd)
    except OSError:
        pass

    return log_path


def resume_optimization(
    run_id: str,
    api_keys: Optional[dict] = None,
    poll_interval: float = 2.0,
    apply_change: bool = False,
    output_mode: str = "rich",
    submit_timeout: Optional[int] = None,
    auto_resume_policy: Optional[AutoResumePolicy] = None,
    additional_steps: Optional[int] = None,
    daemon: bool = False,
    no_open: bool = False,
) -> bool:
    """
    Resume an interrupted or completed run using the queue-based optimization loop.

    Polls for execution tasks, executes locally, and submits results.
    Uses the execution queue flow instead of the legacy direct flow.

    Args:
        run_id: The UUID of the run to resume.
        api_keys: Optional API keys for LLM providers.
        poll_interval: Seconds between polling attempts.
        apply_change: If True, automatically apply best solution; if False, prompt user.
        output_mode: "rich" for interactive terminal UI, "plain" for machine-readable output.
        additional_steps: If set, run this many more evaluations from the last node.
            Required when resuming a completed run; optional for terminated/error.

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
    if run_status_val == "completed":
        if additional_steps is None:
            console.print(f"[yellow]Run {run_id} is already complete. Pass --steps N to resume with N more evaluations.[/]")
            return False
    elif run_status_val not in ("error", "terminated"):
        console.print(
            f"[yellow]Run {run_id} cannot be resumed (status: {run_status_val}). "
            f"Only 'error', 'terminated', or 'completed' runs can be resumed.[/]"
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
    if additional_steps is not None:
        new_total = current_step + additional_steps
        plural = "s" if additional_steps != 1 else ""
        console.print(
            f"  Total Steps: {total_steps} -> [bold]{new_total}[/] | Current Step: {current_step} | "
            f"Will run: [bold]{additional_steps}[/] more evaluation{plural} (--steps)"
        )
        if new_total < total_steps:
            console.print(f"  [yellow]Note: this shrinks the original budget ({total_steps}) to {new_total}.[/]")
    else:
        console.print(f"  Total Steps: {total_steps} | Current Step: {current_step} | Steps Remaining: {steps_remaining}")
    console.print(f"  Last Updated: {status.get('updated_at', 'N/A')}")

    unchanged = Confirm.ask(
        "Have you kept the source files and evaluation command unchanged since the original run?", default=True
    )
    if not unchanged:
        console.print("[yellow]Resume cancelled. Please start a new run if the environment changed.[/]")
        return False

    # Call backend to prepare resume (this sets status to 'running' and, when
    # additional_steps is provided, resets run.steps to last_step + additional_steps)
    resume_resp = resume_optimization_run(
        console=console, run_id=run_id, auth_headers=auth_headers, api_keys=api_keys, steps=additional_steps
    )
    if resume_resp is None:
        return False

    # Refresh total_steps from the resume response — when the budget was
    # extended, the response carries the new total.
    response_steps = resume_resp.get("steps")
    if response_steps is not None:
        total_steps = int(response_steps)

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

    # The backend's resume repair pass may have finalized the run itself: a
    # scored-but-interrupted node was promoted, the step budget was met, and the
    # run was flipped to 'completed' with no runnable work. It signals this with
    # is_done=True (read defensively: an older backend omits the field, yielding
    # False and the unchanged legacy behavior below). Entering the task-polling
    # loop here would find no tasks / a completed run and misreport it as
    # stopped, so short-circuit to a success exit that mirrors normal completion.
    if bool(resume_resp.get("is_done", False)):
        if daemon:
            # Emit the identifiers stdout-watchers (claude, cursor, the wrapper's
            # find_run_ids) rely on before we return — no fork needed, there is
            # no long-running work to detach.
            print(f"Run ID: {run_id}", flush=True)
            print(f"Run name: {run_name}", flush=True)
            print(f"Dashboard: {dashboard_url}", flush=True)
        if output_mode == "plain":
            print("")
            print("=" * 60)
            print("[COMPLETE] Run already complete")
            print("=" * 60, flush=True)
        else:
            console.print("\n[bold green]Run already complete![/]")
        # Offer/apply the best solution exactly as the normal completion path
        # does, so a resume that finds the run already done still lands the
        # winning code in the working tree (or reports where it was saved).
        #
        # offer_apply_best_solution writes candidate files into the working tree,
        # so it MUST hold the consumer lock — same as the normal completion path,
        # which applies under lock_handle and only releases afterwards. Without it,
        # a consumer already draining this lineage in the same tree could be
        # mid-eval (swapping files in/out) when we overwrite its files. If that
        # consumer holds the lock, skip the apply with the same message the normal
        # path gives on lock-unavailable; the run is still complete, so report
        # success either way.
        lock_handle = try_acquire()
        if lock_handle is None:
            console.print(
                "[yellow]Another Weco optimization is already evaluating in this directory; "
                "not starting a second consumer here.[/]"
            )
            return True
        try:
            offer_apply_best_solution(
                console=console,
                run_id=run_id,
                source_code=source_code,
                artifacts=RunArtifacts(log_dir=log_dir, run_id=run_id),
                auth_headers=auth_headers,
                apply_change=apply_change,
            )
        finally:
            release(lock_handle)
        return True

    if daemon:
        # Print everything stdout-watchers (claude, cursor, the wrapper's find_run_ids)
        # care about BEFORE forking — once we're detached, stdout is the log file.
        print(f"Run ID: {run_id}", flush=True)
        print(f"Run name: {run_name}", flush=True)
        print(f"Dashboard: {dashboard_url}", flush=True)
        log_path = _daemonize_to_log(run_id)
        if log_path is None:
            console.print("[yellow]--daemon not supported on this platform; running in foreground.[/]")
        else:
            # Force plain UI in the daemonized child — Rich's ANSI escapes
            # would pollute the log file and serve no purpose without a TTY.
            output_mode = "plain"
            console = Console(force_terminal=False)
            print(f"weco resume daemon started; log: {log_path}", flush=True)
    elif _should_auto_open_browser(no_open):
        open_browser(dashboard_url)

    # Setup artifacts manager
    artifacts = RunArtifacts(log_dir=log_dir, run_id=run_id)

    # Acquire the working-tree consumer lock so no derive starts a competing
    # evaluation loop in this directory while we run. Released on the way out.
    lock_handle = try_acquire()
    if lock_handle is None:
        console.print(
            "[yellow]Another Weco optimization is already evaluating in this directory; "
            "not starting a second consumer here.[/]"
        )
        return False

    lineage_id = status.get("lineage_id") or run_id

    # Start heartbeat thread. Heartbeat the whole lineage (not just this run) so
    # that a run derived from this one mid-flight stays alive until our loop
    # exits and we drain it — for a lone run, lineage_id == run_id, so this is
    # identical to a single-run heartbeat.
    stop_heartbeat_event = threading.Event()
    heartbeat_thread = LineageHeartbeatSender(lineage_id, auth_headers, stop_heartbeat_event)
    heartbeat_thread.start()

    # Extract best solution info from resume response (if available)
    best_metric_value = resume_resp.get("best_metric_value")
    best_step = resume_resp.get("best_step")

    result: Optional[OptimizationResult] = None
    try:
        # Select UI implementation based on output mode
        if output_mode == "plain":
            ui_instance = PlainOptimizationUI(
                run_id, run_name, total_steps, dashboard_url, model=model_name, metric_name=metric_name, maximize=maximize
            )
        else:
            ui_instance = LiveOptimizationUI(
                console,
                run_id,
                run_name,
                total_steps,
                dashboard_url,
                model=model_name,
                metric_name=metric_name,
                maximize=maximize,
            )

        with ui_instance as ui:
            ui.on_init()
            # Populate UI with best solution from previous run if available
            if best_metric_value is not None and best_step is not None:
                ui.on_metric(best_step, best_metric_value)

            def _loop(start_step: int) -> OptimizationResult:
                return run_optimization_loop(
                    ui=ui,
                    run_id=run_id,
                    auth_headers=auth_headers,
                    source_code=source_code,
                    eval_command=eval_command,
                    eval_timeout=eval_timeout,
                    artifacts=artifacts,
                    save_logs=save_logs,
                    start_step=start_step,
                    poll_interval=poll_interval,
                    api_keys=api_keys,
                    submit_timeout=submit_timeout,
                )

            result = _run_loop_with_auto_resume(
                _loop,
                ui=ui,
                run_id=run_id,
                auth_headers=auth_headers,
                api_keys=api_keys,
                policy=auto_resume_policy or AutoResumePolicy(),
                initial_start_step=current_step,
            )

        # Stop heartbeat immediately after loop completes
        stop_heartbeat_event.set()
        heartbeat_thread.join(timeout=2)

        # Drain any runs derived from this one while it was running (still under
        # the consumer lock). No-op when the lineage is quiet.
        _drain_lineage_remainder(
            lineage_id,
            auth_headers,
            source_code,
            eval_command,
            eval_timeout,
            save_logs,
            log_dir,
            api_keys=api_keys,
            submit_timeout=submit_timeout,
        )

        # Show resume message if interrupted
        if result.status == "terminated":
            if output_mode == "plain":
                print(f"\nTo resume this run, use: weco resume {run_id}\n", flush=True)
            else:
                console.print(f"\n[cyan]To resume this run, use:[/] [bold]weco resume {run_id}[/]\n")

        # Offer to apply best solution
        offer_apply_best_solution(
            console=console,
            run_id=run_id,
            source_code=source_code,
            artifacts=artifacts,
            auth_headers=auth_headers,
            apply_change=apply_change,
        )

        release(lock_handle)
        return result.success
    finally:
        # Ensure heartbeat is stopped (in case of early exit/exception)
        stop_heartbeat_event.set()
        heartbeat_thread.join(timeout=2)

        # Release the working-tree consumer lock (idempotent if already released
        # on the success path; OS would also release it on process exit).
        release(lock_handle)

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
    submit_timeout: Optional[int] = None,
    auto_resume_policy: Optional[AutoResumePolicy] = None,
    daemon: bool = False,
    no_open: bool = False,
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
    # Opt into experimental backend behaviors.
    beta = os.environ.get("WECO_BETA", "").lower() in ("1", "true", "yes")
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
        beta=beta,
    )

    if run_response is None:
        return False

    run_id = run_response["run_id"]
    run_name = run_response["run_name"]
    dashboard_url = f"{__dashboard_url__}/runs/{run_id}"

    if daemon:
        # Print everything stdout-watchers (claude, cursor, the wrapper's find_run_ids)
        # care about BEFORE forking — once we're detached, stdout is the log file.
        print(f"Run ID: {run_id}", flush=True)
        print(f"Run name: {run_name}", flush=True)
        print(f"Dashboard: {dashboard_url}", flush=True)
        log_path = _daemonize_to_log(run_id)
        if log_path is None:
            console.print("[yellow]--daemon not supported on this platform; running in foreground.[/]")
        else:
            # Force plain UI in the daemonized child — Rich's ANSI escapes
            # would pollute the log file and serve no purpose without a TTY.
            output_mode = "plain"
            console = Console(force_terminal=False)
            print(f"weco run daemon started; log: {log_path}", flush=True)
    elif _should_auto_open_browser(no_open):
        open_browser(dashboard_url)

    # Setup artifacts manager
    artifacts = RunArtifacts(log_dir=log_dir, run_id=run_id)

    # Acquire the working-tree consumer lock so no derive starts a competing
    # evaluation loop in this directory while we run. Released on the way out.
    lock_handle = try_acquire()
    if lock_handle is None:
        console.print(
            "[yellow]Another Weco optimization is already evaluating in this directory; "
            "not starting a second consumer here.[/]"
        )
        return False

    # A fresh run is its own lineage (lineage_id == run_id); runs derived from it
    # while it runs share this id and are drained by this same process on exit.
    lineage_id = run_id

    # Start heartbeat thread. Heartbeat the whole lineage (not just this run) so
    # that a run derived from this one mid-flight stays alive until our loop
    # exits and we drain it — for a lone run, lineage_id == run_id, so this is
    # identical to a single-run heartbeat.
    stop_heartbeat_event = threading.Event()
    heartbeat_thread = LineageHeartbeatSender(lineage_id, auth_headers, stop_heartbeat_event)
    heartbeat_thread.start()

    result: Optional[OptimizationResult] = None
    try:
        # Select UI implementation based on output mode
        if output_mode == "plain":
            ui_instance = PlainOptimizationUI(
                run_id, run_name, steps, dashboard_url, model=model, metric_name=metric, maximize=maximize
            )
        else:
            ui_instance = LiveOptimizationUI(
                console, run_id, run_name, steps, dashboard_url, model=model, metric_name=metric, maximize=maximize
            )

        with ui_instance as ui:
            ui.on_init()

            def _loop(start_step: int) -> OptimizationResult:
                return run_optimization_loop(
                    ui=ui,
                    run_id=run_id,
                    auth_headers=auth_headers,
                    source_code=source_code,
                    eval_command=eval_command,
                    eval_timeout=eval_timeout,
                    artifacts=artifacts,
                    save_logs=save_logs,
                    start_step=start_step,
                    poll_interval=poll_interval,
                    api_keys=api_keys,
                    submit_timeout=submit_timeout,
                )

            result = _run_loop_with_auto_resume(
                _loop,
                ui=ui,
                run_id=run_id,
                auth_headers=auth_headers,
                api_keys=api_keys,
                policy=auto_resume_policy or AutoResumePolicy(),
                initial_start_step=0,
            )

        # Stop heartbeat immediately after loop completes
        stop_heartbeat_event.set()
        heartbeat_thread.join(timeout=2)

        # Drain any runs derived from this one while it was running (still under
        # the consumer lock). No-op when the lineage is quiet.
        _drain_lineage_remainder(
            lineage_id,
            auth_headers,
            source_code,
            eval_command,
            eval_timeout,
            save_logs,
            log_dir,
            api_keys=api_keys,
            submit_timeout=submit_timeout,
        )

        # Show resume message if interrupted
        if result.status == "terminated":
            if output_mode == "plain":
                print(f"\nTo resume this run, use: weco resume {run_id}\n", flush=True)
            else:
                console.print(f"\n[cyan]To resume this run, use:[/] [bold]weco resume {run_id}[/]\n")

        # Offer to apply best solution
        offer_apply_best_solution(
            console=console,
            run_id=run_id,
            source_code=source_code,
            artifacts=artifacts,
            auth_headers=auth_headers,
            apply_change=apply_change,
        )

        release(lock_handle)
        return result.success
    finally:
        # Ensure heartbeat is stopped (in case of early exit/exception)
        stop_heartbeat_event.set()
        heartbeat_thread.join(timeout=2)

        # Release the working-tree consumer lock (idempotent if already released
        # on the success path; OS would also release it on process exit).
        release(lock_handle)

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
