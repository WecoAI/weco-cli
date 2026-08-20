"""K-parallel lineage scheduler (M3a).

One coordinator owns the lineage heartbeat, the active-run view, and the local
pool of isolated evaluation slots. It is the concurrent generalization of
``run_lineage_loop``:

* evaluations run in up to K isolated slot copies simultaneously — never in
  the user's working tree (the caller holds the working-tree consumer lock and
  applies the best solution there only at the very end);
* results are submitted with their exact ``task_id``/``run_id`` and
  ``skip_generation`` set, so scoring never silently re-reserves capacity for
  the submitting run;
* ALL generation flows through ``POST /runs/{id}/generate`` in fair
  round-robin order over active lineage members, prioritizing members with
  zero queued / in-flight / locally-evaluating work (this is what un-starves a
  deferred derived child). The database's strict lineage-K admission is
  authoritative; ``at_capacity`` is normal backpressure, never an error.

The scheduler exits only after an authoritative queue read confirms: no ready
tasks, no local evaluation futures, no outstanding generation calls, and no
active lineage members.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from typing import Optional

from requests.exceptions import ConnectionError as RequestsConnectionError, HTTPError, ReadTimeout

from .api import claim_execution_task, format_api_error, get_lineage_execution_tasks, submit_execution_result
from .artifacts import RunArtifacts
from .core.api import WecoClient
from .slots import Slot, SlotPathError, prepare_write_target, resolve_candidate_path
from .utils import EvaluationCancelled, run_evaluation, write_to_path


# ---------------------------------------------------------------------------
# Outcomes and jobs
# ---------------------------------------------------------------------------


@dataclass
class _EvalJob:
    """One claimed task assigned to one slot."""

    task_id: str
    run_id: str
    node_id: str
    slot: Slot
    file_map: dict[str, str]
    plan: Optional[str] = None
    # Set by the coordinator's cancellation poller when the job's run leaves
    # 'running' (hard stop / derive-away): the worker kills the eval's process
    # group promptly instead of computing a result nobody will accept.
    cancel_event: threading.Event = field(default_factory=threading.Event)


@dataclass
class _EvalOutcome:
    """What happened to one evaluate-and-submit job (never raises)."""

    kind: str  # "scored" | "run_done" | "conflict" | "account_failure" | "run_failure" | "network" | "cancelled"
    metric: Optional[float] = None
    detail: Optional[str] = None


@dataclass
class _RunView:
    """Cached per-run display/bookkeeping state."""

    ui: object
    artifacts: RunArtifacts
    done: bool = False
    # Locally-observed scored-result count for display (the server-allocated
    # step is per-node; this is just this consumer's progress line numbering).
    scored: int = 0
    # Runs the backend told us to stop generating for (out_of_budget/not_running).
    generation_retired: bool = False
    # Set when /generate returned no_dispatchable_work; cleared when one of the
    # run's results is scored (new information for the search policy).
    awaiting_score: bool = False
    # Incremented each time one of this run's eval futures is reaped. /generate
    # replies dispatched at an older epoch are stale for latching purposes.
    settle_epoch: int = 0


@dataclass
class _GenCall:
    """One outstanding /generate call, stamped with the epochs at dispatch.

    A reply reflects server state at compute time. If a local result settled
    after dispatch, a negative reply (at_capacity / no_dispatchable_work) may
    predate that settle — latching it would silence generation with no future
    event left to clear the latch. Stale replies are discarded and retried
    instead; discarding a fresh reply merely costs one extra round trip.
    """

    run_id: str
    lineage_epoch: int
    run_epoch: int


@dataclass
class _SchedulerState:
    free_slots: list[Slot] = field(default_factory=list)
    eval_futures: dict[Future, _EvalJob] = field(default_factory=dict)
    gen_futures: dict[Future, _GenCall] = field(default_factory=dict)
    runs: dict[str, _RunView] = field(default_factory=dict)
    # Lineage-wide backpressure: the last /generate said at_capacity and no
    # result has settled since. Cleared whenever an eval future completes.
    saturated: bool = False
    # Wallet below the dispatch floor. Credits are user-scoped, not per-run, so
    # one out_of_credits reply retires generation for the WHOLE lineage; never
    # cleared (a topup mid-drain is rare enough that `weco resume` is the
    # recovery path). In-flight evals keep draining; the server terminates each
    # drained member, so the quiescence exit ends the session.
    out_of_credits: bool = False
    fatal: bool = False
    rr_cursor: int = 0
    gen_rr_cursor: int = 0
    # Incremented each time any eval future is reaped (lineage capacity may
    # have changed). Pairs with _GenCall.lineage_epoch for stale-reply discard.
    settle_epoch: int = 0


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


# Serializes appends to the run-level exec_output.jsonl index across slots.
_artifact_lock = threading.Lock()


def _evaluate_and_submit(
    job: _EvalJob,
    *,
    auth_headers: dict,
    originals: dict[str, str],
    eval_command: str,
    eval_timeout: Optional[int],
    save_logs: bool,
    view: _RunView,
    api_keys: Optional[dict],
    submit_timeout: Optional[int],
) -> _EvalOutcome:
    """Runs in a slot's worker thread: write candidate → eval → restore → submit.

    Candidate files are written ONLY inside the assigned slot (path-checked);
    the slot's pristine baseline is restored afterwards so the next task starts
    clean. A path that would escape the slot is never evaluated — the refusal
    is submitted as the execution output so the backend can score it buggy and
    the queue keeps moving instead of stranding a claimed task.
    """
    slot = job.slot
    tag = f"[slot {slot.index}]"

    try:
        try:
            targets = {rel: resolve_candidate_path(slot.cwd, rel) for rel in {**originals, **job.file_map}}
        except (SlotPathError, OSError) as e:
            term_out = f"Weco CLI refused to evaluate this candidate: {e}"
            print(f"{tag} {term_out}", flush=True)
        else:
            with _artifact_lock:
                view.artifacts.save_node_code(job.node_id, job.file_map)
            for rel_path, content in job.file_map.items():
                fp = targets[rel_path]
                fp.parent.mkdir(parents=True, exist_ok=True)
                write_to_path(fp=fp, content=content)
            try:
                term_out = run_evaluation(
                    eval_command=eval_command,
                    timeout=eval_timeout,
                    cwd=slot.cwd,
                    env=slot.full_env(),
                    cancel_event=job.cancel_event,
                )
            except EvaluationCancelled:
                return _EvalOutcome(kind="cancelled")
            finally:
                # The eval ran arbitrary code in the slot: re-validate every
                # restore path at write time (a leaf or ancestor may have been
                # swapped for a symlink escaping the slot). A path that fails
                # re-validation is skipped — the slot is disposable, and the
                # candidate writer re-validates again before the next task.
                for rel_path, content in originals.items():
                    try:
                        write_to_path(fp=prepare_write_target(slot.cwd, rel_path), content=content)
                    except (SlotPathError, OSError) as restore_err:
                        # OSError included: a hostile eval can make unlink/mkdir
                        # fail; the result must still reach submit below rather
                        # than stranding the claimed task on the reaper.
                        print(f"{tag} skipped restoring {rel_path!r}: {restore_err}", flush=True)

        if save_logs:
            with _artifact_lock:
                view.artifacts.save_node_execution_output(job.node_id, term_out)

        # Per-slot resilience: retry a transiently-failed submit before giving
        # the result up — a claimed task with a lost result would otherwise sit
        # until the staleness reaper. WecoClient.suggest already attempts its
        # own recovery (poll-and-synthesize) first; a retry that finds the
        # submit actually landed 409s, which the caller treats as conflict,
        # never as completion.
        submit_attempts = 0
        while True:
            try:
                result = submit_execution_result(
                    run_id=job.run_id,
                    task_id=job.task_id,
                    execution_output=term_out,
                    auth_headers=auth_headers,
                    api_keys=api_keys,
                    timeout=(10, submit_timeout) if submit_timeout is not None else None,
                    skip_generation=True,
                )
                break
            except (RequestsConnectionError, ReadTimeout):
                submit_attempts += 1
                if submit_attempts > 2 or job.cancel_event.is_set():
                    raise
                time.sleep(3 * submit_attempts)
    except HTTPError as e:
        status_code = getattr(e.response, "status_code", None)
        detail = format_api_error(e)
        if status_code in (401, 402):
            return _EvalOutcome(kind="account_failure", detail=detail)
        if status_code == 409:
            # The run/task was finalized under us (hard stop, or the result
            # already landed). NEVER completion — the authoritative poll decides.
            return _EvalOutcome(kind="conflict", detail=detail)
        return _EvalOutcome(kind="run_failure", detail=detail)
    except (RequestsConnectionError, ReadTimeout) as e:
        return _EvalOutcome(kind="network", detail=str(e))
    except Exception as e:  # Defensive: a worker must never raise into the pool.
        return _EvalOutcome(kind="run_failure", detail=f"{type(e).__name__}: {e}")

    metric = result.get("previous_solution_metric_value")
    if result.get("is_done", False):
        return _EvalOutcome(kind="run_done", metric=metric)
    return _EvalOutcome(kind="scored", metric=metric)


def _generate_for_run(run_id: str, auth_headers: dict, api_keys: Optional[dict]) -> dict:
    """Runs in the generation pool: one /generate admission attempt."""
    return WecoClient(auth_headers).generate_candidate(run_id, api_keys=api_keys)


def _submit_daemon_future(fn, /, *args, name: str, **kwargs) -> Future:
    """Run one blocking HTTP call in a daemon thread represented by a Future.

    Generation uses an LLM-length read timeout. Executor workers are joined at
    interpreter exit, which can make Ctrl-C wait for that entire timeout.
    Admission already bounds outstanding calls by local K, so daemonizing each
    admitted request preserves the bound and permits prompt interrupted exits.
    Clean exits still wait in the scheduler loop until every future settles.
    """
    future = Future()

    def _run() -> None:
        if not future.set_running_or_notify_cancel():
            return
        try:
            result = fn(*args, **kwargs)
        except BaseException as exc:
            future.set_exception(exc)
        else:
            future.set_result(result)

    threading.Thread(target=_run, name=name, daemon=True).start()
    return future


def _rotate_run_ids(run_ids: list[str], cursor: int) -> list[str]:
    """Deterministic round-robin order for one equal-priority run bucket."""
    if not run_ids:
        return []
    ordered = sorted(run_ids)
    offset = cursor % len(ordered)
    return ordered[offset:] + ordered[:offset]


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------


class _TaggedPlainUIFactory:
    """Builds the per-run tagged plain UI lazily (import here to avoid cycles)."""

    def __init__(self, auth_headers: dict, log_dir: str, dashboard_base: str) -> None:
        self.auth_headers = auth_headers
        self.log_dir = log_dir
        self.dashboard_base = dashboard_base

    def build(self, run_id: str) -> _RunView:
        from .optimizer import _build_run_state

        state = _build_run_state(run_id, self.auth_headers, self.log_dir, self.dashboard_base)
        return _RunView(ui=state["ui"], artifacts=state["artifacts"])


def _fetch_active_members(lineage_id: str, auth_headers: dict) -> Optional[list[str]]:
    """Run IDs of lineage members still ``running``; None if the read failed."""
    try:
        lineage = WecoClient(auth_headers).get_lineage(lineage_id)
    except Exception:
        return None
    return [m["id"] for m in lineage.get("members", []) if m.get("status") == "running"]


def run_parallel_lineage_loop(
    lineage_id: str,
    auth_headers: dict,
    *,
    slots: list[Slot],
    lineage_k: int,
    originals: dict[str, str],
    eval_command: str,
    eval_timeout: Optional[int],
    save_logs: bool,
    log_dir: str,
    dashboard_base: str,
    api_keys: Optional[dict] = None,
    submit_timeout: Optional[int] = None,
    poll_interval: float = 2.0,
    max_idle_polls: int = 300,
) -> str:
    """Drain a lineage with up to ``len(slots)`` concurrent isolated evaluations.

    The caller MUST hold the working-tree consumer lock for the duration and
    owns slot provisioning/cleanup. Returns one of:

    * ``"ok"`` — exited on confirmed lineage quiescence;
    * ``"out_of_credits"`` — the wallet fell below the dispatch floor:
      generation stopped lineage-wide, in-flight evaluations drained, and the
      server terminates each drained member (results are kept; topup + resume
      continues the work);
    * ``"idle_timeout"`` — exited via the bounded-idle safety valve with the
      lineage possibly still active (the caller must not report this as a
      user interrupt);
    * ``"interrupted"`` — Ctrl-C;
    * ``"fatal"`` — an account-level failure (auth / insufficient credits)
      aborted the consumer.
    """
    from .optimizer import LineageHeartbeatSender  # avoid import cycle

    # A wedged evaluation must never pin a slot forever at K>1: without a
    # timeout there is no per-slot reaping (only whole-run heartbeat expiry).
    if eval_timeout is None:
        eval_timeout = 3600
        print(
            f"[lineage {lineage_id}] no eval timeout configured; parallel slots enforce a "
            f"{eval_timeout}s per-evaluation timeout (set --eval-timeout to control this).",
            flush=True,
        )

    state = _SchedulerState(free_slots=list(slots))
    ui_factory = _TaggedPlainUIFactory(auth_headers, log_dir, dashboard_base)
    # Never keep more work admitted than we can actually evaluate soon: the
    # database enforces lineage K; locally we also stay within our slot count
    # (a resume may hold fewer slots than K — candidates for the difference
    # would go stale waiting for a slot).
    target_in_flight = min(lineage_k, len(slots))

    eval_pool = ThreadPoolExecutor(max_workers=max(1, len(slots)), thread_name_prefix="weco-eval")
    stop_event = threading.Event()
    heartbeat = LineageHeartbeatSender(lineage_id, auth_headers, stop_event)
    heartbeat.start()

    members: list[str] = []
    members_stale = True
    idle_polls = 0
    poll_count = 0

    print(f"[lineage {lineage_id}] parallel scheduler started: {len(slots)} local slot(s), lineage K={lineage_k}.", flush=True)

    def _view(run_id: str) -> _RunView:
        view = state.runs.get(run_id)
        if view is None:
            view = ui_factory.build(run_id)
            state.runs[run_id] = view
        return view

    clean_exit = False
    interrupted = False
    idle_timeout = False
    try:
        while True:
            progressed = False

            # 1. Reap completed evaluation futures; free their slots.
            for future in [f for f in state.eval_futures if f.done()]:
                job = state.eval_futures.pop(future)
                state.free_slots.append(job.slot)
                state.saturated = False  # capacity may have freed
                state.settle_epoch += 1
                members_stale = True
                progressed = True
                view = _view(job.run_id)
                view.awaiting_score = False
                view.settle_epoch += 1
                outcome = future.result()
                if outcome.kind == "account_failure":
                    view.ui.on_error(outcome.detail or "Account-level failure")
                    state.fatal = True
                elif outcome.kind == "conflict":
                    print(f"[{job.run_id}] result submit conflicted (run stopped or already recorded); moving on.", flush=True)
                elif outcome.kind == "run_failure":
                    view.ui.on_error(outcome.detail or "Evaluation submit failed")
                elif outcome.kind == "network":
                    view.ui.on_warning(f"Network error during evaluation submit: {outcome.detail}")
                elif outcome.kind == "cancelled":
                    print(f"[{job.run_id}][slot {job.slot.index}] evaluation cancelled (run stopped).", flush=True)
                else:
                    view.scored += 1
                    if outcome.metric is not None:
                        view.ui.on_metric(view.scored, outcome.metric)
                    if outcome.kind == "run_done":
                        view.done = True
                        view.ui.on_complete(view.scored)
            if state.fatal:
                break

            # 2. Reap completed generation futures.
            for future in [f for f in state.gen_futures if f.done()]:
                call = state.gen_futures.pop(future)
                run_id = call.run_id
                progressed = True
                try:
                    result = future.result()
                except Exception as e:
                    # Ambiguous transport failure: the candidate may have
                    # committed server-side. Do NOT blind-retry this run —
                    # the next authoritative poll will surface any new task,
                    # and the DB cap bounds the damage either way.
                    print(f"[{run_id}] /generate failed transiently: {e}; re-checking queue before retrying.", flush=True)
                    members_stale = True
                    continue
                if result.get("generated"):
                    members_stale = True
                    continue
                reason = result.get("reason")
                # Latch a negative reply ONLY if nothing settled since its
                # dispatch. A reply that raced a settle may predate it — step 1
                # is the sole clearer of `saturated` (and an eval settle one of
                # two clearers of `awaiting_score`), so latching stale info
                # after the last eval reaped would silence generation for good.
                # Discarded replies are simply retried on a later iteration.
                if reason == "at_capacity":
                    if call.lineage_epoch == state.settle_epoch:
                        state.saturated = True
                elif reason == "no_dispatchable_work":
                    view = _view(run_id)
                    if call.run_epoch == view.settle_epoch:
                        view.awaiting_score = True
                elif reason in ("out_of_budget", "not_running"):
                    _view(run_id).generation_retired = True
                    members_stale = True
                elif reason == "out_of_credits":
                    # No epoch check: unlike capacity/dispatchability, solvency
                    # is not restored by a settle — a stale reply is still true.
                    if not state.out_of_credits:
                        state.out_of_credits = True
                        print(
                            f"[lineage {lineage_id}] out of credits: generation stopped for the whole "
                            "lineage; draining in-flight evaluations.",
                            flush=True,
                        )
                    members_stale = True

            # 3. Authoritative queue read.
            poll_count += 1
            tasks_result = get_lineage_execution_tasks(lineage_id, auth_headers)
            read_ok = tasks_result is not None
            ready = (tasks_result.tasks if tasks_result else []) or []
            active_run_count = tasks_result.active_run_count if tasks_result else None

            # A run with ready work self-evidently has dispatchable state again:
            # release its no_dispatchable_work latch even if none of OUR eval
            # futures scored for it (e.g. an approval landed, or the state
            # changed server-side) — otherwise the latch could exclude the run
            # from generation for the rest of the session.
            for task in ready:
                latched = state.runs.get(task["run_id"])
                if latched is not None:
                    latched.awaiting_score = False

            # 3b. Cancellation poller: refresh membership on a short cadence
            # (not only when stale) so a dashboard "stop" frees its slots
            # mid-eval instead of after the eval finishes. A failed membership
            # read cancels nothing.
            if members_stale or poll_count % 3 == 0 or any(job.run_id not in members for job in state.eval_futures.values()):
                fetched = _fetch_active_members(lineage_id, auth_headers)
                if fetched is not None:
                    members = fetched
                    members_stale = False
                    for job in state.eval_futures.values():
                        if job.run_id not in members and not job.cancel_event.is_set():
                            print(
                                f"[{job.run_id}][slot {job.slot.index}] run left 'running'; cancelling its evaluation.",
                                flush=True,
                            )
                            job.cancel_event.set()
                    # A blocking LLM request cannot be forcibly interrupted,
                    # but it runs on a daemon thread. Once its run is
                    # authoritatively no longer active, stop tracking it so a
                    # hard stop can release the consumer promptly; backend
                    # settlement remains guarded by run status/reservations.
                    for future, gen_call in list(state.gen_futures.items()):
                        if gen_call.run_id not in members:
                            future.cancel()
                            state.gen_futures.pop(future)

            # 4. Exit only on confirmed lineage quiescence.
            if read_ok and not ready and not state.eval_futures and not state.gen_futures and active_run_count == 0:
                break

            # 4b. Insolvency drain-complete: once out_of_credits is latched this
            # scheduler stops calling /generate — which is exactly the polling
            # path that triggers the server's drain-terminate backstop — so
            # waiting for active_run_count == 0 would idle for the full valve
            # window (live-verified: 10 minutes). Nothing more can happen
            # locally with no ready work and no futures; exit now and let the
            # finalizer report the honest out_of_credits termination for a run
            # the backstop didn't reach. (Members left 'running' are reaped by
            # the heartbeat cron once our heartbeat stops.)
            if state.out_of_credits and read_ok and not ready and not state.eval_futures and not state.gen_futures:
                break

            # 5. Assign ready tasks to free slots, round-robin by run — one
            # task per run per pass, so no run drains its whole backlog into
            # every free slot before another run gets a turn.
            consumed_task_ids: set[str] = set()
            if read_ok and ready and state.free_slots:
                by_run: dict[str, list[dict]] = {}
                for task in ready:
                    task_run = task.get("run") or {}
                    if task_run.get("status") not in (None, "running"):
                        continue
                    by_run.setdefault(task["run_id"], []).append(task)
                run_order = sorted(by_run)
                if run_order:
                    state.rr_cursor %= len(run_order)
                    rotated = run_order[state.rr_cursor :] + run_order[: state.rr_cursor]
                    state.rr_cursor = (state.rr_cursor + 1) % len(run_order)
                    while state.free_slots and any(by_run[r] for r in rotated):
                        for run_id in rotated:
                            if not state.free_slots:
                                break
                            if not by_run[run_id]:
                                continue
                            task = by_run[run_id].pop(0)
                            # Consumed either way: a failed claim means the task
                            # is gone server-side, not still pending.
                            consumed_task_ids.add(task["id"])
                            claimed = claim_execution_task(task["id"], auth_headers)
                            if claimed is None:
                                continue  # cancelled as its run wound down
                            slot = state.free_slots.pop()
                            view = _view(run_id)
                            plan = claimed["revision"].get("plan")
                            print(f"[{run_id}][slot {slot.index}] evaluating node {claimed['node_id']}", flush=True)
                            view.ui.on_task_claimed(task["id"], plan)
                            job = _EvalJob(
                                task_id=task["id"],
                                run_id=run_id,
                                node_id=claimed["node_id"],
                                slot=slot,
                                file_map=claimed["revision"]["code"],
                                plan=plan,
                            )
                            future = eval_pool.submit(
                                _evaluate_and_submit,
                                job,
                                auth_headers=auth_headers,
                                originals=originals,
                                eval_command=eval_command,
                                eval_timeout=eval_timeout,
                                save_logs=save_logs,
                                view=view,
                                api_keys=api_keys,
                                submit_timeout=submit_timeout,
                            )
                            state.eval_futures[future] = job
                            progressed = True

            # 6. Generation refill — fair rotation, zero-work members first.
            # (Membership is kept fresh by the cancellation poller above.)
            if read_ok and not state.fatal and not state.saturated and not state.out_of_credits:
                # Tasks step 5 just claimed are already counted via
                # eval_futures; counting them again via `ready` would
                # understate `room` by one slot-fill per iteration.
                pending_ready = [t for t in ready if t["id"] not in consumed_task_ids]
                ready_counts: dict[str, int] = {}
                for task in pending_ready:
                    ready_counts[task["run_id"]] = ready_counts.get(task["run_id"], 0) + 1
                evaluating_counts: dict[str, int] = {}
                for job in state.eval_futures.values():
                    evaluating_counts[job.run_id] = evaluating_counts.get(job.run_id, 0) + 1
                generating = {gen_call.run_id for gen_call in state.gen_futures.values()}

                in_flight_estimate = len(state.eval_futures) + len(state.gen_futures) + len(pending_ready)
                room = target_in_flight - in_flight_estimate

                def _work(run_id: str) -> int:
                    return ready_counts.get(run_id, 0) + evaluating_counts.get(run_id, 0) + (1 if run_id in generating else 0)

                eligible = [
                    run_id for run_id in members if not _view(run_id).generation_retired and not _view(run_id).awaiting_score
                ]
                # Zero-work members (e.g. a deferred derived child) always go
                # first; the rest rotate fairly. A run may hold SEVERAL
                # outstanding /generate calls at once (SPEC: "while lineage
                # admission has room, issue concurrent /generate requests") —
                # that is what fills K slots during ramp-up and after a backlog
                # drains; one-at-a-time generation would cap a single run's
                # effective concurrency at ~1 whenever generation latency is
                # comparable to eval time. `room` bounds the burst, the DB
                # reservation is the hard cap, a nothing-to-dispatch run
                # latches awaiting_score after one round trip, and at_capacity
                # flips `saturated` — so over-issuing self-quenches.
                zero_work = [run_id for run_id in eligible if _work(run_id) == 0]
                busy = [run_id for run_id in eligible if _work(run_id) > 0]
                eligible = _rotate_run_ids(zero_work, state.gen_rr_cursor) + _rotate_run_ids(busy, state.gen_rr_cursor)
                submitted = 0
                while room > 0 and eligible:
                    for run_id in eligible:
                        if room <= 0:
                            break
                        future = _submit_daemon_future(
                            _generate_for_run, run_id, auth_headers, api_keys, name=f"weco-gen-{run_id[:8]}"
                        )
                        state.gen_futures[future] = _GenCall(
                            run_id=run_id, lineage_epoch=state.settle_epoch, run_epoch=_view(run_id).settle_epoch
                        )
                        room -= 1
                        submitted += 1
                        progressed = True
                state.gen_rr_cursor += submitted

            # 7. Idle accounting — bounded wait only when no local operation is
            # still making legitimate progress. Evaluations can intentionally
            # run longer than the idle window (their own timeout is authoritative).
            if progressed or state.eval_futures or state.gen_futures:
                idle_polls = 0
            else:
                idle_polls += 1
                if idle_polls >= max_idle_polls:
                    print(f"[lineage {lineage_id}] no progress after {max_idle_polls} polls; stopping scheduler.", flush=True)
                    idle_timeout = True
                    break
            time.sleep(poll_interval if not progressed else 0.2)

        clean_exit = True
    except KeyboardInterrupt:
        interrupted = True
        # Slot evals lead their own process groups. Signal every in-flight job
        # before the provider tears down its worktrees; workers observe the
        # event within about one second and reap their subprocess trees.
        for job in state.eval_futures.values():
            job.cancel_event.set()
        for view in state.runs.values():
            view.ui.on_interrupted()
    finally:
        stop_event.set()
        heartbeat.join(timeout=2)
        if clean_exit:
            eval_pool.shutdown(wait=True)
        else:
            # On interruption, cancel process groups and wait for the workers'
            # prompt cancellation path before their slot directories disappear.
            for job in state.eval_futures.values():
                job.cancel_event.set()
            eval_pool.shutdown(wait=True, cancel_futures=True)
        for future in state.gen_futures:
            future.cancel()

    print(f"[lineage {lineage_id}] parallel scheduler finished.", flush=True)
    if state.fatal:
        return "fatal"
    if interrupted:
        return "interrupted"
    if state.out_of_credits:
        # Ranked above idle_timeout: if the server's drain-terminate backstop
        # didn't end the members (so quiescence never confirmed and the idle
        # valve fired instead), insolvency is still the true cause — the
        # finalizer uses it to record an honest termination reason.
        return "out_of_credits"
    if idle_timeout:
        return "idle_timeout"
    return "ok"
