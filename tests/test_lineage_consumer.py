"""Tests for the lineage consumer's poll-classification and exit behavior.

The consumer must never mistake a failed/empty read for "the lineage is done"
— that would orphan still-running runs. The decision is isolated in the pure
``_classify_lineage_poll`` (unit-tested directly); ``run_lineage_loop`` is then
exercised end-to-end to confirm it acts on those decisions.
"""

from unittest.mock import MagicMock, patch

import weco.optimizer as opt
from weco.optimizer import _PollAction, _classify_lineage_poll


_AUTH = {"Authorization": "Bearer t"}


# ---------------------------------------------------------------------------
# Pure decision function
# ---------------------------------------------------------------------------


def test_failed_read_is_never_done():
    # A read that didn't succeed says nothing about the lineage — keep waiting.
    assert _classify_lineage_poll(read_ok=False, has_ready=False, active_run_count=None) is _PollAction.WAIT
    assert _classify_lineage_poll(read_ok=False, has_ready=True, active_run_count=0) is _PollAction.WAIT


def test_ready_work_is_processed():
    # When there's work, the active count is irrelevant (and omitted by the server).
    assert _classify_lineage_poll(read_ok=True, has_ready=True, active_run_count=None) is _PollAction.PROCESS


def test_done_only_on_confirmed_empty_and_no_active_runs():
    assert _classify_lineage_poll(read_ok=True, has_ready=False, active_run_count=0) is _PollAction.DONE


def test_active_runs_without_work_keeps_waiting():
    # A member between candidates (active, no ready task yet) is not "done".
    assert _classify_lineage_poll(read_ok=True, has_ready=False, active_run_count=2) is _PollAction.WAIT


def test_missing_active_count_does_not_guess_done():
    # Server didn't report the count (e.g. version skew) — wait, don't assume done.
    assert _classify_lineage_poll(read_ok=True, has_ready=False, active_run_count=None) is _PollAction.WAIT


# ---------------------------------------------------------------------------
# Loop behavior
# ---------------------------------------------------------------------------


def _result(*items, active=None):
    """A successful lineage-queue read: given tasks plus an active-run count."""
    return MagicMock(tasks=list(items), active_run_count=active)


def _task(run_id="R", task_id="t1"):
    return {"id": task_id, "run_id": run_id, "run": {"status": "running"}}


def _run(loop_kwargs=None, *, get_tasks_side_effect, submit_return=None):
    """Drive run_lineage_loop with every boundary mocked. Returns the mock
    handles plus the loop's return value."""
    state = {"ui": MagicMock(), "artifacts": MagicMock(), "step": 1, "done": False}
    claimed = {"revision": {"code": {"main.py": "x"}, "plan": "p"}}

    with (
        patch("weco.optimizer.time.sleep"),
        patch("weco.optimizer.LineageHeartbeatSender") as HB,
        patch("weco.optimizer.get_lineage_execution_tasks", side_effect=get_tasks_side_effect) as get_tasks,
        patch("weco.optimizer._build_run_state", return_value=state),
        patch("weco.optimizer.claim_execution_task", return_value=claimed) as claim,
        patch("weco.optimizer.run_evaluation_with_files_swap", return_value="out") as eval_swap,
        patch("weco.optimizer.submit_execution_result", return_value=submit_return or {"is_done": False}) as submit,
    ):
        HB.return_value = MagicMock()
        result = opt.run_lineage_loop(
            lineage_id="L",
            auth_headers=_AUTH,
            originals={"main.py": "orig"},
            eval_command="python e.py",
            eval_timeout=None,
            save_logs=False,
            log_dir=".runs",
            dashboard_base="http://d",
            poll_interval=0,
            **(loop_kwargs or {}),
        )
        return MagicMock(result=result, claim=claim, submit=submit, eval_swap=eval_swap, get_tasks=get_tasks)


def test_transient_read_failure_does_not_orphan_a_ready_task():
    """A failed poll (read returns None) must not be read as quiescence: the
    consumer keeps going and still drains the task that appears next."""
    h = _run(
        # poll 1: read fails. poll 2: a ready task. poll 3: confirmed done.
        get_tasks_side_effect=[None, _result(_task()), _result(active=0)]
    )
    h.claim.assert_called_once()
    h.submit.assert_called_once()
    assert h.result is True


def test_exits_on_a_single_confirmed_done_poll():
    """One authoritative read showing no work and no active runs is enough —
    no arbitrary confirmation streak needed now that the count is consistent."""
    h = _run(get_tasks_side_effect=[_result(active=0)])
    h.claim.assert_not_called()
    h.submit.assert_not_called()
    assert h.get_tasks.call_count == 1
    assert h.result is True


def test_does_not_exit_while_a_run_is_active_with_no_tasks():
    """An active member with no ready task (between candidates, or freshly
    derived) keeps the consumer alive — up to the stuck backstop."""
    h = _run({"max_idle_polls": 3}, get_tasks_side_effect=[_result(active=1)] * 10)
    h.claim.assert_not_called()
    assert h.get_tasks.call_count == 3  # waited, then bailed via backstop
    assert h.result is True


def test_repeated_read_failures_bail_via_backstop_without_orphaning():
    """If the queue read keeps failing, the consumer waits (never declaring
    done) and finally bails via the backstop rather than spinning forever."""
    h = _run({"max_idle_polls": 2}, get_tasks_side_effect=[None] * 10)
    h.claim.assert_not_called()
    assert h.get_tasks.call_count == 2
    assert h.result is True
