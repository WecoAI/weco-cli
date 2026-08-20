"""Manual ``weco resume`` handling of the backend's ``is_done`` resume signal.

Milestone M's ``/runs/{id}/resume`` endpoint may finalize a run itself: it promotes a
scored-but-interrupted node, the step budget is met, and the run flips to
'completed' with no runnable work. It reports this with ``is_done=True`` in the
resume response. The CLI must:

* short-circuit to a success exit WITHOUT entering the task-polling loop when
  ``is_done`` is true (otherwise it polls a completed run and misreports a stop);
* preserve the exact legacy path when ``is_done`` is absent (older backend).
"""

from unittest.mock import MagicMock

import pytest

import weco.optimizer as optimizer
from weco.optimizer import OptimizationResult, resume_optimization


def _status(**overrides):
    base = {
        "status": "terminated",
        "objective": {"metric_name": "accuracy", "maximize": True, "evaluation_command": "python eval.py"},
        "optimizer": {"steps": 10, "code_generator": {"model": "o4-mini"}, "evaluator": {"model": "o4-mini"}},
        "current_step": 4,
        "run_name": "demo-run",
        "updated_at": "2026-07-13T00:00:00Z",
        "lineage_id": "lineage-1",
    }
    base.update(overrides)
    return base


def _resume_resp(**overrides):
    base = {
        "source_code": {"solution.py": "print('fallback')"},
        "run_name": "demo-run",
        "log_dir": ".runs",
        "save_logs": False,
        "eval_timeout": None,
        "steps": 10,
        "best_metric_value": 0.9,
        "best_step": 3,
    }
    base.update(overrides)
    return base


@pytest.fixture
def common_mocks(monkeypatch):
    """Patch resume_optimization's boundaries; return the mock registry."""
    mocks = {
        "auth": MagicMock(return_value=("key", {"Authorization": "Bearer x"})),
        "status": MagicMock(return_value=_status()),
        "resume": MagicMock(),
        "loop": MagicMock(),
        "apply": MagicMock(),
    }
    monkeypatch.setattr(optimizer, "handle_authentication", mocks["auth"])
    monkeypatch.setattr(optimizer, "get_optimization_run_status", mocks["status"])
    monkeypatch.setattr(optimizer, "resume_optimization_run", mocks["resume"])
    monkeypatch.setattr(optimizer, "_run_loop_with_auto_resume", mocks["loop"])
    monkeypatch.setattr(optimizer, "offer_apply_best_solution", mocks["apply"])
    # Confirm.ask ("kept files unchanged?") -> yes
    monkeypatch.setattr(optimizer.Confirm, "ask", staticmethod(lambda *a, **k: True))
    # Never touch the real filesystem lock / heartbeat / drain / termination.
    monkeypatch.setattr(optimizer, "try_acquire", MagicMock(return_value=object()))
    monkeypatch.setattr(optimizer, "release", MagicMock())
    monkeypatch.setattr(optimizer, "_drain_lineage_remainder", MagicMock())
    monkeypatch.setattr(optimizer, "report_termination", MagicMock(return_value=True))
    monkeypatch.setattr(optimizer, "LineageHeartbeatSender", MagicMock())
    return mocks


def test_is_done_true_exits_success_without_entering_task_loop(common_mocks):
    common_mocks["resume"].return_value = _resume_resp(is_done=True)

    result = resume_optimization("run-1", output_mode="plain", no_open=True)

    assert result is True
    # The task-polling loop was never entered.
    common_mocks["loop"].assert_not_called()
    # Best-solution handling still runs, matching normal completion.
    common_mocks["apply"].assert_called_once()


def test_is_done_absent_preserves_legacy_loop_path(common_mocks):
    # Old backend: response carries no is_done field at all.
    common_mocks["resume"].return_value = _resume_resp()
    common_mocks["loop"].return_value = OptimizationResult(
        success=True, final_step=10, status="completed", reason="completed_successfully"
    )

    result = resume_optimization("run-1", output_mode="plain", no_open=True)

    assert result is True
    # Legacy behavior unchanged: the task-polling loop IS entered.
    common_mocks["loop"].assert_called_once()


def test_is_done_false_preserves_legacy_loop_path(common_mocks):
    # Newer backend that explicitly reports there is still work to do.
    common_mocks["resume"].return_value = _resume_resp(is_done=False)
    common_mocks["loop"].return_value = OptimizationResult(
        success=True, final_step=10, status="completed", reason="completed_successfully"
    )

    result = resume_optimization("run-1", output_mode="plain", no_open=True)

    assert result is True
    common_mocks["loop"].assert_called_once()


def test_is_done_acquires_consumer_lock_before_applying(common_mocks, monkeypatch):
    """B2: the is_done short-circuit writes candidate files into the working tree via
    offer_apply_best_solution, so it MUST hold the consumer lock first — exactly as
    the normal completion path does. Assert lock is acquired BEFORE the apply and
    released AFTER, so a concurrent consumer in the same tree is never raced."""
    common_mocks["resume"].return_value = _resume_resp(is_done=True)

    calls = []
    handle = object()
    monkeypatch.setattr(optimizer, "try_acquire", MagicMock(side_effect=lambda *a, **k: calls.append("acquire") or handle))
    monkeypatch.setattr(optimizer, "release", MagicMock(side_effect=lambda h: calls.append("release")))
    common_mocks["apply"].side_effect = lambda *a, **k: calls.append("apply")

    result = resume_optimization("run-1", output_mode="plain", no_open=True)

    assert result is True
    common_mocks["loop"].assert_not_called()
    # Ordering: lock acquired -> apply -> lock released.
    assert calls == ["acquire", "apply", "release"], calls
    optimizer.release.assert_called_once_with(handle)


def test_is_done_lock_unavailable_skips_apply_still_success(common_mocks, monkeypatch):
    """B2: another consumer already holds the working-tree lock. The apply MUST be
    skipped (so we don't clobber that consumer's in-flight eval), but the run is
    still complete, so the resume reports success (returns True)."""
    common_mocks["resume"].return_value = _resume_resp(is_done=True)

    # try_acquire returns None -> lock held by someone else.
    monkeypatch.setattr(optimizer, "try_acquire", MagicMock(return_value=None))
    monkeypatch.setattr(optimizer, "release", MagicMock())

    result = resume_optimization("run-1", output_mode="plain", no_open=True)

    assert result is True
    common_mocks["loop"].assert_not_called()
    # Lock was unavailable -> no apply, and nothing to release.
    common_mocks["apply"].assert_not_called()
    optimizer.release.assert_not_called()
