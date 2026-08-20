"""M5 out_of_credits handling: lineage-wide generation retire in the K>1
scheduler, the serial loop's graceful billing exit, the unknown-reason
fallthrough pin, and the --parallel bound.

Server context: when the wallet falls below the dispatch floor, /generate and
/suggest answer the no-op reason ``out_of_credits`` and the server terminates
each run once its in-flight work drains (the CLI never has to end the lineage
itself — these tests cover the client-side UX on top of that)."""

import argparse

import pytest
from unittest.mock import MagicMock

from weco.core.api import ExecutionTasksResult, RunSummary
from weco.optimizer import OptimizationResult, run_optimization_loop, _is_transient

from .test_parallel_scheduler import FakeBackend, _install, _make_slot, _run


# ---------------------------------------------------------------------------
# K>1 scheduler
# ---------------------------------------------------------------------------


def test_out_of_credits_retires_generation_lineage_wide(tmp_path, monkeypatch):
    """One out_of_credits reply stops /generate for EVERY member (credits are
    user-scoped), and the scheduler exits with the distinct outcome once the
    server-terminated members quiesce."""
    state = {"denied": False}

    def handler(be, run_id):
        state["denied"] = True
        return {"generated": False, "reason": "out_of_credits"}

    def active(be):
        # The server's drain-terminate backstop ends the members once the
        # denial lands and nothing is in flight.
        return 0 if state["denied"] else 2

    backend = FakeBackend(
        members=[{"id": "A", "status": "running"}, {"id": "B", "status": "running"}],
        generate_handler=handler,
        active_count_fn=active,
    )
    _install(monkeypatch, backend)
    slots = [_make_slot(tmp_path / "slots", i) for i in range(2)]

    result = _run(backend, slots, tmp_path / "logs", lineage_k=2, max_idle_polls=50)

    assert result == "out_of_credits"
    # Generation stopped for the WHOLE lineage after the first denial: only the
    # initial same-iteration burst is allowed, never per-member retries.
    assert len(backend.generate_calls) <= 4


def test_out_of_credits_exits_promptly_when_members_stay_running(tmp_path, monkeypatch):
    """Backstop-miss case (live-verified): once out_of_credits is latched this
    CLI stops polling /generate — the very path that triggers the server's
    drain-terminate backstop — so active_run_count may never reach 0. The
    drain-complete exit (step 4b) must end the session promptly instead of
    idling out the full valve window, and the outcome stays out_of_credits so
    the finalizer records the honest termination reason."""
    backend = FakeBackend(
        members=[{"id": "A", "status": "running"}],
        generate_handler=lambda be, rid: {"generated": False, "reason": "out_of_credits"},
        active_count_fn=lambda be: 1,  # never quiesces
    )
    _install(monkeypatch, backend)
    slots = [_make_slot(tmp_path / "slots", 0)]

    result = _run(backend, slots, tmp_path / "logs", lineage_k=1, max_idle_polls=500)

    assert result == "out_of_credits"
    # Prompt exit: with max_idle_polls=500 an idle-valve exit would need ~500
    # polls; the drain-complete break fires within a few. The lineage-wide
    # gate also means no generate-retry churn.
    assert len(backend.generate_calls) <= 2


def test_unknown_generation_reason_fallthrough_is_pinned(tmp_path, monkeypatch):
    """PIN: an unrecognized no-op reason is NOT latched — the run stays
    eligible and /generate is retried on later iterations. Any future reason
    the server may add MUST therefore come with a server-side backstop that
    eventually makes the lineage terminal (as out_of_credits does), or old
    CLIs will retry until their idle valve. This test exists so that adding a
    reason without reading this is hard."""
    backend = FakeBackend(
        members=[{"id": "A", "status": "running"}],
        generate_handler=lambda be, rid: {"generated": False, "reason": "mystery_reason_from_the_future"},
        # End the test through quiescence once the retry behavior is proven.
        active_count_fn=lambda be: 0 if len(be.generate_calls) >= 6 else 1,
    )
    _install(monkeypatch, backend)
    slots = [_make_slot(tmp_path / "slots", 0)]

    result = _run(backend, slots, tmp_path / "logs", lineage_k=1, max_idle_polls=200)

    assert result == "ok"
    # The unknown reason kept the run eligible: generation was retried, not
    # retired after one attempt (contrast the out_of_credits tests above).
    assert len(backend.generate_calls) >= 6


# ---------------------------------------------------------------------------
# Serial (K=1) loop
# ---------------------------------------------------------------------------


def test_serial_out_of_credits_is_graceful_billing_exit(monkeypatch):
    """The scored result is kept; the loop ends with reason=out_of_credits and
    a billing message — never a completion, never a bogus stop/timeout."""
    monkeypatch.setattr("weco.optimizer.get_optimization_run_status", MagicMock(return_value={"status": "running"}))
    tasks_result = ExecutionTasksResult(tasks=[{"id": "task-1"}], run=RunSummary(id="run-1", status="running"))
    monkeypatch.setattr("weco.optimizer.get_execution_tasks", MagicMock(return_value=tasks_result))
    monkeypatch.setattr(
        "weco.optimizer.claim_execution_task", MagicMock(return_value={"revision": {"code": {"main.py": "pass"}, "plan": "p"}})
    )
    monkeypatch.setattr("weco.optimizer.run_evaluation_with_files_swap", MagicMock(return_value="metric: 0.9"))
    monkeypatch.setattr(
        "weco.optimizer.submit_execution_result",
        MagicMock(
            return_value={"is_done": False, "previous_solution_metric_value": 0.9, "reason": "out_of_credits", "balance": 0.25}
        ),
    )

    ui = MagicMock()
    result = run_optimization_loop(
        ui=ui,
        run_id="run-1",
        auth_headers={"Authorization": "Bearer x"},
        source_code={"main.py": "pass"},
        eval_command="python eval.py",
        eval_timeout=None,
        artifacts=MagicMock(),
        save_logs=False,
    )

    assert result.success is False
    assert result.status == "terminated"
    assert result.reason == "out_of_credits"
    # The scored metric reached the UI before the exit.
    ui.on_metric.assert_called_once()
    ui.on_complete.assert_not_called()
    # The user got the billing message with the balance and the recovery path.
    message = ui.on_error.call_args.args[0]
    assert "credits" in message.lower()
    assert "$0.25" in message
    assert "weco resume run-1" in message


def test_out_of_credits_is_not_a_transient_reason():
    """out_of_credits must never trigger the auto-resume retry loop — resuming
    with an empty wallet would 402 (or drain-terminate) forever."""
    result = OptimizationResult(success=False, final_step=3, status="terminated", reason="out_of_credits")
    assert _is_transient(result) is False


# ---------------------------------------------------------------------------
# --parallel bound (mirrors the server's le=32 request validation)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value,ok", [("1", True), ("32", True), ("0", False), ("33", False), ("x", False)])
def test_parallel_flag_bound(value, ok):
    from weco.cli import configure_run_parser

    parser = argparse.ArgumentParser()
    configure_run_parser(parser)
    if ok:
        args = parser.parse_args(["--parallel", value])
        assert args.parallel == int(value)
    else:
        with pytest.raises(SystemExit):
            parser.parse_args(["--parallel", value])
