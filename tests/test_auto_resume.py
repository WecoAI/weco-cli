from unittest.mock import MagicMock

import pytest

from weco.optimizer import AutoResumePolicy, OptimizationResult, _is_transient, _run_loop_with_auto_resume


class FakeUI:
    def __init__(self):
        self.reconnecting: list[tuple[int, int, float]] = []
        self.reconnected: int = 0
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def on_reconnecting(self, attempt: int, max_attempts: int, backoff_s: float) -> None:
        self.reconnecting.append((attempt, max_attempts, backoff_s))

    def on_reconnected(self) -> None:
        self.reconnected += 1

    def on_error(self, message: str) -> None:
        self.errors.append(message)

    def on_warning(self, message: str) -> None:
        self.warnings.append(message)


def _make_result(reason: str, *, status: str = "error", success: bool = False, final_step: int = 0) -> OptimizationResult:
    return OptimizationResult(success=success, final_step=final_step, status=status, reason=reason)


@pytest.fixture
def stub_sleep(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr("weco.optimizer.time.sleep", lambda s: sleeps.append(s))
    return sleeps


@pytest.fixture
def stub_resume(monkeypatch):
    calls: list[str] = []

    def _install(outcomes: list[bool]):
        iterator = iter(outcomes)

        def _fake(run_id, auth_headers, api_keys):
            calls.append(run_id)
            return next(iterator)

        monkeypatch.setattr("weco.optimizer._silent_resume", _fake)
        return calls

    return _install


def _drive(results: list[OptimizationResult], *, policy: AutoResumePolicy | None = None, initial_start_step: int = 0):
    factory = MagicMock(side_effect=results)
    ui = FakeUI()
    returned = _run_loop_with_auto_resume(
        factory,
        ui=ui,
        run_id="run-1",
        auth_headers={},
        api_keys=None,
        policy=policy or AutoResumePolicy(),
        initial_start_step=initial_start_step,
    )
    return returned, factory, ui


@pytest.mark.parametrize(
    "reason,transient",
    [
        ("transient_network_error", True),
        ("http_502", True),
        ("http_503", True),
        ("http_504", True),
        ("http_401", False),
        ("http_402", False),
        ("http_500", False),
        ("user_terminated_sigint", False),
        ("completed_successfully", False),
        ("user_requested_stop", False),
        ("timeout_waiting_for_tasks", False),
        ("unknown", False),
    ],
)
def test_classifies_transient_reasons(reason, transient):
    assert _is_transient(_make_result(reason)) is transient


def test_returns_verbatim_when_not_transient(stub_sleep, stub_resume):
    stub_resume([])
    completed = _make_result("completed_successfully", status="completed", success=True, final_step=7)

    returned, factory, ui = _drive([completed])

    assert returned is completed
    assert factory.call_count == 1
    assert stub_sleep == []
    assert ui.reconnecting == []
    assert ui.reconnected == 0
    assert ui.errors == []


def test_resumes_once_then_continues_from_final_step(stub_sleep, stub_resume):
    resume_calls = stub_resume([True])
    transient = _make_result("transient_network_error", final_step=4)
    completed = _make_result("completed_successfully", status="completed", success=True, final_step=9)

    returned, factory, ui = _drive([transient, completed])

    assert returned is completed
    assert factory.call_count == 2
    assert factory.call_args_list[0].args == (0,)
    assert factory.call_args_list[1].args == (4,)
    assert resume_calls == ["run-1"]
    assert len(stub_sleep) == 1
    assert len(ui.reconnecting) == 1
    assert ui.reconnected == 1
    assert ui.errors == []


def test_exhausts_after_max_attempts_and_returns_original_result(stub_sleep, stub_resume):
    resume_calls = stub_resume([True, True, True])
    transient = _make_result("transient_network_error", final_step=2)
    policy = AutoResumePolicy(max_attempts=3)

    returned, factory, ui = _drive([transient, transient, transient, transient], policy=policy)

    assert returned.reason == "transient_network_error"
    assert factory.call_count == 4
    assert len(resume_calls) == 3
    assert len(ui.reconnecting) == 3
    assert ui.reconnected == 3
    assert len(ui.errors) == 1
    assert "exhausted after 3" in ui.errors[0]


def test_disabled_policy_skips_resume_on_transient(stub_sleep, stub_resume):
    resume_calls = stub_resume([])
    transient = _make_result("transient_network_error", final_step=2)

    returned, factory, ui = _drive([transient], policy=AutoResumePolicy(enabled=False))

    assert returned is transient
    assert factory.call_count == 1
    assert resume_calls == []
    assert stub_sleep == []
    assert ui.reconnecting == []
    assert ui.reconnected == 0
    assert ui.errors == []


def test_silent_resume_failure_retries_without_reinvoking_loop(stub_sleep, stub_resume):
    resume_calls = stub_resume([False, True])
    transient = _make_result("transient_network_error", final_step=3)
    completed = _make_result("completed_successfully", status="completed", success=True)

    returned, factory, ui = _drive([transient, completed], policy=AutoResumePolicy(max_attempts=3))

    assert returned is completed
    assert factory.call_count == 2
    assert len(resume_calls) == 2
    assert len(stub_sleep) == 2
    assert len(ui.reconnecting) == 2
    assert ui.reconnected == 1
    assert ui.errors == []


def test_silent_resume_exhaustion_without_reinvoking_loop(stub_sleep, stub_resume):
    resume_calls = stub_resume([False, False])
    transient = _make_result("transient_network_error", final_step=3)

    returned, factory, ui = _drive([transient], policy=AutoResumePolicy(max_attempts=2))

    assert returned is transient
    assert factory.call_count == 1
    assert len(resume_calls) == 2
    assert len(ui.reconnecting) == 2
    assert ui.reconnected == 0
    assert len(ui.errors) == 1
    assert "exhausted after 2" in ui.errors[0]


def test_backoff_is_exponential_and_capped(stub_sleep, stub_resume):
    stub_resume([True, True, True, True, True])
    transient = _make_result("transient_network_error")
    completed = _make_result("completed_successfully", status="completed", success=True)
    policy = AutoResumePolicy(max_attempts=5, backoff_initial_s=1.0, backoff_factor=2.0, backoff_max_s=5.0)

    _drive([transient, transient, transient, transient, completed], policy=policy)

    assert stub_sleep == [1.0, 2.0, 4.0, 5.0]


def test_keyboard_interrupt_result_propagates_untouched(stub_sleep, stub_resume):
    resume_calls = stub_resume([])
    interrupted = _make_result("user_terminated_sigint", status="terminated")

    returned, factory, ui = _drive([interrupted])

    assert returned is interrupted
    assert factory.call_count == 1
    assert resume_calls == []
    assert stub_sleep == []
    assert ui.reconnecting == []
    assert ui.reconnected == 0
    assert ui.errors == []


def test_reconnecting_event_carries_attempt_and_backoff(stub_sleep, stub_resume):
    stub_resume([True])
    transient = _make_result("transient_network_error", final_step=2)
    completed = _make_result("completed_successfully", status="completed", success=True)
    policy = AutoResumePolicy(max_attempts=5, backoff_initial_s=3.0, backoff_factor=2.0, backoff_max_s=30.0)

    _, _, ui = _drive([transient, completed], policy=policy)

    assert ui.reconnecting == [(1, 5, 3.0)]
