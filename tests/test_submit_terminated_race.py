"""Regression: a /suggest submit that 409s (run terminated mid-race) is an error,
never a false success.

Milestone M's backend converges a raced hard stop to an HTTP 409 on the submit
(``/suggest``) call: if a hard stop wins the completion race, the backend returns
409 instead of a truthful-looking ``is_done=true`` (which the CLI would otherwise
map to ``success=True``/``status="completed"``). This test pins that the CLI's
loop treats that 409 as a stop/error outcome — it must NOT call ``ui.on_complete``
and must NOT report ``success=True`` or ``status="completed"``.
"""

from unittest.mock import MagicMock

import requests

from weco.core.api import ExecutionTasksResult, RunSummary
from weco.optimizer import run_optimization_loop


def _http_409(detail: str = "Run status is `terminated`") -> requests.exceptions.HTTPError:
    resp = requests.Response()
    resp.status_code = 409
    resp._content = f'{{"detail":"{detail}"}}'.encode()
    return requests.exceptions.HTTPError(response=resp)


def test_submit_409_terminated_is_not_reported_as_success(monkeypatch):
    """One task claimed and evaluated; the submit 409s (hard stop won the race).

    The loop must surface an error result (success=False, status != 'completed')
    and never signal completion to the UI.
    """
    # Run is live; a single ready task is waiting.
    monkeypatch.setattr("weco.optimizer.get_optimization_run_status", MagicMock(return_value={"status": "running"}))
    tasks_result = ExecutionTasksResult(tasks=[{"id": "task-1"}], run=RunSummary(id="run-1", status="running"))
    monkeypatch.setattr("weco.optimizer.get_execution_tasks", MagicMock(return_value=tasks_result))
    monkeypatch.setattr(
        "weco.optimizer.claim_execution_task", MagicMock(return_value={"revision": {"code": {"main.py": "pass"}, "plan": "p"}})
    )
    monkeypatch.setattr("weco.optimizer.run_evaluation_with_files_swap", MagicMock(return_value="metric: 0.9"))
    # The submit races a hard stop and 409s.
    monkeypatch.setattr("weco.optimizer.submit_execution_result", MagicMock(side_effect=_http_409()))

    ui = MagicMock()
    artifacts = MagicMock()

    result = run_optimization_loop(
        ui=ui,
        run_id="run-1",
        auth_headers={"Authorization": "Bearer x"},
        source_code={"main.py": "pass"},
        eval_command="python eval.py",
        eval_timeout=None,
        artifacts=artifacts,
        save_logs=False,
    )

    # A raced-terminated 409 is an error outcome, NOT a completion.
    assert result.success is False
    assert result.status != "completed"
    assert result.reason == "http_409"
    # The UI must never be told the run completed.
    ui.on_complete.assert_not_called()
    # The error was surfaced to the user with the backend detail.
    ui.on_error.assert_called_once()
    assert "terminated" in ui.on_error.call_args.args[0].lower()
