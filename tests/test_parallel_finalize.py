"""_run_parallel_and_finalize: the exit-reconciliation edges.

The finalizer decides the whole session's exit from one status GET. These
pin: a transient failure of that GET must not fail a successful session
(retry), a persistent outage must fall back to the scheduler's own
quiescence verdict rather than inventing a failure, the out_of_credits
termination sweep must run even when every status read is down, and — the
contract that must NOT loosen — a stop race is never reported as
completion.
"""

from unittest.mock import MagicMock, patch

import weco.optimizer as optimizer_mod
from weco.optimizer import _run_parallel_and_finalize


def _finalize(outcome, client_factory, monkeypatch):
    monkeypatch.setattr(optimizer_mod.time, "sleep", lambda _s: None)
    with (
        patch("weco.parallel.run_parallel_lineage_loop", return_value=outcome),
        patch.object(optimizer_mod, "WecoClient", client_factory),
        patch.object(optimizer_mod, "report_termination") as mock_report,
    ):
        result = _run_parallel_and_finalize(
            lineage_id="lin-1",
            run_id="run-1",
            auth_headers={},
            slots=[],
            lineage_k=2,
            originals={},
            eval_command="python eval.py",
            eval_timeout=None,
            save_logs=False,
            log_dir=".runs",
            api_keys=None,
            submit_timeout=None,
            poll_interval=0.01,
        )
    return result, mock_report


def _client_factory(get_run_status_effects, lineage=None):
    """WecoClient stand-in: each construction shares the same call script."""
    state = {"calls": 0}

    def factory(_auth_headers):
        client = MagicMock()

        def get_run_status(run_id, include_history=False):
            effect = get_run_status_effects[min(state["calls"], len(get_run_status_effects) - 1)]
            state["calls"] += 1
            if isinstance(effect, Exception):
                raise effect
            return effect

        client.get_run_status = MagicMock(side_effect=get_run_status)
        if isinstance(lineage, Exception):
            client.get_lineage = MagicMock(side_effect=lineage)
        else:
            client.get_lineage = MagicMock(return_value=lineage or {"members": []})
        return client

    factory.state = state
    return factory


def test_transient_status_error_does_not_fail_successful_session(monkeypatch, capsys):
    factory = _client_factory([ConnectionError("502"), ConnectionError("502"), {"status": "completed"}])
    result, mock_report = _finalize("ok", factory, monkeypatch)
    assert result is True, "a successful session must survive transient status errors"
    assert factory.state["calls"] == 3
    mock_report.assert_not_called()


def test_persistent_status_outage_trusts_scheduler_quiescence(monkeypatch, capsys):
    factory = _client_factory([ConnectionError("down")])
    result, _ = _finalize("ok", factory, monkeypatch)
    assert result is True, "scheduler-confirmed quiescence must not be reported as failure"


def test_persistent_outage_with_non_ok_outcome_stays_failed(monkeypatch):
    factory = _client_factory([ConnectionError("down")])
    result, _ = _finalize("fatal", factory, monkeypatch)
    assert result is False


def test_out_of_credits_sweep_survives_status_outage(monkeypatch, capsys):
    """With status AND lineage reads down, the sweep must still report the
    root's termination (assumed running) — skipping it is what mislabels
    drained members heartbeat_timeout later."""
    factory = _client_factory([ConnectionError("down")], lineage=ConnectionError("down"))
    result, mock_report = _finalize("out_of_credits", factory, monkeypatch)
    assert result is False
    mock_report.assert_called_once()
    kwargs = mock_report.call_args.kwargs
    assert kwargs["run_id"] == "run-1"
    assert kwargs["reason"] == "out_of_credits"


def test_stop_race_never_reported_as_completion(monkeypatch):
    """The contract the retry must not loosen: scheduler says ok, but the
    authoritative status is terminated (stop race) -> failure exit."""
    factory = _client_factory([{"status": "terminated"}])
    result, _ = _finalize("ok", factory, monkeypatch)
    assert result is False
