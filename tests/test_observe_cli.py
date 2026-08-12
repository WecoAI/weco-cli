"""Tests for `weco observe` exit codes, error reporting, and retry behavior.

Exit-code policy under test (see weco/observe/cli.py): `init` fails hard on
any error; `log` fails hard only on caller errors and stays exit-0 on
weco-side failures unless --strict is passed.
"""

import argparse
from unittest.mock import Mock, patch

import pytest
import requests

from weco.observe import api
from weco.observe.cli import execute_observe_command


def _log_args(**overrides):
    args = argparse.Namespace(
        observe_command="log",
        run_id="run-123",
        step=1,
        status="completed",
        description=None,
        metrics='{"val_bpb": 0.5}',
        source=None,
        sources=None,
        parent_step=None,
        strict=False,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _init_args(tmp_path, **overrides):
    source = tmp_path / "train.py"
    source.write_text("print('hi')")
    args = argparse.Namespace(
        observe_command="init",
        name="test",
        metric="val_bpb",
        goal="min",
        source=str(source),
        sources=None,
        additional_instructions=None,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


@pytest.fixture
def authed():
    with patch("weco.observe.cli.load_weco_api_key", return_value="wk-test"):
        with patch("weco.observe.cli.send_event"):
            yield


def _response(status=201, json_body=None, headers=None):
    response = Mock(status_code=status, headers=headers or {})
    response.json.return_value = json_body if json_body is not None else {"node_id": "n1"}
    if status >= 400:
        response.raise_for_status.side_effect = requests.HTTPError(f"{status} Error", response=response)
    else:
        response.raise_for_status.return_value = None
    return response


class TestLogExitCodes:
    def test_caller_error_exits_nonzero(self, authed, capsys):
        with patch("weco.observe.cli.api.log_step", side_effect=api.CallerError("failed to log step 1: bad request")):
            with pytest.raises(SystemExit) as exc:
                execute_observe_command(_log_args())
        assert exc.value.code == 1
        assert "failed to log step 1" in capsys.readouterr().err

    def test_transient_error_exits_zero(self, authed, capsys):
        """A weco-side blip must not crash the loop being tracked."""
        with patch("weco.observe.cli.api.log_step", side_effect=api.TransientError("failed to log step 1: HTTP 503")):
            execute_observe_command(_log_args())  # returns normally — no SystemExit
        err = capsys.readouterr().err
        assert "HTTP 503" in err
        assert "dropped" in err

    def test_transient_error_with_strict_exits_nonzero(self, authed):
        with patch("weco.observe.cli.api.log_step", side_effect=api.TransientError("failed to log step 1: HTTP 503")):
            with pytest.raises(SystemExit) as exc:
                execute_observe_command(_log_args(strict=True))
        assert exc.value.code == 1

    def test_success_does_not_exit(self, authed):
        with patch("weco.observe.cli.api.log_step", return_value={"node_id": "n1", "step": 1, "status": "completed"}):
            execute_observe_command(_log_args())  # returns normally

    def test_invalid_metrics_json_exits_nonzero(self, authed, capsys):
        with pytest.raises(SystemExit) as exc:
            execute_observe_command(_log_args(metrics="{not json"))
        assert exc.value.code == 1
        assert "invalid metrics JSON" in capsys.readouterr().err

    def test_unreadable_source_exits_nonzero(self, authed, capsys):
        with pytest.raises(SystemExit) as exc:
            execute_observe_command(_log_args(source="/nonexistent/train.py"))
        assert exc.value.code == 1
        assert "cannot read /nonexistent/train.py" in capsys.readouterr().err

    def test_non_utf8_source_exits_cleanly(self, authed, tmp_path, capsys):
        """UnicodeDecodeError gets the same clean message as OSError, not a traceback."""
        binary = tmp_path / "model.ckpt"
        binary.write_bytes(b"\xff\xfe\x00\x01")
        with pytest.raises(SystemExit) as exc:
            execute_observe_command(_log_args(source=str(binary)))
        assert exc.value.code == 1
        assert f"cannot read {binary}" in capsys.readouterr().err


class TestInitExitCodes:
    """init runs before the tracked loop exists, so any failure is fatal."""

    def test_transient_failure_exits_nonzero_with_empty_stdout(self, authed, tmp_path, capsys):
        """RUN_ID=$(weco observe init ...) must never see exit 0 with no run id."""
        with patch("weco.observe.cli.api.create_run", side_effect=api.TransientError("failed to create run: HTTP 503")):
            with pytest.raises(SystemExit) as exc:
                execute_observe_command(_init_args(tmp_path))
        assert exc.value.code == 1
        assert capsys.readouterr().out == ""

    def test_caller_error_exits_nonzero(self, authed, tmp_path):
        with patch("weco.observe.cli.api.create_run", side_effect=api.CallerError("failed to create run: forbidden")):
            with pytest.raises(SystemExit) as exc:
                execute_observe_command(_init_args(tmp_path))
        assert exc.value.code == 1

    def test_missing_run_id_in_response_exits_nonzero(self, authed, tmp_path, capsys):
        with patch("weco.observe.cli.api.create_run", return_value={"status": "running"}):
            with pytest.raises(SystemExit) as exc:
                execute_observe_command(_init_args(tmp_path))
        assert exc.value.code == 1
        assert capsys.readouterr().out == ""

    def test_success_prints_run_id(self, authed, tmp_path, capsys):
        with patch("weco.observe.cli.api.create_run", return_value={"run_id": "run-123", "status": "running"}):
            with patch("weco.observe.cli.open_browser"):
                execute_observe_command(_init_args(tmp_path))
        assert capsys.readouterr().out.strip() == "run-123"


class TestAuthExitCodes:
    def test_not_logged_in_exits_nonzero(self, capsys):
        with patch("weco.observe.cli.load_weco_api_key", return_value=None):
            with pytest.raises(SystemExit) as exc:
                execute_observe_command(_log_args())
        assert exc.value.code == 1
        assert "not logged in" in capsys.readouterr().err

    def test_missing_subcommand_exits_nonzero(self):
        with pytest.raises(SystemExit) as exc:
            execute_observe_command(argparse.Namespace(observe_command=None))
        assert exc.value.code == 2


class TestApiRetries:
    """Transient failures are retried with backoff before being raised."""

    def test_persistent_5xx_raises_transient_after_retries(self):
        with patch("weco.observe.api.requests.post", return_value=_response(503)) as post:
            with patch("weco.observe.api.time.sleep"):
                with pytest.raises(api.TransientError, match="HTTP 503"):
                    api.log_step(run_id="run-123", step=1, auth_headers={})
        assert post.call_count == 2  # log stays cheap in the hot path: one retry

    def test_timeout_then_success_returns_dict(self):
        ok = _response(201, {"node_id": "n1"})
        with patch("weco.observe.api.requests.post", side_effect=[requests.Timeout("timed out"), ok]):
            with patch("weco.observe.api.time.sleep"):
                result = api.log_step(run_id="run-123", step=1, auth_headers={})
        assert result == {"node_id": "n1"}

    def test_429_honors_capped_retry_after(self):
        throttled = _response(429, headers={"Retry-After": "60"})
        ok = _response(201, {"node_id": "n1"})
        with patch("weco.observe.api.requests.post", side_effect=[throttled, ok]):
            with patch("weco.observe.api.time.sleep") as sleep:
                result = api.log_step(run_id="run-123", step=1, auth_headers={})
        assert result == {"node_id": "n1"}
        sleep.assert_called_once_with(10.0)  # Retry-After capped, not honored verbatim

    def test_4xx_raises_caller_error_without_retry(self):
        response = _response(422, {"detail": "bad metrics"})
        with patch("weco.observe.api.requests.post", return_value=response) as post:
            with pytest.raises(api.CallerError, match="bad metrics"):
                api.log_step(run_id="run-123", step=1, auth_headers={})
        assert post.call_count == 1

    def test_2xx_with_non_json_body_is_transient(self):
        """A proxy/LB interstitial (200 + HTML) is weco-side, not the caller's."""
        response = _response(200)
        response.json.side_effect = ValueError("Expecting value")
        with patch("weco.observe.api.requests.post", return_value=response):
            with patch("weco.observe.api.time.sleep"):
                with pytest.raises(api.TransientError):
                    api.log_step(run_id="run-123", step=1, auth_headers={})

    def test_2xx_with_non_dict_json_body_is_transient(self):
        """Valid JSON of the wrong shape (list/scalar) must not escape as a traceback."""
        response = _response(200, json_body=["not", "a", "dict"])
        with patch("weco.observe.api.requests.post", return_value=response):
            with patch("weco.observe.api.time.sleep"):
                with pytest.raises(api.TransientError, match="unexpected response body"):
                    api.log_step(run_id="run-123", step=1, auth_headers={})

    def test_retryable_status_surfaces_api_detail(self):
        response = _response(503, {"detail": "Unable to log step. Please try again."})
        with patch("weco.observe.api.requests.post", return_value=response):
            with patch("weco.observe.api.time.sleep"):
                with pytest.raises(api.TransientError, match="Unable to log step"):
                    api.log_step(run_id="run-123", step=1, auth_headers={})

    def test_unknown_exception_is_transient(self):
        """Unclassifiable failures default to transient — the cheaper mistake."""
        with patch("weco.observe.api.requests.post", side_effect=RuntimeError("boom")):
            with patch("weco.observe.api.time.sleep"):
                with pytest.raises(api.TransientError, match="boom"):
                    api.log_step(run_id="run-123", step=1, auth_headers={})

    def test_create_run_gets_extra_attempt(self):
        """init is pre-loop, so it absorbs a longer blip than log does."""
        with patch("weco.observe.api.requests.post", return_value=_response(503)) as post:
            with patch("weco.observe.api.time.sleep"):
                with pytest.raises(api.TransientError):
                    api.create_run(source_code={"a.py": "x"}, metric_name="m", maximize=False, auth_headers={})
        assert post.call_count == 3


class TestObserverSdkNeverRaises:
    """The WecoObserver SDK keeps its never-raise contract over the new exceptions."""

    def _observer(self):
        from weco.observe.observer import WecoObserver

        with patch("weco.observe.observer.load_weco_api_key", return_value="wk-test"):
            return WecoObserver()

    def test_create_run_failure_warns_and_returns_none(self):
        obs = self._observer()
        with patch("weco.observe.observer.api.create_run", side_effect=api.TransientError("HTTP 503")):
            with pytest.warns(UserWarning, match="HTTP 503"):
                result = obs.create_run(source_code={"a.py": "x"}, primary_metric="m")
        assert result is None

    def test_create_run_missing_run_id_warns_and_returns_none(self):
        obs = self._observer()
        with patch("weco.observe.observer.api.create_run", return_value={"status": "running"}):
            with pytest.warns(UserWarning, match="no run_id"):
                result = obs.create_run(source_code={"a.py": "x"}, primary_metric="m")
        assert result is None

    def test_log_step_failure_warns_and_does_not_raise(self):
        from weco.observe.observer import ObserveRun

        run = ObserveRun(run_id="run-123", auth_headers={})
        with patch("weco.observe.observer.api.log_step", side_effect=api.CallerError("bad request")):
            with pytest.warns(UserWarning, match="bad request"):
                run.log_step(step=1, metrics={"m": 0.5})


class TestReadCodeFiles:
    def test_partial_read_failure_is_fatal(self, authed, tmp_path, capsys):
        """One unreadable file fails the command — no silent partial snapshots."""
        ok = tmp_path / "ok.py"
        ok.write_text("x = 1")
        with pytest.raises(SystemExit) as exc:
            execute_observe_command(_log_args(source=None, sources=[str(ok), str(tmp_path / "missing.py")]))
        assert exc.value.code == 1
        assert "missing.py" in capsys.readouterr().err
