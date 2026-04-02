"""Unit tests for the ``weco run derive`` command handler.

Mocks every external boundary the handler touches (auth, HTTP client,
optimization loop, heartbeat thread, browser, artifacts, prompts, UI
classes) so the handler can be exercised end-to-end with no network, no
auth, no LLM, and no real threads.

Tests are organized by *what they verify*, not by historical bug reference.
``_resolve_step_to_node_id`` is unit-tested directly; the rest exercise
``derive.handle`` end-to-end via the ``patched`` fixture.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import requests
from rich.console import Console

from weco.commands.run import derive
from weco.optimizer import OptimizationResult


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


_FAKE_DERIVED_FROM = {
    "run_id": "parent-id",
    "node_id": "n1",
    "step": 1,
    "metric_value": 0.7,
}


def _fake_derive_response(
    *, candidate_code: dict[str, str] | None = None, **run_overrides
) -> dict:
    """Build a canonical successful derive response.

    Override any field in the inner ``run`` dict via kwargs (e.g.
    ``_fake_derive_response(eval_timeout=900)``). Override the candidate
    code via the explicit ``candidate_code`` kwarg.
    """
    run = {
        "id": "new-run-id",
        "name": "derived-test",
        "status": "running",
        "lineage_id": "parent-id",
        "derived_from": _FAKE_DERIVED_FROM,
        # Loop-config fields the CLI consumes:
        "evaluation_command": "python test.py",
        "metric_name": "score",
        "maximize": True,
        "steps": 12,
        "model": "gpt-4",
        "eval_timeout": 600,
        "save_logs": True,
        "log_dir": ".weco-runs",
        "source_code": {"main.py": "INHERITED"},
    }
    run.update(run_overrides)
    return {
        "run": run,
        "candidate": {
            "run_id": run["id"],
            "run_name": run["name"],
            "solution_id": "sol-1",
            "code": candidate_code or {"main.py": "CANDIDATE"},
            "plan": "try x",
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    }


# Default kwargs for derive.handle. Most tests don't care about presentation
# so plain mode is the default — it skips the interactive Confirm prompt and
# emits machine-readable error JSON, both of which simplify assertions.
_DEFAULT_CALL_KWARGS = dict(
    run_id="parent-id",
    from_step="best",
    steps=None,
    additional_instructions=None,
    api_keys=None,
    output_mode="plain",
    console=None,  # filled in per-call so each test gets a fresh Console
)


def _call_handle(**overrides):
    """Invoke ``derive.handle`` with sane defaults; overrides win."""
    kwargs = _DEFAULT_CALL_KWARGS | {"console": Console()} | overrides
    return derive.handle(**kwargs)


@pytest.fixture
def patched():
    """Patch every external boundary derive.handle touches.

    Yields a ``SimpleNamespace`` of the most useful mock handles. Both UI
    classes are configured so that ``with ui_instance as ui:`` yields the
    same mock instance — that way ``ui.on_init(...)`` lands directly on
    ``patched.plain_ui.on_init`` (or ``live_ui.on_init``) for inspection.
    """
    with (
        patch(
            "weco.commands.run.derive.handle_authentication",
            return_value=("api-key", {"Authorization": "Bearer t"}),
        ),
        patch("weco.commands.run.derive.WecoClient") as MockClient,
        patch("weco.commands.run.derive.run_optimization_loop") as mock_loop,
        patch("weco.heartbeat.HeartbeatSender"),
        patch("weco.commands.run.derive.open_browser"),
        patch("weco.commands.run.derive.RunArtifacts") as mock_artifacts,
        patch("weco.commands.run.derive.report_termination"),
        patch("weco.commands.run.derive.offer_apply_best_solution") as mock_apply,
        patch("weco.commands.run.derive.Confirm") as MockConfirm,
        patch("weco.commands.run.derive.LiveOptimizationUI") as MockLiveUI,
        patch("weco.commands.run.derive.PlainOptimizationUI") as MockPlainUI,
    ):
        # Defaults: loop succeeds, prompts say yes.
        mock_loop.return_value = OptimizationResult(
            success=True, final_step=2, status="completed", reason="completed_successfully"
        )
        MockConfirm.ask.return_value = True

        # Make `with ui as x:` yield the outer mock so ui.on_init calls land
        # on the same object the test inspects.
        for mock_ui_cls in (MockLiveUI, MockPlainUI):
            mock_ui_cls.return_value.__enter__.return_value = mock_ui_cls.return_value

        client = MockClient.return_value
        client.derive_run.return_value = _fake_derive_response()

        yield SimpleNamespace(
            client=client,
            loop=mock_loop,
            artifacts=mock_artifacts,
            apply_best=mock_apply,
            confirm=MockConfirm,
            live_ui=MockLiveUI.return_value,
            plain_ui=MockPlainUI.return_value,
        )


def _loop_kwargs(patched) -> dict:
    """Pull the kwargs ``run_optimization_loop`` was called with."""
    return patched.loop.call_args.kwargs


# ---------------------------------------------------------------------------
# Step resolution helper: _resolve_step_to_node_id
# ---------------------------------------------------------------------------


def test_resolve_step_to_node_id_returns_first_node():
    fake_client = MagicMock()
    fake_client.list_nodes.return_value = {"nodes": [{"node_id": "node-uuid-7"}]}

    result = derive._resolve_step_to_node_id(fake_client, "run-id", step=7)

    assert result == "node-uuid-7"
    # include_code=False is a deliberate optimisation — the code blob is
    # never needed here, and dropping it saves backend work.
    fake_client.list_nodes.assert_called_once_with("run-id", step=7, include_code=False)


def test_resolve_step_to_node_id_raises_derive_error_when_no_node():
    fake_client = MagicMock()
    fake_client.list_nodes.return_value = {"nodes": []}

    with pytest.raises(derive.DeriveError, match="No node found at step 99"):
        derive._resolve_step_to_node_id(fake_client, "run-id", step=99)


# ---------------------------------------------------------------------------
# End-to-end --from-step wiring through the handler
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "from_step, expected_derive_from, expects_lookup",
    [
        # Both naming conventions for both "best" keywords (the alias dict's reason for being).
        ("best", "lineage_best", False),
        ("lineage-best", "lineage_best", False),
        ("lineage_best", "lineage_best", False),
        ("run-best", "run_best", False),
        ("run_best", "run_best", False),
        # Integer step numbers trigger a node lookup. Negatives parse fine — the
        # backend is responsible for rejecting nonsense step numbers.
        ("0", "node-at-step", True),
        ("3", "node-at-step", True),
        ("-3", "node-at-step", True),
        # Anything else passes through as a presumed node UUID; the backend 404s if bogus.
        ("550e8400-e29b-41d4-a716-446655440000", "550e8400-e29b-41d4-a716-446655440000", False),
        ("abc-def", "abc-def", False),
        ("1.5", "1.5", False),  # float-shaped: not an int
        ("", "", False),  # empty string passes through; backend will 404
    ],
    ids=[
        "lineage_best-bare",
        "lineage_best-hyphen",
        "lineage_best-underscore",
        "run_best-hyphen",
        "run_best-underscore",
        "step-zero",
        "step-positive",
        "step-negative",
        "node-uuid",
        "node-arbitrary",
        "node-float-shaped",
        "node-empty-string",
    ],
)
def test_from_step_wiring(patched, from_step, expected_derive_from, expects_lookup):
    """Each form of --from-step ends up calling derive_run with the right
    derive_from value, and only the integer forms trigger a list_nodes lookup."""
    patched.client.list_nodes.return_value = {"nodes": [{"node_id": "node-at-step"}]}

    _call_handle(from_step=from_step)

    assert patched.client.derive_run.call_args.kwargs["derive_from"] == expected_derive_from
    if expects_lookup:
        patched.client.list_nodes.assert_called_once()
    else:
        patched.client.list_nodes.assert_not_called()


# ---------------------------------------------------------------------------
# Loop configuration plumbing — values flow from response into the loop
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "loop_kwarg, expected",
    [
        ("eval_command", "python test.py"),
        ("save_logs", True),
        ("eval_timeout", 600),
    ],
)
def test_loop_config_inherited_from_response(patched, loop_kwarg, expected):
    """The loop receives its config from the derive response, not from
    defaults or a follow-up GET."""
    _call_handle()
    assert _loop_kwargs(patched)[loop_kwarg] == expected


def test_eval_timeout_none_passes_through(patched):
    """eval_timeout is genuinely Optional[int] — None must not be coerced."""
    patched.client.derive_run.return_value = _fake_derive_response(eval_timeout=None)
    _call_handle()
    assert _loop_kwargs(patched)["eval_timeout"] is None


def test_log_dir_inherited_from_response_into_artifacts(patched):
    """log_dir doesn't go to the loop directly — it goes to RunArtifacts."""
    _call_handle()
    patched.artifacts.assert_called_once()
    assert patched.artifacts.call_args.kwargs["log_dir"] == ".weco-runs"


def test_additional_instructions_flow_through_to_derive_run(patched):
    """The (resolved) --additional-instructions value reaches the backend.
    Locks in the rename from the older `direction` field name."""
    _call_handle(additional_instructions="focus on memory efficiency")
    assert (
        patched.client.derive_run.call_args.kwargs["additional_instructions"]
        == "focus on memory efficiency"
    )


# ---------------------------------------------------------------------------
# Source-code originals: file restoration baseline
# ---------------------------------------------------------------------------


def test_originals_use_inherited_source_when_local_files_missing(patched, tmp_path, monkeypatch):
    """If a file doesn't exist locally, the originals fed to the loop must
    come from the inherited baseline — NEVER the candidate, which would
    pollute the working directory with generated code on every eval cycle."""
    monkeypatch.chdir(tmp_path)  # empty directory: no local files
    patched.client.derive_run.return_value = _fake_derive_response(
        source_code={"main.py": "INHERITED"},
        candidate_code={"main.py": "CANDIDATE"},
    )

    _call_handle()

    source_code = _loop_kwargs(patched)["source_code"]
    assert source_code == {"main.py": "INHERITED"}


def test_originals_prefer_local_files_when_present(patched, tmp_path, monkeypatch):
    """When a file exists locally, prefer it over the inherited baseline —
    the user may have made local edits since the parent run."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "main.py").write_text("LOCAL_EDITED")
    patched.client.derive_run.return_value = _fake_derive_response(
        source_code={"main.py": "INHERITED"}
    )

    _call_handle()

    assert _loop_kwargs(patched)["source_code"] == {"main.py": "LOCAL_EDITED"}


# ---------------------------------------------------------------------------
# UI dispatch: handler picks the right UI and routes derived_from via on_init
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "output_mode, expected_ui, other_ui",
    [("plain", "plain_ui", "live_ui"), ("rich", "live_ui", "plain_ui")],
)
def test_handler_picks_ui_for_output_mode(patched, output_mode, expected_ui, other_ui):
    """Plain mode constructs PlainOptimizationUI, rich mode constructs
    LiveOptimizationUI. Each gets exactly one on_init call."""
    _call_handle(output_mode=output_mode)
    getattr(patched, expected_ui).on_init.assert_called_once()
    getattr(patched, other_ui).on_init.assert_not_called()


def test_on_init_receives_derived_from_payload(patched):
    """The parent reference flows through on_init's payload unchanged — not
    via a side-channel print or constructor argument.

    This is a *flow* test, not a value test: the assertion compares against
    the same fixture constant (``_FAKE_DERIVED_FROM``) deliberately, to
    verify pass-through of whatever the response contained. The specific
    field values are an implementation detail of the fixture.
    """
    _call_handle()
    assert patched.plain_ui.on_init.call_args.kwargs["derived_from"] == _FAKE_DERIVED_FROM


# ---------------------------------------------------------------------------
# Exit code dispatch: handler return value reflects loop result
# ---------------------------------------------------------------------------


def test_handler_returns_false_when_loop_fails(patched):
    """Failed loop must propagate as ``False`` — the dispatcher relies on
    this to set the process exit code. Loop *success* is covered implicitly
    by every other test in this file (the default fixture exercises it)."""
    patched.loop.return_value = OptimizationResult(
        success=False, final_step=3, status="error", reason="submit_failed"
    )
    assert _call_handle() is False


def test_handler_aborts_when_user_cancels_unchanged_files_prompt(patched):
    """Rich mode prompts the user to confirm working-directory state. If
    they say no, the handler aborts before starting the loop and returns
    False without ever calling run_optimization_loop.

    Verifies both that the prompt fired (so a future refactor that drops
    the prompt entirely fails this test) and that the abort happens.
    """
    patched.confirm.ask.return_value = False
    assert _call_handle(output_mode="rich") is False
    patched.confirm.ask.assert_called_once()
    patched.loop.assert_not_called()


def test_plain_mode_skips_unchanged_files_prompt(patched):
    """Agents in plain mode never see the prompt — Confirm.ask is never
    called. This is the positive form of the cancellation test above."""
    _call_handle(output_mode="plain")
    patched.confirm.ask.assert_not_called()


# ---------------------------------------------------------------------------
# Apply-best lifecycle: only after a successful loop
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "loop_success, expect_apply_called",
    [(True, True), (False, False)],
    ids=["success-applies-best", "failure-skips-apply"],
)
def test_apply_best_only_after_successful_loop(patched, loop_success, expect_apply_called):
    patched.loop.return_value = OptimizationResult(
        success=loop_success,
        final_step=2,
        status="completed" if loop_success else "error",
        reason="completed_successfully" if loop_success else "submit_failed",
    )
    _call_handle()
    if expect_apply_called:
        patched.apply_best.assert_called_once()
    else:
        patched.apply_best.assert_not_called()


# ---------------------------------------------------------------------------
# Error handling: HTTPError, network errors, DeriveError
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make_exception",
    [
        # Factories defer construction to test-run time so each parametrize
        # case gets a fresh exception/mock instance — avoiding any chance of
        # cross-test pollution from a shared MagicMock.
        lambda: requests.exceptions.HTTPError(response=MagicMock(status_code=400, text="bad request")),
        lambda: requests.exceptions.ConnectionError("net down"),
    ],
    ids=["http-error", "network-error"],
)
def test_api_errors_exit_cleanly_with_code_1(patched, make_exception):
    """Both HTTP errors and network errors land in distinct except branches
    in the handler — neither should crash inside an exception handler that
    assumes the wrong shape (e.g. AttributeError on ``e.response``)."""
    patched.client.derive_run.side_effect = make_exception()

    with pytest.raises(SystemExit) as exc:
        _call_handle(output_mode="rich")
    assert exc.value.code == 1


@pytest.mark.parametrize(
    "output_mode, expect_json_in_stderr",
    [("plain", True), ("rich", False)],
    ids=["plain", "rich"],
)
def test_derive_error_routed_through_correct_channel(patched, capsys, output_mode, expect_json_in_stderr):
    """A DeriveError reaches the user via stderr+JSON in plain mode and
    styled console output in rich mode. Plain mode in particular must never
    pollute stdout — that channel is reserved for normal output that
    pipelines (jq, NDJSON consumers) might be parsing.

    Asserts on the *structure* of the error envelope, not the exact wording
    of the error message — wording is allowed to evolve without breaking
    this test.
    """
    patched.client.list_nodes.return_value = {"nodes": []}

    with pytest.raises(SystemExit) as exc:
        _call_handle(output_mode=output_mode, from_step="42")

    assert exc.value.code == 1
    captured = capsys.readouterr()

    if expect_json_in_stderr:
        # Plain mode: stdout is clean, stderr contains a parseable JSON
        # error envelope with a non-empty message under the "error" key.
        assert captured.out == ""
        payload = json.loads(captured.err.strip())
        assert "error" in payload
        assert payload["error"]  # non-empty
    else:
        # Rich mode: error is rendered through console.print as styled text.
        # We don't pin a specific channel — rich is for humans, not pipes —
        # but neither stdout nor stderr should contain a raw JSON object.
        assert "{" not in captured.out
        assert "{" not in captured.err


# ---------------------------------------------------------------------------
# API call discipline: derive needs exactly one round-trip to /derive
# ---------------------------------------------------------------------------


def test_handler_makes_one_derive_call_and_no_get_run_status(patched):
    """The pre-fix handler did a follow-up GET /runs/{id} to fetch loop
    config. After the response was enriched with that config, that second
    call is gone. This test guards against re-introducing it."""
    _call_handle()
    patched.client.derive_run.assert_called_once()
    patched.client.get_run_status.assert_not_called()
