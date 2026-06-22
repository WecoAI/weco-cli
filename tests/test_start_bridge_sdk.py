"""Tests for the SDK-driven `weco start claude` bridge.

Covers the parts we own: arg parsing, billing env construction, SDK message
→ transcript envelope conversion, and the in-process approval router. The
actual claude-agent-sdk call is faked so no network or claude binary is
required.
"""

from __future__ import annotations

import asyncio
import io
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from rich.console import Console

from weco.commands.start import tui_bridge as bridge, sdk_config, envelopes, approval_router


# --- Arg parsing ---------------------------------------------------------------


def test_parse_claude_args_extracts_model_and_effort():
    parsed, extra = sdk_config.parse_claude_args(["--model", "opus", "--effort", "high"])
    assert parsed == {"model": "opus", "effort": "high"}
    assert extra == {}


def test_parse_claude_args_pulls_dangerously_skip_permissions_flag():
    parsed, _ = sdk_config.parse_claude_args(["--dangerously-skip-permissions"])
    assert parsed.get("dangerously_skip_permissions") is True


def test_parse_claude_args_forwards_unknown_flags_via_extra_args():
    parsed, extra = sdk_config.parse_claude_args(["--add-dir", "/tmp/foo", "--debug", "--betas", "context-1m-2025-08-07"])
    assert "add-dir" in extra
    assert extra["add-dir"] == "/tmp/foo"
    assert "debug" in extra  # bare flag → None
    assert extra["debug"] is None
    assert extra["betas"] == "context-1m-2025-08-07"
    # Recognised structured fields are untouched here.
    assert parsed == {}


def test_parse_claude_args_handles_equals_form():
    _, extra = sdk_config.parse_claude_args(["--add-dir=/tmp/x"])
    assert extra == {"add-dir": "/tmp/x"}


# --- Billing env --------------------------------------------------------------


def test_build_sdk_env_weco_billing_encodes_session_id_in_base_url():
    """The Anthropic SDK doesn't honor any "custom headers" env var, so we
    smuggle the agent session id through the BASE_URL path itself. The
    proxy reads it from `/anthropic/s/<id>/v1/messages` and uses it to
    broadcast credits_updated events to the right channel."""
    env = sdk_config.build_sdk_env(
        billing="weco", api_key="weco-key-abc", weco_api_base="https://api.weco.ai/v1", session_id="sess-123"
    )
    assert env["ANTHROPIC_BASE_URL"] == "https://api.weco.ai/v1/llm/anthropic/s/sess-123"
    assert env["ANTHROPIC_API_KEY"] == "weco-key-abc"


def test_build_sdk_env_weco_billing_without_session_uses_anonymous_route():
    """If we don't have a session id (channel setup failed, etc.) the
    base URL falls back to the anonymous variant — calls still get billed,
    they just won't trigger credits_updated broadcasts."""
    env = sdk_config.build_sdk_env(
        billing="weco", api_key="weco-key-abc", weco_api_base="https://api.weco.ai/v1", session_id=None
    )
    assert env["ANTHROPIC_BASE_URL"] == "https://api.weco.ai/v1/llm/anthropic"


def test_build_sdk_env_claude_billing_leaves_anthropic_env_alone():
    env = sdk_config.build_sdk_env(
        billing="claude", api_key="weco-key-abc", weco_api_base="https://api.weco.ai/v1", session_id="sess-123"
    )
    # We must not have set anything anthropic-related when the user is on
    # local OAuth or BYO API key — those flows would be hijacked otherwise.
    assert "weco" not in (env.get("ANTHROPIC_BASE_URL") or "")


def test_build_sdk_env_strips_trailing_slash_from_base():
    env = sdk_config.build_sdk_env(billing="weco", api_key="k", weco_api_base="https://api.weco.ai/v1/", session_id="s")
    assert env["ANTHROPIC_BASE_URL"] == "https://api.weco.ai/v1/llm/anthropic/s/s"


# --- Envelope conversion (SDK message → claude stream-json shape) ------------


def test_envelope_for_assistant_message_emits_assistant_event():
    from claude_agent_sdk.types import AssistantMessage, TextBlock, ToolUseBlock

    msg = AssistantMessage(
        content=[TextBlock(text="Hello!"), ToolUseBlock(id="tu_1", name="Bash", input={"command": "ls"})],
        model="claude-opus-4-7",
        parent_tool_use_id=None,
        error=None,
        usage=None,
        message_id="m1",
        stop_reason=None,
        session_id="s1",
        uuid=None,
    )

    env = envelopes.envelope_for(msg)

    assert env is not None
    assert env["type"] == "assistant"
    assert env["model"] == "claude-opus-4-7"
    content = env["message"]["content"]
    assert content[0] == {"type": "text", "text": "Hello!"}
    assert content[1]["type"] == "tool_use"
    assert content[1]["name"] == "Bash"
    assert content[1]["input"] == {"command": "ls"}


def test_envelope_for_includes_thinking_blocks_for_reasoning_dashboard():
    from claude_agent_sdk.types import AssistantMessage, ThinkingBlock

    msg = AssistantMessage(
        content=[ThinkingBlock(thinking="reasoning text", signature="sig")],
        model="claude-opus-4-7",
        parent_tool_use_id=None,
        error=None,
        usage=None,
        message_id="m1",
        stop_reason=None,
        session_id="s1",
        uuid=None,
    )
    env = envelopes.envelope_for(msg)
    assert env["message"]["content"][0]["type"] == "thinking"
    assert env["message"]["content"][0]["thinking"] == "reasoning text"


def test_envelope_for_result_message_emits_result_event_with_cost():
    from claude_agent_sdk.types import ResultMessage

    msg = ResultMessage(
        subtype="success",
        duration_ms=1234,
        duration_api_ms=1100,
        is_error=False,
        num_turns=1,
        session_id="s1",
        stop_reason="end_turn",
        total_cost_usd=0.04,
        usage={"input_tokens": 100},
        result="ok",
        structured_output=None,
        model_usage=None,
        permission_denials=None,
        deferred_tool_use=None,
        errors=None,
        api_error_status=None,
        uuid=None,
    )
    env = envelopes.envelope_for(msg)
    assert env["type"] == "result"
    assert env["total_cost_usd"] == 0.04
    assert env["duration_ms"] == 1234


def test_envelope_for_unknown_message_type_returns_none():
    assert envelopes.envelope_for(SimpleNamespace(type="weird")) is None


# --- Approval router -----------------------------------------------------------


def _queue_publish(q: "asyncio.Queue"):
    """A publish fn that drains into a queue so tests can read what the router
    broadcast. Returns True (published) like a real, non-full session would."""

    def publish(line: str) -> bool:
        q.put_nowait(line)
        return True

    return publish


def test_approval_router_forwards_request_and_awaits_response():
    asyncio.run(_approval_router_forwards_request_and_awaits_response())


async def _approval_router_forwards_request_and_awaits_response():
    outbound: asyncio.Queue = asyncio.Queue()
    router = approval_router.ApprovalRouter(publish=_queue_publish(outbound))

    ctx = SimpleNamespace(tool_use_id="tu_42")

    async def caller():
        return await router.can_use_tool("Bash", {"command": "ls"}, ctx)

    task = asyncio.create_task(caller())
    # The router should have published an approval_request envelope.
    raw = await asyncio.wait_for(outbound.get(), timeout=1.0)
    env = json.loads(raw)
    assert env["type"] == "_weco_meta"
    assert env["event"] == "approval_request"
    assert env["id"] == "tu_42"
    assert env["tool_name"] == "Bash"
    assert env["summary"] == "ls"

    # Simulate the dashboard responding.
    router.resolve("tu_42", "approve", "once")
    result = await asyncio.wait_for(task, timeout=1.0)
    from claude_agent_sdk.types import PermissionResultAllow

    assert isinstance(result, PermissionResultAllow)


def test_approval_router_caches_tool_scope_approvals():
    asyncio.run(_approval_router_caches_tool_scope_approvals())


async def _approval_router_caches_tool_scope_approvals():
    outbound: asyncio.Queue = asyncio.Queue()
    router = approval_router.ApprovalRouter(publish=_queue_publish(outbound))
    ctx = SimpleNamespace(tool_use_id="tu_1")

    task = asyncio.create_task(router.can_use_tool("Read", {"file_path": "a.py"}, ctx))
    await outbound.get()
    router.resolve("tu_1", "approve", "tool")
    await task

    # Second call to the same tool with a different file → no new dashboard
    # request, immediate allow from the cache.
    ctx2 = SimpleNamespace(tool_use_id="tu_2")
    result = await router.can_use_tool("Read", {"file_path": "b.py"}, ctx2)
    from claude_agent_sdk.types import PermissionResultAllow

    assert isinstance(result, PermissionResultAllow)
    # No additional outbound message was produced.
    assert outbound.empty()


def test_approval_router_deny_returns_deny_result():
    asyncio.run(_approval_router_deny_returns_deny_result())


async def _approval_router_deny_returns_deny_result():
    outbound: asyncio.Queue = asyncio.Queue()
    router = approval_router.ApprovalRouter(publish=_queue_publish(outbound))
    ctx = SimpleNamespace(tool_use_id="tu_99")

    task = asyncio.create_task(router.can_use_tool("Bash", {"command": "rm -rf /"}, ctx))
    await outbound.get()
    router.resolve("tu_99", "deny", "once")
    result = await task
    from claude_agent_sdk.types import PermissionResultDeny

    assert isinstance(result, PermissionResultDeny)


def test_approval_router_command_scope_caches_exact_call():
    asyncio.run(_approval_router_command_scope_caches_exact_call())


async def _approval_router_command_scope_caches_exact_call():
    outbound: asyncio.Queue = asyncio.Queue()
    router = approval_router.ApprovalRouter(publish=_queue_publish(outbound))
    ctx = SimpleNamespace(tool_use_id="tu_a")

    task = asyncio.create_task(router.can_use_tool("Bash", {"command": "git status"}, ctx))
    await outbound.get()
    router.resolve("tu_a", "approve", "command")
    await task

    # Same exact command — cached.
    result = await router.can_use_tool("Bash", {"command": "git status"}, SimpleNamespace(tool_use_id="tu_b"))
    from claude_agent_sdk.types import PermissionResultAllow

    assert isinstance(result, PermissionResultAllow)
    assert outbound.empty()

    # Different command — must round-trip.
    task2 = asyncio.create_task(router.can_use_tool("Bash", {"command": "git log"}, SimpleNamespace(tool_use_id="tu_c")))
    raw = await asyncio.wait_for(outbound.get(), timeout=1.0)
    assert json.loads(raw)["event"] == "approval_request"
    router.resolve("tu_c", "approve", "once")
    await task2


# --- AskUserQuestion routing ---------------------------------------------------


def test_ask_user_question_forwards_request_and_returns_answers():
    asyncio.run(_ask_user_question_forwards())


async def _ask_user_question_forwards():
    outbound: asyncio.Queue = asyncio.Queue()
    router = approval_router.ApprovalRouter(publish=_queue_publish(outbound))
    ctx = SimpleNamespace(tool_use_id="q_1")

    questions = [
        {
            "question": "Which DB?",
            "header": "DB",
            "options": [{"label": "Postgres", "description": "Default"}, {"label": "SQLite", "description": "Local dev"}],
            "multiSelect": False,
        }
    ]

    task = asyncio.create_task(router.can_use_tool("AskUserQuestion", {"questions": questions}, ctx))

    # The router should have published a question_request envelope, not the
    # approval_request shape used for tool gating.
    raw = await asyncio.wait_for(outbound.get(), timeout=1.0)
    env = json.loads(raw)
    assert env["event"] == "question_request"
    assert env["id"] == "q_1"
    assert env["questions"] == questions

    # Simulate the dashboard returning the user's pick.
    router.resolve_question("q_1", {"Which DB?": "Postgres"})

    result = await asyncio.wait_for(task, timeout=1.0)
    from claude_agent_sdk.types import PermissionResultAllow

    assert isinstance(result, PermissionResultAllow)
    # The SDK contract requires `updated_input` to carry both the original
    # questions array and the answers map so claude can synthesise the
    # tool result correctly.
    assert result.updated_input == {"questions": questions, "answers": {"Which DB?": "Postgres"}}


def test_ask_user_question_with_empty_questions_short_circuits():
    asyncio.run(_ask_user_question_empty())


async def _ask_user_question_empty():
    outbound: asyncio.Queue = asyncio.Queue()
    router = approval_router.ApprovalRouter(publish=_queue_publish(outbound))
    ctx = SimpleNamespace(tool_use_id="q_e")

    result = await router.can_use_tool("AskUserQuestion", {"questions": []}, ctx)
    from claude_agent_sdk.types import PermissionResultAllow

    assert isinstance(result, PermissionResultAllow)
    assert result.updated_input["answers"] == {}
    # No outbound message — nothing to ask.
    assert outbound.empty()


# --- Stream-keep-open hook ------------------------------------------------------


def test_keep_stream_open_hook_returns_continue():
    """The PreToolUse no-op hook is required so the SDK keeps the input
    channel open long enough for can_use_tool to fire — documented in
    the Agent SDK 'Handle approvals and user input' page."""
    result = asyncio.run(sdk_config.keep_stream_open_hook({}, "tu_1", None))
    assert result == {"continue_": True}


# --- Run-id scanning -----------------------------------------------------------


def test_scan_for_run_ids_picks_id_out_of_tool_result_text():
    from claude_agent_sdk.types import ToolResultBlock, UserMessage

    rw = MagicMock()
    msg = UserMessage(
        content=[
            ToolResultBlock(
                tool_use_id="tu_1", content="Run ID: a14ca1c1-56a7-4ae6-b054-51741adfbee5\nRun Name: foo", is_error=False
            )
        ],
        parent_tool_use_id=None,
        tool_use_result=None,
        uuid=None,
    )
    bridge.scan_for_run_ids(msg, rw)
    rw.watch.assert_called_with("a14ca1c1-56a7-4ae6-b054-51741adfbee5")


def test_scan_for_run_ids_ignores_non_user_messages():
    from claude_agent_sdk.types import AssistantMessage, TextBlock

    rw = MagicMock()
    msg = AssistantMessage(
        content=[TextBlock(text="Run ID: aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")],
        model="m",
        parent_tool_use_id=None,
        error=None,
        usage=None,
        message_id=None,
        stop_reason=None,
        session_id=None,
        uuid=None,
    )
    bridge.scan_for_run_ids(msg, rw)
    rw.watch.assert_not_called()


# --- peek_model ----------------------------------------------------------------


def test_peek_model_reads_space_separated_form():
    assert sdk_config.peek_model(["--model", "opus", "--debug"]) == "opus"


def test_peek_model_reads_equals_form():
    assert sdk_config.peek_model(["--model=sonnet"]) == "sonnet"


def test_peek_model_returns_none_when_absent():
    assert sdk_config.peek_model(["--debug", "--add-dir", "/tmp"]) is None


# --- summarise -----------------------------------------------------------------


def test_summarise_prefers_command_then_truncates():
    assert envelopes.summarise("Bash", {"command": "ls -la"}) == "ls -la"
    long = "x" * 500
    out = envelopes.summarise("Bash", {"command": long})
    assert out.endswith("…") and len(out) == 198  # 197 chars + ellipsis


def test_summarise_non_dict_input_returns_tool_name():
    assert envelopes.summarise("Bash", None) == "Bash"


# --- render_ui_context_preamble ------------------------------------------------


def test_render_ui_context_preamble_empty_when_no_context():
    assert envelopes.render_ui_context_preamble(None) == ""
    assert envelopes.render_ui_context_preamble({}) == ""


def test_render_ui_context_preamble_wraps_fields_in_system_reminder():
    out = envelopes.render_ui_context_preamble({"run_id": "r1", "step": 3, "best_metric": 7.5})
    assert out.startswith("<system-reminder>")
    assert out.rstrip().endswith("</system-reminder>")
    assert "Run ID: r1" in out
    assert "Active step: 3" in out
    assert "Best metric: 7.5" in out


# --- Approval card mount-failure fallback --------------------------------------


def test_on_approval_request_denies_once_when_card_mount_fails():
    """If mounting the inline card raises, the SDK approval must still
    resolve (deny-once) instead of hanging forever on an unmounted card."""
    pytest.importorskip("textual")
    asyncio.run(_on_approval_request_denies_on_mount_failure())


async def _on_approval_request_denies_on_mount_failure():
    from weco.commands.start.tui_bridge import Orchestrator
    from weco.commands.start.session import DashboardSession

    class BrokenApp:
        def mount_inline_card(self, card):
            raise RuntimeError("no terminal")

    orch = Orchestrator(
        app=BrokenApp(),
        claude_args=[],
        api_key="weco-k",
        session=DashboardSession.offline(),
        billing="weco",
        weco_api_base=None,
        effort=None,
    )
    router = approval_router.ApprovalRouter(publish=orch._session.publish)
    orch._approval_router = router

    # Stand in for the SDK's awaited approval future.
    sdk_future = asyncio.get_running_loop().create_future()
    router._pending_approvals["rid"] = sdk_future

    orch._on_approval_request("rid", "Bash", "ls", {"command": "ls"})
    decided = await asyncio.wait_for(sdk_future, timeout=1.0)
    assert decided == {"decision": "deny", "scope": "once"}


# --- Turn-stream drain (off-by-one recovery) -----------------------------------


def _fake_result_message():
    from claude_agent_sdk.types import ResultMessage

    return ResultMessage(
        subtype="success",
        duration_ms=1,
        duration_api_ms=1,
        is_error=False,
        num_turns=1,
        session_id="s",
        stop_reason="end_turn",
        total_cost_usd=0.0,
        usage={},
        result="",
        structured_output=None,
        model_usage=None,
        permission_denials=None,
        deferred_tool_use=None,
        errors=None,
        api_error_status=None,
        uuid=None,
    )


def _fake_orchestrator(**overrides):
    """Minimal Orchestrator stand-in for exercising `_fan_out`/`_consume_messages`
    in isolation. Returns (fake, published, rendered)."""
    published: list = []
    rendered: list = []
    fields = {
        "_session": SimpleNamespace(publish=published.append),
        "_renderer": SimpleNamespace(render=rendered.append),
        "_run_watcher": None,
        "_interrupting": False,
        "_inflight_turns": 0,
        **overrides,
    }
    return SimpleNamespace(**fields), published, rendered


def test_consume_keeps_draining_past_result_message():
    # A ResultMessage ends a turn, but the continuous consumer must keep going —
    # a message produced after it (an autonomous/between-turn one) must surface
    # immediately, not wait for the next prompt. This is the bug the rewrite
    # fixes: the old per-turn receive_response() stopped at ResultMessage, so
    # between-turn output stranded in the transport until the next query().
    seen = []
    rmsg = _fake_result_message()
    after = SimpleNamespace(type="autonomous-after")
    items = [SimpleNamespace(type="reply"), rmsg, after]

    class FakeClient:
        def receive_messages(self):
            async def gen():
                for it in items:
                    yield it

            return gen()

    fake = SimpleNamespace(_sdk_client=FakeClient(), _stop_event=asyncio.Event(), _fan_out=seen.append)
    asyncio.run(bridge.Orchestrator._consume_messages(fake))
    assert seen == items  # every message fanned out, including the one after the ResultMessage
    assert fake._stop_event.is_set()  # stream end brings the session down so the pump unblocks


def test_fan_out_mirrors_and_renders_result():
    fake, published, rendered = _fake_orchestrator(_inflight_turns=1)
    rmsg = _fake_result_message()
    bridge.Orchestrator._fan_out(fake, rmsg)
    assert fake._inflight_turns == 0  # turn settled
    assert rendered == [rmsg]  # shown locally
    assert len(published) == 1  # mirrored to dashboard


def test_fan_out_swallows_interrupted_result_and_clears_flag():
    fake, published, rendered = _fake_orchestrator(_inflight_turns=1, _interrupting=True)
    bridge.Orchestrator._fan_out(fake, _fake_result_message())
    assert fake._inflight_turns == 0  # still settles the count
    assert fake._interrupting is False  # cleared on the interrupted turn's terminal message
    assert published == []  # not mirrored — turn_interrupted already signalled the end
    assert rendered == []  # not rendered as an error bubble


# --- HeadlessUI (no-TTY stand-in) ---------------------------------------------


def test_headless_ui_unknown_methods_are_noops():
    ui = bridge.HeadlessUI(Console(file=io.StringIO()))
    # The full WecoTUI visual surface the Renderer/Orchestrator may call.
    ui.show_thinking()
    ui.hide_thinking()
    ui.post_assistant_delta("hi")
    ui.end_assistant_block()
    ui.post_tool_result("id", "out", is_error=True)
    ui.exit()
    assert callable(ui.some_method_added_later)  # __getattr__ keeps it forward-compatible


def test_headless_ui_lifecycle_lines_reach_the_console():
    buf = io.StringIO()
    ui = bridge.HeadlessUI(Console(file=buf, width=200))
    ui.post_banner(agent="Claude Code", model="opus", billing="weco", session_id="s1")
    ui.post_tool_use("t1", "Bash", "weco run ...", {})
    ui.post_run_update({"run_id": "abcdef123456", "status": "running", "step": 3, "total_steps": 10, "best_metric": 0.9})
    out = buf.getvalue()
    assert "model=opus" in out and "Bash" in out and "abcdef12" in out and "step 3/10" in out


def test_headless_ui_notice_survives_markup_in_text():
    buf = io.StringIO()
    ui = bridge.HeadlessUI(Console(file=buf, width=200))
    ui.post_system_notice("error [not-markup] {brace}", style="bold red")  # must not raise on stray brackets
    assert "not-markup" in buf.getvalue()


def test_orchestrator_stores_seed_prompt():
    orch = bridge.Orchestrator(
        app=bridge.HeadlessUI(Console(file=io.StringIO())),
        claude_args=[],
        api_key="k",
        session=bridge.DashboardSession.offline(),
        billing="claude",
        weco_api_base=None,
        effort=None,
        seed_prompt="optimize the repo",
    )
    assert orch._seed_prompt == "optimize the repo"
