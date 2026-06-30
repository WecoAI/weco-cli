"""Approval + AskUserQuestion routing for the Claude bridge.

Bridges the SDK's ``can_use_tool`` callback to two surfaces — the dashboard
(over the relay channel) and the local TUI (via orchestrator callbacks) —
and resolves the SDK call with whichever answers first. Approvals can be
cached per-tool or per-exact-call so a granted scope doesn't re-prompt.
"""

from __future__ import annotations

import asyncio
import itertools
import json
from typing import Any, Callable

from claude_agent_sdk.types import PermissionResultAllow, PermissionResultDeny, ToolPermissionContext

from .envelopes import summarise


# Publishes a JSON line to the dashboard; returns False if it couldn't (queue
# full / dropped). The router fails open — if it can't even ask, it allows.
PublishFn = Callable[[str], bool]


_id_seq = itertools.count(1)


def new_id() -> str:
    return f"weco-approval-{next(_id_seq)}"


def cache_key(tool_name: str, tool_input: Any) -> str:
    if not isinstance(tool_input, dict):
        return tool_name
    for key in ("command", "file_path", "path", "url", "query"):
        v = tool_input.get(key)
        if isinstance(v, str) and v:
            return f"{tool_name}:{v}"
    return tool_name


class ApprovalRouter:
    """Bridges SDK ``can_use_tool`` calls to the dashboard + local modal.

    For tool approvals: publishes an ``approval_request`` envelope and
    awaits an ``approval_response`` from any surface (dashboard channel
    or local TUI modal).

    For ``AskUserQuestion``: publishes a ``question_request`` and awaits
    a ``question_response``; on resolution returns ``PermissionResultAllow``
    with ``{questions, answers}`` folded into ``updated_input`` per the
    Agent SDK contract.
    """

    def __init__(self, *, publish: PublishFn, on_approval_request=None, on_question_request=None) -> None:
        self._publish = publish
        # No-op defaults let unit tests construct the router without hooks.
        self._on_approval_request = on_approval_request or (lambda *a, **k: None)
        self._on_question_request = on_question_request or (lambda *a, **k: None)
        self._pending_approvals: dict[str, asyncio.Future] = {}
        self._pending_questions: dict[str, asyncio.Future] = {}
        self._approvals: dict[str, str] = {}

    async def can_use_tool(
        self, tool_name: str, tool_input: dict[str, Any], context: ToolPermissionContext
    ) -> PermissionResultAllow | PermissionResultDeny:
        if tool_name == "AskUserQuestion":
            return await self._ask_user_question(tool_input, context)
        return await self._tool_approval(tool_name, tool_input, context)

    async def _tool_approval(
        self, tool_name: str, tool_input: dict[str, Any], context: ToolPermissionContext
    ) -> PermissionResultAllow | PermissionResultDeny:
        key = cache_key(tool_name, tool_input)
        cached = self._approvals.get(key) or self._approvals.get(tool_name)
        if cached == "approve":
            return PermissionResultAllow(updated_input=tool_input)

        request_id = getattr(context, "tool_use_id", None) or new_id()
        future = asyncio.get_running_loop().create_future()
        self._pending_approvals[request_id] = future
        summary = summarise(tool_name, tool_input)

        published = self._publish(
            json.dumps(
                {
                    "type": "_weco_meta",
                    "event": "approval_request",
                    "id": request_id,
                    "tool_name": tool_name,
                    "summary": summary,
                    "tool_input": tool_input,
                }
            )
        )
        if not published:
            # Couldn't reach the dashboard — fail open rather than block claude.
            self._pending_approvals.pop(request_id, None)
            return PermissionResultAllow(updated_input=tool_input)

        # Let the orchestrator pop a local modal alongside the dashboard
        # card. Best-effort — if no local UI is wired the dashboard still
        # resolves the future.
        try:
            self._on_approval_request(request_id, tool_name, summary, tool_input)
        except Exception:
            pass

        try:
            response = await future
        except asyncio.CancelledError:
            return PermissionResultDeny(message="Approval cancelled")

        decision = response.get("decision", "ask")
        scope = response.get("scope", "once")
        if decision == "approve":
            # "always" comes from the TUI modal's Allow-always button and is
            # treated as a tool-wide whitelist; "tool"/"command" are the more
            # granular scopes the dashboard sends.
            if scope in ("tool", "always"):
                self._approvals[tool_name] = "approve"
            elif scope == "command":
                self._approvals[key] = "approve"
            return PermissionResultAllow(updated_input=tool_input)
        return PermissionResultDeny(message=response.get("reason") or "Denied by user")

    async def _ask_user_question(
        self, tool_input: dict[str, Any], context: ToolPermissionContext
    ) -> PermissionResultAllow | PermissionResultDeny:
        questions = tool_input.get("questions") if isinstance(tool_input, dict) else None
        if not isinstance(questions, list) or not questions:
            return PermissionResultAllow(updated_input={"questions": questions or [], "answers": {}})

        request_id = getattr(context, "tool_use_id", None) or new_id()
        future = asyncio.get_running_loop().create_future()
        self._pending_questions[request_id] = future

        published = self._publish(
            json.dumps({"type": "_weco_meta", "event": "question_request", "id": request_id, "questions": questions})
        )
        if not published:
            self._pending_questions.pop(request_id, None)
            return PermissionResultAllow(updated_input={"questions": questions, "answers": {}})

        # Surface the same question inline in the TUI; whichever surface
        # (TUI or dashboard) answers first wins.
        try:
            self._on_question_request(request_id, questions)
        except Exception:
            pass

        try:
            response = await future
        except asyncio.CancelledError:
            return PermissionResultDeny(message="Question cancelled")

        answers = response.get("answers")
        if not isinstance(answers, dict):
            answers = {}
        return PermissionResultAllow(updated_input={"questions": questions, "answers": answers})

    def resolve(self, request_id: str, decision: str, scope: str = "once") -> None:
        future = self._pending_approvals.pop(request_id, None)
        if future is None or future.done():
            return
        future.set_result({"decision": decision, "scope": scope})

    def resolve_question(self, request_id: str, answers: dict[str, Any]) -> None:
        future = self._pending_questions.pop(request_id, None)
        if future is None or future.done():
            return
        future.set_result({"answers": answers})
