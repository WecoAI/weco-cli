"""SDK message ↔ dashboard transcript conversion + display formatting.

The dashboard consumes the same stream-json envelope shape that
``claude -p`` emits, so its transcript parser works unchanged. These
helpers turn the Agent SDK's typed messages into that wire shape, plus a
few small string formatters shared by the orchestrator and approval router.
"""

from __future__ import annotations

from typing import Any, Optional

from claude_agent_sdk.types import (
    AssistantMessage,
    ResultMessage,
    StreamEvent,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)


INTERRUPT_MARKER = "Request interrupted by user"


def envelope_for(message: Any) -> Optional[dict]:
    """SDK message → claude stream-json envelope for the dashboard relay."""
    if isinstance(message, StreamEvent):
        return {
            "type": "stream_event",
            "event": message.event,
            "session_id": message.session_id,
            "parent_tool_use_id": message.parent_tool_use_id,
        }
    if isinstance(message, AssistantMessage):
        msg: dict[str, Any] = {"role": "assistant", "content": [block_to_dict(b) for b in message.content]}
        # `id` is what claude's native stream-json emits; the dashboard
        # parser keys its streamed-vs-bulk dedupe off it.
        if message.message_id:
            msg["id"] = message.message_id
        return {"type": "assistant", "message": msg, "model": message.model, "session_id": message.session_id}
    if isinstance(message, UserMessage):
        content = message.content
        if isinstance(content, list):
            content_json = [block_to_dict(b) for b in content]
        else:
            content_json = content
        return {"type": "user", "message": {"role": "user", "content": content_json}}
    if isinstance(message, SystemMessage):
        return {"type": "system", "subtype": message.subtype, **message.data}
    if isinstance(message, ResultMessage):
        return {
            "type": "result",
            "subtype": message.subtype,
            "duration_ms": message.duration_ms,
            "duration_api_ms": message.duration_api_ms,
            "is_error": message.is_error,
            "num_turns": message.num_turns,
            "session_id": message.session_id,
            "stop_reason": message.stop_reason,
            "total_cost_usd": message.total_cost_usd,
            "usage": message.usage,
            "result": message.result,
        }
    return None


def block_to_dict(block: Any) -> dict:
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    if isinstance(block, ThinkingBlock):
        return {"type": "thinking", "thinking": block.thinking, "signature": block.signature}
    if isinstance(block, ToolUseBlock):
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    if isinstance(block, ToolResultBlock):
        return {"type": "tool_result", "tool_use_id": block.tool_use_id, "content": block.content, "is_error": block.is_error}
    return {"type": "unknown", "repr": repr(block)}


def is_synthetic_interrupt_message(message: Any) -> bool:
    """Detect the SDK's post-interrupt synthetic UserMessage.

    After `client.interrupt()`, the SDK appends a UserMessage whose sole
    content reads roughly ``[Request interrupted by user]``, then emits a
    final ResultMessage and ends the turn. That synthetic message is an
    internal protocol artifact — it should not render as if the user typed
    it. Match on the marker substring so a real message that merely mentions
    an interrupt isn't swallowed.
    """
    if not isinstance(message, UserMessage):
        return False
    content = message.content
    if isinstance(content, str):
        return INTERRUPT_MARKER in content
    if isinstance(content, list):
        for block in content:
            text = getattr(block, "text", None)
            if isinstance(text, str) and INTERRUPT_MARKER in text:
                return True
    return False


def stringify_tool_result(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                t = item.get("text")
                if isinstance(t, str):
                    parts.append(t)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return ""


def summarise(tool_name: str, tool_input: Any) -> str:
    """A short one-line summary of a tool call for cards + approval prompts."""
    if not isinstance(tool_input, dict):
        return tool_name
    for key in ("command", "file_path", "path", "url", "query", "pattern"):
        v = tool_input.get(key)
        if isinstance(v, str) and v:
            return v if len(v) <= 200 else v[:197] + "…"
    return ""


def render_ui_context_preamble(ctx: Optional[dict]) -> str:
    """Render a dashboard UI snapshot as a hidden ``<system-reminder>``.

    Empty string when nothing's available. The dashboard's transcript
    parser filters the reminder from the visible chat (see
    ``isSyntheticCCInjection`` on the dashboard side). Plain key-value
    rather than JSON so claude reads it without parsing overhead and a
    noisy field can't blow out the user's actual prompt.
    """
    if not ctx:
        return ""
    rows: list[str] = []
    if ctx.get("summary"):
        rows.append(f"Run: {ctx['summary']}")
    if ctx.get("run_id"):
        rows.append(f"Run ID: {ctx['run_id']}")
    if ctx.get("lineage_id"):
        rows.append(f"Lineage: {ctx['lineage_id']}")
    if ctx.get("step") is not None:
        rows.append(f"Active step: {ctx['step']}")
    if ctx.get("best_metric") is not None:
        rows.append(f"Best metric: {ctx['best_metric']}")
    if ctx.get("url"):
        rows.append(f"URL: {ctx['url']}")
    if not rows:
        return ""
    body = "\n".join(rows)
    return (
        "<system-reminder>\n"
        "The user is currently viewing this in the Weco dashboard. Use as "
        "context for their next prompt; don't repeat it back verbatim.\n\n"
        f"{body}\n"
        "</system-reminder>\n\n"
    )
