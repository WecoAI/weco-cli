"""SDK message stream → WecoTUI rendering.

The dashboard side of "SDK message → wire" lives in ``envelopes``; this is
its local-terminal twin: it turns the same Agent SDK message stream into
``WecoTUI.post_*`` calls. It owns the small amount of token-streaming state
needed to avoid double-rendering text that already arrived as deltas.
"""

from __future__ import annotations

from typing import Any, Optional

from claude_agent_sdk.types import (
    AssistantMessage,
    ResultMessage,
    StreamEvent,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from weco.ui.tui import WecoTUI

from .envelopes import stringify_tool_result, summarise


class Renderer:
    """Translates the SDK message stream into ``WecoTUI`` post_* calls."""

    def __init__(self, app: WecoTUI) -> None:
        self.app = app
        # `streamed_message_ids` collects every message-id we've already
        # rendered text deltas for; the trailing bulk AssistantMessage then
        # skips those text blocks to avoid double-rendering.
        # `current_stream_message_id` tracks the active stream so deltas can
        # register themselves.
        self._streamed_message_ids: set[str] = set()
        self._current_stream_message_id: Optional[str] = None

    def render(self, message: Any) -> None:
        if isinstance(message, StreamEvent):
            self._render_stream_event(message)
        elif isinstance(message, AssistantMessage):
            self._render_assistant(message)
        elif isinstance(message, UserMessage):
            self._render_user(message)
        elif isinstance(message, ResultMessage):
            self._render_result(message)
        # SystemMessage events are forwarded to the dashboard but need no
        # local render — the banner already shows the model.

    def _render_result(self, message: ResultMessage) -> None:
        self.app.hide_thinking()
        sub = message.subtype or ""
        if sub and sub != "success":
            self.app.post_turn_end(str(sub), cost=message.total_cost_usd, duration_ms=message.duration_ms)

    def _render_assistant(self, message: AssistantMessage) -> None:
        # If deltas for this message already streamed, skip text blocks in the
        # bulk event — otherwise the response would render twice.
        message_id = getattr(message, "message_id", None)
        already_streamed = bool(message_id and message_id in self._streamed_message_ids)

        any_text = False
        for block in message.content:
            if isinstance(block, TextBlock) and block.text:
                if already_streamed:
                    # Tool calls still need to come through; text is the only
                    # block kind that got pre-rendered.
                    continue
                self.app.hide_thinking()
                self.app.post_assistant_delta(block.text)
                any_text = True
            elif isinstance(block, ToolUseBlock):
                self.app.end_assistant_block()
                self.app.hide_thinking()
                name = block.name or "tool"
                tool_input = block.input if isinstance(block.input, dict) else {}
                self.app.post_tool_use(block.id or "", name, summarise(name, tool_input), tool_input)
        if any_text:
            self.app.end_assistant_block()

    def _render_stream_event(self, message: StreamEvent) -> None:
        """Render token-level deltas from `include_partial_messages`.

        We only care about `content_block_delta` events of type `text_delta`;
        tool-use streaming is left to the trailing bulk AssistantMessage so we
        have the complete tool input for the card.
        """
        inner = message.event if isinstance(message.event, dict) else None
        if not inner:
            return
        inner_type = inner.get("type")
        if inner_type == "message_start":
            msg = inner.get("message")
            if isinstance(msg, dict) and isinstance(msg.get("id"), str):
                self._current_stream_message_id = msg["id"]
        elif inner_type == "content_block_start":
            # Tokens are about to flow — drop the spinner so it doesn't sit on
            # top of streaming output.
            self.app.hide_thinking()
        elif inner_type == "content_block_delta":
            delta = inner.get("delta")
            if isinstance(delta, dict) and delta.get("type") == "text_delta":
                text = delta.get("text")
                if isinstance(text, str) and text:
                    self.app.hide_thinking()
                    self.app.post_assistant_delta(text)
                    if self._current_stream_message_id:
                        self._streamed_message_ids.add(self._current_stream_message_id)
        elif inner_type == "message_stop":
            self._current_stream_message_id = None

    def _render_user(self, message: UserMessage) -> None:
        blocks = message.content if isinstance(message.content, list) else []
        had_result = False
        for block in blocks:
            if isinstance(block, ToolResultBlock):
                text = stringify_tool_result(block.content)
                self.app.post_tool_result(block.tool_use_id or "", text, is_error=bool(block.is_error))
                had_result = True
        if had_result:
            # Tool finished — claude will be thinking about its next response;
            # spin until the next block arrives.
            self.app.show_thinking()
