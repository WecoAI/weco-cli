"""Smoke tests for the Textual TUI app (no real `claude` subprocess).

We drive WecoTUI via its public ``post_*`` surface and assert that the
expected widgets land in the message log. Uses Textual's headless test
harness so no real terminal is required.
"""

from __future__ import annotations

import pytest

pytest.importorskip("textual")

from weco.ui.tui import WecoTUI
from weco.ui.tui.widgets import AssistantMessage, RunUpdate, SystemNotice, ToolCallCard, UserMessage


@pytest.mark.asyncio
async def test_app_renders_full_turn_sequence():
    """Drive a representative event sequence; verify each kind of widget mounts."""
    app = WecoTUI()
    async with app.run_test() as pilot:
        app.post_system_notice("— claude ready —")
        app.post_user_message("read the file and update the greeting")
        app.post_assistant_delta("I'll take a look ")
        app.post_assistant_delta("and update it.\n")
        app.post_tool_use("t1", "Read", "/tmp/example.py", {"file_path": "/tmp/example.py"})
        app.post_tool_result("t1", "def greet(name):\n    return f'hi {name}'\n", is_error=False)
        app.post_tool_use(
            "t2",
            "Edit",
            "/tmp/example.py",
            {
                "file_path": "/tmp/example.py",
                "old_string": "    return f'hi {name}'",
                "new_string": "    return f'hello, {name}!'",
            },
        )
        app.post_tool_result("t2", "ok", is_error=False)
        app.post_run_update(
            {
                "kind": "new_best",
                "run_id": "abc12345-aaaa-bbbb-cccc-ddddeeeeffff",
                "text": "new best metric: 0.92",
                "hints": ["view: weco run show <id>"],
            }
        )
        app.post_turn_end("success", cost=0.0123, duration_ms=4210)

        # Yield to the event loop so mounts settle.
        await pilot.pause()

        app.query(".message")  # no class filter — count by type instead
        assert app.query(UserMessage)
        assert app.query(AssistantMessage)
        assert len(app.query(ToolCallCard)) == 2
        assert app.query(RunUpdate)
        # We expect at least: claude-ready + turn-end notices.
        assert len(app.query(SystemNotice)) >= 2


@pytest.mark.asyncio
async def test_user_input_invokes_submit_callback():
    """Pressing Enter in the input should call the bridge's submit callback."""
    received: list[str] = []

    async def cb(text: str) -> None:
        received.append(text)

    app = WecoTUI(submit_callback=cb)
    async with app.run_test() as pilot:
        app.query_one("#prompt").value = "hello"
        await pilot.press("enter")
        await pilot.pause()
        assert received == ["hello"]
        # The app should also have echoed the prompt as a UserMessage.
        assert app.query(UserMessage)
