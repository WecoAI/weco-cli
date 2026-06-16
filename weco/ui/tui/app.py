"""Textual App for the `weco start claude --tui` orchestrator.

The App is purely a view: it owns the widget tree, the input box, and the
status bar, and exposes a small ``post_*`` surface that the bridge calls to
push stream events. The bridge owns subprocess + relay + approval + run
watcher state; it injects a ``submit_callback`` so the App can route user
input back to the orchestrator.
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, VerticalScroll
from textual.widgets import Input

from .question import QuestionCard
from .widgets import AssistantMessage, RunUpdate, SystemNotice, ThinkingIndicator, ToolCallCard, UserMessage, WecoBanner


# Bridge-provided callback. Receives the prompt text. May return an awaitable
# (e.g. when the bridge needs to do async cleanup before queuing). Errors
# from this callback are surfaced as a SystemNotice and otherwise ignored —
# the App must keep accepting input.
SubmitCallback = Callable[[str], Optional[Awaitable[None]]]

# Called once, after widgets are mounted and the App's loop is running.
# Use this to schedule any long-lived bridge work (subprocess loops, relay
# clients, watchers) — those need the App's loop to drive their tasks.
StartupCallback = Callable[["WecoTUI"], Optional[Awaitable[None]]]

# Called when the user presses Escape — bridge should SIGINT the current
# claude turn (if any). Returns truthy if something was actually preempted.
PreemptCallback = Callable[[], Optional[Awaitable[None]]]


class WecoTUI(App):
    """Conversation TUI for the claude orchestrator."""

    CSS = """
    Screen {
        layers: base modal;
    }
    #messages {
        height: 1fr;
        padding: 0 1;
    }
    #bottom-bar {
        dock: bottom;
        height: auto;
    }
    #thinking {
        height: auto;
    }
    Input {
        margin: 1 0 0 0;
    }
    """

    BINDINGS = [
        # Ctrl-C is two-tier: first press interrupts the current claude turn
        # and arms the exit flag; a second press within CTRL_C_EXIT_WINDOW_S
        # exits the TUI and asks the bridge to stop any active weco runs.
        # If no second press lands within the window, the exit flag resets
        # so an inadvertent earlier Ctrl-C doesn't haunt a later session.
        Binding("ctrl+c", "ctrl_c_press", "Interrupt", priority=True),
        Binding("ctrl+l", "scroll_end", "Jump to bottom"),
        Binding("escape", "preempt", "Interrupt", show=False, priority=True),
    ]

    # How long the "press Ctrl-C again to exit" warning stays armed after
    # the first press. 3s is short enough that a fat-fingered Ctrl-C
    # doesn't sit there forever; long enough that the user can register
    # the message and choose.
    CTRL_C_EXIT_WINDOW_S = 3.0

    def __init__(
        self,
        *,
        submit_callback: Optional[SubmitCallback] = None,
        startup_callback: Optional[StartupCallback] = None,
        preempt_callback: Optional[PreemptCallback] = None,
        title: str = "weco · claude",
    ) -> None:
        super().__init__()
        self.title = title
        self._submit_callback: Optional[SubmitCallback] = submit_callback
        self._startup_callback: Optional[StartupCallback] = startup_callback
        self._preempt_callback: Optional[PreemptCallback] = preempt_callback
        # Set on first Ctrl-C; cleared by a timer after CTRL_C_EXIT_WINDOW_S.
        # The second Ctrl-C while armed triggers exit-with-run-stop.
        self._ctrl_c_armed: bool = False
        self._ctrl_c_reset_timer = None
        # Late-bound by the bridge — called on the second Ctrl-C so the
        # bridge can stop any background `weco run` processes the agent
        # spawned before we tear down the UI.
        self._exit_with_stop_callback = None
        self._messages: Optional[VerticalScroll] = None
        self._input: Optional[Input] = None
        self._thinking: Optional[ThinkingIndicator] = None
        # Per-turn streaming state.
        self._current_assistant: Optional[AssistantMessage] = None
        self._tool_cards: dict[str, ToolCallCard] = {}

    # --- Wiring ----------------------------------------------------------

    def set_submit_callback(self, cb: SubmitCallback) -> None:
        """Late-bind the bridge's submit handler.

        Used when the App is constructed before the bridge has its
        pending-prompts queue ready.
        """
        self._submit_callback = cb

    def set_startup_callback(self, cb: StartupCallback) -> None:
        """Late-bind a callback to fire once the App is mounted and running."""
        self._startup_callback = cb

    def set_preempt_callback(self, cb: PreemptCallback) -> None:
        """Late-bind the bridge's Escape-key preempt handler."""
        self._preempt_callback = cb

    # --- Layout ----------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="messages")
        with Container(id="bottom-bar"):
            yield ThinkingIndicator(id="thinking")
            yield Input(placeholder="Type a prompt and press Enter (Ctrl+C to quit)", id="prompt")

    async def on_mount(self) -> None:
        self._messages = self.query_one("#messages", VerticalScroll)
        self._input = self.query_one("#prompt", Input)
        self._thinking = self.query_one("#thinking", ThinkingIndicator)
        self._input.focus()
        if self._startup_callback is not None:
            try:
                result = self._startup_callback(self)
                if asyncio.iscoroutine(result):
                    # Don't await inline — the startup callback typically
                    # kicks off a long-running task that should run for the
                    # lifetime of the app. Schedule it on the running loop.
                    asyncio.create_task(result)
            except Exception as exc:  # noqa: BLE001
                self.post_system_notice(f"startup error: {exc}", style="bold red")

    # --- Input handling --------------------------------------------------

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input is not self._input:
            return
        text = event.value.strip()
        if not text:
            return
        self._input.value = ""
        # Echo immediately so the user sees their prompt land before the
        # subprocess turns it into output.
        self.post_user_message(text)
        if self._submit_callback is None:
            self.post_system_notice("(no orchestrator wired up — prompt dropped)")
            return
        try:
            result = self._submit_callback(text)
            if asyncio.iscoroutine(result):
                await result
        except Exception as exc:  # noqa: BLE001
            self.post_system_notice(f"submit error: {exc}", style="bold red")

    def action_scroll_end(self) -> None:
        if self._messages is not None:
            self._messages.scroll_end(animate=False)

    async def action_preempt(self) -> None:
        if self._preempt_callback is None:
            return
        try:
            result = self._preempt_callback()
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            # Best-effort — don't crash the UI if the bridge is gone.
            pass

    def set_exit_with_stop_callback(self, cb) -> None:
        """Late-bind the bridge's exit-and-stop-runs handler. Invoked on
        the second Ctrl-C within the warning window."""
        self._exit_with_stop_callback = cb

    async def action_ctrl_c_press(self) -> None:
        """Two-tier Ctrl-C. First press interrupts the in-flight turn and
        warns the user; second press within the window exits the TUI and
        asks the bridge to stop any active weco runs."""
        if self._ctrl_c_armed:
            # Second press → exit + stop runs.
            self._ctrl_c_armed = False
            if self._ctrl_c_reset_timer is not None:
                try:
                    self._ctrl_c_reset_timer.stop()
                except Exception:
                    pass
                self._ctrl_c_reset_timer = None
            if self._exit_with_stop_callback is not None:
                try:
                    result = self._exit_with_stop_callback()
                    if asyncio.iscoroutine(result):
                        await result
                except Exception:
                    pass
            self.exit()
            return

        # First press → interrupt current turn (same as Esc) and warn.
        await self.action_preempt()
        self._ctrl_c_armed = True
        self.post_system_notice("Press Ctrl-C again to exit and terminate any active Weco runs.", style="bold yellow")
        # Reset the armed flag after the window so a stale Ctrl-C from
        # earlier doesn't sneak through later.
        self._ctrl_c_reset_timer = self.set_timer(self.CTRL_C_EXIT_WINDOW_S, self._disarm_ctrl_c)

    def _disarm_ctrl_c(self) -> None:
        self._ctrl_c_armed = False
        self._ctrl_c_reset_timer = None

    # --- Bridge → App: message posting ----------------------------------

    def post_system_notice(self, text: str, *, style: str = "dim italic") -> None:
        self._mount(SystemNotice(text, style=style))

    def post_banner(
        self,
        *,
        agent: str = "Claude Code",
        model: Optional[str] = None,
        billing: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """Mount the gradient WECO wordmark + welcome + session-info block.
        Called once at startup."""
        self._mount(WecoBanner(agent=agent, model=model, billing=billing, session_id=session_id))

    def post_user_message(self, text: str) -> None:
        self._end_assistant_block()
        self._mount(UserMessage(text))

    def post_assistant_delta(self, text: str) -> None:
        if not text:
            return
        if self._current_assistant is None:
            self._current_assistant = AssistantMessage()
            self._mount(self._current_assistant)
        self._current_assistant.append_delta(text)
        self._scroll_to_end()

    def end_assistant_block(self) -> None:
        self._end_assistant_block()

    def post_tool_use(self, tool_id: str, name: str, summary: str, tool_input: dict) -> None:
        self._end_assistant_block()
        card = ToolCallCard(name=name, summary=summary, tool_input=tool_input)
        if tool_id:
            self._tool_cards[tool_id] = card
        self._mount(card)

    def post_tool_result(self, tool_id: str, content: str, *, is_error: bool = False) -> None:
        card = self._tool_cards.pop(tool_id, None) if tool_id else None
        if card is None:
            # Result without a matching tool_use — surface as a generic
            # notice so we don't silently drop it.
            self.post_system_notice(f"(orphan tool_result) {content[:200]}", style="dim red" if is_error else "dim")
            return
        card.set_result(content, is_error=is_error)
        self._scroll_to_end()

    def post_run_update(self, update: dict) -> None:
        self._end_assistant_block()
        self._mount(RunUpdate(update))

    def post_turn_end(self, subtype: str, *, cost: Optional[float] = None, duration_ms: Optional[int] = None) -> None:
        self._end_assistant_block()
        parts: list[str] = []
        if isinstance(cost, (int, float)):
            parts.append(f"${cost:.4f} api equiv")
        if isinstance(duration_ms, (int, float)):
            parts.append(f"{int(duration_ms)}ms")
        suffix = f" ({', '.join(parts)})" if parts else ""
        if subtype == "success":
            self.post_system_notice(f"— done{suffix} —")
        else:
            self.post_system_notice(f"— claude {subtype}{suffix} —", style="bold yellow")

    # --- Bridge → App: thinking indicator -------------------------------

    def show_thinking(self) -> None:
        """Start the spinner above the input (idempotent)."""
        if self._thinking is not None:
            self._thinking.start()

    def hide_thinking(self) -> None:
        """Stop and hide the spinner (idempotent)."""
        if self._thinking is not None:
            self._thinking.stop()

    # --- Bridge → App: approval + questions ------------------------------

    def mount_inline_card(self, card: QuestionCard) -> None:
        """Mount a pre-built inline picker — a `QuestionCard` or its
        `ApprovalCard` subclass (each with its own future) — into the chat
        scroll. The orchestrator owns the future and the card's lifecycle
        in cross-surface sync paths.
        """
        self._mount(card)

    # --- Internals -------------------------------------------------------

    def _mount(self, widget) -> None:
        if self._messages is None:
            # Pre-mount; defer by attaching to the app once mounted.
            self.call_later(self._mount, widget)
            return
        self._messages.mount(widget)
        self._scroll_to_end()

    def _scroll_to_end(self) -> None:
        if self._messages is not None:
            self._messages.scroll_end(animate=False)

    def _end_assistant_block(self) -> None:
        self._current_assistant = None
