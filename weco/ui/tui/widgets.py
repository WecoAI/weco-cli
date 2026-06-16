"""Message widgets shown in the TUI conversation log.

Each widget renders one conceptual unit of the chat:
* :class:`UserMessage` — the user's prompt for this turn.
* :class:`AssistantMessage` — a streamable Markdown bubble.
* :class:`ToolCallCard` — a tool_use + its eventual tool_result, paired by id.
* :class:`SystemNotice` — `— claude ready —`, `— done —`, etc.
* :class:`RunUpdate` — wrapper-side weco-run progress lines.
* :class:`ThinkingIndicator` — Claude-Code-style spinner+word shown while the
  model is generating but no tokens have streamed yet.

Widgets accept Rich renderables, so prose stays markdown-rendered (via
:class:`rich.markdown.Markdown`) and tool output stays verbatim with no
auto-highlighting surprises.
"""

from __future__ import annotations

import random
import time
from typing import Optional

from rich.console import Group, RenderableType
from rich.markdown import Markdown
from rich.text import Text
from textual.widgets import Static


# How many lines / chars of tool output we show before collapsing into a
# "+N lines" marker. Claude still sees the full content; this is purely the
# local display budget.
_RESULT_MAX_LINES = 12
_RESULT_MAX_CHARS = 2000

# Diff line budgets when rendering Edit/Write.
_DIFF_MAX_OLD = 8
_DIFF_MAX_NEW = 12


class UserMessage(Static):
    """User prompt, rendered with a `›` left marker."""

    DEFAULT_CSS = """
    UserMessage {
        padding: 0 1;
        margin-top: 1;
    }
    """

    def __init__(self, text: str) -> None:
        super().__init__()
        self._text = text

    def on_mount(self) -> None:
        body = Text()
        body.append("› ", style="bold cyan")
        body.append(self._text, style="bold")
        self.update(body)


class AssistantMessage(Static):
    """A streamed assistant turn, rendered as Markdown.

    Call :meth:`append_delta` for each text fragment as the model streams.
    The widget re-renders the accumulated buffer as Markdown each call,
    so bold/lists/headings/fenced code all surface live.
    """

    DEFAULT_CSS = """
    AssistantMessage {
        padding: 0 1;
        margin-top: 1;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._buf = ""

    def append_delta(self, text: str) -> None:
        if not text:
            return
        self._buf += text
        # `code_theme` matches the dark CC look; Rich falls back to plain
        # if the terminal doesn't support truecolor.
        self.update(Markdown(self._buf, code_theme="github-dark"))

    @property
    def buffer(self) -> str:
        return self._buf


class SystemNotice(Static):
    """Dim italic ambient notice — model ready, turn done, etc."""

    DEFAULT_CSS = """
    SystemNotice {
        padding: 0 1;
        margin-top: 1;
    }
    """

    def __init__(self, text: str, *, style: str = "dim italic") -> None:
        super().__init__()
        self._text = text
        self._style = style

    def on_mount(self) -> None:
        self.update(Text(self._text, style=self._style))


class WecoBanner(Static):
    """Gradient WECO wordmark + welcome line + session-info block.

    Shown once at TUI startup. Fields default to ``None`` and are simply
    omitted from the info block when missing.
    """

    DEFAULT_CSS = """
    WecoBanner {
        padding: 1 2 0 2;
    }
    """

    def __init__(
        self,
        *,
        agent: str = "Claude Code",
        model: Optional[str] = None,
        billing: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._agent = agent
        self._model = model
        self._billing = billing
        self._session_id = session_id

    def on_mount(self) -> None:
        # Imported lazily so the widget module stays free of cli-only deps.
        from weco.ui.banner import WORDMARK_LINES, WORDMARK_COLORS, WELCOME_LINE, session_info_rows

        text = Text()
        for i, (line, color) in enumerate(zip(WORDMARK_LINES, WORDMARK_COLORS)):
            if i:
                text.append("\n")
            text.append(line, style=f"bold {color}")

        text.append("\n\n")
        text.append(WELCOME_LINE, style="bold")
        text.append("\n")

        rows = session_info_rows(
            agent=self._agent,
            model=self._model,
            billing=self._billing,
            session_id=self._session_id,
            dashboard_url=None,  # The TUI intentionally omits the dashboard link.
        )
        label_w = max(len(label) for label, _ in rows)
        for label, value in rows:
            text.append("\n  ")
            text.append(label.ljust(label_w), style="dim")
            text.append("  ")
            text.append(value)

        self.update(text)


class RunUpdate(Static):
    """Single-line wrapper-rendered weco-run update with optional hints."""

    DEFAULT_CSS = """
    RunUpdate {
        padding: 0 1;
        margin-top: 1;
    }
    """

    _STYLES = {
        "new_best": ("✦", "bold green"),
        "completed": ("✓", "bold green"),
        "step_advance": ("›", "cyan"),
        "idle": ("…", "dim"),
        "errored": ("✗", "bold red"),
        "stopped": ("◼", "yellow"),
        "pending_review": ("⚠", "yellow"),
        "attached": ("→", "dim cyan"),
    }

    def __init__(self, update: dict) -> None:
        super().__init__()
        self._update = update

    def on_mount(self) -> None:
        kind = self._update.get("kind", "")
        icon, style = self._STYLES.get(kind, ("›", "cyan"))
        run_id = self._update.get("run_id") or ""
        short_id = run_id[:8] if run_id else "????????"
        body = Text()
        body.append(f"{icon}", style=style)
        body.append(f" [{short_id}] ", style="dim")
        body.append(str(self._update.get("text", "")))
        hints = self._update.get("hints") or []
        for hint in hints:
            if isinstance(hint, str) and hint:
                body.append("\n   → ", style="dim")
                body.append(hint, style="dim")
        self.update(body)


class ThinkingIndicator(Static):
    """Animated `✻ Thinking…` line shown while we're waiting on Claude.

    Sits above the input box (see :class:`weco.ui.tui.app.WecoTUI.compose`).
    Hidden when nothing is in flight; visible from prompt-submit through
    first token, and again between tool_result and the next assistant
    chunk. The label rotates through a small list of evocative verbs to
    match Claude Code's vibe.
    """

    DEFAULT_CSS = """
    ThinkingIndicator {
        height: 1;
        padding: 0 1;
        color: $text-muted;
    }
    ThinkingIndicator.-hidden {
        display: none;
    }
    """

    # Single-glyph spinner — six-pointed-star variants give a CC-ish twinkle.
    _SPINNER_FRAMES = ("✻", "✺", "✹", "✸", "✷", "✶", "✵", "✴", "✳", "✲", "✱", "✺")
    # Word rotates every few seconds so long waits feel alive without
    # being noisy.
    _WORDS = (
        "Thinking",
        "Cogitating",
        "Pondering",
        "Marinating",
        "Brewing",
        "Synthesizing",
        "Mulling",
        "Deliberating",
        "Weighing",
        "Ruminating",
    )

    def __init__(self, *args, **kwargs) -> None:
        super().__init__("", *args, **kwargs)
        self.add_class("-hidden")
        self._timer = None
        self._frame = 0
        self._word_idx = 0
        self._started_at: Optional[float] = None

    def start(self) -> None:
        """Show the indicator and start animating."""
        self._frame = 0
        self._word_idx = random.randrange(len(self._WORDS))
        self._started_at = time.monotonic()
        self.remove_class("-hidden")
        self._tick()
        if self._timer is None:
            self._timer = self.set_interval(0.1, self._tick)

    def stop(self) -> None:
        """Hide the indicator and pause the animation timer."""
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        self._started_at = None
        self.add_class("-hidden")

    def _tick(self) -> None:
        self._frame += 1
        spinner = self._SPINNER_FRAMES[self._frame % len(self._SPINNER_FRAMES)]
        # Cycle through verbs every ~4 seconds (40 ticks at 100ms).
        if self._frame % 40 == 0:
            self._word_idx = (self._word_idx + 1) % len(self._WORDS)
        word = self._WORDS[self._word_idx]
        elapsed = f" ({int(time.monotonic() - self._started_at)}s)" if self._started_at is not None else ""
        body = Text()
        body.append(spinner, style="bold magenta")
        body.append(f" {word}…", style="dim")
        if elapsed:
            body.append(elapsed, style="dim")
        body.append("   esc to interrupt", style="dim italic")
        self.update(body)


class ToolCallCard(Static):
    """One tool_use, paired with its tool_result by id.

    The card mounts immediately when the model emits the tool_use (bullet
    in yellow, "running"). When the result arrives we re-render with a
    green/red bullet and a tool-specific body.
    """

    DEFAULT_CSS = """
    ToolCallCard {
        padding: 0 1;
        margin-top: 1;
    }
    """

    def __init__(self, name: str, summary: str, tool_input: dict) -> None:
        super().__init__()
        self._name = name
        self._summary = summary
        self._input = tool_input if isinstance(tool_input, dict) else {}
        self._result_text: Optional[str] = None
        self._is_error = False
        self._status = "running"

    def on_mount(self) -> None:
        self._refresh_card()

    def set_result(self, text: str, *, is_error: bool) -> None:
        self._result_text = text or ""
        self._is_error = is_error
        self._status = "error" if is_error else "ok"
        self._refresh_card()

    def _refresh_card(self) -> None:
        bullet_style = {"running": "bold yellow", "ok": "bold green", "error": "bold red"}[self._status]
        head = Text()
        head.append("●", style=bullet_style)
        head.append(f" {self._name}", style="bold")
        if self._summary:
            head.append("(", style="default")
            head.append(self._summary, style="dim")
            head.append(")", style="default")

        parts: list[RenderableType] = [head]
        body = self._build_body()
        if body is not None:
            parts.append(body)
        self.update(Group(*parts))

    def _build_body(self) -> Optional[RenderableType]:
        if self._status == "running":
            return None

        # Tool-specific renderers
        if self._name == "TodoWrite" and not self._is_error:
            todos = self._input.get("todos") if isinstance(self._input, dict) else None
            if isinstance(todos, list):
                return _render_todos(todos)
        if self._name in ("Edit", "Write") and not self._is_error:
            diff = _render_diff(self._name, self._input)
            if diff is not None:
                return diff

        return _render_generic_body(self._result_text or "", is_error=self._is_error)


# --- Renderable builders -----------------------------------------------------


def _render_generic_body(text: str, *, is_error: bool) -> RenderableType:
    text = (text or "").rstrip()
    glyph_style = "red" if is_error else "dim"
    body_style = "red" if is_error else "default"
    if not text:
        out = Text("  ⎿  ", style=glyph_style)
        out.append("(no output)", style="dim")
        return out

    if len(text) > _RESULT_MAX_CHARS:
        text = text[:_RESULT_MAX_CHARS] + " …"

    lines = text.split("\n")
    extra = 0
    if len(lines) > _RESULT_MAX_LINES:
        extra = len(lines) - _RESULT_MAX_LINES
        lines = lines[:_RESULT_MAX_LINES]

    body = Text()
    for i, line in enumerate(lines):
        if i == 0:
            body.append("  ⎿  ", style=glyph_style)
        else:
            body.append("     ")
        body.append(line, style=body_style)
        if i < len(lines) - 1 or extra:
            body.append("\n")
    if extra:
        body.append(f"     … +{extra} lines", style="dim")
    return body


def _render_todos(todos: list) -> RenderableType:
    body = Text()
    body.append("  ⎿  ", style="dim")
    first = True
    for todo in todos:
        if not isinstance(todo, dict):
            continue
        if not first:
            body.append("\n     ")
        first = False
        status = todo.get("status", "")
        content = str(todo.get("content", ""))
        if status == "completed":
            body.append("☒ ", style="green")
            body.append(content, style="dim")
        elif status == "in_progress":
            body.append("☐ ", style="yellow")
            body.append(content, style="bold")
        else:
            body.append("☐ ")
            body.append(content)
    return body


def _render_diff(tool_name: str, tool_input: dict) -> Optional[RenderableType]:
    path = str(tool_input.get("file_path", "")) if isinstance(tool_input, dict) else ""
    if tool_name == "Write":
        old = ""
        new = tool_input.get("content") if isinstance(tool_input, dict) else None
    else:
        old = tool_input.get("old_string") if isinstance(tool_input, dict) else None
        new = tool_input.get("new_string") if isinstance(tool_input, dict) else None
    if not isinstance(old, str) or not isinstance(new, str):
        return None

    old_lines = old.splitlines()
    new_lines = new.splitlines()
    verb = "Created" if tool_name == "Write" else "Updated"
    summary = f"{verb} {path}"
    if tool_name == "Edit":
        summary += f" ({len(old_lines)} → {len(new_lines)} lines)"
    elif new_lines:
        summary += f" ({len(new_lines)} lines)"

    body = Text()
    body.append("  ⎿  ", style="dim")
    body.append(summary, style="dim")

    truncated = False
    for line in old_lines[:_DIFF_MAX_OLD]:
        body.append("\n       - ", style="red")
        body.append(line, style="red")
    if len(old_lines) > _DIFF_MAX_OLD:
        truncated = True
    for line in new_lines[:_DIFF_MAX_NEW]:
        body.append("\n       + ", style="green")
        body.append(line, style="green")
    if len(new_lines) > _DIFF_MAX_NEW:
        truncated = True
    if truncated:
        body.append("\n       … (diff truncated)", style="dim")
    return body
