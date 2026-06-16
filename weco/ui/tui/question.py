"""Inline AskUserQuestion widget — sequential, one-question-at-a-time.

Mounts in the chat scroll. Renders **only the active question** at any
moment; previously answered questions collapse into a one-line summary
above the active picker so the user only has one decision in focus.
Matches Claude Code's native AskUserQuestion picker UX.

Per-question controls:

* single-select → `OptionList`. Arrow keys + number-jump; Enter
  selects and **auto-advances** to the next question (or submits if
  this was the last).
* multi-select  → `SelectionList`. Space toggles items, Enter on the
  "Continue" button below advances. (Multi-select can't auto-advance
  because there's no "done picking" signal.)

Submission resolves the future with ``{question_text: answer}`` where
answer is a string (single-select) or list[str] (multi-select). Esc
resolves with ``{}`` which the SDK treats as "no input".
"""

from __future__ import annotations

import asyncio
from typing import Union

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, OptionList, SelectionList, Static
from textual.widgets.option_list import Option


AnswerValue = Union[str, list[str]]
AnswersDict = dict[str, AnswerValue]


class QuestionCard(Static):
    """Inline question card. Sequential per-question prompting."""

    can_focus = True

    DEFAULT_CSS = """
    QuestionCard {
        padding: 1 0;
        margin-top: 1;
        height: auto;
    }
    QuestionCard .q-header {
        text-style: bold;
        color: $text-muted;
        padding-bottom: 1;
        height: auto;
    }
    QuestionCard .q-done {
        color: $text-muted;
        height: 1;
        padding: 0;
    }
    QuestionCard .q-text {
        padding-bottom: 0;
        height: auto;
    }
    QuestionCard .q-detail {
        color: $text-muted;
        height: auto;
        padding-bottom: 1;
    }
    QuestionCard .q-hint {
        color: $text-muted;
        height: 1;
        padding-top: 0;
    }
    QuestionCard #q-done,
    QuestionCard #q-current {
        height: auto;
        padding: 0;
    }
    QuestionCard OptionList,
    QuestionCard SelectionList {
        background: transparent;
        border: none;
        padding: 0;
        height: auto;
    }
    QuestionCard #continue-row {
        height: 3;
        align-horizontal: right;
        margin-top: 0;
    }
    QuestionCard #continue-row Button {
        margin: 0 1;
    }
    """

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(
        self,
        questions: list[dict],
        future: "asyncio.Future[AnswersDict]",
        *,
        active_title: str = "Agent needs your input",
        done_title: str = "Agent's questions",
    ) -> None:
        super().__init__()
        self._questions = [q for q in (questions or []) if isinstance(q, dict)]
        self._future = future
        self._answered = False
        self._idx = 0
        self._answers: AnswersDict = {}
        # Header copy — overridable so the same picker fronts both
        # AskUserQuestion prompts and tool-call approvals (see
        # `ApprovalCard`, which subclasses this with approval-flavoured
        # titles). Defaults preserve the question-flow wording.
        self._active_title = active_title
        self._done_title = done_title

    def compose(self) -> ComposeResult:
        yield Static(self._header_text(), id="q-header", classes="q-header", markup=False)
        # Container for "already answered" one-line summaries — grows
        # as questions are answered.
        yield Vertical(id="q-done")
        # Container for the active question's widgets — cleared and
        # repopulated on every advance.
        yield Vertical(id="q-current")

    def on_mount(self) -> None:
        if not self._questions:
            self._submit()
            return
        self._show_current_question()

    # --- Header / progress ---------------------------------------------

    def _header_text(self) -> str:
        n = len(self._questions)
        if self._answered:
            return f"{self._done_title} · answered"
        if n == 1:
            return self._active_title
        return f"{self._active_title} · {self._idx + 1} of {n}"

    def _refresh_header(self) -> None:
        try:
            self.query_one("#q-header", Static).update(self._header_text())
        except Exception:
            pass

    # --- Sequential prompting ------------------------------------------

    def _show_current_question(self) -> None:
        """Clear the active-question container and render the current
        question's prompt + picker. Auto-submits when past the last
        question.

        Implementation note — child widget ids are suffixed with the
        per-question index (`picker-0`, `picker-1`, …). `Widget.remove()`
        is scheduled, not immediate: when we clear the container and
        immediately mount the next question's widgets, the old children
        are still in the tree for one event-loop tick. Using a fresh
        id per question avoids the resulting `DuplicateIds` error
        without needing to make the handler async.
        """
        try:
            current = self.query_one("#q-current", Vertical)
        except Exception:
            return
        for child in list(current.children):
            child.remove()

        if self._idx >= len(self._questions):
            self._submit()
            return

        q = self._questions[self._idx]
        qtext = str(q.get("question") or "")
        detail = str(q.get("detail") or "")
        options = q.get("options") if isinstance(q.get("options"), list) else []
        multi = bool(q.get("multiSelect"))

        picker_id = self._picker_id()
        continue_id = f"continue-{self._idx}"
        row_id = f"continue-row-{self._idx}"

        # Question text
        current.mount(Static(qtext, classes="q-text", markup=False))
        # Optional muted detail line under the question — e.g. the tool
        # summary on an approval (file path / shell command / JSON). Kept
        # out of `qtext` so the collapsed "✓ … → answer" summary stays a
        # readable one-liner.
        if detail:
            current.mount(Static(detail, classes="q-detail", markup=False))

        if multi:
            entries = [(Text(str(o.get("label") or "")), str(o.get("label") or "")) for o in options if isinstance(o, dict)]
            picker = SelectionList[str](*entries, id=picker_id)
            current.mount(picker)
            current.mount(Static("  Space to toggle  ·  Tab → Continue to advance", classes="q-hint", markup=False))
            # Continue button — the explicit "done picking" signal that
            # multi-select needs (Enter on a SelectionList toggles, so
            # we can't reuse it for advance).
            row = Horizontal(id=row_id)
            current.mount(row)
            row.mount(Button("Continue", id=continue_id, variant="primary"))
        else:
            opts = [
                Option(Text(str(o.get("label") or "")), id=str(o.get("label") or "")) for o in options if isinstance(o, dict)
            ]
            picker = OptionList(*opts, id=picker_id)
            current.mount(picker)

        self._refresh_header()

        # Focus the picker so arrow keys / number-jump / Space / Enter
        # work without the user having to click first.
        def _focus_picker():
            try:
                self.app.set_focus(self.query_one(f"#{picker_id}"))
            except Exception:
                pass

        self.app.call_after_refresh(_focus_picker)

    def _picker_id(self) -> str:
        return f"picker-{self._idx}"

    def _record_and_advance(self, answer: AnswerValue) -> None:
        q = self._questions[self._idx]
        qtext = str(q.get("question") or "")
        self._answers[qtext] = answer
        # Mount a collapsed summary into the "done" container so the
        # user can see what they've already answered without it
        # competing for attention with the active question.
        try:
            done = self.query_one("#q-done", Vertical)
            done.mount(Static(f"  ✓ {qtext} → {_format_answer(answer)}", classes="q-done", markup=False))
        except Exception:
            pass
        self._idx += 1
        self._show_current_question()

    # --- Events ---------------------------------------------------------

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Single-select Enter — answer is the option id (= the label
        string, since we set them equal at construction). Auto-advance.

        Per-question-index ids (`picker-0`, `picker-1`, …) let us tell
        the active picker apart from any still-tearing-down picker
        from a previous question, so a late OptionSelected fired by
        an unmounting widget can't double-advance.
        """
        if self._answered or self._future.done():
            return
        if self._idx >= len(self._questions):
            return
        if (event.option_list.id or "") != self._picker_id():
            return
        answer = str(event.option.id or "")
        self._record_and_advance(answer)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid == f"continue-{self._idx}":
            try:
                picker = self.query_one(f"#{self._picker_id()}", SelectionList)
            except Exception:
                return
            picks = list(picker.selected)
            self._record_and_advance(picks)

    def action_cancel(self) -> None:
        self._cancel()

    # --- Cross-surface sync --------------------------------------------

    def resolve_remotely(self, answers: AnswersDict) -> None:
        """Called when the dashboard answered first — freeze the card."""
        if self._answered or self._future.done():
            return
        self._future.set_result(answers)
        self._mark_done("answered on dashboard")

    # --- Terminal state ------------------------------------------------

    def _submit(self) -> None:
        if self._answered or self._future.done():
            return
        self._future.set_result(self._answers)
        self._mark_done("answered")

    def _cancel(self) -> None:
        if self._answered or self._future.done():
            return
        self._future.set_result({})
        self._mark_done("dismissed")

    def _mark_done(self, status: str) -> None:
        self._answered = True
        # Clear the active-question container — only the "done" summary
        # column stays as conversation history.
        try:
            current = self.query_one("#q-current", Vertical)
            for child in list(current.children):
                child.remove()
        except Exception:
            pass
        try:
            self.query_one("#q-header", Static).update(f"{self._done_title} · {status}")
        except Exception:
            pass


def _format_answer(answer: AnswerValue) -> str:
    if isinstance(answer, list):
        return ", ".join(answer) if answer else "(none)"
    return str(answer)
