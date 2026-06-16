"""In-TUI tool-call approval.

Rendered with the **same inline picker component as AskUserQuestion**
(:class:`~weco.ui.tui.question.QuestionCard`) so confirmations and
option-choosing share one widget — and one cross-surface sync path with
the dashboard.

Fired by the bridge when the SDK's ``can_use_tool`` callback gates on a
tool: the bridge mounts an :class:`ApprovalCard` inline in the chat scroll
and awaits a ``(decision, scope)`` decision.

* decision: ``"approve"`` | ``"deny"`` | ``"ask"``
* scope:    ``"once"``    | ``"always"``
"""

from __future__ import annotations

import asyncio
from typing import Tuple

from .question import QuestionCard


Decision = Tuple[str, str]


# Option label → (decision, scope), in display order. "always" is treated
# as a tool-wide whitelist by the approval router (it maps it the same way
# as the dashboard's "tool" scope); finer-grained "command"/exact-call
# scoping stays a dashboard-only affordance for now.
_APPROVAL_CHOICES: list[tuple[str, Decision]] = [
    ("Allow once", ("approve", "once")),
    ("Allow always", ("approve", "always")),
    ("Deny", ("deny", "once")),
]
_LABEL_TO_DECISION: dict[str, Decision] = {label: decision for label, decision in _APPROVAL_CHOICES}

# Short status word shown in the collapsed header after the user picks.
_DECISION_STATUS: dict[Decision, str] = {
    ("approve", "once"): "allowed once",
    ("approve", "always"): "always allowed",
    ("deny", "once"): "denied",
}


class ApprovalCard(QuestionCard):
    """Inline tool-call approval — a single-select :class:`QuestionCard`
    whose options map to ``(decision, scope)`` tuples.

    Resolves ``decision_future`` with the chosen :data:`Decision` (or
    ``("deny", "once")`` on Escape). When the dashboard answers the same
    approval first, the bridge calls :meth:`resolve_remotely_decision` to
    freeze the card.
    """

    def __init__(self, tool_name: str, summary: str, decision_future: "asyncio.Future[Decision]") -> None:
        self._decision_future = decision_future
        question = {
            # Short question text → clean collapsed "✓ … → answer" line.
            "question": f"Allow {tool_name}?",
            # The arbitrary tool summary (paths, commands, JSON) rides as
            # the muted detail sub-line, not the question text.
            "detail": summary,
            "options": [{"label": label} for label, _ in _APPROVAL_CHOICES],
            "multiSelect": False,
        }
        # The parent drives an answers-future we never await — we translate
        # the selected label into a Decision instead (see `_submit`).
        answers_future: "asyncio.Future[dict]" = asyncio.get_running_loop().create_future()
        super().__init__([question], answers_future, active_title="Tool call needs approval", done_title="Tool call")

    # --- Decision plumbing ---------------------------------------------

    def _resolve_decision(self, decision: Decision) -> None:
        if not self._decision_future.done():
            self._decision_future.set_result(decision)

    def _submit(self) -> None:
        # A single question → exactly one recorded answer; translate its
        # selected label into a Decision rather than handing back the
        # answers map the question flow uses.
        if self._answered or self._decision_future.done():
            return
        label = next(iter(self._answers.values()), "")
        decision = _LABEL_TO_DECISION.get(str(label), ("deny", "once"))
        self._resolve_decision(decision)
        self._mark_done(_DECISION_STATUS.get(decision, "answered"))

    def _cancel(self) -> None:
        # Escape on an approval = deny once.
        if self._answered or self._decision_future.done():
            return
        self._resolve_decision(("deny", "once"))
        self._mark_done("denied")

    # --- Cross-surface sync --------------------------------------------

    def resolve_remotely_decision(self, decision: str, scope: str) -> None:
        """Dashboard answered first — freeze the card with its decision."""
        if self._answered or self._decision_future.done():
            return
        self._resolve_decision((decision, scope))
        self._mark_done("answered on dashboard")

    def resolve_remotely(self, answers) -> None:  # type: ignore[override]
        """Compatibility shim for the generic turn-cancel cleanup path,
        which calls ``resolve_remotely({})`` on every active card. For an
        approval, a cancelled turn means the gated tool call is moot →
        deny once."""
        if self._answered or self._decision_future.done():
            return
        self._resolve_decision(("deny", "once"))
        self._mark_done("dismissed")
