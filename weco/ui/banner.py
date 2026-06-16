"""Weco ASCII banner + session info.

Single source of truth for the startup splash. Rendered in two places:

* SDK bridge (``weco.commands.start.bridge``) — printed via Rich to the
  console before the session-create flow.
* TUI bridge (``weco.commands.start.tui_bridge``) — mounted as a Textual
  widget at app start.

The wordmark uses the logo's gradient stops (orange → pink → purple) row
by row. The session-info block sits underneath as a key/value list.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple


WORDMARK_LINES: Sequence[str] = (
    "██╗    ██╗███████╗ ██████╗ ██████╗ ",
    "██║    ██║██╔════╝██╔════╝██╔═══██╗",
    "██║ █╗ ██║█████╗  ██║     ██║   ██║",
    "██║███╗██║██╔══╝  ██║     ██║   ██║",
    "╚███╔███╔╝███████╗╚██████╗╚██████╔╝",
    " ╚══╝╚══╝ ╚══════╝ ╚═════╝ ╚═════╝ ",
)

# Per-line colours sampled from the logo's gradient stops
# (#f99d24 → #ff3c82 → #c649b6 → #9854e1 → #8759f2).
WORDMARK_COLORS: Sequence[str] = ("#f99d24", "#ff5c70", "#ff3c82", "#c649b6", "#9854e1", "#8759f2")


WELCOME_LINE = "Welcome to Weco — let's optimize something."


def session_info_rows(
    *, agent: str, model: Optional[str], billing: Optional[str], session_id: Optional[str], dashboard_url: Optional[str]
) -> Sequence[Tuple[str, str]]:
    """The ordered (label, value) pairs shown under the welcome line."""
    rows: list[Tuple[str, str]] = [("Agent", agent)]
    rows.append(("Model", model or "default"))
    if billing:
        rows.append(("Billing", _format_billing(billing)))
    if session_id:
        rows.append(("Session", session_id))
    if dashboard_url:
        rows.append(("Dashboard", dashboard_url))
    return rows


def _format_billing(billing: str) -> str:
    if billing == "weco":
        return "Weco credits"
    if billing == "claude":
        return "Claude (local OAuth / ANTHROPIC_API_KEY)"
    return billing


def render_console(
    console,
    *,
    agent: str = "Claude Code",
    model: Optional[str] = None,
    billing: Optional[str] = None,
    session_id: Optional[str] = None,
    dashboard_url: Optional[str] = None,
) -> None:
    """Print the gradient wordmark + session-info block to a Rich Console."""
    for line, color in zip(WORDMARK_LINES, WORDMARK_COLORS):
        console.print(f"[bold {color}]{line}[/]")
    console.print()
    console.print(f"  [bold]{WELCOME_LINE}[/]")
    console.print()
    label_w = max(
        len(label)
        for label, _ in session_info_rows(
            agent=agent, model=model, billing=billing, session_id=session_id, dashboard_url=dashboard_url
        )
    )
    for label, value in session_info_rows(
        agent=agent, model=model, billing=billing, session_id=session_id, dashboard_url=dashboard_url
    ):
        console.print(f"  [dim]{label.ljust(label_w)}[/]  {value}")
    console.print()
