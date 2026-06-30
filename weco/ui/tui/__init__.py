"""Textual-based TUI for the `weco start claude --tui` orchestrator.

This package owns the *rendering* and *input* surfaces. The
:mod:`weco.commands.start.tui_bridge` module owns the subprocess +
relay/approval/run-watcher plumbing and pushes events into the App via the
public ``post_*`` methods on :class:`WecoTUI`.
"""

from .app import WecoTUI

__all__ = ["WecoTUI"]
