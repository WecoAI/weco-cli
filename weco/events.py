# weco/events.py
"""Event reporting module for analytics.

This module provides fire-and-forget event reporting to the Weco backend.
Events are sent asynchronously in a background thread to avoid blocking
the CLI execution. Failures are silently ignored.

Usage:
    from weco.events import send_event, EventContext, CLIInvokedEvent

    # Create a context for the CLI invocation
    ctx = EventContext()

    # Send an event
    send_event(CLIInvokedEvent(command="run"), ctx)

Environment variables:
    WECO_DISABLE_EVENTS: Set to "1" or "true" to disable event reporting.
"""

import os
import uuid
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import requests

from pydantic import BaseModel, Field

from . import __base_url__, __pkg_version__
from .config import get_or_create_installation_id, load_weco_api_key


# =============================================================================
# Event Definitions
# =============================================================================


class BaseEvent(BaseModel):
    """Base class for all events. Subclass to define specific events."""

    # Client-side timestamp set at event creation time
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def event_name(self) -> str:
        """Return the event name. Must be implemented by subclasses."""
        raise NotImplementedError

    def to_properties(self) -> dict[str, Any]:
        """Convert event fields to properties dict (excludes timestamp)."""
        return self.model_dump(exclude={"timestamp"}, exclude_none=True)


class CLIInvokedEvent(BaseEvent):
    """Tracked when the CLI is invoked."""

    command: str  # The command being run (run, login, setup, etc.)

    @property
    def event_name(self) -> str:
        return "cli.invoked"


class SkillInstallStartedEvent(BaseEvent):
    """Tracked when skill installation begins."""

    tool: str  # 'cursor' or 'claude_code'
    source: str  # 'local' or 'repo'

    @property
    def event_name(self) -> str:
        return "skill.install.started"


class SkillInstallCompletedEvent(BaseEvent):
    """Tracked when skill installation completes successfully."""

    tool: str
    source: str
    duration_ms: int

    @property
    def event_name(self) -> str:
        return "skill.install.completed"


class SkillInstallFailedEvent(BaseEvent):
    """Tracked when skill installation fails."""

    tool: str
    source: str
    error_type: str
    stage: str  # 'git_operation', 'setup', etc.

    @property
    def event_name(self) -> str:
        return "skill.install.failed"


class RunStartAttemptedEvent(BaseEvent):
    """Tracked when a run is attempted (before server contact)."""

    output_mode: str
    require_review: bool
    save_logs: bool
    steps: int
    model: str

    @property
    def event_name(self) -> str:
        return "run.start.attempted"


class AuthStartedEvent(BaseEvent):
    """Tracked when authentication flow begins."""

    @property
    def event_name(self) -> str:
        return "auth.started"


class AuthCompletedEvent(BaseEvent):
    """Tracked when authentication completes successfully."""

    @property
    def event_name(self) -> str:
        return "auth.completed"


class AuthFailedEvent(BaseEvent):
    """Tracked when authentication fails."""

    reason: str  # 'timeout', 'denied', 'expired', 'error', etc.

    @property
    def event_name(self) -> str:
        return "auth.failed"


# =============================================================================
# Event Context
# =============================================================================


def _is_events_disabled() -> bool:
    """Check if event reporting is disabled via environment variable."""
    val = os.environ.get("WECO_DISABLE_EVENTS", "").lower()
    return val in ("1", "true", "yes")


@dataclass
class EventContext:
    """Context for a CLI invocation.

    This holds information that should be attached to all events
    during a single CLI invocation.
    """

    installation_id: str = field(default_factory=get_or_create_installation_id)
    invocation_id: str = field(default_factory=lambda: f"inv_{uuid.uuid4().hex[:16]}")
    client_version: str = field(default_factory=lambda: __pkg_version__)
    invoked_via: str = "cli"  # Set via create_event_context()


def create_event_context(via_skill: bool = False) -> EventContext:
    """Create an event context for a CLI invocation.

    Args:
        via_skill: If True, indicates this invocation is via an AI skill.

    Returns:
        An EventContext instance. Returns a minimal context if creation fails.
    """
    invoked_via = "skill" if via_skill else "cli"
    try:
        return EventContext(invoked_via=invoked_via)
    except Exception:
        # Return a minimal context if anything fails
        return EventContext(
            installation_id="unknown",
            invocation_id=f"inv_{uuid.uuid4().hex[:16]}",
            client_version="unknown",
            invoked_via="cli",
        )


# Global event context for the current CLI invocation
_event_ctx: EventContext | None = None


def get_event_context() -> EventContext:
    """Get the global event context, creating one if needed."""
    global _event_ctx
    if _event_ctx is None:
        _event_ctx = create_event_context()
    return _event_ctx


def set_event_context(ctx: EventContext) -> None:
    """Set the global event context."""
    global _event_ctx
    _event_ctx = ctx


# =============================================================================
# Event Sending
# =============================================================================


def _send_event_request(
    event_name: str,
    timestamp: datetime,
    installation_id: str | None,
    invocation_id: str | None,
    client_version: str | None,
    invoked_via: str | None,
    properties: dict[str, Any],
    auth_headers: dict[str, str],
) -> None:
    """Send an event to the backend (runs in background thread)."""
    try:
        payload = {
            "event": event_name,
            "timestamp": timestamp.isoformat(),
            "installation_id": installation_id,
            "invocation_id": invocation_id,
            "client_version": client_version,
            "invoked_via": invoked_via,
            "properties": properties,
        }

        # Remove None values
        payload = {k: v for k, v in payload.items() if v is not None}

        requests.post(
            f"{__base_url__}/events/",
            json=payload,
            headers=auth_headers,
            timeout=5,  # Short timeout to avoid blocking
        )
    except Exception:
        # Silently ignore errors - they should never affect the CLI
        pass


def send_event(event: BaseEvent, ctx: EventContext | None = None, auth_headers: dict[str, str] | None = None) -> None:
    """Send an event asynchronously.

    This function returns immediately and sends the event in a background
    thread to avoid blocking the CLI. Failures are silently ignored.

    Args:
        event: The event to send (a subclass of BaseEvent).
        ctx: The event context (installation_id, invocation_id, etc.).
             If None, a new context will be created.
        auth_headers: Optional authentication headers for the API.
    """
    try:
        # Don't send if disabled
        if _is_events_disabled():
            return

        # Create context if not provided
        if ctx is None:
            ctx = create_event_context()

        # Build auth headers if not provided
        if auth_headers is None:
            api_key = load_weco_api_key()
            auth_headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

        # Get event properties
        properties = event.to_properties()
        # Send in background thread (fire-and-forget)
        thread = threading.Thread(
            target=_send_event_request,
            args=(
                event.event_name,
                event.timestamp,
                ctx.installation_id,
                ctx.invocation_id,
                ctx.client_version,
                ctx.invoked_via,
                properties,
                auth_headers,
            ),
            daemon=True,
        )
        thread.start()
    except Exception:
        # Silently ignore all errors - event reporting should never break the CLI
        pass
