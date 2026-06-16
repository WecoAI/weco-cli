"""``DashboardSession`` — a live bridge between a local agent and the Weco dashboard.

Wraps the REST session row (via :class:`~weco.core.api.WecoClient`) and its
Supabase Realtime channel behind a four-call surface: ``create`` / ``offline``,
``publish``, ``run``, plus ``id`` / ``dashboard_url``. JWT refresh, heartbeat,
scrollback replay, payload chunking and graceful close all happen inside —
callers never see api keys, queues, or channel credentials.
"""

from __future__ import annotations

import asyncio
from typing import Optional

import requests

from weco.core.api import WecoClient

from . import relay
from .relay import OUTBOUND_QUEUE_MAX, InboundHandler, enqueue


class SetupError(Exception):
    """Raised when the dashboard session can't be created (HTTP/network)."""


class DashboardSession:
    """A dashboard-bridged agent session.

    Construct with :meth:`create` (which does the REST handshake) or
    :meth:`offline` (a no-op stand-in when the relay is unavailable). Start
    the channel with :meth:`run`; push transcript/meta JSON lines with
    :meth:`publish`.
    """

    def __init__(self, *, client: Optional[WecoClient], data: dict) -> None:
        self._client = client
        self._data = data  # Weco API response, or {} when offline
        self._outbound: asyncio.Queue[str] = asyncio.Queue(maxsize=OUTBOUND_QUEUE_MAX)

    @classmethod
    def create(cls, *, api_key: str, agent_type: str) -> "DashboardSession":
        """REST-create the session. Raises :class:`SetupError` on failure."""
        client = WecoClient({"Authorization": f"Bearer {api_key}"})
        try:
            data = client.create_session(agent_type)
        except requests.HTTPError as e:
            resp = e.response
            detail = (resp.text or "")[:200] if resp is not None else str(e)
            status = resp.status_code if resp is not None else "?"
            raise SetupError(f"HTTP {status}: {detail}") from e
        except requests.RequestException as e:
            raise SetupError(str(e)) from e
        return cls(client=client, data=data)

    @classmethod
    def offline(cls) -> "DashboardSession":
        """A no-op session: ``publish`` drops, ``run`` returns immediately, and
        the id/url are ``None``. Lets bridges skip ``if session`` checks."""
        return cls(client=None, data={})

    @property
    def online(self) -> bool:
        return self._client is not None and bool(self._data)

    @property
    def id(self) -> Optional[str]:
        return (self._data.get("session") or {}).get("id")

    @property
    def dashboard_url(self) -> Optional[str]:
        return self._data.get("dashboard_url")

    def publish(self, line: str) -> bool:
        """Queue a JSON line for broadcast to the dashboard. Returns False if
        the outbound queue is full (the line is dropped). Offline sessions
        accept lines into an undrained queue, same as the legacy behaviour."""
        return enqueue(self._outbound, line)

    async def run(self, *, on_inbound: InboundHandler, stop_event: asyncio.Event) -> None:
        """Open the Realtime channel and pump until ``stop_event`` is set.
        No-op for an offline session."""
        if not self.online:
            return
        assert self._client is not None  # online ⇒ client present
        await relay.run_channel(
            weco_client=self._client,
            session_data=self._data,
            outbound=self._outbound,
            on_inbound=on_inbound,
            stop_event=stop_event,
        )
