"""Tests for the low-level Realtime channel pump (`relay`).

Live Realtime connection / JWT refresh / channel subscription is exercised
end-to-end against a real Supabase instance, not here. These cover the
local-only seams: UTF-8-safe chunking and timestamp parsing. Session REST
now lives on ``WecoClient`` / ``DashboardSession`` (see ``test_start_session``).
"""

from __future__ import annotations

import asyncio

from unittest.mock import AsyncMock, MagicMock

from weco.commands.start import relay
from weco.core.api import SessionInactiveError


# --- _chunk_utf8 ---------------------------------------------------------------


def test_chunk_short_strings_pass_through_unchanged():
    chunks = list(relay._chunk_utf8("hello", 100))
    assert chunks == ["hello"]


def test_chunk_splits_long_ascii_at_byte_limit():
    s = "a" * 1000
    chunks = list(relay._chunk_utf8(s, 256))
    assert "".join(chunks) == s
    assert all(len(c.encode("utf-8")) <= 256 for c in chunks)
    assert len(chunks) == 4  # 1000 / 256 rounded up


def test_chunk_does_not_split_multibyte_chars():
    # 4-byte char repeated; naive byte slicing would corrupt mid-char.
    s = "𠮷" * 200  # each char is 4 bytes in UTF-8
    chunks = list(relay._chunk_utf8(s, 17))  # awkward boundary
    assert "".join(chunks) == s
    for c in chunks:
        c.encode("utf-8").decode("utf-8")


# --- _parse_iso ----------------------------------------------------------------


def test_parse_iso_handles_z_suffix():
    ts = relay._parse_iso("2026-05-05T12:00:00Z")
    # Just sanity: it's a unix timestamp around the right magnitude.
    assert ts > 1_700_000_000


# --- terminal 409 stops the bridge ---------------------------------------------


def test_heartbeat_loop_stops_bridge_on_session_inactive(monkeypatch):
    monkeypatch.setattr(relay, "HEARTBEAT_INTERVAL_SECONDS", 0.01)
    client = MagicMock()
    client.session_heartbeat.side_effect = SessionInactiveError("sess-1")
    stop_event = asyncio.Event()

    async def _run():
        await asyncio.wait_for(relay._heartbeat_loop(client, "sess-1", stop_event), timeout=2.0)

    asyncio.run(_run())
    assert stop_event.is_set()


def test_heartbeat_loop_keeps_going_on_transient_error(monkeypatch):
    monkeypatch.setattr(relay, "HEARTBEAT_INTERVAL_SECONDS", 0.01)
    client = MagicMock()
    calls = {"n": 0}

    def _flaky(_sid):
        calls["n"] += 1
        if calls["n"] >= 3:
            stop_event.set()
        raise RuntimeError("network blip")

    client.session_heartbeat.side_effect = _flaky
    stop_event = asyncio.Event()

    async def _run():
        await asyncio.wait_for(relay._heartbeat_loop(client, "sess-1", stop_event), timeout=2.0)

    asyncio.run(_run())
    # Transient errors don't tear the bridge down — it kept pinging.
    assert calls["n"] >= 3


def test_jwt_refresher_stops_bridge_on_session_inactive(monkeypatch):
    client = MagicMock()
    client.refresh_realtime_token.side_effect = SessionInactiveError("sess-1")
    stop_event = asyncio.Event()
    # Already-elapsed expiry → the loop's sleep floors at 1s, then refreshes.
    expiry_state = {"unix": relay.time.time() - 1000}

    async def _run():
        await asyncio.wait_for(
            relay._jwt_refresher(MagicMock(), MagicMock(), client, "sess-1", expiry_state, stop_event), timeout=3.0
        )

    asyncio.run(_run())
    assert stop_event.is_set()


def test_jwt_refresher_restamps_join_token_and_rejoins_after_reconnect():
    # After a reconnect that outlived the original JWT, the channel rejoins with
    # a stale baked-in token unless we re-stamp the join payload on refresh.
    fresh = {"token": "T2", "expires_at": "2999-01-01T00:00:00+00:00"}
    weco_client = MagicMock()
    stop_event = asyncio.Event()

    def _mint(_sid):
        stop_event.set()  # let the loop run exactly one iteration
        return fresh

    weco_client.refresh_realtime_token.side_effect = _mint

    rt_client = MagicMock()
    rt_client.set_auth = AsyncMock()

    channel = MagicMock()
    channel.is_joined = False  # a reconnect left it un-joined
    channel._rejoin = AsyncMock()

    expiry_state = {"unix": relay.time.time() - 1000}

    async def _run():
        await asyncio.wait_for(
            relay._jwt_refresher(rt_client, channel, weco_client, "sess-1", expiry_state, stop_event), timeout=3.0
        )

    asyncio.run(_run())
    rt_client.set_auth.assert_awaited_once_with("T2")
    channel.join_push.update_payload.assert_called_once_with({"access_token": "T2"})
    channel._rejoin.assert_awaited_once()
