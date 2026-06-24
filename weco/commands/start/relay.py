"""Low-level Supabase Realtime channel pump for the dashboard bridge.

The high-level object the bridge uses is :class:`~weco.commands.start.session.DashboardSession`;
this module is the engine it drives. ``run_channel`` opens the channel for a
session, pumps its outbound queue to broadcasts (chunking large payloads),
keeps the JWT fresh + the session heartbeating, replays scrollback to late
joiners, and marks the session closed on exit. Inbound = control events from
the dashboard (``inject_prompt``, ``approval_response``, ``scrollback_request``)
re-synthesized into JSON-line form for the bridge dispatch code.

Wire format on the channel:
    event ``transcript_batch``   payload ``{"lines": [{"line", "seq"}, ...], "replay"?: bool}``
                                 (rate-limited live path + scrollback replay)
    event ``transcript_chunk``   payload ``{"chunk_id", "seq", "of", "data"}``  (lines >2.5MB)
    event ``transcript``         payload ``{"line": <json-line text>, "seq": <int>}``  (legacy/unused)
    event ``scrollback_replay``  payload ``{"line", "seq", "replay": true}``  (legacy/unused)
    event ``inject_prompt``      payload ``{"text", "id"?}``  (dashboard -> CLI)
    event ``approval_response``  payload ``{"id", "decision", "scope"?}``
    event ``scrollback_request`` payload ``{}``  (dashboard -> CLI)

Persistence model: nothing is written to Weco servers. Transcript history lives:
  * In-memory in this process (capped ring buffer) for replay-on-rejoin.
  * Optionally, on the user's disk at ~/.config/weco/sessions/<id>.jsonl
    (opt-out with ``WECO_NO_LOCAL_HISTORY=1``).

The CLI is the only owner-side publisher. The dashboard sends control messages
through the Weco API, which publishes them to the channel on the CLI's behalf.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import pathlib
import time
import uuid
from collections import deque
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import TYPE_CHECKING, Awaitable, Callable, Iterator, Optional, TextIO

from weco.core.api import SessionInactiveError

if TYPE_CHECKING:
    from weco.core.api import WecoClient


# --- Constants -----------------------------------------------------------------

OUTBOUND_QUEUE_MAX = 2048

# Supabase Realtime caps Broadcast payloads at ~3 MB; chunk anything larger.
# The threshold is in bytes, applied to the UTF-8 encoded JSON-line string.
MAX_PAYLOAD_BYTES = 2_500_000

# Coalescing publisher: enforce a minimum interval between transcript broadcasts
# so a single session can never exceed ~`1000 / RELAY_FLUSH_MS` messages/sec on
# Realtime, regardless of how fast the SDK emits. Lines that arrive within the
# interval are batched into one `transcript_batch` broadcast — coalesced, never
# dropped. This is a hard, self-imposed ceiling (the SDK's own ~10/s cadence is
# undocumented and version-dependent, so we don't rely on it). Override via env.
RELAY_FLUSH_MS = max(0, int(os.environ.get("WECO_RELAY_FLUSH_MS", "100")))

# JWT refresh: kick a fresh token in this many seconds before the current
# one would otherwise expire. Set generously so a short-lived expiry stalls
# don't drop the channel.
JWT_REFRESH_LEAD_SECONDS = 60

# In-memory scrollback ring buffer. Cap by total bytes of stored payloads;
# drop oldest events when full. ~4 MB covers a typical Claude session;
# enough for "user refreshed their dashboard tab" cases.
SCROLLBACK_BUFFER_BYTES = 4 * 1024 * 1024

# Coalesce scrollback replays: when one or more dashboard tabs ask for
# history, wait a short beat and replay the buffer ONCE for all of them.
# A replay is a broadcast every channel subscriber receives (deduped by
# seq), so one replay serves every tab that asked within the window.
#
# Trailing-edge on purpose. A leading-edge throttle (serve the first
# requester, drop everyone for the next N seconds) starves a tab that
# takes over the session moments after another joined — it asks, gets
# silently dropped, and shows empty history. Debouncing instead means a
# late/takeover request always schedules its own replay.
SCROLLBACK_REPLAY_DEBOUNCE = 1.5


# Errors during a session (websocket disconnects, etc.) can't be printed to the
# user's terminal because the wrapped agent owns it and our prints would corrupt
# its TUI. Log them to a file the user can tail if they want detail.
_LOG_DIR = pathlib.Path.home() / ".config" / "weco"
_LOG_FILE = _LOG_DIR / "session.log"
_SESSIONS_DIR = _LOG_DIR / "sessions"

# Cap the shared diagnostics log so it can't grow without bound across runs.
# It's low-volume (errors/info only, never the transcript), so a handful of
# small rotations is ample: at most MAX_BYTES x (BACKUP_COUNT + 1) on disk.
SESSION_LOG_MAX_BYTES = 5 * 1024 * 1024
SESSION_LOG_BACKUP_COUNT = 3


def _get_logger() -> logging.Logger:
    logger = logging.getLogger("weco.start.relay")
    if logger.handlers:
        return logger
    logger.propagate = False
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(_LOG_FILE, maxBytes=SESSION_LOG_MAX_BYTES, backupCount=SESSION_LOG_BACKUP_COUNT)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    except OSError:
        logger.addHandler(logging.NullHandler())
    return logger


logger = _get_logger()


# --- Outbound queue helpers ----------------------------------------------------


# Inbound handler: receives a single text frame (JSON line). Bridges parse it
# to dispatch on the ``type`` field.
InboundHandler = Callable[[str], Optional[Awaitable[None]]]


def enqueue(outbound: "asyncio.Queue[str]", text: str) -> bool:
    """Best-effort enqueue. Returns False (and drops the line) on overflow."""
    try:
        outbound.put_nowait(text)
        return True
    except asyncio.QueueFull:
        return False


# --- Scrollback buffer & local disk log ---------------------------------------


class ScrollbackBuffer:
    """Bounded in-memory buffer of recent transcript events.

    Each entry is a (seq, line) pair. Total stored bytes is capped; oldest
    entries get evicted on overflow. Used to replay history to dashboard
    tabs that join after the session started.
    """

    def __init__(self, max_bytes: int = SCROLLBACK_BUFFER_BYTES) -> None:
        self._items: "deque[tuple[int, str]]" = deque()
        self._bytes = 0
        self._max_bytes = max_bytes

    def append(self, seq: int, line: str) -> None:
        size = len(line.encode("utf-8"))
        self._items.append((seq, line))
        self._bytes += size
        while self._bytes > self._max_bytes and self._items:
            _, popped = self._items.popleft()
            self._bytes -= len(popped.encode("utf-8"))

    def snapshot(self) -> list[tuple[int, str]]:
        return list(self._items)

    def __len__(self) -> int:
        return len(self._items)


def _open_local_log(session_id: str) -> Optional[TextIO]:
    """Open a per-session JSONL file under ~/.config/weco/sessions/.

    Returns a writable file handle, or None if disabled or unavailable.
    """
    if os.environ.get("WECO_NO_LOCAL_HISTORY") == "1":
        return None
    try:
        _SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        return open(_SESSIONS_DIR / f"{session_id}.jsonl", "a", buffering=1, encoding="utf-8")
    except OSError as e:
        logger.info("local session log disabled: %s", e)
        return None


# --- Realtime client -----------------------------------------------------------


def _parse_iso(ts: str) -> float:
    """Parse a Supabase ISO-8601 timestamp into a unix timestamp."""
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()


def _chunk_utf8(s: str, max_bytes: int) -> Iterator[str]:
    """Split a string into chunks whose UTF-8 encoding fits under ``max_bytes``.

    Avoids splitting multi-byte sequences mid-character.
    """
    encoded = s.encode("utf-8")
    if len(encoded) <= max_bytes:
        yield s
        return
    pos = 0
    while pos < len(encoded):
        end = min(pos + max_bytes, len(encoded))
        # Walk back if we landed on a UTF-8 continuation byte (10xxxxxx).
        while end > pos and end < len(encoded) and (encoded[end] & 0xC0) == 0x80:
            end -= 1
        yield encoded[pos:end].decode("utf-8")
        pos = end


async def run_channel(
    *,
    weco_client: "WecoClient",
    session_data: dict,
    outbound: "asyncio.Queue[str]",
    on_inbound: InboundHandler,
    stop_event: asyncio.Event,
) -> None:
    """Connect to the Realtime channel for ``session_data`` and pump messages.

    Owns three internal tasks for the lifetime of the channel:
      * publisher  — drains ``outbound`` to broadcasts (with chunking)
      * refresher  — keeps the JWT fresh by minting new ones server-side
      * heartbeater — rolls the session TTL forward

    Also listens for ``scrollback_request`` events and replays the in-memory
    buffer in response. Marks the session closed on exit. Returns when
    ``stop_event`` is set or the connection terminates.
    """
    try:
        from realtime import AsyncRealtimeClient
    except ImportError:
        # Logged silently; printing to TTY would corrupt the wrapped agent's UI.
        logger.warning("realtime package not installed; dashboard bridge disabled.")
        return

    rt_creds = session_data.get("realtime") or {}
    session_id = session_data["session"]["id"]
    rt_url = rt_creds["url"]
    rt_token = rt_creds["token"]
    channel_name = rt_creds["channel"]
    expires_at_unix = _parse_iso(rt_creds["expires_at"])

    # The gateway gates the WebSocket handshake on `apikey` and rejects a user
    # JWT there (HTTP 401). Connect with the public anon key as apikey, then
    # authenticate the user via `set_auth(rt_token)` before subscribing — that
    # user token is what authorizes the channel join.
    rt_apikey = rt_creds["apikey"]

    client = AsyncRealtimeClient(rt_url, rt_apikey, auto_reconnect=True)

    seq_counter = {"value": 0}  # CLI-assigned monotonic seq per session
    scrollback = ScrollbackBuffer()
    expiry_state = {"unix": expires_at_unix}
    replay_pending: dict = {"task": None}
    local_log = _open_local_log(session_id)

    try:
        await client.connect()
    except Exception as e:
        if not stop_event.is_set():
            logger.warning("Realtime connect failed for session %s: %s", session_id, e)
        if local_log is not None:
            try:
                local_log.close()
            except Exception:
                pass
        return

    channel = client.channel(channel_name, {"config": {"private": True}})

    # Inbound: synthesize JSON-line strings so existing bridge dispatch keeps
    # working. The original protocol used `{"type": "inject_prompt", ...}`
    # text frames; we re-emit that envelope from broadcast event + payload.
    def _on_control_broadcast(event_type: str):
        def _cb(message: dict) -> None:
            payload = (message or {}).get("payload") or {}
            try:
                line = json.dumps({"type": event_type, **payload})
            except (TypeError, ValueError):
                return
            result = on_inbound(line)
            if asyncio.iscoroutine(result):
                asyncio.create_task(result)

        return _cb

    channel.on_broadcast("inject_prompt", _on_control_broadcast("inject_prompt"))
    channel.on_broadcast("approval_response", _on_control_broadcast("approval_response"))
    # AskUserQuestion answers — Claude SDK's clarifying-question gate
    # round-trips through the same control plane as tool approvals.
    channel.on_broadcast("question_response", _on_control_broadcast("question_response"))
    # Dashboard stop button — asks the CLI to abort the in-flight turn
    # without tearing down the session (same semantics as a local Ctrl-C).
    channel.on_broadcast("interrupt", _on_control_broadcast("interrupt"))
    # Dashboard "Explore a new path" — structured derived-run request the
    # bridge turns into agent instructions (not a plain chat message).
    channel.on_broadcast("derive_request", _on_control_broadcast("derive_request"))

    def _on_scrollback_request(_message: dict) -> None:
        # A replay is already queued — it broadcasts to every subscriber,
        # so this requester is covered. Coalesce instead of stacking.
        if replay_pending["task"] is not None:
            return

        async def _debounced_replay() -> None:
            try:
                await asyncio.sleep(SCROLLBACK_REPLAY_DEBOUNCE)
                if stop_event.is_set():
                    return
                await _replay_scrollback(channel, scrollback)
            except asyncio.CancelledError:
                raise
            finally:
                replay_pending["task"] = None

        replay_pending["task"] = asyncio.create_task(_debounced_replay())

    channel.on_broadcast("scrollback_request", _on_scrollback_request)

    # Authenticate the user before joining: the channel join carries the
    # socket's access_token, which authorizes subscribe/publish on the session's
    # channel. The anon key in the URL only gets us past the connection
    # handshake; it has no per-channel authority on its own.
    try:
        await client.set_auth(rt_token)
    except Exception as e:
        if not stop_event.is_set():
            logger.warning("Realtime set_auth failed for session %s: %s", session_id, e)

    try:
        await channel.subscribe()
    except Exception as e:
        if not stop_event.is_set():
            logger.warning("Realtime subscribe failed for session %s: %s", session_id, e)
        await _safe_close(client)
        if local_log is not None:
            try:
                local_log.close()
            except Exception:
                pass
        return

    publisher = asyncio.create_task(_publisher_loop(channel, outbound, scrollback, seq_counter, local_log, stop_event))
    refresher = asyncio.create_task(_jwt_refresher(client, channel, weco_client, session_id, expiry_state, stop_event))
    heartbeater = asyncio.create_task(_heartbeat_loop(weco_client, session_id, stop_event))

    try:
        # Sleep until something tells us to stop (stop_event, or any task ending).
        stop_task = asyncio.create_task(stop_event.wait())
        await asyncio.wait({stop_task, publisher, refresher, heartbeater}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        stop_event.set()
        pending_replay = replay_pending["task"]
        if pending_replay is not None:
            pending_replay.cancel()
        for t in (publisher, refresher, heartbeater):
            t.cancel()
        for t in (publisher, refresher, heartbeater):
            try:
                await asyncio.wait_for(t, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                pass
        # Graceful-exit close: tells the Weco API the session is over so the
        # dashboard's active list updates immediately rather than waiting for
        # the TTL to elapse. Run in a thread so we don't block the event loop's
        # teardown if the network is slow. Best-effort — TTL expiry is the
        # backstop if this fails.
        try:
            await asyncio.to_thread(weco_client.close_session, session_id)
        except Exception:
            pass
        await _safe_close(client)
        if local_log is not None:
            try:
                local_log.close()
            except Exception:
                pass


async def _safe_close(client) -> None:
    try:
        await client.close()
    except Exception:
        pass


async def _send_entries(channel, entries: list[tuple[int, str]], *, replay: bool = False) -> None:
    """Emit ``(seq, line)`` pairs as coalesced ``transcript_batch`` broadcasts.

    Lines are packed into batches kept under ``MAX_PAYLOAD_BYTES`` so a normal
    flush is a SINGLE broadcast. A lone line larger than the cap falls back to
    the ``transcript_chunk`` path (reassembled client-side). ``replay=True``
    marks the batch as historical so the dashboard dedupes it against anything
    already rendered live (by ``seq``).
    """
    group: list[dict] = []
    group_bytes = 0

    async def flush_group() -> None:
        nonlocal group, group_bytes
        if not group:
            return
        payload: dict = {"lines": group}
        if replay:
            payload["replay"] = True
        await channel.send_broadcast("transcript_batch", payload)
        group = []
        group_bytes = 0

    for seq, line in entries:
        size = len(line.encode("utf-8"))
        if size > MAX_PAYLOAD_BYTES:
            # Oversized single line (rare; never in chat): flush the batch so far,
            # then chunk this one on its own. Adds frames only for >2.5MB lines.
            await flush_group()
            chunk_id = uuid.uuid4().hex
            pieces = list(_chunk_utf8(line, MAX_PAYLOAD_BYTES))
            of = len(pieces)
            for i, piece in enumerate(pieces):
                await channel.send_broadcast("transcript_chunk", {"chunk_id": chunk_id, "seq": i, "of": of, "data": piece})
            continue
        if group and group_bytes + size > MAX_PAYLOAD_BYTES:
            await flush_group()
        group.append({"seq": seq, "line": line})
        group_bytes += size
    await flush_group()


async def _publisher_loop(
    channel,
    outbound: "asyncio.Queue[str]",
    scrollback: ScrollbackBuffer,
    seq_counter: dict,
    local_log,
    stop_event: asyncio.Event,
) -> None:
    """Drain the outbound queue and publish lines as rate-limited broadcasts.

    Enforces a minimum ``RELAY_FLUSH_MS`` gap between broadcasts: lines that
    arrive within the gap are coalesced into one ``transcript_batch`` (nothing is
    dropped), guaranteeing a single session can't exceed ~``1000/RELAY_FLUSH_MS``
    messages/sec on Realtime — independent of how fast the SDK emits. When
    traffic is sparse the gap is already satisfied, so a lone line is sent
    immediately (no added latency); throttling only kicks in under load.

    Each line is appended to the in-memory ring buffer (for live replay) and to
    the optional local disk log before broadcast.
    """
    loop = asyncio.get_running_loop()
    flush_s = RELAY_FLUSH_MS / 1000.0
    last_send = 0.0
    while not stop_event.is_set():
        try:
            first = await asyncio.wait_for(outbound.get(), timeout=0.5)
        except asyncio.TimeoutError:
            continue

        # Throttle: ensure >= flush_s since the last broadcast. While we wait,
        # more lines pile into `outbound` and get coalesced into this batch.
        wait = flush_s - (loop.time() - last_send)
        if wait > 0:
            await asyncio.sleep(wait)

        batch = [first]
        while True:
            try:
                batch.append(outbound.get_nowait())
            except asyncio.QueueEmpty:
                break

        # Persist to in-memory buffer + local disk before broadcast — guarantees
        # a tab that joins right after a line is sent will see it on replay,
        # even if the broadcast itself is lost in flight.
        entries: list[tuple[int, str]] = []
        for line in batch:
            seq_counter["value"] += 1
            seq = seq_counter["value"]
            scrollback.append(seq, line)
            if local_log is not None:
                try:
                    local_log.write(json.dumps({"seq": seq, "line": line}) + "\n")
                except Exception as e:
                    logger.info("local log write failed: %s", e)
            entries.append((seq, line))

        try:
            await _send_entries(channel, entries)
            last_send = loop.time()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # Don't crash the publisher on a single bad batch — log and move on.
            logger.warning("publisher send failed: %s", e)


async def _replay_scrollback(channel, scrollback: ScrollbackBuffer) -> None:
    """Re-broadcast the buffer as ``transcript_batch`` frames marked ``replay``.

    Batched (size-partitioned under MAX_PAYLOAD_BYTES) so a reconnect replays the
    whole history in a handful of broadcasts instead of one-per-line — keeping
    the per-session message-rate ceiling intact even on takeover bursts. The
    dashboard dedupes by ``seq`` against anything already rendered live.
    """
    items = scrollback.snapshot()
    if not items:
        return
    try:
        await _send_entries(channel, items, replay=True)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.info("scrollback replay failed: %s", e)


HEARTBEAT_INTERVAL_SECONDS = 30.0


async def _heartbeat_loop(weco_client: "WecoClient", session_id: str, stop_event: asyncio.Event) -> None:
    """Tell the Weco API we're still alive every ~30s.

    Each ping rolls the session's `expires_at` forward (the server stamps
    `now + 2 minutes` per heartbeat). When this loop stops, the TTL elapses
    and the server marks the session expired.

    Best-effort: a failed ping is logged and we wait for the next tick.
    The grace built into the TTL (4x the heartbeat interval) tolerates a
    couple of consecutive failures before the session is considered stale.
    """
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=HEARTBEAT_INTERVAL_SECONDS)
            return  # stop_event set
        except asyncio.TimeoutError:
            pass

        try:
            await asyncio.to_thread(weco_client.session_heartbeat, session_id)
        except SessionInactiveError:
            # Session is gone for good (closed, or expired past the revive
            # grace). Tear the bridge down instead of pinging a dead session
            # forever — the supervisor cancels the sibling loops on stop.
            logger.info("session %s is no longer active; stopping dashboard bridge", session_id)
            stop_event.set()
            return
        except Exception as e:
            logger.info("heartbeat failed for session %s: %s", session_id, e)


async def _jwt_refresher(
    rt_client, channel, weco_client: "WecoClient", session_id: str, expiry_state: dict, stop_event: asyncio.Event
) -> None:
    """Mint a new JWT before the current one expires; push it into the channel."""
    while not stop_event.is_set():
        now = time.time()
        sleep_for = max(1.0, expiry_state["unix"] - now - JWT_REFRESH_LEAD_SECONDS)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=sleep_for)
            return  # stop_event set
        except asyncio.TimeoutError:
            pass

        try:
            fresh = await asyncio.to_thread(weco_client.refresh_realtime_token, session_id)
        except SessionInactiveError:
            # Session is gone for good (closed, or expired past the revive
            # grace) — minting will never succeed again. Stop the bridge rather
            # than 409-storm the endpoint every 5s until the user Ctrl-Cs.
            logger.info("session %s is no longer active; stopping dashboard bridge", session_id)
            stop_event.set()
            return
        except Exception as e:
            logger.warning("JWT refresh failed for session %s: %s", session_id, e)
            # Back off and retry; if expiry has passed the channel will drop and
            # auto-reconnect on the next attempt with whatever token we have.
            await asyncio.sleep(5.0)
            continue

        try:
            await rt_client.set_auth(fresh["token"])
        except Exception as e:
            logger.warning("set_auth failed for session %s: %s", session_id, e)
            continue

        # Keep the channel's *join* payload token current. realtime-py bakes the
        # access_token into the join push once at subscribe() time and reuses it
        # verbatim on every reconnect rejoin (join_push.resend); set_auth updates
        # live channels but NOT that baked payload. So after a reconnect that
        # outlives the original 60-min JWT — now common inside the 2h revive
        # grace — the channel would rejoin with a dead token, authorization
        # fails, and inbound broadcasts (dashboard -> CLI) silently stop while
        # outbound still limps through (push() only checks _joined_once, not the
        # real join state). Re-stamping here means any future rejoin carries a
        # live token; forcing a rejoin when we're not currently joined recovers a
        # channel a reconnect already broke, promptly rather than on the rejoin
        # timer's backoff.
        try:
            channel.join_push.update_payload({"access_token": fresh["token"]})
            if not channel.is_joined:
                await channel._rejoin()
        except Exception as e:
            logger.warning("channel rejoin after token refresh failed for session %s: %s", session_id, e)

        expiry_state["unix"] = _parse_iso(fresh["expires_at"])
