"""Wrapper-side weco-run watcher.

Polls ``weco run status <id>`` and emits structured update events to a caller-
supplied notifier. The notifier renders directly to the user's terminal and
broadcasts to the dashboard — claude is not in the loop for these pings, which
makes them fast (no LLM round-trip), cheap (no tokens), and predictable
(deterministic format).

Each update is a dict shaped like::

    {
        "kind": "step_advance" | "new_best" | "idle" | "completed" | "errored" | "stopped" | "pending_review" | "attached",
        "run_id": "...",
        # Lineage root id (= the original non-derived run). For a derived
        # sub-run this differs from `run_id`; the dashboard navigates/associates
        # by `lineage_id` so a sub-run surfaces under its root rather than as a
        # dead-end tab on its own id. None until the first status poll resolves it.
        "lineage_id": "..." | None,
        "level": "info" | "success" | "warning" | "error",
        "text": "step 3/15 done, best still 7.51x at step 0",
        "hints": ["weco run results <id> --top 5", ...],  # optional follow-up commands
        # Plus event-specific fields where useful for richer dashboard rendering:
        # current_step, total_steps, best_metric, best_step, idle_seconds, ...
    }

The wrapper renders the update locally with an icon + colour by level/kind and
forwards it to the dashboard as a ``_weco_meta:run_update`` broadcast.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Awaitable, Callable, Optional, Union


# Idle heartbeat: if nothing has changed for this many seconds, surface a
# "still running, no progress yet" prompt so the user gets reassured that
# the run hasn't quietly stalled. Re-fires every IDLE_HEARTBEAT_SECONDS while
# state remains unchanged.
IDLE_HEARTBEAT_SECONDS = 60.0


# Anchored regex: `Run ID: <uuid>` on its own line is what `weco run` emits in
# its plain output. UUID v4 by spec but we accept any 36-char hyphenated form
# to avoid coupling to the format too tightly.
_RUN_ID_RE = re.compile(r"^Run ID: ([0-9a-fA-F-]{36})\b", re.MULTILINE)


# Notifier signature: receives one update dict per call. May be sync or async.
UpdateNotifier = Callable[[dict], Union[None, Awaitable[None]]]


def find_run_ids(text: str) -> list[str]:
    """Return the run ids appearing in `text` (de-duplicated, in order).

    Handles two input shapes:
      - raw text (e.g. from a plain stdout dump)
      - claude's stream-json JSONL line, where tool_result content has the
        run-id line buried inside a JSON-escaped string (newlines as ``\\n``,
        not real ``\\n``). We decode the JSON first and search the
        nested string values so the anchored regex still works on real
        newlines.
    """
    seen: set[str] = set()
    out: list[str] = []

    def take(s: str) -> None:
        for match in _RUN_ID_RE.finditer(s):
            rid = match.group(1)
            if rid not in seen:
                seen.add(rid)
                out.append(rid)

    # Try JSON first — the common case for stream-json.
    try:
        decoded = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        decoded = None

    if decoded is not None:
        _walk_strings(decoded, take)
    else:
        take(text)

    return out


def _walk_strings(obj, on_string) -> None:
    """Walk a JSON-decoded value, calling ``on_string`` with each string leaf."""
    if isinstance(obj, str):
        on_string(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _walk_strings(v, on_string)
    elif isinstance(obj, list):
        for v in obj:
            _walk_strings(v, on_string)


# Statuses that mean "stop polling".
_TERMINAL_STATUSES = {"completed", "complete", "error", "errored", "stopped", "terminated", "cancelled"}


class RunWatcher:
    """Polls ``weco run status <id>`` and posts structured progress updates.

    A ``RunWatcher`` may be watching multiple runs concurrently. Each watch is
    its own asyncio task; cancelling the watcher cancels all of them.
    """

    def __init__(
        self, *, weco_bin: Optional[str], notify: UpdateNotifier, stop_event: asyncio.Event, poll_interval_s: float = 10.0
    ) -> None:
        # Caller is responsible for resolution (so tests can pass None to
        # exercise the no-bin path independent of the host's PATH).
        self.weco_bin = weco_bin
        self.notify = notify
        self.stop_event = stop_event
        self.poll_interval_s = poll_interval_s
        self._tasks: dict[str, asyncio.Task] = {}

    def watch(self, run_id: str) -> bool:
        """Begin watching ``run_id``. Returns True if a new watch was started,
        False if we were already watching it."""
        if run_id in self._tasks and not self._tasks[run_id].done():
            return False
        if not self.weco_bin:
            return False
        self._tasks[run_id] = asyncio.create_task(self._poll(run_id))
        return True

    def watching(self) -> list[str]:
        return [rid for rid, t in self._tasks.items() if not t.done()]

    async def stop(self) -> None:
        for task in list(self._tasks.values()):
            if not task.done():
                task.cancel()
        for task in list(self._tasks.values()):
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                pass
        self._tasks.clear()

    # --- Internals ---

    async def _emit(self, update: Optional[dict]) -> None:
        if not update:
            return
        result = self.notify(update)
        if asyncio.iscoroutine(result):
            await result

    async def _poll(self, run_id: str) -> None:
        last_status: Optional[str] = None
        last_best_step: Optional[int] = None
        last_current_step: Optional[int] = None
        last_pending_count: Optional[int] = None
        last_change_at = time.monotonic()
        last_heartbeat_at = last_change_at
        # Fire an immediate "attached" event so consumers (dashboard
        # toast, terminal renderer) learn the run id without waiting
        # for the first poll cycle. `build_status_change_message`
        # suppresses the same kind on cur_step == 0 (which is almost
        # always true at run start), so we'd otherwise wait minutes
        # before the dashboard learns there's a new run.
        await self._emit({"kind": "attached", "run_id": run_id, "level": "info", "text": "watching run", "hints": []})
        try:
            while not self.stop_event.is_set():
                await asyncio.sleep(self.poll_interval_s)
                status = await self._fetch_status(run_id)
                if status is None:
                    continue
                cur_status = (status.get("status") or "").lower()
                cur_best_step = status.get("best_step")
                cur_step = status.get("current_step")
                pending_nodes = status.get("pending_nodes") or []
                cur_pending_count = len(pending_nodes)

                changed = False
                # Status change wins outright — covers terminal states
                # (completed/error/stopped) and any movement out of "running".
                if cur_status and cur_status != last_status:
                    await self._emit(build_status_change_message(run_id, status, prev=last_status))
                    changed = True
                else:
                    # Within a single status (typically "running"), fire on
                    # whichever finer-grained signal advanced this poll. We
                    # check best-step first so a step-with-improvement reads
                    # as the more interesting message rather than a generic
                    # progress tick.
                    update: Optional[dict] = None
                    if cur_best_step is not None and last_best_step is not None and cur_best_step != last_best_step:
                        update = build_new_best_message(run_id, status)
                        changed = True
                    elif cur_step is not None and last_current_step is not None and cur_step != last_current_step:
                        update = build_step_advance_message(run_id, status)
                        changed = True
                    elif last_pending_count is not None and cur_pending_count > last_pending_count:
                        update = build_pending_review_message(run_id, status)
                        changed = True
                    await self._emit(update)

                now = time.monotonic()
                if changed:
                    last_change_at = now
                    last_heartbeat_at = now
                elif (
                    cur_status == "running"
                    and now - last_change_at >= IDLE_HEARTBEAT_SECONDS
                    and now - last_heartbeat_at >= IDLE_HEARTBEAT_SECONDS
                ):
                    await self._emit(build_idle_heartbeat_message(run_id, status, idle_seconds=now - last_change_at))
                    last_heartbeat_at = now

                last_status = cur_status
                if cur_best_step is not None:
                    last_best_step = cur_best_step
                if cur_step is not None:
                    last_current_step = cur_step
                last_pending_count = cur_pending_count
                if cur_status in _TERMINAL_STATUSES:
                    return
        except asyncio.CancelledError:
            return

    async def _fetch_status(self, run_id: str) -> Optional[dict]:
        if not self.weco_bin:
            return None
        try:
            proc = await asyncio.create_subprocess_exec(
                self.weco_bin, "run", "status", run_id, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL
            )
            try:
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10.0)
            except asyncio.TimeoutError:
                proc.kill()
                return None
        except (FileNotFoundError, OSError):
            return None
        try:
            return json.loads(stdout.decode("utf-8", errors="replace"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None


# --- Update builders -----------------------------------------------------------
#
# Each returns a dict shaped per the module docstring, or None if there isn't
# enough information in `status` to construct a meaningful update.


def _format_metric(value) -> str:
    """Render a numeric metric compactly (3 sig figs, strip trailing zeros)."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if f == int(f):
        return f"{int(f)}"
    return f"{f:.3g}"


def _progress_str(cur_step, total) -> str:
    if cur_step is None:
        return ""
    if total is None:
        return f"step {cur_step}"
    return f"step {cur_step}/{total}"


def _best_clause(best, best_step) -> str:
    if best is None or best_step is None:
        return ""
    return f"best {_format_metric(best)} at step {best_step}"


def build_new_best_message(run_id: str, status: dict) -> Optional[dict]:
    """User-direct update for a new best metric."""
    best = status.get("best_metric")
    best_step = status.get("best_step")
    cur_step = status.get("current_step")
    total = status.get("total_steps")
    if best is None or best_step is None:
        return None
    progress = _progress_str(cur_step, total)
    progress_part = f" ({progress})" if progress else ""
    return {
        "kind": "new_best",
        "run_id": run_id,
        "lineage_id": status.get("lineage_id"),
        "level": "success",
        "text": f"new best {_format_metric(best)} at step {best_step}{progress_part}",
        "hints": [],
        "current_step": cur_step,
        "total_steps": total,
        "best_metric": best,
        "best_step": best_step,
    }


def build_step_advance_message(run_id: str, status: dict) -> Optional[dict]:
    """User-direct update for a generic step transition (no improvement)."""
    cur_step = status.get("current_step")
    total = status.get("total_steps")
    if cur_step is None:
        return None
    best = status.get("best_metric")
    best_step = status.get("best_step")
    progress = _progress_str(cur_step, total)
    bc = _best_clause(best, best_step)
    text = f"{progress} done"
    if bc:
        text += f" — {bc}"
    return {
        "kind": "step_advance",
        "run_id": run_id,
        "lineage_id": status.get("lineage_id"),
        "level": "info",
        "text": text,
        "hints": [],
        "current_step": cur_step,
        "total_steps": total,
        "best_metric": best,
        "best_step": best_step,
    }


def build_idle_heartbeat_message(run_id: str, status: dict, *, idle_seconds: float) -> Optional[dict]:
    """User-direct update when the run has been quiet for a while."""
    cur_step = status.get("current_step")
    total = status.get("total_steps")
    best = status.get("best_metric")
    best_step = status.get("best_step")
    idle_mins = max(1, int(round(idle_seconds / 60)))
    plural = "minute" if idle_mins == 1 else "minutes"
    progress = _progress_str(cur_step, total)
    progress_part = f" ({progress})" if progress else ""
    bc = _best_clause(best, best_step)
    bc_part = f" — {bc}" if bc else ""
    return {
        "kind": "idle",
        "run_id": run_id,
        "lineage_id": status.get("lineage_id"),
        "level": "info",
        "text": f"still running{progress_part}, no progress in ~{idle_mins} {plural}{bc_part}",
        "hints": [],
        "current_step": cur_step,
        "total_steps": total,
        "best_metric": best,
        "best_step": best_step,
        "idle_seconds": idle_seconds,
    }


def build_pending_review_message(run_id: str, status: dict) -> Optional[dict]:
    """User-direct update when nodes are awaiting review/evaluation."""
    pending = status.get("pending_nodes") or []
    if not pending:
        return None
    count = len(pending)
    plural = "node" if count == 1 else "nodes"
    return {
        "kind": "pending_review",
        "run_id": run_id,
        "lineage_id": status.get("lineage_id"),
        "level": "warning",
        "text": f"{count} {plural} awaiting review",
        "hints": [f"weco run review {run_id}"],
        "pending_count": count,
    }


def build_status_change_message(run_id: str, status: dict, *, prev: Optional[str]) -> Optional[dict]:
    """User-direct update for a status delta. Pulled out of ``RunWatcher`` so it
    can be unit-tested without touching async machinery."""
    cur = (status.get("status") or "").lower()
    if not cur:
        return None
    best = status.get("best_metric")
    best_step = status.get("best_step")
    cur_step = status.get("current_step")
    total = status.get("total_steps")

    progress = _progress_str(cur_step, total)
    progress_part = f" ({progress})" if progress else ""
    bc = _best_clause(best, best_step)
    bc_part = f" — {bc}" if bc else ""

    if cur in ("completed", "complete"):
        return {
            "kind": "completed",
            "run_id": run_id,
            "lineage_id": status.get("lineage_id"),
            "level": "success",
            "text": f"run completed{progress_part}{bc_part}",
            "hints": [f"weco run results {run_id} --top 5", f"weco run diff {run_id} --step best"],
            "current_step": cur_step,
            "total_steps": total,
            "best_metric": best,
            "best_step": best_step,
        }
    if cur in ("error", "errored"):
        return {
            "kind": "errored",
            "run_id": run_id,
            "lineage_id": status.get("lineage_id"),
            "level": "error",
            "text": f"run errored{progress_part}{bc_part}",
            "hints": [f"weco run status {run_id}"],
            "current_step": cur_step,
            "total_steps": total,
            "best_metric": best,
            "best_step": best_step,
        }
    if cur in ("stopped", "terminated", "cancelled"):
        return {
            "kind": "stopped",
            "run_id": run_id,
            "lineage_id": status.get("lineage_id"),
            "level": "warning",
            "text": f"run stopped (status: {cur}){progress_part}{bc_part}",
            "hints": [f"weco resume {run_id} --daemon"],
            "current_step": cur_step,
            "total_steps": total,
            "best_metric": best,
            "best_step": best_step,
        }
    if prev is None:
        # First observation of a non-terminal status. `_poll` already emitted
        # an immediate `attached` event for run pickup, so don't announce a
        # second one here — progress flows via step_advance / new_best / idle.
        return None
    # Generic running-state transition.
    if cur == "running" and cur != prev:
        return {
            "kind": "attached",
            "run_id": run_id,
            "lineage_id": status.get("lineage_id"),
            "level": "info",
            "text": f"now {cur}{progress_part}{bc_part}",
            "hints": [],
            "current_step": cur_step,
            "total_steps": total,
            "best_metric": best,
            "best_step": best_step,
        }
    return None
