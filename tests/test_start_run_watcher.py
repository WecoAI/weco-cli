"""Tests for the wrapper-side weco-run watcher."""

import asyncio
import json
from unittest.mock import patch

from weco.commands.start.run_watcher import (
    RunWatcher,
    build_idle_heartbeat_message,
    build_new_best_message,
    build_pending_review_message,
    build_status_change_message,
    build_step_advance_message,
    find_run_ids,
)


# --- find_run_ids ---


def test_find_run_ids_picks_canonical_format():
    text = "Run ID: a14ca1c1-56a7-4ae6-b054-51741adfbee5\nRun Name: hello\n"
    assert find_run_ids(text) == ["a14ca1c1-56a7-4ae6-b054-51741adfbee5"]


def test_find_run_ids_dedupes_and_preserves_order():
    text = (
        "Run ID: aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee\n"
        "Run ID: 11111111-2222-3333-4444-555555555555\n"
        "Run ID: aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee\n"
    )
    assert find_run_ids(text) == ["aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", "11111111-2222-3333-4444-555555555555"]


def test_find_run_ids_returns_empty_when_absent():
    assert find_run_ids("just a normal line\nno ids here") == []


def test_find_run_ids_only_matches_anchored_lines():
    """The marker has to start the line — random "Run ID:" mentions in prose
    shouldn't trigger a false start of polling."""
    text = "I'm thinking about: Run ID: aaaa-..."
    assert find_run_ids(text) == []


def test_find_run_ids_extracts_from_stream_json_tool_result():
    """The bridge feeds us claude's stream-json line. The run-id line is buried
    in a JSON-escaped string with `\\n` literals — we should still find it."""
    rid = "a14ca1c1-56a7-4ae6-b054-51741adfbee5"
    event = {
        "type": "user",
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "content": (f"WECO OPTIMIZATION RUN\n============================\nRun ID: {rid}\nRun Name: hello\n"),
                }
            ],
        },
    }
    assert find_run_ids(json.dumps(event)) == [rid]


# --- build_status_change_message ---


def test_build_status_change_message_for_completion_includes_results_hint():
    msg = build_status_change_message(
        "rid-1",
        {"status": "completed", "best_metric": 3.21, "best_step": 7, "current_step": 15, "total_steps": 15},
        prev="running",
    )
    assert msg is not None
    assert msg["kind"] == "completed"
    assert msg["level"] == "success"
    assert "completed" in msg["text"]
    assert "3.21" in msg["text"]
    assert any("weco run results rid-1" in h for h in msg["hints"])


def test_build_status_change_message_for_error_marks_error_level():
    msg = build_status_change_message("rid-2", {"status": "error", "current_step": 3, "total_steps": 15}, prev="running")
    assert msg is not None
    assert msg["kind"] == "errored"
    assert msg["level"] == "error"
    assert "errored" in msg["text"]


def test_build_status_change_message_skips_first_observation_when_idle():
    """A run we just started watching that's already in 'running' but step 0
    isn't worth a message — it's just initial pickup."""
    out = build_status_change_message("r", {"status": "running", "current_step": 0, "total_steps": 15}, prev=None)
    assert out is None


def test_build_status_change_message_skips_first_observation_even_mid_run():
    """`_poll` already emits an immediate `attached` event for run pickup, so
    a first observation of a running run — even one already past step 0 —
    must not produce a duplicate announcement here."""
    out = build_status_change_message("r", {"status": "running", "current_step": 8, "total_steps": 15}, prev=None)
    assert out is None


def test_build_status_change_message_returns_none_when_status_missing():
    assert build_status_change_message("r", {}, prev=None) is None


def test_build_status_change_message_handles_stopped_status():
    msg = build_status_change_message("r", {"status": "stopped"}, prev="running")
    assert msg is not None
    assert msg["kind"] == "stopped"
    assert msg["level"] == "warning"
    assert "stopped" in msg["text"].lower()


def test_update_carries_lineage_root_id_when_present():
    # Derived sub-runs poll under their own run_id but the dashboard navigates
    # by lineage root, so the root id must ride along on the update.
    status = {"status": "running", "current_step": 3, "total_steps": 5, "lineage_id": "root-1"}
    advance = build_step_advance_message("derived-9", status)
    assert advance is not None
    assert advance["run_id"] == "derived-9"
    assert advance["lineage_id"] == "root-1"
    errored = build_status_change_message("derived-9", {"status": "error", "lineage_id": "root-1"}, prev="running")
    assert errored is not None
    assert errored["lineage_id"] == "root-1"


def test_update_lineage_id_is_none_for_non_derived_runs():
    # No lineage info in status (plain run) → lineage_id is None; the dashboard
    # falls back to run_id.
    msg = build_step_advance_message("r", {"current_step": 1, "total_steps": 5})
    assert msg is not None
    assert msg["lineage_id"] is None


# --- build_new_best_message ---


def test_build_new_best_message_includes_metric_and_step():
    msg = build_new_best_message("r-abc", {"best_metric": 3.14, "best_step": 4, "current_step": 5, "total_steps": 15})
    assert msg is not None
    assert msg["kind"] == "new_best"
    assert msg["level"] == "success"
    assert "3.14" in msg["text"]
    assert "step 4" in msg["text"]
    assert "5/15" in msg["text"]


def test_build_new_best_message_returns_none_when_no_metric():
    assert build_new_best_message("r", {}) is None
    assert build_new_best_message("r", {"best_metric": 1.5}) is None  # missing best_step


# --- build_step_advance_message ---


def test_build_step_advance_message_includes_progress_and_best():
    msg = build_step_advance_message("r-x", {"current_step": 4, "total_steps": 12, "best_metric": 1.7, "best_step": 2})
    assert msg is not None
    assert msg["kind"] == "step_advance"
    assert "step 4/12" in msg["text"]
    assert "1.7" in msg["text"]


def test_build_step_advance_message_returns_none_without_step():
    assert build_step_advance_message("r", {}) is None


# --- build_pending_review_message ---


def test_build_pending_review_message_mentions_count_and_command():
    msg = build_pending_review_message("r-pr", {"pending_nodes": [{"node_id": "a"}, {"node_id": "b"}]})
    assert msg is not None
    assert msg["kind"] == "pending_review"
    assert "2 nodes" in msg["text"]
    assert any("weco run review r-pr" in h for h in msg["hints"])


def test_build_pending_review_message_returns_none_when_empty():
    assert build_pending_review_message("r", {}) is None
    assert build_pending_review_message("r", {"pending_nodes": []}) is None


# --- build_idle_heartbeat_message ---


def test_build_idle_heartbeat_message_includes_idle_minutes_and_progress():
    msg = build_idle_heartbeat_message(
        "r-idle", {"current_step": 3, "total_steps": 10, "best_metric": 1.5, "best_step": 1}, idle_seconds=125.0
    )
    assert msg is not None
    assert msg["kind"] == "idle"
    assert "step 3/10" in msg["text"]
    assert "best 1.5 at step 1" in msg["text"]
    assert "~2 minutes" in msg["text"]


def test_build_idle_heartbeat_message_uses_singular_minute_for_short_idle():
    msg = build_idle_heartbeat_message("r", {"current_step": 0}, idle_seconds=45.0)
    assert msg is not None
    assert "~1 minute" in msg["text"]
    assert "minutes" not in msg["text"]


# --- RunWatcher integration with mocked subprocess ---


def _collect_via_watcher(statuses: list[dict]) -> list[dict]:
    """Drive the watcher with a canned status sequence; return all updates emitted."""
    iterator = iter(statuses)

    async def fake_fetch(_self, run_id):
        try:
            return next(iterator)
        except StopIteration:
            return None

    received: list[dict] = []

    def notify(update: dict) -> None:
        received.append(update)

    async def body():
        stop_event = asyncio.Event()
        watcher = RunWatcher(weco_bin="/usr/bin/true", notify=notify, stop_event=stop_event, poll_interval_s=0.01)
        with patch.object(RunWatcher, "_fetch_status", fake_fetch):
            assert watcher.watch("r1") is True
            for _ in range(200):
                if not watcher.watching():
                    break
                await asyncio.sleep(0.02)
            await watcher.stop()

    asyncio.run(body())
    return received


def test_watcher_starts_polling_and_announces_completion():
    out = _collect_via_watcher(
        [
            {"status": "running", "current_step": 1, "total_steps": 5},
            {"status": "running", "current_step": 3, "total_steps": 5},
            {"status": "completed", "best_metric": 2.4, "best_step": 4, "current_step": 5, "total_steps": 5},
        ]
    )
    assert any(u["kind"] == "completed" for u in out)


def test_watcher_announces_on_new_best_metric_within_running():
    out = _collect_via_watcher(
        [
            {"status": "running", "current_step": 0, "total_steps": 5, "best_step": 0, "best_metric": 1.0},
            {"status": "running", "current_step": 1, "total_steps": 5, "best_step": 0, "best_metric": 1.0},
            {"status": "running", "current_step": 2, "total_steps": 5, "best_step": 2, "best_metric": 2.5},
            {"status": "completed", "current_step": 5, "total_steps": 5, "best_step": 2, "best_metric": 2.5},
        ]
    )
    assert any(u["kind"] == "new_best" for u in out)
    assert any(u["kind"] == "completed" for u in out)


def test_watcher_announces_step_transitions_without_improvement():
    out = _collect_via_watcher(
        [
            {"status": "running", "current_step": 0, "total_steps": 5, "best_step": 0, "best_metric": 1.0},
            {"status": "running", "current_step": 1, "total_steps": 5, "best_step": 0, "best_metric": 1.0},
            {"status": "running", "current_step": 2, "total_steps": 5, "best_step": 0, "best_metric": 1.0},
            {"status": "running", "current_step": 3, "total_steps": 5, "best_step": 0, "best_metric": 1.0},
            {"status": "completed", "current_step": 5, "total_steps": 5, "best_step": 0, "best_metric": 1.0},
        ]
    )
    assert sum(1 for u in out if u["kind"] == "new_best") == 0
    assert sum(1 for u in out if u["kind"] == "step_advance") >= 1
    assert any(u["kind"] == "completed" for u in out)


def test_watcher_announces_when_pending_review_grows():
    out = _collect_via_watcher(
        [
            {
                "status": "running",
                "current_step": 1,
                "total_steps": 5,
                "best_step": 0,
                "best_metric": 1.0,
                "pending_nodes": [],
            },
            {
                "status": "running",
                "current_step": 1,
                "total_steps": 5,
                "best_step": 0,
                "best_metric": 1.0,
                "pending_nodes": [{"node_id": "a"}, {"node_id": "b"}],
            },
            {
                "status": "completed",
                "current_step": 5,
                "total_steps": 5,
                "best_step": 0,
                "best_metric": 1.0,
                "pending_nodes": [],
            },
        ]
    )
    assert any(u["kind"] == "pending_review" for u in out)


def test_watcher_dedupes_concurrent_watches():
    """Calling watch() twice for the same id is a no-op the second time."""

    async def body():
        stop_event = asyncio.Event()
        watcher = RunWatcher(weco_bin="/usr/bin/true", notify=lambda _u: None, stop_event=stop_event, poll_interval_s=10.0)
        first = watcher.watch("r-dup")
        second = watcher.watch("r-dup")
        await watcher.stop()
        return first, second

    first, second = asyncio.run(body())
    assert first is True
    assert second is False


def test_watcher_no_op_without_weco_bin():
    """If `weco` isn't on PATH, the watcher silently does nothing."""

    async def body() -> bool:
        stop_event = asyncio.Event()
        watcher = RunWatcher(weco_bin=None, notify=lambda _u: None, stop_event=stop_event)
        return watcher.watch("r")

    assert asyncio.run(body()) is False


def test_watcher_stop_idempotent():
    """Calling stop() multiple times is safe."""

    async def body():
        stop_event = asyncio.Event()
        watcher = RunWatcher(weco_bin="/usr/bin/true", notify=lambda _u: None, stop_event=stop_event, poll_interval_s=10.0)
        watcher.watch("r")
        await watcher.stop()
        await watcher.stop()

    asyncio.run(body())
