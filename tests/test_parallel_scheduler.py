"""K-parallel scheduler tests (run_parallel_lineage_loop).

Every backend call is monkeypatched in the ``weco.parallel`` namespace against a
small scripted ``FakeBackend``; evaluations run the *real* ``run_evaluation``
subprocess inside real temp slot directories, so overlap/isolation claims are
observed, not simulated. No network, no real API.

These pin SPEC.md "Scheduler behavior" and "Required tests": never claim more
tasks than free slots, genuine wall-clock overlap in distinct slots, candidate
writes stay slot-local, exact task/run identity on submit, fair deferred-child
generation, at_capacity backpressure without a busy loop, budget/not-running
retirement, stop/409 never reported as completion, account failure aborts
coherently, quiescent-only exit, path-escape refusal, and node-keyed artifacts.
"""

from __future__ import annotations

import json
import pathlib
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import requests

import weco.optimizer as optimizer
import weco.parallel as parallel
from weco.artifacts import RunArtifacts
from weco.slots import Slot, build_slot_env


AUTH = {"Authorization": "Bearer t"}
LINEAGE = "lineage-1"

# Quick, quoted-safe eval commands.
NOOP_CMD = f'{sys.executable} -c "pass"'
SLEEP_CMD = f'{sys.executable} -c "import time; time.sleep(0.5)"'


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


def _make_slot(base: pathlib.Path, index: int) -> Slot:
    """A real, on-disk slot with a project cwd and slot-owned temp/cache dirs."""
    root = base / f"slot-{index}"
    cwd = root / "project"
    cwd.mkdir(parents=True)
    env = build_slot_env(root, index, None)
    for key in ("TMPDIR", "XDG_CACHE_HOME", "TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR"):
        pathlib.Path(env[key]).mkdir(parents=True, exist_ok=True)
    return Slot(index=index, root=root, cwd=cwd, env=env)


def _http_error(status: int, text: str = "err") -> requests.exceptions.HTTPError:
    resp = MagicMock()
    resp.status_code = status
    resp.json.side_effect = ValueError("no json")
    resp.text = text
    return requests.exceptions.HTTPError(response=resp)


class FakeBackend:
    """Scripted, thread-safe stand-in for the lineage queue + generation API."""

    def __init__(
        self,
        *,
        tasks=None,
        members=None,
        generate_handler=None,
        submit_handler=None,
        active_count_fn=None,
        read_returns_none=False,
    ):
        self.lock = threading.Lock()
        # task dict: id, run_id, code(file_map), and mutable _claimed/_done flags.
        self.tasks = tasks or []
        self.members = members or []  # [{"id":.., "status":..}]
        self.generate_handler = generate_handler
        self.submit_handler = submit_handler
        self.active_count_fn = active_count_fn
        self.read_returns_none = read_returns_none

        self.claims: list[tuple[str, float]] = []
        self.submits: list[dict] = []
        self.generate_calls: list[tuple[str, float]] = []
        self.first_submit_time: float | None = None

    # -- installed as weco.parallel.get_lineage_execution_tasks --
    def get_lineage_execution_tasks(self, lineage_id, auth_headers=None):
        if self.read_returns_none:
            return None
        with self.lock:
            ready = [
                {"id": t["id"], "run_id": t["run_id"], "run": {"status": "running"}}
                for t in self.tasks
                if not t["_claimed"] and not t["_done"]
            ]
            active = self.active_count_fn(self) if self.active_count_fn else self._default_active()
            return SimpleNamespace(tasks=ready, active_run_count=active)

    def _default_active(self) -> int:
        return len({t["run_id"] for t in self.tasks if not t["_done"]})

    # -- installed as weco.parallel.claim_execution_task --
    def claim_execution_task(self, task_id, auth_headers=None):
        with self.lock:
            self.claims.append((task_id, time.time()))
            for t in self.tasks:
                if t["id"] == task_id:
                    if t["_claimed"] or t["_done"]:
                        return None
                    t["_claimed"] = True
                    return {
                        "node_id": t.get("node_id", f"node-{task_id}"),
                        "revision": {"plan": t.get("plan"), "code": t.get("code", {"m.py": "x"})},
                    }
            return None

    # -- installed as weco.parallel.submit_execution_result --
    def submit_execution_result(
        self, run_id, task_id, execution_output, auth_headers=None, api_keys=None, timeout=None, skip_generation=False
    ):
        with self.lock:
            record = {
                "run_id": run_id,
                "task_id": task_id,
                "execution_output": execution_output,
                "skip_generation": skip_generation,
                "time": time.time(),
            }
            self.submits.append(record)
            if self.first_submit_time is None:
                self.first_submit_time = record["time"]
            if self.submit_handler is not None:
                return self.submit_handler(self, record)  # may mark done / raise
            for t in self.tasks:
                if t["id"] == task_id:
                    t["_done"] = True
            return {"previous_solution_metric_value": 1.0, "is_done": False}

    # -- WecoClient(auth) adapter --
    def client(self, auth_headers):
        backend = self

        class _Client:
            def generate_candidate(self, run_id, api_keys=None):
                return backend._generate(run_id, api_keys)

            def get_lineage(self, lineage_id):
                with backend.lock:
                    return {"members": list(backend.members)}

        return _Client()

    def _generate(self, run_id, api_keys):
        with self.lock:
            self.generate_calls.append((run_id, time.time()))
        if self.generate_handler is not None:
            return self.generate_handler(self, run_id)  # may raise
        return {"generated": False, "reason": "no_dispatchable_work"}

    def generate_run_ids(self) -> list[str]:
        return [rid for rid, _ in self.generate_calls]


def _install(monkeypatch, backend: FakeBackend):
    """Wire the fake backend + no-op heartbeat + stub run-state builder in."""
    monkeypatch.setattr(parallel, "get_lineage_execution_tasks", backend.get_lineage_execution_tasks)
    monkeypatch.setattr(parallel, "claim_execution_task", backend.claim_execution_task)
    monkeypatch.setattr(parallel, "submit_execution_result", backend.submit_execution_result)
    monkeypatch.setattr(parallel, "WecoClient", backend.client)

    class _DummyHeartbeat:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def join(self, timeout=None):
            pass

    monkeypatch.setattr(optimizer, "LineageHeartbeatSender", _DummyHeartbeat)

    ui_by_run: dict[str, MagicMock] = {}

    def fake_build_run_state(run_id, auth_headers, log_dir, dashboard_base):
        ui = MagicMock()
        ui_by_run[run_id] = ui
        return {"ui": ui, "artifacts": RunArtifacts(log_dir=log_dir, run_id=run_id)}

    monkeypatch.setattr(optimizer, "_build_run_state", fake_build_run_state)
    return ui_by_run


def _run(backend, slots, log_dir, *, lineage_k, eval_command=NOOP_CMD, originals=None, save_logs=False, max_idle_polls=200):
    return parallel.run_parallel_lineage_loop(
        LINEAGE,
        AUTH,
        slots=slots,
        lineage_k=lineage_k,
        originals=originals or {},
        eval_command=eval_command,
        eval_timeout=None,
        save_logs=save_logs,
        log_dir=str(log_dir),
        dashboard_base="https://dash.test",
        poll_interval=0.02,
        max_idle_polls=max_idle_polls,
    )


def _task(task_id, run_id, code=None, node_id=None):
    return {
        "id": task_id,
        "run_id": run_id,
        "code": code or {"m.py": task_id},
        "node_id": node_id or f"node-{task_id}",
        "_claimed": False,
        "_done": False,
    }


def _running(*run_ids):
    """Lineage membership: the given runs are all 'running'."""
    return [{"id": rid, "status": "running"} for rid in run_ids]


# ---------------------------------------------------------------------------
# Slot capacity
# ---------------------------------------------------------------------------


def test_never_claims_more_tasks_than_free_slots(tmp_path, monkeypatch):
    """With 2 slots and 4 ready tasks, at most 2 tasks are claimed before a slot frees."""
    backend = FakeBackend(tasks=[_task(f"t{i}", "A") for i in range(4)], members=_running("A"))
    _install(monkeypatch, backend)
    slots = [_make_slot(tmp_path / "slots", i) for i in range(2)]

    result = _run(backend, slots, tmp_path / "logs", lineage_k=2, eval_command=SLEEP_CMD)

    assert result == "ok"
    # No submit can land before an eval finishes, so claims recorded before the
    # first submit are exactly the initial fill — never more than the slot count.
    assert backend.first_submit_time is not None
    claims_before_first_submit = [t for _, t in backend.claims if t < backend.first_submit_time]
    assert len(claims_before_first_submit) <= 2
    assert len(backend.submits) == 4  # all work eventually drained


# ---------------------------------------------------------------------------
# Genuine overlap + isolation
# ---------------------------------------------------------------------------


def test_two_tasks_overlap_in_distinct_slots(tmp_path, monkeypatch):
    """Two evals run concurrently, each in its own cwd with its own WECO_SLOT."""
    backend = FakeBackend(tasks=[_task("t0", "A"), _task("t1", "A")], members=_running("A"))
    _install(monkeypatch, backend)
    slots = [_make_slot(tmp_path / "slots", i) for i in range(2)]

    script = tmp_path / "overlap.py"
    script.write_text(
        "import os, time, json, pathlib\n"
        "t0 = time.time()\n"
        "time.sleep(0.5)\n"
        "t1 = time.time()\n"
        "pathlib.Path('marker.json').write_text(json.dumps("
        "{'slot': os.environ.get('WECO_SLOT'), 'cwd': os.getcwd(), 't0': t0, 't1': t1}))\n"
    )
    result = _run(backend, slots, tmp_path / "logs", lineage_k=2, eval_command=f'{sys.executable} "{script}"')
    assert result == "ok"

    markers = [json.loads((s.cwd / "marker.json").read_text()) for s in slots]
    assert len(markers) == 2
    a, b = markers
    # Distinct isolation identity.
    assert a["slot"] != b["slot"]
    assert a["cwd"] != b["cwd"]
    # Genuine wall-clock overlap: each started before the other finished.
    assert max(a["t0"], b["t0"]) < min(a["t1"], b["t1"])


def test_candidate_writes_stay_inside_slots(tmp_path, monkeypatch):
    """Candidate files land only in slot cwds; the source project is untouched."""
    source = tmp_path / "source_project"
    source.mkdir()
    (source / "model.py").write_text("ORIGINAL")

    backend = FakeBackend(
        tasks=[
            _task("t0", "A", code={"model.py": "CANDIDATE0", "cand.py": "NEW0"}),
            _task("t1", "A", code={"model.py": "CANDIDATE1", "cand.py": "NEW1"}),
        ],
        members=_running("A"),
    )
    _install(monkeypatch, backend)
    slots = [_make_slot(tmp_path / "slots", i) for i in range(2)]
    for s in slots:
        (s.cwd / "model.py").write_text("ORIGINAL")

    result = _run(backend, slots, tmp_path / "logs", lineage_k=2, originals={"model.py": "ORIGINAL"})
    assert result == "ok"

    for s in slots:
        # Baseline restored after eval; the extra candidate file persists slot-local.
        assert (s.cwd / "model.py").read_text() == "ORIGINAL"
        assert (s.cwd / "cand.py").read_text().startswith("NEW")
    # The user's project never saw a candidate write.
    assert (source / "model.py").read_text() == "ORIGINAL"
    assert not (source / "cand.py").exists()


def test_submit_carries_exact_identity_and_skip_generation(tmp_path, monkeypatch):
    """Each result is submitted with its own task_id/run_id and skip_generation=True."""
    backend = FakeBackend(tasks=[_task("task-xyz", "run-abc")], members=_running("run-abc"))
    _install(monkeypatch, backend)
    slots = [_make_slot(tmp_path / "slots", 0)]

    result = _run(backend, slots, tmp_path / "logs", lineage_k=1)
    assert result == "ok"
    assert len(backend.submits) == 1
    rec = backend.submits[0]
    assert rec["task_id"] == "task-xyz"
    assert rec["run_id"] == "run-abc"
    assert rec["skip_generation"] is True


# ---------------------------------------------------------------------------
# Fairness / generation
# ---------------------------------------------------------------------------


def test_generation_tie_order_rotates_by_cursor():
    """Equal-priority runs do not repeatedly favor the lowest run ID."""
    run_ids = ["C", "A", "B"]
    assert parallel._rotate_run_ids(run_ids, 0) == ["A", "B", "C"]
    assert parallel._rotate_run_ids(run_ids, 1) == ["B", "C", "A"]
    assert parallel._rotate_run_ids(run_ids, 2) == ["C", "A", "B"]


def test_deferred_child_generated_before_busy_run(tmp_path, monkeypatch):
    """A zero-work member (deferred child) gets the next /generate turn first."""
    backend = FakeBackend(
        tasks=[_task("t0", "A")],
        members=[{"id": "A", "status": "running"}, {"id": "B", "status": "running"}],
        generate_handler=lambda be, rid: {"generated": False, "reason": "at_capacity"},
        active_count_fn=lambda be: 1,  # keep the lineage "active"; exit via idle bound
    )
    _install(monkeypatch, backend)
    slots = [_make_slot(tmp_path / "slots", i) for i in range(2)]

    # A keeps a task in flight long enough that B is the only zero-work member.
    result = _run(backend, slots, tmp_path / "logs", lineage_k=2, eval_command=SLEEP_CMD, max_idle_polls=12)
    assert result == "idle_timeout"
    gen_ids = backend.generate_run_ids()
    assert "B" in gen_ids
    # B is prioritized: its first generation precedes A's first generation
    # (A holds in-flight work while B has none).
    assert gen_ids.index("B") < (gen_ids.index("A") if "A" in gen_ids else len(gen_ids))


def test_at_capacity_does_not_busy_loop(tmp_path, monkeypatch):
    """at_capacity is backpressure: generation pauses until a result settles."""
    backend = FakeBackend(
        tasks=[_task("t0", "A")],
        members=[{"id": "A", "status": "running"}, {"id": "B", "status": "running"}],
        generate_handler=lambda be, rid: {"generated": False, "reason": "at_capacity"},
        active_count_fn=lambda be: 1,
    )
    _install(monkeypatch, backend)
    slots = [_make_slot(tmp_path / "slots", i) for i in range(2)]

    result = _run(backend, slots, tmp_path / "logs", lineage_k=2, eval_command=SLEEP_CMD, max_idle_polls=12)
    assert result == "idle_timeout"
    # Without backpressure this would spin thousands of /generate calls; the
    # saturated gate keeps it to a small, settle-driven handful.
    assert len(backend.generate_calls) <= 20


def test_budget_and_not_running_retire_run_from_generation(tmp_path, monkeypatch):
    """out_of_budget / not_running retire a run: it is generated for at most once."""

    def handler(be, run_id):
        return {"generated": False, "reason": "out_of_budget" if run_id == "C" else "not_running"}

    backend = FakeBackend(
        members=[{"id": "C", "status": "running"}, {"id": "D", "status": "running"}],
        generate_handler=handler,
        active_count_fn=lambda be: 1,  # nothing quiesces; exit via idle bound
    )
    _install(monkeypatch, backend)
    slots = [_make_slot(tmp_path / "slots", i) for i in range(2)]

    result = _run(backend, slots, tmp_path / "logs", lineage_k=2, max_idle_polls=5)
    assert result == "idle_timeout"
    gen_ids = backend.generate_run_ids()
    assert gen_ids.count("C") == 1  # retired after the budget refusal
    assert gen_ids.count("D") == 1  # retired after not_running


def test_stale_at_capacity_reply_does_not_silence_generation(tmp_path, monkeypatch):
    """An at_capacity computed before the last eval settled must be discarded.

    Latching saturation on it after that settle would silence generation
    lineage-wide with no eval future left to clear the latch: the scheduler
    would idle out with the lineage still active. The stale reply must instead
    be retried, at which point the backend hands out new work.
    """
    release = threading.Event()

    def generate_handler(be, run_id):
        with be.lock:
            call_no = len(be.generate_calls)
        if call_no == 1:
            # Reply computed "before" the settle but delivered after it: block
            # until the eval result submits, then claim the lineage was full.
            release.wait(timeout=10)
            return {"generated": False, "reason": "at_capacity"}
        if call_no == 2:
            with be.lock:
                be.tasks.append(_task("t1", "A"))
            return {"generated": True}
        return {"generated": False, "reason": "no_dispatchable_work"}

    def submit_handler(be, record):
        for t in be.tasks:
            if t["id"] == record["task_id"]:
                t["_done"] = True
        release.set()
        return {"previous_solution_metric_value": 1.0, "is_done": False}

    backend = FakeBackend(
        tasks=[_task("t0", "A")],
        members=_running("A"),
        generate_handler=generate_handler,
        submit_handler=submit_handler,
        # The run stays 'running' server-side (budget remains) until the
        # retry-generated t1 drains — quiescence must come from draining it,
        # not from the fake conveniently going inactive after t0.
        active_count_fn=lambda be: 0 if any(t["id"] == "t1" and t["_done"] for t in be.tasks) else 1,
    )
    _install(monkeypatch, backend)
    slots = [_make_slot(tmp_path / "slots", i) for i in range(2)]

    result = _run(backend, slots, tmp_path / "logs", lineage_k=2, eval_command=SLEEP_CMD, max_idle_polls=100)

    assert result == "ok"  # quiesced by draining t1, not by idling out
    assert len(backend.submits) == 2  # the post-settle retry produced and drained t1


def test_stale_no_dispatchable_reply_does_not_starve_run(tmp_path, monkeypatch):
    """A no_dispatchable_work computed before the run's last result settled is
    stale: latching awaiting_score on it would exclude the run from generation
    forever — this scheduler owns ALL generation (skip_generation submits), so
    no ready task can ever appear to release the latch."""
    release = threading.Event()

    def generate_handler(be, run_id):
        with be.lock:
            call_no = len(be.generate_calls)
        if call_no == 1:
            release.wait(timeout=10)
            return {"generated": False, "reason": "no_dispatchable_work"}
        if call_no == 2:
            with be.lock:
                be.tasks.append(_task("t1", "A"))
            return {"generated": True}
        return {"generated": False, "reason": "no_dispatchable_work"}

    def submit_handler(be, record):
        for t in be.tasks:
            if t["id"] == record["task_id"]:
                t["_done"] = True
        release.set()
        return {"previous_solution_metric_value": 1.0, "is_done": False}

    backend = FakeBackend(
        tasks=[_task("t0", "A")],
        members=_running("A"),
        generate_handler=generate_handler,
        submit_handler=submit_handler,
        # The run stays 'running' server-side (budget remains) until the
        # retry-generated t1 drains — quiescence must come from draining it,
        # not from the fake conveniently going inactive after t0.
        active_count_fn=lambda be: 0 if any(t["id"] == "t1" and t["_done"] for t in be.tasks) else 1,
    )
    _install(monkeypatch, backend)
    slots = [_make_slot(tmp_path / "slots", i) for i in range(2)]

    result = _run(backend, slots, tmp_path / "logs", lineage_k=2, eval_command=SLEEP_CMD, max_idle_polls=100)

    assert result == "ok"
    assert len(backend.submits) == 2


# ---------------------------------------------------------------------------
# Lifecycle: stop / conflict / account failure / quiescence
# ---------------------------------------------------------------------------


def test_409_submit_is_never_reported_as_completion(tmp_path, monkeypatch):
    """A 409 on submit (stopped/late) must never trigger the completion UI."""

    def submit_handler(be, record):
        # Server finalized the task under us: mark done (so the queue drains)
        # but reply 409 to the client.
        for t in be.tasks:
            if t["id"] == record["task_id"]:
                t["_done"] = True
        raise _http_error(409)

    backend = FakeBackend(tasks=[_task("t0", "A")], members=_running("A"), submit_handler=submit_handler)
    ui_by_run = _install(monkeypatch, backend)
    slots = [_make_slot(tmp_path / "slots", 0)]

    result = _run(backend, slots, tmp_path / "logs", lineage_k=1)
    assert result == "ok"  # conflict is not fatal
    ui_by_run["A"].on_complete.assert_not_called()


def test_account_level_failure_aborts_promptly(tmp_path, monkeypatch):
    """A 402 on submit is an account-level failure: the loop returns False."""

    def submit_handler(be, record):
        for t in be.tasks:
            if t["id"] == record["task_id"]:
                t["_done"] = True
        raise _http_error(402, "insufficient credits")

    backend = FakeBackend(tasks=[_task("t0", "A")], members=_running("A"), submit_handler=submit_handler)
    ui_by_run = _install(monkeypatch, backend)
    slots = [_make_slot(tmp_path / "slots", 0)]

    result = _run(backend, slots, tmp_path / "logs", lineage_k=1)
    assert result == "fatal"
    ui_by_run["A"].on_error.assert_called()


def test_failed_read_keeps_waiting_and_exits_on_bound(tmp_path, monkeypatch):
    """A failed queue read is never mistaken for quiescence; it waits then bounds out."""
    backend = FakeBackend(read_returns_none=True)
    _install(monkeypatch, backend)
    slots = [_make_slot(tmp_path / "slots", 0)]

    result = _run(backend, slots, tmp_path / "logs", lineage_k=1, max_idle_polls=3)
    assert result == "idle_timeout"
    # It never claimed anything on unreadable state.
    assert backend.claims == []


def test_in_flight_evaluation_is_not_counted_as_scheduler_idle(tmp_path, monkeypatch):
    """A legitimate long eval outlives the idle-poll bound and is still reaped.

    The evaluation timeout, not the no-work poll bound, governs active workers.
    """
    backend = FakeBackend(tasks=[_task("t0", "A")], members=_running("A"))
    ui_by_run = _install(monkeypatch, backend)
    slots = [_make_slot(tmp_path / "slots", 0)]

    result = _run(
        backend,
        slots,
        tmp_path / "logs",
        lineage_k=1,
        eval_command=f'{sys.executable} -c "import time; time.sleep(0.2)"',
        max_idle_polls=1,
    )

    assert result == "ok"
    assert len(backend.submits) == 1
    ui_by_run["A"].on_metric.assert_called_once()


# ---------------------------------------------------------------------------
# Path escape + artifacts
# ---------------------------------------------------------------------------


def test_path_escape_candidate_is_refused_not_evaluated(tmp_path, monkeypatch):
    """A traversal candidate is refused, submitted as buggy, and writes nothing outside."""
    backend = FakeBackend(tasks=[_task("t0", "A", code={"../evil.py": "EVIL"})], members=_running("A"))
    _install(monkeypatch, backend)
    slots = [_make_slot(tmp_path / "slots", 0)]

    result = _run(backend, slots, tmp_path / "logs", lineage_k=1, save_logs=True)
    assert result == "ok"
    assert len(backend.submits) == 1
    assert "refused" in backend.submits[0]["execution_output"].lower()
    # Nothing escaped the slot.
    assert not (slots[0].cwd.parent / "evil.py").exists()
    assert not (slots[0].root.parent / "evil.py").exists()


def test_concurrent_artifacts_are_node_keyed_and_uncorrupted(tmp_path, monkeypatch):
    """Simultaneous tasks write distinct node-keyed dirs and a valid JSONL index."""
    backend = FakeBackend(
        tasks=[_task("t0", "A", node_id="node-t0"), _task("t1", "A", node_id="node-t1")], members=_running("A")
    )
    _install(monkeypatch, backend)
    slots = [_make_slot(tmp_path / "slots", i) for i in range(2)]

    result = _run(backend, slots, tmp_path / "logs", lineage_k=2, eval_command=SLEEP_CMD, save_logs=True)
    assert result == "ok"

    run_root = tmp_path / "logs" / "A"
    # Both node code snapshots exist, keyed by node id (never a shared counter).
    assert (run_root / "nodes" / "node-t0").is_dir()
    assert (run_root / "nodes" / "node-t1").is_dir()
    # The shared JSONL index is append-serialized: every line is valid JSON.
    lines = [ln for ln in (run_root / "exec_output.jsonl").read_text().splitlines() if ln.strip()]
    assert len(lines) == 2
    node_ids = {json.loads(ln)["node_id"] for ln in lines}
    assert node_ids == {"node-t0", "node-t1"}
