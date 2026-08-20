"""Cancellable-evaluation tests (M3b).

Two layers:

* ``run_evaluation`` gains a ``cancel_event`` — when set it kills the whole
  process group promptly and raises :class:`EvaluationCancelled`; the cancellable
  loop still honors ``timeout``; and with no event behavior is unchanged.
* The K-parallel scheduler cancels an in-flight eval when its run leaves the
  lineage (hard stop): the worker returns without submitting and the loop exits
  quickly. Uses the same monkeypatch approach as ``test_parallel_scheduler.py``
  (patch ``weco.parallel.*`` + ``optimizer._build_run_state`` /
  ``LineageHeartbeatSender``); the eval runs the real ``run_evaluation``.

No network. Real subprocesses under ``tmp_path``.
"""

from __future__ import annotations

import os
import pathlib
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import weco.optimizer as optimizer
import weco.parallel as parallel
from weco.artifacts import RunArtifacts
from weco.slots import Slot, build_slot_env
from weco.utils import EvaluationCancelled, run_evaluation


AUTH = {"Authorization": "Bearer t"}
LINEAGE = "lineage-1"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# ---------------------------------------------------------------------------
# run_evaluation cancel_event
# ---------------------------------------------------------------------------


def test_cancel_event_kills_child_and_raises(tmp_path):
    """Setting the event mid-eval raises EvaluationCancelled and reaps the child."""
    pid_file = tmp_path / "pid.txt"
    script = tmp_path / "sleeper.py"
    script.write_text(
        "import os, sys, time, pathlib\npathlib.Path(sys.argv[1]).write_text(str(os.getpid()))\ntime.sleep(60)\n"
    )
    cmd = f'{sys.executable} "{script}" "{pid_file}"'

    cancel_event = threading.Event()
    threading.Timer(0.5, cancel_event.set).start()

    started = time.monotonic()
    try:
        run_evaluation(cmd, cancel_event=cancel_event)
        raise AssertionError("expected EvaluationCancelled")
    except EvaluationCancelled:
        pass
    elapsed = time.monotonic() - started

    assert elapsed < 10  # cancellation lands well within a couple of beats
    # The eval wrote its own pid before sleeping; that process must now be gone.
    child_pid = int(pid_file.read_text())
    deadline = time.monotonic() + 3
    while _pid_alive(child_pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _pid_alive(child_pid)


def test_cancellable_path_still_times_out(tmp_path):
    """With a timeout and an unset event, the cancellable loop still times out."""
    cmd = f'{sys.executable} -c "import time; time.sleep(60)"'
    cancel_event = threading.Event()  # never set

    started = time.monotonic()
    out = run_evaluation(cmd, timeout=1, cancel_event=cancel_event)
    elapsed = time.monotonic() - started

    assert "timed out" in out.lower()
    assert elapsed < 5


def test_no_cancel_event_behaves_normally():
    """Without a cancel_event, a fast command runs to completion unchanged."""
    out = run_evaluation(f"{sys.executable} -c \"print('hello-eval')\"")
    assert "hello-eval" in out


# ---------------------------------------------------------------------------
# Scheduler-level cancellation
# ---------------------------------------------------------------------------


def _make_slot(base: pathlib.Path, index: int) -> Slot:
    root = base / f"slot-{index}"
    cwd = root / "project"
    cwd.mkdir(parents=True)
    env = build_slot_env(root, index, None)
    for key in ("TMPDIR", "XDG_CACHE_HOME", "TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR"):
        pathlib.Path(env[key]).mkdir(parents=True, exist_ok=True)
    return Slot(index=index, root=root, cwd=cwd, env=env)


class _CancelFake:
    """A one-run/one-task lineage that hard-stops the moment the task is claimed:
    membership drops the run and the queue reports zero active runs."""

    def __init__(self):
        self.lock = threading.Lock()
        self.claimed = False
        self.submits: list[dict] = []

    def get_lineage_execution_tasks(self, lineage_id, auth_headers=None):
        with self.lock:
            if self.claimed:
                return SimpleNamespace(tasks=[], active_run_count=0)
            return SimpleNamespace(tasks=[{"id": "t0", "run_id": "A", "run": {"status": "running"}}], active_run_count=1)

    def claim_execution_task(self, task_id, auth_headers=None):
        with self.lock:
            if self.claimed:
                return None
            self.claimed = True
            return {"node_id": "node-t0", "revision": {"plan": None, "code": {"m.py": "x"}}}

    def submit_execution_result(self, run_id, task_id, execution_output, auth_headers=None, **kwargs):
        with self.lock:
            self.submits.append({"run_id": run_id, "task_id": task_id})
        return {"previous_solution_metric_value": 1.0, "is_done": False}

    def client(self, auth_headers):
        fake = self

        class _Client:
            def generate_candidate(self, run_id, api_keys=None):
                return {"generated": False, "reason": "no_dispatchable_work"}

            def get_lineage(self, lineage_id):
                with fake.lock:
                    members = [] if fake.claimed else [{"id": "A", "status": "running"}]
                return {"members": members}

        return _Client()


class _QuiescentFake:
    """An empty lineage: nothing ready, nothing active — immediate clean exit."""

    def get_lineage_execution_tasks(self, lineage_id, auth_headers=None):
        return SimpleNamespace(tasks=[], active_run_count=0)

    def claim_execution_task(self, task_id, auth_headers=None):
        return None

    def submit_execution_result(self, *args, **kwargs):
        return {"previous_solution_metric_value": 1.0, "is_done": False}

    def client(self, auth_headers):
        class _Client:
            def generate_candidate(self, run_id, api_keys=None):
                return {"generated": False, "reason": "no_dispatchable_work"}

            def get_lineage(self, lineage_id):
                return {"members": []}

        return _Client()


class _InterruptFake(_CancelFake):
    """Start one eval, then simulate terminal Ctrl-C on the next queue poll."""

    def get_lineage_execution_tasks(self, lineage_id, auth_headers=None):
        with self.lock:
            if self.claimed:
                raise KeyboardInterrupt
            return SimpleNamespace(tasks=[{"id": "t0", "run_id": "A", "run": {"status": "running"}}], active_run_count=1)

    def client(self, auth_headers):
        class _Client:
            def generate_candidate(self, run_id, api_keys=None):
                return {"generated": False, "reason": "no_dispatchable_work"}

            def get_lineage(self, lineage_id):
                return {"members": [{"id": "A", "status": "running"}]}

        return _Client()


class _GenerateInterruptFake:
    """Interrupt while a deliberately blocked /generate request is in flight."""

    def __init__(self):
        self.polls = 0
        self.generate_started = threading.Event()
        self.release_generate = threading.Event()

    def get_lineage_execution_tasks(self, lineage_id, auth_headers=None):
        self.polls += 1
        if self.polls > 1:
            raise KeyboardInterrupt
        return SimpleNamespace(tasks=[], active_run_count=1)

    def claim_execution_task(self, task_id, auth_headers=None):
        return None

    def submit_execution_result(self, *args, **kwargs):
        raise AssertionError("no evaluation should be submitted")

    def client(self, auth_headers):
        fake = self

        class _Client:
            def generate_candidate(self, run_id, api_keys=None):
                fake.generate_started.set()
                fake.release_generate.wait(timeout=30)
                return {"generated": False, "reason": "no_dispatchable_work"}

            def get_lineage(self, lineage_id):
                return {"members": [{"id": "A", "status": "running"}]}

        return _Client()


class _GenerateStopFake(_GenerateInterruptFake):
    """The lineage hard-stops while a generation request remains blocked."""

    def get_lineage_execution_tasks(self, lineage_id, auth_headers=None):
        self.polls += 1
        return SimpleNamespace(tasks=[], active_run_count=1 if self.polls == 1 else 0)

    def client(self, auth_headers):
        fake = self

        class _Client:
            def generate_candidate(self, run_id, api_keys=None):
                fake.generate_started.set()
                fake.release_generate.wait(timeout=30)
                return {"generated": False, "reason": "not_running"}

            def get_lineage(self, lineage_id):
                return {"members": [{"id": "A", "status": "running"}] if fake.polls == 1 else []}

        return _Client()


def _install(monkeypatch, backend):
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

    def fake_build_run_state(run_id, auth_headers, log_dir, dashboard_base):
        return {"ui": MagicMock(), "artifacts": RunArtifacts(log_dir=log_dir, run_id=run_id)}

    monkeypatch.setattr(optimizer, "_build_run_state", fake_build_run_state)


def test_scheduler_cancels_eval_when_run_leaves_lineage(tmp_path, monkeypatch):
    """When a run's membership drops mid-eval, the scheduler cancels the eval,
    never submits its result, and exits promptly via the cancelled path."""
    backend = _CancelFake()
    _install(monkeypatch, backend)
    slots = [_make_slot(tmp_path / "slots", 0)]

    long_eval = f'{sys.executable} -c "import time; time.sleep(30)"'
    started = time.monotonic()
    result = parallel.run_parallel_lineage_loop(
        LINEAGE,
        AUTH,
        slots=slots,
        lineage_k=1,
        originals={},
        eval_command=long_eval,
        eval_timeout=None,
        save_logs=False,
        log_dir=str(tmp_path / "logs"),
        dashboard_base="https://dash.test",
        poll_interval=0.02,
        max_idle_polls=2000,
    )
    elapsed = time.monotonic() - started

    assert result == "ok"  # cancelled path is a clean exit, not a failure
    assert elapsed < 15  # the 30s eval was killed, not awaited
    assert backend.submits == []  # cancelled work is never submitted


def test_scheduler_eval_timeout_none_defaults_to_3600(tmp_path, monkeypatch, capsys):
    """eval_timeout=None prints the enforced 3600s parallel-slot default."""
    backend = _QuiescentFake()
    _install(monkeypatch, backend)
    slots = [_make_slot(tmp_path / "slots", 0)]

    result = parallel.run_parallel_lineage_loop(
        LINEAGE,
        AUTH,
        slots=slots,
        lineage_k=1,
        originals={},
        eval_command=f'{sys.executable} -c "pass"',
        eval_timeout=None,
        save_logs=False,
        log_dir=str(tmp_path / "logs"),
        dashboard_base="https://dash.test",
        poll_interval=0.02,
        max_idle_polls=5,
    )

    assert result == "ok"
    assert "3600" in capsys.readouterr().out


def test_scheduler_interrupt_cancels_in_flight_eval_before_return(tmp_path, monkeypatch):
    """Ctrl-C signals slot workers and waits for their process trees to stop."""
    backend = _InterruptFake()
    _install(monkeypatch, backend)
    slots = [_make_slot(tmp_path / "slots", 0)]

    started = time.monotonic()
    result = parallel.run_parallel_lineage_loop(
        LINEAGE,
        AUTH,
        slots=slots,
        lineage_k=1,
        originals={},
        eval_command=f'{sys.executable} -c "import time; time.sleep(30)"',
        eval_timeout=60,
        save_logs=False,
        log_dir=str(tmp_path / "logs"),
        dashboard_base="https://dash.test",
        poll_interval=0.02,
        max_idle_polls=2000,
    )

    assert result == "interrupted"
    assert time.monotonic() - started < 10
    assert backend.submits == []


def test_scheduler_interrupt_does_not_wait_for_generation_http_timeout(tmp_path, monkeypatch):
    """A blocked generation request runs on a daemon and cannot pin Ctrl-C."""
    backend = _GenerateInterruptFake()
    _install(monkeypatch, backend)
    slots = [_make_slot(tmp_path / "slots", 0)]

    started = time.monotonic()
    try:
        result = parallel.run_parallel_lineage_loop(
            LINEAGE,
            AUTH,
            slots=slots,
            lineage_k=1,
            originals={},
            eval_command=f'{sys.executable} -c "pass"',
            eval_timeout=60,
            save_logs=False,
            log_dir=str(tmp_path / "logs"),
            dashboard_base="https://dash.test",
            poll_interval=0.02,
            max_idle_polls=2000,
        )
        assert backend.generate_started.is_set()
        assert result == "interrupted"
        assert time.monotonic() - started < 5
    finally:
        backend.release_generate.set()


def test_scheduler_hard_stop_does_not_wait_for_generation_http_timeout(tmp_path, monkeypatch):
    """Authoritative zero active members detaches a blocked daemon generation."""
    backend = _GenerateStopFake()
    _install(monkeypatch, backend)
    slots = [_make_slot(tmp_path / "slots", 0)]

    started = time.monotonic()
    try:
        result = parallel.run_parallel_lineage_loop(
            LINEAGE,
            AUTH,
            slots=slots,
            lineage_k=1,
            originals={},
            eval_command=f'{sys.executable} -c "pass"',
            eval_timeout=60,
            save_logs=False,
            log_dir=str(tmp_path / "logs"),
            dashboard_base="https://dash.test",
            poll_interval=0.02,
            max_idle_polls=2000,
        )
        assert backend.generate_started.is_set()
        assert result == "ok"
        assert time.monotonic() - started < 5
    finally:
        backend.release_generate.set()
