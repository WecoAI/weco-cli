"""`weco slots` CLI parsing + handler tests (M3b).

Parsing is exercised through ``weco.cli.main`` with the slots handlers stubbed
(recording their kwargs) and the network side-effects — update check + event
send — neutered, mirroring how ``test_parallel_cli.py`` keeps the CLI offline.
The ``handle_verify`` / ``handle_clean`` behavior is then driven directly.
"""

from __future__ import annotations

import subprocess
import sys

import pytest
from rich.console import Console

import weco.cli as cli
import weco.commands.slots as slots_cmd
import weco.slots as slots
from weco.commands.slots import handle_clean, handle_verify
from weco.env import WecoEnv


def _neuter_network(monkeypatch):
    """Silence the update check and telemetry the CLI fires before dispatch."""
    monkeypatch.setattr(WecoEnv, "check_for_updates", lambda self: None)
    monkeypatch.setattr(cli, "send_event", lambda *a, **k: None)


# ---------------------------------------------------------------------------
# Parser wiring
# ---------------------------------------------------------------------------


def test_verify_requires_eval_command(monkeypatch):
    """`weco slots verify` without -c is an argparse error (SystemExit)."""
    _neuter_network(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["weco", "slots", "verify"])
    with pytest.raises(SystemExit):
        cli.main()


def test_verify_defaults_parallel_two_and_timeout_600(monkeypatch):
    """Absent -p / --eval-timeout default to 2 and 600."""
    _neuter_network(monkeypatch)
    captured: dict = {}

    def fake_verify(*, eval_command, parallel, eval_timeout, console):
        captured.update(eval_command=eval_command, parallel=parallel, eval_timeout=eval_timeout)
        return True

    monkeypatch.setattr(slots_cmd, "handle_verify", fake_verify)
    monkeypatch.setattr(sys, "argv", ["weco", "slots", "verify", "-c", "python eval.py"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0
    assert captured == {"eval_command": "python eval.py", "parallel": 2, "eval_timeout": 600}


def test_verify_accepts_overrides(monkeypatch):
    """-p and --eval-timeout override the defaults."""
    _neuter_network(monkeypatch)
    captured: dict = {}

    def fake_verify(*, eval_command, parallel, eval_timeout, console):
        captured.update(parallel=parallel, eval_timeout=eval_timeout)
        return True

    monkeypatch.setattr(slots_cmd, "handle_verify", fake_verify)
    monkeypatch.setattr(sys, "argv", ["weco", "slots", "verify", "-c", "cmd", "-p", "4", "--eval-timeout", "30"])
    with pytest.raises(SystemExit):
        cli.main()
    assert captured == {"parallel": 4, "eval_timeout": 30}


def test_clean_dry_run_parses(monkeypatch):
    """`weco slots clean --dry-run` reaches handle_clean with dry_run=True."""
    _neuter_network(monkeypatch)
    captured: dict = {}

    def fake_clean(*, console, dry_run=False):
        captured["dry_run"] = dry_run
        return True

    monkeypatch.setattr(slots_cmd, "handle_clean", fake_clean)
    monkeypatch.setattr(sys, "argv", ["weco", "slots", "clean", "--dry-run"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0
    assert captured["dry_run"] is True


def test_clean_defaults_to_not_dry_run(monkeypatch):
    """Without --dry-run, dry_run defaults to False."""
    _neuter_network(monkeypatch)
    captured: dict = {}

    def fake_clean(*, console, dry_run=False):
        captured["dry_run"] = dry_run
        return True

    monkeypatch.setattr(slots_cmd, "handle_clean", fake_clean)
    monkeypatch.setattr(sys, "argv", ["weco", "slots", "clean"])
    with pytest.raises(SystemExit):
        cli.main()
    assert captured["dry_run"] is False


# ---------------------------------------------------------------------------
# handle_verify
# ---------------------------------------------------------------------------


def _plain_project(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "m.py").write_text("x\n")
    return project


def test_handle_verify_passes_on_concurrent_success(tmp_path, monkeypatch):
    """Two slots running a brief overlapping command → True."""
    monkeypatch.setattr(slots, "_discover_cuda_pool", lambda: None)
    monkeypatch.chdir(_plain_project(tmp_path))
    cmd = f"{sys.executable} -c \"import time; time.sleep(0.4); print('ok')\""
    assert handle_verify(eval_command=cmd, parallel=2, eval_timeout=30, console=Console()) is True


def test_handle_verify_fails_on_nonzero_exit(tmp_path, monkeypatch):
    """A slot command that exits non-zero fails the verification."""
    monkeypatch.setattr(slots, "_discover_cuda_pool", lambda: None)
    monkeypatch.chdir(_plain_project(tmp_path))
    cmd = f'{sys.executable} -c "import sys; sys.exit(1)"'
    assert handle_verify(eval_command=cmd, parallel=2, eval_timeout=30, console=Console()) is False


def test_handle_verify_rejects_parallel_below_two(tmp_path, monkeypatch):
    """parallel < 2 can't demonstrate concurrency — refused before provisioning."""
    provisioned = False

    def _boom(*args, **kwargs):
        nonlocal provisioned
        provisioned = True
        raise AssertionError("should not provision")

    monkeypatch.setattr(slots_cmd, "create_slot_provider", _boom)
    monkeypatch.chdir(_plain_project(tmp_path))
    assert handle_verify(eval_command="true", parallel=1, eval_timeout=30, console=Console()) is False
    assert provisioned is False


# ---------------------------------------------------------------------------
# handle_clean
# ---------------------------------------------------------------------------


def test_handle_clean_dry_run_lists_without_removing(tmp_path, monkeypatch):
    """A dry run reports stale dirs but leaves them in place."""
    import json

    fake_tmp = tmp_path / "faketmp"
    fake_tmp.mkdir()
    monkeypatch.setattr(slots.tempfile, "gettempdir", lambda: str(fake_tmp))

    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    assert slots._pid_alive(dead.pid) is False

    stale = fake_tmp / "weco-slots-stale"
    stale.mkdir()
    (stale / "weco-slots-meta.json").write_text(
        json.dumps({"schema": "weco-slots-v1", "pid": dead.pid, "provider": "copy", "project": str(tmp_path / "project")})
    )

    assert handle_clean(console=Console(), dry_run=True) is True
    assert stale.exists()  # dry run removed nothing
