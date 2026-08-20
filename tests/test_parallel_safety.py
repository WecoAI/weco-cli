"""Safety regressions for isolated evaluation slots and process handling."""

import json
import os
import threading
import time

import pytest

from weco.slots import CopySlotProvider, SlotPathError, SlotProvisionError, prepare_write_target, resolve_candidate_path
from weco.utils import _kill_process_tree, run_evaluation


# ---------------------------------------------------------------------------
# Symlink escape prevention (resolve time and write time)
# ---------------------------------------------------------------------------


def test_resolve_candidate_path_rejects_leaf_symlink(tmp_path):
    """A candidate path whose FILE is a symlink is refused — writing through it
    would land outside the slot even though every ancestor is inside."""
    outside = tmp_path / "outside.txt"
    outside.write_text("victim")
    base = tmp_path / "slot"
    base.mkdir()
    (base / "config.py").symlink_to(outside)

    with pytest.raises(SlotPathError, match="symlink"):
        resolve_candidate_path(base, "config.py")


def test_prepare_write_target_unlinks_leaf_symlink(tmp_path):
    """The post-eval restore recreates a regular file instead of following a
    symlink the eval planted at a restore path."""
    outside = tmp_path / "outside.txt"
    outside.write_text("victim")
    base = tmp_path / "slot"
    base.mkdir()
    (base / "main.py").symlink_to(outside)

    target = prepare_write_target(base, "main.py")
    target.write_text("restored")

    assert outside.read_text() == "victim"
    assert not (base / "main.py").is_symlink()
    assert (base / "main.py").read_text() == "restored"


def test_prepare_write_target_rejects_symlinked_ancestor(tmp_path):
    """An eval that swaps a directory for an outside-pointing symlink cannot
    receive restore writes through it."""
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    base = tmp_path / "slot"
    (base / "pkg").mkdir(parents=True)
    (base / "pkg").rmdir()
    (base / "pkg").symlink_to(outside_dir)

    with pytest.raises(SlotPathError, match="ancestor"):
        prepare_write_target(base, "pkg/mod.py")


# ---------------------------------------------------------------------------
# Evaluation process groups and Ctrl-C reachability
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not hasattr(os, "getpgid"), reason="POSIX process groups required")
def test_serial_eval_shares_process_group(tmp_path):
    """Without a cancel_event (the K=1 in-place path) the eval child must stay
    in the CLI's process group so a terminal SIGINT reaches it — the
    pre-parallelization behavior."""
    pgid_file = tmp_path / "pgid"
    run_evaluation(f"python3 -c \"import os; open({str(pgid_file)!r}, 'w').write(str(os.getpgrp()))\"")
    assert int(pgid_file.read_text()) == os.getpgrp()


@pytest.mark.skipif(not hasattr(os, "getpgid"), reason="POSIX process groups required")
def test_cancellable_eval_owns_its_process_group(tmp_path):
    """With a cancel_event (slot evals) the child leads its own group so the
    whole tree can be killed without signalling the CLI itself."""
    pgid_file = tmp_path / "pgid"
    run_evaluation(
        f"python3 -c \"import os; open({str(pgid_file)!r}, 'w').write(str(os.getpgrp()) + ',' + str(os.getpid()))\"",
        cancel_event=threading.Event(),
    )
    for _ in range(20):  # the writer may still be flushing when communicate returns
        if pgid_file.exists() and "," in pgid_file.read_text():
            break
        time.sleep(0.05)
    pgid, _pid = (int(v) for v in pgid_file.read_text().split(","))
    # The eval must run in a NEW process group so a group kill can never
    # signal the CLI itself. Whether the group leader is the wrapping shell
    # or the python child depends on the platform shell (bash execs a simple
    # command, dash forks before it) — either satisfies the safety property,
    # so assert only that the group is not ours.
    assert pgid != os.getpgrp()


@pytest.mark.skipif(not hasattr(os, "getpgid"), reason="POSIX process groups required")
def test_kill_process_tree_never_signals_a_shared_group(monkeypatch):
    """Killing a serial (shared-group) eval must never use a group signal:
    the child shares the CLI's process group, so ``killpg`` would take the CLI
    down with it. The child must still die via the individual psutil walk."""
    import subprocess
    import sys

    group_signals: list[tuple[int, int]] = []
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: group_signals.append((pgid, sig)))

    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert os.getpgid(proc.pid) == os.getpgrp()  # genuinely shares our group
        _kill_process_tree(proc)
        assert proc.poll() is not None  # terminated individually
        assert group_signals == []  # and never via the group
    finally:
        if proc.poll() is None:
            proc.kill()


# ---------------------------------------------------------------------------
# Slot-registry ownership after a daemon fork
# ---------------------------------------------------------------------------


def test_refresh_registry_meta_restamps_current_pid(tmp_path, monkeypatch):
    """A daemon child (new pid) re-stamps the slot registry so `weco slots
    clean` never classifies its live slots as stale."""
    fake_tmp = tmp_path / "tmp"
    fake_tmp.mkdir()
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(fake_tmp))
    monkeypatch.setattr("tempfile.tempdir", str(fake_tmp))

    project = tmp_path / "proj"
    project.mkdir()
    (project / "main.py").write_text("print('x')\n")

    provider = CopySlotProvider(project, 1)
    try:
        provider.provision()
        meta_path = provider.base_dir / "weco-slots-meta.json"
        # Simulate the stale parent pid a daemon fork leaves behind.
        stale = json.loads(meta_path.read_text())
        stale["pid"] = 1  # pid 1 is init/launchd — "alive", but not ours; any wrong pid works
        meta_path.write_text(json.dumps(stale))

        provider.refresh_registry_meta()

        assert json.loads(meta_path.read_text())["pid"] == os.getpid()
    finally:
        provider.cleanup()


def test_provision_refuses_tmpdir_inside_project(tmp_path, monkeypatch):
    """Slots must never nest inside the source project (recursive copies)."""
    project = tmp_path / "proj"
    (project / "tmp").mkdir(parents=True)
    (project / "main.py").write_text("print('x')\n")
    monkeypatch.setenv("TMPDIR", str(project / "tmp"))
    monkeypatch.setattr("tempfile.tempdir", None)  # force re-read of TMPDIR

    provider = CopySlotProvider(project, 2)
    with pytest.raises(SlotProvisionError, match="inside the project"):
        provider.provision()
    monkeypatch.setattr("tempfile.tempdir", None)  # don't leak the override
