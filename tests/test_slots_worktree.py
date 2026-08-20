"""Worktree-provider + overlay-config + registry tests (M3b).

Pins the M3b machinery layered onto the M3a copy provider: ``.weco/parallel.toml``
overlay parsing, git-project detection, the ``git worktree``-based provider that
overlays the working tree's uncommitted state (modified/untracked copied, deleted
removed) plus declared/`.env`/symlink overlays, worktree teardown, the
``create_slot_provider`` factory choices, and stale-slot registry discovery/clean.

Real git subprocesses are used (git is available); every temp-dir scan is
redirected into ``tmp_path`` so the real system temp is never touched.
"""

from __future__ import annotations

import os
import subprocess

import pytest

import weco.slots as slots
from weco.slots import (
    CopySlotProvider,
    SlotProvisionError,
    WorktreeSlotProvider,
    clean_stale_slots,
    create_slot_provider,
    find_stale_slot_dirs,
    is_git_project,
    load_overlay_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True)


def _init_repo(repo):
    """A git repo with an initial commit of a.py, sub/b.py, c.py."""
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init")
    # Deterministic, no dependence on global git identity.
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "a.py").write_text("A_ORIGINAL\n")
    (repo / "sub").mkdir()
    (repo / "sub" / "b.py").write_text("B\n")
    (repo / "c.py").write_text("C\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "init")


def _worktree_paths(repo):
    out = subprocess.run(
        ["git", "-C", str(repo), "worktree", "list", "--porcelain"], check=True, capture_output=True, text=True
    ).stdout
    return [line[len("worktree ") :] for line in out.splitlines() if line.startswith("worktree ")]


# ---------------------------------------------------------------------------
# load_overlay_config
# ---------------------------------------------------------------------------


def test_load_overlay_config_absent_returns_defaults(tmp_path):
    """No .weco/parallel.toml → empty defaults (auto provider, no extras)."""
    cfg = load_overlay_config(tmp_path)
    assert cfg.provider is None
    assert cfg.copy == []
    assert cfg.symlink == []


def test_load_overlay_config_parses_valid_toml(tmp_path):
    """A well-formed [slots] table yields provider/copy/symlink."""
    (tmp_path / ".weco").mkdir()
    (tmp_path / ".weco" / "parallel.toml").write_text(
        '[slots]\nprovider = "worktree"\ncopy = ["data/small", ".env"]\nsymlink = ["data/huge"]\n'
    )
    cfg = load_overlay_config(tmp_path)
    assert cfg.provider == "worktree"
    assert cfg.copy == ["data/small", ".env"]
    assert cfg.symlink == ["data/huge"]


def test_load_overlay_config_malformed_toml_returns_defaults(tmp_path):
    """Unparseable TOML degrades to defaults rather than raising."""
    (tmp_path / ".weco").mkdir()
    (tmp_path / ".weco" / "parallel.toml").write_text("this is = = not valid toml [[[\n")
    cfg = load_overlay_config(tmp_path)
    assert cfg.provider is None
    assert cfg.copy == []
    assert cfg.symlink == []


def test_load_overlay_config_bogus_provider_is_dropped(tmp_path):
    """An unrecognized provider value falls back to None (auto)."""
    (tmp_path / ".weco").mkdir()
    (tmp_path / ".weco" / "parallel.toml").write_text('[slots]\nprovider = "banana"\n')
    cfg = load_overlay_config(tmp_path)
    assert cfg.provider is None


@pytest.mark.parametrize(
    ("body", "expected_copy", "expected_symlink"),
    [
        ('[slots]\ncopy = "not-a-list"\nsymlink = 3\n', [], []),
        ('[slots]\ncopy = ["../outside", "/tmp/absolute", "C:/drive", "safe/data"]\n', ["safe/data"], []),
    ],
)
def test_load_overlay_config_rejects_bad_types_and_escaping_paths(tmp_path, body, expected_copy, expected_symlink):
    """.weco overlays cannot escape either the project or slot target."""
    (tmp_path / ".weco").mkdir()
    (tmp_path / ".weco" / "parallel.toml").write_text(body)
    cfg = load_overlay_config(tmp_path)
    assert cfg.copy == expected_copy
    assert cfg.symlink == expected_symlink


# ---------------------------------------------------------------------------
# is_git_project
# ---------------------------------------------------------------------------


def test_is_git_project_false_for_plain_dir(tmp_path):
    """A directory that is not a git work tree is not a git project."""
    assert is_git_project(tmp_path) is False


def test_is_git_project_false_without_commit(tmp_path):
    """An initialized repo with no HEAD commit is not yet usable."""
    _git(tmp_path, "init")
    assert is_git_project(tmp_path) is False


def test_is_git_project_true_after_commit(tmp_path):
    """A repo with at least one commit is a git project."""
    _init_repo(tmp_path)
    assert is_git_project(tmp_path) is True


# ---------------------------------------------------------------------------
# WorktreeSlotProvider.provision
# ---------------------------------------------------------------------------


def _dirty_repo(tmp_path, monkeypatch):
    """Init a repo, then apply the full spread of staged/unstaged working state."""
    monkeypatch.setattr(slots, "_discover_cuda_pool", lambda: None)
    repo = tmp_path / "repo"
    _init_repo(repo)
    # Staged-only modified tracked file (working tree matches the index).
    (repo / "a.py").write_text("A_MODIFIED\n")
    _git(repo, "add", "a.py")
    # Untracked (non-ignored) file.
    (repo / "notes.txt").write_text("scratch notes\n")
    # Ignore .env and data/, then create both (gitignored, untracked).
    (repo / ".gitignore").write_text(".env\ndata/\n")
    (repo / ".env").write_text("SECRET=1\n")
    (repo / "data").mkdir()
    (repo / "data" / "big.bin").write_text("payload\n")
    # Staged deletion.
    (repo / "c.py").unlink()
    _git(repo, "add", "-u", "c.py")
    return repo


def test_worktree_provision_overlays_working_state(tmp_path, monkeypatch):
    """Slots reflect staged and unstaged state: modified/untracked in, deleted
    out, .env copied, ignored-undeclared dir absent."""
    repo = _dirty_repo(tmp_path, monkeypatch)
    provider = WorktreeSlotProvider(repo, 2)
    slots_made = provider.provision()
    try:
        assert len(slots_made) == 2
        for slot in slots_made:
            cwd = slot.cwd
            # Staged-only modified content, not HEAD content.
            assert (cwd / "a.py").read_text() == "A_MODIFIED\n"
            # Untracked non-ignored file copied in.
            assert (cwd / "notes.txt").read_text() == "scratch notes\n"
            # .env is gitignored yet copied per-slot so the eval can run.
            assert (cwd / ".env").read_text() == "SECRET=1\n"
            # Committed-then-deleted file removed from the slot.
            assert not (cwd / "c.py").exists()
            # A tracked, untouched file is present from HEAD.
            assert (cwd / "sub" / "b.py").read_text() == "B\n"
            # Ignored + undeclared directory does not appear.
            assert not (cwd / "data").exists()
    finally:
        provider.cleanup()


def test_worktree_provision_symlinks_declared_overlay(tmp_path, monkeypatch):
    """A declared symlink overlay materializes the ignored dir as a symlink."""
    repo = _dirty_repo(tmp_path, monkeypatch)
    (repo / ".weco").mkdir()
    (repo / ".weco" / "parallel.toml").write_text('[slots]\nsymlink = ["data"]\n')
    provider = create_slot_provider(repo, 2)
    assert isinstance(provider, WorktreeSlotProvider)
    slots_made = provider.provision()
    try:
        assert slots_made
        for slot in slots_made:
            linked = slot.cwd / "data"
            assert linked.is_symlink()
            assert linked.resolve() == (repo / "data").resolve()
            assert (linked / "big.bin").read_text() == "payload\n"
    finally:
        provider.cleanup()


def test_worktree_cleanup_removes_worktrees_and_base(tmp_path, monkeypatch):
    """Cleanup prunes every slot worktree and removes the base dir."""
    repo = _dirty_repo(tmp_path, monkeypatch)
    provider = WorktreeSlotProvider(repo, 2)
    slots_made = provider.provision()
    base = provider.base_dir
    assert base is not None and base.exists()
    # During the run the slot worktrees are registered.
    assert len(_worktree_paths(repo)) == 1 + len(slots_made)

    provider.cleanup()

    # Only the main work tree remains; the base dir is gone.
    remaining = _worktree_paths(repo)
    assert len(remaining) == 1
    assert os.path.realpath(remaining[0]) == os.path.realpath(str(repo))
    assert not base.exists()


# ---------------------------------------------------------------------------
# create_slot_provider factory
# ---------------------------------------------------------------------------


def test_factory_picks_worktree_for_git_repo(tmp_path, monkeypatch):
    """A git project with a commit selects the worktree provider by default."""
    monkeypatch.setattr(slots, "_discover_cuda_pool", lambda: None)
    repo = tmp_path / "repo"
    _init_repo(repo)
    provider = create_slot_provider(repo, 2)
    assert isinstance(provider, WorktreeSlotProvider)


def test_factory_picks_copy_for_non_git(tmp_path):
    """A plain directory falls back to the copy provider."""
    project = tmp_path / "plain"
    project.mkdir()
    (project / "m.py").write_text("x\n")
    provider = create_slot_provider(project, 2)
    assert isinstance(provider, CopySlotProvider)
    assert not isinstance(provider, WorktreeSlotProvider)


def test_factory_config_forces_copy_in_git_repo(tmp_path):
    """provider = "copy" overrides the git-default worktree choice."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / ".weco").mkdir()
    (repo / ".weco" / "parallel.toml").write_text('[slots]\nprovider = "copy"\n')
    provider = create_slot_provider(repo, 2)
    assert isinstance(provider, CopySlotProvider)
    assert not isinstance(provider, WorktreeSlotProvider)


def test_factory_raises_on_windows(tmp_path, monkeypatch):
    """Windows has no working-tree consumer lock, so K>1 is refused."""
    repo = tmp_path / "repo"
    _init_repo(repo)

    # Override only ``os.name`` as slots sees it — patching the real os.name
    # would break pathlib's platform-flavour selection on this host.
    class _NtOS:
        name = "nt"

        def __getattr__(self, item):
            return getattr(os, item)

    monkeypatch.setattr(slots, "os", _NtOS())
    with pytest.raises(SlotProvisionError):
        create_slot_provider(repo, 2)


# ---------------------------------------------------------------------------
# Registry / stale-slot discovery + clean
# ---------------------------------------------------------------------------


def _fake_stale_dir(fake_tmp, name, meta):
    import json

    base = fake_tmp / name
    base.mkdir()
    (base / "weco-slots-meta.json").write_text(json.dumps({"schema": "weco-slots-v1", **meta}))
    return base


def test_live_slot_dir_is_not_stale_and_survives_clean(tmp_path, monkeypatch):
    """A provisioned dir (our live pid) is skipped by discovery and never removed;
    a sibling with a dead pid is found and removed."""
    fake_tmp = tmp_path / "faketmp"
    fake_tmp.mkdir()
    monkeypatch.setattr(slots.tempfile, "gettempdir", lambda: str(fake_tmp))
    monkeypatch.setattr(slots, "_discover_cuda_pool", lambda: None)

    project = tmp_path / "project"
    project.mkdir()
    (project / "m.py").write_text("x\n")

    provider = CopySlotProvider(project, 1)
    provider.provision()
    try:
        live_base = provider.base_dir
        assert live_base is not None
        # Registry marker written with our own (live) pid.
        meta_text = (live_base / "weco-slots-meta.json").read_text()
        assert f'"pid": {os.getpid()}' in meta_text

        # A dead pid: spawn a child, wait for it to exit, reuse its (now-freed) pid.
        dead = subprocess.Popen([__import__("sys").executable, "-c", "pass"])
        dead.wait()
        dead_pid = dead.pid
        assert slots._pid_alive(dead_pid) is False
        stale_base = _fake_stale_dir(
            fake_tmp, "weco-slots-deadxyz", {"pid": dead_pid, "provider": "copy", "project": str(project)}
        )

        # Discovery skips the live dir, includes the dead-pid one.
        found = {b for b, _ in find_stale_slot_dirs()}
        assert live_base not in found
        assert stale_base in found

        # Clean removes only the stale dir; the live dir is untouched.
        removed = clean_stale_slots()
        assert stale_base in removed
        assert live_base not in removed
        assert not stale_base.exists()
        assert live_base.exists()
    finally:
        provider.cleanup()


def test_missing_or_corrupt_registry_marker_is_never_cleanup_authority(tmp_path, monkeypatch):
    """A name prefix without an authenticated Weco marker is left untouched."""
    fake_tmp = tmp_path / "faketmp"
    fake_tmp.mkdir()
    monkeypatch.setattr(slots.tempfile, "gettempdir", lambda: str(fake_tmp))
    missing = fake_tmp / "weco-slots-unrelated"
    missing.mkdir()
    corrupt = fake_tmp / "weco-slots-corrupt"
    corrupt.mkdir()
    (corrupt / "weco-slots-meta.json").write_text("{bad")

    assert find_stale_slot_dirs() == []
    assert clean_stale_slots() == []
    assert missing.exists()
    assert corrupt.exists()


@pytest.mark.parametrize(
    "meta",
    [
        {"provider": "copy", "project": "/tmp/project"},
        {"provider": "copy", "project": "/tmp/project", "pid": "123"},
        {"provider": "copy", "project": "/tmp/project", "pid": True},
        {"provider": "copy", "project": "/tmp/project", "pid": 0},
        {"provider": "copy", "pid": 123},
        {"provider": "copy", "project": 123, "pid": 123},
    ],
)
def test_incomplete_registry_marker_is_never_cleanup_authority(tmp_path, monkeypatch, meta):
    """Only a complete, typed marker from Weco can authorize stale cleanup."""
    fake_tmp = tmp_path / "faketmp"
    fake_tmp.mkdir()
    monkeypatch.setattr(slots.tempfile, "gettempdir", lambda: str(fake_tmp))
    base = _fake_stale_dir(fake_tmp, "weco-slots-incomplete", meta)

    assert find_stale_slot_dirs() == []
    assert clean_stale_slots() == []
    assert base.exists()


def test_worktree_provision_from_repo_subdirectory(tmp_path, monkeypatch):
    """weco run invoked from a SUBDIRECTORY of a git repo: the slot's eval cwd
    must be that subdirectory's counterpart inside the worktree (with the
    repo context around it), the uncommitted overlay must land at
    repo-relative paths, and cleanup must remove the worktree root."""
    repo = _dirty_repo(tmp_path, monkeypatch)
    sub = repo / "sub"
    (sub / "untracked_note.txt").write_text("local\n")

    provider = create_slot_provider(sub, 2)
    assert isinstance(provider, WorktreeSlotProvider)
    slots_made = provider.provision()
    try:
        assert len(slots_made) == 2
        for slot in slots_made:
            # Evals run where the user invoked weco, not the repo root.
            assert (slot.cwd / "b.py").read_text() == "B\n"
            # Uncommitted state belonging to the subdir arrives IN the subdir.
            assert (slot.cwd / "untracked_note.txt").read_text() == "local\n"
            # The enclosing repo is present around it, with working-tree state.
            assert (slot.cwd.parent / "a.py").read_text() == "A_MODIFIED\n"
            assert not (slot.cwd.parent / "c.py").exists()
    finally:
        provider.cleanup()
    # Worktrees removed even though cwd was the subdir, not the worktree root.
    assert _worktree_paths(repo) == [str(repo)]
