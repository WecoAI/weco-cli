"""Slot-provider safety tests (M3a copy-based isolation).

These pin the invariants from SPEC.md "Copy-slot provider" and "Evaluation and
slot environment": candidate paths cannot escape a slot, the per-slot
environment is distinct and slot-owned, project copies exclude control/build
artifacts, and cleanup is bounded and never touches the source project.
"""

from __future__ import annotations

import os
import pathlib

import pytest

import weco.slots as slots
from weco.slots import CopySlotProvider, Slot, SlotPathError, build_slot_env, resolve_candidate_path


# ---------------------------------------------------------------------------
# resolve_candidate_path
# ---------------------------------------------------------------------------


def test_resolve_candidate_path_accepts_nested_relative(tmp_path):
    """A normal nested relative path resolves to a location inside the slot."""
    target = resolve_candidate_path(tmp_path, "src/pkg/mod.py")
    assert target == tmp_path / "src" / "pkg" / "mod.py"
    assert tmp_path.resolve() in target.resolve().parents


def test_resolve_candidate_path_rejects_absolute(tmp_path):
    """An absolute candidate path is refused (would write outside the slot)."""
    with pytest.raises(SlotPathError):
        resolve_candidate_path(tmp_path, "/etc/passwd")


def test_resolve_candidate_path_rejects_parent_traversal(tmp_path):
    """A `..` component is refused before it can climb out of the slot."""
    with pytest.raises(SlotPathError):
        resolve_candidate_path(tmp_path, "../evil.py")


def test_resolve_candidate_path_rejects_drive_prefix(tmp_path):
    """A Windows drive-prefixed path is refused (POSIX-relative but escaping)."""
    with pytest.raises(SlotPathError):
        resolve_candidate_path(tmp_path, "C:/x")


def test_resolve_candidate_path_rejects_symlink_escape(tmp_path):
    """A path routed through an in-slot symlink that points outside is refused."""
    base = tmp_path / "slot"
    base.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (base / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(SlotPathError):
        resolve_candidate_path(base, "link/evil.py")


# ---------------------------------------------------------------------------
# build_slot_env
# ---------------------------------------------------------------------------


def test_build_slot_env_dirs_are_slot_owned_and_distinct(tmp_path):
    """Temp/cache dirs live under each slot's own root and differ per index."""
    root0 = tmp_path / "slot-0"
    root1 = tmp_path / "slot-1"
    env0 = build_slot_env(root0, 0, None)
    env1 = build_slot_env(root1, 1, None)

    assert env0["WECO_SLOT"] == "0"
    assert env1["WECO_SLOT"] == "1"
    for key in ("TMPDIR", "XDG_CACHE_HOME", "TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR"):
        assert pathlib.Path(env0[key]).is_relative_to(root0), key
        assert pathlib.Path(env1[key]).is_relative_to(root1), key
        assert env0[key] != env1[key], key


def test_build_slot_env_port_offset_is_index_times_ten(tmp_path):
    """WECO_PORT_OFFSET is a deterministic index*10 hook for eval scripts."""
    assert build_slot_env(tmp_path, 0, None)["WECO_PORT_OFFSET"] == "0"
    assert build_slot_env(tmp_path, 3, None)["WECO_PORT_OFFSET"] == "30"


def test_build_slot_env_no_cuda_key_when_pool_none(tmp_path):
    """When the GPU pool is unknowable, CUDA_VISIBLE_DEVICES is left untouched."""
    env = build_slot_env(tmp_path, 0, None)
    assert "CUDA_VISIBLE_DEVICES" not in env


def test_build_slot_env_cuda_round_robins_within_pool(tmp_path):
    """Slots round-robin over the discovered GPU pool; never invent a device."""
    pool = ["0", "1"]
    assert build_slot_env(tmp_path, 0, pool)["CUDA_VISIBLE_DEVICES"] == "0"
    assert build_slot_env(tmp_path, 1, pool)["CUDA_VISIBLE_DEVICES"] == "1"
    assert build_slot_env(tmp_path, 2, pool)["CUDA_VISIBLE_DEVICES"] == "0"


def test_build_slot_env_preserves_empty_user_restriction(tmp_path):
    """An empty user CUDA restriction (pool == [""]) stays empty for every slot."""
    for index in range(3):
        assert build_slot_env(tmp_path, index, [""])["CUDA_VISIBLE_DEVICES"] == ""


# ---------------------------------------------------------------------------
# Slot.full_env
# ---------------------------------------------------------------------------


def test_full_env_overlays_process_environment(tmp_path, monkeypatch):
    """full_env() overlays the slot env on os.environ, never discarding it."""
    monkeypatch.setenv("WECO_TEST_SENTINEL", "keep-me")
    slot = Slot(index=2, root=tmp_path, cwd=tmp_path, env={"WECO_SLOT": "2"})
    merged = slot.full_env()
    assert merged["WECO_TEST_SENTINEL"] == "keep-me"  # inherited auth/PATH/etc survive
    assert merged["WECO_SLOT"] == "2"  # slot overlay applied


# ---------------------------------------------------------------------------
# CopySlotProvider.provision
# ---------------------------------------------------------------------------


def _make_project(root: pathlib.Path) -> pathlib.Path:
    """A fake project tree with the artifacts a provider must exclude/symlink."""
    project = root / "project"
    project.mkdir()
    (project / "model.py").write_text("print('model')\n")
    (project / "README.md").write_text("readme\n")
    (project / "data").mkdir()
    (project / "data" / "nested.txt").write_text("nested\n")
    for control in (".git", ".weco", ".runs", "__pycache__"):
        (project / control).mkdir()
        (project / control / "junk").write_text("junk\n")
    venv = project / ".venv"
    venv.mkdir()
    (venv / "pyvenv.cfg").write_text("home = /usr\n")
    return project


def test_provision_creates_k_isolated_slots(tmp_path, monkeypatch):
    """K>1 provisions K copies with distinct roots/cwds all outside the project."""
    monkeypatch.setattr(slots, "_discover_cuda_pool", lambda: None)
    project = _make_project(tmp_path)
    provider = CopySlotProvider(project, 3)
    made = provider.provision()
    try:
        assert len(made) == 3
        project_resolved = project.resolve()
        roots = {s.root for s in made}
        cwds = {s.cwd for s in made}
        assert len(roots) == 3 and len(cwds) == 3  # distinct per slot
        for slot in made:
            cwd_resolved = slot.cwd.resolve()
            assert project_resolved not in cwd_resolved.parents  # outside source
            # Project files copied in.
            assert (slot.cwd / "model.py").read_text() == "print('model')\n"
            assert (slot.cwd / "data" / "nested.txt").read_text() == "nested\n"
            # Control/build artifacts excluded.
            for control in (".git", ".weco", ".runs", "__pycache__"):
                assert not (slot.cwd / control).exists(), control
            # Heavy env dir shared as a symlink to the original, not copied.
            linked = slot.cwd / ".venv"
            assert linked.is_symlink()
            assert linked.resolve() == (project / ".venv").resolve()
    finally:
        provider.cleanup()


def test_provision_excludes_custom_log_dir(tmp_path, monkeypatch):
    """A non-default log_dir top-level name is excluded from the copy."""
    monkeypatch.setattr(slots, "_discover_cuda_pool", lambda: None)
    project = _make_project(tmp_path)
    (project / "mylogs").mkdir()
    (project / "mylogs" / "run.json").write_text("{}\n")
    provider = CopySlotProvider(project, 2, log_dir="mylogs")
    made = provider.provision()
    try:
        assert made
        for slot in made:
            assert not (slot.cwd / "mylogs").exists()
    finally:
        provider.cleanup()


def test_cleanup_removes_base_dir_and_is_idempotent(tmp_path, monkeypatch):
    """Cleanup removes the whole slot base; a second cleanup is a safe no-op."""
    monkeypatch.setattr(slots, "_discover_cuda_pool", lambda: None)
    project = _make_project(tmp_path)
    provider = CopySlotProvider(project, 2)
    provider.provision()
    base = provider.base_dir
    assert base is not None and base.exists()

    provider.cleanup()
    assert not base.exists()
    assert provider.base_dir is None
    provider.cleanup()  # must not raise
    # The source project is never touched by cleanup.
    assert (project / "model.py").read_text() == "print('model')\n"
    assert (project / "data" / "nested.txt").exists()


def test_provision_failure_midway_leaves_no_partial_slot(tmp_path, monkeypatch):
    """A copytree failure on slot 2 yields exactly one slot and no orphan dir."""
    monkeypatch.setattr(slots, "_discover_cuda_pool", lambda: None)
    project = _make_project(tmp_path)
    real_copytree = slots.shutil.copytree

    # copytree recurses through the public name, so key the failure on the
    # destination: slot 0 copies fully, slot 1's copy fails at its root.
    def flaky_copytree(src, dst, *args, **kwargs):
        if "slot-1" in str(dst):
            raise OSError("simulated disk failure")
        return real_copytree(src, dst, *args, **kwargs)

    monkeypatch.setattr(slots.shutil, "copytree", flaky_copytree)
    provider = CopySlotProvider(project, 3)
    made = provider.provision()
    try:
        assert len(made) == 1  # effective K = slots that fully succeeded
        assert provider.base_dir is not None
        assert not (provider.base_dir / "slot-1").exists()  # partial slot discarded
    finally:
        monkeypatch.setattr(slots.shutil, "copytree", real_copytree)
        provider.cleanup()


def test_full_env_reflects_current_process_env_at_call_time(tmp_path):
    """Sanity: full_env snapshots os.environ each call (overlay, not replacement)."""
    slot = Slot(index=0, root=tmp_path, cwd=tmp_path, env={"WECO_SLOT": "0"})
    assert set(os.environ).issubset(set(slot.full_env()))
