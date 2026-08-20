"""Isolated evaluation slots for K-parallel runs.

A *slot* is an isolated checkout of the user's project plus a per-slot
environment overlay. K>1 evaluation writes candidate code only inside its
assigned slot and runs the eval command there, so concurrent evaluations never
touch the user's working tree (which is only modified by the explicit final
"apply best solution" step, under the working-tree consumer lock).

Git projects default to detached worktrees with the user's complete uncommitted
state overlaid. Non-git projects use plain directory copies placed *outside*
the source project (system temp), excluding recursive/control artifacts.

The zero-config copy provider is best effort: it establishes filesystem
isolation but cannot prove an arbitrary evaluation command avoids shared
external resources (fixed ports, GPUs, databases, absolute-path writes).
Callers must surface :data:`BEST_EFFORT_WARNING` when provisioning for a run.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field

from .utils import safe_remove_directory, UnsafeRemoveError


# Names never copied into a slot: recursive/control artifacts. The run's actual
# log_dir is added at provision time (it defaults to `.runs` but is user-set).
CONTROL_EXCLUDES = {".git", ".weco", ".runs", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".DS_Store"}

# Heavy environment directories that are symlinked into the slot instead of
# copied: copying them is prohibitively slow and their absolute-path internals
# don't survive relocation anyway. A read-mostly symlink keeps `./.venv/bin/…`
# eval commands working; concurrent *writes* through them (e.g. `pip install`
# inside the eval) are part of the documented best-effort boundary.
SYMLINK_SHARED = {".venv", "venv", "node_modules"}

BEST_EFFORT_WARNING = (
    "Parallel slots are best-effort copies: files are isolated (separate slot dirs, temp dirs, "
    "compile caches), but the eval command itself must avoid fixed ports, shared databases, "
    "absolute-path writes, and unmanaged GPU contention. venvs/node_modules are shared symlinks "
    "and must be treated as read-only. Use WECO_SLOT / WECO_PORT_OFFSET in your eval script to disambiguate slots."
)


class SlotProvisionError(Exception):
    """Provisioning a slot failed (copy error, disk, unexpected layout)."""


class SlotPathError(Exception):
    """A candidate file path would escape its slot (absolute, `..`, or symlink)."""


@dataclass
class Slot:
    """One isolated evaluation slot.

    ``root`` is the slot-owned area (outside the project); ``cwd`` is the
    project copy evaluations run in; ``env`` is the per-slot *overlay* — the
    evaluator merges it over the parent process environment, never replacing it
    (auth, PATH, and user settings must survive).
    """

    index: int
    root: pathlib.Path
    cwd: pathlib.Path
    env: dict[str, str] = field(default_factory=dict)

    def full_env(self) -> dict[str, str]:
        """The complete environment for this slot's evaluations."""
        merged = dict(os.environ)
        merged.update(self.env)
        return merged


def resolve_candidate_path(base: pathlib.Path, rel_path: str) -> pathlib.Path:
    """Resolve a candidate file path strictly inside ``base``.

    Rejects absolute paths, Windows drive prefixes, any ``..`` component, and
    paths whose existing ancestors resolve (via symlinks) outside ``base``.
    Returns the path to write to. Callers must write via
    :func:`prepare_write_target`, which re-checks the LEAF at write time — an
    eval command may have replaced the file itself with a symlink after this
    resolution (opening one for write would follow it out of the slot).
    """
    normalized = rel_path.replace("\\", "/")
    pure = pathlib.PurePosixPath(normalized)
    if pure.is_absolute() or normalized.startswith("/"):
        raise SlotPathError(f"Refusing candidate path outside the slot (absolute): {rel_path!r}")
    parts = pure.parts
    if not parts:
        raise SlotPathError(f"Refusing empty candidate path: {rel_path!r}")
    if any(part == ".." for part in parts):
        raise SlotPathError(f"Refusing candidate path with traversal: {rel_path!r}")
    if ":" in parts[0]:
        raise SlotPathError(f"Refusing candidate path with drive prefix: {rel_path!r}")

    base_resolved = base.resolve()
    target = base.joinpath(*parts)

    # Symlink escape: resolve the deepest *existing* ancestor and require it to
    # stay inside the slot (the file itself may not exist yet).
    probe = target.parent
    while not probe.exists() and probe != base:
        probe = probe.parent
    probe_resolved = probe.resolve()
    if probe_resolved != base_resolved and base_resolved not in probe_resolved.parents:
        raise SlotPathError(f"Refusing candidate path escaping the slot via symlink: {rel_path!r}")
    if target.is_symlink():
        raise SlotPathError(f"Refusing candidate path that is a symlink: {rel_path!r}")
    return target


def prepare_write_target(base: pathlib.Path, rel_path: str) -> pathlib.Path:
    """Re-validate a slot path at WRITE time and make it safe to open.

    The eval command runs arbitrary code inside the slot, so between path
    resolution and a later write (the post-eval restore) it may have replaced
    the file — or any ancestor directory — with a symlink pointing outside the
    slot; ``open(..., "w")`` would follow it. Re-run the ancestor escape probe,
    unlink a symlink leaf so the write recreates a regular file, and create
    parent directories. Raises :class:`SlotPathError` when ancestors escape.
    """
    parts = pathlib.PurePosixPath(rel_path.replace("\\", "/")).parts
    base_resolved = base.resolve()
    target = base.joinpath(*parts)
    probe = target.parent
    while not probe.exists() and probe != base:
        probe = probe.parent
    probe_resolved = probe.resolve()
    if probe_resolved != base_resolved and base_resolved not in probe_resolved.parents:
        raise SlotPathError(f"Refusing write through an ancestor escaping the slot: {rel_path!r}")
    if target.is_symlink():
        target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _discover_cuda_pool() -> list[str] | None:
    """The GPU indices slots may be assigned, or ``None`` when unknowable.

    An explicit user ``CUDA_VISIBLE_DEVICES`` is preserved as the pool (slots
    round-robin within the user's restriction; an empty restriction stays empty
    for every slot). Otherwise ``nvidia-smi`` is probed; when no device info is
    discoverable we leave the variable untouched — never invent a GPU.
    """
    user = os.environ.get("CUDA_VISIBLE_DEVICES")
    if user is not None:
        ids = [d.strip() for d in user.split(",") if d.strip()]
        return ids if ids else [""]
    try:
        result = subprocess.run(["nvidia-smi", "--list-gpus"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    count = len([line for line in result.stdout.splitlines() if line.strip()])
    return [str(i) for i in range(count)] if count else None


def build_slot_env(slot_root: pathlib.Path, index: int, cuda_pool: list[str] | None) -> dict[str, str]:
    """The per-slot environment overlay.

    Separate temp and compile-cache trees are what keep the flagship CUDA/Triton
    examples from corrupting each other's JIT caches; ``WECO_SLOT`` and
    ``WECO_PORT_OFFSET`` are deterministic hooks for eval scripts (we never
    rewrite arbitrary user port variables).
    """
    tmp_dir = slot_root / "tmp"
    cache_dir = slot_root / "cache"
    env = {
        "WECO_SLOT": str(index),
        "TMPDIR": str(tmp_dir),
        "XDG_CACHE_HOME": str(cache_dir),
        "TORCH_EXTENSIONS_DIR": str(cache_dir / "torch_extensions"),
        "TRITON_CACHE_DIR": str(cache_dir / "triton"),
        "WECO_PORT_OFFSET": str(index * 10),
    }
    if cuda_pool is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_pool[index % len(cuda_pool)]
    return env


# Optional per-project overlay configuration, authored by the user or the M4
# setup skill. Plain deterministic TOML the platform executes:
#
#   [slots]
#   provider = "worktree"          # or "copy"; omit for auto (worktree on git)
#   copy = ["data/small", ".env"]  # copied into every slot (per-slot private)
#   symlink = ["data/huge"]        # symlinked (shared, read-mostly)
#
# Paths are project-relative; entries that don't exist are skipped.
CONFIG_PATH = pathlib.Path(".weco") / "parallel.toml"


@dataclass
class SlotOverlayConfig:
    provider: str | None = None  # "worktree" | "copy" | None (auto)
    copy: list[str] = field(default_factory=list)
    symlink: list[str] = field(default_factory=list)


def load_overlay_config(project_dir: pathlib.Path) -> SlotOverlayConfig:
    """Read ``.weco/parallel.toml``; absent or malformed → defaults."""
    path = project_dir / CONFIG_PATH
    if not path.is_file():
        return SlotOverlayConfig()
    import tomllib

    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return SlotOverlayConfig()
    section = data.get("slots") or {}
    if not isinstance(section, dict):
        return SlotOverlayConfig()
    provider = section.get("provider")
    if provider not in (None, "worktree", "copy"):
        provider = None

    def _paths(key: str) -> list[str]:
        value = section.get(key, [])
        if not isinstance(value, list):
            return []
        valid: list[str] = []
        for entry in value:
            if not isinstance(entry, str):
                continue
            normalized = entry.replace("\\", "/")
            pure = pathlib.PurePosixPath(normalized)
            if not pure.parts or pure.is_absolute() or any(part == ".." for part in pure.parts) or ":" in pure.parts[0]:
                continue
            valid.append(pure.as_posix())
        return valid

    return SlotOverlayConfig(provider=provider, copy=_paths("copy"), symlink=_paths("symlink"))


class CopySlotProvider:
    """Provision up to K isolated project copies under a temp base directory.

    The base directory lives in the system temp area — never inside the source
    project — and is removed as a whole on cleanup via the repository's
    defensive removal helper. Provisioning is all-or-per-slot: each slot copy
    either fully succeeds or is discarded; the effective K is however many
    slots succeeded (the caller clamps behavior accordingly, never exceeding
    the request).
    """

    provider_name = "copy"

    def __init__(
        self, project_dir: pathlib.Path | str, k: int, *, log_dir: str = ".runs", overlay: SlotOverlayConfig | None = None
    ) -> None:
        if k < 1:
            raise ValueError("k must be >= 1")
        if os.name == "nt":
            # The working-tree consumer lock is a no-op on Windows, so nothing
            # enforces the single-consumer invariant parallel slots rely on.
            raise SlotProvisionError("Parallel slots are not supported on Windows yet; run with --parallel 1.")
        self.project_dir = pathlib.Path(project_dir).resolve()
        self.k = k
        self.overlay = overlay or SlotOverlayConfig()
        self.excludes = set(CONTROL_EXCLUDES)
        # The run's configured log dir may differ from the default; exclude the
        # top-level name (artifacts recurse into themselves otherwise).
        log_name = pathlib.PurePosixPath(log_dir.replace("\\", "/")).parts
        if log_name:
            self.excludes.add(log_name[0])
        self.base_dir: pathlib.Path | None = None
        self.slots: list[Slot] = []

    # -- provisioning --

    def provision(self) -> list[Slot]:
        """Create up to ``k`` slots; returns the ones that fully succeeded."""
        if not self.project_dir.is_dir():
            raise SlotProvisionError(f"Project directory not found: {self.project_dir}")
        self.base_dir = pathlib.Path(tempfile.mkdtemp(prefix="weco-slots-"))
        # mkdtemp honors the ambient TMPDIR: if that points inside the project,
        # every slot copy would recursively include the slot base itself.
        if self.base_dir.resolve().is_relative_to(self.project_dir):
            shutil.rmtree(self.base_dir, ignore_errors=True)
            self.base_dir = None
            raise SlotProvisionError("The temp directory (TMPDIR) resolves inside the project; slots must live outside it.")
        _write_registry_meta(self.base_dir, self.project_dir, self.provider_name)
        cuda_pool = _discover_cuda_pool()

        for index in range(self.k):
            slot_root = self.base_dir / f"slot-{index}"
            try:
                self.slots.append(self._provision_one(slot_root, index, cuda_pool))
            except Exception:
                # A partial slot is worse than a missing one — discard it and
                # stop trying (later copies will almost certainly fail the same
                # way, and effective K = what succeeded).
                self._discard_partial(slot_root)
                break
        return list(self.slots)

    def _discard_partial(self, slot_root: pathlib.Path) -> None:
        shutil.rmtree(slot_root, ignore_errors=True)

    def _apply_overlay(self, cwd: pathlib.Path) -> None:
        """Apply the declared per-slot extras (config-listed copies/symlinks)."""
        for rel in self.overlay.copy:
            source = self.project_dir / rel
            target = resolve_candidate_path(cwd, rel)
            if not source.exists() or target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                if source.is_dir():
                    shutil.copytree(source, target, symlinks=True)
                else:
                    shutil.copy2(source, target)
            except (OSError, shutil.Error) as e:
                raise SlotProvisionError(f"Failed to copy declared overlay entry {rel!r}: {e}") from e
        for rel in self.overlay.symlink:
            source = self.project_dir / rel
            target = resolve_candidate_path(cwd, rel)
            if not source.exists() or target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                target.symlink_to(source)
            except OSError as e:
                raise SlotProvisionError(f"Failed to symlink declared overlay entry {rel!r}: {e}") from e

    def _provision_one(self, slot_root: pathlib.Path, index: int, cuda_pool: list[str] | None) -> Slot:
        cwd = slot_root / "project"
        exclude = self.excludes
        symlink_shared = SYMLINK_SHARED

        def _ignore(directory: str, names: list[str]) -> list[str]:
            skipped = [n for n in names if n in exclude]
            # Top-level only: shared env dirs are re-linked after the copy.
            if pathlib.Path(directory).resolve() == self.project_dir:
                skipped += [n for n in names if n in symlink_shared]
            return skipped

        try:
            shutil.copytree(self.project_dir, cwd, ignore=_ignore, symlinks=True)
        except (OSError, shutil.Error) as e:
            raise SlotProvisionError(f"Failed to copy project into slot {index}: {e}") from e

        for name in symlink_shared:
            source = self.project_dir / name
            if source.exists() and not (cwd / name).exists():
                try:
                    (cwd / name).symlink_to(source)
                except OSError:
                    pass  # The eval may not need it; best-effort by design.

        self._apply_overlay(cwd)

        env = build_slot_env(slot_root, index, cuda_pool)
        for key in ("TMPDIR", "XDG_CACHE_HOME", "TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR"):
            pathlib.Path(env[key]).mkdir(parents=True, exist_ok=True)
        return Slot(index=index, root=slot_root, cwd=cwd, env=env)

    def refresh_registry_meta(self) -> None:
        """Re-stamp the slot registry with the CURRENT pid.

        Provisioning may happen in a parent process that later daemon-forks and
        exits (``weco run --parallel K --daemon``): the recorded parent pid dies
        immediately, which would make ``weco slots clean`` classify the daemon's
        live slots as stale and delete them mid-evaluation. The consumer calls
        this once it is the process that will actually run the scheduler.
        """
        if self.base_dir is not None:
            _write_registry_meta(self.base_dir, self.project_dir, self.provider_name)

    # -- teardown --

    def cleanup(self) -> None:
        """Best-effort removal of every slot (called on all exit paths)."""
        if self.base_dir is None:
            return
        try:
            safe_remove_directory(
                self.base_dir, allowed_parents={pathlib.Path(tempfile.gettempdir())}, expected_name=self.base_dir.name
            )
        except UnsafeRemoveError:
            # Refusing to delete is the correct failure mode; leave the dir for
            # manual cleanup (`weco slots clean` finds it later).
            pass
        except OSError:
            pass
        self.base_dir = None
        self.slots = []


def _git(project_dir: pathlib.Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(["git", "-C", str(project_dir), *args], capture_output=True, text=True, timeout=120)
    if check and result.returncode != 0:
        raise SlotProvisionError(f"git {' '.join(args)} failed: {result.stderr.strip() or result.stdout.strip()}")
    return result


def is_git_project(project_dir: pathlib.Path) -> bool:
    """True when the project is a git work tree with at least one commit."""
    try:
        inside = _git(project_dir, "rev-parse", "--is-inside-work-tree", check=False)
        if inside.returncode != 0 or inside.stdout.strip() != "true":
            return False
        head = _git(project_dir, "rev-parse", "--verify", "HEAD", check=False)
        return head.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


class WorktreeSlotProvider(CopySlotProvider):
    """Git-worktree-based slots: the M3b default for git projects.

    ``git worktree add --detach`` materializes the tracked tree at HEAD nearly
    for free; on top of that each slot receives (1) the working tree's
    uncommitted state — modified and untracked non-ignored files are copied in,
    deleted files removed — so slots match what the user actually has, and
    (2) the declared overlay for things git deliberately doesn't track (data
    dirs, ``.env``, venvs). The same shared-env symlinks as the copy provider
    apply by default; ``.weco/parallel.toml`` extends or overrides.
    """

    provider_name = "worktree"

    def provision(self) -> list[Slot]:
        if not is_git_project(self.project_dir):
            raise SlotProvisionError(f"Not a git repository with a commit: {self.project_dir}")
        return super().provision()

    def _provision_one(self, slot_root: pathlib.Path, index: int, cuda_pool: list[str] | None) -> Slot:
        cwd = slot_root / "project"
        slot_root.mkdir(parents=True, exist_ok=True)
        _git(self.project_dir, "worktree", "add", "--detach", str(cwd), "HEAD")

        # Overlay the complete uncommitted state: what the user runs is their
        # tree, not HEAD. Diff against HEAD so staged-only modifications are
        # included; add untracked non-ignored files separately. Disable rename
        # detection so a rename is represented as an old-path deletion plus a
        # new-path copy. NUL-separated throughout to survive any filename.
        tracked_changed = _git(self.project_dir, "diff", "--name-only", "--no-renames", "--diff-filter=ACMRTUXB", "-z", "HEAD")
        untracked = _git(self.project_dir, "ls-files", "--others", "--exclude-standard", "-z")
        changed = set(tracked_changed.stdout.split("\0")) | set(untracked.stdout.split("\0"))
        for rel in sorted(p for p in changed if p):
            if pathlib.PurePosixPath(rel).parts and pathlib.PurePosixPath(rel).parts[0] in self.excludes:
                continue
            source = self.project_dir / rel
            if not source.is_file():
                continue
            target = cwd / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        deleted = _git(self.project_dir, "diff", "--name-only", "--no-renames", "--diff-filter=D", "-z", "HEAD")
        for rel in [p for p in deleted.stdout.split("\0") if p]:
            try:
                (cwd / rel).unlink(missing_ok=True)
            except OSError:
                pass

        # Shared env dirs (gitignored, so absent from the worktree) — same
        # best-effort symlinks as the copy provider, then the declared overlay.
        for name in SYMLINK_SHARED:
            source = self.project_dir / name
            if source.exists() and not (cwd / name).exists():
                try:
                    (cwd / name).symlink_to(source)
                except OSError:
                    pass
        # `.env` is almost always gitignored yet required to run; copy it (per
        # slot, so an eval editing it can't corrupt siblings) unless declared.
        env_file = self.project_dir / ".env"
        declared = set(self.overlay.copy) | set(self.overlay.symlink)
        if env_file.is_file() and ".env" not in declared and not (cwd / ".env").exists():
            try:
                shutil.copy2(env_file, cwd / ".env")
            except OSError:
                pass

        self._apply_overlay(cwd)

        env = build_slot_env(slot_root, index, cuda_pool)
        for key in ("TMPDIR", "XDG_CACHE_HOME", "TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR"):
            pathlib.Path(env[key]).mkdir(parents=True, exist_ok=True)
        return Slot(index=index, root=slot_root, cwd=cwd, env=env)

    def _discard_partial(self, slot_root: pathlib.Path) -> None:
        self._remove_worktree(slot_root / "project")
        shutil.rmtree(slot_root, ignore_errors=True)

    def _remove_worktree(self, cwd: pathlib.Path) -> None:
        try:
            _git(self.project_dir, "worktree", "remove", "--force", str(cwd), check=False)
        except (OSError, subprocess.SubprocessError):
            pass

    def cleanup(self) -> None:
        for slot in self.slots:
            self._remove_worktree(slot.cwd)
        try:
            _git(self.project_dir, "worktree", "prune", check=False)
        except (OSError, subprocess.SubprocessError):
            pass
        super().cleanup()


# ---------------------------------------------------------------------------
# Registry (stale-slot discovery for `weco slots clean`)
# ---------------------------------------------------------------------------

_REGISTRY_META = "weco-slots-meta.json"
_REGISTRY_SCHEMA = "weco-slots-v1"


def _write_registry_meta(base_dir: pathlib.Path, project_dir: pathlib.Path, provider: str) -> None:
    import json

    meta = {
        "schema": _REGISTRY_SCHEMA,
        "pid": os.getpid(),
        "project": str(project_dir),
        "provider": provider,
        "created_at": __import__("datetime").datetime.now().isoformat(),
    }
    try:
        (base_dir / _REGISTRY_META).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    except OSError:
        pass


def find_stale_slot_dirs() -> list[tuple[pathlib.Path, dict]]:
    """Slot base dirs in the temp area whose owning process is gone.

    Returns ``(base_dir, meta)`` pairs. Missing, corrupt, or unrecognized
    markers are skipped: a name prefix alone is never enough authority to
    delete a temp directory.
    """
    import json

    stale: list[tuple[pathlib.Path, dict]] = []
    temp_root = pathlib.Path(tempfile.gettempdir())
    try:
        candidates = [p for p in temp_root.iterdir() if p.is_dir() and p.name.startswith("weco-slots-")]
    except OSError:
        return []
    for base in candidates:
        try:
            meta = json.loads((base / _REGISTRY_META).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if meta.get("schema") != _REGISTRY_SCHEMA or meta.get("provider") not in {"copy", "worktree"}:
            continue
        pid = meta.get("pid")
        project = meta.get("project")
        # Cleanup is destructive, so require a complete marker written by this
        # version before using it as deletion authority. bool is an int subclass
        # but never a valid owner PID.
        if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0 or not isinstance(project, str) or not project:
            continue
        if _pid_alive(pid):
            continue  # in use by a live consumer
        stale.append((base, meta))
    return stale


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def clean_stale_slots() -> list[pathlib.Path]:
    """Remove every stale slot base dir; returns the paths removed."""
    removed: list[pathlib.Path] = []
    for base, meta in find_stale_slot_dirs():
        project = meta.get("project")
        if meta.get("provider") == "worktree" and project and pathlib.Path(project).is_dir():
            for slot_project in base.glob("slot-*/project"):
                try:
                    _git(pathlib.Path(project), "worktree", "remove", "--force", str(slot_project), check=False)
                except (OSError, subprocess.SubprocessError):
                    pass
            try:
                _git(pathlib.Path(project), "worktree", "prune", check=False)
            except (OSError, subprocess.SubprocessError):
                pass
        try:
            safe_remove_directory(base, allowed_parents={pathlib.Path(tempfile.gettempdir())}, expected_name=base.name)
            removed.append(base)
        except (UnsafeRemoveError, OSError):
            continue
    return removed


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_slot_provider(project_dir: pathlib.Path | str, k: int, *, log_dir: str = ".runs") -> CopySlotProvider:
    """The provider for this project: config choice > worktree-on-git > copy.

    Raises :class:`SlotProvisionError` on unsupported platforms (Windows —
    the consumer lock is a no-op there, so K>1 cannot be made safe).
    """
    project = pathlib.Path(project_dir).resolve()
    overlay = load_overlay_config(project)
    if overlay.provider == "copy":
        return CopySlotProvider(project, k, log_dir=log_dir, overlay=overlay)
    if overlay.provider == "worktree" or is_git_project(project):
        return WorktreeSlotProvider(project, k, log_dir=log_dir, overlay=overlay)
    return CopySlotProvider(project, k, log_dir=log_dir, overlay=overlay)
