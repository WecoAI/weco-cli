"""Skill safety and installation primitives."""

import pathlib

from rich.console import Console

from ...utils import (
    DownloadError,
    UnsafeRemoveError,
    copy_directory,
    copy_file,
    download_github_archive,
    safe_remove_directory,
)
from .targets import SETUP_TARGETS, SetupTarget


class SetupError(Exception):
    """Base exception for setup failures."""


class InvalidLocalRepoError(SetupError):
    """Raised when a local path is not a valid skill repository."""


class SafetyError(SetupError):
    """Raised when a safety check fails during directory operations."""


WECO_SKILL_REPO_URL = "https://github.com/WecoAI/weco-skill"
WECO_SKILL_BRANCH = "main"

_COPY_IGNORE_PATTERNS = {".git", "__pycache__", ".DS_Store"}
_ALLOWED_SKILL_PARENTS = {target.install_parent for target in SETUP_TARGETS}
_SKILL_DIR_NAME = "weco"


def _safe_remove_skill_dir(path: pathlib.Path) -> None:
    """Remove an installed skill directory, enforcing skill-specific safety."""
    try:
        safe_remove_directory(path, allowed_parents=_ALLOWED_SKILL_PARENTS, expected_name=_SKILL_DIR_NAME)
    except UnsafeRemoveError as e:
        raise SafetyError(str(e)) from e


def _validate_local_skill_repo(local_path: pathlib.Path) -> None:
    """Validate that a local path is a valid weco-skill repository."""
    if not local_path.exists():
        raise InvalidLocalRepoError(f"Local path does not exist: {local_path}")
    if not local_path.is_dir():
        raise InvalidLocalRepoError(f"Local path is not a directory: {local_path}")
    if not (local_path / "SKILL.md").exists():
        raise InvalidLocalRepoError(
            f"Local path does not appear to be a weco-skill repository (expected SKILL.md at {local_path / 'SKILL.md'})"
        )


def download_skill_archive(dest: pathlib.Path, console: Console) -> None:
    """Download the Weco skill archive into ``dest`` and validate it looks like a skill repo."""
    dest.mkdir(parents=True, exist_ok=True)
    url = f"{WECO_SKILL_REPO_URL}/archive/refs/heads/{WECO_SKILL_BRANCH}.zip"
    console.print("[cyan]Downloading Weco skill...[/] ", end="")
    download_github_archive(url, dest)
    console.print("[green]done.[/]\n")
    if not (dest / "SKILL.md").exists():
        raise DownloadError("Downloaded content does not appear to be a valid weco-skill repository")


def install_target(target: SetupTarget, console: Console, source_path: pathlib.Path) -> None:
    """Install the Weco skill for a single target by copying from ``source_path``."""
    _validate_local_skill_repo(source_path)

    target.install_dir.parent.mkdir(parents=True, exist_ok=True)
    if target.install_dir.exists():
        _safe_remove_skill_dir(target.install_dir)

    copy_directory(source_path, target.install_dir, ignore_patterns=_COPY_IGNORE_PATTERNS)

    for src_rel, dst_rel in target.extra_files:
        copy_file(target.install_dir / src_rel, target.install_dir / dst_rel)

    console.print(f"Installing Weco for {target.label}... [green]done[/] [dim]({_shorten(target.install_dir)})[/]")


def _shorten(path: pathlib.Path) -> str:
    """Return a ``~``-abbreviated path string, or the absolute path if not under home."""
    home = pathlib.Path.home()
    try:
        return f"~/{path.relative_to(home)}"
    except ValueError:
        return str(path)
