# weco/git.py
"""
Git utilities for command execution and validation.
"""

import pathlib
import shutil
import subprocess


class GitError(Exception):
    """Raised when a git command fails."""

    def __init__(self, message: str, stderr: str = ""):
        super().__init__(message)
        self.stderr = stderr


class GitNotFoundError(Exception):
    """Raised when git is not available on the system."""

    pass


def is_available() -> bool:
    """Check if git is available on the system."""
    return shutil.which("git") is not None


def is_repo(path: pathlib.Path) -> bool:
    """Check if a directory is a git repository."""
    return (path / ".git").is_dir()


def validate_ref(ref: str) -> None:
    """
    Validate a git ref to prevent option injection.

    Raises:
        ValueError: If ref could be interpreted as a git option.
    """
    if ref.startswith("-"):
        raise ValueError(f"Invalid git ref: {ref!r} (cannot start with '-')")


def validate_repo_url(url: str) -> None:
    """
    Validate a git repository URL.

    Raises:
        ValueError: If URL doesn't match expected patterns.
    """
    valid_prefixes = ("git@", "https://", "http://", "ssh://", "file://", "/", "./", "../")
    if not any(url.startswith(prefix) for prefix in valid_prefixes):
        raise ValueError(f"Invalid repository URL: {url!r}")


def run(*args: str, cwd: pathlib.Path | None = None, error_msg: str = "Git command failed") -> subprocess.CompletedProcess:
    """
    Run a git command and return the result.

    Args:
        *args: Git subcommand and arguments (e.g., "clone", url, path).
               Do NOT include "git" — it's prepended automatically.
        cwd: Working directory for the command.
        error_msg: Message to include in exception on failure.

    Raises:
        GitError: If the command fails or returns non-zero.
    """
    cmd = ["git", *args]
    try:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    except Exception as e:
        raise GitError(f"{error_msg}: {e}") from e

    if result.returncode != 0:
        raise GitError(error_msg, result.stderr)
    return result


def clone(repo_url: str, dest: pathlib.Path, ref: str | None = None) -> None:
    """
    Clone a git repository.

    Args:
        repo_url: The repository URL to clone.
        dest: Destination directory.
        ref: Optional branch, tag, or commit to checkout after cloning.

    Raises:
        GitError: If clone or checkout fails.
    """
    run("clone", repo_url, str(dest), error_msg="Failed to clone repository")
    if ref:
        run("checkout", ref, cwd=dest, error_msg=f"Failed to checkout '{ref}'")


def pull(repo_path: pathlib.Path) -> None:
    """
    Pull latest changes in a git repository.

    Raises:
        GitError: If pull fails.
    """
    run("pull", cwd=repo_path, error_msg="Failed to pull repository")


def fetch_and_checkout(repo_path: pathlib.Path, ref: str) -> None:
    """
    Fetch all remotes and checkout a specific ref.

    Args:
        repo_path: Path to the git repository.
        ref: Branch, tag, or commit to checkout.

    Raises:
        GitError: If fetch or checkout fails.
    """
    run("fetch", "--all", cwd=repo_path, error_msg="Failed to fetch from repository")
    run("checkout", ref, cwd=repo_path, error_msg=f"Failed to checkout '{ref}'")
