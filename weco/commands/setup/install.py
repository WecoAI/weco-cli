"""Skill download, validation, and installation."""

import io
import pathlib
import shutil
import tempfile
import zipfile
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from rich.console import Console

from ...core.files import copy_directory, copy_file, safe_remove_directory
from .errors import DownloadError, InvalidLocalRepoError
from .paths import (
    ALLOWED_SKILL_PARENTS,
    COPY_IGNORE_PATTERNS,
    WECO_SKILL_BRANCH,
    WECO_SKILL_REPO_URL,
)


def validate_local_skill_repo(local_path: pathlib.Path) -> None:
    """Validate that a local path is a valid weco-skill repository.

    Raises:
        InvalidLocalRepoError: If validation fails.
    """
    if not local_path.exists():
        raise InvalidLocalRepoError(f"Local path does not exist: {local_path}")
    if not local_path.is_dir():
        raise InvalidLocalRepoError(f"Local path is not a directory: {local_path}")
    if not (local_path / "SKILL.md").exists():
        raise InvalidLocalRepoError(
            f"Local path does not appear to be a weco-skill repository "
            f"(expected SKILL.md at {local_path / 'SKILL.md'})"
        )


def _get_zip_url() -> str:
    """Build the GitHub zip download URL."""
    return f"{WECO_SKILL_REPO_URL}/archive/refs/heads/{WECO_SKILL_BRANCH}.zip"


def _download_and_extract_zip(dest: pathlib.Path, console: Console) -> None:
    """Download the weco-skill zip and extract its contents to *dest*.

    Args:
        dest: Destination directory for extracted contents.
        console: Console for progress output.

    Raises:
        DownloadError: If download or extraction fails.
    """
    url = _get_zip_url()
    console.print(f"[cyan]Downloading skill from {url}...[/]")

    try:
        with urlopen(url, timeout=60) as response:
            zip_data = io.BytesIO(response.read())
    except HTTPError as e:
        raise DownloadError(f"Failed to download: HTTP {e.code} - {e.reason}")
    except URLError as e:
        raise DownloadError(f"Failed to download: {e.reason}")
    except TimeoutError:
        raise DownloadError("Failed to download: connection timed out")
    except Exception as e:
        raise DownloadError(f"Failed to download: {e}")

    try:
        with zipfile.ZipFile(zip_data) as zf:
            top_level_dirs = {name.split("/")[0] for name in zf.namelist() if "/" in name}
            if len(top_level_dirs) != 1:
                raise DownloadError("Unexpected zip structure: expected single top-level directory")

            top_dir = top_level_dirs.pop()
            prefix = f"{top_dir}/"
            dest.mkdir(parents=True, exist_ok=True)

            for member in zf.namelist():
                if not member.startswith(prefix):
                    continue
                relative_path = member[len(prefix):]
                if not relative_path:
                    continue
                target_path = dest / relative_path
                if member.endswith("/"):
                    target_path.mkdir(parents=True, exist_ok=True)
                else:
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(member) as source, open(target_path, "wb") as target:
                        target.write(source.read())

            console.print("[green]Skill downloaded and extracted successfully.[/]")

    except zipfile.BadZipFile:
        raise DownloadError("Downloaded file is not a valid zip archive")
    except DownloadError:
        raise
    except Exception as e:
        raise DownloadError(f"Failed to extract zip: {e}")


def install_from_zip(skill_dir: pathlib.Path, console: Console) -> None:
    """Download and install the skill from GitHub.

    Args:
        skill_dir: Destination directory for the skill.
        console: Console for progress output.

    Raises:
        DownloadError: If download or extraction fails.
    """
    skill_dir.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir) / "skill"
        _download_and_extract_zip(tmp_path, console)

        if not (tmp_path / "SKILL.md").exists():
            raise DownloadError("Downloaded content does not appear to be a valid weco-skill repository")

        if skill_dir.exists():
            console.print(f"[cyan]Replacing existing skill at {skill_dir}...[/]")
            safe_remove_directory(skill_dir, ALLOWED_SKILL_PARENTS)

        shutil.move(str(tmp_path), str(skill_dir))

    console.print("[green]Skill installed successfully.[/]")


def install_from_local(skill_dir: pathlib.Path, console: Console, local_path: pathlib.Path) -> None:
    """Copy the skill from a local directory.

    Raises:
        InvalidLocalRepoError: If *local_path* is invalid.
    """
    validate_local_skill_repo(local_path)

    skill_dir.parent.mkdir(parents=True, exist_ok=True)
    if skill_dir.exists():
        console.print(f"[cyan]Removing existing directory at {skill_dir}...[/]")
        safe_remove_directory(skill_dir, ALLOWED_SKILL_PARENTS)

    copy_directory(local_path, skill_dir, ignore_patterns=COPY_IGNORE_PATTERNS)
    console.print(f"[green]Copied local repo from: {local_path}[/]")


def install_skill(skill_dir: pathlib.Path, console: Console, local_path: pathlib.Path | None) -> None:
    """Install the skill from a local path or GitHub."""
    if local_path:
        install_from_local(skill_dir, console, local_path)
    else:
        install_from_zip(skill_dir, console)
