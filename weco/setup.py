# weco/setup.py
"""
Setup commands for integrating Weco with various AI tools.
"""

import io
import pathlib
import shutil
import sys
import tempfile
import time
import zipfile
from collections.abc import Callable
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from rich.console import Console
from rich.prompt import Prompt

from .events import (
    send_event,
    create_event_context,
    SkillInstallStartedEvent,
    SkillInstallCompletedEvent,
    SkillInstallFailedEvent,
)
from .setup_targets import (
    ALL_SETUP_OPTION_LABEL,
    ALL_SETUP_OPTION_NAME,
    SETUP_TARGET_BY_NAME,
    SETUP_TARGET_NAMES,
    SETUP_TARGETS,
)
from .utils import copy_directory, copy_file


# =============================================================================
# Exceptions
# =============================================================================


class SetupError(Exception):
    """Base exception for setup failures."""

    pass


class InvalidLocalRepoError(SetupError):
    """Raised when a local path is not a valid skill repository."""

    pass


class DownloadError(SetupError):
    """Raised when downloading the skill fails."""

    pass


class SafetyError(SetupError):
    """Raised when a safety check fails during directory operations."""

    pass


# =============================================================================
# Path constants
# =============================================================================

# Skill repository
WECO_SKILL_REPO_URL = "https://github.com/WecoAI/weco-skill"
WECO_SKILL_BRANCH = "main"

# Claude Code paths
WECO_SKILL_DIR = SETUP_TARGET_BY_NAME["claude-code"].install_dir
WECO_CLAUDE_SNIPPET_PATH = WECO_SKILL_DIR / "snippets" / "claude.md"
WECO_CLAUDE_MD_PATH = WECO_SKILL_DIR / "CLAUDE.md"
# Other tool paths
CURSOR_WECO_SKILL_DIR = SETUP_TARGET_BY_NAME["cursor"].install_dir
CODEX_WECO_SKILL_DIR = SETUP_TARGET_BY_NAME["codex"].install_dir
OPENCLAW_WECO_SKILL_DIR = SETUP_TARGET_BY_NAME["openclaw"].install_dir

# Files/directories to skip when copying local repos
_COPY_IGNORE_PATTERNS = {".git", "__pycache__", ".DS_Store"}

# Allowed parent directories for safe removal (defense in depth)
_ALLOWED_SKILL_PARENTS = {target.install_parent for target in SETUP_TARGETS}


# =============================================================================
# Safety utilities
# =============================================================================


def safe_remove_directory(path: pathlib.Path, allowed_parents: set[pathlib.Path] | None = None) -> None:
    """
    Safely remove a directory with multiple defensive checks.

    This function is paranoid by design to prevent accidental deletion of
    important directories due to bugs, path traversal, or misconfiguration.

    Args:
        path: The directory to remove.
        allowed_parents: Optional set of allowed parent directories. If provided,
                         the path must be a direct child of one of these directories.

    Raises:
        SafetyError: If any safety check fails.
    """
    if allowed_parents is None:
        allowed_parents = _ALLOWED_SKILL_PARENTS

    # Resolve to absolute path to prevent path traversal tricks
    resolved_path = path.resolve()

    # Safety check 1: Path must exist (if not, nothing to do)
    if not resolved_path.exists():
        return

    # Safety check 2: Must be a directory, not a file or symlink to file
    if not resolved_path.is_dir():
        raise SafetyError(f"Refusing to remove: not a directory: {resolved_path}")

    # Safety check 3: Must not be a symlink (could point anywhere)
    if resolved_path.is_symlink():
        raise SafetyError(f"Refusing to remove: path is a symlink: {resolved_path}")

    # Safety check 4: Must not be the home directory or root
    home = pathlib.Path.home().resolve()
    if resolved_path == home:
        raise SafetyError(f"Refusing to remove: path is home directory: {resolved_path}")
    if resolved_path == pathlib.Path("/").resolve():
        raise SafetyError(f"Refusing to remove: path is root directory: {resolved_path}")

    # Safety check 5: Must not be a parent of home directory
    try:
        home.relative_to(resolved_path)
        raise SafetyError(f"Refusing to remove: path is a parent of home directory: {resolved_path}")
    except ValueError:
        pass  # Good - resolved_path is not a parent of home

    # Safety check 6: Must be a direct child of one of the allowed parent directories
    resolved_allowed = {p.resolve() for p in allowed_parents}
    parent = resolved_path.parent

    if parent not in resolved_allowed:
        raise SafetyError(
            f"Refusing to remove: path {resolved_path} is not a direct child of allowed directories: "
            f"{[str(p) for p in resolved_allowed]}"
        )

    # Safety check 7: Verify the path is exactly 1 level below the allowed parent
    # This is redundant with check 6 but provides defense in depth
    try:
        relative = resolved_path.relative_to(parent)
        if len(relative.parts) != 1:
            raise SafetyError(f"Refusing to remove: path {resolved_path} is not exactly 1 level below parent {parent}")
    except ValueError:
        raise SafetyError(f"Refusing to remove: path {resolved_path} is not relative to any allowed parent")

    # Safety check 8: Directory name must be 'weco' (our expected skill directory name)
    if resolved_path.name != "weco":
        raise SafetyError(f"Refusing to remove: directory name is not 'weco': {resolved_path}")

    # All checks passed - safe to remove
    shutil.rmtree(resolved_path)


# =============================================================================
# Domain helpers
# =============================================================================


def validate_local_skill_repo(local_path: pathlib.Path) -> None:
    """
    Validate that a local path is a valid weco-skill repository.

    Raises:
        InvalidLocalRepoError: If validation fails.
    """
    if not local_path.exists():
        raise InvalidLocalRepoError(f"Local path does not exist: {local_path}")
    if not local_path.is_dir():
        raise InvalidLocalRepoError(f"Local path is not a directory: {local_path}")
    if not (local_path / "SKILL.md").exists():
        raise InvalidLocalRepoError(
            f"Local path does not appear to be a weco-skill repository (expected SKILL.md at {local_path / 'SKILL.md'})"
        )


# =============================================================================
# Download functions
# =============================================================================


def get_zip_url() -> str:
    """
    Build the GitHub zip download URL.

    Returns:
        The zip download URL for the weco-skill repository.
    """
    return f"{WECO_SKILL_REPO_URL}/archive/refs/heads/{WECO_SKILL_BRANCH}.zip"


def download_and_extract_zip(dest: pathlib.Path, console: Console) -> None:
    """
    Download the weco-skill zip and extract its contents.

    GitHub archives have a single top-level directory (e.g., 'weco-skill-main/'),
    so we extract that directory's contents directly to dest.

    Args:
        dest: Destination directory for extracted contents.
        console: Console for output.

    Raises:
        DownloadError: If download or extraction fails.
    """
    url = get_zip_url()
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
            # GitHub zips have a single top-level dir like 'repo-branch/'
            # Find it and extract contents to dest
            top_level_dirs = {name.split("/")[0] for name in zf.namelist() if "/" in name}

            if len(top_level_dirs) != 1:
                raise DownloadError("Unexpected zip structure: expected single top-level directory")

            top_dir = top_level_dirs.pop()
            prefix = f"{top_dir}/"

            dest.mkdir(parents=True, exist_ok=True)

            for member in zf.namelist():
                if not member.startswith(prefix):
                    continue

                # Strip the top-level directory prefix
                relative_path = member[len(prefix) :]
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


# =============================================================================
# Installation functions
# =============================================================================


def install_skill_from_zip(skill_dir: pathlib.Path, console: Console) -> None:
    """
    Download and install skill from GitHub zip archive.

    Downloads to a temporary directory first, validates the contents,
    then moves to the final location for safer installation.

    Args:
        skill_dir: Destination directory for the skill.
        console: Console for output.

    Raises:
        DownloadError: If download or extraction fails.
        SafetyError: If directory removal fails safety checks.
    """
    skill_dir.parent.mkdir(parents=True, exist_ok=True)

    # Download to a temp location first for atomic replacement
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir) / "skill"

        download_and_extract_zip(tmp_path, console)

        # Validate before replacing
        if not (tmp_path / "SKILL.md").exists():
            raise DownloadError("Downloaded content does not appear to be a valid weco-skill repository")

        # Now replace existing skill safely
        if skill_dir.exists():
            console.print(f"[cyan]Replacing existing skill at {skill_dir}...[/]")
            safe_remove_directory(skill_dir)

        # Move from temp to final location
        shutil.move(str(tmp_path), str(skill_dir))

    console.print("[green]Skill installed successfully.[/]")


def install_skill_from_local(skill_dir: pathlib.Path, console: Console, local_path: pathlib.Path) -> None:
    """
    Copy skill from local path.

    Raises:
        InvalidLocalRepoError: If local path is invalid.
        SafetyError: If directory removal fails safety checks.
        OSError: If copy fails.
    """
    validate_local_skill_repo(local_path)

    skill_dir.parent.mkdir(parents=True, exist_ok=True)

    if skill_dir.exists():
        console.print(f"[cyan]Removing existing directory at {skill_dir}...[/]")
        safe_remove_directory(skill_dir)

    copy_directory(local_path, skill_dir, ignore_patterns=_COPY_IGNORE_PATTERNS)
    console.print(f"[green]Copied local repo from: {local_path}[/]")


def install_skill(skill_dir: pathlib.Path, console: Console, local_path: pathlib.Path | None) -> None:
    """Install skill by copying from local path or downloading from GitHub."""
    if local_path:
        install_skill_from_local(skill_dir, console, local_path)
    else:
        install_skill_from_zip(skill_dir, console)


# =============================================================================
# Setup commands
# =============================================================================


def _print_setup_complete(console: Console, skill_dir: pathlib.Path, local_path: pathlib.Path | None) -> None:
    """Print a shared success footer for setup commands."""
    console.print("\n[bold green]Setup complete![/]")
    if local_path:
        console.print(f"[dim]Skill copied from: {local_path}[/]")
    console.print(f"[dim]Skill installed at: {skill_dir}[/]")


def _setup_skill_target(
    console: Console,
    *,
    label: str,
    skill_dir: pathlib.Path,
    local_path: pathlib.Path | None = None,
    after_install: Callable[[], None] | None = None,
) -> None:
    """Install the Weco skill into a tool-specific directory."""
    console.print(f"[bold blue]Setting up Weco for {label}...[/]\n")
    install_skill(skill_dir, console, local_path)
    if after_install is not None:
        after_install()
    _print_setup_complete(console, skill_dir, local_path)


def setup_claude_code(console: Console, local_path: pathlib.Path | None = None) -> None:
    """Set up Weco skill for Claude Code."""
    def install_claude_snippet() -> None:
        copy_file(WECO_CLAUDE_SNIPPET_PATH, WECO_CLAUDE_MD_PATH)
        console.print("[green]CLAUDE.md installed to skill directory.[/]")

    _setup_skill_target(
        console,
        label="Claude Code",
        skill_dir=WECO_SKILL_DIR,
        local_path=local_path,
        after_install=install_claude_snippet,
    )


def setup_cursor(console: Console, local_path: pathlib.Path | None = None) -> None:
    """Set up Weco skill for Cursor."""
    _setup_skill_target(console, label="Cursor", skill_dir=CURSOR_WECO_SKILL_DIR, local_path=local_path)


def setup_codex(console: Console, local_path: pathlib.Path | None = None) -> None:
    """Set up Weco skill for Codex."""
    _setup_skill_target(console, label="Codex", skill_dir=CODEX_WECO_SKILL_DIR, local_path=local_path)


def setup_openclaw(console: Console, local_path: pathlib.Path | None = None) -> None:
    """Set up Weco skill for OpenClaw."""
    _setup_skill_target(console, label="OpenClaw", skill_dir=OPENCLAW_WECO_SKILL_DIR, local_path=local_path)


# =============================================================================
# CLI entry point
# =============================================================================

SETUP_HANDLERS = {
    "claude-code": setup_claude_code,
    "cursor": setup_cursor,
    "codex": setup_codex,
    "openclaw": setup_openclaw,
}


def prompt_tool_selection(console: Console) -> list[str]:
    """Prompt the user to select which tool(s) to set up.

    Returns:
        List of tool names to set up.
    """
    tool_names = list(SETUP_TARGET_NAMES)
    all_option = len(tool_names) + 1

    console.print("\n[bold cyan]Available tools to set up:[/]\n")
    for i, target in enumerate(SETUP_TARGETS, 1):
        console.print(f"  {i}. {target.label} [dim]({target.name})[/]")
    console.print(f"  {all_option}. {ALL_SETUP_OPTION_LABEL} [dim](default)[/]")

    valid_choices = [str(i) for i in range(1, all_option + 1)]
    choice = Prompt.ask("\n[bold]Select an option[/]", choices=valid_choices, default=str(all_option), show_choices=True)

    idx = int(choice)
    if idx == all_option:
        return tool_names
    return [tool_names[idx - 1]]


def _run_setup_for_tool(tool: str, console: Console, local_path: pathlib.Path | None, ctx) -> None:
    """Run setup for a single tool with event tracking and error handling."""
    source = "local" if local_path else "download"

    send_event(SkillInstallStartedEvent(tool=tool, source=source), ctx)

    start_time = time.time()

    try:
        handler = SETUP_HANDLERS[tool]
        handler(console, local_path=local_path)

        duration_ms = int((time.time() - start_time) * 1000)
        send_event(SkillInstallCompletedEvent(tool=tool, source=source, duration_ms=duration_ms), ctx)

    except DownloadError as e:
        send_event(SkillInstallFailedEvent(tool=tool, source=source, error_type="download_error", stage="download"), ctx)
        console.print(f"[bold red]Error:[/] {e}")
        sys.exit(1)
    except SafetyError as e:
        send_event(SkillInstallFailedEvent(tool=tool, source=source, error_type="safety_error", stage="setup"), ctx)
        console.print(f"[bold red]Safety Error:[/] {e}")
        sys.exit(1)
    except (SetupError, FileNotFoundError, OSError, ValueError) as e:
        error_type = type(e).__name__
        send_event(SkillInstallFailedEvent(tool=tool, source=source, error_type=error_type, stage="setup"), ctx)
        console.print(f"[bold red]Error:[/] {e}")
        sys.exit(1)


def handle_setup_command(args, console: Console) -> None:
    """Handle the setup command with its subcommands."""
    ctx = create_event_context()

    if args.tool is None:
        selected_tools = prompt_tool_selection(console)
    elif args.tool == ALL_SETUP_OPTION_NAME:
        selected_tools = list(SETUP_TARGET_NAMES)
    else:
        handler = SETUP_HANDLERS.get(args.tool)
        if handler is None:
            available_tools = ", ".join((*SETUP_HANDLERS, ALL_SETUP_OPTION_NAME))
            console.print(f"[bold red]Error:[/] Unknown tool: {args.tool}")
            console.print(f"Available tools: {available_tools}")
            sys.exit(1)
        selected_tools = [args.tool]

    # Extract local path if provided
    local_path = None
    if hasattr(args, "local") and args.local:
        local_path = pathlib.Path(args.local).expanduser().resolve()

    for tool in selected_tools:
        _run_setup_for_tool(tool, console, local_path, ctx)
