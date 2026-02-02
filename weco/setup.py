# weco/setup.py
"""
Setup commands for integrating Weco with various AI tools.
"""

import pathlib
import sys

from rich.console import Console
from rich.prompt import Confirm

from . import git
from .utils import copy_directory, copy_file, read_from_path, remove_directory, write_to_path


# =============================================================================
# Exceptions
# =============================================================================


class SetupError(Exception):
    """Base exception for setup failures."""

    pass


class InvalidLocalRepoError(SetupError):
    """Raised when a local path is not a valid skill repository."""

    pass


# =============================================================================
# Path constants
# =============================================================================

# Claude Code paths
CLAUDE_DIR = pathlib.Path.home() / ".claude"
CLAUDE_SKILLS_DIR = CLAUDE_DIR / "skills"
WECO_SKILL_DIR = CLAUDE_SKILLS_DIR / "weco"
WECO_SKILL_REPO = "git@github.com:WecoAI/weco-skill.git"
WECO_CLAUDE_SNIPPET_PATH = WECO_SKILL_DIR / "snippets" / "claude.md"
WECO_CLAUDE_MD_PATH = WECO_SKILL_DIR / "CLAUDE.md"

# Cursor paths
CURSOR_DIR = pathlib.Path.home() / ".cursor"
CURSOR_RULES_DIR = CURSOR_DIR / "rules"
CURSOR_WECO_RULES_PATH = CURSOR_RULES_DIR / "weco.mdc"
CURSOR_SKILLS_DIR = CURSOR_DIR / "skills"
CURSOR_WECO_SKILL_DIR = CURSOR_SKILLS_DIR / "weco"
CURSOR_RULES_SNIPPET_PATH = CURSOR_WECO_SKILL_DIR / "snippets" / "cursor.md"

# Files/directories to skip when copying local repos
_COPY_IGNORE_PATTERNS = {".git", "__pycache__", ".DS_Store"}


# =============================================================================
# Domain helpers
# =============================================================================


def generate_cursor_mdc_content(snippet_content: str) -> str:
    """Generate Cursor MDC file content with YAML frontmatter."""
    return f"""---
description: Weco code optimization skill. Weco automates optimization by iteratively refining code against any metric you define — invoke for speed, accuracy, latency, cost, or anything else you can measure.
alwaysApply: true
---
{snippet_content}
"""


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
# Installation functions
# =============================================================================


def install_skill_from_git(skill_dir: pathlib.Path, console: Console, repo_url: str | None, ref: str | None) -> None:
    """
    Clone or update skill from git.

    Raises:
        git.GitNotFoundError: If git is not available.
        git.GitError: If git operations fail.
    """
    if not git.is_available():
        raise git.GitNotFoundError("git is not installed or not in PATH")

    skill_dir.parent.mkdir(parents=True, exist_ok=True)

    if skill_dir.exists():
        if git.is_repo(skill_dir):
            console.print(f"[cyan]Updating existing skill at {skill_dir}...[/]")
            if ref:
                git.fetch_and_checkout(skill_dir, ref)
                console.print(f"[green]Checked out {ref}.[/]")
            else:
                git.pull(skill_dir)
                console.print("[green]Skill updated successfully.[/]")
            return

        # Not a git repo — clear and re-clone
        console.print(f"[cyan]Removing existing directory at {skill_dir}...[/]")
        remove_directory(skill_dir)

    console.print(f"[cyan]Cloning Weco skill to {skill_dir}...[/]")
    git.clone(repo_url or WECO_SKILL_REPO, skill_dir, ref=ref)
    if ref:
        console.print(f"[green]Cloned and checked out {ref}.[/]")
    else:
        console.print("[green]Skill cloned successfully.[/]")


def install_skill_from_local(skill_dir: pathlib.Path, console: Console, local_path: pathlib.Path) -> None:
    """
    Copy skill from local path.

    Raises:
        InvalidLocalRepoError: If local path is invalid.
        OSError: If copy fails.
    """
    validate_local_skill_repo(local_path)

    skill_dir.parent.mkdir(parents=True, exist_ok=True)

    if skill_dir.exists():
        console.print(f"[cyan]Removing existing directory at {skill_dir}...[/]")
        remove_directory(skill_dir)

    copy_directory(local_path, skill_dir, ignore_patterns=_COPY_IGNORE_PATTERNS)
    console.print(f"[green]Copied local repo from: {local_path}[/]")


def install_skill(
    skill_dir: pathlib.Path, console: Console, local_path: pathlib.Path | None, repo_url: str | None, ref: str | None
) -> None:
    """Install skill by copying from local path or cloning from git."""
    if local_path:
        if ref:
            console.print("[bold yellow]Warning:[/] --ref is ignored when using --local")
        install_skill_from_local(skill_dir, console, local_path)
    else:
        install_skill_from_git(skill_dir, console, repo_url, ref)


# =============================================================================
# Setup commands
# =============================================================================


def setup_claude_code(
    console: Console, local_path: pathlib.Path | None = None, repo_url: str | None = None, ref: str | None = None
) -> None:
    """Set up Weco skill for Claude Code."""
    console.print("[bold blue]Setting up Weco for Claude Code...[/]\n")

    # Claude Code setup is intentionally "skill-centric":
    # - Install the skill into Claude's skills directory.
    # - `CLAUDE.md` lives *inside* that installed skill folder, so configuration is just
    #   copying a file within the skill directory.
    # - There is no separate global config file to reconcile or prompt before overwriting.
    install_skill(WECO_SKILL_DIR, console, local_path, repo_url, ref)

    # Copy snippets/claude.md to CLAUDE.md (skip for local - user manages their own)
    if not local_path:
        copy_file(WECO_CLAUDE_SNIPPET_PATH, WECO_CLAUDE_MD_PATH)
        console.print("[green]CLAUDE.md installed to skill directory.[/]")

    console.print("\n[bold green]Setup complete![/]")
    if local_path:
        console.print(f"[dim]Skill copied from: {local_path}[/]")
    console.print(f"[dim]Skill installed at: {WECO_SKILL_DIR}[/]")


def setup_cursor(
    console: Console, local_path: pathlib.Path | None = None, repo_url: str | None = None, ref: str | None = None
) -> None:
    """Set up Weco rules for Cursor."""
    console.print("[bold blue]Setting up Weco for Cursor...[/]\n")

    # Cursor setup is intentionally "editor-config-centric":
    # - Install/copy the skill into Cursor's skills directory (so we can read snippets).
    # - The behavior change is controlled by `~/.cursor/rules/weco.mdc`, which is *global*
    #   editor state (not part of the installed skill folder).
    # - Because users may have customized that file, we:
    #   1) compute desired content from the snippet
    #   2) check if it is already up to date
    #   3) prompt before creating/updating it
    install_skill(CURSOR_WECO_SKILL_DIR, console, local_path, repo_url, ref)

    snippet_content = read_from_path(CURSOR_RULES_SNIPPET_PATH)
    mdc_content = generate_cursor_mdc_content(snippet_content.strip())

    # Check if already up to date
    existing_content = None
    if CURSOR_WECO_RULES_PATH.exists():
        try:
            existing_content = read_from_path(CURSOR_WECO_RULES_PATH)
        except Exception:
            pass

    if existing_content is not None and existing_content.strip() == mdc_content.strip():
        console.print("[dim]weco.mdc already contains the latest Weco rules.[/]")
        console.print("\n[bold green]Setup complete![/]")
        console.print(f"[dim]Rules file at: {CURSOR_WECO_RULES_PATH}[/]")
        return

    # Prompt user for creation/update
    if existing_content is not None:
        console.print("\n[bold yellow]weco.mdc Update[/]")
        console.print("The Weco rules file can be updated to the latest version.")
        if not Confirm.ask("Would you like to update weco.mdc?", default=True):
            console.print("\n[yellow]Skipping weco.mdc update.[/]")
            console.print(f"[dim]Skill installed but rules not configured. Create manually at {CURSOR_WECO_RULES_PATH}[/]")
            return
    else:
        console.print("\n[bold yellow]weco.mdc Creation[/]")
        console.print("To enable Weco optimization rules, we can create a weco.mdc file.")
        if not Confirm.ask("Would you like to create weco.mdc?", default=True):
            console.print("\n[yellow]Skipping weco.mdc creation.[/]")
            console.print(f"[dim]Skill installed but rules not configured. Create manually at {CURSOR_WECO_RULES_PATH}[/]")
            return

    write_to_path(CURSOR_WECO_RULES_PATH, mdc_content, mkdir=True)
    console.print("[green]weco.mdc created successfully.[/]")

    console.print("\n[bold green]Setup complete![/]")
    if local_path:
        console.print(f"[dim]Skill copied from: {local_path}[/]")
    console.print(f"[dim]Skill installed at: {CURSOR_WECO_SKILL_DIR}[/]")
    console.print(f"[dim]Rules file at: {CURSOR_WECO_RULES_PATH}[/]")


# =============================================================================
# CLI entry point
# =============================================================================

SETUP_HANDLERS = {"claude-code": setup_claude_code, "cursor": setup_cursor}


def handle_setup_command(args, console: Console) -> None:
    """Handle the setup command with its subcommands."""
    available_tools = ", ".join(SETUP_HANDLERS)

    if args.tool is None:
        console.print("[bold red]Error:[/] Please specify a tool to set up.")
        console.print(f"Available tools: {available_tools}")
        console.print("\nUsage: weco setup <tool>")
        sys.exit(1)

    handler = SETUP_HANDLERS.get(args.tool)
    if handler is None:
        console.print(f"[bold red]Error:[/] Unknown tool: {args.tool}")
        console.print(f"Available tools: {available_tools}")
        sys.exit(1)

    # Validate and extract args
    repo_url = getattr(args, "repo", None)
    ref = getattr(args, "ref", None)
    local_path = None
    if hasattr(args, "local") and args.local:
        local_path = pathlib.Path(args.local).expanduser().resolve()

    try:
        if repo_url:
            git.validate_repo_url(repo_url)
        if ref:
            git.validate_ref(ref)

        handler(console, local_path=local_path, repo_url=repo_url, ref=ref)

    except git.GitError as e:
        console.print(f"[bold red]Error:[/] {e}")
        if e.stderr:
            console.print(f"[dim]{e.stderr}[/]")
        sys.exit(1)
    except (SetupError, git.GitNotFoundError, FileNotFoundError, OSError, ValueError) as e:
        console.print(f"[bold red]Error:[/] {e}")
        sys.exit(1)
