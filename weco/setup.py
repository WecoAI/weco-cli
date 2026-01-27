# weco/setup.py
"""
Setup commands for integrating Weco with various AI tools.
"""

import pathlib
import shutil
import subprocess
from rich.console import Console
from rich.prompt import Confirm

# Claude Code paths
CLAUDE_DIR = pathlib.Path.home() / ".claude"
CLAUDE_SKILLS_DIR = CLAUDE_DIR / "skills"
CLAUDE_MD_PATH = CLAUDE_DIR / "CLAUDE.md"
WECO_SKILL_DIR = CLAUDE_SKILLS_DIR / "weco"
WECO_SKILL_REPO = "git@github.com:WecoAI/weco-skill.git"
WECO_RULES_SNIPPET_PATH = WECO_SKILL_DIR / "rules-snippet.md"

# Delimiters for agent rules files - allows automatic updates
WECO_RULES_BEGIN_DELIMITER = "<!-- BEGIN WECO_RULES -->"
WECO_RULES_END_DELIMITER = "<!-- END WECO_RULES -->"


def is_git_available() -> bool:
    """Check if git is available on the system."""
    return shutil.which("git") is not None


def read_rules_snippet(console: Console) -> str | None:
    """
    Read the rules snippet from the cloned skill repository.

    Returns:
        The snippet content wrapped in delimiters, or None if not found.
    """
    if not WECO_RULES_SNIPPET_PATH.exists():
        console.print(f"[bold red]Error:[/] Snippet file not found at {WECO_RULES_SNIPPET_PATH}")
        return None

    try:
        snippet_content = WECO_RULES_SNIPPET_PATH.read_text().strip()
        return f"\n{WECO_RULES_BEGIN_DELIMITER}\n{snippet_content}\n{WECO_RULES_END_DELIMITER}\n"
    except Exception as e:
        console.print(f"[bold red]Error:[/] Failed to read snippet file: {e}")
        return None


def is_git_repo(path: pathlib.Path) -> bool:
    """Check if a directory is a git repository."""
    return (path / ".git").is_dir()


def clone_skill_repo(console: Console) -> bool:
    """
    Clone or update the weco-skill repository.

    Returns:
        True if successful, False otherwise.
    """
    if not is_git_available():
        console.print("[bold red]Error:[/] git is not installed or not in PATH.")
        console.print("Please install git and try again.")
        return False

    # Ensure the skills directory exists
    CLAUDE_SKILLS_DIR.mkdir(parents=True, exist_ok=True)

    if WECO_SKILL_DIR.exists():
        if is_git_repo(WECO_SKILL_DIR):
            # Directory exists and is a git repo - pull latest
            console.print(f"[cyan]Updating existing skill at {WECO_SKILL_DIR}...[/]")
            try:
                result = subprocess.run(["git", "pull"], cwd=WECO_SKILL_DIR, capture_output=True, text=True)
                if result.returncode != 0:
                    console.print("[bold red]Error:[/] Failed to update skill repository.")
                    console.print(f"[dim]{result.stderr}[/]")
                    return False
                console.print("[green]Skill updated successfully.[/]")
                return True
            except Exception as e:
                console.print(f"[bold red]Error:[/] Failed to update skill repository: {e}")
                return False
        else:
            # Directory exists but is not a git repo
            console.print(f"[bold red]Error:[/] Directory {WECO_SKILL_DIR} exists but is not a git repository.")
            console.print("Please remove it manually and try again.")
            return False
    else:
        # Clone the repository
        console.print(f"[cyan]Cloning Weco skill to {WECO_SKILL_DIR}...[/]")
        try:
            result = subprocess.run(["git", "clone", WECO_SKILL_REPO, str(WECO_SKILL_DIR)], capture_output=True, text=True)
            if result.returncode != 0:
                console.print("[bold red]Error:[/] Failed to clone skill repository.")
                console.print(f"[dim]{result.stderr}[/]")
                return False
            console.print("[green]Skill cloned successfully.[/]")
            return True
        except Exception as e:
            console.print(f"[bold red]Error:[/] Failed to clone skill repository: {e}")
            return False


def update_agent_rules_file(rules_file: pathlib.Path, console: Console) -> bool:
    """
    Update an agent's rules file with the Weco skill reference.

    Uses delimiters to allow automatic updates if the snippet changes.

    Args:
        rules_file: Path to the agent's rules file (e.g., ~/.claude/CLAUDE.md)
        console: Rich console for output.

    Returns:
        True if updated or user declined, False on error.
    """
    import re

    rules_file_name = rules_file.name

    # Read the snippet from the cloned skill repo
    snippet_section = read_rules_snippet(console)
    if snippet_section is None:
        return False

    # Check if the section already exists with delimiters
    existing_content = ""
    has_existing_section = False
    if rules_file.exists():
        try:
            existing_content = rules_file.read_text()
            has_existing_section = WECO_RULES_BEGIN_DELIMITER in existing_content and WECO_RULES_END_DELIMITER in existing_content
        except Exception as e:
            console.print(f"[bold yellow]Warning:[/] Could not read {rules_file_name}: {e}")

    # Determine what action to take
    if has_existing_section:
        # Check if content is already up to date
        pattern = re.escape(WECO_RULES_BEGIN_DELIMITER) + r".*?" + re.escape(WECO_RULES_END_DELIMITER)
        match = re.search(pattern, existing_content, re.DOTALL)
        if match and match.group(0).strip() == snippet_section.strip():
            console.print(f"[dim]{rules_file_name} already contains the latest Weco rules.[/]")
            return True

        # Prompt for update
        console.print(f"\n[bold yellow]{rules_file_name} Update[/]")
        console.print(f"The Weco rules in your {rules_file_name} can be updated to the latest version.")
        should_update = Confirm.ask("Would you like to update the Weco section?", default=True)
    elif rules_file.exists():
        console.print(f"\n[bold yellow]{rules_file_name} Update[/]")
        console.print(f"To enable automatic skill discovery, we can add Weco rules to your {rules_file_name} file.")
        should_update = Confirm.ask(
            f"Would you like to update your {rules_file_name}?", default=True
        )
    else:
        console.print(f"\n[bold yellow]{rules_file_name} Creation[/]")
        console.print(f"To enable automatic skill discovery, we can create a {rules_file_name} file.")
        should_update = Confirm.ask(f"Would you like to create {rules_file_name}?", default=True)

    if not should_update:
        console.print(f"\n[yellow]Skipping {rules_file_name} update.[/]")
        console.print(
            "[dim]The Weco skill has been installed but may not be discovered automatically.\n"
            f"You can manually reference it at {WECO_SKILL_DIR}[/]"
        )
        return True

    # Update or create the file
    try:
        rules_file.parent.mkdir(parents=True, exist_ok=True)

        if has_existing_section:
            # Replace existing section between delimiters
            pattern = re.escape(WECO_RULES_BEGIN_DELIMITER) + r".*?" + re.escape(WECO_RULES_END_DELIMITER)
            new_content = re.sub(pattern, snippet_section.strip(), existing_content, flags=re.DOTALL)
            rules_file.write_text(new_content)
            console.print(f"[green]{rules_file_name} updated successfully.[/]")
        elif rules_file.exists():
            # Append to existing file
            with open(rules_file, "a") as f:
                f.write(snippet_section)
            console.print(f"[green]{rules_file_name} updated successfully.[/]")
        else:
            # Create new file
            with open(rules_file, "w") as f:
                f.write(snippet_section.lstrip())
            console.print(f"[green]{rules_file_name} created successfully.[/]")
        return True
    except Exception as e:
        console.print(f"[bold red]Error:[/] Failed to update {rules_file_name}: {e}")
        return False


def setup_claude_code(console: Console) -> bool:
    """
    Set up Weco skill for Claude Code.

    Returns:
        True if setup was successful, False otherwise.
    """
    console.print("[bold blue]Setting up Weco for Claude Code...[/]\n")

    # Step 1: Clone or update the skill repository
    if not clone_skill_repo(console):
        return False

    # Step 2: Update CLAUDE.md
    if not update_agent_rules_file(CLAUDE_MD_PATH, console):
        return False

    console.print("\n[bold green]Setup complete![/]")
    console.print(f"[dim]Skill installed at: {WECO_SKILL_DIR}[/]")
    return True


def handle_setup_command(args, console: Console) -> None:
    """Handle the setup command with its subcommands."""
    if args.tool == "claude-code":
        success = setup_claude_code(console)
        if not success:
            import sys

            sys.exit(1)
    elif args.tool is None:
        console.print("[bold red]Error:[/] Please specify a tool to set up.")
        console.print("Available tools: claude-code")
        console.print("\nUsage: weco setup claude-code")
        import sys

        sys.exit(1)
    else:
        console.print(f"[bold red]Error:[/] Unknown tool: {args.tool}")
        console.print("Available tools: claude-code")
        import sys

        sys.exit(1)
