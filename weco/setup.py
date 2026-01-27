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

# Cursor paths
CURSOR_DIR = pathlib.Path.home() / ".cursor"
CURSOR_RULES_DIR = CURSOR_DIR / "rules"
CURSOR_WECO_RULES_PATH = CURSOR_RULES_DIR / "weco.mdc"
CURSOR_SKILLS_DIR = CURSOR_DIR / "skills"
CURSOR_WECO_SKILL_DIR = CURSOR_SKILLS_DIR / "weco"
CURSOR_RULES_SNIPPET_PATH = CURSOR_WECO_SKILL_DIR / "rules-snippet.md"

# Delimiters for agent rules files - allows automatic updates
WECO_RULES_BEGIN_DELIMITER = "<!-- BEGIN WECO_RULES -->"
WECO_RULES_END_DELIMITER = "<!-- END WECO_RULES -->"


def is_git_available() -> bool:
    """Check if git is available on the system."""
    return shutil.which("git") is not None


def read_rules_snippet(snippet_path: pathlib.Path, console: Console) -> str | None:
    """
    Read the rules snippet from the cloned skill repository.

    Args:
        snippet_path: Path to the rules-snippet.md file.
        console: Rich console for output.

    Returns:
        The snippet content wrapped in delimiters, or None if not found.
    """
    if not snippet_path.exists():
        console.print(f"[bold red]Error:[/] Snippet file not found at {snippet_path}")
        return None

    try:
        snippet_content = snippet_path.read_text().strip()
        return f"\n{WECO_RULES_BEGIN_DELIMITER}\n{snippet_content}\n{WECO_RULES_END_DELIMITER}\n"
    except Exception as e:
        console.print(f"[bold red]Error:[/] Failed to read snippet file: {e}")
        return None


def read_rules_snippet_raw(snippet_path: pathlib.Path, console: Console) -> str | None:
    """
    Read the raw rules snippet from the cloned skill repository (without delimiters).

    Args:
        snippet_path: Path to the rules-snippet.md file.
        console: Rich console for output.

    Returns:
        The raw snippet content, or None if not found.
    """
    if not snippet_path.exists():
        console.print(f"[bold red]Error:[/] Snippet file not found at {snippet_path}")
        return None

    try:
        return snippet_path.read_text().strip()
    except Exception as e:
        console.print(f"[bold red]Error:[/] Failed to read snippet file: {e}")
        return None


def generate_cursor_mdc_content(snippet_content: str) -> str:
    """
    Generate Cursor MDC file content with YAML frontmatter.

    Args:
        snippet_content: The raw rules snippet content.

    Returns:
        MDC formatted content with frontmatter.
    """
    return f"""---
description: Weco code optimization skill - invoke for speed, accuracy, loss optimization
alwaysApply: true
---
{snippet_content}
"""


def is_git_repo(path: pathlib.Path) -> bool:
    """Check if a directory is a git repository."""
    return (path / ".git").is_dir()


def clone_skill_repo(skill_dir: pathlib.Path, console: Console) -> bool:
    """
    Clone or update the weco-skill repository to the specified directory.

    Args:
        skill_dir: The directory to clone/update the skill repository in.
        console: Rich console for output.

    Returns:
        True if successful, False otherwise.
    """
    if not is_git_available():
        console.print("[bold red]Error:[/] git is not installed or not in PATH.")
        console.print("Please install git and try again.")
        return False

    # Ensure the parent skills directory exists
    skill_dir.parent.mkdir(parents=True, exist_ok=True)

    if skill_dir.exists():
        if is_git_repo(skill_dir):
            # Directory exists and is a git repo - pull latest
            console.print(f"[cyan]Updating existing skill at {skill_dir}...[/]")
            try:
                result = subprocess.run(["git", "pull"], cwd=skill_dir, capture_output=True, text=True)
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
            console.print(f"[bold red]Error:[/] Directory {skill_dir} exists but is not a git repository.")
            console.print("Please remove it manually and try again.")
            return False
    else:
        # Clone the repository
        console.print(f"[cyan]Cloning Weco skill to {skill_dir}...[/]")
        try:
            result = subprocess.run(["git", "clone", WECO_SKILL_REPO, str(skill_dir)], capture_output=True, text=True)
            if result.returncode != 0:
                console.print("[bold red]Error:[/] Failed to clone skill repository.")
                console.print(f"[dim]{result.stderr}[/]")
                return False
            console.print("[green]Skill cloned successfully.[/]")
            return True
        except Exception as e:
            console.print(f"[bold red]Error:[/] Failed to clone skill repository: {e}")
            return False


def update_agent_rules_file(
    rules_file: pathlib.Path, snippet_path: pathlib.Path, skill_dir: pathlib.Path, console: Console
) -> bool:
    """
    Update an agent's rules file with the Weco skill reference.

    Uses delimiters to allow automatic updates if the snippet changes.

    Args:
        rules_file: Path to the agent's rules file (e.g., ~/.claude/CLAUDE.md)
        snippet_path: Path to the rules-snippet.md file.
        skill_dir: Path to the skill directory (for user messaging).
        console: Rich console for output.

    Returns:
        True if updated or user declined, False on error.
    """
    import re

    rules_file_name = rules_file.name

    # Read the snippet from the cloned skill repo
    snippet_section = read_rules_snippet(snippet_path, console)
    if snippet_section is None:
        return False

    # Check if the section already exists with delimiters
    existing_content = ""
    has_existing_section = False
    if rules_file.exists():
        try:
            existing_content = rules_file.read_text()
            has_existing_section = (
                WECO_RULES_BEGIN_DELIMITER in existing_content and WECO_RULES_END_DELIMITER in existing_content
            )
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
        should_update = Confirm.ask(f"Would you like to update your {rules_file_name}?", default=True)
    else:
        console.print(f"\n[bold yellow]{rules_file_name} Creation[/]")
        console.print(f"To enable automatic skill discovery, we can create a {rules_file_name} file.")
        should_update = Confirm.ask(f"Would you like to create {rules_file_name}?", default=True)

    if not should_update:
        console.print(f"\n[yellow]Skipping {rules_file_name} update.[/]")
        console.print(
            "[dim]The Weco skill has been installed but may not be discovered automatically.\n"
            f"You can manually reference it at {skill_dir}[/]"
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
    if not clone_skill_repo(WECO_SKILL_DIR, console):
        return False

    # Step 2: Update CLAUDE.md
    if not update_agent_rules_file(CLAUDE_MD_PATH, WECO_RULES_SNIPPET_PATH, WECO_SKILL_DIR, console):
        return False

    console.print("\n[bold green]Setup complete![/]")
    console.print(f"[dim]Skill installed at: {WECO_SKILL_DIR}[/]")
    return True


def setup_cursor(console: Console) -> bool:
    """
    Set up Weco rules for Cursor.

    Creates a weco.mdc file in ~/.cursor/rules/ with the Weco optimization rules.

    Returns:
        True if setup was successful, False otherwise.
    """
    console.print("[bold blue]Setting up Weco for Cursor...[/]\n")

    # Step 1: Clone or update the skill repository to Cursor's path
    if not clone_skill_repo(CURSOR_WECO_SKILL_DIR, console):
        return False

    # Step 2: Read the rules snippet
    snippet_content = read_rules_snippet_raw(CURSOR_RULES_SNIPPET_PATH, console)
    if snippet_content is None:
        return False

    # Step 3: Check if weco.mdc already exists
    if CURSOR_WECO_RULES_PATH.exists():
        try:
            existing_content = CURSOR_WECO_RULES_PATH.read_text()
            new_content = generate_cursor_mdc_content(snippet_content)
            if existing_content.strip() == new_content.strip():
                console.print("[dim]weco.mdc already contains the latest Weco rules.[/]")
                console.print("\n[bold green]Setup complete![/]")
                console.print(f"[dim]Rules file at: {CURSOR_WECO_RULES_PATH}[/]")
                return True
        except Exception as e:
            console.print(f"[bold yellow]Warning:[/] Could not read existing weco.mdc: {e}")

        console.print("\n[bold yellow]weco.mdc Update[/]")
        console.print("The Weco rules file can be updated to the latest version.")
        should_update = Confirm.ask("Would you like to update weco.mdc?", default=True)
    else:
        console.print("\n[bold yellow]weco.mdc Creation[/]")
        console.print("To enable Weco optimization rules, we can create a weco.mdc file.")
        should_update = Confirm.ask("Would you like to create weco.mdc?", default=True)

    if not should_update:
        console.print("\n[yellow]Skipping weco.mdc update.[/]")
        console.print(
            "[dim]The Weco skill has been installed but rules are not configured.\n"
            f"You can manually create the rules file at {CURSOR_WECO_RULES_PATH}[/]"
        )
        return True

    # Step 4: Write the MDC file
    try:
        CURSOR_RULES_DIR.mkdir(parents=True, exist_ok=True)
        mdc_content = generate_cursor_mdc_content(snippet_content)
        CURSOR_WECO_RULES_PATH.write_text(mdc_content)
        console.print("[green]weco.mdc created successfully.[/]")
    except Exception as e:
        console.print(f"[bold red]Error:[/] Failed to write weco.mdc: {e}")
        return False

    console.print("\n[bold green]Setup complete![/]")
    console.print(f"[dim]Rules file at: {CURSOR_WECO_RULES_PATH}[/]")
    return True


def handle_setup_command(args, console: Console) -> None:
    """Handle the setup command with its subcommands."""
    if args.tool == "claude-code":
        success = setup_claude_code(console)
        if not success:
            import sys

            sys.exit(1)
    elif args.tool == "cursor":
        success = setup_cursor(console)
        if not success:
            import sys

            sys.exit(1)
    elif args.tool is None:
        console.print("[bold red]Error:[/] Please specify a tool to set up.")
        console.print("Available tools: claude-code, cursor")
        console.print("\nUsage: weco setup <tool>")
        import sys

        sys.exit(1)
    else:
        console.print(f"[bold red]Error:[/] Unknown tool: {args.tool}")
        console.print("Available tools: claude-code, cursor")
        import sys

        sys.exit(1)
