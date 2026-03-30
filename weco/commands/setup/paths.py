"""Filesystem paths for skill installation targets."""

import pathlib

# Skill repository
WECO_SKILL_REPO_URL = "https://github.com/WecoAI/weco-skill"
WECO_SKILL_BRANCH = "main"

# Claude Code paths
CLAUDE_DIR = pathlib.Path.home() / ".claude"
CLAUDE_SKILLS_DIR = CLAUDE_DIR / "skills"
WECO_SKILL_DIR = CLAUDE_SKILLS_DIR / "weco"
WECO_CLAUDE_SNIPPET_PATH = WECO_SKILL_DIR / "snippets" / "claude.md"
WECO_CLAUDE_MD_PATH = WECO_SKILL_DIR / "CLAUDE.md"

# Cursor paths
CURSOR_DIR = pathlib.Path.home() / ".cursor"
CURSOR_SKILLS_DIR = CURSOR_DIR / "skills"
CURSOR_WECO_SKILL_DIR = CURSOR_SKILLS_DIR / "weco"

# Allowed parent directories for safe removal (defense in depth)
ALLOWED_SKILL_PARENTS = {CLAUDE_SKILLS_DIR, CURSOR_SKILLS_DIR}

# Files/directories to skip when copying local repos
COPY_IGNORE_PATTERNS = {".git", "__pycache__", ".DS_Store"}
