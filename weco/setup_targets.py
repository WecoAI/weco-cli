"""Shared setup target definitions for AI tool integrations."""

from dataclasses import dataclass
import pathlib


@dataclass(frozen=True)
class SetupTarget:
    """Metadata for a supported ``weco setup`` target."""

    name: str
    label: str
    help_text: str
    install_dir: pathlib.Path

    @property
    def install_parent(self) -> pathlib.Path:
        """Return the parent directory that contains the installed skill."""
        return self.install_dir.parent


SETUP_TARGETS = (
    SetupTarget(
        name="claude-code",
        label="Claude Code",
        help_text="Set up Weco skill for Claude Code",
        install_dir=pathlib.Path.home() / ".claude" / "skills" / "weco",
    ),
    SetupTarget(
        name="cursor",
        label="Cursor",
        help_text="Set up Weco skill for Cursor",
        install_dir=pathlib.Path.home() / ".cursor" / "skills" / "weco",
    ),
    SetupTarget(
        name="codex",
        label="Codex",
        help_text="Set up Weco skill for Codex",
        install_dir=pathlib.Path.home() / ".codex" / "skills" / "weco",
    ),
    SetupTarget(
        name="openclaw",
        label="OpenClaw",
        help_text="Set up Weco skill for OpenClaw",
        install_dir=pathlib.Path.home() / ".openclaw" / "skills" / "weco",
    ),
)

SETUP_TARGET_BY_NAME = {target.name: target for target in SETUP_TARGETS}
SETUP_TARGET_NAMES = tuple(target.name for target in SETUP_TARGETS)

ALL_SETUP_OPTION_NAME = "all"
ALL_SETUP_OPTION_LABEL = "All of the above"
