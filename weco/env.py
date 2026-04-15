# weco/env.py
"""High-level interface to the weco CLI environment.

Provides a single object that encapsulates version info, authentication
state, installed skills, event context, and update checking.

Usage::

    env = WecoEnv(via_skill=args.via_skill)
    env.check_for_updates()

    if env.is_authenticated:
        ...

    send_event(SomeEvent(), env.event_context)
"""

import pathlib
import time
from dataclasses import dataclass

import requests
from packaging.version import parse as parse_version

from . import __pkg_version__, __base_url__, __dashboard_url__
from .config import load_weco_api_key
from .events import EventContext, create_event_context, set_event_context
from .commands.setup.targets import SETUP_TARGETS


_UNSET = object()

# Update checking
_CLI_UPDATE_PYPI_URL = "https://pypi.org/pypi/weco/json"


@dataclass(frozen=True)
class InstalledSkill:
    """A locally installed weco skill."""

    tool: str
    path: pathlib.Path
    version: str  # Installed skill version ("" if unknown)


class WecoEnv:
    """High-level interface to the weco CLI environment.

    Encapsulates version info, authentication state, installed skills,
    event context, and update checking behind a single object that CLI
    commands can depend on.
    """

    # Read once at class-load time — these never change within a process.
    version: str = __pkg_version__
    base_url: str = __base_url__
    dashboard_url: str = __dashboard_url__

    def __init__(self, via_skill: bool = False):
        self._via_skill = via_skill
        # _UNSET distinguishes "not yet loaded" from None ("no key configured").
        self._api_key = _UNSET
        self._event_ctx: EventContext | None = None

    # ── Authentication ──────────────────────────────────────────

    @property
    def api_key(self) -> str | None:
        """Weco API key (loaded lazily from env var or credentials file)."""
        if self._api_key is _UNSET:
            self._api_key = load_weco_api_key()
        return self._api_key

    @property
    def is_authenticated(self) -> bool:
        """Whether a valid API key is available."""
        return self.api_key is not None

    @property
    def auth_headers(self) -> dict[str, str]:
        """Authorization headers for API requests (empty if not authenticated)."""
        if self.api_key:
            return {"Authorization": f"Bearer {self.api_key}"}
        return {}

    def clear_cached_api_key(self) -> None:
        """Reset the cached API key so it will be re-loaded on next access."""
        self._api_key = _UNSET

    # ── Installed Skills ────────────────────────────────────────

    @property
    def installed_skills(self) -> list[InstalledSkill]:
        """Discover locally installed weco skills."""
        skills = []
        for target in SETUP_TARGETS:
            if target.install_dir.exists():
                version = ""
                try:
                    version = (target.install_dir / "VERSION").read_text().strip()
                except (OSError, FileNotFoundError):
                    pass
                skills.append(InstalledSkill(tool=target.name, path=target.install_dir, version=version))
        return skills

    # ── Event Context ───────────────────────────────────────────

    @property
    def event_context(self) -> EventContext:
        """Event context for this CLI invocation (created lazily).

        Also sets the module-level global so that code using
        ``get_event_context()`` continues to work.
        """
        if self._event_ctx is None:
            self._event_ctx = create_event_context(via_skill=self._via_skill)
            set_event_context(self._event_ctx)
        return self._event_ctx

    # ── Update Checking ─────────────────────────────────────────

    def check_for_updates(self) -> None:
        """Check for CLI package and skill updates.

        Prints a yellow warning and pauses briefly for each available
        update.  Fails silently — never disrupts the user.
        """
        self._check_cli_updates()
        self._check_skill_updates()

    def _check_cli_updates(self) -> None:
        """Check PyPI for a newer CLI version."""
        try:
            response = requests.get(_CLI_UPDATE_PYPI_URL, timeout=5)
            response.raise_for_status()
            latest_version_str = response.json()["info"]["version"]

            if parse_version(latest_version_str) > parse_version(self.version):
                _print_update_warning(
                    f"New Weco CLI version ({latest_version_str}) available (you have {self.version}). Run: pipx upgrade weco"
                )
        except Exception:
            pass

    def _check_skill_updates(self) -> None:
        """Check the Weco API for a newer skill version."""
        try:
            if not self.installed_skills:
                return

            skill = self.installed_skills[0]

            response = requests.get(f"{self.base_url}/version", timeout=5)
            if response.status_code != 200:
                return

            latest_version = response.json().get("latest_skill_version", "")
            if not latest_version:
                return

            # If installed skill has no version (pre-VERSION file install), always prompt to update
            if not skill.version:
                commands = ", ".join(f"weco setup {s.tool}" for s in self.installed_skills)
                _print_update_warning(f"New weco skill version ({latest_version}) available. Run: {commands}")
                return

            if parse_version(latest_version) > parse_version(skill.version):
                commands = ", ".join(f"weco setup {s.tool}" for s in self.installed_skills)
                _print_update_warning(
                    f"New weco skill version ({latest_version}) available (you have {skill.version}). Run: {commands}"
                )
        except Exception:
            pass


def _print_update_warning(message: str) -> None:
    """Print a yellow warning and pause briefly so the user sees it."""
    print(f"\033[93m{message}\033[0m")
    time.sleep(2)
