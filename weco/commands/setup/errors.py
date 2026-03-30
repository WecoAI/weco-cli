"""Exceptions raised during skill installation."""


class SetupError(Exception):
    """Base exception for setup failures."""


class InvalidLocalRepoError(SetupError):
    """Raised when a local path is not a valid skill repository."""


class DownloadError(SetupError):
    """Raised when downloading the skill fails."""
