"""Input validation for the Weco CLI.

Provides early validation of user inputs with helpful, actionable error messages.
Validation happens before expensive operations (auth, API calls) to fail fast.
"""

import pathlib
from difflib import get_close_matches

from rich.console import Console


class ValidationError(Exception):
    """Raised when user input validation fails."""

    def __init__(self, message: str, suggestion: str | None = None):
        self.message = message
        self.suggestion = suggestion
        super().__init__(message)


def validate_source_file(source: str) -> None:
    """
    Validate that the source file exists and is readable.

    Args:
        source: Path to the source file.

    Raises:
        ValidationError: If the file doesn't exist, isn't readable, or isn't a valid text file.
    """
    path = pathlib.Path(source)

    if not path.exists():
        suggestion = _find_similar_files(path)
        raise ValidationError(f"Source file '{source}' not found.", suggestion=suggestion)

    if path.is_dir():
        raise ValidationError(
            f"'{source}' is a directory, not a file.", suggestion="Please specify a file path, e.g., 'src/model.py'"
        )

    # Try reading the file to catch permission and encoding issues early
    try:
        path.read_text(encoding="utf-8")
    except PermissionError:
        raise ValidationError(f"Cannot read '{source}' — permission denied.")
    except UnicodeDecodeError:
        raise ValidationError(
            f"'{source}' doesn't appear to be a valid text file.",
            suggestion="Weco optimizes source code files (e.g., .py, .cu, .rs)",
        )
    except OSError as e:
        raise ValidationError(f"Cannot read '{source}': {e}")


def validate_log_directory(log_dir: str) -> None:
    """
    Validate that the log directory is writable.

    Args:
        log_dir: Path to the log directory.

    Raises:
        ValidationError: If the directory can't be created or isn't writable.
    """
    path = pathlib.Path(log_dir)

    try:
        # Attempt to create the directory (no-op if exists)
        path.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        raise ValidationError(f"Cannot create log directory '{log_dir}' — permission denied.")
    except OSError as e:
        raise ValidationError(f"Cannot create log directory '{log_dir}': {e}")

    # Check if writable by attempting to create a temp file
    test_file = path / ".weco_write_test"
    try:
        test_file.touch()
        test_file.unlink()
    except PermissionError:
        raise ValidationError(f"Log directory '{log_dir}' is not writable.")
    except OSError:
        pass  # Directory exists and is likely fine


def _find_similar_files(path: pathlib.Path) -> str | None:
    """Find similar filenames in the same directory to suggest as alternatives."""
    parent = path.parent if path.parent.exists() else pathlib.Path(".")

    try:
        # Get files with the same extension, or all files if no extension
        if path.suffix:
            candidates = [f.name for f in parent.iterdir() if f.is_file() and f.suffix == path.suffix]
        else:
            candidates = [f.name for f in parent.iterdir() if f.is_file()]

        matches = get_close_matches(path.name, candidates, n=3, cutoff=0.4)
        if matches:
            return f"Did you mean: {', '.join(matches)}?"
    except OSError:
        pass

    return None


def print_validation_error(error: ValidationError, console: Console) -> None:
    """Print a validation error in a user-friendly format."""
    console.print(f"[bold red]Error:[/] {error.message}")
    if error.suggestion:
        console.print(f"[dim]{error.suggestion}[/]")
