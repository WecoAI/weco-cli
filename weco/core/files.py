"""File I/O helpers."""

import json
import pathlib
import shutil
from typing import Any, Union

from .constants import SUPPORTED_FILE_EXTENSIONS


def read_from_path(fp: pathlib.Path, is_json: bool = False) -> Union[str, dict[str, Any]]:
    """Read content from a file path, optionally parsing as JSON."""
    with fp.open("r", encoding="utf-8") as f:
        if is_json:
            return json.load(f)
        return f.read()


def write_to_path(fp: pathlib.Path, content: Union[str, dict[str, Any]], is_json: bool = False, mkdir: bool = False) -> None:
    """Write content to a file path, optionally as JSON."""
    if mkdir:
        fp.parent.mkdir(parents=True, exist_ok=True)

    with fp.open("w", encoding="utf-8") as f:
        if is_json:
            json.dump(content, f, indent=4)
        elif isinstance(content, str):
            f.write(content)
        else:
            raise TypeError("Error writing to file. Please verify the file path and try again.")


def copy_file(src: pathlib.Path, dest: pathlib.Path, mkdir: bool = False) -> None:
    """Copy a single file."""
    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")
    if mkdir:
        dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dest)


def copy_directory(src: pathlib.Path, dest: pathlib.Path, ignore_patterns: set[str] | None = None) -> None:
    """Copy a directory tree, optionally ignoring certain names."""
    if not src.exists():
        raise FileNotFoundError(f"Source directory not found: {src}")

    def ignore_func(_: str, files: list[str]) -> list[str]:
        if not ignore_patterns:
            return []
        return [f for f in files if f in ignore_patterns]

    shutil.copytree(src, dest, ignore=ignore_func)


def safe_remove_directory(path: pathlib.Path, allowed_parents: set[pathlib.Path]) -> None:
    """Safely remove a directory with multiple defensive checks.

    This function is intentionally paranoid to prevent accidental deletion of
    important directories due to bugs, path traversal, or misconfiguration.

    Args:
        path: The directory to remove.
        allowed_parents: The path must be a direct child of one of these directories.

    Raises:
        SafetyError: If any safety check fails.
    """
    resolved_path = path.resolve()

    if not resolved_path.exists():
        return

    if not resolved_path.is_dir():
        raise SafetyError(f"Refusing to remove: not a directory: {resolved_path}")

    if resolved_path.is_symlink():
        raise SafetyError(f"Refusing to remove: path is a symlink: {resolved_path}")

    home = pathlib.Path.home().resolve()
    if resolved_path == home:
        raise SafetyError(f"Refusing to remove: path is home directory: {resolved_path}")
    if resolved_path == pathlib.Path("/").resolve():
        raise SafetyError(f"Refusing to remove: path is root directory: {resolved_path}")

    try:
        home.relative_to(resolved_path)
        raise SafetyError(f"Refusing to remove: path is a parent of home directory: {resolved_path}")
    except ValueError:
        pass

    resolved_allowed = {p.resolve() for p in allowed_parents}
    parent = resolved_path.parent

    if parent not in resolved_allowed:
        raise SafetyError(
            f"Refusing to remove: path {resolved_path} is not a direct child of allowed directories: "
            f"{[str(p) for p in resolved_allowed]}"
        )

    try:
        relative = resolved_path.relative_to(parent)
        if len(relative.parts) != 1:
            raise SafetyError(f"Refusing to remove: path {resolved_path} is not exactly 1 level below parent {parent}")
    except ValueError:
        raise SafetyError(f"Refusing to remove: path {resolved_path} is not relative to any allowed parent")

    if resolved_path.name != "weco":
        raise SafetyError(f"Refusing to remove: directory name is not 'weco': {resolved_path}")

    shutil.rmtree(resolved_path)


class SafetyError(Exception):
    """Raised when a safety check fails during directory operations."""

    pass


def read_additional_instructions(additional_instructions: str | None) -> str | None:
    """Read additional instructions from a file path or return the string as-is."""
    if additional_instructions is None:
        return None

    potential_path = pathlib.Path(additional_instructions)
    try:
        if potential_path.exists() and potential_path.is_file():
            if potential_path.suffix.lower() not in SUPPORTED_FILE_EXTENSIONS:
                raise ValueError(
                    f"Unsupported file extension: {potential_path.suffix.lower()}. "
                    f"Supported extensions are: {', '.join(SUPPORTED_FILE_EXTENSIONS)}"
                )
            return read_from_path(potential_path, is_json=False)
        else:
            return additional_instructions
    except OSError:
        return additional_instructions
