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
