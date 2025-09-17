from typing import Any, Dict, List, Tuple, Union
import json
import os
import time
import subprocess
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
import pathlib
import requests
from packaging.version import parse as parse_version

from .constants import TRUNCATION_THRESHOLD, TRUNCATION_KEEP_LENGTH, DEFAULT_MODEL, SUPPORTED_FILE_EXTENSIONS


# Env/arg helper functions
def read_api_keys_from_env() -> Dict[str, Any]:
    """Read API keys from environment variables."""
    keys = {}
    keys_to_check = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"]
    for key in keys_to_check:
        value = os.getenv(key)
        if value is not None and len(value) > 0:
            keys[key] = value
    return keys


def determine_model_for_onboarding() -> str:
    """Determine which model to use for onboarding chatbot. Defaults to o4-mini."""
    return DEFAULT_MODEL


def read_additional_instructions(additional_instructions: str | None) -> str | None:
    """Read additional instructions from a file path string or return the string itself."""
    if additional_instructions is None:
        return None

    # Try interpreting as a path first
    potential_path = pathlib.Path(additional_instructions)
    try:
        if potential_path.exists() and potential_path.is_file():
            # If it's a valid file path, check if we support the file extension
            if potential_path.suffix.lower() not in SUPPORTED_FILE_EXTENSIONS:
                raise ValueError(f"Unsupported file extension: {potential_path.suffix.lower()}. Supported extensions are: {', '.join(SUPPORTED_FILE_EXTENSIONS)}")
            return read_from_path(potential_path, is_json=False)  # type: ignore # read_from_path returns str when is_json=False
        else:
            # If it's not a valid file path, return the string itself
            return additional_instructions
    except OSError:
        # If the path can't be read, return the string itself
        return additional_instructions


# File helper functions
def read_from_path(fp: pathlib.Path, is_json: bool = False) -> Union[str, Dict[str, Any]]:
    """Read content from a file path, optionally parsing as JSON."""
    with fp.open("r", encoding="utf-8") as f:
        if is_json:
            return json.load(f)
        return f.read()


def write_to_path(fp: pathlib.Path, content: Union[str, Dict[str, Any]], is_json: bool = False) -> None:
    """Write content to a file path, optionally as JSON."""
    with fp.open("w", encoding="utf-8") as f:
        if is_json:
            json.dump(content, f, indent=4)
        elif isinstance(content, str):
            f.write(content)
        else:
            raise TypeError("Error writing to file. Please verify the file path and try again.")


# Visualization helper functions
def smooth_update(
    live: Live, layout: Layout, sections_to_update: List[Tuple[str, Panel]], transition_delay: float = 0.05
) -> None:
    """
    Update sections of the layout with a small delay between each update for a smoother transition effect.

    Args:
        live: The Live display instance
        layout: The Layout to update
        sections_to_update: List of (section_name, content) tuples to update
        transition_delay: Delay in seconds between updates (default: 0.05)
    """
    for section, content in sections_to_update:
        layout[section].update(content)
        live.refresh()
        time.sleep(transition_delay)


# Other helper functions
def truncate_output(output: str) -> str:
    """Truncate long output to a manageable size.

    If output exceeds TRUNCATION_THRESHOLD characters, keeps the first
    TRUNCATION_KEEP_LENGTH and last TRUNCATION_KEEP_LENGTH characters
    with a truncation message.

    Args:
        output: The output string to truncate
    """
    # Check if the length of the string is longer than the threshold
    if len(output) > TRUNCATION_THRESHOLD:
        # Output the first TRUNCATION_KEEP_LENGTH and last TRUNCATION_KEEP_LENGTH characters
        first_k_chars = output[:TRUNCATION_KEEP_LENGTH]
        last_k_chars = output[-TRUNCATION_KEEP_LENGTH:]

        truncated_len = len(output) - 2 * TRUNCATION_KEEP_LENGTH

        if truncated_len <= 0:
            return output
        return f"{first_k_chars}\n ... [{truncated_len} characters truncated] ... \n{last_k_chars}"
    else:
        return output


def run_evaluation(eval_command: str, timeout: int | None = None) -> str:
    """Run the evaluation command on the code and return the output."""

    # Run the eval command as is
    try:
        result = subprocess.run(eval_command, shell=True, capture_output=True, text=True, check=False, timeout=timeout)
        # Combine stdout and stderr for complete output
        output = result.stderr if result.stderr else ""
        if result.stdout:
            if len(output) > 0:
                output += "\n"
            output += result.stdout
        return output  # Return full output, no truncation
    except subprocess.TimeoutExpired:
        return f"Evaluation timed out after {'an unspecified duration' if timeout is None else f'{timeout} seconds'}."


# Update Check Function
def check_for_cli_updates():
    """Checks PyPI for a newer version of the weco package and notifies the user."""
    try:
        from . import __pkg_version__

        pypi_url = "https://pypi.org/pypi/weco/json"
        response = requests.get(pypi_url, timeout=5)  # Short timeout for non-critical check
        response.raise_for_status()
        latest_version_str = response.json()["info"]["version"]

        current_version = parse_version(__pkg_version__)
        latest_version = parse_version(latest_version_str)

        if latest_version > current_version:
            yellow_start = "\033[93m"
            reset_color = "\033[0m"
            message = f"WARNING: New weco version ({latest_version_str}) available (you have {__pkg_version__}). Run: pip install --upgrade weco"
            print(f"{yellow_start}{message}{reset_color}")
            time.sleep(2)  # Wait for 2 second

    except requests.exceptions.RequestException:
        # Silently fail on network errors, etc. Don't disrupt user.
        pass
    except (KeyError, ValueError):
        # Handle cases where the PyPI response format might be unexpected
        pass
    except Exception:
        # Catch any other unexpected error during the check
        pass
