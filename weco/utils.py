from typing import Any, Dict, List, Tuple, Union
import json
import time
import subprocess
import psutil
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
import pathlib
import requests
from packaging.version import parse as parse_version
from .constants import TRUNCATION_THRESHOLD, TRUNCATION_KEEP_LENGTH, SUPPORTED_FILE_EXTENSIONS, DEFAULT_MODELS


class UnrecognizedAPIKeysError(Exception):
    """Exception raised when unrecognized API keys are provided."""

    def __init__(self, api_keys: dict[str, str]):
        self.api_keys = api_keys
        providers = {provider for provider, _ in DEFAULT_MODELS}
        super().__init__(
            f"Unrecognized API key provider in {set(api_keys.keys())}. Supported providers: {', '.join(providers)}"
        )


class DefaultModelNotFoundError(Exception):
    """Exception raised when no default model is found for the API keys."""

    def __init__(self, api_keys: dict[str, str]):
        self.api_keys = api_keys
        super().__init__(f"No default model found for any of the provided API keys: {set(api_keys.keys())}")


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
                raise ValueError(
                    f"Unsupported file extension: {potential_path.suffix.lower()}. Supported extensions are: {', '.join(SUPPORTED_FILE_EXTENSIONS)}"
                )
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


def run_evaluation_with_file_swap(
    file_path: pathlib.Path, new_content: str, original_content: str, eval_command: str, timeout: int | None = None
) -> str:
    """
    Temporarily write new content to a file, run evaluation, then restore original.

    This function ensures the file is always restored to its original state,
    even if an exception occurs during evaluation.

    Args:
        file_path: Path to the file to temporarily modify
        new_content: The new content to write for evaluation
        original_content: The original content to restore after evaluation
        eval_command: The shell command to run for evaluation
        timeout: Optional timeout for the evaluation command

    Returns:
        The output from running the evaluation command

    Raises:
        Any exception raised by run_evaluation will be re-raised after
        the file is restored to its original state.
    """
    # Write the new content
    write_to_path(fp=file_path, content=new_content)

    try:
        # Run the evaluation
        output = run_evaluation(eval_command=eval_command, timeout=timeout)
        return output
    finally:
        # Always restore the original file, even if evaluation fails
        write_to_path(fp=file_path, content=original_content)


def run_evaluation(eval_command: str, timeout: int | None = None) -> str:
    """Run the evaluation command on the code and return the output."""
    process = subprocess.Popen(
        eval_command, shell=True, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )

    try:
        # NOTE: Process tree cleanup only happens on timeout. Normal completion relies on the OS/shell to clean up child processes, which works for typical evaluation scripts.
        output, _ = process.communicate(timeout=timeout)
        return output

    except subprocess.TimeoutExpired:
        # Kill process tree
        try:
            parent = psutil.Process(process.pid)
            children = parent.children(recursive=True)

            # Terminate gracefully
            for child in children:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            try:
                parent.terminate()
            except psutil.NoSuchProcess:
                pass

            # Wait, then force kill survivors
            _, alive = psutil.wait_procs(children + [parent], timeout=1)
            for proc in alive:
                try:
                    proc.kill()
                except psutil.NoSuchProcess:
                    pass

        except psutil.NoSuchProcess:
            pass

        # Drain pipes
        try:
            process.communicate(timeout=1)
        except (subprocess.TimeoutExpired, ValueError, OSError):
            pass

        return f"Evaluation timed out after {'an unspecified duration' if timeout is None else f'{timeout} seconds'}."


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


def get_default_model(api_keys: dict[str, str] | None = None) -> str:
    """Determine the default model to use based on the API keys."""
    providers = {provider for provider, _ in DEFAULT_MODELS}
    if api_keys and not all(provider in providers for provider in api_keys.keys()):
        raise UnrecognizedAPIKeysError(api_keys)

    if api_keys:
        for provider, model in DEFAULT_MODELS:
            if provider in api_keys:
                return model
        # Should never happen, but just in case
        raise DefaultModelNotFoundError(api_keys)
    return DEFAULT_MODELS[0][1]
