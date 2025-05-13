from typing import Any, Dict, List, Tuple, Union, Callable
import json
import os
import time
import subprocess
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
import pathlib


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


def read_additional_instructions(additional_instructions: str | None) -> str | None:
    """Read additional instructions from a file path string or return the string itself."""
    if additional_instructions is None:
        return None

    # Try interpreting as a path first
    potential_path = pathlib.Path(additional_instructions)
    try:
        if potential_path.exists() and potential_path.is_file():
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
            raise TypeError("Content must be str or Dict[str, Any]")


# Visualization helper functions
def format_number(n: Union[int, float]) -> str:
    """Format large numbers with K, M, B, T suffixes for better readability."""
    if n >= 1e12:
        return f"{n / 1e12:.1f}T"
    elif n >= 1e9:
        return f"{n / 1e9:.1f}B"
    elif n >= 1e6:
        return f"{n / 1e6:.1f}M"
    elif n >= 1e3:
        return f"{n / 1e3:.1f}K"
    # Handle potential floats that don't need suffix but might need formatting
    if isinstance(n, float):
        # Format floats nicely, avoid excessive precision unless needed
        return f"{n:.4g}"  # Use general format, up to 4 significant digits
    return str(n)


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
def run_evaluation(
    eval_command: str,
    *,
    log_path: pathlib.Path | None = None,
    on_update: Callable[[str], None] | None = None,
) -> str:
    """
    Run `eval_command` and stream *combined* stdout+stderr.

    Parameters
    ----------
    eval_command : str
        Shell command to execute.
    log_path : pathlib.Path | None
        If given, write the full output to this file when finished.
    on_update : Callable[[str], None] | None
        Callback that receives the *current* accumulated output every time
        a new line arrives – handy for updating the Rich panel live.

    Returns
    -------
    str
        The complete output after the process exits.
    """
    proc = subprocess.Popen(
        eval_command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1, # line-buffered
    )

    lines: List[str] = []
    assert proc.stdout is not None # for type-checkers
    with proc.stdout:
        for line in iter(proc.stdout.readline, ""):
            lines.append(line)
            if on_update:
                on_update("".join(lines))

    proc.wait()

    full_output = "".join(lines)
    if log_path:
        log_path.write_text(full_output, encoding="utf-8")

    # If the eval script crashed, surface it
    if proc.returncode and proc.returncode != 0:
        full_output += f"\n\n[Process exited with code {proc.returncode}]"

    return full_output