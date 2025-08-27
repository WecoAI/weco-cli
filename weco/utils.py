from typing import Any, Dict, List, Tuple, Union
import json
import os
import time
import subprocess
import re
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
import pathlib
import requests
from packaging.version import parse as parse_version


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


def determine_default_model(llm_api_keys: Dict[str, Any]) -> str:
    """Determine the default model based on available API keys.

    Uses priority: OpenAI > Anthropic > Gemini

    Args:
        llm_api_keys: Dictionary of available LLM API keys

    Returns:
        str: The default model name to use

    Raises:
        ValueError: If no LLM API keys are found
    """
    if "OPENAI_API_KEY" in llm_api_keys:
        return "o4-mini"
    elif "ANTHROPIC_API_KEY" in llm_api_keys:
        return "claude-sonnet-4-0"
    elif "GEMINI_API_KEY" in llm_api_keys:
        return "gemini-2.5-pro"
    else:
        raise ValueError(
            "No LLM API keys found in environment variables. Please set one of the following: OPENAI_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY based on your model of choice."
        )


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
            raise TypeError("Error writing to file. Please verify the file path and try again.")


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
DEFAULT_MAX_LINES = 50
DEFAULT_MAX_CHARS = 5000


def truncate_output(output: str, max_lines: int = DEFAULT_MAX_LINES, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """Truncate the output to a reasonable size while preserving lines with metrics.

    This function identifies and preserves important metric lines even when truncating,
    ensuring that evaluation scores and metrics are not lost in multi-threaded output.
    """
    lines = output.splitlines()

    # If output is already small enough, return as is
    if len(lines) <= max_lines and len(output) <= max_chars:
        return output

    # Patterns that identify important metric lines
    # These patterns are case-insensitive and look for common metric formats
    metric_patterns = [
        # Final scores/metrics with various formats
        r"(?i)final[\s_-]*(?:score|metric|accuracy|result).*?[:\s=]+[\d.]+",
        r"(?i)(?:test|eval|evaluation)[\s_-]*(?:score|metric|accuracy).*?[:\s=]+[\d.]+",
        r"(?i)total[\s_-]*(?:score|metric|accuracy|result).*?[:\s=]+[\d.]+",
        # Standalone metric lines
        r"(?i)^[\s]*(?:score|accuracy|precision|recall|f1|loss|error|metric)[\s]*[:\s=]+[\s]*[\d.]+",
        # Percentage formats
        r"(?i)(?:score|accuracy|pass[\s_-]*rate).*?[:\s=]+[\s]*[\d.]+[\s]*%",
        # Test results summary
        r"(?i)(?:passed|failed|tests?).*?(\d+[\s]*/[\s]*\d+)",
        r"(?i)(\d+[\s]*/[\s]*\d+).*?(?:passed|failed|tests?)",
        # JSON-like metric lines
        r'["\'](?:score|metric|accuracy|result)["\'][\s]*:[\s]*[\d.]+',
        # Common evaluation framework outputs
        r"(?i)PASS:[\s]*[\d.]+",
        r"(?i)FAIL:[\s]*[\d.]+",
        r"(?i)Result:[\s]*[\d.]+",
    ]

    # Compile patterns for efficiency
    compiled_patterns = [re.compile(pattern) for pattern in metric_patterns]

    # Separate lines into important (metrics) and regular
    metric_lines = []
    metric_line_indices = []

    for i, line in enumerate(lines):
        # Check if line contains any metric pattern
        if any(pattern.search(line) for pattern in compiled_patterns):
            metric_lines.append((i, line))
            metric_line_indices.append(i)

    # Strategy: Preserve metric lines and as much context as possible
    if metric_lines:
        # Reserve space for metric lines (max 20 metric lines to prevent abuse)
        max_metric_lines = min(20, len(metric_lines))
        # Prioritize recent metrics (last N metric lines)
        preserved_metrics = metric_lines[-max_metric_lines:]
        preserved_indices = set(idx for idx, _ in preserved_metrics)

        # Calculate remaining space for context
        remaining_lines = max_lines - max_metric_lines

        if remaining_lines > 0:
            # Get non-metric lines
            context_lines = [(i, line) for i, line in enumerate(lines) if i not in preserved_indices]

            # Take the most recent context lines
            if len(context_lines) > remaining_lines:
                context_lines = context_lines[-remaining_lines:]

            # Combine and sort by original index to maintain order
            all_lines = context_lines + preserved_metrics
            all_lines.sort(key=lambda x: x[0])

            # Extract just the lines
            result_lines = [line for _, line in all_lines]
        else:
            # If no space for context, just keep metrics
            result_lines = [line for _, line in preserved_metrics]

        output = "\n".join(result_lines)

        # Apply character limit if needed, but try to keep last metric
        if len(output) > max_chars:
            # Find the last metric line in the output
            last_metric_pos = -1
            for _, line in reversed(preserved_metrics):
                pos = output.rfind(line)
                if pos != -1:
                    last_metric_pos = pos + len(line)
                    break

            if last_metric_pos > 0 and last_metric_pos <= max_chars:
                # Truncate but ensure we include the last metric
                output = output[-max_chars:]
                # Make sure we didn't cut off the last metric line
                if not any(pattern.search(output.split("\n")[-1]) for pattern in compiled_patterns):
                    # Try to include at least the last metric line
                    for _, line in reversed(preserved_metrics):
                        if len(line) < max_chars:
                            output = "...\n" + line
                            break
            else:
                output = output[-max_chars:]

        # Add truncation notice
        truncation_notice = f"... (truncated to {len(result_lines)} lines with {max_metric_lines} metric lines preserved)\n"
        output = truncation_notice + output

    else:
        # No metrics found, fall back to simple truncation
        if len(lines) > max_lines:
            output = "\n".join(lines[-max_lines:])

        if len(output) > max_chars:
            output = output[-max_chars:]

        # Add truncation notice
        prefixes = []
        if len(lines) > max_lines:
            prefixes.append(f"truncated to last {max_lines} lines")
        if len(output) > max_chars:
            prefixes.append(f"truncated to last {max_chars} characters")

        if prefixes:
            prefix_text = ", ".join(prefixes)
            output = f"... ({prefix_text})\n{output}"

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
        return truncate_output(output)
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
