from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union
import io
import json
import os
import shutil
import time
import subprocess
import zipfile
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
import psutil
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
import pathlib
from .constants import TRUNCATION_THRESHOLD, TRUNCATION_KEEP_LENGTH, SUPPORTED_FILE_EXTENSIONS, DEFAULT_MODELS

if TYPE_CHECKING:
    import threading


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


def write_to_path(fp: pathlib.Path, content: Union[str, Dict[str, Any]], is_json: bool = False, mkdir: bool = False) -> None:
    """
    Write content to a file path, optionally as JSON.

    Args:
        fp: File path to write to.
        content: Content to write (string or dict for JSON).
        is_json: If True, write as JSON.
        mkdir: If True, create parent directories if they don't exist.
    """
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
    """
    Copy a single file.

    Args:
        src: Source file path.
        dest: Destination file path.
        mkdir: If True, create parent directories if they don't exist.

    Raises:
        FileNotFoundError: If source doesn't exist.
        OSError: If copy fails.
    """
    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")
    if mkdir:
        dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dest)


def copy_directory(src: pathlib.Path, dest: pathlib.Path, ignore_patterns: set[str] | None = None) -> None:
    """
    Copy a directory tree.

    Args:
        src: Source directory path.
        dest: Destination directory path.
        ignore_patterns: Optional set of file/directory names to skip.

    Raises:
        FileNotFoundError: If source doesn't exist.
        OSError: If copy fails.
    """
    if not src.exists():
        raise FileNotFoundError(f"Source directory not found: {src}")

    def ignore_func(_: str, files: list[str]) -> list[str]:
        if not ignore_patterns:
            return []
        return [f for f in files if f in ignore_patterns]

    shutil.copytree(src, dest, ignore=ignore_func)


def remove_directory(path: pathlib.Path) -> None:
    """
    Remove a directory and all its contents.

    Does nothing if the directory doesn't exist.
    """
    if path.exists():
        shutil.rmtree(path)


class UnsafeRemoveError(Exception):
    """Raised when ``safe_remove_directory`` refuses to remove a path."""


def safe_remove_directory(path: pathlib.Path, *, allowed_parents: set[pathlib.Path], expected_name: str | None = None) -> None:
    """Remove a directory only if it passes defensive safety checks.

    The path must:
      * exist and be a real directory (not a symlink)
      * not be home, root, or an ancestor of home
      * be a direct child of one of ``allowed_parents``
      * match ``expected_name`` if supplied

    Does nothing if the path doesn't exist. Raises ``UnsafeRemoveError`` if any
    other check fails.
    """
    resolved = path.resolve()

    if not resolved.exists():
        return
    if not resolved.is_dir():
        raise UnsafeRemoveError(f"Refusing to remove: not a directory: {resolved}")
    if resolved.is_symlink():
        raise UnsafeRemoveError(f"Refusing to remove: path is a symlink: {resolved}")

    home = pathlib.Path.home().resolve()
    if resolved == home:
        raise UnsafeRemoveError(f"Refusing to remove: path is home directory: {resolved}")
    if resolved == pathlib.Path("/").resolve():
        raise UnsafeRemoveError(f"Refusing to remove: path is root directory: {resolved}")
    try:
        home.relative_to(resolved)
        raise UnsafeRemoveError(f"Refusing to remove: path is a parent of home directory: {resolved}")
    except ValueError:
        pass

    resolved_allowed = {p.resolve() for p in allowed_parents}
    if resolved.parent not in resolved_allowed:
        raise UnsafeRemoveError(
            f"Refusing to remove: path {resolved} is not a direct child of allowed directories: "
            f"{[str(p) for p in resolved_allowed]}"
        )

    if expected_name is not None and resolved.name != expected_name:
        raise UnsafeRemoveError(f"Refusing to remove: directory name is not {expected_name!r}: {resolved}")

    shutil.rmtree(resolved)


class DownloadError(Exception):
    """Raised when downloading or extracting a remote archive fails."""


def download_github_archive(url: str, dest: pathlib.Path, *, timeout: int = 60) -> None:
    """Download a GitHub-style zip archive and extract its contents to ``dest``.

    GitHub archive zips wrap everything in a single top-level directory (e.g.
    ``repo-main/``); this function strips that prefix so ``dest`` receives the
    repo contents directly.
    """
    try:
        with urlopen(url, timeout=timeout) as response:
            zip_data = io.BytesIO(response.read())
    except HTTPError as e:
        raise DownloadError(f"Failed to download: HTTP {e.code} - {e.reason}")
    except URLError as e:
        raise DownloadError(f"Failed to download: {e.reason}")
    except TimeoutError:
        raise DownloadError("Failed to download: connection timed out")
    except Exception as e:
        raise DownloadError(f"Failed to download: {e}")

    try:
        with zipfile.ZipFile(zip_data) as zf:
            top_level_dirs = {name.split("/")[0] for name in zf.namelist() if "/" in name}
            if len(top_level_dirs) != 1:
                raise DownloadError("Unexpected zip structure: expected single top-level directory")

            prefix = f"{top_level_dirs.pop()}/"
            dest.mkdir(parents=True, exist_ok=True)

            for member in zf.namelist():
                if not member.startswith(prefix):
                    continue
                relative_path = member[len(prefix) :]
                if not relative_path:
                    continue
                target_path = dest / relative_path
                if member.endswith("/"):
                    target_path.mkdir(parents=True, exist_ok=True)
                else:
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(member) as source, open(target_path, "wb") as out:
                        out.write(source.read())
    except zipfile.BadZipFile:
        raise DownloadError("Downloaded file is not a valid zip archive")
    except DownloadError:
        raise
    except Exception as e:
        raise DownloadError(f"Failed to extract zip: {e}")


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


def run_evaluation_with_files_swap(
    file_map: dict[str, str],
    originals: dict[str, str],
    eval_command: str,
    timeout: int | None = None,
    cwd: "pathlib.Path | str | None" = None,
    env: dict[str, str] | None = None,
) -> str:
    """
    Temporarily write multiple files, run evaluation, then restore all originals.

    File paths in ``file_map`` and ``originals`` are relative to ``cwd`` (the
    current working directory when omitted — the serial in-place path). Parent
    directories are created as needed. ``cwd``/``env`` exist for isolated slot
    evaluation (K>1): the candidate is written into and evaluated inside the
    slot's own project copy, never the user's tree.

    Args:
        file_map: Dict mapping relative file paths to their new content.
        originals: Dict mapping relative file paths to their original content.
        eval_command: The shell command to run for evaluation.
        timeout: Optional timeout for the evaluation command.
        cwd: Directory the paths resolve against and the command runs in.
        env: Full environment for the command (``None`` inherits the caller's).

    Returns:
        The output from running the evaluation command.
    """
    base = pathlib.Path(cwd) if cwd is not None else None

    def _resolve(rel_path: str) -> pathlib.Path:
        return (base / rel_path) if base is not None else pathlib.Path(rel_path)

    # Write all new files
    for rel_path, content in file_map.items():
        fp = _resolve(rel_path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        write_to_path(fp=fp, content=content)

    try:
        output = run_evaluation(eval_command=eval_command, timeout=timeout, cwd=cwd, env=env)
        return output
    finally:
        # Always restore all originals
        for rel_path, content in originals.items():
            fp = _resolve(rel_path)
            write_to_path(fp=fp, content=content)


class EvaluationCancelled(Exception):
    """The evaluation was killed because its cancel event was set (run stopped)."""


def _kill_process_tree(process: "subprocess.Popen") -> None:
    """Terminate a shell command and everything it spawned.

    Kills the whole process group first (catches daemonized/double-forked
    children that escape the psutil parent walk — the eval runs with
    ``start_new_session=True`` so the group is ours to kill), then walks the
    remaining tree via psutil as a fallback for platforms without process
    groups.
    """
    import signal as _signal

    def _own_group() -> bool:
        # Only ever signal a group the child LEADS (start_new_session path).
        # A serial eval shares the CLI's group — killing that would kill us.
        try:
            return os.getpgid(process.pid) == process.pid
        except (ProcessLookupError, OSError):
            return False

    if hasattr(os, "killpg") and _own_group():
        try:
            os.killpg(process.pid, _signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            pass

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

    if hasattr(os, "killpg") and _own_group():
        try:
            os.killpg(process.pid, _signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass

    # Drain pipes
    try:
        process.communicate(timeout=1)
    except (subprocess.TimeoutExpired, ValueError, OSError):
        pass


def run_evaluation(
    eval_command: str,
    timeout: int | None = None,
    cwd: "pathlib.Path | str | None" = None,
    env: dict[str, str] | None = None,
    cancel_event: "threading.Event | None" = None,
) -> str:
    """Run the evaluation command on the code and return the output.

    ``cwd`` and ``env`` default to the caller's own working directory and
    environment (the serial behavior); slot-based evaluation passes the slot's
    project copy and its per-slot environment overlay.

    ``cancel_event`` makes the evaluation externally interruptible: when the
    event is set (e.g. the run was stopped from the dashboard) the whole
    process group is killed promptly and :class:`EvaluationCancelled` is
    raised. Without it, behavior is unchanged — the process runs to completion
    or timeout.
    """
    process = subprocess.Popen(
        eval_command,
        shell=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        # Cancellable (K>1 slot) evals get their own session/process group so
        # cancellation and timeout can kill the entire tree, including children
        # that re-parent themselves. Serial evals keep sharing the CLI's group —
        # exactly the pre-parallelization behavior, so a terminal Ctrl-C still
        # reaches the eval process directly.
        start_new_session=cancel_event is not None and hasattr(os, "setsid"),
    )

    if cancel_event is None:
        try:
            # NOTE: Process tree cleanup only happens on timeout. Normal completion relies on the OS/shell to clean up child processes, which works for typical evaluation scripts.
            output, _ = process.communicate(timeout=timeout)
            return output
        except subprocess.TimeoutExpired:
            _kill_process_tree(process)
            return f"Evaluation timed out after {'an unspecified duration' if timeout is None else f'{timeout} seconds'}."

    # Cancellable path: wait in short beats so a stop lands within ~a second.
    deadline = (time.monotonic() + timeout) if timeout is not None else None
    while True:
        if cancel_event.is_set():
            _kill_process_tree(process)
            raise EvaluationCancelled(eval_command)
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
            _kill_process_tree(process)
            return f"Evaluation timed out after {timeout} seconds."
        try:
            output, _ = process.communicate(timeout=min(1.0, remaining) if remaining is not None else 1.0)
            return output
        except subprocess.TimeoutExpired:
            continue


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
