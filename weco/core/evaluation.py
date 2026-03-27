"""Local evaluation execution — runs eval commands and manages file swaps."""

import pathlib
import subprocess
import sys

import psutil


def run_evaluation(eval_command: str, timeout: int | None = None) -> str:
    """Run the evaluation command and return the captured output."""
    process = subprocess.Popen(
        eval_command, shell=True, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )

    try:
        output, _ = process.communicate(timeout=timeout)
        return output

    except subprocess.TimeoutExpired:
        # Kill the entire process tree
        try:
            parent = psutil.Process(process.pid)
            children = parent.children(recursive=True)

            for child in children:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            try:
                parent.terminate()
            except psutil.NoSuchProcess:
                pass

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


def run_evaluation_with_files_swap(
    file_map: dict[str, str], originals: dict[str, str], eval_command: str, timeout: int | None = None
) -> str:
    """Temporarily write candidate code, run evaluation, then restore originals.

    File paths in *file_map* and *originals* are relative to the current
    working directory. Parent directories are created as needed.
    """
    for rel_path, content in file_map.items():
        fp = pathlib.Path(rel_path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content, encoding="utf-8")

    try:
        return run_evaluation(eval_command=eval_command, timeout=timeout)
    finally:
        for rel_path, content in originals.items():
            pathlib.Path(rel_path).write_text(content, encoding="utf-8")
