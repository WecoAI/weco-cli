"""Utility functions for the LangFuse setup wizard."""

import ast
from pathlib import Path


def discover_functions(source_files: list[str]) -> dict:
    """Scan Python files using AST to find target and evaluator candidates.

    Targets: functions with an ``inputs`` parameter (convention for LangFuse/LangSmith targets).
    Evaluators: functions with keyword-only parameters matching LangFuse's evaluator
    signature (input, output, expected_output).

    Also scans ``evaluators.py`` in the same directory as each source file (the
    conventional location for custom evaluators).
    """
    targets: list[dict] = []
    evaluators: list[dict] = []
    seen_files: set[str] = set()

    files_to_scan = list(source_files)

    # Also look for evaluators.py alongside each source file
    for src in source_files:
        src_path = Path(src)
        for candidate in [src_path.parent / "evaluators.py", Path("evaluators.py")]:
            if candidate.is_file() and str(candidate) not in seen_files:
                files_to_scan.append(str(candidate))

    for filepath in files_to_scan:
        if filepath in seen_files:
            continue
        seen_files.add(filepath)

        try:
            source = Path(filepath).read_text(encoding="utf-8")
            tree = ast.parse(source)
        except Exception:
            continue

        # Derive module name from file path (strip .py, replace / with .)
        module = Path(filepath).stem

        for node in ast.iter_child_nodes(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name.startswith("_"):
                continue

            params = [arg.arg for arg in node.args.args]
            kwonly_params = [arg.arg for arg in node.args.kwonlyargs]
            docstring = ast.get_docstring(node) or ""
            spec = f"{module}:{node.name}"

            # LangFuse evaluators use keyword-only params: (*, input, output, expected_output, ...)
            if "input" in kwonly_params and "output" in kwonly_params:
                evaluators.append({"spec": spec, "name": node.name, "file": filepath, "doc": docstring[:80]})
            # Also detect LangSmith-style evaluators for compatibility
            elif params == ["run", "example"]:
                evaluators.append({"spec": spec, "name": node.name, "file": filepath, "doc": docstring[:80]})
            elif "inputs" in params:
                targets.append({"spec": spec, "name": node.name, "file": filepath, "doc": docstring[:80]})

    return {"targets": targets, "evaluators": evaluators}


_SKIP_DIRS = {"__pycache__", "node_modules", ".git", ".venv", "venv", ".tox", ".mypy_cache", ".ruff_cache", ".eggs"}


def list_directory(dir_path: Path, project_root: Path) -> list[dict]:
    """List one level of a directory, returning .py files and sub-directories.

    Skips hidden entries and common non-source directories.
    Entries are sorted directories-first, then alphabetically.
    """
    entries: list[dict] = []

    for child in sorted(dir_path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
        if child.name.startswith("."):
            continue
        if child.name in _SKIP_DIRS:
            continue

        if child.is_dir():
            entries.append({"name": child.name, "path": str(child.relative_to(project_root)), "is_dir": True, "size": None})
        elif child.suffix == ".py":
            entries.append(
                {
                    "name": child.name,
                    "path": str(child.relative_to(project_root)),
                    "is_dir": False,
                    "size": child.stat().st_size,
                }
            )

    return entries
