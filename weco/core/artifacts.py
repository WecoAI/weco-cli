"""On-disk artifact management for optimization runs.

Centralizes the directory layout and write logic for all artifacts
produced during an optimization run under .runs/<run_id>/.
"""

import json
import pathlib
from datetime import datetime


def _sanitize_artifact_path(path_value: str) -> pathlib.Path:
    """Convert a source path into a safe relative artifact path.

    Strips traversal components (..), absolute prefixes, and Windows
    drive letters so that artifacts are always written under the
    intended directory.
    """
    normalized = path_value.replace("\\", "/")
    parts = pathlib.PurePosixPath(normalized).parts
    safe_parts: list[str] = []
    for part in parts:
        if part in ("", ".", "/"):
            continue
        if part == "..":
            continue
        if not safe_parts and ":" in part:
            part = part.replace(":", "_")
        safe_parts.append(part)

    if not safe_parts:
        return pathlib.Path("unnamed_file")
    return pathlib.Path(*safe_parts)


class RunArtifacts:
    """Manages the on-disk artifact layout for a single optimization run.

    Layout::

        <root>/
            steps/<step>/
                files/<relative_path>   # actual code files
                manifest.json           # machine-readable index
            best/
                files/<relative_path>
                manifest.json
            outputs/
                step_<n>.out.txt        # execution stdout/stderr
            exec_output.jsonl           # centralized output index
    """

    def __init__(self, log_dir: str, run_id: str) -> None:
        self.root = pathlib.Path(log_dir) / run_id
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Code snapshots
    # ------------------------------------------------------------------

    def save_step_code(self, step: int, file_map: dict[str, str]) -> pathlib.Path:
        """Write code snapshot + manifest for a given step.

        Returns the bundle directory path.
        """
        return self._write_code_bundle(file_map, label=("steps", str(step)))

    def save_best_code(self, file_map: dict[str, str]) -> pathlib.Path:
        """Write code snapshot + manifest for the best result.

        Returns the bundle directory path.
        """
        return self._write_code_bundle(file_map, label=("best",))

    # ------------------------------------------------------------------
    # Execution output
    # ------------------------------------------------------------------

    def save_execution_output(self, step: int, output: str) -> None:
        """Save execution output as a per-step file and append to the JSONL index."""
        timestamp = datetime.now().isoformat()

        outputs_dir = self.root / "outputs"
        # Keep raw execution output per step for easy local inspection.
        outputs_dir.mkdir(parents=True, exist_ok=True)

        step_file = outputs_dir / f"step_{step}.out.txt"
        # Store full stdout/stderr for this exact step.
        step_file.write_text(output, encoding="utf-8")

        jsonl_file = self.root / "exec_output.jsonl"
        entry = {
            "step": step,
            "timestamp": timestamp,
            "output_file": step_file.relative_to(self.root).as_posix(),
            "output_length": len(output),
        }
        # Append compact metadata so tooling can stream/index outputs.
        with open(jsonl_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_code_bundle(self, file_map: dict[str, str], label: tuple[str, ...]) -> pathlib.Path:
        bundle_dir = self.root.joinpath(*label)
        files_dir = bundle_dir / "files"
        files_dir.mkdir(parents=True, exist_ok=True)

        files_manifest: list[dict[str, str | int]] = []
        for source_path, content in sorted(file_map.items()):
            artifact_rel = _sanitize_artifact_path(source_path)
            artifact_path = files_dir / artifact_rel
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_text(content, encoding="utf-8")
            files_manifest.append(
                {"path": source_path, "artifact_path": artifact_rel.as_posix(), "bytes": len(content.encode("utf-8"))}
            )

        is_step = label[0] == "steps"
        manifest: dict = {
            "type": "step_code_snapshot" if is_step else "best_code_snapshot",
            "created_at": datetime.now().isoformat(),
            "file_count": len(files_manifest),
            "files": files_manifest,
        }
        if is_step:
            manifest["step"] = int(label[1])

        manifest_path = bundle_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return bundle_dir
