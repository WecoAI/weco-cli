## Multi-file optimization support

Extends the CLI to optimize across multiple source files in a single run, while remaining fully backwards-compatible with the existing single-file (`-s/--source`) workflow.

### Changes

**CLI (`cli.py`)**
- Add `--sources` flag (mutually exclusive with `-s/--source`) accepting multiple file paths.
- Normalize both input modes into a unified `list[str]` before passing downstream.

**Optimizer (`optimizer.py`)**
- Represent source code as `dict[str, str]` (path -> content) throughout the optimization loop and resume flow.
- Normalize server responses (single string or dict) into a file map at every boundary.
- Replace inline log-writing with the new `RunArtifacts` manager.
- Multi-file apply: `_offer_apply_best_solution` writes back all modified files and displays which paths changed.

**Artifacts (`artifacts.py`) — new**
- `RunArtifacts` class that owns the on-disk layout under `.runs/<run_id>/`.
- Step snapshots (`steps/<n>/files/` + `manifest.json`), best snapshot (`best/`), and execution output (`outputs/` + `exec_output.jsonl`).
- Path sanitization (`_sanitize_artifact_path`) strips traversal components, absolute prefixes, and Windows drive letters.

**Utils (`utils.py`)**
- Add `run_evaluation_with_files_swap()` — atomically swaps multiple files before evaluation and restores originals in a `finally` block.

**Validation (`validation.py`)**
- Add `validate_source_files()` with per-file size (200 KB), total size (500 KB), file count (10), and duplicate-path checks.
- Add `validate_sources()` dispatcher that routes single vs. multi-file input.

**API (`api.py`)**
- Widen `start_optimization_run` signature: `source_code` accepts `str | dict[str, str]`, `source_path` accepts `str | None`.

**Tests (`test_artifacts.py`) — new**
- Parametrized path-sanitization tests (relative, absolute, traversal, Windows, empty).
- Step/best snapshot round-trip tests, step independence, JSONL index verification.
- Path-traversal containment tests for both step and best snapshots.

## Test plan

- [ ] `weco run -s single.py ...` still works as before (single-file backwards compat)
- [ ] `weco run --sources a.py b.py ...` runs multi-file optimization end-to-end
- [ ] `-s` and `--sources` are mutually exclusive (argparse rejects both together)
- [ ] Validation rejects: >10 files, >200 KB per file, >500 KB total, duplicate paths
- [ ] Step artifacts written to `.runs/<id>/steps/<n>/files/` with correct manifest
- [ ] Best solution artifacts written to `.runs/<id>/best/files/`
- [ ] Path traversal payloads (e.g. `../../etc/passwd`) are contained within the artifact directory
- [ ] `--apply-change` writes back all modified files for multi-file runs
- [ ] Resume flow correctly reconstructs the file map from server response
- [ ] `pytest tests/test_artifacts.py` passes
