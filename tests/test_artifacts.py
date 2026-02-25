"""Tests for run artifact persistence and path sanitization."""

import json

import pytest

from weco.artifacts import RunArtifacts, _sanitize_artifact_path


@pytest.fixture
def artifacts(tmp_path):
    return RunArtifacts(log_dir=str(tmp_path), run_id="test-run")


def _read_manifest(path):
    return json.loads(path.read_text())


@pytest.mark.parametrize(
    ("raw_path", "expected_parts"),
    [
        ("model.py", ("model.py",)),
        ("src/model.py", ("src", "model.py")),
        ("./src/model.py", ("src", "model.py")),
        ("/absolute/path.py", ("absolute", "path.py")),
        ("src\\utils\\helper.py", ("src", "utils", "helper.py")),
        ("../../etc/passwd", ("etc", "passwd")),
        ("", ("unnamed_file",)),
        ("../../..", ("unnamed_file",)),
    ],
)
def test_sanitize_artifact_path(raw_path, expected_parts):
    assert _sanitize_artifact_path(raw_path).parts == expected_parts


def test_save_step_code_writes_files_and_manifest(artifacts):
    bundle = artifacts.save_step_code(
        step=3, file_map={"src/model.py": "class Model: pass", "src/utils.py": "def helper(): pass"}
    )

    assert bundle == artifacts.root / "steps" / "3"
    assert (bundle / "files" / "src" / "model.py").read_text() == "class Model: pass"
    assert (bundle / "files" / "src" / "utils.py").read_text() == "def helper(): pass"

    manifest = _read_manifest(bundle / "manifest.json")
    assert manifest["type"] == "step_code_snapshot"
    assert manifest["step"] == 3
    assert manifest["file_count"] == 2
    assert [file_entry["path"] for file_entry in manifest["files"]] == ["src/model.py", "src/utils.py"]
    assert [file_entry["artifact_path"] for file_entry in manifest["files"]] == ["src/model.py", "src/utils.py"]


def test_save_step_code_keeps_steps_independent(artifacts):
    artifacts.save_step_code(step=0, file_map={"f.py": "v1"})
    artifacts.save_step_code(step=1, file_map={"f.py": "v2"})

    assert (artifacts.root / "steps" / "0" / "files" / "f.py").read_text() == "v1"
    assert (artifacts.root / "steps" / "1" / "files" / "f.py").read_text() == "v2"


def test_save_best_code_writes_manifest_without_step(artifacts):
    bundle = artifacts.save_best_code({"model.py": "optimized = True"})

    assert bundle == artifacts.root / "best"
    assert (bundle / "files" / "model.py").read_text() == "optimized = True"

    manifest = _read_manifest(bundle / "manifest.json")
    assert manifest["type"] == "best_code_snapshot"
    assert manifest["file_count"] == 1
    assert "step" not in manifest


def test_save_execution_output_writes_step_file_and_jsonl_index(artifacts):
    artifacts.save_execution_output(step=0, output="first")
    artifacts.save_execution_output(step=1, output="second")

    assert (artifacts.root / "outputs" / "step_0.out.txt").read_text() == "first"
    assert (artifacts.root / "outputs" / "step_1.out.txt").read_text() == "second"

    lines = (artifacts.root / "exec_output.jsonl").read_text().strip().split("\n")
    assert len(lines) == 2
    first_entry = json.loads(lines[0])
    second_entry = json.loads(lines[1])

    assert first_entry["step"] == 0
    assert first_entry["output_file"] == "outputs/step_0.out.txt"
    assert first_entry["output_length"] == len("first")
    assert second_entry["step"] == 1
    assert second_entry["output_file"] == "outputs/step_1.out.txt"
    assert second_entry["output_length"] == len("second")


def test_root_directory_creation_is_idempotent(tmp_path):
    first = RunArtifacts(log_dir=str(tmp_path), run_id="abc-123")
    second = RunArtifacts(log_dir=str(tmp_path), run_id="abc-123")

    assert first.root == second.root == (tmp_path / "abc-123")
    assert first.root.exists()


def test_step_snapshot_sanitizes_path_traversal(artifacts, tmp_path):
    artifacts.save_step_code(step=0, file_map={"../../etc/evil.py": "malicious"})

    assert not (tmp_path / "etc" / "evil.py").exists()
    assert (artifacts.root / "steps" / "0" / "files" / "etc" / "evil.py").exists()


def test_best_snapshot_sanitizes_path_traversal(artifacts, tmp_path):
    artifacts.save_best_code({"../../../tmp/evil.py": "malicious"})

    assert not (tmp_path.parent / "tmp" / "evil.py").exists()
    assert (artifacts.root / "best" / "files" / "tmp" / "evil.py").exists()
