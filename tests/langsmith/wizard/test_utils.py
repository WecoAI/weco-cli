"""Tests for the LangSmith wizard utility functions."""

import os

from weco.integrations.langsmith.wizard.utils import discover_functions, list_directory


class TestDiscoverFunctions:
    def test_discovers_target_function(self, tmp_path):
        """discover_functions finds functions with an 'inputs' parameter."""
        src = tmp_path / "agent.py"
        src.write_text('def answer_question(inputs: dict) -> dict:\n    """Answer a question."""\n    return {}\n')

        result = discover_functions([str(src)])
        assert len(result["targets"]) == 1
        assert result["targets"][0]["name"] == "answer_question"
        assert result["targets"][0]["spec"] == "agent:answer_question"
        assert "Answer a question" in result["targets"][0]["doc"]

    def test_discovers_evaluator_function(self, tmp_path):
        """discover_functions finds functions with (run, example) parameters."""
        src = tmp_path / "evaluators.py"
        src.write_text(
            'def correctness(run, example):\n    """Check correctness."""\n    return {"key": "correctness", "score": 1}\n'
        )

        result = discover_functions([str(src)])
        assert len(result["evaluators"]) == 1
        assert result["evaluators"][0]["name"] == "correctness"
        assert result["evaluators"][0]["spec"] == "evaluators:correctness"

    def test_skips_private_functions(self, tmp_path):
        """discover_functions skips functions starting with underscore."""
        src = tmp_path / "mod.py"
        src.write_text("def _helper(inputs):\n    pass\n\ndef run(inputs):\n    pass\n")

        result = discover_functions([str(src)])
        assert len(result["targets"]) == 1
        assert result["targets"][0]["name"] == "run"

    def test_auto_scans_evaluators_py(self, tmp_path):
        """discover_functions auto-discovers evaluators.py alongside source files."""
        src = tmp_path / "agent.py"
        src.write_text("def run(inputs):\n    pass\n")
        evals = tmp_path / "evaluators.py"
        evals.write_text("def check(run, example):\n    pass\n")

        result = discover_functions([str(src)])
        assert len(result["targets"]) == 1
        assert len(result["evaluators"]) == 1
        assert result["evaluators"][0]["name"] == "check"

    def test_handles_missing_files(self):
        """discover_functions handles non-existent files gracefully."""
        result = discover_functions(["/nonexistent/file.py"])
        assert result["targets"] == []
        assert result["evaluators"] == []


class TestListDirectory:
    def test_lists_py_files_and_dirs(self, tmp_path):
        """list_directory returns .py files and directories, skipping others."""
        (tmp_path / "agent.py").write_text("pass")
        (tmp_path / "readme.txt").write_text("hi")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "inner.py").write_text("pass")

        entries = list_directory(tmp_path, tmp_path)

        names = [e["name"] for e in entries]
        assert "sub" in names
        assert "agent.py" in names
        assert "readme.txt" not in names

    def test_directories_first(self, tmp_path):
        """Directories appear before files."""
        (tmp_path / "z_file.py").write_text("pass")
        (tmp_path / "a_dir").mkdir()

        entries = list_directory(tmp_path, tmp_path)

        assert entries[0]["name"] == "a_dir"
        assert entries[0]["is_dir"] is True
        assert entries[1]["name"] == "z_file.py"
        assert entries[1]["is_dir"] is False

    def test_skips_hidden_and_pycache(self, tmp_path):
        """Hidden files/dirs and __pycache__ are excluded."""
        (tmp_path / ".hidden").mkdir()
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "node_modules").mkdir()
        (tmp_path / ".env").write_text("SECRET=x")
        (tmp_path / "app.py").write_text("pass")

        entries = list_directory(tmp_path, tmp_path)

        names = [e["name"] for e in entries]
        assert names == ["app.py"]

    def test_relative_paths(self, tmp_path):
        """Paths are relative to project_root."""
        sub = tmp_path / "src"
        sub.mkdir()
        (sub / "main.py").write_text("pass")

        entries = list_directory(sub, tmp_path)

        assert entries[0]["path"] == os.path.join("src", "main.py")

    def test_file_size(self, tmp_path):
        """Files include size; directories have size=None."""
        (tmp_path / "small.py").write_text("x = 1")
        (tmp_path / "pkg").mkdir()

        entries = list_directory(tmp_path, tmp_path)

        dir_entry = next(e for e in entries if e["is_dir"])
        file_entry = next(e for e in entries if not e["is_dir"])
        assert dir_entry["size"] is None
        assert file_entry["size"] > 0
