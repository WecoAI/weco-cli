"""Tests for the LangSmith backend module (register_args, validate_args, build_eval_command)."""

import argparse
from unittest.mock import patch

import pytest

from weco.integrations.langsmith.backend import build_eval_command, register_args, validate_args


# ---------------------------------------------------------------------------
# Helper to create a parser with LangSmith args registered
# ---------------------------------------------------------------------------


def _make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", default="accuracy")
    parser.add_argument("--eval-backend", default="shell")
    register_args(parser)
    return parser


# ---------------------------------------------------------------------------
# register_args
# ---------------------------------------------------------------------------


class TestRegisterArgs:
    def test_adds_langsmith_flags(self):
        """register_args adds all expected --langsmith-* flags."""
        parser = _make_parser()
        args = parser.parse_args([])

        assert hasattr(args, "langsmith_dataset")
        assert hasattr(args, "langsmith_target")
        assert hasattr(args, "langsmith_evaluators")
        assert hasattr(args, "langsmith_experiment_prefix")
        assert hasattr(args, "langsmith_summary")
        assert hasattr(args, "langsmith_max_examples")
        assert hasattr(args, "langsmith_max_concurrency")
        assert hasattr(args, "langsmith_target_adapter")
        assert hasattr(args, "langsmith_metric_function")
        assert hasattr(args, "langsmith_dashboard_evaluators")
        assert hasattr(args, "langsmith_dashboard_evaluator_timeout")
        assert hasattr(args, "langsmith_splits")

    def test_defaults_are_none_or_expected(self):
        """Default values are None for optional args, 'mean' for summary, 'raw' for adapter."""
        parser = _make_parser()
        args = parser.parse_args([])

        assert args.langsmith_dataset is None
        assert args.langsmith_target is None
        assert args.langsmith_evaluators is None
        assert args.langsmith_summary == "mean"
        assert args.langsmith_target_adapter == "raw"
        assert args.langsmith_metric_function is None
        assert args.langsmith_dashboard_evaluators is None
        assert args.langsmith_dashboard_evaluator_timeout == 900
        assert args.langsmith_splits is None

    def test_parses_all_flags(self):
        """All flags can be parsed from command line."""
        parser = _make_parser()
        args = parser.parse_args(
            [
                "--langsmith-dataset",
                "my-data",
                "--langsmith-target",
                "agent:run",
                "--langsmith-evaluators",
                "acc",
                "rel",
                "--langsmith-experiment-prefix",
                "weco-test",
                "--langsmith-summary",
                "median",
                "--langsmith-max-examples",
                "10",
                "--langsmith-max-concurrency",
                "4",
                "--langsmith-target-adapter",
                "langchain",
                "--langsmith-metric-function",
                "scoring:combine",
                "--langsmith-dashboard-evaluators",
                "Conciseness",
                "--langsmith-dashboard-evaluator-timeout",
                "60",
                "--langsmith-splits",
                "train",
                "test",
            ]
        )

        assert args.langsmith_dataset == "my-data"
        assert args.langsmith_target == "agent:run"
        assert args.langsmith_evaluators == ["acc", "rel"]
        assert args.langsmith_experiment_prefix == "weco-test"
        assert args.langsmith_summary == "median"
        assert args.langsmith_max_examples == 10
        assert args.langsmith_max_concurrency == 4
        assert args.langsmith_target_adapter == "langchain"
        assert args.langsmith_metric_function == "scoring:combine"
        assert args.langsmith_dashboard_evaluators == ["Conciseness"]
        assert args.langsmith_dashboard_evaluator_timeout == 60
        assert args.langsmith_splits == ["train", "test"]


# ---------------------------------------------------------------------------
# validate_args
# ---------------------------------------------------------------------------


class TestValidateArgs:
    def test_missing_dataset_exits(self):
        """validate_args exits when --langsmith-dataset is missing."""
        parser = _make_parser()
        args = parser.parse_args(["--langsmith-target", "m:f"])
        with pytest.raises(SystemExit):
            validate_args(args)

    def test_missing_target_exits(self):
        """validate_args exits when --langsmith-target is missing."""
        parser = _make_parser()
        args = parser.parse_args(["--langsmith-dataset", "data"])
        with pytest.raises(SystemExit):
            validate_args(args)

    def test_defaults_evaluators_to_metric(self):
        """When no evaluators specified, defaults to [metric]."""
        parser = _make_parser()
        args = parser.parse_args(["--metric", "f1", "--langsmith-dataset", "data", "--langsmith-target", "m:f"])
        validate_args(args)
        assert args.langsmith_evaluators == ["f1"]

    def test_dashboard_evaluators_default_timeout(self):
        """When dashboard evaluators are set, timeout defaults to 900."""
        parser = _make_parser()
        args = parser.parse_args(
            ["--langsmith-dataset", "data", "--langsmith-target", "m:f", "--langsmith-dashboard-evaluators", "Conciseness"]
        )
        validate_args(args)
        assert args.langsmith_dashboard_evaluator_timeout == 900

    def test_dashboard_evaluators_empty_code_evaluators(self):
        """Dashboard-only mode sets evaluators to empty list."""
        parser = _make_parser()
        args = parser.parse_args(
            ["--langsmith-dataset", "data", "--langsmith-target", "m:f", "--langsmith-dashboard-evaluators", "Conciseness"]
        )
        validate_args(args)
        assert args.langsmith_evaluators == []

    def test_no_dashboard_timeout_keeps_argparse_default(self):
        """Without dashboard evaluators, timeout stays at argparse default (900)."""
        parser = _make_parser()
        args = parser.parse_args(["--langsmith-dataset", "data", "--langsmith-target", "m:f", "--langsmith-evaluators", "acc"])
        validate_args(args)
        assert args.langsmith_dashboard_evaluator_timeout == 900

    def test_explicit_timeout_preserved(self):
        """Explicit --langsmith-dashboard-evaluator-timeout is preserved."""
        parser = _make_parser()
        args = parser.parse_args(
            [
                "--langsmith-dataset",
                "data",
                "--langsmith-target",
                "m:f",
                "--langsmith-dashboard-evaluators",
                "Conciseness",
                "--langsmith-dashboard-evaluator-timeout",
                "60",
            ]
        )
        validate_args(args)
        assert args.langsmith_dashboard_evaluator_timeout == 60


# ---------------------------------------------------------------------------
# build_eval_command
# ---------------------------------------------------------------------------


class TestBuildEvalCommand:
    def test_basic_command(self):
        """Builds a correct basic command."""
        parser = _make_parser()
        args = parser.parse_args(
            [
                "--metric",
                "acc",
                "--langsmith-dataset",
                "my-data",
                "--langsmith-target",
                "agent:run",
                "--langsmith-evaluators",
                "acc",
            ]
        )
        validate_args(args)
        cmd = build_eval_command(args)
        assert "bridge.py" in cmd
        assert "--dataset my-data" in cmd
        assert "--target agent:run" in cmd
        assert "--evaluators acc" in cmd
        assert "--metric acc" in cmd

    def test_omits_defaults(self):
        """Doesn't include flags that are at their default values."""
        parser = _make_parser()
        args = parser.parse_args(
            ["--metric", "acc", "--langsmith-dataset", "data", "--langsmith-target", "m:f", "--langsmith-evaluators", "acc"]
        )
        validate_args(args)
        cmd = build_eval_command(args)
        assert "--summary" not in cmd
        assert "--target-adapter" not in cmd
        assert "--metric-function" not in cmd
        assert "--dashboard-evaluators" not in cmd
        assert "--dashboard-evaluator-timeout" not in cmd

    def test_includes_non_default_flags(self):
        """Includes flags that differ from defaults."""
        parser = _make_parser()
        args = parser.parse_args(
            [
                "--metric",
                "acc",
                "--langsmith-dataset",
                "data",
                "--langsmith-target",
                "m:f",
                "--langsmith-evaluators",
                "acc",
                "rel",
                "--langsmith-summary",
                "median",
                "--langsmith-experiment-prefix",
                "test-exp",
                "--langsmith-max-concurrency",
                "4",
                "--langsmith-max-examples",
                "50",
                "--langsmith-target-adapter",
                "langchain",
            ]
        )
        validate_args(args)
        cmd = build_eval_command(args)
        assert "--summary median" in cmd
        assert "--experiment-prefix test-exp" in cmd
        assert "--max-concurrency 4" in cmd
        assert "--max-examples 50" in cmd
        assert "--target-adapter langchain" in cmd
        assert "--evaluators acc rel" in cmd

    def test_dashboard_evaluators_in_command(self):
        """Dashboard evaluators and timeout appear in command."""
        parser = _make_parser()
        args = parser.parse_args(
            [
                "--metric",
                "Conciseness",
                "--langsmith-dataset",
                "data",
                "--langsmith-target",
                "m:f",
                "--langsmith-dashboard-evaluators",
                "Conciseness",
                "--langsmith-dashboard-evaluator-timeout",
                "30",
            ]
        )
        validate_args(args)
        cmd = build_eval_command(args)
        assert "--dashboard-evaluators Conciseness" in cmd
        assert "--dashboard-evaluator-timeout 30" in cmd

    def test_metric_function_in_command(self):
        """Metric function spec appears in command when set."""
        parser = _make_parser()
        args = parser.parse_args(
            [
                "--metric",
                "composite",
                "--langsmith-dataset",
                "data",
                "--langsmith-target",
                "m:f",
                "--langsmith-evaluators",
                "acc",
                "rel",
                "--langsmith-metric-function",
                "scoring:combine",
            ]
        )
        validate_args(args)
        cmd = build_eval_command(args)
        assert "--metric-function scoring:combine" in cmd

    def test_no_evaluators_flag_when_empty(self):
        """When evaluators list is empty (dashboard-only), --evaluators is omitted."""
        parser = _make_parser()
        args = parser.parse_args(
            [
                "--metric",
                "Conciseness",
                "--langsmith-dataset",
                "data",
                "--langsmith-target",
                "m:f",
                "--langsmith-dashboard-evaluators",
                "Conciseness",
            ]
        )
        validate_args(args)
        cmd = build_eval_command(args)
        assert "--evaluators" not in cmd

    def test_splits_in_command(self):
        """--langsmith-splits appears in command when set."""
        parser = _make_parser()
        args = parser.parse_args(
            [
                "--metric",
                "acc",
                "--langsmith-dataset",
                "data",
                "--langsmith-target",
                "m:f",
                "--langsmith-evaluators",
                "acc",
                "--langsmith-splits",
                "train",
                "test",
            ]
        )
        validate_args(args)
        cmd = build_eval_command(args)
        assert "--splits train test" in cmd

    def test_splits_omitted_when_none(self):
        """--splits is not in command when no splits specified."""
        parser = _make_parser()
        args = parser.parse_args(
            ["--metric", "acc", "--langsmith-dataset", "data", "--langsmith-target", "m:f", "--langsmith-evaluators", "acc"]
        )
        validate_args(args)
        cmd = build_eval_command(args)
        assert "--splits" not in cmd


# ---------------------------------------------------------------------------
# Backend dispatch (_load_backend)
# ---------------------------------------------------------------------------


class TestBackendDispatch:
    def test_load_langsmith_backend(self):
        """_load_backend returns the langsmith backend module."""
        from weco.cli import _load_backend

        backend = _load_backend("langsmith")
        assert hasattr(backend, "register_args")
        assert hasattr(backend, "validate_args")
        assert hasattr(backend, "build_eval_command")

    def test_load_unknown_backend_raises(self):
        """_load_backend raises KeyError for unknown backends."""
        from weco.cli import _load_backend

        with pytest.raises(KeyError):
            _load_backend("nonexistent")


# ---------------------------------------------------------------------------
# Wizard trigger logic
# ---------------------------------------------------------------------------


class TestWizardTrigger:
    @patch("weco.integrations.langsmith.backend.sys")
    @patch("weco.integrations.langsmith.wizard.run_wizard")
    def test_wizard_triggered_when_tty(self, mock_wizard, mock_sys):
        """When required args are missing and stdin is TTY, wizard is called."""
        mock_sys.stdin.isatty.return_value = True
        mock_sys.exit.side_effect = SystemExit

        parser = _make_parser()
        args = parser.parse_args(["--metric", "acc"])

        def fake_wizard(a):
            a.langsmith_dataset = "test-data"
            a.langsmith_target = "m:f"

        mock_wizard.side_effect = fake_wizard

        validate_args(args)

        mock_wizard.assert_called_once_with(args)
        assert args.langsmith_dataset == "test-data"
        assert args.langsmith_target == "m:f"

    @patch("weco.integrations.langsmith.backend.sys")
    def test_no_wizard_in_non_tty(self, mock_sys):
        """When required args are missing and stdin is not TTY, exits with error."""
        mock_sys.stdin.isatty.return_value = False
        mock_sys.exit.side_effect = SystemExit

        parser = _make_parser()
        args = parser.parse_args(["--metric", "acc"])

        with pytest.raises(SystemExit):
            validate_args(args)

        mock_sys.exit.assert_called_with(1)

    def test_wizard_not_triggered_when_args_provided(self):
        """When all required args are provided, wizard is not called regardless of TTY."""
        parser = _make_parser()
        args = parser.parse_args(["--metric", "acc", "--langsmith-dataset", "data", "--langsmith-target", "m:f"])

        with patch("weco.integrations.langsmith.wizard.run_wizard") as mock_wizard:
            validate_args(args)
            mock_wizard.assert_not_called()
