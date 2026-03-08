"""Tests for wizard config → args mapping (run_wizard)."""

import argparse
from unittest.mock import MagicMock, patch

from weco.integrations.langsmith.wizard import run_wizard


class TestWizardArgsMapping:
    """Tests that run_wizard correctly maps config_result back to args."""

    def _make_args(self, **overrides):
        """Create a minimal Namespace mimicking CLI-parsed args."""
        defaults = {
            "metric": "accuracy",
            "goal": "maximize",
            "source": "agent.py",
            "sources": None,
            "steps": 100,
            "model": None,
            "log_dir": ".runs",
            "additional_instructions": None,
            "eval_timeout": None,
            "save_logs": False,
            "apply_change": False,
            "require_review": False,
            "langsmith_summary": "mean",
            "langsmith_experiment_prefix": None,
            "langsmith_max_examples": None,
            "langsmith_max_concurrency": None,
            "langsmith_dashboard_evaluator_timeout": 900,
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def _run_wizard_with_config(self, args, config):
        """Run the wizard with mocked infrastructure, injecting config_result."""
        with (
            patch("weco.integrations.langsmith.wizard.open_browser", return_value=True),
            patch("weco.integrations.langsmith.wizard.console"),
            patch("weco.integrations.langsmith.wizard.WizardServer") as mock_srv,
            patch("weco.integrations.langsmith.wizard.threading") as mock_threading,
        ):
            mock_srv.return_value.server_address = ("127.0.0.1", 9999)
            mock_event = MagicMock()
            mock_event.wait.return_value = None
            mock_threading.Event.return_value = mock_event
            mock_threading.Thread.return_value = MagicMock()

            def server_side_effect(addr, handler, *, done_event, config_result, initial_state, html_path):
                config_result.update(config)
                mock = MagicMock()
                mock.server_address = ("127.0.0.1", 9999)
                return mock

            mock_srv.side_effect = server_side_effect

            run_wizard(args)

    def test_single_source_file(self):
        """Single source file sets args.source, clears args.sources."""
        args = self._make_args()
        self._run_wizard_with_config(args, {"dataset": "d", "target": "m:f", "source_files": ["new_agent.py"]})
        assert args.source == "new_agent.py"
        assert args.sources is None

    def test_multiple_source_files(self):
        """Multiple source files sets args.sources, clears args.source."""
        args = self._make_args()
        self._run_wizard_with_config(args, {"dataset": "d", "target": "m:f", "source_files": ["agent.py", "utils.py"]})
        assert args.source is None
        assert args.sources == ["agent.py", "utils.py"]

    def test_core_params_mapped(self):
        """Core run params are mapped from wizard config to args."""
        args = self._make_args()
        self._run_wizard_with_config(
            args,
            {
                "dataset": "d",
                "target": "m:f",
                "metric": "f1",
                "goal": "maximize",
                "steps": 200,
                "model": "gpt-4o",
                "log_dir": ".logs",
                "additional_instructions": "Be concise",
                "eval_timeout": 60,
            },
        )
        assert args.metric == "f1"
        assert args.steps == 200
        assert args.model == "gpt-4o"
        assert args.log_dir == ".logs"
        assert args.additional_instructions == "Be concise"
        assert args.eval_timeout == 60

    def test_boolean_flags_mapped(self):
        """Boolean flags including False values are correctly mapped."""
        args = self._make_args(save_logs=True, apply_change=True)
        self._run_wizard_with_config(
            args, {"dataset": "d", "target": "m:f", "save_logs": False, "apply_change": False, "require_review": True}
        )
        assert args.save_logs is False
        assert args.apply_change is False
        assert args.require_review is True

    def test_langsmith_params_mapped(self):
        """LangSmith-specific params are mapped from wizard config."""
        args = self._make_args()
        self._run_wizard_with_config(
            args,
            {
                "dataset": "d",
                "target": "m:f",
                "langsmith_summary": "median",
                "langsmith_experiment_prefix": "weco-test",
                "langsmith_max_examples": 50,
                "langsmith_max_concurrency": 4,
                "langsmith_dashboard_evaluator_timeout": 120,
            },
        )
        assert args.langsmith_summary == "median"
        assert args.langsmith_experiment_prefix == "weco-test"
        assert args.langsmith_max_examples == 50
        assert args.langsmith_max_concurrency == 4
        assert args.langsmith_dashboard_evaluator_timeout == 120

    def test_required_langsmith_args_mapped(self):
        """Dataset, target, adapter, and evaluators are mapped."""
        args = self._make_args()
        self._run_wizard_with_config(
            args,
            {
                "dataset": "my-dataset",
                "target": "agent:run",
                "adapter": "langchain",
                "code_evaluators": ["acc", "rel"],
                "dashboard_evaluators": ["Conciseness"],
            },
        )
        assert args.langsmith_dataset == "my-dataset"
        assert args.langsmith_target == "agent:run"
        assert args.langsmith_target_adapter == "langchain"
        assert args.langsmith_evaluators == ["acc", "rel"]
        assert args.langsmith_dashboard_evaluators == ["Conciseness"]
