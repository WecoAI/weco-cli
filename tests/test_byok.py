"""Tests to verify API keys are correctly passed through the system and sent to the API."""

import pytest
from unittest.mock import patch, MagicMock
from rich.console import Console

from weco.api import start_optimization_run, evaluate_feedback_then_suggest_next_solution


class TestApiKeysInStartOptimizationRun:
    """Test that api_keys are correctly included in start_optimization_run requests."""

    @pytest.fixture
    def mock_console(self):
        """Create a mock console for testing."""
        return MagicMock(spec=Console)

    @pytest.fixture
    def base_params(self, mock_console):
        """Base parameters for start_optimization_run."""
        return {
            "console": mock_console,
            "source_code": "print('hello')",
            "source_path": "test.py",
            "evaluation_command": "python test.py",
            "metric_name": "accuracy",
            "maximize": True,
            "steps": 10,
            "code_generator_config": {"model": "o4-mini"},
            "evaluator_config": {"model": "o4-mini"},
            "search_policy_config": {"num_drafts": 2},
        }

    @patch("weco.api.requests.post")
    def test_api_keys_included_in_request(self, mock_post, base_params):
        """Test that api_keys are included in the request JSON when provided."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "run_id": "test-run-id",
            "solution_id": "test-solution-id",
            "code": "print('hello')",
            "plan": "test plan",
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        api_keys = {"openai": "sk-test-key", "anthropic": "sk-ant-test"}
        start_optimization_run(**base_params, api_keys=api_keys)

        # Verify the request was made with api_keys in the JSON payload
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        request_json = call_kwargs.kwargs["json"]
        assert "api_keys" in request_json
        assert request_json["api_keys"] == {"openai": "sk-test-key", "anthropic": "sk-ant-test"}

    @patch("weco.api.requests.post")
    def test_api_keys_not_included_when_none(self, mock_post, base_params):
        """Test that api_keys field is not included when api_keys is None."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "run_id": "test-run-id",
            "solution_id": "test-solution-id",
            "code": "print('hello')",
            "plan": "test plan",
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        start_optimization_run(**base_params, api_keys=None)

        # Verify the request was made without api_keys
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        request_json = call_kwargs.kwargs["json"]
        assert "api_keys" not in request_json

    @patch("weco.api.requests.post")
    def test_api_keys_not_included_when_empty_dict(self, mock_post, base_params):
        """Test that api_keys field is not included when api_keys is an empty dict."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "run_id": "test-run-id",
            "solution_id": "test-solution-id",
            "code": "print('hello')",
            "plan": "test plan",
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        # Empty dict is falsy, so api_keys should not be included
        start_optimization_run(**base_params, api_keys={})

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        request_json = call_kwargs.kwargs["json"]
        assert "api_keys" not in request_json


class TestApiKeysInEvaluateFeedbackThenSuggest:
    """Test that api_keys are correctly included in evaluate_feedback_then_suggest_next_solution requests."""

    @pytest.fixture
    def mock_console(self):
        """Create a mock console for testing."""
        return MagicMock(spec=Console)

    @patch("weco.api.requests.post")
    def test_api_keys_included_in_suggest_request(self, mock_post, mock_console):
        """Test that api_keys are included in the suggest request JSON when provided."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "run_id": "test-run-id",
            "solution_id": "new-solution-id",
            "code": "print('improved')",
            "plan": "improvement plan",
            "is_done": False,
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        api_keys = {"openai": "sk-test-key"}
        evaluate_feedback_then_suggest_next_solution(
            console=mock_console,
            run_id="test-run-id",
            step=1,
            execution_output="accuracy: 0.95",
            auth_headers={"Authorization": "Bearer test-token"},
            api_keys=api_keys,
        )

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        request_json = call_kwargs.kwargs["json"]
        assert "api_keys" in request_json
        assert request_json["api_keys"] == {"openai": "sk-test-key"}

    @patch("weco.api.requests.post")
    def test_api_keys_not_included_in_suggest_when_none(self, mock_post, mock_console):
        """Test that api_keys field is not included in suggest request when api_keys is None."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "run_id": "test-run-id",
            "solution_id": "new-solution-id",
            "code": "print('improved')",
            "plan": "improvement plan",
            "is_done": False,
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        evaluate_feedback_then_suggest_next_solution(
            console=mock_console,
            run_id="test-run-id",
            step=1,
            execution_output="accuracy: 0.95",
            auth_headers={"Authorization": "Bearer test-token"},
            api_keys=None,
        )

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        request_json = call_kwargs.kwargs["json"]
        assert "api_keys" not in request_json

    @patch("weco.api.requests.post")
    def test_api_keys_not_included_in_suggest_when_empty_dict(self, mock_post, mock_console):
        """Test that api_keys field is not included in suggest request when api_keys is None."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "run_id": "test-run-id",
            "solution_id": "new-solution-id",
            "code": "print('improved')",
            "plan": "improvement plan",
            "is_done": False,
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        evaluate_feedback_then_suggest_next_solution(
            console=mock_console,
            run_id="test-run-id",
            step=1,
            execution_output="accuracy: 0.95",
            auth_headers={"Authorization": "Bearer test-token"},
            api_keys={},
        )

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        request_json = call_kwargs.kwargs["json"]
        assert "api_keys" not in request_json
