"""Tests to verify API keys are correctly passed through the system and sent to the API."""

import pytest
from unittest.mock import patch, MagicMock

from weco.core.api import WecoClient


class TestApiKeysInStartRun:
    """Test that api_keys are correctly included in WecoClient.start_run requests."""

    @pytest.fixture
    def base_params(self):
        """Base parameters for start_run."""
        return {
            "source_code": {"test.py": "print('hello')"},
            "source_path": "test.py",
            "evaluation_command": "python test.py",
            "metric_name": "accuracy",
            "maximize": True,
            "steps": 10,
            "code_generator_config": {"model": "o4-mini"},
            "evaluator_config": {"model": "o4-mini"},
            "search_policy_config": {"num_drafts": 2},
        }

    def _make_client(self, mock_session_cls):
        """Create a WecoClient with a mocked session."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "run_id": "test-run-id",
            "solution_id": "test-solution-id",
            "code": "print('hello')",
            "plan": "test plan",
        }
        mock_response.raise_for_status = MagicMock()
        mock_session.post.return_value = mock_response
        mock_session_cls.return_value = mock_session
        client = WecoClient({"Authorization": "Bearer test-token"})
        return client, mock_session

    @patch("weco.core.api.requests.Session")
    def test_api_keys_included_in_request(self, mock_session_cls, base_params):
        """Test that api_keys are included in the request JSON when provided."""
        client, mock_session = self._make_client(mock_session_cls)

        api_keys = {"openai": "sk-test-key", "anthropic": "sk-ant-test"}
        client.start_run(**base_params, api_keys=api_keys)

        mock_session.post.assert_called_once()
        call_kwargs = mock_session.post.call_args
        request_json = call_kwargs.kwargs["json"]
        assert "api_keys" in request_json
        assert request_json["api_keys"] == {"openai": "sk-test-key", "anthropic": "sk-ant-test"}

    @patch("weco.core.api.requests.Session")
    def test_api_keys_not_included_when_none(self, mock_session_cls, base_params):
        """Test that api_keys field is not included when api_keys is None."""
        client, mock_session = self._make_client(mock_session_cls)

        client.start_run(**base_params, api_keys=None)

        mock_session.post.assert_called_once()
        call_kwargs = mock_session.post.call_args
        request_json = call_kwargs.kwargs["json"]
        assert "api_keys" not in request_json

    @patch("weco.core.api.requests.Session")
    def test_api_keys_not_included_when_empty_dict(self, mock_session_cls, base_params):
        """Test that api_keys field is not included when api_keys is an empty dict."""
        client, mock_session = self._make_client(mock_session_cls)

        # Empty dict is falsy, so api_keys should not be included
        client.start_run(**base_params, api_keys={})

        mock_session.post.assert_called_once()
        call_kwargs = mock_session.post.call_args
        request_json = call_kwargs.kwargs["json"]
        assert "api_keys" not in request_json


class TestApiKeysInSuggest:
    """Test that api_keys are correctly included in WecoClient.suggest requests."""

    def _make_client(self, mock_session_cls):
        """Create a WecoClient with a mocked session."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "run_id": "test-run-id",
            "solution_id": "new-solution-id",
            "code": "print('improved')",
            "plan": "improvement plan",
            "is_done": False,
        }
        mock_response.raise_for_status = MagicMock()
        mock_session.post.return_value = mock_response
        mock_session_cls.return_value = mock_session
        client = WecoClient({"Authorization": "Bearer test-token"})
        return client, mock_session

    @patch("weco.core.api.requests.Session")
    def test_api_keys_included_in_suggest_request(self, mock_session_cls):
        """Test that api_keys are included in the suggest request JSON when provided."""
        client, mock_session = self._make_client(mock_session_cls)

        api_keys = {"openai": "sk-test-key"}
        client.suggest(
            "test-run-id",
            execution_output="accuracy: 0.95",
            step=1,
            api_keys=api_keys,
        )

        mock_session.post.assert_called_once()
        call_kwargs = mock_session.post.call_args
        request_json = call_kwargs.kwargs["json"]
        assert "api_keys" in request_json
        assert request_json["api_keys"] == {"openai": "sk-test-key"}

    @patch("weco.core.api.requests.Session")
    def test_api_keys_not_included_in_suggest_when_none(self, mock_session_cls):
        """Test that api_keys field is not included in suggest request when api_keys is None."""
        client, mock_session = self._make_client(mock_session_cls)

        client.suggest(
            "test-run-id",
            execution_output="accuracy: 0.95",
            step=1,
            api_keys=None,
        )

        mock_session.post.assert_called_once()
        call_kwargs = mock_session.post.call_args
        request_json = call_kwargs.kwargs["json"]
        assert "api_keys" not in request_json

    @patch("weco.core.api.requests.Session")
    def test_api_keys_not_included_in_suggest_when_empty_dict(self, mock_session_cls):
        """Test that api_keys field is not included in suggest request when api_keys is empty."""
        client, mock_session = self._make_client(mock_session_cls)

        client.suggest(
            "test-run-id",
            execution_output="accuracy: 0.95",
            step=1,
            api_keys={},
        )

        mock_session.post.assert_called_once()
        call_kwargs = mock_session.post.call_args
        request_json = call_kwargs.kwargs["json"]
        assert "api_keys" not in request_json
