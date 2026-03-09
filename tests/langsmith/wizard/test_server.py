"""Tests for the LangSmith wizard HTTP server."""

import json
import threading
from datetime import datetime, timezone
from http.client import HTTPConnection
from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from weco.integrations.langsmith.wizard.server import WizardHandler, WizardServer


# -- Request helpers --


def get_json(conn, path):
    conn.request("GET", path)
    resp = conn.getresponse()
    return resp, json.loads(resp.read())


def post_json(conn, path, body):
    conn.request("POST", path, body=json.dumps(body), headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    return resp, json.loads(resp.read())


# -- Fixture --


@pytest.fixture
def wizard_server(tmp_path):
    """Start a wizard server on a random port and yield (connection, done_event, config_result)."""
    done_event = threading.Event()
    config_result = {}
    initial_state = {
        "api_key_set": True,
        "metric": "accuracy",
        "goal": "maximize",
        "source": "agent.py",
        "sources": None,
        "steps": 50,
        "model": "o4-mini",
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

    html_file = tmp_path / "page.html"
    html_file.write_text("<html><body>Test</body></html>")

    server = WizardServer(
        ("127.0.0.1", 0),
        WizardHandler,
        done_event=done_event,
        config_result=config_result,
        initial_state=initial_state,
        html_path=html_file,
    )
    port = server.server_address[1]

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    conn = HTTPConnection("127.0.0.1", port)

    yield conn, done_event, config_result

    server.shutdown()
    thread.join(timeout=2)
    conn.close()


# -- Tests --


class TestWizardServer:
    def test_serves_html_page(self, wizard_server):
        """GET / returns the HTML page."""
        conn, _, _ = wizard_server
        conn.request("GET", "/")
        resp = conn.getresponse()
        assert resp.status == 200
        assert "text/html" in resp.getheader("Content-Type")
        assert "<html>" in resp.read().decode()

    @patch.object(WizardServer, "client", new_callable=PropertyMock)
    @patch("weco.integrations.langsmith.wizard.server.os")
    def test_status_endpoint(self, mock_os, mock_client_prop, wizard_server):
        """GET /api/status returns initial state including all run params."""
        mock_os.environ.get.return_value = "ls-key"
        mock_client = MagicMock()
        mock_client.list_datasets.return_value = [object()]
        mock_client_prop.return_value = mock_client

        conn, _, _ = wizard_server
        resp, data = get_json(conn, "/api/status")
        assert resp.status == 200
        assert data["api_key_set"] is True
        assert data["connected"] is True
        assert data["metric"] == "accuracy"
        assert data["goal"] == "maximize"
        assert data["source_files"] == ["agent.py"]
        assert data["steps"] == 50
        assert data["model"] == "o4-mini"
        assert data["log_dir"] == ".runs"
        assert data["additional_instructions"] is None
        assert data["eval_timeout"] is None
        assert data["save_logs"] is False
        assert data["apply_change"] is False
        assert data["require_review"] is False
        assert data["langsmith_summary"] == "mean"
        assert data["langsmith_experiment_prefix"] is None
        assert data["langsmith_max_examples"] is None
        assert data["langsmith_max_concurrency"] is None
        assert data["langsmith_dashboard_evaluator_timeout"] == 900
        mock_client.list_datasets.assert_called_once_with(limit=1)

    @patch("weco.integrations.langsmith.wizard.server.os")
    def test_status_no_api_key(self, mock_os, wizard_server):
        """GET /api/status reflects missing API key."""
        mock_os.environ.get.return_value = None
        conn, _, _ = wizard_server
        resp, data = get_json(conn, "/api/status")
        assert resp.status == 200
        assert data["api_key_set"] is False
        assert data["connected"] is False

    def test_configure_success(self, wizard_server):
        """POST /api/configure stores config and sets done_event."""
        conn, done_event, config_result = wizard_server
        resp, data = post_json(
            conn,
            "/api/configure",
            {
                "dataset": "my-dataset",
                "target": "agent:run",
                "adapter": "raw",
                "code_evaluators": ["accuracy"],
                "dashboard_evaluators": [],
            },
        )
        assert resp.status == 200
        assert data["ok"] is True
        assert done_event.is_set()
        assert config_result["dataset"] == "my-dataset"
        assert config_result["target"] == "agent:run"

    @pytest.mark.parametrize("body", [{"target": "m:f"}, {"dataset": "data"}])
    def test_configure_rejects_incomplete(self, wizard_server, body):
        """POST /api/configure rejects missing dataset or target."""
        conn, done_event, _ = wizard_server
        resp, data = post_json(conn, "/api/configure", body)
        assert resp.status == 400
        assert "error" in data
        assert not done_event.is_set()

    def test_404_on_unknown_route(self, wizard_server):
        """Unknown routes return 404."""
        conn, _, _ = wizard_server
        resp, _ = get_json(conn, "/api/nonexistent")
        assert resp.status == 404

    @patch("weco.integrations.langsmith.wizard.server.os")
    def test_set_key_empty_rejected(self, mock_os, wizard_server):
        """POST /api/set-key with empty key is rejected."""
        conn, _, _ = wizard_server
        resp, data = post_json(conn, "/api/set-key", {"key": ""})
        assert resp.status == 400
        assert "error" in data

    @patch.object(WizardServer, "client", new_callable=PropertyMock)
    @patch("weco.integrations.langsmith.wizard.server.os")
    def test_set_key_success_sets_connected(self, mock_os, mock_client_prop, wizard_server):
        """POST /api/set-key stores key and returns connected=True."""
        mock_client = MagicMock()
        mock_client.list_datasets.return_value = [object()]
        mock_client_prop.return_value = mock_client

        conn, _, _ = wizard_server
        resp, data = post_json(conn, "/api/set-key", {"key": "ls-key"})
        assert resp.status == 200
        assert data == {"connected": True, "error": None}
        mock_os.environ.__setitem__.assert_called_once_with("LANGCHAIN_API_KEY", "ls-key")
        mock_client.list_datasets.assert_called_once_with(limit=1)

    @patch.object(WizardServer, "client", new_callable=PropertyMock)
    @patch("weco.integrations.langsmith.wizard.server.os")
    def test_set_key_failure_clears_key(self, mock_os, mock_client_prop, wizard_server):
        """POST /api/set-key clears key when connection check fails."""
        mock_client = MagicMock()
        mock_client.list_datasets.side_effect = RuntimeError("bad key")
        mock_client_prop.return_value = mock_client

        conn, _, _ = wizard_server
        resp, data = post_json(conn, "/api/set-key", {"key": "bad-key"})
        assert resp.status == 200
        assert data["connected"] is False
        assert data["error"] == "Connection failed. Check that your API key is valid."
        mock_os.environ.pop.assert_called_once_with("LANGCHAIN_API_KEY", None)

    @patch.object(WizardServer, "client", new_callable=PropertyMock)
    def test_list_datasets_filters_and_sorts(self, mock_client_prop, wizard_server):
        """GET /api/datasets excludes evaluator datasets and sorts newest-first."""
        mock_client = MagicMock()
        mock_client.list_datasets.return_value = [
            SimpleNamespace(
                name="Evaluator: internal",
                description="skip",
                example_count=10,
                created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ),
            SimpleNamespace(
                name="older", description="old", example_count=1, created_at=datetime(2025, 1, 1, tzinfo=timezone.utc)
            ),
            SimpleNamespace(
                name="newer", description="new", example_count=2, created_at=datetime(2026, 2, 1, tzinfo=timezone.utc)
            ),
        ]
        mock_client_prop.return_value = mock_client

        conn, _, _ = wizard_server
        resp, data = get_json(conn, "/api/datasets")
        assert resp.status == 200
        assert [d["name"] for d in data["datasets"]] == ["newer", "older"]

    @patch.object(WizardServer, "client", new_callable=PropertyMock)
    def test_list_examples_returns_inputs_outputs(self, mock_client_prop, wizard_server):
        """GET /api/datasets/<name>/examples returns example payloads."""
        mock_client = MagicMock()
        mock_client.list_examples.return_value = [
            SimpleNamespace(inputs={"q": "hi"}, outputs={"a": "hello"}),
            SimpleNamespace(inputs=None, outputs=None),
        ]
        mock_client_prop.return_value = mock_client

        conn, _, _ = wizard_server
        resp, data = get_json(conn, "/api/datasets/my-dataset/examples")
        assert resp.status == 200
        assert data["examples"] == [{"inputs": {"q": "hi"}, "outputs": {"a": "hello"}}, {"inputs": {}, "outputs": {}}]
        mock_client.list_examples.assert_called_once_with(dataset_name="my-dataset", limit=5)

    @patch.object(WizardServer, "client", new_callable=PropertyMock)
    def test_list_splits(self, mock_client_prop, wizard_server):
        """GET /api/datasets/<name>/splits returns available splits."""
        mock_client = MagicMock()
        mock_client.list_dataset_splits.return_value = ["train", "test"]
        mock_client_prop.return_value = mock_client

        conn, _, _ = wizard_server
        resp, data = get_json(conn, "/api/datasets/my-dataset/splits")
        assert resp.status == 200
        assert data["splits"] == ["train", "test"]
        mock_client.list_dataset_splits.assert_called_once_with(dataset_name="my-dataset")

    def test_file_tree_lists_cwd(self, wizard_server, tmp_path, monkeypatch):
        """GET /api/file-tree returns directory entries."""
        (tmp_path / "agent.py").write_text("pass")
        (tmp_path / "utils").mkdir()
        monkeypatch.chdir(tmp_path)

        conn, _, _ = wizard_server
        resp, data = get_json(conn, "/api/file-tree?path=.")
        assert resp.status == 200
        assert "entries" in data
        names = [e["name"] for e in data["entries"]]
        assert "agent.py" in names

    def test_file_tree_rejects_traversal(self, wizard_server, tmp_path, monkeypatch):
        """GET /api/file-tree rejects paths outside project root."""
        monkeypatch.chdir(tmp_path)

        conn, _, _ = wizard_server
        resp, data = get_json(conn, "/api/file-tree?path=../../etc")
        assert resp.status == 403
        assert "error" in data

    def test_file_tree_invalid_dir(self, wizard_server, tmp_path, monkeypatch):
        """GET /api/file-tree returns 400 for non-directory."""
        (tmp_path / "file.py").write_text("pass")
        monkeypatch.chdir(tmp_path)

        conn, _, _ = wizard_server
        resp, data = get_json(conn, "/api/file-tree?path=file.py")
        assert resp.status == 400
        assert "error" in data

    def test_discover_get_uses_initial_source_files(self, wizard_server):
        """GET /api/discover returns targets/evaluators from initial source files."""
        conn, _, _ = wizard_server
        resp, data = get_json(conn, "/api/discover")
        assert resp.status == 200
        assert "targets" in data
        assert "evaluators" in data

    def test_discover_post_with_files(self, wizard_server, tmp_path):
        """POST /api/discover returns discovered functions from provided files."""
        src = tmp_path / "mymod.py"
        src.write_text("def run(inputs):\n    pass\n")

        conn, _, _ = wizard_server
        resp, data = post_json(conn, "/api/discover", {"source_files": [str(src)]})
        assert resp.status == 200
        assert len(data["targets"]) == 1
        assert data["targets"][0]["name"] == "run"

    @pytest.mark.parametrize("body", [{"source_files": []}, {}])
    def test_discover_post_rejects_invalid(self, wizard_server, body):
        """POST /api/discover rejects empty or missing source_files."""
        conn, _, _ = wizard_server
        resp, data = post_json(conn, "/api/discover", body)
        assert resp.status == 400
        assert "error" in data
