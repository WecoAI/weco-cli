"""HTTP server and API handlers for the LangSmith setup wizard."""

import json
import os
import re
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from .utils import discover_functions, list_directory


class WizardServer(HTTPServer):
    """HTTP server that holds shared wizard state.

    Replaces the closure-based make_handler() pattern with explicit state
    on the server instance, accessible from handlers via ``self.server``.
    """

    def __init__(self, addr, handler_class, *, done_event, config_result, initial_state, html_path):
        super().__init__(addr, handler_class)
        self.done_event = done_event
        self.config_result = config_result
        self.initial_state = initial_state
        self.html_path = html_path
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from langsmith import Client

            self._client = Client()
        return self._client

    def reset_client(self):
        self._client = None

    def resolve_source_files(self):
        """Resolve source files from initial state."""
        if self.initial_state.get("sources"):
            return self.initial_state["sources"]
        elif self.initial_state.get("source"):
            return [self.initial_state["source"]]
        return []


class WizardHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the LangSmith setup wizard.

    Accesses shared state via ``self.server`` (a WizardServer instance).
    """

    server: WizardServer

    def log_message(self, format, *args):
        pass

    # -- Response helpers --

    def send_json(self, data: dict, status: int = 200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_html(self, content: str):
        body = content.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, message: str, status: int = 400):
        self.send_json({"error": message}, status=status)

    def read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        return json.loads(raw) if raw else {}

    # -- Routing --

    GET_ROUTES: list[tuple[re.Pattern, str]] = [
        (re.compile(r"^/$"), "handle_page"),
        (re.compile(r"^/api/status$"), "handle_status"),
        (re.compile(r"^/api/datasets$"), "handle_list_datasets"),
        (re.compile(r"^/api/datasets/(?P<name>[^/]+)/examples$"), "handle_list_examples"),
        (re.compile(r"^/api/datasets/(?P<name>[^/]+)/splits$"), "handle_list_splits"),
        (re.compile(r"^/api/discover$"), "handle_discover"),
        (re.compile(r"^/api/file-tree$"), "handle_file_tree"),
    ]

    POST_ROUTES: list[tuple[re.Pattern, str]] = [
        (re.compile(r"^/api/set-key$"), "handle_set_key"),
        (re.compile(r"^/api/configure$"), "handle_configure"),
        (re.compile(r"^/api/discover$"), "handle_discover_post"),
    ]

    def dispatch(self, routes):
        path = self.path.split("?")[0]
        for pattern, handler_name in routes:
            match = pattern.match(path)
            if match:
                kwargs = {k: unquote(v) for k, v in match.groupdict().items()}
                getattr(self, handler_name)(**kwargs)
                return
        self.send_error_json("Not found", 404)

    def do_GET(self):
        self.dispatch(self.GET_ROUTES)

    def do_POST(self):
        self.dispatch(self.POST_ROUTES)

    # -- Route handlers --

    def handle_page(self):
        try:
            content = self.server.html_path.read_text(encoding="utf-8")
            self.send_html(content)
        except FileNotFoundError:
            self.send_error_json("Wizard page not found", 500)

    def handle_status(self):
        api_key_set = bool(os.environ.get("LANGCHAIN_API_KEY"))
        connected = False

        if api_key_set:
            try:
                self.server.reset_client()
                list(self.server.client.list_datasets(limit=1))
                connected = True
            except Exception:
                pass

        state = self.server.initial_state
        self.send_json(
            {
                "api_key_set": api_key_set,
                "connected": connected,
                # Core params
                "metric": state.get("metric", ""),
                "goal": state.get("goal", ""),
                "source_files": self.server.resolve_source_files(),
                "steps": state.get("steps", 100),
                "model": state.get("model", None),
                "log_dir": state.get("log_dir", ".runs"),
                "additional_instructions": state.get("additional_instructions", None),
                "eval_timeout": state.get("eval_timeout", None),
                "save_logs": state.get("save_logs", False),
                "apply_change": state.get("apply_change", False),
                "require_review": state.get("require_review", False),
                # LangSmith params
                "langsmith_summary": state.get("langsmith_summary", "mean"),
                "langsmith_experiment_prefix": state.get("langsmith_experiment_prefix", None),
                "langsmith_max_examples": state.get("langsmith_max_examples", None),
                "langsmith_max_concurrency": state.get("langsmith_max_concurrency", None),
                "langsmith_dashboard_evaluator_timeout": state.get("langsmith_dashboard_evaluator_timeout", 900),
            }
        )

    def handle_set_key(self):
        body = self.read_body()
        key = body.get("key", "").strip()

        if not key:
            self.send_error_json("API key cannot be empty")
            return

        os.environ["LANGCHAIN_API_KEY"] = key
        self.server.reset_client()

        try:
            list(self.server.client.list_datasets(limit=1))
            self.send_json({"connected": True, "error": None})
        except Exception:
            os.environ.pop("LANGCHAIN_API_KEY", None)
            self.server.reset_client()
            self.send_json({"connected": False, "error": "Connection failed. Check that your API key is valid."})

    def handle_list_datasets(self):
        try:
            datasets = list(self.server.client.list_datasets())
            result = []
            for ds in datasets:
                result.append(
                    {
                        "name": ds.name,
                        "description": ds.description or "",
                        "example_count": ds.example_count or 0,
                        "created_at": ds.created_at.isoformat() if ds.created_at else None,
                    }
                )
            result = [d for d in result if not d["name"].startswith("Evaluator: ")]
            result.sort(key=lambda d: d["created_at"] or "", reverse=True)
            self.send_json({"datasets": result})
        except Exception as e:
            self.send_error_json(f"Failed to fetch datasets: {e}", 500)

    def handle_list_examples(self, name: str):
        try:
            examples = list(self.server.client.list_examples(dataset_name=name, limit=5))
            result = []
            for ex in examples:
                result.append({"inputs": ex.inputs or {}, "outputs": ex.outputs or {}})
            self.send_json({"examples": result})
        except Exception as e:
            self.send_error_json(f"Failed to fetch examples: {e}", 500)

    def handle_list_splits(self, name: str):
        try:
            splits = self.server.client.list_dataset_splits(dataset_name=name)
            self.send_json({"splits": splits})
        except Exception as e:
            self.send_error_json(f"Failed to fetch splits: {e}", 500)

    def handle_discover(self):
        source_files = self.server.resolve_source_files()
        try:
            result = discover_functions(source_files)
            self.send_json(result)
        except Exception as e:
            self.send_error_json(f"Discovery failed: {e}", 500)

    def handle_discover_post(self):
        body = self.read_body()
        source_files = body.get("source_files", [])

        if not source_files:
            self.send_error_json("source_files is required")
            return

        try:
            result = discover_functions(source_files)
            self.send_json(result)
        except Exception as e:
            self.send_error_json(f"Discovery failed: {e}", 500)

    def handle_file_tree(self):
        query = parse_qs(urlparse(self.path).query)
        rel_path = query.get("path", ["."])[0]

        project_root = Path.cwd().resolve()
        target = (project_root / rel_path).resolve()

        # Security: must be within project root
        if not str(target).startswith(str(project_root)):
            self.send_error_json("Access denied: path outside project directory", 403)
            return

        if not target.is_dir():
            self.send_error_json("Not a directory", 400)
            return

        try:
            entries = list_directory(target, project_root)
            self.send_json({"entries": entries})
        except PermissionError:
            self.send_error_json("Permission denied", 403)

    def handle_configure(self):
        body = self.read_body()

        if not body.get("dataset"):
            self.send_error_json("Dataset is required")
            return
        if not body.get("target"):
            self.send_error_json("Target function is required")
            return

        self.server.config_result.clear()
        self.server.config_result.update(body)
        self.send_json({"ok": True})
        self.server.done_event.set()
