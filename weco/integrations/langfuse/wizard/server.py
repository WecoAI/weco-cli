"""HTTP server and API handlers for the LangFuse setup wizard."""

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
            from langfuse import Langfuse

            self._client = Langfuse(
                public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"),
                secret_key=os.environ.get("LANGFUSE_SECRET_KEY"),
                host=os.environ.get("LANGFUSE_BASE_URL"),
            )
        return self._client

    def reset_client(self):
        if self._client is not None:
            try:
                self._client.flush()
            except Exception:
                pass
        self._client = None

    def resolve_source_files(self):
        """Resolve source files from initial state."""
        if self.initial_state.get("sources"):
            return self.initial_state["sources"]
        elif self.initial_state.get("source"):
            return [self.initial_state["source"]]
        return []


class WizardHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the LangFuse setup wizard.

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
        (re.compile(r"^/api/discover$"), "handle_discover"),
        (re.compile(r"^/api/file-tree$"), "handle_file_tree"),
    ]

    POST_ROUTES: list[tuple[re.Pattern, str]] = [
        (re.compile(r"^/api/set-keys$"), "handle_set_keys"),
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
        secret_key_set = bool(os.environ.get("LANGFUSE_SECRET_KEY"))
        public_key_set = bool(os.environ.get("LANGFUSE_PUBLIC_KEY"))
        connected = False

        if secret_key_set and public_key_set:
            try:
                self.server.reset_client()
                connected = self.server.client.auth_check()
            except Exception:
                connected = False

        state = self.server.initial_state
        self.send_json(
            {
                "secret_key_set": secret_key_set,
                "public_key_set": public_key_set,
                "connected": connected,
                "host": os.environ.get("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"),
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
                # LangFuse params
                "langfuse_summary": state.get("langfuse_summary", "mean"),
                "langfuse_experiment_name": state.get("langfuse_experiment_name", None),
                "langfuse_max_concurrency": state.get("langfuse_max_concurrency", None),
                "langfuse_managed_evaluator_timeout": state.get("langfuse_managed_evaluator_timeout", 900),
            }
        )

    def handle_set_keys(self):
        body = self.read_body()
        secret_key = body.get("secret_key", "").strip()
        public_key = body.get("public_key", "").strip()
        host = body.get("host", "").strip()

        if not secret_key or not public_key:
            self.send_error_json("Both public and secret keys are required")
            return

        os.environ["LANGFUSE_SECRET_KEY"] = secret_key
        os.environ["LANGFUSE_PUBLIC_KEY"] = public_key
        if host:
            os.environ["LANGFUSE_BASE_URL"] = host
        self.server.reset_client()

        try:
            connected = self.server.client.auth_check()
            if connected:
                self.send_json({"connected": True, "error": None})
                return
            raise RuntimeError("auth_check returned False")
        except Exception as e:
            os.environ.pop("LANGFUSE_SECRET_KEY", None)
            os.environ.pop("LANGFUSE_PUBLIC_KEY", None)
            self.server.reset_client()
            error_type = type(e).__name__
            detail = ""
            if "401" in str(e) or "403" in str(e):
                detail = "API keys were rejected (check they are valid and for the correct project)."
            elif "404" in str(e):
                detail = "LangFuse API endpoint not found (check LANGFUSE_BASE_URL)."
            elif "ConnectionError" in error_type or "timeout" in str(e).lower():
                detail = "Could not reach LangFuse API (check your network connection and host URL)."
            else:
                detail = "Check that your API keys are valid."
            self.send_json({"connected": False, "error": f"Connection failed: {detail}"})

    def handle_list_datasets(self):
        try:
            client = self.server.client
            # The Langfuse Python SDK uses client.api.datasets.list() or client.get_datasets()
            # Try the REST API approach which is more reliable across SDK versions
            response = client.api.datasets.list()
            datasets = response.data if hasattr(response, "data") else response
            result = []
            for ds in datasets:
                result.append(
                    {
                        "name": ds.name,
                        "description": getattr(ds, "description", "") or "",
                        "created_at": ds.created_at.isoformat() if hasattr(ds, "created_at") and ds.created_at else None,
                    }
                )
            result.sort(key=lambda d: d["created_at"] or "", reverse=True)
            self.send_json({"datasets": result})
        except Exception as e:
            self.send_error_json(f"Failed to fetch datasets: {e}", 500)

    def handle_list_examples(self, name: str):
        try:
            dataset = self.server.client.get_dataset(name)
            result = []
            items = dataset.items if hasattr(dataset, "items") and dataset.items else []
            for item in items[:5]:
                result.append(
                    {"input": getattr(item, "input", {}) or {}, "expected_output": getattr(item, "expected_output", {}) or {}}
                )
            self.send_json({"examples": result})
        except Exception as e:
            self.send_error_json(f"Failed to fetch examples: {e}", 500)

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
