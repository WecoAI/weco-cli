"""HTTP client for the Weco API."""

import sys
from dataclasses import dataclass
from typing import Any, Optional

import requests

from .. import __base_url__, __pkg_version__
from .constants import TRUNCATION_THRESHOLD, TRUNCATION_KEEP_LENGTH


def _truncate_output(output: str) -> str:
    """Truncate long execution output to a manageable size for the API."""
    if len(output) <= TRUNCATION_THRESHOLD:
        return output
    first = output[:TRUNCATION_KEEP_LENGTH]
    last = output[-TRUNCATION_KEEP_LENGTH:]
    truncated_len = len(output) - 2 * TRUNCATION_KEEP_LENGTH
    if truncated_len <= 0:
        return output
    return f"{first}\n ... [{truncated_len} characters truncated] ... \n{last}"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RunSummary:
    """Brief run summary from execution task response."""

    id: str
    status: str
    name: Optional[str] = None
    require_review: bool = False


@dataclass
class ExecutionTasksResult:
    """Result from get_execution_tasks containing tasks and run info."""

    tasks: list
    run: Optional[RunSummary] = None


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def handle_api_error(e: requests.exceptions.HTTPError, console) -> None:
    """Extract and display error messages from API responses in a structured format."""
    status = getattr(e.response, "status_code", None)
    try:
        payload = e.response.json()
        detail = payload.get("detail", payload)
    except (ValueError, AttributeError):
        detail = getattr(e.response, "text", "") or f"HTTP {status} Error"

    def _render(detail_obj: Any) -> None:
        if isinstance(detail_obj, str):
            console.print(f"[bold red]{detail_obj}[/]")
        elif isinstance(detail_obj, dict):
            message_keys = ("message", "error", "msg", "detail")
            message = next((detail_obj.get(key) for key in message_keys if detail_obj.get(key)), None)
            suggestion = detail_obj.get("suggestion")
            if message:
                console.print(f"[bold red]{message}[/]")
            else:
                console.print(f"[bold red]HTTP {status} Error[/]")
            if suggestion:
                console.print(f"[yellow]{suggestion}[/]")
            extras = {
                k: v
                for k, v in detail_obj.items()
                if k not in {"message", "error", "msg", "detail", "suggestion"} and v not in (None, "")
            }
            for key, value in extras.items():
                console.print(f"[dim]{key}: {value}[/]")
        elif isinstance(detail_obj, list) and detail_obj:
            _render(detail_obj[0])
            for extra in detail_obj[1:]:
                console.print(f"[yellow]{extra}[/]")
        else:
            console.print(f"[bold red]{detail_obj or f'HTTP {status} Error'}[/]")

    _render(detail)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class WecoClient:
    """HTTP client for the Weco API.

    Holds a ``requests.Session`` with pre-configured base URL and auth so
    that callers never need to construct URLs or pass headers manually.
    """

    def __init__(self, auth_headers: dict, *, base_url: str = __base_url__) -> None:
        self._base_url = base_url.rstrip("/")
        self._session = requests.Session()
        self._session.headers.update(auth_headers)

    # -- helpers --

    def _url(self, path: str) -> str:
        return f"{self._base_url}{path}"

    def _get(self, path: str, *, params: dict | None = None, timeout=(5, 10)) -> requests.Response:
        return self._session.get(self._url(path), params=params, timeout=timeout)

    def _post(self, path: str, *, json: dict | None = None, timeout=(5, 30)) -> requests.Response:
        return self._session.post(self._url(path), json=json, timeout=timeout)

    def _put(self, path: str, *, timeout=(5, 10)) -> requests.Response:
        return self._session.put(self._url(path), timeout=timeout)

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------

    def start_run(
        self,
        *,
        source_code: dict[str, str],
        source_path: str | None,
        evaluation_command: str,
        metric_name: str,
        maximize: bool,
        steps: int,
        code_generator_config: dict,
        evaluator_config: dict,
        search_policy_config: dict,
        additional_instructions: str | None = None,
        eval_timeout: int | None = None,
        save_logs: bool = False,
        log_dir: str = ".runs",
        api_keys: dict[str, str] | None = None,
        require_review: bool = False,
        installation_id: str | None = None,
        invocation_id: str | None = None,
        invoked_via: str | None = None,
    ) -> dict | None:
        """POST /runs/ — start a new optimization run."""
        metadata = {"client_name": invoked_via or "cli", "client_version": __pkg_version__}
        if installation_id:
            metadata["installation_id"] = installation_id
        if invocation_id:
            metadata["invocation_id"] = invocation_id

        body: dict[str, Any] = {
            "source_code": source_code,
            "source_path": source_path,
            "additional_instructions": additional_instructions,
            "objective": {"evaluation_command": evaluation_command, "metric_name": metric_name, "maximize": maximize},
            "optimizer": {
                "steps": steps,
                "code_generator": code_generator_config,
                "evaluator": evaluator_config,
                "search_policy": search_policy_config,
            },
            "eval_timeout": eval_timeout,
            "save_logs": save_logs,
            "log_dir": log_dir,
            "require_review": require_review,
            "metadata": metadata,
        }
        if api_keys:
            body["api_keys"] = api_keys

        resp = self._post("/runs/", json=body, timeout=(10, 3650))
        resp.raise_for_status()
        result = resp.json()
        result.setdefault("plan", "")
        result.setdefault("code", "")
        return result

    def get_run_status(self, run_id: str, *, include_history: bool = False) -> dict:
        """GET /runs/{run_id} — fetch run status and optionally the full node history."""
        resp = self._get(f"/runs/{run_id}", params={"include_history": include_history})
        resp.raise_for_status()
        result = resp.json()
        # Normalise nullable code/plan fields
        if result.get("best_result"):
            result["best_result"].setdefault("code", "")
            result["best_result"].setdefault("plan", "")
        for node in result.get("nodes") or []:
            node.setdefault("plan", "")
            node.setdefault("code", "")
        return result

    def resume_run(self, run_id: str, *, api_keys: dict[str, str] | None = None) -> dict | None:
        """POST /runs/{run_id}/resume — resume an interrupted run."""
        body: dict[str, Any] = {"metadata": {"client_name": "cli", "client_version": __pkg_version__}}
        if api_keys:
            body["api_keys"] = api_keys
        resp = self._post(f"/runs/{run_id}/resume", json=body, timeout=(5, 10))
        resp.raise_for_status()
        return resp.json()

    def suggest(
        self,
        run_id: str,
        *,
        execution_output: str,
        step: int | None = None,
        task_id: str | None = None,
        api_keys: dict[str, str] | None = None,
    ) -> dict:
        """POST /runs/{run_id}/suggest — submit execution output, get next candidate.

        If *step* is provided, transport errors (ReadTimeout, 502, ConnectionError)
        trigger an automatic recovery attempt via ``get_run_status``.
        """
        body: dict[str, Any] = {"execution_output": _truncate_output(execution_output), "metadata": {}}
        if task_id:
            body["task_id"] = task_id
        if api_keys:
            body["api_keys"] = api_keys

        try:
            resp = self._post(f"/runs/{run_id}/suggest", json=body, timeout=(10, 3650))
            resp.raise_for_status()
            result = resp.json()
            result.setdefault("plan", "")
            result.setdefault("code", "")
            return result
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as exc:
            if step is not None:
                recovered = self._recover_suggest(run_id, step)
                if recovered is not None:
                    return recovered
            raise type(exc)(exc) from exc
        except requests.exceptions.HTTPError as exc:
            if step is not None and getattr(exc.response, "status_code", None) == 502:
                recovered = self._recover_suggest(run_id, step)
                if recovered is not None:
                    return recovered
            raise

    def heartbeat(self, run_id: str) -> bool:
        """PUT /runs/{run_id}/heartbeat — keep the run alive."""
        try:
            resp = self._put(f"/runs/{run_id}/heartbeat")
            resp.raise_for_status()
            return True
        except requests.exceptions.HTTPError as e:
            code = getattr(e.response, "status_code", None)
            if code == 409:
                print(f"Polling ignore: Run {run_id} is not running.", file=sys.stderr)
            else:
                print(f"Polling failed for run {run_id}: HTTP {code}", file=sys.stderr)
            return False
        except Exception as e:
            print(f"Error sending heartbeat for run {run_id}: {e}", file=sys.stderr)
            return False

    def terminate(self, run_id: str, *, reason: str, details: str | None = None) -> bool:
        """POST /runs/{run_id}/terminate — report termination."""
        try:
            resp = self._post(
                f"/runs/{run_id}/terminate",
                json={"status_update": "terminated", "termination_reason": reason, "termination_details": details},
            )
            resp.raise_for_status()
            return True
        except Exception as e:
            print(f"Warning: Failed to report termination for run {run_id}: {e}", file=sys.stderr)
            return False

    def update_instructions(self, run_id: str, instructions: str | None) -> dict | None:
        """POST /runs/{run_id}/additional_instructions — update mid-run instructions."""
        try:
            resp = self._post(f"/runs/{run_id}/additional_instructions", json={"additional_instructions": instructions})
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return None

    def share(self, run_id: str) -> str | None:
        """POST /runs/{run_id}/share — create a public share link. Returns the share ID."""
        try:
            resp = self._post(f"/runs/{run_id}/share")
            resp.raise_for_status()
            return resp.json().get("share_id")
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Execution tasks
    # ------------------------------------------------------------------

    def get_execution_tasks(self, run_id: str) -> ExecutionTasksResult | None:
        """GET /execution-tasks/ — poll for ready tasks."""
        try:
            resp = self._get("/execution-tasks/", params={"run_id": run_id}, timeout=(5, 30))
            resp.raise_for_status()
            data = resp.json()

            run_summary = None
            if data.get("run"):
                rd = data["run"]
                run_summary = RunSummary(
                    id=rd["id"], status=rd["status"], name=rd.get("name"), require_review=rd.get("require_review", False)
                )
            return ExecutionTasksResult(tasks=data.get("tasks", []), run=run_summary)
        except Exception:
            return None

    def claim_task(self, task_id: str) -> dict | None:
        """POST /execution-tasks/{task_id}/claim — claim a task for evaluation."""
        try:
            resp = self._post(f"/execution-tasks/{task_id}/claim", timeout=(5, 30))
            if resp.status_code == 409:
                return None  # Already claimed
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Nodes / Revisions (review mode)
    # ------------------------------------------------------------------

    def create_revision(self, node_id: str, code: dict[str, str]) -> dict | None:
        """POST /nodes/{node_id}/revisions — create a new code revision."""
        try:
            resp = self._post(f"/nodes/{node_id}/revisions", json={"code": code}, timeout=(10, 60))
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return None

    def submit_node(self, node_id: str) -> dict | None:
        """POST /nodes/{node_id}/submit — submit a pending_approval node for evaluation."""
        try:
            resp = self._post(f"/nodes/{node_id}/submit")
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Observe (external runs)
    # ------------------------------------------------------------------

    def create_external_run(
        self,
        *,
        source_code: dict[str, str],
        metric_name: str,
        maximize: bool,
        name: str | None = None,
        additional_instructions: str | None = None,
        metadata: dict | None = None,
    ) -> dict | None:
        """POST /external/runs — create an external run for tracking."""
        body: dict = {"source_code": source_code, "metric_name": metric_name, "maximize": maximize}
        if name is not None:
            body["name"] = name
        if additional_instructions is not None:
            body["additional_instructions"] = additional_instructions
        if metadata:
            body["metadata"] = metadata
        try:
            resp = self._post("/external/runs", json=body, timeout=(5, 30))
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return None

    def log_external_step(
        self,
        run_id: str,
        *,
        step: int,
        status: str = "completed",
        description: str | None = None,
        metrics: dict[str, float] | None = None,
        code: dict[str, str] | None = None,
        parent_step: int | None = None,
        metadata: dict | None = None,
    ) -> dict | None:
        """POST /external/runs/{run_id}/steps — log a step to an external run."""
        body: dict = {"step": step, "status": status}
        if description is not None:
            body["description"] = description
        if metrics:
            body["metrics"] = metrics
        if code is not None:
            body["code"] = code
        if parent_step is not None:
            body["parent_step"] = parent_step
        if metadata:
            body["metadata"] = metadata
        try:
            resp = self._post(f"/external/runs/{run_id}/steps", json=body, timeout=(5, 30))
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _recover_suggest(self, run_id: str, step: int) -> dict | None:
        """Try to reconstruct a /suggest response after a transport error."""
        try:
            data = self.get_run_status(run_id, include_history=True)
        except Exception:
            return None

        if data.get("status") != "running" or data.get("current_step") != step:
            return None

        nodes = data.get("nodes") or []
        if len(nodes) < 2:
            return None

        by_step = sorted(nodes, key=lambda n: n["step"])
        latest, prev = by_step[-1], by_step[-2]
        if latest["step"] != step:
            return None

        return {
            "run_id": run_id,
            "previous_solution_metric_value": prev.get("metric_value"),
            "solution_id": latest.get("solution_id"),
            "code": latest.get("code"),
            "plan": latest.get("plan"),
            "is_done": False,
        }
