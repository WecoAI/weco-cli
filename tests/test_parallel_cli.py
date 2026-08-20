"""CLI parsing + API-client wire-contract tests for K-parallel (M3a).

Pins SPEC.md "M3a wire contract" and "Required tests": ``weco run`` validates
``--parallel`` (absent -> 1, invalid -> argparse error); ``start_run`` sends K
inside ``optimizer`` only when >1 (byte-compat at K=1); ``derive_run`` never
sends K and only opts into the deferred-candidate capability when asked;
``suggest`` gates ``skip_generation`` on both the flag and a ``task_id``; and
``generate_candidate`` posts to ``/runs/{id}/generate`` and returns parsed JSON.
No network — the client's ``_post`` is captured.
"""

from __future__ import annotations

import argparse

import pytest

from weco.cli import configure_run_parser
from weco.core.api import WecoClient


AUTH = {"Authorization": "Bearer t"}


# ---------------------------------------------------------------------------
# --parallel argument parsing
# ---------------------------------------------------------------------------


def _run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    configure_run_parser(parser)
    return parser


def test_parallel_defaults_to_one():
    """Absent --parallel means serial (K=1)."""
    args, _ = _run_parser().parse_known_args([])
    assert args.parallel == 1


def test_parallel_accepts_valid_integer():
    """A valid K>=1 is parsed as an int."""
    args, _ = _run_parser().parse_known_args(["--parallel", "4"])
    assert args.parallel == 4
    args, _ = _run_parser().parse_known_args(["-p", "2"])
    assert args.parallel == 2


@pytest.mark.parametrize("bad", ["0", "-1", "x", "1.5"])
def test_parallel_rejects_invalid_values(bad):
    """K<1 or non-integer is rejected via an argparse error (SystemExit)."""
    with pytest.raises(SystemExit):
        _run_parser().parse_known_args(["--parallel", bad])


# ---------------------------------------------------------------------------
# WecoClient request-body capture
# ---------------------------------------------------------------------------


class _FakeResp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise AssertionError(f"unexpected status {self.status_code}")

    def json(self):
        return self._payload


def _client_capturing(payload=None):
    """A WecoClient whose _post records (path, json) instead of hitting the network."""
    client = WecoClient(AUTH)
    captured: dict = {}

    def fake_post(path, *, json=None, timeout=None):
        captured["path"] = path
        captured["json"] = json
        return _FakeResp(payload if payload is not None else {})

    client._post = fake_post  # type: ignore[assignment]
    return client, captured


_START_KWARGS = dict(
    source_code={"m.py": "x"},
    source_path=None,
    evaluation_command="python eval.py",
    metric_name="acc",
    maximize=True,
    steps=10,
    code_generator_config={"model": "g"},
    evaluator_config={},
    search_policy_config={},
)


def test_start_run_omits_parallelism_at_k1():
    """K=1 keeps the optimizer body byte-identical to older serial CLIs."""
    client, captured = _client_capturing({"id": "r", "plan": None, "code": None})
    client.start_run(parallelism=1, **_START_KWARGS)
    assert "parallelism" not in captured["json"]["optimizer"]


def test_start_run_sends_parallelism_inside_optimizer_when_above_one():
    """K>1 carries the strict shared K inside the optimizer body."""
    client, captured = _client_capturing({"id": "r", "plan": None, "code": None})
    client.start_run(parallelism=3, **_START_KWARGS)
    assert captured["json"]["optimizer"]["parallelism"] == 3
    # K is never a top-level field.
    assert "parallelism" not in captured["json"]


def test_derive_run_never_sends_k_and_defaults_without_deferred_flag():
    """Derived runs use the root's pool: no K, and no deferred flag by default."""
    client, captured = _client_capturing({"id": "child"})
    client.derive_run("run-1")
    body = captured["json"]
    assert "parallelism" not in body
    assert "allow_deferred_candidate" not in body


def test_derive_run_opts_into_deferred_candidate_when_requested():
    """The M3a scheduler opts into the deferred zero-work child capability."""
    client, captured = _client_capturing({"id": "child"})
    client.derive_run("run-1", allow_deferred_candidate=True)
    assert captured["json"]["allow_deferred_candidate"] is True


def test_suggest_sends_skip_generation_with_task_id():
    """skip_generation is sent when set alongside a task_id (queue flow)."""
    client, captured = _client_capturing({"run_id": "r", "is_done": False, "plan": None, "code": None})
    client.suggest("run-1", execution_output="out", task_id="t1", skip_generation=True)
    assert captured["json"]["skip_generation"] is True


def test_suggest_omits_skip_generation_without_task_id():
    """skip_generation is meaningless without a task_id, so it is not sent."""
    client, captured = _client_capturing({"run_id": "r", "is_done": False, "plan": None, "code": None})
    client.suggest("run-1", execution_output="out", skip_generation=True)
    assert "skip_generation" not in captured["json"]


def test_suggest_omits_skip_generation_by_default():
    """Serial clients never send skip_generation."""
    client, captured = _client_capturing({"run_id": "r", "is_done": False, "plan": None, "code": None})
    client.suggest("run-1", execution_output="out", task_id="t1")
    assert "skip_generation" not in captured["json"]


def test_generate_candidate_posts_to_generate_and_returns_json():
    """generate_candidate targets /runs/{id}/generate and returns the parsed body."""
    payload = {"generated": False, "reason": "at_capacity"}
    client, captured = _client_capturing(payload)
    result = client.generate_candidate("run-42")
    assert captured["path"] == "/runs/run-42/generate"
    assert result == payload
