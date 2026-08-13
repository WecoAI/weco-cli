"""Tests for the POST /runs/ body built by WecoClient.start_run."""

from unittest.mock import Mock

from weco.core.api import WecoClient


def _start_run(**kwargs):
    client = WecoClient({})
    captured = {}

    def fake_post(path, *, json=None, timeout=None):
        captured["path"] = path
        captured["body"] = json
        resp = Mock()
        resp.raise_for_status = Mock()
        resp.json.return_value = {"run_id": "r1", "run_name": "n", "plan": "", "code": ""}
        return resp

    client._post = fake_post
    client.start_run(
        source_code={"main.py": "pass"},
        source_path=None,
        evaluation_command="python eval.py",
        metric_name="score",
        maximize=True,
        steps=5,
        code_generator_config={"model": "gpt-4"},
        evaluator_config={"model": "gpt-4"},
        search_policy_config={"num_drafts": 3},
        **kwargs,
    )
    return captured["body"]


def test_enable_web_search_in_body():
    body = _start_run(enable_web_search=True)
    assert body["enable_web_search"] is True


def test_enable_web_search_defaults_false():
    body = _start_run()
    assert body["enable_web_search"] is False
    assert body["require_review"] is False
