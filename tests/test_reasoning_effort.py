"""weco run --reasoning-effort: parsing and run-init config threading.

The backend's code_generator/evaluator config models use Extra.allow and
forward unknown kwargs to the model call, so the flag needs no server or
client-API change — only the CLI surface and the config build.
"""

import argparse

import pytest

from weco.optimizer import _build_model_configs


@pytest.mark.parametrize("value", ["low", "medium", "high"])
def test_reasoning_effort_flag_accepts_levels(value):
    from weco.cli import configure_run_parser

    parser = argparse.ArgumentParser()
    configure_run_parser(parser)
    args = parser.parse_args(["--reasoning-effort", value])
    assert args.reasoning_effort == value


def test_reasoning_effort_flag_rejects_junk():
    from weco.cli import configure_run_parser

    parser = argparse.ArgumentParser()
    configure_run_parser(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(["--reasoning-effort", "extreme"])


def test_reasoning_effort_defaults_to_none():
    from weco.cli import configure_run_parser

    parser = argparse.ArgumentParser()
    configure_run_parser(parser)
    assert parser.parse_args([]).reasoning_effort is None


def test_configs_carry_reasoning_effort_on_both_sides():
    """Generation AND evaluation: the effort applies to drafting/improving and
    to the LLM judge alike (matching the rsi-weco harness semantics)."""
    code_gen, evaluator = _build_model_configs("gemini-3-flash-preview", "medium")
    assert code_gen == {"model": "gemini-3-flash-preview", "reasoning_effort": "medium"}
    assert evaluator["model"] == "gemini-3-flash-preview"
    assert evaluator["reasoning_effort"] == "medium"
    assert evaluator["include_analysis"] is True


def test_configs_omit_reasoning_effort_when_unset():
    code_gen, evaluator = _build_model_configs("gemini-3-flash-preview", None)
    assert "reasoning_effort" not in code_gen
    assert "reasoning_effort" not in evaluator
