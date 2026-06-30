"""Translating `weco start claude` inputs into Claude Agent SDK configuration.

Owns the three things we hand the SDK before connecting: the parsed
``claude_args``, the subprocess env (which encodes the billing route), and
the system-prompt delta we append to Claude Code's preset.
"""

from __future__ import annotations

import os
from typing import Any, Optional


# `effort` strings the SDK accepts, in display order. Shared with the
# argparse `--effort` choices so the CLI and the SDK agree on one list.
VALID_EFFORTS: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")


# Appended to claude's built-in Claude Code system prompt (we don't replace
# it — `preset: "claude_code"` keeps the baseline). Keeps the agent aware
# that it's running inside a Weco bridge and that the `weco` skill is the
# canonical entry point for anything Weco-related.
WECO_SYSTEM_PROMPT_APPEND = """\
You are running inside a Weco-bridged Claude Code session. The user is a Weco
user; they may be on a run-detail page in the Weco dashboard while talking
to you (a `<system-reminder>` block at the start of any prompt will give you
the current view: run id, lineage, step, best metric).

For any Weco-related operation — running an optimization, configuring an
evaluator, tuning a prompt or skill against a measurable metric, or anything
that involves Weco's optimization machinery — use the `/weco` skill rather
than writing ad-hoc loops or scripts yourself. The skill wraps Weco's
evaluation + search loop correctly; reinventing it is almost always wrong.

Load the `/weco` skill BEFORE running any `weco` command. The skill carries
the canonical workflow — how to start a run, monitor it without blocking,
steer with `derive`, and read results — and a bare `weco` invocation without
it almost always goes wrong. So before you run ANY `weco …` shell command
(`weco run`, `weco run derive`, `weco run status`, `weco run stop`,
`weco run results`, `weco resume`, …), make sure the `/weco` skill is loaded
in this session; if you have not already invoked it, invoke `/weco` first and
follow its guidance.
"""


async def keep_stream_open_hook(input_data, tool_use_id, context):
    """PreToolUse no-op hook that keeps the SDK's streaming-mode input
    channel open long enough for can_use_tool to fire — documented
    workaround in the Agent SDK docs."""
    return {"continue_": True}


# Default model for `weco start claude` when the user doesn't pass `--model`.
# Opus 4.7 is the strongest agent model; the Weco-billing proxy pins Claude
# traffic to the first-party Anthropic API, which serves this id.
DEFAULT_MODEL = "claude-opus-4-7"


def peek_model(args: list[str]) -> Optional[str]:
    """Cheap scan for ``--model VALUE``. ``None`` if absent (caller applies
    `DEFAULT_MODEL`)."""
    for i, a in enumerate(args):
        if a == "--model" and i + 1 < len(args):
            return args[i + 1]
        if a.startswith("--model="):
            return a.split("=", 1)[1]
    return None


def resolve_model(args: list[str]) -> str:
    """The model the session will actually run on: an explicit `--model`,
    else `DEFAULT_MODEL`. Used for both the SDK options and the banner so
    they never disagree."""
    return peek_model(args) or DEFAULT_MODEL


def parse_claude_args(args: list[str]) -> tuple[dict[str, Any], dict[str, Optional[str]]]:
    """Return ({recognized_fields}, {extra_args_for_sdk}).

    Recognized: ``--model VALUE``, ``--effort VALUE``,
    ``--dangerously-skip-permissions``. Everything else passes through to
    the SDK's ``extra_args`` dict.
    """
    parsed: dict[str, Any] = {}
    extra: dict[str, Optional[str]] = {}
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--model" and i + 1 < len(args):
            parsed["model"] = args[i + 1]
            i += 2
            continue
        if a == "--effort" and i + 1 < len(args):
            parsed["effort"] = args[i + 1]
            i += 2
            continue
        if a == "--dangerously-skip-permissions":
            parsed["dangerously_skip_permissions"] = True
            i += 1
            continue
        if a.startswith("--"):
            key = a[2:]
            if "=" in key:
                k, _, v = key.partition("=")
                extra[k] = v
            elif i + 1 < len(args) and not args[i + 1].startswith("--"):
                extra[key] = args[i + 1]
                i += 2
                continue
            else:
                extra[key] = None
        i += 1
    return parsed, extra


def build_sdk_env(*, billing: str, api_key: str, weco_api_base: Optional[str], session_id: Optional[str]) -> dict[str, str]:
    """Construct the env the SDK subprocess sees.

    For ``--billing weco``, the Anthropic BASE_URL is pointed at Weco's
    proxy (with the session id encoded in the path so the proxy can
    broadcast ``credits_updated`` to the right channel), and the user's
    Weco API key is used as the bearer. The proxy resolves the key to the
    user, charges credits, and forwards the request to the model provider.

    For ``--billing claude``, the SDK runs against Anthropic directly.
    The Claude Code CLI binary (which the SDK subprocesses out to) will
    use, in order of precedence:

      1. ``ANTHROPIC_API_KEY`` if set (BYO API-key flow).
      2. OAuth credentials stored locally by ``claude login`` otherwise.

    We sanitize the inherited env so a previous ``--billing weco`` run
    (or an explicit override in the user's shell pointing at our proxy)
    can't accidentally re-route the call back through Weco:

      * Drop ``ANTHROPIC_BASE_URL`` if it points at our proxy.
      * Drop ``ANTHROPIC_API_KEY`` if it's a Weco key (``weco-…``) —
        Anthropic would reject it, and the user almost certainly wants
        their ``claude login`` OAuth credentials to kick in instead.

    Genuine BYO setup (a real ``sk-ant-…`` key, or a non-Weco custom
    BASE_URL for an Anthropic-compatible gateway) is left untouched.

    ``weco_api_base`` should be the full Weco API base URL including the
    version segment (e.g. ``https://api.weco.ai/v1``).
    """
    env = dict(os.environ)
    # Signal to nested `weco` CLI invocations (e.g. `weco run` kicked off
    # by the agent) that they're running inside a `weco start` session, so
    # they suppress their own `webbrowser.open()` — the attached dashboard
    # surfaces the new run via an in-page toast instead.
    if session_id:
        env["WECO_CC_SESSION_ID"] = session_id
    if billing == "weco" and weco_api_base:
        base = weco_api_base.rstrip("/")
        # Encode session id as a path segment rather than a header — the
        # Anthropic Python SDK doesn't honor `ANTHROPIC_CUSTOM_HEADERS`,
        # so the proxy reads the id off the URL instead.
        if session_id:
            env["ANTHROPIC_BASE_URL"] = f"{base}/llm/anthropic/s/{session_id}"
        else:
            env["ANTHROPIC_BASE_URL"] = f"{base}/llm/anthropic"
        env["ANTHROPIC_API_KEY"] = api_key
        return env

    if billing == "claude":
        weco_base_marker = (weco_api_base or "").rstrip("/")
        cur_base = env.get("ANTHROPIC_BASE_URL", "")
        if weco_base_marker and cur_base.startswith(weco_base_marker):
            env.pop("ANTHROPIC_BASE_URL", None)
        cur_key = env.get("ANTHROPIC_API_KEY", "")
        if cur_key.startswith("weco-"):
            env.pop("ANTHROPIC_API_KEY", None)
    return env
