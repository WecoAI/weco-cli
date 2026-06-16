"""Tests for the inline tool-call approval card.

`ApprovalCard` is the AskUserQuestion picker (`QuestionCard`) reused for
tool gating — these cover the label→decision translation and the
cross-surface resolve paths the bridge depends on. No Textual app is
mounted; the card's `_mark_done` query is guarded, so we drive its
terminal methods directly.
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("textual")

from weco.ui.tui.approval import ApprovalCard


def _make_card(tool_name="Bash", summary="rm -rf build/"):
    future: asyncio.Future = asyncio.get_running_loop().create_future()
    return ApprovalCard(tool_name, summary, future), future


def test_selected_label_maps_to_decision():
    asyncio.run(_selected_label_maps_to_decision())


async def _selected_label_maps_to_decision():
    for label, expected in (
        ("Allow once", ("approve", "once")),
        ("Allow always", ("approve", "always")),
        ("Deny", ("deny", "once")),
    ):
        card, future = _make_card()
        # The OptionList records the picked label under the question key;
        # `_submit` is what the picker fires once a single question is done.
        card._answers = {"Allow Bash?": label}
        card._submit()
        assert future.result() == expected


def test_summary_rides_as_detail_not_question_text():
    asyncio.run(_summary_rides_as_detail())


async def _summary_rides_as_detail():
    card, _ = _make_card(tool_name="Bash", summary="rm -rf build/")
    q = card._questions[0]
    # Short question text keeps the collapsed "✓ … → answer" line readable;
    # the arbitrary summary lives on the detail sub-line.
    assert q["question"] == "Allow Bash?"
    assert q["detail"] == "rm -rf build/"


def test_escape_denies_once():
    asyncio.run(_escape_denies_once())


async def _escape_denies_once():
    card, future = _make_card()
    card._cancel()
    assert future.result() == ("deny", "once")


def test_dashboard_answer_freezes_card_with_its_decision():
    asyncio.run(_dashboard_answer_freezes_card())


async def _dashboard_answer_freezes_card():
    card, future = _make_card()
    card.resolve_remotely_decision("approve", "command")
    assert future.result() == ("approve", "command")
    # A late local pick can't double-resolve the SDK call.
    card._answers = {"Allow Bash?": "Deny"}
    card._submit()
    assert future.result() == ("approve", "command")


def test_turn_cancel_shim_denies_once():
    asyncio.run(_turn_cancel_shim_denies_once())


async def _turn_cancel_shim_denies_once():
    card, future = _make_card()
    # The bridge's generic cleanup calls resolve_remotely({}) on every
    # active card; for an approval that means deny-once.
    card.resolve_remotely({})
    assert future.result() == ("deny", "once")
