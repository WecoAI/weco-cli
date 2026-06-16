"""Tests for the low-level Realtime channel pump (`relay`).

Live Realtime connection / JWT refresh / channel subscription is exercised
end-to-end against a real Supabase instance, not here. These cover the
local-only seams: UTF-8-safe chunking and timestamp parsing. Session REST
now lives on ``WecoClient`` / ``DashboardSession`` (see ``test_start_session``).
"""

from __future__ import annotations

from weco.commands.start import relay


# --- _chunk_utf8 ---------------------------------------------------------------


def test_chunk_short_strings_pass_through_unchanged():
    chunks = list(relay._chunk_utf8("hello", 100))
    assert chunks == ["hello"]


def test_chunk_splits_long_ascii_at_byte_limit():
    s = "a" * 1000
    chunks = list(relay._chunk_utf8(s, 256))
    assert "".join(chunks) == s
    assert all(len(c.encode("utf-8")) <= 256 for c in chunks)
    assert len(chunks) == 4  # 1000 / 256 rounded up


def test_chunk_does_not_split_multibyte_chars():
    # 4-byte char repeated; naive byte slicing would corrupt mid-char.
    s = "𠮷" * 200  # each char is 4 bytes in UTF-8
    chunks = list(relay._chunk_utf8(s, 17))  # awkward boundary
    assert "".join(chunks) == s
    for c in chunks:
        c.encode("utf-8").decode("utf-8")


# --- _parse_iso ----------------------------------------------------------------


def test_parse_iso_handles_z_suffix():
    ts = relay._parse_iso("2026-05-05T12:00:00Z")
    # Just sanity: it's a unix timestamp around the right magnitude.
    assert ts > 1_700_000_000
