"""Tests for DashboardSession + the WecoClient session-REST methods.

Real Realtime/channel behaviour is exercised end-to-end elsewhere; here we
cover the local seams: the offline no-op session, the create handshake, and
error translation to SetupError.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest
import requests

from weco.commands.start.session import DashboardSession, SetupError


FAKE_SESSION = {
    "session": {"id": "sess-1"},
    "dashboard_url": "https://dashboard.weco.ai/sessions/sess-1",
    "realtime": {"url": "wss://x", "token": "t", "expires_at": "2026-05-05T13:00:00+00:00", "channel": "agent:sess-1"},
}


# --- offline -------------------------------------------------------------------


def test_offline_session_is_inert():
    asyncio.run(_offline_session_is_inert())


async def _offline_session_is_inert():
    s = DashboardSession.offline()
    assert s.online is False
    assert s.id is None
    assert s.dashboard_url is None
    # publish still accepts into the (undrained) queue, matching legacy behaviour.
    assert s.publish("a line") is True
    # run() returns immediately rather than trying to open a channel.
    await asyncio.wait_for(s.run(on_inbound=lambda _m: None, stop_event=asyncio.Event()), timeout=1.0)


# --- create --------------------------------------------------------------------


def test_create_exposes_id_and_url_and_marks_online():
    with patch("weco.commands.start.session.WecoClient") as MockClient:
        MockClient.return_value.create_session.return_value = FAKE_SESSION
        s = DashboardSession.create(api_key="weco-k", agent_type="claude-code")
    assert s.online is True
    assert s.id == "sess-1"
    assert s.dashboard_url == "https://dashboard.weco.ai/sessions/sess-1"
    # Built with a bearer header derived from the api key.
    assert MockClient.call_args.args[0] == {"Authorization": "Bearer weco-k"}
    MockClient.return_value.create_session.assert_called_once_with("claude-code")


def test_create_translates_http_error_to_setup_error():
    resp = MagicMock(status_code=402, text="insufficient credits")
    err = requests.HTTPError(response=resp)
    with patch("weco.commands.start.session.WecoClient") as MockClient:
        MockClient.return_value.create_session.side_effect = err
        with pytest.raises(SetupError) as excinfo:
            DashboardSession.create(api_key="weco-k", agent_type="claude-code")
    assert "402" in str(excinfo.value)


def test_create_translates_network_error_to_setup_error():
    with patch("weco.commands.start.session.WecoClient") as MockClient:
        MockClient.return_value.create_session.side_effect = requests.ConnectionError("no route")
        with pytest.raises(SetupError):
            DashboardSession.create(api_key="weco-k", agent_type="claude-code")


# --- WecoClient session REST ---------------------------------------------------


def test_weco_client_create_session_posts_agent_type():
    from weco.core.api import WecoClient

    client = WecoClient({"Authorization": "Bearer weco-k"})
    fake = MagicMock(status_code=200)
    fake.json.return_value = FAKE_SESSION
    fake.raise_for_status.return_value = None
    with patch.object(client._session, "post", return_value=fake) as post:
        out = client.create_session("claude-code")
    assert out == FAKE_SESSION
    url = post.call_args.args[0]
    assert url.endswith("/sessions")
    assert post.call_args.kwargs["json"] == {"agent_type": "claude-code"}


def test_weco_client_close_session_patches_status():
    from weco.core.api import WecoClient

    client = WecoClient({"Authorization": "Bearer weco-k"})
    with patch.object(client._session, "patch", return_value=MagicMock(status_code=200)) as patch_call:
        client.close_session("sess-1")
    url = patch_call.call_args.args[0]
    assert url.endswith("/sessions/sess-1")
    assert patch_call.call_args.kwargs["json"] == {"status": "closed"}


def test_refresh_realtime_token_raises_on_409():
    from weco.core.api import SessionInactiveError, WecoClient

    client = WecoClient({"Authorization": "Bearer weco-k"})
    with patch.object(client._session, "post", return_value=MagicMock(status_code=409)):
        with pytest.raises(SessionInactiveError):
            client.refresh_realtime_token("sess-1")


def test_session_heartbeat_raises_on_409():
    from weco.core.api import SessionInactiveError, WecoClient

    client = WecoClient({"Authorization": "Bearer weco-k"})
    with patch.object(client._session, "post", return_value=MagicMock(status_code=409)):
        with pytest.raises(SessionInactiveError):
            client.session_heartbeat("sess-1")
