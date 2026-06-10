"""Reusable context manager for the run heartbeat thread.

Wraps :class:`weco.optimizer.HeartbeatSender` so callers can write::

    with heartbeat(run_id, auth_headers):
        ...

instead of repeating the ``Event() / start() / try / finally / set / join``
boilerplate. The thread is started on entry and stopped (with a bounded
join) on exit, even if the body raises.
"""

import threading
from contextlib import contextmanager

from .optimizer import HeartbeatSender


# How long to wait for the heartbeat thread to exit cleanly on shutdown.
# The thread is a daemon, so this is purely a courtesy join — exceeding it
# is harmless.
HEARTBEAT_SHUTDOWN_TIMEOUT_S = 2


@contextmanager
def heartbeat(run_id: str, auth_headers: dict[str, str]):
    """Run a heartbeat thread for the lifetime of the ``with`` block."""
    stop_event = threading.Event()
    sender = HeartbeatSender(run_id, auth_headers, stop_event)
    sender.start()
    try:
        yield
    finally:
        stop_event.set()
        sender.join(timeout=HEARTBEAT_SHUTDOWN_TIMEOUT_S)
