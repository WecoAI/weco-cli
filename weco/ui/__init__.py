"""Optimization loop UI components.

This package contains the ``OptimizationUI`` Protocol and its two
implementations:

* :class:`LiveOptimizationUI` — Rich Live panel for interactive terminals
* :class:`PlainOptimizationUI` — bracket-tagged text for LLM agents and
  machine consumers

The Protocol and shared state live in ``base``; each implementation has its
own submodule. Import everything from ``weco.ui`` directly — the symbols are
re-exported here for backward compatibility.
"""

from .base import OptimizationUI, UIState
from .live import LiveOptimizationUI
from .plain import PlainOptimizationUI

__all__ = ["OptimizationUI", "UIState", "LiveOptimizationUI", "PlainOptimizationUI"]
