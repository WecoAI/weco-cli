"""Shared types: the ``OptimizationUI`` Protocol and the ``UIState`` dataclass."""

from dataclasses import dataclass, field
from typing import List, Optional, Protocol


class OptimizationUI(Protocol):
    """Protocol for optimization UI event handlers."""

    def on_init(self, derived_from: dict | None = None) -> None:
        """Called once after entering the UI context, before any loop events.

        Implementations display the run header / panel here. Optionally accepts
        a ``derived_from`` dict (with ``run_id``, ``node_id``, ``step``,
        ``metric_value``) for runs created via ``weco run derive``.
        """
        ...

    def on_polling(self, step: int) -> None:
        """Called when polling for execution tasks."""
        ...

    def on_task_claimed(self, task_id: str, plan: Optional[str]) -> None:
        """Called when a task is successfully claimed."""
        ...

    def on_executing(self, step: int) -> None:
        """Called when starting to execute code."""
        ...

    def on_output(self, output: str, max_preview: int = 200) -> None:
        """Called with execution output."""
        ...

    def on_submitting(self) -> None:
        """Called when submitting result to backend."""
        ...

    def on_metric(self, step: int, value: float) -> None:
        """Called when a metric value is received."""
        ...

    def on_complete(self, total_steps: int) -> None:
        """Called when optimization completes successfully."""
        ...

    def on_stop_requested(self) -> None:
        """Called when a stop request is received from dashboard."""
        ...

    def on_interrupted(self) -> None:
        """Called when interrupted by user (Ctrl+C)."""
        ...

    def on_warning(self, message: str) -> None:
        """Called for non-fatal warnings."""
        ...

    def on_error(self, message: str) -> None:
        """Called for errors."""
        ...


@dataclass
class UIState:
    """Reactive state for the live optimization UI."""

    step: int = 0
    total_steps: int = 0
    status: str = "initializing"  # polling, executing, submitting, complete, stopped, error
    plan_preview: str = ""
    output_preview: str = ""
    metrics: List[tuple] = field(default_factory=list)  # (step, value)
    error: Optional[str] = None
