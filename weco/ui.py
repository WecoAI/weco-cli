"""
Optimization loop UI components.

This module contains the UI protocol and implementations for displaying
optimization progress in the CLI.
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional, Protocol

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class OptimizationUI(Protocol):
    """Protocol for optimization UI event handlers."""

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


class LiveOptimizationUI:
    """
    Rich Live implementation of OptimizationUI with dynamic single-panel updates.

    Displays a compact, updating panel showing:
    - Run info (ID, name, dashboard link)
    - Current step and status with visual indicator
    - Plan preview
    - Output preview
    - Metric history as sparkline
    """

    SPARKLINE_CHARS = "▁▂▃▄▅▆▇█"
    SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    # Statuses that show the spinner animation
    ACTIVE_STATUSES = {"initializing", "polling", "executing", "submitting"}
    STATUS_INDICATORS = {
        "initializing": ("⏳", "dim"),
        "polling": ("🔄", "cyan"),
        "executing": ("⚡", "yellow"),
        "submitting": ("🧠", "blue"),
        "complete": ("✅", "green"),
        "stopped": ("⏹", "yellow"),
        "interrupted": ("⚠", "yellow"),
        "error": ("❌", "red"),
    }

    def __init__(
        self,
        console: Console,
        run_id: str,
        run_name: str,
        total_steps: int,
        dashboard_url: str,
        model: str = "",
        metric_name: str = "",
    ):
        self.console = console
        self.run_id = run_id
        self.run_name = run_name
        self.dashboard_url = dashboard_url
        self.model = model
        self.metric_name = metric_name
        self.state = UIState(total_steps=total_steps)
        self._live: Optional[Live] = None

    def _sparkline(self, values: List[float], max_width: int) -> str:
        """
        Create a mini sparkline chart from metric values.

        Automatically slides to show most recent values when they exceed max_width.
        Shows "···" prefix when older values are hidden.
        """
        if not values:
            return ""

        # Reserve space for "···" prefix if we need to truncate
        if len(values) > max_width:
            prefix = "··"
            available = max_width - len(prefix)
            vals = values[-available:]  # Take most recent values that fit
            sparkline_prefix = f"[dim]{prefix}[/]"
        else:
            vals = values
            sparkline_prefix = ""

        min_v, max_v = min(vals), max(vals)
        if max_v == min_v:
            return sparkline_prefix + self.SPARKLINE_CHARS[4] * len(vals)

        chars = self.SPARKLINE_CHARS
        sparkline = "".join(chars[int((v - min_v) / (max_v - min_v) * 7)] for v in vals)
        return sparkline_prefix + sparkline

    def _render(self) -> Group:
        """Render the current UI state as a Rich Panel with top margin."""
        emoji, style = self.STATUS_INDICATORS.get(self.state.status, ("⏳", "dim"))

        # Build content grid - expands to full terminal width
        grid = Table.grid(padding=(0, 1), expand=True)
        grid.add_column(style="dim", width=10)
        grid.add_column(overflow="ellipsis", no_wrap=True, ratio=1)

        # Run info (always shown)
        run_display = f"[bold]{self.run_name}[/] [dim]({self.run_id})[/]"
        grid.add_row("Run", run_display)
        grid.add_row("Dashboard", f"[link={self.dashboard_url}]{self.dashboard_url}[/link]")
        if self.model:
            grid.add_row("Model", f"[cyan]{self.model}[/]")
        if self.metric_name:
            grid.add_row("Metric", f"[magenta]{self.metric_name}[/]")
        grid.add_row("", "")

        # Progress (always shown)
        progress_bar = self._render_progress_bar()
        grid.add_row("Progress", progress_bar)

        # Status (always shown) - with spinner for active states
        status_text = Text()
        status_text.append(f"{emoji} ", style=style)
        status_text.append(self.state.status.replace("_", " ").title(), style=f"bold {style}")
        if self.state.status in self.ACTIVE_STATUSES:
            # Time-based frame calculation: ~10 fps spinner animation
            frame = int(time.time() * 10) % len(self.SPINNER_FRAMES)
            spinner = self.SPINNER_FRAMES[frame]
            status_text.append(f" {spinner}", style=f"bold {style}")
        grid.add_row("Status", status_text)

        # Plan (always shown, placeholder when empty)
        if self.state.plan_preview:
            grid.add_row("Plan", f"[dim italic]{self.state.plan_preview}[/]")
        else:
            grid.add_row("Plan", "[dim]—[/]")

        # Output (always shown, placeholder when empty)
        if self.state.output_preview:
            output_text = self.state.output_preview.replace("\n", " ")
            grid.add_row("Output", f"[dim]{output_text}[/]")
        else:
            grid.add_row("Output", "[dim]—[/]")

        # Metrics section (always shown, 3 rows: current, best, chart)
        if self.state.metrics:
            values = [m[1] for m in self.state.metrics]
            latest = self.state.metrics[-1][1]
            best = max(values)

            # Current and best on separate lines
            grid.add_row("Current", f"[bold cyan]{latest:.6g}[/]")
            grid.add_row("Best", f"[bold green]{best:.6g}[/]")

            # Chart line - calculate available width for sparkline
            # Console width minus: label(10) + padding(4) + panel borders(4) + panel padding(4)
            chart_width = max(self.console.width - 22, 20)
            sparkline = self._sparkline(values, chart_width)
            grid.add_row("History", f"[green]{sparkline}[/]")
        else:
            grid.add_row("Current", "[dim]—[/]")
            grid.add_row("Best", "[dim]—[/]")
            grid.add_row("History", "[dim]—[/]")

        # Error row (always present, empty when no error)
        if self.state.error:
            grid.add_row("Error", f"[bold red]{self.state.error}[/]")
        else:
            grid.add_row("", "")  # Empty row to maintain height

        panel = Panel(grid, title="[bold blue]⚡ Weco Optimization[/]", border_style="blue", padding=(1, 2), expand=True)
        # Wrap panel with top margin for spacing
        return Group(Text(""), panel)

    def _render_progress_bar(self) -> Text:
        """Render a simple ASCII progress bar."""
        total = self.state.total_steps
        current = min(self.state.step, total)  # Clamp to total to avoid >100%
        width = 40

        if total <= 0:
            return Text(f"Step {self.state.step}", style="bold")

        filled = min(int((current / total) * width), width)  # Clamp filled bars
        bar = "█" * filled + "░" * (width - filled)
        pct = min((current / total) * 100, 100)  # Clamp percentage
        return Text(f"[{bar}] {current}/{total} ({pct:.0f}%)", style="bold")

    def __rich__(self) -> Group:
        """Called by Rich on each refresh cycle - enables auto-animated spinner."""
        return self._render()

    def _update(self) -> None:
        """Trigger an immediate live update (for state changes)."""
        if self._live:
            self._live.refresh()

    # --- Context manager for Live display ---
    def __enter__(self) -> "LiveOptimizationUI":
        # Pass self so Rich calls __rich__() on every auto-refresh (enables spinner animation)
        # Use vertical_overflow="visible" to prevent clipping issues on exit
        self._live = Live(self, console=self.console, refresh_per_second=10, transient=False, vertical_overflow="visible")
        self._live.__enter__()
        return self

    def __exit__(self, *args) -> None:
        if self._live:
            self._live.__exit__(*args)
            self._live = None

    # --- OptimizationUI Protocol Implementation ---
    def on_polling(self, step: int) -> None:
        self.state.step = step
        self.state.status = "polling"
        self.state.output_preview = ""
        self._update()

    def on_task_claimed(self, task_id: str, plan: Optional[str]) -> None:
        self.state.plan_preview = plan or ""
        self._update()

    def on_executing(self, step: int) -> None:
        self.state.step = step
        self.state.status = "executing"
        self._update()

    def on_output(self, output: str, max_preview: int = 200) -> None:
        self.state.output_preview = output[:max_preview]
        self._update()

    def on_submitting(self) -> None:
        self.state.status = "submitting"
        self._update()

    def on_metric(self, step: int, value: float) -> None:
        self.state.metrics.append((step, value))
        self._update()

    def on_complete(self, total_steps: int) -> None:
        self.state.step = total_steps
        self.state.status = "complete"
        self._update()

    def on_stop_requested(self) -> None:
        self.state.status = "stopped"
        self._update()

    def on_interrupted(self) -> None:
        self.state.status = "interrupted"
        self._update()

    def on_warning(self, message: str) -> None:
        # Warnings are less critical; we could add a warnings list but keeping it simple
        pass

    def on_error(self, message: str) -> None:
        self.state.error = message
        self.state.status = "error"
        self._update()


class PlainOptimizationUI:
    """
    Plain text implementation of OptimizationUI for machine-readable output.

    Designed to be consumed by LLM agents - outputs structured, parseable text
    without Rich formatting, ANSI codes, or interactive elements.
    Includes full execution output for agent consumption.
    """

    def __init__(
        self, run_id: str, run_name: str, total_steps: int, dashboard_url: str, model: str = "", metric_name: str = ""
    ):
        self.run_id = run_id
        self.run_name = run_name
        self.total_steps = total_steps
        self.dashboard_url = dashboard_url
        self.model = model
        self.metric_name = metric_name
        self.current_step = 0
        self.metrics: List[tuple] = []  # (step, value)
        self._header_printed = False

    def _print(self, message: str) -> None:
        """Print a message to stdout with flush for immediate output."""
        print(message, flush=True)

    def _print_header(self) -> None:
        """Print run header info once at start."""
        if self._header_printed:
            return
        self._header_printed = True
        self._print("=" * 60)
        self._print("WECO OPTIMIZATION RUN")
        self._print("=" * 60)
        self._print(f"Run ID: {self.run_id}")
        self._print(f"Run Name: {self.run_name}")
        self._print(f"Dashboard: {self.dashboard_url}")
        if self.model:
            self._print(f"Model: {self.model}")
        if self.metric_name:
            self._print(f"Metric: {self.metric_name}")
        self._print(f"Total Steps: {self.total_steps}")
        self._print("=" * 60)
        self._print("")

    # --- Context manager (no-op for plain output) ---
    def __enter__(self) -> "PlainOptimizationUI":
        self._print_header()
        return self

    def __exit__(self, *args) -> None:
        pass

    # --- OptimizationUI Protocol Implementation ---
    def on_polling(self, step: int) -> None:
        self.current_step = step
        self._print(f"[STEP {step}/{self.total_steps}] Polling for task...")

    def on_task_claimed(self, task_id: str, plan: Optional[str]) -> None:
        self._print(f"[TASK CLAIMED] {task_id}")
        if plan:
            self._print(f"[PLAN] {plan}")

    def on_executing(self, step: int) -> None:
        self.current_step = step
        self._print(f"[STEP {step}/{self.total_steps}] Executing code...")

    def on_output(self, output: str, max_preview: int = 200) -> None:
        # For plain mode, output the full execution result for LLM consumption
        self._print("[EXECUTION OUTPUT START]")
        self._print(output)
        self._print("[EXECUTION OUTPUT END]")

    def on_submitting(self) -> None:
        self._print("[SUBMITTING] Sending result to backend...")

    def on_metric(self, step: int, value: float) -> None:
        self.metrics.append((step, value))
        best = max(m[1] for m in self.metrics) if self.metrics else value
        self._print(f"[METRIC] Step {step}: {value:.6g} (best so far: {best:.6g})")

    def on_complete(self, total_steps: int) -> None:
        self._print("")
        self._print("=" * 60)
        self._print("[COMPLETE] Optimization finished successfully")
        self._print(f"Total steps completed: {total_steps}")
        if self.metrics:
            values = [m[1] for m in self.metrics]
            self._print(f"Best metric value: {max(values):.6g}")
        self._print("=" * 60)

    def on_stop_requested(self) -> None:
        self._print("")
        self._print("[STOPPED] Run stopped by user request")

    def on_interrupted(self) -> None:
        self._print("")
        self._print("[INTERRUPTED] Run interrupted (Ctrl+C)")

    def on_warning(self, message: str) -> None:
        self._print(f"[WARNING] {message}")

    def on_error(self, message: str) -> None:
        self._print(f"[ERROR] {message}")
