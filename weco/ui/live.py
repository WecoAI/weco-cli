"""Rich Live panel implementation of ``OptimizationUI``."""

import time
from typing import List, Optional

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .base import UIState


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
    ACTIVE_STATUSES = {"initializing", "polling", "executing", "submitting", "reconnecting"}
    STATUS_INDICATORS = {
        "initializing": ("⏳", "dim"),
        "polling": ("🔄", "cyan"),
        "executing": ("⚡", "yellow"),
        "submitting": ("🧠", "blue"),
        "reconnecting": ("📡", "yellow"),
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
        maximize: bool = True,
    ):
        self.console = console
        self.run_id = run_id
        self.run_name = run_name
        self.dashboard_url = dashboard_url
        self.model = model
        self.metric_name = metric_name
        self.maximize = maximize
        self.state = UIState(total_steps=total_steps)
        self._live: Optional[Live] = None
        self._derived_from: Optional[dict] = None  # populated by on_init for derived runs

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
        if self._derived_from:
            df = self._derived_from
            metric_suffix = f" [dim](metric: {df['metric_value']:.6g})[/]" if df.get("metric_value") is not None else ""
            grid.add_row("From", f"[yellow]{df['run_id']}[/] [dim]@ step {df['step']}[/]{metric_suffix}")
        grid.add_row("", "")

        # Progress (always shown)
        progress_bar = self._render_progress_bar()
        grid.add_row("Progress", progress_bar)

        # Status (always shown) - with spinner for active states
        status_text = Text()
        status_text.append(f"{emoji} ", style=style)
        status_text.append(self.state.status.replace("_", " ").title(), style=f"bold {style}")
        if self.state.status == "reconnecting" and self.state.reconnect_max_attempts > 0:
            status_text.append(
                f" (attempt {self.state.reconnect_attempt}/{self.state.reconnect_max_attempts}"
                f", retry in {self.state.reconnect_backoff_s:.0f}s)",
                style=f"bold {style}",
            )
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
            best = max(values) if self.maximize else min(values)

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
    def on_init(self, derived_from: Optional[dict] = None) -> None:
        """Display the run header. For derived runs, the parent reference is
        included as a "From" row in the panel."""
        self._derived_from = derived_from
        self._update()

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

    def on_reconnecting(self, attempt: int, max_attempts: int, backoff_s: float) -> None:
        self.state.status = "reconnecting"
        self.state.reconnect_attempt = attempt
        self.state.reconnect_max_attempts = max_attempts
        self.state.reconnect_backoff_s = backoff_s
        self._update()

    def on_reconnected(self) -> None:
        self.state.reconnect_attempt = 0
        self.state.reconnect_max_attempts = 0
        self.state.reconnect_backoff_s = 0.0
        # Status will be overwritten by the next on_polling call; clear here so
        # any error/status rendering in between reads a clean slate.
        self.state.status = "polling"
        self._update()
