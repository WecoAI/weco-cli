"""Plain text output handler for CLI."""

from typing import Optional
from rich.console import Console


class PlainOutputHandler:
    """Handles plain text output without rich Live/panels but with colors.

    Used when --output plain is specified to provide simple text updates
    that still support colors but avoid Live displays, panels, and tables.
    """

    def __init__(self, metric_name: str, maximize: bool, total_steps: int, console: Console):
        self.metric_name = metric_name
        self.maximize = maximize
        self.total_steps = total_steps
        self.console = console
        self.best_value: Optional[float] = None

    def print_run_started(self, run_id: str, dashboard_url: str, model: str, runs_dir: str) -> None:
        """Print run initialization info."""
        self.console.print("[cyan]Starting optimization run...[/]")
        self.console.print(f"[bold]Run ID:[/] {run_id}")
        self.console.print(f"[bold]Dashboard:[/] [link={dashboard_url}]{dashboard_url}[/link]")
        self.console.print(f"[bold]Model:[/] {model}")
        self.console.print(f"[bold]Logs:[/] {runs_dir}/{run_id}")
        self.console.print(f"[bold]Objective:[/] {'Maximize' if self.maximize else 'Minimize'} {self.metric_name}")
        self.console.print()

    def print_step_started(self, step: int) -> None:
        """Print step start notification."""
        self.console.print(f"[dim]Step {step}/{self.total_steps}: evaluating...[/]")

    def print_step_completed(
        self,
        step: int,
        metric_value: Optional[float],
        is_buggy: bool,
        is_new_best: bool,
    ) -> None:
        """Print step completion with metric value."""
        if is_buggy:
            self.console.print(f"[red]Step {step}/{self.total_steps}: bug detected[/]")
        elif metric_value is not None:
            if is_new_best:
                self.console.print(f"[green]Step {step}/{self.total_steps}: {self.metric_name} = {metric_value:.4f} (new best!)[/]")
                self.best_value = metric_value
            else:
                self.console.print(f"Step {step}/{self.total_steps}: {self.metric_name} = {metric_value:.4f}")
        else:
            self.console.print(f"[yellow]Step {step}/{self.total_steps}: metric not available[/]")

    def print_best_update(self, metric_value: float) -> None:
        """Print when best value changes."""
        self.best_value = metric_value
        self.console.print(f"[green]Best: {self.metric_name} = {metric_value:.4f}[/]")

    def print_run_completed(self, best_value: Optional[float], runs_dir: str, run_id: str) -> None:
        """Print final run completion summary."""
        self.console.print()
        self.console.print("[bold green]Optimization complete![/]")
        if best_value is not None:
            goal_word = "maximized" if self.maximize else "minimized"
            self.console.print(f"[green]{self.metric_name.capitalize()} {goal_word}![/]")
            self.console.print(f"[bold]Best {self.metric_name}:[/] [green]{best_value:.4f}[/]")
        else:
            self.console.print("[red]No valid solution found.[/]")
        self.console.print(f"[dim]Logs saved to: {runs_dir}/{run_id}/[/]")

    def print_run_stopped(self, run_id: str) -> None:
        """Print when run is stopped by user."""
        self.console.print()
        self.console.print("[yellow]Run terminated by user request.[/]")
        self.console.print(f"[cyan]To resume this run, use:[/] [bold cyan]weco resume {run_id}[/]")

    def print_error(self, message: str) -> None:
        """Print error message."""
        self.console.print(f"[bold red]Error:[/] {message}")

    def print_resume_info(
        self,
        run_id: str,
        run_name: str,
        status: str,
        current_step: int,
        total_steps: int,
        model: str,
    ) -> None:
        """Print resume confirmation info."""
        self.console.print("[cyan]Resuming optimization run...[/]")
        self.console.print(f"[bold]Run ID:[/] {run_id}")
        self.console.print(f"[bold]Run Name:[/] {run_name}")
        self.console.print(f"[bold]Status:[/] {status}")
        self.console.print(f"[bold]Model:[/] {model}")
        self.console.print(f"[bold]Progress:[/] Step {current_step}/{total_steps}")
        self.console.print()
