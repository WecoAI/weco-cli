"""Plain text output handler for CLI."""

from typing import Optional


class PlainOutputHandler:
    """Handles plain text output without rich formatting.

    Used when --output plain is specified to provide simple,
    colorless text updates suitable for non-interactive environments.
    """

    def __init__(self, metric_name: str, maximize: bool, total_steps: int):
        self.metric_name = metric_name
        self.maximize = maximize
        self.total_steps = total_steps
        self.best_value: Optional[float] = None

    def print_run_started(self, run_id: str, dashboard_url: str, model: str, runs_dir: str) -> None:
        """Print run initialization info."""
        print("Starting optimization run...")
        print(f"Run ID: {run_id}")
        print(f"Dashboard: {dashboard_url}")
        print(f"Model: {model}")
        print(f"Logs: {runs_dir}/{run_id}")
        print(f"Objective: {'Maximize' if self.maximize else 'Minimize'} {self.metric_name}")
        print()

    def print_step_started(self, step: int) -> None:
        """Print step start notification."""
        print(f"Step {step}/{self.total_steps}: evaluating...")

    def print_step_completed(
        self,
        step: int,
        metric_value: Optional[float],
        is_buggy: bool,
        is_new_best: bool,
    ) -> None:
        """Print step completion with metric value."""
        if is_buggy:
            print(f"Step {step}/{self.total_steps}: bug detected")
        elif metric_value is not None:
            best_marker = " (new best!)" if is_new_best else ""
            print(f"Step {step}/{self.total_steps}: {self.metric_name} = {metric_value:.4f}{best_marker}")
            if is_new_best:
                self.best_value = metric_value
        else:
            print(f"Step {step}/{self.total_steps}: metric not available")

    def print_best_update(self, metric_value: float) -> None:
        """Print when best value changes."""
        self.best_value = metric_value
        print(f"Best: {self.metric_name} = {metric_value:.4f}")

    def print_run_completed(self, best_value: Optional[float], runs_dir: str, run_id: str) -> None:
        """Print final run completion summary."""
        print()
        print("Optimization complete!")
        if best_value is not None:
            goal_word = "maximized" if self.maximize else "minimized"
            print(f"{self.metric_name.capitalize()} {goal_word}!")
            print(f"Best {self.metric_name}: {best_value:.4f}")
        else:
            print("No valid solution found.")
        print(f"Logs saved to: {runs_dir}/{run_id}/")

    def print_run_stopped(self, run_id: str) -> None:
        """Print when run is stopped by user."""
        print()
        print("Run terminated by user request.")
        print(f"To resume this run, use: weco resume {run_id}")

    def print_error(self, message: str) -> None:
        """Print error message."""
        print(f"Error: {message}")

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
        print("Resuming optimization run...")
        print(f"Run ID: {run_id}")
        print(f"Run Name: {run_name}")
        print(f"Status: {status}")
        print(f"Model: {model}")
        print(f"Progress: Step {current_step}/{total_steps}")
        print()
