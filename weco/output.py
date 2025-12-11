"""Output handlers for CLI - event-based interface for rich and plain output modes."""

from abc import ABC, abstractmethod
from typing import Optional
from rich.console import Console
from rich.live import Live

from .panels import (
    SummaryPanel,
    SolutionPanels,
    EvaluationOutputPanel,
    MetricTreePanel,
    Node,
    create_optimization_layout,
    create_end_optimization_layout,
)
from .utils import smooth_update


class OutputHandler(ABC):
    """Abstract base class for output handlers.

    Defines the event-based interface that both rich and plain output modes implement.
    The optimizer calls these methods at key points during execution.
    """

    @abstractmethod
    def __enter__(self):
        """Enter the output context (e.g., start Live display for rich mode)."""
        return self

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the output context (e.g., stop Live display)."""
        pass

    @abstractmethod
    def on_run_started(
        self,
        run_id: str,
        run_name: str,
        dashboard_url: str,
        model: str,
        runs_dir: str,
        plan: str,
        initial_code: str,
        initial_solution_id: str,
    ) -> None:
        """Called when the optimization run starts."""
        pass

    @abstractmethod
    def on_baseline_evaluating(self) -> None:
        """Called before evaluating the baseline solution."""
        pass

    @abstractmethod
    def on_baseline_completed(self, eval_output: str, best_metric: Optional[float] = None) -> None:
        """Called after baseline evaluation completes."""
        pass

    @abstractmethod
    def on_step_starting(self, step: int, previous_best_metric: Optional[float] = None) -> None:
        """Called at the start of a step (before API call that evaluates previous and generates next)."""
        pass

    @abstractmethod
    def on_step_generated(
        self, step: int, code: str, plan: str, nodes: list, current_node: Node, best_node: Optional[Node], solution_id: str
    ) -> None:
        """Called after code generation, with updated state."""
        pass

    @abstractmethod
    def on_step_completed(self, step: int, eval_output: str) -> None:
        """Called after local evaluation completes (result not yet known)."""
        pass

    @abstractmethod
    def on_step_result(self, step: int, metric: Optional[float], is_new_best: bool) -> None:
        """Called when a step's metric result becomes known (after API evaluates it).

        If metric is None, the solution was buggy/invalid.
        """
        pass

    @abstractmethod
    def on_run_completed(self, best_node: Optional[Node], best_metric_value: Optional[float], nodes: list) -> None:
        """Called when run completes successfully."""
        pass

    @abstractmethod
    def on_run_stopped(self) -> None:
        """Called when run is stopped by user."""
        pass

    @abstractmethod
    def on_error(self, message: str, run_id: Optional[str] = None) -> None:
        """Called when an error occurs."""
        pass

    @abstractmethod
    def on_warning(self, message: str) -> None:
        """Called for warning messages."""
        pass

    @abstractmethod
    def on_stop_requested(self) -> None:
        """Called when a stop request is received."""
        pass

    @abstractmethod
    def get_live_ref(self) -> Optional[Live]:
        """Get the Live reference for signal handlers (rich mode only)."""
        pass


class RichOutputHandler(OutputHandler):
    """Rich output handler using Live displays, panels, and smooth updates."""

    def __init__(
        self, console: Console, metric_name: str, maximize: bool, total_steps: int, model: str, runs_dir: str, source_fp
    ):
        self.console = console
        self.metric_name = metric_name
        self.maximize = maximize
        self.total_steps = total_steps
        self.model = model
        self.runs_dir = runs_dir
        self.source_fp = source_fp

        # Create panels
        self.summary_panel = SummaryPanel(
            maximize=maximize, metric_name=metric_name, total_steps=total_steps, model=model, runs_dir=runs_dir
        )
        self.solution_panels = SolutionPanels(metric_name=metric_name, source_fp=source_fp)
        self.eval_output_panel = EvaluationOutputPanel()
        self.tree_panel = MetricTreePanel(maximize=maximize)
        self.layout = create_optimization_layout()
        self.end_optimization_layout = create_end_optimization_layout()

        self.live: Optional[Live] = None
        self.current_step = 0

    def __enter__(self):
        self.live = Live(self.layout, refresh_per_second=4)
        self.live.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.live:
            self.live.__exit__(exc_type, exc_val, exc_tb)

    def get_live_ref(self) -> Optional[Live]:
        return self.live

    def _update_display(self, is_done: bool = False) -> None:
        """Update the live display with current panel states."""
        if not self.live:
            return

        current_solution_panel, best_solution_panel = self.solution_panels.get_display(current_step=self.current_step)
        smooth_update(
            live=self.live,
            layout=self.layout,
            sections_to_update=[
                ("summary", self.summary_panel.get_display()),
                ("tree", self.tree_panel.get_display(is_done=is_done)),
                ("current_solution", current_solution_panel),
                ("best_solution", best_solution_panel),
                ("eval_output", self.eval_output_panel.get_display()),
            ],
            transition_delay=0.1,
        )

    def on_run_started(
        self,
        run_id: str,
        run_name: str,
        dashboard_url: str,
        model: str,
        runs_dir: str,
        plan: str,
        initial_code: str,
        initial_solution_id: str,
    ) -> None:
        self.summary_panel.set_run_id(run_id=run_id)
        self.summary_panel.set_run_name(run_name=run_name)
        self.summary_panel.set_step(step=0)
        self.summary_panel.update_thinking(thinking=plan)

        # Build initial metric tree
        self.tree_panel.build_metric_tree(
            nodes=[
                {
                    "solution_id": initial_solution_id,
                    "parent_id": None,
                    "code": initial_code,
                    "step": 0,
                    "metric_value": None,
                    "is_buggy": None,
                }
            ]
        )
        self.tree_panel.set_unevaluated_node(node_id=initial_solution_id)

        # Update solution panels
        self.solution_panels.update(
            current_node=Node(id=initial_solution_id, parent_id=None, code=initial_code, metric=None, is_buggy=None),
            best_node=None,
        )

        self._update_display()

    def on_baseline_evaluating(self) -> None:
        # No visible change for rich mode - already showing initial state
        pass

    def on_baseline_completed(self, eval_output: str, best_metric: Optional[float] = None) -> None:
        self.eval_output_panel.update(output=eval_output)
        if self.live:
            smooth_update(
                live=self.live,
                layout=self.layout,
                sections_to_update=[("eval_output", self.eval_output_panel.get_display())],
                transition_delay=0.1,
            )

    def on_step_starting(self, step: int, previous_best_metric: Optional[float] = None) -> None:
        # No visible change for rich mode - Live display updates automatically
        pass

    def on_step_generated(
        self, step: int, code: str, plan: str, nodes: list, current_node: Node, best_node: Optional[Node], solution_id: str
    ) -> None:
        self.current_step = step
        self.summary_panel.set_step(step=step)
        self.summary_panel.update_thinking(thinking=plan)

        self.tree_panel.build_metric_tree(nodes=nodes)
        self.tree_panel.set_unevaluated_node(node_id=solution_id)

        self.solution_panels.update(current_node=current_node, best_node=best_node)
        self.eval_output_panel.clear()

        self._update_display()

    def on_step_completed(self, step: int, eval_output: str) -> None:
        self.eval_output_panel.update(output=eval_output)
        if self.live:
            smooth_update(
                live=self.live,
                layout=self.layout,
                sections_to_update=[("eval_output", self.eval_output_panel.get_display())],
                transition_delay=0.1,
            )

    def on_step_result(self, step: int, metric: Optional[float], is_new_best: bool) -> None:
        # Rich mode updates the tree panel which shows metrics - no separate action needed
        pass

    def on_run_completed(self, best_node: Optional[Node], best_metric_value: Optional[float], nodes: list) -> None:
        self.summary_panel.set_step(step=self.total_steps)
        self.tree_panel.build_metric_tree(nodes=nodes)

        self.solution_panels.update(current_node=None, best_node=best_node)
        _, best_solution_panel = self.solution_panels.get_display(current_step=self.total_steps)

        final_message = (
            f"{self.metric_name.capitalize()} {'maximized' if self.maximize else 'minimized'}! "
            f"Best solution {self.metric_name.lower()} = [green]{best_metric_value}[/] 🏆"
            if best_node is not None and best_node.metric is not None
            else "[red] No valid solution found.[/]"
        )

        self.end_optimization_layout["summary"].update(self.summary_panel.get_display(final_message=final_message))
        self.end_optimization_layout["tree"].update(self.tree_panel.get_display(is_done=True))
        self.end_optimization_layout["best_solution"].update(best_solution_panel)

        if self.live:
            self.live.update(self.end_optimization_layout)

    def on_run_stopped(self) -> None:
        self.console.print("[yellow]Run terminated by user request.[/]")

    def on_error(self, message: str, run_id: Optional[str] = None) -> None:
        from rich.panel import Panel

        self.console.print(Panel(f"[bold red]Error: {message}", title="[bold red]Optimization Error", border_style="red"))
        if run_id:
            self.console.print(f"\n[cyan]To resume this run, use:[/] [bold cyan]weco resume {run_id}[/]\n")

    def on_warning(self, message: str) -> None:
        self.console.print(f"\n[bold red]Warning: {message}[/]")

    def on_stop_requested(self) -> None:
        self.console.print("\n[bold yellow]Stop request received. Terminating run gracefully...[/]")


class PlainOutputHandler(OutputHandler):
    """Plain output handler using simple console.print with colors and spinners."""

    def __init__(self, console: Console, metric_name: str, maximize: bool, total_steps: int):
        self.console = console
        self.metric_name = metric_name
        self.maximize = maximize
        self.total_steps = total_steps
        self.run_id: Optional[str] = None
        self._status = None  # Active spinner status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop_spinner()

    def _stop_spinner(self):
        """Stop any active spinner."""
        if self._status is not None:
            self._status.stop()
            self._status = None

    def _start_spinner(self, message: str):
        """Start a new spinner with the given message."""
        self._stop_spinner()
        self._status = self.console.status(message, spinner="dots")
        self._status.start()

    def get_live_ref(self) -> Optional[Live]:
        return None

    def on_run_started(
        self,
        run_id: str,
        run_name: str,
        dashboard_url: str,
        model: str,
        runs_dir: str,
        plan: str,
        initial_code: str,
        initial_solution_id: str,
    ) -> None:
        self.run_id = run_id
        goal = "Maximize" if self.maximize else "Minimize"
        self.console.print("[cyan]Optimization started[/]")
        self.console.print(f"├─ Run:       {run_id}")
        self.console.print(f"├─ Dashboard: [link={dashboard_url}]{dashboard_url}[/link]")
        self.console.print(f"├─ Model:     {model}")
        self.console.print(f"└─ Objective: {goal} {self.metric_name}")
        self.console.print()

    def on_baseline_evaluating(self) -> None:
        self._start_spinner("[dim]Evaluating baseline...[/]")

    def on_baseline_completed(self, eval_output: str, best_metric: Optional[float] = None) -> None:
        self._stop_spinner()
        if best_metric is not None:
            self.console.print(f"Baseline {self.metric_name} = {best_metric:.4f}")

    def on_step_starting(self, step: int, previous_best_metric: Optional[float] = None) -> None:
        # Don't start spinner here - wait until previous result is shown
        pass

    def on_step_generated(
        self, step: int, code: str, plan: str, nodes: list, current_node: Node, best_node: Optional[Node], solution_id: str
    ) -> None:
        self._start_spinner(f"[dim][{step}/{self.total_steps}] Evaluating...[/]")

    def on_step_completed(self, step: int, eval_output: str) -> None:
        self._stop_spinner()
        # Start spinner for the analysis phase (API call that evaluates this step's output)
        self._start_spinner(f"[dim][{step}/{self.total_steps}] Analyzing results...[/]")

    def on_step_result(self, step: int, metric: Optional[float], is_new_best: bool) -> None:
        self._stop_spinner()  # Ensure no spinner is running when printing result
        if metric is None:
            self.console.print(f"[dim][{step}/{self.total_steps}] buggy solution[/]")
        elif is_new_best:
            self.console.print(f"[green][{step}/{self.total_steps}] {self.metric_name} = {metric:.4f} ★ new best[/]")
        else:
            self.console.print(f"[{step}/{self.total_steps}] {self.metric_name} = {metric:.4f}")

    def on_run_completed(self, best_node: Optional[Node], best_metric_value: Optional[float], nodes: list) -> None:
        self._stop_spinner()
        self.console.print()
        if best_metric_value is not None:
            self.console.print("[bold green]Optimization complete![/]")
            self.console.print(f"Best {self.metric_name}: [green]{best_metric_value:.4f}[/]")
        else:
            self.console.print("[bold yellow]Optimization complete![/]")
            self.console.print("[red]No valid solution found.[/]")

    def on_run_stopped(self) -> None:
        self._stop_spinner()
        self.console.print("[yellow]Run terminated by user request.[/]")

    def on_error(self, message: str, run_id: Optional[str] = None) -> None:
        self._stop_spinner()
        self.console.print(f"\n[bold red]Error:[/] {message}")
        if run_id:
            self.console.print(f"\n[cyan]To resume this run, use:[/] [bold cyan]weco resume {run_id}[/]\n")

    def on_warning(self, message: str) -> None:
        self._stop_spinner()
        self.console.print(f"\n[bold red]Warning: {message}[/]")

    def on_stop_requested(self) -> None:
        self._stop_spinner()
        self.console.print("\n[bold yellow]Stop request received. Terminating run gracefully...[/]")


def create_output_handler(
    output_mode: str,
    console: Console,
    metric_name: str,
    maximize: bool,
    total_steps: int,
    model: str = "",
    runs_dir: str = "",
    source_fp=None,
) -> OutputHandler:
    """Factory function to create the appropriate output handler."""
    if output_mode == "rich":
        return RichOutputHandler(
            console=console,
            metric_name=metric_name,
            maximize=maximize,
            total_steps=total_steps,
            model=model,
            runs_dir=runs_dir,
            source_fp=source_fp,
        )
    else:
        return PlainOutputHandler(console=console, metric_name=metric_name, maximize=maximize, total_steps=total_steps)
