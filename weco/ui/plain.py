"""Plain-text implementation of ``OptimizationUI`` for LLM agents and machine consumers."""

from typing import List, Optional


class PlainOptimizationUI:
    """
    Plain text implementation of OptimizationUI for machine-readable output.

    Designed to be consumed by LLM agents - outputs structured, parseable text
    without Rich formatting, ANSI codes, or interactive elements.
    Includes full execution output for agent consumption.
    """

    def __init__(
        self,
        run_id: str,
        run_name: str,
        total_steps: int,
        dashboard_url: str,
        model: str = "",
        metric_name: str = "",
        maximize: bool = True,
    ):
        self.run_id = run_id
        self.run_name = run_name
        self.total_steps = total_steps
        self.dashboard_url = dashboard_url
        self.model = model
        self.metric_name = metric_name
        self.maximize = maximize
        self.current_step = 0
        self.metrics: List[tuple] = []  # (step, value)
        self._header_printed = False
        self._derived_from: Optional[dict] = None  # populated by on_init for derived runs

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
        if self._derived_from:
            df = self._derived_from
            metric = df.get("metric_value")
            metric_suffix = f" (metric: {metric:.6g})" if metric is not None else ""
            self._print(f"Derived from: {df['run_id']} @ step {df['step']}{metric_suffix}")
        self._print("=" * 60)
        self._print("")

    # --- Context manager (header printing happens via on_init, not __enter__) ---
    def __enter__(self) -> "PlainOptimizationUI":
        return self

    def __exit__(self, *args) -> None:
        pass

    # --- OptimizationUI Protocol Implementation ---
    def on_init(self, derived_from: Optional[dict] = None) -> None:
        """Print the run header. Includes a "Derived from" line when the run
        was created via ``weco run derive``."""
        self._derived_from = derived_from
        self._print_header()

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
        best_fn = max if self.maximize else min
        best = best_fn(m[1] for m in self.metrics) if self.metrics else value
        self._print(f"[METRIC] Step {step}: {value:.6g} (best so far: {best:.6g})")

    def on_complete(self, total_steps: int) -> None:
        self._print("")
        self._print("=" * 60)
        self._print("[COMPLETE] Optimization finished successfully")
        self._print(f"Total steps completed: {total_steps}")
        if self.metrics:
            values = [m[1] for m in self.metrics]
            best_fn = max if self.maximize else min
            self._print(f"Best metric value: {best_fn(values):.6g}")
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
