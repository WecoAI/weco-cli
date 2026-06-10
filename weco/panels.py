from rich.tree import Tree
from rich.table import Table
from rich.progress import BarColumn, Progress, TextColumn
from rich.layout import Layout
from rich.panel import Panel
from rich.syntax import Syntax
from rich import box
from rich.console import Console

from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
from .__init__ import __dashboard_url__


class SummaryPanel:
    """Holds a summary of the optimization run."""

    def __init__(
        self,
        maximize: bool,
        metric_name: str,
        total_steps: int,
        model: str,
        runs_dir: str,
        run_id: str = None,
        run_name: str = None,
        console: Optional[Console] = None,
    ):
        self.maximize = maximize
        self.metric_name = metric_name
        self.total_steps = total_steps
        self.model = model
        self.runs_dir = runs_dir
        self.run_id = run_id if run_id is not None else "N/A"
        self.run_name = run_name if run_name is not None else "N/A"
        self.dashboard_url = "N/A"
        self.thinking_content = ""
        self.user_input = ""
        self.console = Console()
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=20),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TextColumn("[bold]{task.completed}/{task.total} Steps "),
            expand=False,
        )
        self.task_id = self.progress.add_task("", total=total_steps)

    def set_run_id(self, run_id: str):
        """Set the run ID."""
        self.run_id = run_id
        self.set_dashboard_url(run_id=run_id)

    def set_run_name(self, run_name: str):
        """Set the run name."""
        self.run_name = run_name

    def set_dashboard_url(self, run_id: str):
        """Set the dashboard URL."""
        self.dashboard_url = f"{__dashboard_url__}/runs/{run_id}"

    def set_step(self, step: int):
        """Set the current step."""
        self.progress.update(self.task_id, completed=step)

    def update_thinking(self, thinking: str):
        """Update the thinking content."""
        self.thinking_content = thinking

    def clear_thinking(self):
        """Clear the thinking content."""
        self.thinking_content = ""

    def get_display(self, final_message: Optional[str] = None) -> Panel:
        """Return a Rich panel summarising the current run."""
        # ───────────────────── summary grid ──────────────────────
        summary_table = Table.grid(expand=True, padding=(0, 1))
        summary_table.add_column(ratio=1)
        summary_table.add_column(justify="right")
        summary_table.add_row("")

        # Run id
        summary_table.add_row(f" Run ID: [bold cyan]{self.run_id}[/]")
        summary_table.add_row("")

        # Dashboard url
        summary_table.add_row(f" Dashboard: [underline blue]{self.dashboard_url}[/]")
        summary_table.add_row("")

        if final_message is not None:
            # Add the final message
            summary_table.add_row(f"[bold cyan] Result:[/] {final_message}", "")
            summary_table.add_row("")

        # Model info
        summary_table.add_row(f" Model: [bold cyan]{self.model}[/]")
        summary_table.add_row("")

        # Progress bar
        summary_table.add_row(self.progress)
        summary_table.add_row("")

        # Logs url
        logs_url = Path(self.runs_dir) / self.run_id
        summary_table.add_row(f" Logs: [underline blue]{logs_url}[/]")
        summary_table.add_row("")

        if final_message is not None:
            # Don't include the thinking section
            return Panel(
                summary_table,
                title=f"[bold]📊 {'Maximizing' if self.maximize else 'Minimizing'} {self.run_name}",
                border_style="magenta",
                expand=True,
                padding=(0, 1),
            )

        # Include the thinking section
        layout = Layout(name="summary")
        layout.split_column(
            Layout(summary_table, name="main_summary", ratio=1),
            Layout(
                Panel(
                    self.thinking_content or "[dim]No thinking content yet...[/]",
                    title="[bold]📝 Thinking...",
                    border_style="cyan",
                    expand=True,
                    padding=(0, 1),
                ),
                name="thinking_section",
                ratio=1,
            ),
        )

        return Panel(
            layout,
            title=f"[bold]📊 {'Maximizing' if self.maximize else 'Minimizing'} {self.run_name}",
            border_style="magenta",
            expand=True,
            padding=(0, 1),
        )


class Node:
    """Represents a node in the solution tree."""

    def __init__(
        self,
        id: str,
        parent_id: Union[str, None],
        code: Union[str, None],
        metric: Union[float, None],
        is_buggy: Union[bool, None],
    ):
        self.id = id
        self.parent_id = parent_id
        self.children: List["Node"] = []
        self.code = code
        self.metric = metric
        self.is_buggy = is_buggy
        self.evaluated = True
        self.name = ""


class MetricTree:
    """Manages the tree structure of optimization solutions."""

    def __init__(self, maximize: bool):
        self.nodes: Dict[str, Node] = {}
        self.maximize = maximize

    def clear(self):
        """Clear the tree."""
        self.nodes = {}

    def add_node(self, node: Node):
        """Add a node to the tree."""
        # Add the node to the tree
        self.nodes[node.id] = node

        # Add node to node's parent's children
        if node.parent_id is not None:
            if node.parent_id not in self.nodes:
                raise ValueError("Cannot construct optimization tree.")
            self.nodes[node.parent_id].children.append(node)

    def get_root_node(self) -> Node:
        """Get the root node from the tree."""
        nodes = [node for node in self.nodes.values() if node.parent_id is None]
        if len(nodes) != 1:
            raise ValueError("Cannot construct optimization tree.")
        return nodes[0]

    def get_best_node(self) -> Optional[Node]:
        """Get the best node from the tree."""
        measured_nodes = [
            node
            for node in self.nodes.values()
            if node.evaluated  # evaluated
            and node.is_buggy
            is False  # not buggy => is_buggy can exist in 3 states: None (solution has not yet been evaluated for bugs), True (solution has bug), False (solution does not have a bug)
            and node.metric is not None  # has metric
        ]
        if len(measured_nodes) == 0:
            return None
        if self.maximize:
            return max(measured_nodes, key=lambda node: node.metric)
        else:
            return min(measured_nodes, key=lambda node: node.metric)


class MetricTreePanel:
    """Displays the solution tree with depth limiting."""

    def __init__(self, maximize: bool):
        self.metric_tree = MetricTree(maximize=maximize)

    def build_metric_tree(self, nodes: List[dict]):
        """Build the tree from the list of nodes."""
        # Defensive: treat None as empty list
        if nodes is None:
            nodes = []
        # First clear then tree
        self.metric_tree.clear()

        # Then sort the nodes by step number
        nodes.sort(key=lambda x: x["step"])

        # Finally build the new tree
        for i, node in enumerate(nodes):
            node = Node(
                id=node["solution_id"],
                parent_id=node["parent_id"],
                code=node["code"],
                metric=node["metric_value"],
                is_buggy=node["is_buggy"],
            )
            if i == 0:
                node.name = "baseline"
            self.metric_tree.add_node(node)

    def set_unevaluated_node(self, node_id: str):
        """Set the unevaluated node."""
        self.metric_tree.nodes[node_id].evaluated = False

    def _build_rich_tree(self) -> Tree:
        """Get a Rich Tree representation of the solution tree using a DFS like traversal."""
        if len(self.metric_tree.nodes) == 0:
            return Tree("[bold green]Building first solution...")

        best_node = self.metric_tree.get_best_node()

        def append_rec(node: Node, tree: Tree):
            if not node.evaluated:
                # not evaluated
                color = "yellow"
                style = None
                text = "evaluating..."
            elif node.is_buggy:
                # buggy node
                color = "red"
                style = None
                text = "bug"
            else:
                # evaluated non-buggy node
                if node.id == best_node.id:
                    # best node
                    color = "green"
                    style = "bold"
                    text = f"{node.metric:.3f} 🏆"
                elif node.metric is None:
                    # metric not extracted from evaluated solution
                    color = "yellow"
                    style = None
                    text = "N/A"
                else:
                    # evaluated node with metric
                    color = "green"
                    style = None
                    text = f"{node.metric:.3f}"

                # add the node name info
                text = f"{node.name} {text}".strip()

            s = f"[{f'{style} ' if style is not None else ''}{color}]● {text}"
            subtree = tree.add(s)
            for child in node.children:
                append_rec(child, subtree)

        tree = Tree("", hide_root=True)
        root_node = self.metric_tree.get_root_node()
        append_rec(node=root_node, tree=tree)

        return tree

    def get_display(self, is_done: bool) -> Panel:
        """Get a panel displaying the solution tree."""
        # Make sure the metric tree is built before calling build_rich_tree
        return Panel(
            self._build_rich_tree(),
            title=("[bold]🔎 Exploring Solutions..." if not is_done else "[bold]🔎 Optimization Complete!"),
            border_style="green",
            expand=True,
            padding=(0, 1),
        )


class EvaluationOutputPanel:
    """Displays evaluation output with truncation for long outputs."""

    def __init__(self):
        self.output = ""

    def update(self, output: str) -> None:
        """Update the evaluation output."""
        self.output = output

    def clear(self) -> None:
        """Clear the evaluation output."""
        self.output = ""

    def get_display(self) -> Panel:
        """Create a panel displaying the evaluation output with truncation if needed."""
        return Panel(self.output, title="[bold]📋 Evaluation Output", border_style="blue", expand=True, padding=(0, 1))


class SolutionPanels:
    """Displays the current and best solutions side by side."""

    def __init__(self, metric_name: str, source_fp: Path):
        # Current solution
        self.current_node = None
        # Best solution
        self.best_node = None
        # Metric name
        self.metric_name = metric_name.capitalize()
        # Determine the lexer for the source file
        self.lexer = self._determine_lexer(source_fp)

    def _determine_lexer(self, source_fp: Path) -> str:
        """Determine the lexer for the source file."""
        return Syntax.from_path(source_fp).lexer

    def update(self, current_node: Union[Node, None], best_node: Union[Node, None]):
        """Update the current and best solutions."""
        # Update current solution
        self.current_node = current_node
        # Update best solution
        self.best_node = best_node

    def get_display(self, current_step: int) -> Tuple[Panel, Panel]:
        """Return the current and best solutions as panels."""
        current_code = self.current_node.code if self.current_node is not None else ""
        best_code = self.best_node.code if self.best_node is not None else ""
        best_score = self.best_node.metric if self.best_node is not None else None

        # Current solution (without score)
        current_title = f"[bold]💡 Current Solution (Step {current_step})"
        current_panel = Panel(
            Syntax(str(current_code), self.lexer, theme="monokai", line_numbers=True, word_wrap=False),
            title=current_title,
            border_style="yellow",
            expand=True,
            padding=(0, 1),
        )

        # Best solution
        best_title = f"[bold]🏆 Best Solution ([green]{self.metric_name}: {f'{best_score:.4f}' if best_score is not None else 'N/A'}[/])"
        best_panel = Panel(
            Syntax(str(best_code), self.lexer, theme="monokai", line_numbers=True, word_wrap=False),
            title=best_title,
            border_style="green",
            expand=True,
            padding=(0, 1),
        )

        return current_panel, best_panel


def create_optimization_layout() -> Layout:
    """Create the main layout for the CLI."""
    layout = Layout()

    # First split into top, middle, and bottom sections
    layout.split_column(
        Layout(name="top_section", ratio=3), Layout(name="middle_section", ratio=4), Layout(name="eval_output", ratio=2)
    )

    # Split the top section into left and right
    layout["top_section"].split_row(Layout(name="summary", ratio=1), Layout(name="tree", ratio=1))

    # Split the middle section into left and right
    layout["middle_section"].split_row(Layout(name="current_solution", ratio=1), Layout(name="best_solution", ratio=1))

    return layout


def create_end_optimization_layout() -> Layout:
    """Create the final layout after optimization is complete."""
    layout = Layout()

    # Create a top section for summary
    layout.split_column(Layout(name="summary", ratio=1), Layout(name="bottom_section", ratio=3))

    # Split the bottom section into left (best solution) and right ( tree)
    layout["bottom_section"].split_row(Layout(name="best_solution", ratio=1), Layout(name="tree", ratio=1))

    return layout


class OptimizationOptionsPanel:
    """Panel for displaying optimization options in a table.

    Creates a formatted table showing optimization suggestions with details
    like target file, description, estimated cost, and predicted gains.
    """

    def get_display(self, options: List[Dict[str, str]]) -> Table:
        """Create optimization options table as a renderable object."""
        table = Table(title="Optimization Options", show_lines=True, box=box.ROUNDED, border_style="cyan", padding=(1, 1))
        table.add_column("No.", style="bold white", width=5, header_style="bold white", justify="center")
        table.add_column("Target File", style="cyan", width=20, header_style="bold white")
        table.add_column("Description", style="magenta", width=40, header_style="bold white")
        table.add_column("Est. Token Cost", style="yellow", width=15, header_style="bold white")
        table.add_column("Pred. Perf. Gain", style="green", width=20, header_style="bold white")

        for i, opt in enumerate(options):
            table.add_row(
                str(i + 1),
                opt["target_file"],
                opt["description"],
                opt["estimated_token_cost"],
                opt["predicted_performance_gain"],
            )
        return table


class EvaluationScriptPanel:
    """Panel for displaying evaluation scripts with syntax highlighting.

    Shows Python evaluation scripts with proper syntax highlighting,
    line numbers, and a descriptive title.
    """

    def get_display(self, script_content: str, script_path: str = "evaluate.py") -> Panel:
        """Create a panel displaying the evaluation script with syntax highlighting."""
        return Panel(
            Syntax(script_content, "python", theme="monokai", line_numbers=True),
            title=f"[bold]📄 Evaluation Script: {script_path}",
            border_style="cyan",
            expand=True,
            padding=(0, 1),
        )
