from rich.tree import Tree
from rich.table import Table
from rich.progress import BarColumn, Progress, TextColumn
from rich.layout import Layout
from rich.panel import Panel
from rich.syntax import Syntax
from typing import Dict, List, Optional, Union, Tuple
from .utils import format_number
import pathlib
from .__init__ import __dashboard_url__


class SummaryPanel:
    """Holds a summary of the optimization run."""

    def __init__(self, maximize: bool, metric_name: str, total_steps: int, model: str, runs_dir: str, run_id: str = None):
        self.maximize = maximize
        self.metric_name = metric_name
        self.goal = ("Maximizing" if self.maximize else "Minimizing") + f" {self.metric_name}..."
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_steps = total_steps
        self.model = model
        self.runs_dir = runs_dir
        self.run_id = run_id if run_id is not None else "N/A"
        self.dashboard_url = "N/A"
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=20),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TextColumn("[bold]{task.completed}/{task.total} Steps"),
            expand=False,
        )
        self.task_id = self.progress.add_task("", total=total_steps)

    def set_run_id(self, run_id: str):
        """Set the run ID."""
        self.run_id = run_id
        self.set_dashboard_url(run_id=run_id)

    def set_dashboard_url(self, run_id: str):
        """Set the dashboard URL."""
        self.dashboard_url = f"{__dashboard_url__}/runs/{run_id}"

    def set_step(self, step: int):
        """Set the current step."""
        self.progress.update(self.task_id, completed=step)

    def update_token_counts(self, usage: Dict[str, int]):
        """Update token counts from usage data."""
        if not isinstance(usage, dict) or "input_tokens" not in usage or "output_tokens" not in usage:
            raise ValueError("Invalid token usage response from API.")
        self.total_input_tokens += usage["input_tokens"]
        self.total_output_tokens += usage["output_tokens"]

    def get_display(self, final_message: Optional[str] = None) -> Panel:
        """Create a summary panel with the relevant information."""
        layout = Layout(name="summary")
        summary_table = Table(show_header=False, box=None, padding=(0, 1))

        summary_table.add_row("")
        # Goal
        if final_message is not None:
            summary_table.add_row(f"[bold cyan]Result:[/] {final_message}")
        else:
            summary_table.add_row(f"[bold cyan]Goal:[/] {self.goal}")
        summary_table.add_row("")
        # Model used
        summary_table.add_row(f"[bold cyan]Model:[/] {self.model}")
        summary_table.add_row("")
        # Log directory
        summary_table.add_row(f"[bold cyan]Logs:[/] [blue underline]{self.runs_dir}/{self.run_id}[/]")
        summary_table.add_row("")
        # Dashboard link
        summary_table.add_row(f"[bold cyan]Dashboard:[/] [blue underline]{self.dashboard_url}[/]")
        summary_table.add_row("")
        # Token counts
        summary_table.add_row(
            f"[bold cyan]Tokens:[/] ↑[yellow]{format_number(self.total_input_tokens)}[/] ↓[yellow]{format_number(self.total_output_tokens)}[/] = [green]{format_number(self.total_input_tokens + self.total_output_tokens)}[/]"
        )
        summary_table.add_row("")
        # Progress bar
        summary_table.add_row(self.progress)

        # Update layout
        layout.update(summary_table)

        return Panel(layout, title="[bold]📊 Summary", border_style="magenta", expand=True, padding=(0, 1))


class PlanPanel:
    """Displays the optimization plan with truncation for long plans."""

    def __init__(self):
        self.plan = ""

    def update(self, plan: str):
        """Update the plan text."""
        self.plan = plan

    def clear(self):
        """Clear the plan text."""
        self.plan = ""

    def get_display(self) -> Panel:
        """Create a panel displaying the plan with truncation if needed."""
        return Panel(self.plan, title="[bold]📝 Thinking...", border_style="cyan", expand=True, padding=(0, 1))


class Node:
    """Represents a node in the solution tree."""

    def __init__(
        self, id: str, parent_id: Union[str, None], code: Union[str, None], metric: Union[float, None], is_buggy: bool
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
                raise ValueError("Could not construct tree: parent node not found.")
            self.nodes[node.parent_id].children.append(node)

    def get_draft_nodes(self) -> List[Node]:
        """Get all draft nodes from the tree."""
        return [node for node in self.nodes.values() if node.parent_id is None]

    def get_best_node(self) -> Optional[Node]:
        """Get the best node from the tree."""
        measured_nodes = [
            node
            for node in self.nodes.values()
            if node.evaluated  # evaluated
            and not node.is_buggy  # not buggy
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
                if best_node is not None and node.id == best_node.id:
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
        for n in self.metric_tree.get_draft_nodes():
            append_rec(n, tree)

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

    def __init__(self, metric_name: str, source_fp: pathlib.Path):
        # Current solution
        self.current_node = None
        # Best solution
        self.best_node = None
        # Metric name
        self.metric_name = metric_name.capitalize()
        # Determine the lexer for the source file
        self.lexer = self._determine_lexer(source_fp)

    def _determine_lexer(self, source_fp: pathlib.Path) -> str:
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
    layout["top_section"].split_row(Layout(name="left_panels", ratio=1), Layout(name="tree", ratio=1))

    # Split the left panels into summary and thinking
    layout["left_panels"].split_column(Layout(name="summary", ratio=2), Layout(name="plan", ratio=1))

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
