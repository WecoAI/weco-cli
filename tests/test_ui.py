"""Tests for optimization UI best-metric display."""

from rich.console import Console

from weco.ui import LiveOptimizationUI, PlainOptimizationUI


def _render_to_text(renderable) -> str:
    console = Console(record=True, force_terminal=False, width=120, color_system=None)
    console.print(renderable)
    return console.export_text()


def test_live_optimization_ui_best_respects_minimize_goal():
    ui = LiveOptimizationUI(
        console=Console(force_terminal=False, color_system=None),
        run_id="run-1",
        run_name="demo",
        total_steps=3,
        dashboard_url="https://example.com",
        metric_name="val_bpb",
        maximize=False,
    )
    ui.state.metrics = [(0, 2.0), (1, 1.5), (2, 1.8)]

    text = _render_to_text(ui._render())

    assert "Best" in text
    assert "1.5" in text
    assert "2.0" not in text


def test_plain_optimization_ui_best_respects_minimize_goal():
    ui = PlainOptimizationUI(
        run_id="run-1",
        run_name="demo",
        total_steps=3,
        dashboard_url="https://example.com",
        metric_name="val_bpb",
        maximize=False,
    )

    lines: list[str] = []
    ui._print = lines.append  # Capture output without writing to stdout

    ui.on_metric(0, 2.0)
    ui.on_metric(1, 1.5)
    ui.on_metric(2, 1.8)
    ui.on_complete(3)

    assert any("best so far: 1.5" in line for line in lines)
    assert any("Best metric value: 1.5" in line for line in lines)
