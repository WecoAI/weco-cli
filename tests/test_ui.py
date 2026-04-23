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


def _make_plain_ui_with_capture() -> tuple[PlainOptimizationUI, list[str]]:
    """Construct a PlainOptimizationUI that records output to a list instead of stdout."""
    ui = PlainOptimizationUI(
        run_id="run-1",
        run_name="demo",
        total_steps=5,
        dashboard_url="https://example.com",
        model="gpt-4",
        metric_name="accuracy",
    )
    lines: list[str] = []
    ui._print = lines.append
    return ui, lines


def test_plain_ui_on_init_prints_header():
    """on_init prints the run banner exactly once. (Header used to be printed
    by __enter__; the move to on_init keeps the same observable behavior for
    non-derived runs.)"""
    ui, lines = _make_plain_ui_with_capture()

    ui.on_init()

    output = "\n".join(lines)
    assert "WECO OPTIMIZATION RUN" in output
    assert "Run ID: run-1" in output
    assert "Run Name: demo" in output
    assert "Dashboard: https://example.com" in output
    assert "Model: gpt-4" in output
    assert "Metric: accuracy" in output
    assert "Total Steps: 5" in output
    # Non-derived run: no "Derived from" line
    assert not any("Derived from" in line for line in lines)


def test_plain_ui_on_init_includes_derived_from_line():
    ui, lines = _make_plain_ui_with_capture()

    ui.on_init(derived_from={"run_id": "parent-uuid", "node_id": "node-uuid", "step": 7, "metric_value": 0.842})

    derived_lines = [line for line in lines if "Derived from" in line]
    assert len(derived_lines) == 1
    assert "parent-uuid" in derived_lines[0]
    assert "step 7" in derived_lines[0]
    assert "0.842" in derived_lines[0]


def test_plain_ui_on_init_handles_derived_from_without_metric():
    """A node with no metric_value (e.g., still pending eval) shouldn't crash
    the header rendering."""
    ui, lines = _make_plain_ui_with_capture()

    ui.on_init(derived_from={"run_id": "parent-uuid", "node_id": "node-uuid", "step": 0, "metric_value": None})

    derived_lines = [line for line in lines if "Derived from" in line]
    assert len(derived_lines) == 1
    assert "parent-uuid" in derived_lines[0]
    assert "step 0" in derived_lines[0]
    # No "(metric: ...)" suffix when metric_value is None. Specific to the
    # suffix's literal form so the assertion is robust to metric_name values
    # that happen to contain the substring "metric".
    assert "(metric:" not in derived_lines[0]


def test_plain_ui_enter_no_longer_prints_header():
    """Header printing must happen via on_init now, not __enter__. This guards
    against accidentally re-introducing the auto-print and double-printing the
    header."""
    ui, lines = _make_plain_ui_with_capture()

    with ui:
        pass

    assert lines == []


def test_live_ui_on_init_renders_derived_from_row():
    """The Live panel grid gains a "From" row when derived_from is set."""
    ui = LiveOptimizationUI(
        console=Console(force_terminal=False, color_system=None),
        run_id="run-1",
        run_name="demo",
        total_steps=3,
        dashboard_url="https://example.com",
        metric_name="accuracy",
    )

    ui.on_init(derived_from={"run_id": "parent-uuid", "node_id": "node-uuid", "step": 4, "metric_value": 0.91})

    text = _render_to_text(ui._render())
    assert "parent-uuid" in text
    assert "step 4" in text
    assert "0.91" in text


def test_live_ui_on_init_without_derived_from_does_not_render_from_row():
    """Negative cousin to test_live_ui_on_init_renders_derived_from_row.

    Renders the same UI both with and without ``derived_from`` and asserts
    the parent reference (the marker added by the "From" row) appears only
    in the derived render. Comparing the two renders directly avoids
    fragile substring checks against unrelated panel chrome.
    """

    def render(derived_from):
        ui = LiveOptimizationUI(
            console=Console(force_terminal=False, color_system=None),
            run_id="run-1",
            run_name="demo",
            total_steps=3,
            dashboard_url="https://example.com",
        )
        ui.on_init(derived_from=derived_from)
        return _render_to_text(ui._render())

    derived_text = render({"run_id": "parent-uuid", "node_id": "n", "step": 4, "metric_value": 0.91})
    plain_text = render(None)

    # The parent reference is added only by the "From" row, so its presence
    # in one render and absence in the other proves the row is conditional.
    assert "parent-uuid" in derived_text
    assert "parent-uuid" not in plain_text
    # Belt and braces: the row label itself only appears in the derived
    # render (no other panel label contains "From" as a substring).
    assert "From" not in plain_text
