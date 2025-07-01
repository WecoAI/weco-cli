import cProfile
import pstats
import io
import pathlib
import tempfile
from unittest.mock import patch

from rich.console import Console

# --------------------
# Dummy backend helpers
# --------------------

def _dummy_start_optimization_run(**kwargs):
    # Minimal payload resembling real backend response
    return {
        "run_id": "dummy_run",
        "code": "# dummy generated code\nprint('hello world')\n",
        "solution_id": "sol_0",
        "plan": "Initial plan",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


def _dummy_evaluate_feedback_then_suggest_next_solution(**kwargs):
    step = getattr(_dummy_evaluate_feedback_then_suggest_next_solution, "_step", 0) + 1
    _dummy_evaluate_feedback_then_suggest_next_solution._step = step
    return {
        "code": f"# dummy solution step {step}\nprint('step {step}')\n",
        "solution_id": f"sol_{step}",
        "plan": f"Plan for step {step}",
        "usage": {"prompt_tokens": 8, "completion_tokens": 4},
    }


def _dummy_get_optimization_run_status(**kwargs):
    step = getattr(_dummy_evaluate_feedback_then_suggest_next_solution, "_step", 0)
    # Build synthetic node list for tree panel
    nodes = [
        {
            "solution_id": f"sol_{i}",
            "parent_id": None if i == 0 else f"sol_{i-1}",
            "code": f"# code {i}",
            "step": i,
            "metric_value": i * 1.0,
            "is_buggy": False,
        }
        for i in range(step + 1)
    ]
    best_node = nodes[-1]
    return {
        "status": "running",
        "nodes": nodes,
        "best_result": {
            "solution_id": best_node["solution_id"],
            "parent_id": best_node["parent_id"],
            "code": best_node["code"],
            "metric_value": best_node["metric_value"],
            "is_buggy": False,
        },
    }


# Heartbeat / termination just return True so optimizer proceeds
_dummy_true = lambda *args, **kwargs: True  # noqa: E731


# Mock for utils.run_evaluation – returns constant output

def _dummy_run_evaluation(eval_command: str) -> str:  # noqa: D401
    return "metric_value: 1.0"


# No-op write to avoid file I/O
_no_op = lambda *args, **kwargs: None  # noqa: E731


# --------------------
# Main test runner
# --------------------

def run_test():
    import weco.optimizer as optimizer

    example_source = pathlib.Path(tempfile.gettempdir()) / "dummy_source.py"
    example_source.write_text("# baseline code\nprint('baseline')\n")

    console = Console(file=io.StringIO())  # discard rich output

    with patch("weco.api.start_optimization_run", _dummy_start_optimization_run), \
         patch("weco.api.evaluate_feedback_then_suggest_next_solution", _dummy_evaluate_feedback_then_suggest_next_solution), \
         patch("weco.api.get_optimization_run_status", _dummy_get_optimization_run_status), \
         patch("weco.api.send_heartbeat", _dummy_true), \
         patch("weco.api.report_termination", _dummy_true), \
         patch("weco.auth.handle_authentication", lambda *a, **k: ("dummy", {})), \
         patch("weco.utils.read_api_keys_from_env", lambda: {"OPENAI_API_KEY": "dummy"}), \
         patch("weco.utils.run_evaluation", _dummy_run_evaluation), \
         patch("weco.utils.write_to_path", _no_op):

        optimizer.execute_optimization(
            source=str(example_source),
            eval_command="echo hi",  # not executed thanks to mock
            metric="accuracy",
            goal="maximize",
            steps=3,
            model="o4-mini",
            log_dir=tempfile.gettempdir(),
            additional_instructions=None,
            console=console,
        )


if __name__ == "__main__":
    profile_file = "weco_profile.prof"
    cProfile.run("run_test()", profile_file)
    print("\n--- Profiling Summary (top 30 cumulative) ---")
    stats = pstats.Stats(profile_file)
    stats.strip_dirs().sort_stats("cumulative").print_stats(30)