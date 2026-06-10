"""Tests for the LangSmith evaluation bridge module."""

from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from weco.integrations.langsmith.bridge import (
    _adapt_target,
    _aggregate,
    _poll_dashboard_scores,
    import_target,
    main,
    resolve_evaluators,
    run_langsmith_eval,
)


# ---------------------------------------------------------------------------
# import_target
# ---------------------------------------------------------------------------


class TestImportTarget:
    def test_valid_import(self, tmp_path, monkeypatch):
        """Test importing a function from a temporary module."""
        mod_file = tmp_path / "sample_mod.py"
        mod_file.write_text("def my_func(x): return x * 2\n")
        monkeypatch.syspath_prepend(str(tmp_path))

        func = import_target("sample_mod:my_func")
        assert callable(func)
        assert func(3) == 6

    def test_missing_colon_raises(self):
        """Target spec must contain a colon."""
        with pytest.raises(ValueError, match="must be 'module:function'"):
            import_target("no_colon_here")

    def test_empty_module_raises(self):
        """Empty module part raises ValueError."""
        with pytest.raises(ValueError, match="must be non-empty"):
            import_target(":func")

    def test_empty_function_raises(self):
        """Empty function part raises ValueError."""
        with pytest.raises(ValueError, match="must be non-empty"):
            import_target("module:")

    def test_nonexistent_module_raises(self):
        """Importing a module that doesn't exist raises ImportError."""
        with pytest.raises(ImportError):
            import_target("definitely_nonexistent_module_xyz:func")

    def test_nonexistent_function_raises(self, tmp_path, monkeypatch):
        """Importing a function that doesn't exist in the module raises AttributeError."""
        mod_file = tmp_path / "empty_mod.py"
        mod_file.write_text("x = 1\n")
        monkeypatch.syspath_prepend(str(tmp_path))

        with pytest.raises(AttributeError):
            import_target("empty_mod:nonexistent_func")

    def test_non_callable_raises(self, tmp_path, monkeypatch):
        """If the attribute exists but isn't callable, raise TypeError."""
        mod_file = tmp_path / "const_mod.py"
        mod_file.write_text("MY_CONST = 42\n")
        monkeypatch.syspath_prepend(str(tmp_path))

        with pytest.raises(TypeError, match="expected a callable"):
            import_target("const_mod:MY_CONST")

    def test_dotted_module_path(self, tmp_path, monkeypatch):
        """Test importing from a dotted module path like pkg.mod:func."""
        pkg_dir = tmp_path / "test_pkg"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text("")
        (pkg_dir / "inner.py").write_text("def greet(): return 'hello'\n")
        monkeypatch.syspath_prepend(str(tmp_path))

        func = import_target("test_pkg.inner:greet")
        assert func() == "hello"

    def test_syntax_error_in_module(self, tmp_path, monkeypatch):
        """A syntax error in the source module raises SyntaxError."""
        mod_file = tmp_path / "broken_mod.py"
        mod_file.write_text("def bad(:\n")
        monkeypatch.syspath_prepend(str(tmp_path))

        with pytest.raises(SyntaxError):
            import_target("broken_mod:bad")


# ---------------------------------------------------------------------------
# _aggregate
# ---------------------------------------------------------------------------


class TestAggregate:
    def test_mean(self):
        assert _aggregate([1.0, 2.0, 3.0], "mean") == 2.0

    def test_median_odd(self):
        assert _aggregate([1.0, 2.0, 3.0], "median") == 2.0

    def test_median_even(self):
        assert _aggregate([1.0, 2.0, 3.0, 4.0], "median") == 2.5

    def test_min(self):
        assert _aggregate([3.0, 1.0, 2.0], "min") == 1.0

    def test_max(self):
        assert _aggregate([3.0, 1.0, 2.0], "max") == 3.0

    def test_empty_returns_zero(self):
        assert _aggregate([], "mean") == 0.0

    def test_single_value(self):
        assert _aggregate([5.0], "mean") == 5.0

    def test_unknown_mode_defaults_to_mean(self):
        assert _aggregate([1.0, 3.0], "unknown") == 2.0


# ---------------------------------------------------------------------------
# _adapt_target
# ---------------------------------------------------------------------------


class TestAdaptTarget:
    def test_raw_passthrough(self):
        """Raw adapter returns the function unchanged."""

        def fn(x):
            return x

        assert _adapt_target(fn, "raw") is fn

    def test_langchain_adapter(self):
        """Langchain adapter calls .invoke() on the target."""
        mock_runnable = MagicMock()
        mock_runnable.invoke.return_value = {"answer": "yes"}
        adapted = _adapt_target(mock_runnable, "langchain")
        result = adapted({"question": "test?"})
        mock_runnable.invoke.assert_called_once_with({"question": "test?"})
        assert result == {"answer": "yes"}

    def test_langchain_adapter_wraps_non_dict(self):
        """Langchain adapter wraps non-dict results."""
        mock_runnable = MagicMock()
        mock_runnable.invoke.return_value = "plain text"
        adapted = _adapt_target(mock_runnable, "langchain")
        result = adapted({"question": "test?"})
        assert result == {"output": "plain text"}

    def test_single_input_adapter(self):
        """Single-input adapter extracts text from common input keys."""

        def fn(text):
            return f"response to {text}"

        adapted = _adapt_target(fn, "single-input")

        result = adapted({"input": "hello"})
        assert result == {"output": "response to hello"}

        result = adapted({"question": "what?"})
        assert result == {"output": "response to what?"}

        result = adapted({"text": "some text"})
        assert result == {"output": "response to some text"}

    def test_single_input_adapter_dict_return(self):
        """Single-input adapter passes through dict returns."""

        def fn(text):
            return {"answer": text}

        adapted = _adapt_target(fn, "single-input")
        result = adapted({"input": "hello"})
        assert result == {"answer": "hello"}


# ---------------------------------------------------------------------------
# resolve_evaluators
# ---------------------------------------------------------------------------


class TestResolveEvaluators:
    def test_module_function_spec(self, tmp_path, monkeypatch):
        """Evaluator specified as module:function is imported."""
        mod_file = tmp_path / "my_evals.py"
        mod_file.write_text("def check_accuracy(run, example): return {'score': 1}\n")
        monkeypatch.syspath_prepend(str(tmp_path))

        result = resolve_evaluators(["my_evals:check_accuracy"])
        assert len(result) == 1
        assert callable(result[0])

    def test_unresolvable_raises(self):
        """Evaluator name that can't be resolved raises ValueError."""
        with pytest.raises(ValueError, match="Could not resolve evaluator"):
            resolve_evaluators(["definitely_not_a_real_evaluator_xyz"])


# ---------------------------------------------------------------------------
# run_langsmith_eval (mocked)
# ---------------------------------------------------------------------------


class TestRunLangsmithEval:
    def _make_mock_results(self, metrics_per_example):
        """Create mock evaluation results.

        Args:
            metrics_per_example: list of dicts, each mapping metric_name -> score
        """
        results = []
        for metrics in metrics_per_example:
            eval_results = []
            for key, score in metrics.items():
                mock_result = MagicMock()
                mock_result.key = key
                mock_result.score = score
                eval_results.append(mock_result)
            results.append({"evaluation_results": {"results": eval_results}})
        return results

    @patch("weco.integrations.langsmith.bridge.resolve_evaluators")
    def test_basic_evaluation(self, mock_resolve):
        """Test end-to-end evaluation with mocked LangSmith client."""
        mock_resolve.return_value = [lambda r, e: {"score": 1}]

        mock_results = self._make_mock_results(
            [{"correctness": 0.8, "relevance": 0.9}, {"correctness": 1.0, "relevance": 0.7}]
        )

        mock_client_cls = MagicMock()
        mock_client_cls.return_value.evaluate.return_value = mock_results

        with patch.dict("sys.modules", {"langsmith": MagicMock(Client=mock_client_cls)}):
            metrics = run_langsmith_eval(
                dataset_name="test-data", target=lambda x: x, evaluator_names=["correctness"], metric_name="correctness"
            )

        assert metrics["correctness"] == pytest.approx(0.9)
        assert metrics["relevance"] == pytest.approx(0.8)

    @patch("weco.integrations.langsmith.bridge.resolve_evaluators")
    def test_max_examples_limit(self, mock_resolve):
        """Test that max_examples limits the number of examples processed."""
        mock_resolve.return_value = [lambda r, e: {"score": 1}]

        mock_results = self._make_mock_results([{"accuracy": 0.5}, {"accuracy": 1.0}, {"accuracy": 0.0}])

        mock_client_cls = MagicMock()
        mock_client_cls.return_value.evaluate.return_value = mock_results

        with patch.dict("sys.modules", {"langsmith": MagicMock(Client=mock_client_cls)}):
            metrics = run_langsmith_eval(
                dataset_name="test-data",
                target=lambda x: x,
                evaluator_names=["accuracy"],
                metric_name="accuracy",
                max_examples=2,
            )

        # Only first 2 examples: (0.5 + 1.0) / 2 = 0.75
        assert metrics["accuracy"] == pytest.approx(0.75)

    @patch("weco.integrations.langsmith.bridge.resolve_evaluators")
    def test_summary_modes(self, mock_resolve):
        """Test different aggregation modes."""
        mock_resolve.return_value = [lambda r, e: {"score": 1}]

        mock_results = self._make_mock_results([{"score": 1.0}, {"score": 2.0}, {"score": 3.0}])

        mock_client_cls = MagicMock()

        for mode, expected in [("mean", 2.0), ("median", 2.0), ("min", 1.0), ("max", 3.0)]:
            mock_client_cls.return_value.evaluate.return_value = list(mock_results)
            with patch.dict("sys.modules", {"langsmith": MagicMock(Client=mock_client_cls)}):
                metrics = run_langsmith_eval(
                    dataset_name="test-data",
                    target=lambda x: x,
                    evaluator_names=["score"],
                    metric_name="score",
                    summary_mode=mode,
                )
            assert metrics["score"] == pytest.approx(expected), f"Failed for mode={mode}"

    @patch("weco.integrations.langsmith.bridge.resolve_evaluators")
    def test_empty_results(self, mock_resolve):
        """Test handling of empty evaluation results."""
        mock_resolve.return_value = [lambda r, e: {"score": 1}]

        mock_client_cls = MagicMock()
        mock_client_cls.return_value.evaluate.return_value = []

        with patch.dict("sys.modules", {"langsmith": MagicMock(Client=mock_client_cls)}):
            metrics = run_langsmith_eval(
                dataset_name="test-data", target=lambda x: x, evaluator_names=["score"], metric_name="score"
            )

        assert metrics == {}

    @patch("weco.integrations.langsmith.bridge.resolve_evaluators")
    def test_splits_filters_examples(self, mock_resolve):
        """When splits are provided, list_examples is called with splits parameter."""
        mock_resolve.return_value = [lambda r, e: {"score": 1}]

        mock_results = self._make_mock_results([{"accuracy": 1.0}])

        mock_client_cls = MagicMock()
        mock_client = mock_client_cls.return_value
        mock_client.list_examples.return_value = ["example1"]
        mock_client.evaluate.return_value = mock_results

        with patch.dict("sys.modules", {"langsmith": MagicMock(Client=mock_client_cls)}):
            run_langsmith_eval(
                dataset_name="test-data",
                target=lambda x: x,
                evaluator_names=["accuracy"],
                metric_name="accuracy",
                splits=["train"],
            )

        mock_client.list_examples.assert_called_once_with(dataset_name="test-data", splits=["train"])
        # data should be the filtered examples, not the dataset name
        call_kwargs = mock_client.evaluate.call_args
        assert call_kwargs[1]["data"] == ["example1"]


# ---------------------------------------------------------------------------
# main() — output format
# ---------------------------------------------------------------------------


class TestMain:
    def test_missing_api_key(self, monkeypatch):
        """main() exits with error when LANGCHAIN_API_KEY is not set."""
        monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
        monkeypatch.setattr("sys.argv", ["prog", "--dataset", "d", "--target", "m:f", "--evaluators", "e", "--metric", "m"])

        captured = StringIO()
        monkeypatch.setattr("sys.stdout", captured)

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1
        assert "LANGCHAIN_API_KEY" in captured.getvalue()

    def test_import_error_handled(self, monkeypatch):
        """main() handles import errors gracefully."""
        monkeypatch.setenv("LANGCHAIN_API_KEY", "test-key")
        monkeypatch.setattr(
            "sys.argv",
            ["prog", "--dataset", "d", "--target", "nonexistent_module_xyz:func", "--evaluators", "e", "--metric", "m"],
        )

        captured = StringIO()
        monkeypatch.setattr("sys.stdout", captured)

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1
        assert "Import error" in captured.getvalue()

    @patch("weco.integrations.langsmith.bridge.run_langsmith_eval")
    @patch("weco.integrations.langsmith.bridge.import_target")
    def test_output_format(self, mock_import, mock_eval, monkeypatch, capsys):
        """main() prints metrics in Weco's expected 'key: value' format."""
        monkeypatch.setenv("LANGCHAIN_API_KEY", "test-key")
        monkeypatch.setattr(
            "sys.argv", ["prog", "--dataset", "d", "--target", "m:f", "--evaluators", "acc", "--metric", "acc"]
        )

        mock_import.return_value = lambda x: x
        mock_eval.return_value = {"acc": 0.85, "relevance": 0.72}

        main()

        captured = capsys.readouterr()
        assert "acc: 0.850000" in captured.out
        assert "relevance: 0.720000" in captured.out

    @patch("weco.integrations.langsmith.bridge.run_langsmith_eval")
    @patch("weco.integrations.langsmith.bridge.import_target")
    def test_prints_all_metrics(self, mock_import, mock_eval, monkeypatch, capsys):
        """main() prints ALL metrics, not just the target one."""
        monkeypatch.setenv("LANGCHAIN_API_KEY", "test-key")
        monkeypatch.setattr(
            "sys.argv", ["prog", "--dataset", "d", "--target", "m:f", "--evaluators", "acc", "rel", "--metric", "acc"]
        )

        mock_import.return_value = lambda x: x
        mock_eval.return_value = {"acc": 0.85, "rel": 0.72}

        # main() should not raise SystemExit on success
        main()

        captured = capsys.readouterr()
        assert "acc: 0.850000" in captured.out
        assert "rel: 0.720000" in captured.out

    @patch("weco.integrations.langsmith.bridge.run_langsmith_eval")
    @patch("weco.integrations.langsmith.bridge.import_target")
    def test_empty_metrics_exits(self, mock_import, mock_eval, monkeypatch, capsys):
        """main() exits with warning when no metrics are returned."""
        monkeypatch.setenv("LANGCHAIN_API_KEY", "test-key")
        monkeypatch.setattr(
            "sys.argv", ["prog", "--dataset", "d", "--target", "m:f", "--evaluators", "acc", "--metric", "acc"]
        )

        mock_import.return_value = lambda x: x
        mock_eval.return_value = {}

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "No metrics returned" in captured.out


# ---------------------------------------------------------------------------
# _poll_dashboard_scores
# ---------------------------------------------------------------------------


class TestPollDashboardScores:
    def test_finds_new_scores(self):
        """Polling finds scores from dashboard evaluators not in known_keys."""
        mock_client = MagicMock()

        # First poll: no feedback yet. Second poll: feedback appears.
        fb = MagicMock()
        fb.key = "dashboard_metric"
        fb.score = 0.95
        mock_client.list_feedback.side_effect = [[], [fb], [fb]]

        scores = _poll_dashboard_scores(
            mock_client,
            run_ids=["run-1"],
            known_keys={"correctness"},
            expected_keys={"dashboard_metric"},
            timeout=3,
            poll_interval=0.1,
        )

        assert "dashboard_metric" in scores
        assert scores["dashboard_metric"] == [0.95]

    def test_ignores_known_keys(self):
        """Polling skips metrics already captured from code evaluators."""
        mock_client = MagicMock()

        fb_known = MagicMock()
        fb_known.key = "correctness"
        fb_known.score = 0.5
        fb_new = MagicMock()
        fb_new.key = "dashboard_metric"
        fb_new.score = 0.8
        mock_client.list_feedback.side_effect = [[fb_known, fb_new], [fb_known, fb_new]]

        scores = _poll_dashboard_scores(
            mock_client,
            run_ids=["run-1"],
            known_keys={"correctness"},
            expected_keys={"dashboard_metric"},
            timeout=3,
            poll_interval=0.1,
        )

        assert "correctness" not in scores
        assert "dashboard_metric" in scores

    def test_empty_runs(self):
        """Returns empty dict when no run IDs provided."""
        mock_client = MagicMock()

        scores = _poll_dashboard_scores(
            mock_client, run_ids=[], known_keys=set(), expected_keys=set(), timeout=1, poll_interval=0.1
        )

        assert scores == {}
