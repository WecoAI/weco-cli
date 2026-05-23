"""Tests for the consecutive-identical-failures halt logic in run_optimization_loop."""

from weco.optimizer import _extract_error_signature, _extract_error_tail


# The actual term_out from a real failed run that surfaced this issue
# (5-step run on Windows where every step failed because the eval command
# `python evaluate.py` resolved to system Python, not the venv Python with sklearn).
SKLEARN_MISSING_OUTPUT = """\
Traceback (most recent call last):
  File "C:\\Users\\jzyji\\AppData\\Local\\Temp\\weco-experiment\\evaluate.py", line 4, in <module>
    from sklearn.datasets import load_breast_cancer
ModuleNotFoundError: No module named 'sklearn'
"""

HEALTHY_OUTPUT = "accuracy: 0.9543\n"

DIFFERENT_ERROR_OUTPUT = """\
Traceback (most recent call last):
  File "evaluate.py", line 12, in <module>
    raise ValueError("Invalid input shape")
ValueError: Invalid input shape
"""


class TestExtractErrorSignature:
    def test_healthy_output_returns_none(self):
        # A successful eval (just a metric line) is not an error
        assert _extract_error_signature(HEALTHY_OUTPUT) is None

    def test_empty_output_returns_none(self):
        assert _extract_error_signature("") is None
        assert _extract_error_signature("   \n   \n") is None

    def test_traceback_returns_signature(self):
        sig = _extract_error_signature(SKLEARN_MISSING_OUTPUT)
        assert sig is not None
        assert len(sig) == 40  # sha1 hex length

    def test_identical_outputs_produce_identical_signatures(self):
        sig1 = _extract_error_signature(SKLEARN_MISSING_OUTPUT)
        sig2 = _extract_error_signature(SKLEARN_MISSING_OUTPUT)
        assert sig1 == sig2

    def test_different_errors_produce_different_signatures(self):
        sig_sklearn = _extract_error_signature(SKLEARN_MISSING_OUTPUT)
        sig_value = _extract_error_signature(DIFFERENT_ERROR_OUTPUT)
        assert sig_sklearn != sig_value

    def test_signature_robust_to_leading_lines(self):
        # If two outputs share the same trailing error but differ in earlier lines,
        # they should still hash identically (because we hash the trailing lines).
        with_preamble = (
            "Step 3 of 5\n"
            "Running evaluation...\n"
            + SKLEARN_MISSING_OUTPUT
        )
        sig1 = _extract_error_signature(SKLEARN_MISSING_OUTPUT)
        sig2 = _extract_error_signature(with_preamble)
        assert sig1 == sig2

    def test_error_indicators_picked_up(self):
        # Any of the recognized indicators should trigger signature extraction
        for indicator_output in [
            "Traceback (most recent call last):\n  ...\nValueError: bad",
            "Exception: something broke",
            "ImportError: No module named 'x'",
            "FATAL: cannot proceed",
        ]:
            assert _extract_error_signature(indicator_output) is not None


class TestExtractErrorTail:
    def test_returns_last_n_lines(self):
        output = "line1\nline2\nline3\nline4\nline5\nline6\n"
        tail = _extract_error_tail(output, max_lines=3)
        assert tail == "line4\nline5\nline6"

    def test_skips_blank_lines(self):
        output = "line1\n\n\nline2\n\nline3\n"
        tail = _extract_error_tail(output, max_lines=5)
        assert "line1" in tail and "line2" in tail and "line3" in tail

    def test_empty_output_returns_empty_string(self):
        assert _extract_error_tail("") == ""
