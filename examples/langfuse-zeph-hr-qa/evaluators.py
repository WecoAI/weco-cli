"""Evaluators and metric function for ZephHR QA (LangFuse format).

Code evaluators (run locally via Weco bridge):
- json_schema_validity: checks the agent output has the required JSON schema
- conciseness: penalises excessively long or empty answers

LLM judges (configured in LangFuse UI as managed evaluators):
- helpfulness: how helpful and complete the answer is (1-5 scale)
- correctness: binary factual accuracy against expected answer and required facts (0 or 1)

Metric function:
- qa_score: multiplies correctness (gate) by normalized helpfulness (signal),
  so incorrect answers score 0 regardless of helpfulness.
"""

from langfuse import Evaluation


def json_schema_validity(*, input, output, expected_output=None, **kwargs):
    """Check that the agent output contains the required fields with correct types."""
    outputs = output or {}

    checks = {
        "answer": isinstance(outputs.get("answer"), str) and len(outputs["answer"]) > 0,
        "confidence": outputs.get("confidence") in ("high", "medium", "low"),
        "relevant_sections": isinstance(outputs.get("relevant_sections"), list),
    }

    passed = all(checks.values())
    failed_fields = [k for k, v in checks.items() if not v]
    comment = "All fields valid" if passed else f"Invalid fields: {', '.join(failed_fields)}"

    return Evaluation(name="json_schema_validity", value=1.0 if passed else 0.0, comment=comment)


def conciseness(*, input, output, expected_output=None, **kwargs):
    """Score based on answer length — penalise empty or excessively verbose answers."""
    answer = (output or {}).get("answer", "")
    word_count = len(answer.split())

    if word_count == 0:
        score, comment = 0.0, "Empty answer"
    elif word_count <= 150:
        score, comment = 1.0, f"{word_count} words — concise"
    elif word_count <= 250:
        score, comment = 0.5, f"{word_count} words — verbose"
    else:
        score, comment = 0.0, f"{word_count} words — excessively long"

    return Evaluation(name="conciseness", value=score, comment=comment)


def qa_score(scores: dict) -> float:
    """Combine correctness (binary gate) with helpfulness (0-1 signal).

    correctness * helpfulness

    - Incorrect answers always score 0.
    - Correct answers are ranked by helpfulness.
    """
    correctness = scores.get("Correctness", 0.0)
    helpfulness = scores.get("Helpfulness", 0.0)
    return correctness * helpfulness
