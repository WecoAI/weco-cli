"""Evaluators and metric function for ZephHR QA.

Code evaluators (run locally):
- json_schema_validity: checks the agent output has the required JSON schema
- conciseness: penalises excessively long or empty answers

LLM judges (configured in LangSmith dashboard, not in code):
- helpfulness: how helpful and complete the answer is (1-5 scale)
- correctness: binary factual accuracy against expected answer and required facts (0 or 1)

Metric function:
- qa_score: multiplies correctness (gate) by normalized helpfulness (signal),
  so incorrect answers score 0 regardless of helpfulness.
"""


def json_schema_validity(run, example) -> dict:
    """Check that the agent output contains the required fields with correct types."""
    outputs = run.outputs or {}

    checks = {
        "answer": isinstance(outputs.get("answer"), str) and len(outputs["answer"]) > 0,
        "confidence": outputs.get("confidence") in ("high", "medium", "low"),
        "relevant_sections": isinstance(outputs.get("relevant_sections"), list),
    }

    passed = all(checks.values())
    failed_fields = [k for k, v in checks.items() if not v]
    comment = "All fields valid" if passed else f"Invalid fields: {', '.join(failed_fields)}"

    return {"key": "json_schema_validity", "score": 1.0 if passed else 0.0, "comment": comment}


def conciseness(run, example) -> dict:
    """Score based on answer length — penalise empty or excessively verbose answers."""
    answer = (run.outputs or {}).get("answer", "")
    word_count = len(answer.split())

    if word_count == 0:
        score, comment = 0.0, "Empty answer"
    elif word_count <= 150:
        score, comment = 1.0, f"{word_count} words — concise"
    elif word_count <= 250:
        score, comment = 0.5, f"{word_count} words — verbose"
    else:
        score, comment = 0.0, f"{word_count} words — excessively long"

    return {"key": "conciseness", "score": score, "comment": comment}


def qa_score(scores: dict) -> float:
    """Combine correctness (binary gate) with helpfulness (1-5 signal).

    correctness * (helpfulness - 1) / 4

    - Incorrect answers always score 0.
    - Correct answers are ranked by helpfulness, normalized to 0-1.
    """
    correctness = scores.get("correctness", 0.0)
    helpfulness = scores.get("helpfulness", 1.0)
    return correctness * (helpfulness - 1.0) / 4.0
