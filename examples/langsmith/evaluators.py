"""Evaluators for the reply-drafting LangSmith example.

Primary metric:
- grounded_reply_quality (LLM-as-judge)

Deterministic diagnostics:
- retrieval_hit_at_2
- escalation_match
- reply_length_ok
"""

import json
from typing import List

from openai import OpenAI
from knowledge_base import DOCS_BY_ID

_judge_client = OpenAI()

GROUNDING_PROMPT = """\
You are grading a customer-support reply.

Ticket subject: {subject}
Customer message: {message}

Retrieved policy docs:
{retrieved_context}

Reference reply:
{ideal_reply}

Agent reply:
{predicted_reply}

Score the agent reply from 0.0 to 1.0 based on:
1. Factual correctness relative to the retrieved docs
2. Usefulness and clarity for the customer
3. Specific next steps when appropriate
4. No unsupported policy claims

Scoring guide:
- 1.0: Fully grounded, helpful, specific, no hallucinations
- 0.7-0.9: Mostly good, minor omissions or imprecision
- 0.4-0.6: Partially helpful, missing key details, or slightly unsupported
- 0.1-0.3: Mostly unhelpful or significantly unsupported
- 0.0: Misleading, clearly hallucinated, or unsafe

Respond with JSON: {{"score": <float>, "reasoning": "<brief explanation>"}}\
"""


def _as_list(value):
    if isinstance(value, list):
        return value
    return []


def _reply_text(run) -> str:
    return ((run.outputs or {}).get("reply") or "").strip()


def _retrieved_doc_ids(run) -> List[str]:
    return [str(x) for x in _as_list((run.outputs or {}).get("retrieved_doc_ids"))]


def grounded_reply_quality(run, example) -> dict:
    predicted_reply = _reply_text(run)
    subject = (example.inputs or {}).get("subject", "")
    message = (example.inputs or {}).get("message", "")
    ideal_reply = (example.outputs or {}).get("ideal_reply", "")

    context_parts = []
    for doc_id in _retrieved_doc_ids(run):
        doc = DOCS_BY_ID.get(doc_id)
        if doc:
            context_parts.append(f"{doc_id}: {doc['body']}")
    if not context_parts:
        context_parts.append("(no valid retrieved docs)")

    prompt = GROUNDING_PROMPT.format(
        subject=subject,
        message=message,
        retrieved_context="\n\n".join(context_parts),
        ideal_reply=ideal_reply,
        predicted_reply=predicted_reply,
    )

    response = _judge_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )

    try:
        result = json.loads(response.choices[0].message.content)
        score = float(result.get("score", 0.0))
        comment = result.get("reasoning", "")
    except (json.JSONDecodeError, ValueError, TypeError):
        score = 0.0
        comment = "Failed to parse LLM judge response"

    return {"key": "grounded_reply_quality", "score": score, "comment": comment}


def retrieval_hit_at_2(run, example) -> dict:
    predicted_ids = set(_retrieved_doc_ids(run))
    gold_ids = set(_as_list((example.outputs or {}).get("gold_doc_ids")))
    hit = 1.0 if predicted_ids & gold_ids else 0.0
    return {"key": "retrieval_hit_at_2", "score": hit, "comment": "Retrieved a gold doc" if hit else "Missed gold docs"}


def escalation_match(run, example) -> dict:
    predicted = (run.outputs or {}).get("should_escalate")
    expected = (example.outputs or {}).get("should_escalate")
    ok = predicted == expected
    return {
        "key": "escalation_match",
        "score": 1.0 if ok else 0.0,
        "comment": "Escalation matches" if ok else "Escalation mismatch",
    }


def reply_length_ok(run, example) -> dict:
    word_count = len(_reply_text(run).split())
    ok = 35 <= word_count <= 140
    return {
        "key": "reply_length_ok",
        "score": 1.0 if ok else 0.0,
        "comment": f"Reply length {word_count} words",
    }
