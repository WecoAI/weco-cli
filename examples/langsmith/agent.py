"""Reply-drafting agent for the Weco + LangSmith optimization example.

This benchmark drafts grounded customer replies using a tiny local retriever.
Weco optimizes this file.
"""

import json

from openai import OpenAI
from retrieval import retrieve_documents

client = OpenAI()

SYSTEM_PROMPT = """You are a customer support agent for Acme Cloud.
Draft a short reply to the customer using only the provided policy context.

Return JSON with exactly these fields:
- reply
- retrieved_doc_ids
- should_escalate

If the context does not fully support a confident answer, say what you can,
avoid inventing policy details, and set should_escalate to true."""


def answer_question(inputs: dict) -> dict:
    """Draft a grounded customer-support reply."""
    subject = inputs.get("subject", "")
    message = inputs.get("message", "")
    docs = retrieve_documents(subject=subject, message=message, top_k=2)
    retrieved_doc_ids = [doc["doc_id"] for doc in docs]
    context = "\n\n".join(
        f"[{idx}] {doc['title']}\n{doc['body']}" for idx, doc in enumerate(docs, start=1)
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Subject: {subject}\n"
                    f"Customer message: {message}\n\n"
                    f"Policy context:\n{context}\n\n"
                    "Return only JSON."
                ),
            },
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )

    try:
        parsed = json.loads(response.choices[0].message.content)
    except (TypeError, json.JSONDecodeError):
        parsed = {}

    should_escalate = parsed.get("should_escalate")
    if not isinstance(should_escalate, bool):
        should_escalate = False

    return {
        "reply": parsed.get("reply", ""),
        "retrieved_doc_ids": parsed.get("retrieved_doc_ids", retrieved_doc_ids),
        "should_escalate": should_escalate,
    }
