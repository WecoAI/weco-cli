"""Baseline triage agent for the LangSmith Zendesk-style example.

Weco optimizes this file. The baseline is intentionally simple so optimization
can improve label consistency and decision quality.
"""

import json

from openai import OpenAI

client = OpenAI()

SYSTEM_PROMPT = """Classify each support ticket and return JSON only.

Required fields:
- category: "account_access" | "billing" | "shipping" | "refund" | "product_issue" | "general_question"
- priority: "low" | "medium" | "high"
- requires_human: true | false

Use the ticket subject and description only.
Set requires_human=true when the issue likely needs manual intervention."""


def triage_ticket(inputs: dict) -> dict:
    """Classify one ticket into structured triage labels."""
    subject = (inputs or {}).get("subject", "")
    description = (inputs or {}).get("description", "")

    response = client.chat.completions.create(
        model="gpt-5.2",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (f"Ticket:\nSubject: {subject}\nDescription: {description}\n\nReturn only JSON.")},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    try:
        parsed = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        parsed = {}

    # Always return all fields for deterministic scoring.
    return {
        "category": parsed.get("category"),
        "priority": parsed.get("priority"),
        "requires_human": parsed.get("requires_human"),
    }
