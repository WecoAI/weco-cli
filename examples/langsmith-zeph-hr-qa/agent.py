"""Baseline QA agent for ZephHR documentation.

Answers HR policy questions using only the docs.md knowledge base.
Weco optimizes this file — specifically the SYSTEM_PROMPT and USER_TEMPLATE.
"""

import json
from pathlib import Path

from openai import OpenAI

client = OpenAI()

DOCS = Path(__file__).with_name("docs.md").read_text()

SYSTEM_PROMPT = """You are a ZephHR support assistant. Answer the user's question
using ONLY the provided documentation. Do not guess or invent policy details.

If the documentation does not contain enough information to fully answer,
say so clearly and state what IS covered.

Return your answer as JSON with exactly these fields:
- answer: your response to the question (string)
- confidence: how confident you are the answer is fully supported by the docs (high/medium/low)
- relevant_sections: list of section names from the docs you referenced"""

USER_TEMPLATE = """Documentation:
{docs}

Question: {question}

Return only JSON."""


def answer_hr_question(inputs: dict) -> dict:
    """Answer an HR policy question from the ZephHR docs."""
    question = inputs.get("question", "")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_TEMPLATE.format(docs=DOCS, question=question),
            },
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )

    try:
        parsed = json.loads(response.choices[0].message.content)
    except (TypeError, json.JSONDecodeError):
        parsed = {}

    confidence = parsed.get("confidence", "low")
    if confidence not in ("high", "medium", "low"):
        confidence = "low"

    relevant_sections = parsed.get("relevant_sections", [])
    if not isinstance(relevant_sections, list):
        relevant_sections = []

    return {
        "answer": parsed.get("answer", ""),
        "confidence": confidence,
        "relevant_sections": relevant_sections,
    }
