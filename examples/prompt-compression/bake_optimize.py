"""One-shot script: bakes the bloated baseline prompt into optimize.py.

Run once before kicking off Weco. After this, Weco mutates optimize.py
directly — DO NOT re-run this script mid-optimization (it will overwrite
Weco's progress with the original bloated baseline).

Usage:
    python bake_optimize.py
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_bloated_prompt import build_prompt  # noqa: E402

ROOT = Path(__file__).resolve().parent
OPTIMIZE_PATH = ROOT / "optimize.py"

PROMPT = build_prompt()
escaped = PROMPT.replace('"""', '\\"\\"\\"')

OPTIMIZE_BODY = f'''\
"""BANKING77 intent classifier — the file Weco edits.

Weco mutates SYSTEM_PROMPT below to compress it while preserving classification
accuracy. The classify() function and surrounding plumbing must remain
intact — only the SYSTEM_PROMPT string content should be modified.

Baseline prompt: {len(PROMPT):,} chars across all 77 classes.
"""
from __future__ import annotations
import os
from openai import OpenAI

# ---------------------------------------------------------------------------
# WECO-MUTABLE REGION: only the SYSTEM_PROMPT string content should be edited.
# Do not change the variable name, the assignment, or the surrounding code.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """{escaped}"""
# ---------------------------------------------------------------------------
# END WECO-MUTABLE REGION
# ---------------------------------------------------------------------------


_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def classify(query: str, model: str = "gpt-5-mini") -> str:
    """Classify a single user query into one of the 77 BANKING77 intents.

    Returns the model's raw text response. Label parsing happens in eval.py.
    """
    response = _get_client().chat.completions.create(
        model=model,
        messages=[
            {{"role": "system", "content": SYSTEM_PROMPT}},
            {{"role": "user", "content": query}},
        ],
        max_completion_tokens=256,
        reasoning_effort="minimal",
    )
    return (response.choices[0].message.content or "").strip()
'''

OPTIMIZE_PATH.write_text(OPTIMIZE_BODY, encoding="utf-8")
print(f"wrote {OPTIMIZE_PATH} ({OPTIMIZE_PATH.stat().st_size:,} bytes)")
print(f"  SYSTEM_PROMPT: {len(PROMPT):,} chars")
