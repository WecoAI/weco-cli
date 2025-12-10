"""
optimize.py

Exposes a single public entry point `extract_csv` that turns a chart image into CSV text.
All helper utilities remain private to this module.
"""

import base64
from pathlib import Path
from typing import Optional, Tuple

from openai import OpenAI

__all__ = ["extract_csv"]

_DEFAULT_MODEL = "gpt-4o-mini"
_CLIENT = OpenAI()


def _build_prompt() -> str:
    return (
        "You are a precise data extraction model. Given a chart image, extract the underlying data table.\n"
        "Return ONLY the CSV text with a header row and no markdown code fences.\n"
        "Rules:\n"
        "- The first column must be the x-axis values with its exact axis label as the header.\n"
        "- Include one column per data series using the legend labels as headers.\n"
        "- Preserve the original order of x-axis ticks as they appear.\n"
        "- Use plain CSV (comma-separated), no explanations, no extra text.\n"
    )


def _image_to_data_uri(image_path: Path) -> str:
    mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
    data = image_path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _clean_to_csv(text: str) -> str:
    return text.strip()


def _pricing_for_model(model_name: str) -> dict:
    """Return pricing information for the given model in USD per token."""
    name = (model_name or "").lower()
    per_million = {
        "gpt-5": {"in": 1.250, "in_cached": 0.125, "out": 10.000},
        "gpt-5-mini": {"in": 0.250, "in_cached": 0.025, "out": 2.000},
        "gpt-5-nano": {"in": 0.050, "in_cached": 0.005, "out": 0.400},
    }
    if name.startswith("gpt-5-nano"):
        chosen = per_million["gpt-5-nano"]
    elif name.startswith("gpt-5-mini"):
        chosen = per_million["gpt-5-mini"]
    elif name.startswith("gpt-5"):
        chosen = per_million["gpt-5"]
    else:
        chosen = per_million["gpt-5-mini"]
    return {k: v / 1_000_000.0 for k, v in chosen.items()}


def extract_csv(image_path: Path, model: Optional[str] = None) -> Tuple[str, float]:
    """
    Extract CSV text from an image and return (csv_text, cost_usd).

    The caller can optionally override the model name; otherwise the default is used.
    """
    effective_model = model or _DEFAULT_MODEL
    prompt = _build_prompt()
    image_uri = _image_to_data_uri(image_path)
    response = _CLIENT.chat.completions.create(
        model=effective_model,
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": image_uri}}],
            }
        ],
    )

    usage = getattr(response, "usage", None)
    cost_usd = 0.0
    if usage is not None:
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        details = getattr(usage, "prompt_tokens_details", None)
        cached_tokens = 0
        if details is not None:
            cached_tokens = int(getattr(details, "cached_tokens", 0) or 0)
        non_cached_prompt_tokens = max(0, prompt_tokens - cached_tokens)
        rates = _pricing_for_model(effective_model)
        cost_usd = (
            non_cached_prompt_tokens * rates["in"] + cached_tokens * rates["in_cached"] + completion_tokens * rates["out"]
        )

    text = response.choices[0].message.content or ""
    return _clean_to_csv(text), cost_usd
