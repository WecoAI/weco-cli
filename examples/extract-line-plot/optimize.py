"""
optimize.py

Baseline implementation of a VLM-driven function that takes an image and returns CSV.
Weco will optimize the prompt and logic here.
"""

import base64
import threading
from pathlib import Path
from typing import Optional

from openai import OpenAI


def build_prompt() -> str:
    return (
        "You are a precise data extraction model. Given a chart image, extract the underlying data table.\n"
        "Return ONLY the CSV text with a header row and no markdown code fences.\n"
        "Rules:\n"
        "- The first column must be the x-axis values with its exact axis label as the header.\n"
        "- Include one column per data series using the legend labels as headers.\n"
        "- Preserve the original order of x-axis ticks as they appear.\n"
        "- Use plain CSV (comma-separated), no explanations, no extra text.\n"
    )


def image_to_data_uri(image_path: Path) -> str:
    mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
    data = image_path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def clean_to_csv(text: str) -> str:
    return text.strip()


class VLMExtractor:
    """Baseline VLM wrapper for chart-to-CSV extraction."""

    def __init__(self, model: str = "gpt-5-mini", client: Optional[OpenAI] = None) -> None:
        self.model = model
        self.client = client or OpenAI()
        # Aggregates
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_cost_usd: float = 0.0
        self.num_queries: int = 0
        self._usage_lock = threading.Lock()

    def _pricing_for_model(self) -> dict:
        """Return pricing for current model in USD per token.

        Structure: {"in": x, "in_cached": y, "out": z}
        Defaults to GPT-5 mini if model not matched.
        """
        name = (self.model or "").lower()
        # Prices are given per 1M tokens in the spec; convert to per-token
        per_million = {
            "gpt-5": {"in": 1.250, "in_cached": 0.125, "out": 10.000},
            "gpt-5-mini": {"in": 0.250, "in_cached": 0.025, "out": 2.000},
            "gpt-5-nano": {"in": 0.050, "in_cached": 0.005, "out": 0.400},
        }
        # Pick by prefix
        if name.startswith("gpt-5-nano"):
            chosen = per_million["gpt-5-nano"]
        elif name.startswith("gpt-5-mini"):
            chosen = per_million["gpt-5-mini"]
        elif name.startswith("gpt-5"):
            chosen = per_million["gpt-5"]
        else:
            chosen = per_million["gpt-5-mini"]
        # Convert per 1M to per token
        return {k: v / 1_000_000.0 for k, v in chosen.items()}

    def image_to_csv(self, image_path: Path) -> str:
        prompt = build_prompt()
        image_uri = image_to_data_uri(image_path)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_uri}},
                    ],
                }
            ],
        )
        # Track usage and cost if available
        usage = getattr(response, "usage", None)
        with self._usage_lock:
            if usage is not None:
                prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
                # Attempt to detect cached tokens if available
                details = getattr(usage, "prompt_tokens_details", None)
                cached_tokens = 0
                if details is not None:
                    cached_tokens = int(getattr(details, "cached_tokens", 0) or 0)
                non_cached_prompt_tokens = max(0, prompt_tokens - cached_tokens)

                rates = self._pricing_for_model()
                cost = (
                    non_cached_prompt_tokens * rates["in"]
                    + cached_tokens * rates["in_cached"]
                    + completion_tokens * rates["out"]
                )

                self.total_prompt_tokens += prompt_tokens
                self.total_completion_tokens += completion_tokens
                self.total_cost_usd += cost
                self.num_queries += 1
            else:
                self.num_queries += 1
        text = response.choices[0].message.content or ""
        return clean_to_csv(text)