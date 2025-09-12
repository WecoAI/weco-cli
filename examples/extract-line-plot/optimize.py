"""
optimize.py

Baseline implementation of a VLM-driven function that takes an image and returns CSV.
Weco will optimize the prompt and logic here.
"""

import base64
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

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0, client: Optional[OpenAI] = None) -> None:
        self.model = model
        self.temperature = temperature
        self.client = client or OpenAI()

    def image_to_csv(self, image_path: Path) -> str:
        prompt = build_prompt()
        image_uri = image_to_data_uri(image_path)
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
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
        text = response.choices[0].message.content or ""
        return clean_to_csv(text)


