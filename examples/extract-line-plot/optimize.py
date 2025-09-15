"""
optimize.py

Structured JSON-first extraction: prompt the VLM to return strict JSON, then convert to CSV locally.
This aims to reduce spurious text and improve alignment of x/series values while keeping cost low.

Constraints:
- No backticks in code content.
- Maintain accurate cost tracking.
"""

import base64
import csv
import io
import json
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI


def build_prompt() -> str:
    return (
        "You are a precise data extraction model. Given a chart image, extract the underlying data table as strict JSON only.\n"
        "Return ONLY compact JSON (no code fences, no explanations). Do not include units in numeric values.\n"
        "Schema:\n"
        "{\n"
        '  "x_label": string,\n'
        '  "x": [string or number, ...],\n'
        '  "series": [\n'
        "    {\"name\": string, \"values\": [string or number, ...]},\n"
        "    ...\n"
        "  ]\n"
        "}\n"
        "Rules:\n"
        "- x must be the first column values and preserve original order.\n"
        "- Use legend labels for series.name and exact x-axis label for x_label.\n"
        "- Ensure each series.values aligns by index to x; keep lengths equal where possible.\n"
        "- If values are approximate (e.g., reading from bars/lines), use best estimates consistently.\n"
        "- Output only valid JSON matching the schema above."
    )


def image_to_data_uri(image_path: Path) -> str:
    mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
    data = image_path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _strip_code_fences(text: str) -> str:
    # Remove common code fences using \x60 to represent the backtick character.
    text = re.sub(r"(?is)^[\x60~]{3,}[^\n]*\n?", "", text.strip())
    text = re.sub(r"(?is)\n?[\x60~]{3,}\s*$", "", text.strip())
    return text.strip()


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    raw = _strip_code_fences(text)
    # Heuristically isolate the largest JSON object
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = raw[start : end + 1].strip()

    # Light cleanup: normalize smart quotes and stray trailing commas
    candidate = candidate.replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'")
    # Remove trailing commas before } or ]
    candidate = re.sub(r",(\s*[}\]])", r"\1", candidate)

    # Prefer strict JSON; try original and a single-quote to double-quote fallback
    try:
        return json.loads(candidate)
    except Exception:
        pass
    try:
        candidate2 = re.sub(r"'", '"', candidate)
        return json.loads(candidate2)
    except Exception:
        return None


def _coerce_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (int, float)):
        return f"{v}"
    return str(v).strip()


def _json_to_csv(data: Dict[str, Any]) -> str:
    x_label = _coerce_str(data.get("x_label", "x"))
    x_vals_raw = data.get("x", [])
    series_raw = data.get("series", [])

    x_vals = [_coerce_str(v) for v in (x_vals_raw if isinstance(x_vals_raw, list) else [])]

    series_list: List[Dict[str, Any]] = []
    if isinstance(series_raw, list):
        for s in series_raw:
            if not isinstance(s, dict):
                continue
            name = _coerce_str(s.get("name") or s.get("label") or s.get("header") or "")
            vals = s.get("values") or s.get("y") or []
            if not isinstance(vals, list):
                vals = []
            vals = [_coerce_str(v) for v in vals]
            series_list.append({"name": name if name else "series", "values": vals})

    lengths = [len(x_vals)] + [len(s["values"]) for s in series_list]
    row_count = min(lengths) if lengths and all(l > 0 for l in lengths) else 0

    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    header = [x_label] + [s["name"] for s in series_list]
    writer.writerow(header)
    for i in range(row_count):
        row = [x_vals[i]] + [series_list[j]["values"][i] for j in range(len(series_list))]
        writer.writerow(row)
    return output.getvalue().strip()


def clean_to_csv(text: str) -> str:
    return text.strip()


class VLMExtractor:
    """JSON-first VLM wrapper for chart-to-CSV extraction."""

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
        name = (self.model or "").lower()
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

    def _update_usage_costs(self, usage: Any) -> None:
        with self._usage_lock:
            if usage is None:
                self.num_queries += 1
                return
            # Support both chat.completions (prompt/completion) and responses API (input/output) naming
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
            output_tokens = int(getattr(usage, "output_tokens", 0) or 0)

            in_tokens = input_tokens or prompt_tokens
            out_tokens = output_tokens or completion_tokens

            details = getattr(usage, "prompt_tokens_details", None)
            cached_tokens = 0
            if details is not None:
                cached_tokens = int(getattr(details, "cached_tokens", 0) or 0)

            # Use prompt_tokens when available (to account for caching), otherwise fall back to in_tokens
            prompt_for_cost = prompt_tokens or in_tokens
            non_cached_prompt_tokens = max(0, prompt_for_cost - cached_tokens)

            rates = self._pricing_for_model()
            cost = (
                non_cached_prompt_tokens * rates["in"]
                + cached_tokens * rates["in_cached"]
                + out_tokens * rates["out"]
            )

            self.total_prompt_tokens += in_tokens
            self.total_completion_tokens += out_tokens
            self.total_cost_usd += cost
            self.num_queries += 1

    def image_to_csv(self, image_path: Path) -> str:
        prompt = build_prompt()
        image_uri = image_to_data_uri(image_path)
        # Request strict JSON by instruction; omit unsupported temperature overrides for this model.
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
        usage = getattr(response, "usage", None)
        self._update_usage_costs(usage)

        # Defensive extraction of text
        text = ""
        try:
            text = response.choices[0].message.content or ""
        except Exception:
            text = ""

        data = _extract_json(text)
        if data is not None:
            try:
                csv_text = _json_to_csv(data)
                if csv_text.strip():
                    return clean_to_csv(csv_text)
            except Exception:
                pass

        # Fallback: if JSON parsing failed, try to salvage any CSV-like content
        raw = _strip_code_fences(text)
        lines = [ln for ln in raw.splitlines() if ln.strip()]
        if len(lines) >= 2 and ("," in lines[0] or ";" in lines[0] or "\t" in lines[0]):
            sep = "," if "," in lines[0] else ("\t" if "\t" in lines[0] else ";")
            norm = "\n".join([ln.replace(sep, ",") for ln in lines])
            return clean_to_csv(norm)

        # Final resort: return minimal header only to avoid extra text
        return clean_to_csv("x")