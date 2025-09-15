## Extract Line Plot (Chart → CSV) with a VLM

This example evaluates and optimizes a vision-language baseline that converts line-chart images into CSV.

### Prerequisites

- Python 3.9+
- `uv` installed (see `https://docs.astral.sh/uv/`)
- An OpenAI API key in your environment:

```bash
export OPENAI_API_KEY=your_key_here
```

### Files

- `prepare_data.py`: downloads ChartQA (full) and prepares a 100-sample subset of line charts.
- `optimize.py`: baseline VLM function (`VLMExtractor.image_to_csv`) to be optimized.
- `eval.py`: evaluation harness that runs the baseline on images and reports a similarity score as "accuracy".

Generated artifacts (gitignored):
- `subset_line_100/` and `subset_line_100.zip`
- `predictions/`

### 1) Prepare the data

From the repo root or this directory:

```bash
cd examples/extract-line-plot
uv run --with huggingface_hub python prepare_data.py
```

Notes:
- Downloads the ChartQA dataset snapshot and auto-extracts `ChartQA Dataset.zip` if needed.
- Produces `subset_line_100/` with `index.csv`, `images/`, and `tables/`.

### 2) Run a baseline evaluation once

```bash
uv run --with openai python eval.py --max-samples 10 --num-workers 4
```

This writes predicted CSVs to `predictions/` and prints a final line like `accuracy: 0.32`.

Metric definition (summarized):
- Per-sample score = 0.2 × header match + 0.8 × Jaccard(similarity of content rows).
- Reported `accuracy` is the mean score over all evaluated samples.

### 3) Optimize the baseline with Weco

Run Weco to iteratively improve `optimize.py` using 100 examples and many workers:

```bash
weco run --source optimize.py --eval-command 'uv run --with openai python eval.py --max-samples 100 --num-workers 50' --metric accuracy --goal maximize --steps 20 --model gpt-5
```

Arguments:
- `--source optimize.py`: file that Weco will edit to improve results.
- `--eval-command '…'`: command Weco executes to measure the metric.
- `--metric accuracy`: Weco parses `accuracy: <value>` from `eval.py` output.
- `--goal maximize`: higher is better.
- `--steps 20`: number of optimization iterations.
- `--model gpt-5`: model used by Weco to propose edits (change as desired).

### Tips

- Ensure your OpenAI key has access to a vision-capable model (default: `gpt-4o-mini` in the eval; change via `--model`).
- Adjust `--num-workers` to balance throughput and rate limits.
- You can tweak baseline behavior in `optimize.py` (prompt, temperature) — Weco will explore modifications automatically during optimization.


