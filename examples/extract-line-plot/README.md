## Extract Line Plot (Chart → CSV): Accuracy/Cost Optimization for Agentic Workflow

This example demonstrates optimizing an AI feature that turns chart images into CSV tables, showcasing how to use Weco to improve accuracy or reduce cost of a VLM-based extraction workflow.

### Prerequisites

- Python 3.9+
- `uv` installed (see `https://docs.astral.sh/uv/`)
- An OpenAI API key in your environment:

```bash
export OPENAI_API_KEY=your_key_here
```

### Files

- `prepare_data.py`: downloads ChartQA (full) and prepares a 100-sample subset of line charts.
- `optimize.py`: exposes `extract_csv(image_path)` which returns CSV text plus the per-call cost (helpers stay private).
- `eval.py`: evaluation harness that runs the baseline on images and reports a similarity score as "accuracy".
- `guide.md`: optional additional instructions you can feed to Weco via `--additional-instructions guide.md`.

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

To emit a secondary `cost` metric that Weco can minimize (while enforcing `accuracy > 0.45`), append `--cost-metric`:

```bash
uv run --with openai python eval.py --max-samples 10 --num-workers 4 --cost-metric
```

If the final accuracy falls at or below `0.45`, the reported cost is replaced with a large penalty so Weco keeps searching for higher-accuracy solutions.
You can tighten or relax this constraint with `--cost-accuracy-threshold`, e.g. `--cost-accuracy-threshold 0.50`.

### 3) Optimize the baseline with Weco

Run Weco to iteratively improve `optimize.py` using 100 examples and many workers:

```bash
weco run --source optimize.py --eval-command 'uv run --with openai python eval.py --max-samples 100 --num-workers 50' --metric accuracy --goal maximize --steps 20 --model gpt-5 --additional-instructions guide.md
```

Arguments:
- `--source optimize.py`: file that Weco will edit to improve results.
- `--eval-command '…'`: command Weco executes to measure the metric.
- `--metric accuracy`: Weco parses `accuracy: <value>` from `eval.py` output.
- `--goal maximize`: higher is better.
- `--steps 20`: number of optimization iterations.
- `--model gpt-5`: model used by Weco to propose edits (change as desired).

To minimize cost instead (subject to the accuracy constraint), enable the flag in the eval command and switch the optimization target:

```bash
weco run --source optimize.py --eval-command 'uv run --with openai python eval.py --max-samples 100 --num-workers 50 --cost-metric' --metric cost --goal minimize --steps 20 --model gpt-5 --additional-instructions guide.md
```

#### Cost optimization workflow
- Run the evaluation command with `--cost-metric` once to confirm accuracy meets your threshold and note the baseline cost.
- Adjust `--cost-accuracy-threshold` if you want to tighten or relax the constraint before launching optimization.
- Kick off Weco with `--metric cost --goal minimize --additional-instructions guide.md` so the optimizer respects the constraint while acting on the extra tips.

### Tips

- Ensure your OpenAI key has access to a vision-capable model (default: `gpt-4o-mini` in the eval; change via `--model`).
- Adjust `--num-workers` to balance throughput and rate limits.
- You can tweak baseline behavior in `optimize.py` (prompt, temperature) — Weco will explore modifications automatically during optimization.
- Include `--additional-instructions guide.md` whenever you run Weco so those cost-conscious hints influence the generated proposals.
