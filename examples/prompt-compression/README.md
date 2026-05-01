# Prompt Compression

This example shows how **Weco** compresses a long, over-engineered LLM system
prompt while keeping its classification accuracy intact. The classifier task
is the public [PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77)
77-intent banking dataset; the prompt to compress is a deliberately bloated
65,887-character system prompt that mimics real-world enterprise prompt patterns
(verbose preamble, per-class blocks with Description / Typical phrasings /
Disambiguation / Output sections, FAQ, worked examples).

Treating character count as a cost to minimize and accuracy as a constraint to
preserve, Weco searches the prompt space and finds dramatic compression with
no measurable accuracy drop.

> **Headline result** with `--model claude-opus-4-7` × 50 steps:
> **65,887 → 3,229 chars (95.1% reduction)** holding accuracy at the
> baseline-minus-2pp threshold. ([share link](https://weco.ai/share/XSRQdS7vfMdt9beD3KR1tlhg7By-FFIo))

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/WecoAI/weco-cli.git
   cd weco-cli/examples/prompt-compression
   ```

2. Install the CLI and dependencies:
   ```bash
   pip install weco openai "datasets<4.0"
   ```

3. Set your OpenAI API key (used by the classifier; Weco's optimizer LLM is
   billed via Weco credits):
   ```bash
   export OPENAI_API_KEY="your_key_here"
   ```

4. Bake the bloated baseline into `optimize.py`:
   ```bash
   python bake_optimize.py
   ```
   This deterministically generates a 65,887-char `SYSTEM_PROMPT` and writes
   `optimize.py`. Run this **once** before kicking off Weco — re-running it
   afterwards will overwrite Weco's progress.

5. (Optional) Re-measure the baseline accuracy for your environment:
   ```bash
   python measure_baseline.py
   ```
   This writes `baseline_accuracy.json`. A pre-measured value of 0.7700 (200
   samples, seed=0, `gpt-5-mini`, `reasoning_effort=minimal`) is committed; if
   you change the classifier model or sample size you should re-measure.

## Files in this folder

| File | Purpose |
| :--- | :--- |
| `optimize.py` | Holds the `SYSTEM_PROMPT` string and the `classify(query, model)` function. **Weco mutates only the `SYSTEM_PROMPT` string content** in the marked WECO-MUTABLE REGION. The `classify()` function and surrounding code remain intact. |
| `eval.py` | Loads 200 BANKING77 test samples (seed=0), runs `optimize.classify` in parallel, parses the LLM output to a canonical label, and emits the three lines `accuracy:`, `chars:`, `metric:`. The composite `metric = chars` if `accuracy ≥ ACCURACY_THRESHOLD` else `chars + 10⁷` (penalty). |
| `labels.py` | The 77 BANKING77 canonical labels in dataset-index order, plus `parse_predicted_label()` for robust label extraction from free-text LLM responses. |
| `build_bloated_prompt.py` | Deterministically generates the 65,887-char baseline prompt. Idempotent. |
| `bake_optimize.py` | Bakes the generated baseline into `optimize.py`. Run once at setup. |
| `measure_baseline.py` | Measures baseline accuracy on the 200-sample slice and writes `baseline_accuracy.json`. |
| `baseline_accuracy.json` | Frozen baseline accuracy (consumed by `eval.py` to set the threshold). |
| `prompt_guide.md` | The `--additional-instructions` content the optimizer LLM reads. |

## Run Weco

```bash
weco run --source optimize.py \
     --eval-command "python eval.py" \
     --metric metric \
     --goal minimize \
     --steps 50 \
     --model claude-opus-4-7 \
     --additional-instructions prompt_guide.md \
     --apply-change
```

You'll see eval output streaming for each step, then a step summary like:

```text
[setup] loading 200 BANKING77 test samples (seed=0)
[setup] running 200 classifications via gpt-5-mini (20 workers)
[progress] 40/200 completed, accuracy: 0.7250, parse-rate: 1.0000, elapsed 5.7s
...
[progress] 200/200 completed, accuracy: 0.7700, parse-rate: 1.0000, elapsed 21.6s
accuracy: 0.7700
chars: 65887
metric: 65887
```

Weco then proposes mutations to `SYSTEM_PROMPT`, re-evaluates, and pushes the
metric (chars) down while the constraint (accuracy ≥ baseline − 2pp) holds.

## What to expect

In our 50-step runs the optimizer found two qualitatively different solutions:

| Optimizer | best chars | reduction | acc | strategy |
| :--- | ---: | ---: | ---: | :--- |
| `claude-opus-4-7` | **3,229** | **95.1%** | 0.7500 | bare label list + 12 cluster disambiguation rules (`Refund: initiating=request_refund; missing=Refund_not_showing_up; …`) |
| `gpt-5.5` | 6,828 | 89.6% | 0.7550 | one terse cue per label (`activate_my_card: activate received/new card`, ~80 chars × 77) |

Both held the accuracy threshold. Opus's strategy is roughly **2× more
compressed** at the same quality — it trusts the model to infer meaning from
label *names* alone for most classes, only adding explicit hints for the
clusters of confusable intents.

## How it works

* **Loss**: `metric = chars` when accuracy clears the threshold; else
  `chars + 10_000_000`. Weco minimizes `metric`, so any variant that loses
  accuracy is heavily penalized.
* **Threshold**: `baseline_accuracy − 2pp`, loaded from
  `baseline_accuracy.json`. The 2pp slack covers per-call sampling variance
  on 200 samples.
* **Eval cost**: ~$0.50–1.00 per evaluation pass at full prompt size, dropping
  linearly as the prompt shrinks. A 50-step run is ~$5–15 for evaluation; the
  optimizer-LLM cost is billed separately via Weco credits.
* **Reproducibility**: the baseline prompt is generated by
  `build_bloated_prompt.py` from a deterministic template + the 77-label list.
  Anyone who runs `python bake_optimize.py` gets bit-identical
  `SYSTEM_PROMPT`. The 200-sample test slice is fixed by `EVAL_SEED=0`.

## A note on the baseline

The 65K-char "bloated" baseline is **synthetic** — generated programmatically
to mimic real production-style classifier prompts (operating principles,
per-class blocks, FAQ, worked examples). PolyAI's BANKING77 dataset itself is
real and public, but it ships only `(text, label)` pairs and predates the
prompt-engineering era. We use the synthetic baseline so the example is fully
public and reproducible. The compression ratios we observe (~95%) line up with
what we've seen on real customer prompts of similar size and shape.

## Next Steps

* Read the [Prompt Engineering example](../prompt/README.md) for the
  *maximize-accuracy* shape (AIME math), which complements the
  *minimize-chars-with-accuracy-floor* shape used here.
* See the [CLI Reference](https://docs.weco.ai/cli/cli-reference) for all
  `weco run` options.
