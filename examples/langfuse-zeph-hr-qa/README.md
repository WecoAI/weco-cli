# ZephHR QA — LangFuse + Weco Example

Optimize a QA agent that answers HR policy questions over fictional ZephHR documentation.

This example demonstrates using Weco with LangFuse as the evaluation backend. It uses LangFuse datasets, local code evaluators, and managed LLM-as-a-Judge evaluators configured in the LangFuse UI.

## Prerequisites

- Python 3.10+
- `uv pip install 'weco[langfuse]' openai langfuse`
- Environment variables:
  ```bash
  export OPENAI_API_KEY="..."
  export LANGFUSE_SECRET_KEY="sk-lf-..."
  export LANGFUSE_PUBLIC_KEY="pk-lf-..."
  export LANGFUSE_BASE_URL="https://cloud.langfuse.com"  # or https://us.cloud.langfuse.com
  ```

## LangFuse UI Setup

Before running optimization, configure two **managed evaluators** (LLM-as-a-Judge) in your LangFuse project. These run server-side and score each agent response automatically.

1. Go to your project in [LangFuse](https://cloud.langfuse.com/) → **Evaluation** → **Evaluators**
2. Click **+ New Evaluator** and create two evaluators:

### Correctness evaluator

- **Name**: `Correctness`
- **Score**: 0 or 1 (binary factual accuracy)
- **Variable mappings**:
  - `{{input}}` → `$.input.question` (the user's question)
  - `{{output}}` → `$.output.answer` (the agent's answer)
  - `{{expected_output}}` → `$.expected_output.expected_answer` (the ground truth)

### Helpfulness evaluator

- **Name**: `Helpfulness`
- **Score**: 0–1 continuous scale
- **Variable mappings**:
  - `{{input}}` → `$.input.question`
  - `{{output}}` → `$.output.answer`

> **Important:** Use the **live preview** when configuring each evaluator to verify the variable mappings are picking up the correct data from your traces. The evaluator names are case-sensitive — `Correctness` and `Helpfulness` must match exactly what you pass to `--langfuse-managed-evaluators`.

The custom metric function `evaluators:qa_score` combines these scores locally: `Correctness * Helpfulness`.

## Setup

Create the LangFuse datasets:

```bash
cd examples/langfuse-zeph-hr-qa
python setup_dataset.py
```

This creates two datasets: `zephhr-qa-opt` (optimization) and `zephhr-qa-holdout` (validation).

## Optimize

```bash
weco run --source agent.py \
  --eval-backend langfuse \
  --langfuse-dataset zephhr-qa-opt \
  --langfuse-target agent:answer_hr_question \
  --langfuse-evaluators evaluators:json_schema_validity evaluators:conciseness \
  --langfuse-managed-evaluators Correctness Helpfulness \
  --langfuse-metric-function evaluators:qa_score \
  --additional-instructions optimizer_exemplars.md \
  --metric qa_score --goal maximize --steps 30
```

## Holdout Validation

```bash
weco run --source agent.py \
  --eval-backend langfuse \
  --langfuse-dataset zephhr-qa-holdout \
  --langfuse-target agent:answer_hr_question \
  --langfuse-evaluators evaluators:json_schema_validity evaluators:conciseness \
  --langfuse-managed-evaluators Correctness Helpfulness \
  --langfuse-metric-function evaluators:qa_score \
  --metric qa_score --goal maximize --steps 1
```

## File Overview

| File | Purpose |
|------|---------|
| `agent.py` | Baseline QA agent (gpt-4o-mini) — Weco optimizes the prompt |
| `evaluators.py` | LangFuse-format evaluators + `qa_score` metric function |
| `setup_dataset.py` | Idempotent LangFuse dataset creation from JSON |
| `docs.md` | ZephHR product documentation (knowledge base) |
| `optimizer_exemplars.md` | Few-shot Q&A examples passed via `--additional-instructions` |
| `data/optimization_questions.json` | Optimization set (15 questions) |
| `data/holdout_questions.json` | Held-out validation set (10 questions) |
