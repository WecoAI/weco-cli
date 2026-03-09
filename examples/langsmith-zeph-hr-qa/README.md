# ZephHR QA — LangSmith + Weco Example

Optimize a QA agent that answers HR policy questions over fictional ZephHR documentation.

This example accompanies the [LangSmith integration tutorial](https://docs.weco.ai/integrations/langsmith). See the tutorial for a full walkthrough of connecting Weco to LangSmith datasets, evaluators, and dashboard metrics.

## Prerequisites

- Python 3.10+
- `pip install weco openai langsmith`
- Environment variables:
  ```bash
  export OPENAI_API_KEY="..."
  export LANGCHAIN_API_KEY="..."
  ```

## LangSmith Dashboard Setup

Before running optimization, configure two **online evaluators** in your LangSmith project:

1. **helpfulness** — scores how complete and useful the answer is (1–5 scale)
2. **correctness** — binary factual accuracy against the expected answer and required facts (0 or 1)

The custom metric function `evaluators:qa_score` handles aggregation locally — no dashboard metric needed.

## Setup

Create the LangSmith datasets:

```bash
cd examples/langsmith-zeph-hr-qa
python setup_dataset.py
```

This creates a single dataset `zephhr-qa` with two splits: `opt` (optimization) and `holdout` (validation).

## Optimize

```bash
weco run --source agent.py \
  --eval-backend langsmith \
  --langsmith-dataset zephhr-qa \
  --langsmith-splits opt \
  --langsmith-target agent:answer_hr_question \
  --langsmith-evaluators evaluators:json_schema_validity evaluators:conciseness \
  --langsmith-dashboard-evaluators helpfulness correctness \
  --langsmith-metric-function evaluators:qa_score \
  --additional-instructions optimizer_exemplars.md \
  --metric qa_score --goal maximize --steps 30
```

## Holdout Validation

```bash
weco run --source agent.py \
  --eval-backend langsmith \
  --langsmith-dataset zephhr-qa \
  --langsmith-splits holdout \
  --langsmith-target agent:answer_hr_question \
  --langsmith-evaluators evaluators:json_schema_validity evaluators:conciseness \
  --langsmith-dashboard-evaluators helpfulness correctness \
  --langsmith-metric-function evaluators:qa_score \
  --metric qa_score --goal maximize --steps 1
```

## File Overview

| File | Purpose |
|------|---------|
| `agent.py` | Baseline QA agent (gpt-4o-mini) — Weco optimizes the prompt |
| `evaluators.py` | Deterministic checks + `qa_score` metric function |
| `setup_dataset.py` | Idempotent LangSmith dataset creation from JSON splits |
| `docs.md` | ZephHR product documentation (knowledge base) |
| `optimizer_exemplars.md` | Few-shot Q&A examples passed via `--additional-instructions` |
| `data/optimization_questions.json` | Optimization split (15 questions) |
| `data/holdout_questions.json` | Held-out validation split (10 questions) |
