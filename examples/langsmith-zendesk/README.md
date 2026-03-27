# Weco + LangSmith: Zendesk-Style Triage Benchmark

Optimize a prompt-based support ticket triage agent with LangSmith datasets and deterministic evaluators.

## Why this example exists

This example is intentionally small and synthetic. Its goal is not to model the full complexity of support operations, but to provide a fast, reproducible prompt optimization task with realistic structure and objectively scorable outputs.

## Task

Input:

```json
{
  "subject": "...",
  "description": "..."
}
```

Output:

```json
{
  "category": "account_access | billing | shipping | refund | product_issue | general_question",
  "priority": "low | medium | high",
  "requires_human": true
}
```

## Dataset Split

- Optimization split dataset: `zendesk-triage-opt` (10 tickets)
- Validation split dataset: `zendesk-triage-val` (10 tickets)

The split is stored in two separate files:
- `optimization_tickets.py`
- `validation_tickets.py`

If you want strict holdout isolation during optimization, keep only
`optimization_tickets.py` in the optimization workspace and run validation
from a separate workspace that contains `validation_tickets.py`.

## Metrics (deterministic)

- Primary optimization metric: `record_exact_match`
- Secondary diagnostics:
  - `schema_validity`
  - `category_accuracy`
  - `priority_accuracy`
  - `requires_human_accuracy`

No LLM judge is used in the optimization objective.

## Prerequisites

```bash
export LANGCHAIN_API_KEY="your-langsmith-api-key"
export OPENAI_API_KEY="your-openai-api-key"
```

## Setup

Create or update both datasets (idempotent by `ticket_id` metadata):

```bash
python setup_dataset.py
```

## Run Optimization

```bash
weco run --source agent.py \
  --eval-backend langsmith \
  --langsmith-dataset zendesk-triage \
  --langsmith-splits opt \
  --langsmith-target agent:triage_ticket \
  --langsmith-evaluators evaluators:record_exact_match evaluators:schema_validity evaluators:category_accuracy evaluators:priority_accuracy evaluators:requires_human_accuracy \
  --metric record_exact_match \
  --goal maximize \
  --steps 30
```

To expose exemplar guidance to the optimizer, pass:

```bash
weco run --source agent.py \
  --eval-backend langsmith \
  --langsmith-dataset zendesk-triage \
  --langsmith-splits opt \
  --langsmith-target agent:triage_ticket \
  --langsmith-evaluators evaluators:record_exact_match evaluators:schema_validity evaluators:category_accuracy evaluators:priority_accuracy evaluators:requires_human_accuracy \
  --metric record_exact_match \
  --goal maximize \
  --steps 30 \
  --additional-instructions optimizer_exemplars.md
```

## Run Holdout Validation

Use the same evaluators against the validation split to check generalization:

```bash
weco run --source agent.py \
  --eval-backend langsmith \
  --langsmith-dataset zendesk-triage \
  --langsmith-splits val \
  --langsmith-target agent:triage_ticket \
  --langsmith-evaluators evaluators:record_exact_match evaluators:schema_validity evaluators:category_accuracy evaluators:priority_accuracy evaluators:requires_human_accuracy \
  --metric record_exact_match \
  --goal maximize \
  --steps 1
```

## Files

- `agent.py` — baseline triage prompt/function Weco optimizes
- `evaluators.py` — deterministic evaluators and primary objective
- `setup_dataset.py` — creates optimization + validation datasets idempotently
- `optimization_tickets.py` — optimization split tickets and labels
- `validation_tickets.py` — validation split tickets and labels
- `optimizer_exemplars.md` — optimizer-visible exemplar guidance (kept separate from optimization/validation tickets)

