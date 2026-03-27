# Weco + LangSmith: Reply Drafting Example

Optimize a context-grounded customer-support reply agent using LangSmith datasets and evaluators.

## Why this example exists

This is a small synthetic benchmark modeled on real support workflows. It is not a full production support system. The goal is to provide a fast, readable example of reply drafting with simple local retrieval, one primary LLM-judge metric, and a holdout validation split.

## Task

Input:

```json
{
  "subject": "...",
  "message": "..."
}
```

Output:

```json
{
  "reply": "...",
  "retrieved_doc_ids": ["doc_a", "doc_b"],
  "should_escalate": false
}
```

## Retrieval Design

The agent does a tiny local retrieval step before drafting a reply:

- support docs live in `knowledge_base.py`
- retrieval uses simple lexical matching in `retrieval.py`
- top 2 docs are passed into the prompt as policy context

This makes the example more realistic than directly embedding the exact answer context into the dataset, while staying simple enough to explain in a blog post.

## Dataset Split

- Optimization split dataset: `acme-reply-drafting-opt` (12 cases)
- Validation split dataset: `acme-reply-drafting-val` (8 cases)

## Evaluators

Primary optimization metric:

- `grounded_reply_quality`

Deterministic diagnostics:

- `retrieval_hit_at_2`
- `must_include_coverage`
- `forbidden_phrase_violation`
- `escalation_match`
- `reply_length_ok`

Only `grounded_reply_quality` is optimized. The other metrics are there to help explain what changed.

## Prerequisites

```bash
export LANGCHAIN_API_KEY="your-langsmith-api-key"
export OPENAI_API_KEY="your-openai-api-key"
```

## Setup

Create or update the LangSmith datasets:

```bash
python setup_dataset.py
```

The setup script is idempotent and uses stable `case_id` metadata so reruns do not duplicate cases.

## Run Optimization

```bash
weco run --source agent.py \
  --eval-backend langsmith \
  --langsmith-dataset acme-reply-drafting \
  --langsmith-splits opt \
  --langsmith-target agent:answer_question \
  --langsmith-evaluators evaluators:grounded_reply_quality evaluators:retrieval_hit_at_2 evaluators:must_include_coverage evaluators:forbidden_phrase_violation evaluators:escalation_match evaluators:reply_length_ok \
  --metric grounded_reply_quality \
  --goal maximize \
  --steps 30
```

## Run Holdout Validation

Use the same evaluator set on the validation split to check whether improvements generalize:

```bash
weco run --source agent.py \
  --eval-backend langsmith \
  --langsmith-dataset acme-reply-drafting \
  --langsmith-splits val \
  --langsmith-target agent:answer_question \
  --langsmith-evaluators evaluators:grounded_reply_quality evaluators:retrieval_hit_at_2 evaluators:must_include_coverage evaluators:forbidden_phrase_violation evaluators:escalation_match evaluators:reply_length_ok \
  --metric grounded_reply_quality \
  --goal maximize \
  --steps 1
```

## What happens

1. Weco mutates `agent.py` (prompt, structure, and simple logic).
2. The agent retrieves a small set of policy docs for each case.
3. LangSmith evaluates the drafted reply using one judge metric and several deterministic diagnostics.
4. Weco uses the primary metric to guide search and reports the secondary diagnostics for inspection.

## Files

- `agent.py` — baseline reply-drafting agent Weco optimizes
- `cases.py` — synthetic support tickets, expected outputs, and split metadata
- `retrieval.py` — simple local retriever
- `knowledge_base.py` — Acme Cloud policy docs
- `evaluators.py` — one primary LLM-judge metric plus deterministic diagnostics
- `setup_dataset.py` — creates optimization and validation datasets idempotently
