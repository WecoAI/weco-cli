"""Create ZephHR QA datasets in LangFuse (idempotent).

Reads the JSON data files and creates two separate datasets:
- zephhr-qa-opt       (optimization — 15 questions)
- zephhr-qa-holdout   (held-out validation — 10 questions)

LangFuse does not have native dataset splits, so we use separate datasets.
"""

import json
from pathlib import Path
from typing import Set, Tuple

from langfuse import Langfuse

DATA_DIR = Path(__file__).with_name("data")

DATASETS = {
    "zephhr-qa-opt": {"file": "optimization_questions.json", "desc": "ZephHR QA optimization set"},
    "zephhr-qa-holdout": {"file": "holdout_questions.json", "desc": "ZephHR QA holdout validation set"},
}


def _get_or_create_dataset(client: Langfuse, name: str, description: str):
    try:
        dataset = client.create_dataset(name=name, description=description)
        print(f"Created dataset: {name}")
        return dataset
    except Exception:
        dataset = client.get_dataset(name=name)
        print(f"Using existing dataset: {name}")
        return dataset


def _existing_ids(client: Langfuse, dataset_name: str) -> Set[str]:
    existing = set()
    dataset = client.get_dataset(dataset_name)
    for item in dataset.items:
        metadata = getattr(item, "metadata", None) or {}
        case_id = metadata.get("case_id")
        if case_id:
            existing.add(str(case_id))
    return existing


def _populate(client: Langfuse, dataset_name: str, records: list) -> Tuple[int, int]:
    existing = _existing_ids(client, dataset_name)
    added = skipped = 0

    for record in records:
        case_id = record["id"]
        if case_id in existing:
            skipped += 1
            continue

        client.create_dataset_item(
            dataset_name=dataset_name,
            input={"question": record["question"]},
            expected_output={"expected_answer": record["expected_answer"]},
            metadata={"case_id": case_id},
        )
        added += 1

    return added, skipped


def main():
    client = Langfuse()

    for dataset_name, config in DATASETS.items():
        _get_or_create_dataset(client, dataset_name, config["desc"])
        records = json.loads((DATA_DIR / config["file"]).read_text())
        added, skipped = _populate(client, dataset_name, records)
        print(f"  {dataset_name}: added={added}, skipped_existing={skipped}, total_target={len(records)}")

    client.flush()

    print("\n--- Run optimization ---")
    print("weco run --source agent.py \\")
    print("  --eval-backend langfuse \\")
    print("  --langfuse-dataset zephhr-qa-opt \\")
    print("  --langfuse-target agent:answer_hr_question \\")
    print("  --langfuse-evaluators evaluators:json_schema_validity evaluators:conciseness \\")
    print("  --langfuse-managed-evaluators helpfulness correctness \\")
    print("  --langfuse-metric-function evaluators:qa_score \\")
    print("  --additional-instructions optimizer_exemplars.md \\")
    print("  --metric qa_score --goal maximize --steps 30")

    print("\n--- Run holdout validation ---")
    print("weco run --source agent.py \\")
    print("  --eval-backend langfuse \\")
    print("  --langfuse-dataset zephhr-qa-holdout \\")
    print("  --langfuse-target agent:answer_hr_question \\")
    print("  --langfuse-evaluators evaluators:json_schema_validity evaluators:conciseness \\")
    print("  --langfuse-managed-evaluators helpfulness correctness \\")
    print("  --langfuse-metric-function evaluators:qa_score \\")
    print("  --metric qa_score --goal maximize --steps 1")


if __name__ == "__main__":
    main()
