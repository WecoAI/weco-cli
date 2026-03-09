"""Create ZephHR QA dataset with splits in LangSmith (idempotent).

Reads the JSON data files and creates/updates a single dataset with two splits:
- opt        (optimization split — 15 questions)
- holdout    (held-out validation split — 10 questions)
"""

import json
from pathlib import Path
from typing import Set, Tuple

from langsmith import Client

DATA_DIR = Path(__file__).with_name("data")

DATASET_NAME = "zephhr-qa"
DATASET_DESCRIPTION = "ZephHR QA benchmark"

SPLITS = {
    "opt": "optimization_questions.json",
    "holdout": "holdout_questions.json",
}


def _get_or_create_dataset(client: Client, name: str, description: str):
    try:
        dataset = client.create_dataset(dataset_name=name, description=description)
        print(f"Created dataset: {name}")
        return dataset
    except Exception:
        dataset = client.read_dataset(dataset_name=name)
        print(f"Using existing dataset: {name}")
        return dataset


def _existing_ids(client: Client, dataset_id) -> Set[str]:
    existing = set()
    for example in client.list_examples(dataset_id=dataset_id):
        metadata = getattr(example, "metadata", None) or {}
        case_id = metadata.get("case_id")
        if case_id:
            existing.add(str(case_id))
    return existing


def _populate(client: Client, dataset, split: str, records: list) -> Tuple[int, int]:
    existing = _existing_ids(client, dataset.id)
    added = skipped = 0

    for record in records:
        case_id = record["id"]
        if case_id in existing:
            skipped += 1
            continue

        client.create_example(
            inputs={"question": record["question"]},
            outputs={"expected_answer": record["expected_answer"]},
            dataset_id=dataset.id,
            metadata={"case_id": case_id, "split": split},
            split=split,
        )
        added += 1

    return added, skipped


def main():
    client = Client()
    dataset = _get_or_create_dataset(client, DATASET_NAME, DATASET_DESCRIPTION)

    for split, filename in SPLITS.items():
        records = json.loads((DATA_DIR / filename).read_text())
        added, skipped = _populate(client, dataset, split, records)
        print(f"  {DATASET_NAME} [{split}]: added={added}, skipped_existing={skipped}, total_target={len(records)}")

    print("\n--- Run optimization ---")
    print("weco run --source agent.py \\")
    print("  --eval-backend langsmith \\")
    print(f"  --langsmith-dataset {DATASET_NAME} \\")
    print("  --langsmith-splits opt \\")
    print("  --langsmith-target agent:answer_hr_question \\")
    print("  --langsmith-evaluators evaluators:json_schema_validity evaluators:conciseness \\")
    print("  --langsmith-dashboard-evaluators helpfulness correctness \\")
    print("  --langsmith-metric-function evaluators:qa_score \\")
    print("  --additional-instructions optimizer_exemplars.md \\")
    print("  --metric qa_score --goal maximize --steps 30")

    print("\n--- Run holdout validation ---")
    print("weco run --source agent.py \\")
    print("  --eval-backend langsmith \\")
    print(f"  --langsmith-dataset {DATASET_NAME} \\")
    print("  --langsmith-splits holdout \\")
    print("  --langsmith-target agent:answer_hr_question \\")
    print("  --langsmith-evaluators evaluators:json_schema_validity evaluators:conciseness \\")
    print("  --langsmith-dashboard-evaluators helpfulness correctness \\")
    print("  --langsmith-metric-function evaluators:qa_score \\")
    print("  --metric qa_score --goal maximize --steps 1")


if __name__ == "__main__":
    main()
