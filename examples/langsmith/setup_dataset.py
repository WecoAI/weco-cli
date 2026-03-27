"""Create optimization and validation dataset for reply drafting."""

from typing import Dict, List, Set, Tuple

from langsmith import Client

from cases import CASES, OPTIMIZATION_CASE_IDS, VALIDATION_CASE_IDS

DATASET_NAME = "acme-reply-drafting"
DATASET_DESCRIPTION = "Acme Cloud reply-drafting benchmark"

SPLITS = {
    "opt": OPTIMIZATION_CASE_IDS,
    "val": VALIDATION_CASE_IDS,
}


def _cases_by_id() -> Dict[str, dict]:
    return {case["case_id"]: case for case in CASES}


def _get_or_create_dataset(client: Client, dataset_name: str, description: str):
    try:
        dataset = client.create_dataset(dataset_name=dataset_name, description=description)
        print(f"Created dataset: {dataset_name}")
        return dataset
    except Exception:
        dataset = client.read_dataset(dataset_name=dataset_name)
        print(f"Using existing dataset: {dataset_name}")
        return dataset


def _existing_case_ids(client: Client, dataset_id) -> Set[str]:
    existing = set()
    for example in client.list_examples(dataset_id=dataset_id):
        metadata = getattr(example, "metadata", None) or {}
        case_id = metadata.get("case_id")
        if case_id:
            existing.add(str(case_id))
    return existing


def _populate_split(client: Client, dataset, split_name: str, case_ids: List[str]) -> Tuple[int, int]:
    by_id = _cases_by_id()
    existing = _existing_case_ids(client, dataset.id)
    added = 0
    skipped = 0

    for case_id in case_ids:
        if case_id in existing:
            skipped += 1
            continue

        case = by_id[case_id]
        client.create_example(
            inputs={
                "case_id": case_id,
                "subject": case["subject"],
                "message": case["message"],
            },
            outputs={
                "ideal_reply": case["ideal_reply"],
                "gold_doc_ids": case["gold_doc_ids"],
                "should_escalate": case["should_escalate"],
            },
            dataset_id=dataset.id,
            metadata={"case_id": case_id, "split": split_name},
            split=split_name,
        )
        added += 1

    return added, skipped


def main():
    client = Client()
    dataset = _get_or_create_dataset(client, DATASET_NAME, DATASET_DESCRIPTION)

    for split_name, case_ids in SPLITS.items():
        added, skipped = _populate_split(client, dataset, split_name, case_ids)
        print(
            f"{DATASET_NAME} [{split_name}]: "
            f"added={added}, skipped_existing={skipped}, total_target={len(case_ids)}"
        )

    print("\nRun optimization:")
    print("  weco run --source agent.py \\")
    print("    --eval-backend langsmith \\")
    print(f"    --langsmith-dataset {DATASET_NAME} \\")
    print("    --langsmith-splits opt \\")
    print(
        "    --langsmith-evaluators "
        "evaluators:grounded_reply_quality "
        "evaluators:retrieval_hit_at_2 "
        "evaluators:escalation_match "
        "evaluators:reply_length_ok \\"
    )
    print("    --metric grounded_reply_quality --goal maximize --steps 30")

    print("\nRun holdout validation (same command, validation split):")
    print("  weco run --source agent.py \\")
    print("    --eval-backend langsmith \\")
    print(f"    --langsmith-dataset {DATASET_NAME} \\")
    print("    --langsmith-splits val \\")
    print(
        "    --langsmith-evaluators "
        "evaluators:grounded_reply_quality "
        "evaluators:retrieval_hit_at_2 "
        "evaluators:escalation_match "
        "evaluators:reply_length_ok \\"
    )
    print("    --metric grounded_reply_quality --goal maximize --steps 1")


if __name__ == "__main__":
    main()
