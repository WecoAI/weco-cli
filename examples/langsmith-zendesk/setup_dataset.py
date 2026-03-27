"""Create optimization and validation dataset for Zendesk-style triage."""

from typing import Dict, List, Set, Tuple

from langsmith import Client

from optimization_tickets import OPTIMIZATION_TICKETS
from validation_tickets import VALIDATION_TICKETS

DATASET_NAME = "zendesk-triage"
DATASET_DESCRIPTION = "Synthetic Zendesk-style triage benchmark"

SPLITS = {
    "opt": OPTIMIZATION_TICKETS,
    "val": VALIDATION_TICKETS,
}


def _ticket_map(tickets: List[dict]) -> Dict[int, dict]:
    return {ticket["id"]: ticket for ticket in tickets}


def _get_or_create_dataset(client: Client, dataset_name: str, description: str):
    try:
        dataset = client.create_dataset(dataset_name=dataset_name, description=description)
        print(f"Created dataset: {dataset_name}")
        return dataset
    except Exception:
        dataset = client.read_dataset(dataset_name=dataset_name)
        print(f"Using existing dataset: {dataset_name}")
        return dataset


def _existing_ticket_ids(client: Client, dataset_id) -> Set[int]:
    existing = set()
    for example in client.list_examples(dataset_id=dataset_id):
        metadata = getattr(example, "metadata", None) or {}
        raw_ticket_id = metadata.get("ticket_id")
        try:
            existing.add(int(raw_ticket_id))
        except (TypeError, ValueError):
            continue
    return existing


def _populate_split(client: Client, dataset, split_name: str, tickets: List[dict]) -> Tuple[int, int]:
    by_id = _ticket_map(tickets)
    ticket_ids = [ticket["id"] for ticket in tickets]
    existing = _existing_ticket_ids(client, dataset.id)
    added = 0
    skipped = 0

    for ticket_id in ticket_ids:
        if ticket_id in existing:
            skipped += 1
            continue

        ticket = by_id[ticket_id]
        client.create_example(
            inputs={
                "ticket_id": ticket_id,
                "subject": ticket["subject"],
                "description": ticket["description"],
            },
            outputs=ticket["label"],
            dataset_id=dataset.id,
            metadata={"ticket_id": ticket_id, "split": split_name},
            split=split_name,
        )
        added += 1

    return added, skipped


def main():
    client = Client()
    dataset = _get_or_create_dataset(client, DATASET_NAME, DATASET_DESCRIPTION)

    for split_name, tickets in SPLITS.items():
        added, skipped = _populate_split(client, dataset, split_name, tickets)
        print(
            f"{DATASET_NAME} [{split_name}]: "
            f"added={added}, skipped_existing={skipped}, total_target={len(tickets)}"
        )

    print("\nRun optimization:")
    print("  weco run --source agent.py \\")
    print("    --eval-backend langsmith \\")
    print(f"    --langsmith-dataset {DATASET_NAME} \\")
    print("    --langsmith-splits opt \\")
    print(
        "    --langsmith-evaluators "
        "evaluators:record_exact_match evaluators:schema_validity "
        "evaluators:category_accuracy evaluators:priority_accuracy "
        "evaluators:requires_human_accuracy \\"
    )
    print("    --metric record_exact_match --goal maximize --steps 30")

    print("\nRun holdout validation (same command, validation split):")
    print("  weco run --source agent.py \\")
    print("    --eval-backend langsmith \\")
    print(f"    --langsmith-dataset {DATASET_NAME} \\")
    print("    --langsmith-splits val \\")
    print(
        "    --langsmith-evaluators "
        "evaluators:record_exact_match evaluators:schema_validity "
        "evaluators:category_accuracy evaluators:priority_accuracy "
        "evaluators:requires_human_accuracy \\"
    )
    print("    --metric record_exact_match --goal maximize --steps 1")


if __name__ == "__main__":
    main()
