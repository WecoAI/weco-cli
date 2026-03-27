"""Deterministic evaluators for the Zendesk-style triage benchmark."""

VALID_CATEGORIES = {
    "account_access",
    "billing",
    "shipping",
    "refund",
    "product_issue",
    "general_question",
}
VALID_PRIORITIES = {"low", "medium", "high"}


def _to_bool_or_none(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "yes", "1"}:
            return True
        if v in {"false", "no", "0"}:
            return False
    return None


def _extract(run, example):
    predicted = run.outputs or {}
    expected = example.outputs or {}

    pred_category = predicted.get("category")
    pred_priority = predicted.get("priority")
    pred_requires_human = _to_bool_or_none(predicted.get("requires_human"))

    exp_category = expected.get("category")
    exp_priority = expected.get("priority")
    exp_requires_human = _to_bool_or_none(expected.get("requires_human"))

    return {
        "pred_category": pred_category,
        "pred_priority": pred_priority,
        "pred_requires_human": pred_requires_human,
        "exp_category": exp_category,
        "exp_priority": exp_priority,
        "exp_requires_human": exp_requires_human,
    }


def schema_validity(run, example) -> dict:
    values = _extract(run, example)
    is_valid = (
        values["pred_category"] in VALID_CATEGORIES
        and values["pred_priority"] in VALID_PRIORITIES
        and values["pred_requires_human"] is not None
    )
    return {
        "key": "schema_validity",
        "score": 1.0 if is_valid else 0.0,
        "comment": "Valid schema output" if is_valid else "Invalid schema or labels",
    }


def category_accuracy(run, example) -> dict:
    values = _extract(run, example)
    ok = values["pred_category"] == values["exp_category"]
    return {
        "key": "category_accuracy",
        "score": 1.0 if ok else 0.0,
        "comment": "Category match" if ok else "Category mismatch",
    }


def priority_accuracy(run, example) -> dict:
    values = _extract(run, example)
    ok = values["pred_priority"] == values["exp_priority"]
    return {
        "key": "priority_accuracy",
        "score": 1.0 if ok else 0.0,
        "comment": "Priority match" if ok else "Priority mismatch",
    }


def requires_human_accuracy(run, example) -> dict:
    values = _extract(run, example)
    ok = values["pred_requires_human"] == values["exp_requires_human"]
    return {
        "key": "requires_human_accuracy",
        "score": 1.0 if ok else 0.0,
        "comment": "Requires-human match" if ok else "Requires-human mismatch",
    }


def record_exact_match(run, example) -> dict:
    values = _extract(run, example)
    ok = (
        values["pred_category"] == values["exp_category"]
        and values["pred_priority"] == values["exp_priority"]
        and values["pred_requires_human"] == values["exp_requires_human"]
    )
    return {
        "key": "record_exact_match",
        "score": 1.0 if ok else 0.0,
        "comment": "All fields match" if ok else "At least one field differs",
    }

