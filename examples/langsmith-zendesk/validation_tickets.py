"""Validation split tickets and labels for Zendesk-style triage benchmark."""

VALIDATION_TICKETS = [
    {
        "id": 9,
        "subject": "Package says delivered but isn't here",
        "description": "Tracking shows delivered this morning but nothing arrived.",
        "label": {"category": "shipping", "priority": "high", "requires_human": True},
    },
    {
        "id": 12,
        "subject": "Can't log in after email change",
        "description": "I updated my email and now I can't log into my account.",
        "label": {"category": "account_access", "priority": "high", "requires_human": False},
    },
    {
        "id": 13,
        "subject": "Refund not received",
        "description": "Customer service told me a refund was issued but I still haven't received it.",
        "label": {"category": "refund", "priority": "medium", "requires_human": True},
    },
    {
        "id": 14,
        "subject": "Subscription shows active",
        "description": "I cancelled my plan but it still shows active in my dashboard.",
        "label": {"category": "billing", "priority": "medium", "requires_human": False},
    },
    {
        "id": 15,
        "subject": "Order delayed",
        "description": "My order was supposed to arrive Monday but hasn't arrived yet.",
        "label": {"category": "shipping", "priority": "medium", "requires_human": False},
    },
    {
        "id": 16,
        "subject": "Bug when exporting data",
        "description": "The export button doesn't produce a file.",
        "label": {"category": "product_issue", "priority": "medium", "requires_human": False},
    },
    {
        "id": 17,
        "subject": "Payment failed but charged",
        "description": "Checkout said payment failed but my bank shows the charge.",
        "label": {"category": "billing", "priority": "high", "requires_human": True},
    },
    {
        "id": 18,
        "subject": "Need invoice copy",
        "description": "Can you send me a copy of last month's invoice?",
        "label": {"category": "general_question", "priority": "low", "requires_human": False},
    },
    {
        "id": 19,
        "subject": "Order delivered damaged",
        "description": "The product arrived but the screen is cracked.",
        "label": {"category": "product_issue", "priority": "medium", "requires_human": True},
    },
    {
        "id": 20,
        "subject": "How long are refunds taking?",
        "description": "Just wondering how long refunds usually take.",
        "label": {"category": "refund", "priority": "low", "requires_human": False},
    },
]

