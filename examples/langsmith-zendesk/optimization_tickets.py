"""Optimization split tickets and labels for Zendesk-style triage benchmark."""

OPTIMIZATION_TICKETS = [
    {
        "id": 1,
        "subject": "Password reset not working",
        "description": "I tried resetting my password three times but the reset link keeps saying it's expired.",
        "label": {"category": "account_access", "priority": "high", "requires_human": False},
    },
    {
        "id": 2,
        "subject": "Still waiting for my order",
        "description": "Tracking says shipped but it hasn't moved in 5 days.",
        "label": {"category": "shipping", "priority": "medium", "requires_human": False},
    },
    {
        "id": 3,
        "subject": "Charged after cancellation",
        "description": "I cancelled my subscription last week but my card was charged again today.",
        "label": {"category": "billing", "priority": "high", "requires_human": True},
    },
    {
        "id": 4,
        "subject": "Refund status",
        "description": "I returned the item 10 days ago and haven't seen the refund yet.",
        "label": {"category": "refund", "priority": "medium", "requires_human": True},
    },
    {
        "id": 5,
        "subject": "App crashes on upload",
        "description": "Whenever I try to upload a document the app immediately crashes.",
        "label": {"category": "product_issue", "priority": "medium", "requires_human": False},
    },
    {
        "id": 6,
        "subject": "Delivery address mistake",
        "description": "I just placed an order but noticed the shipping address is wrong.",
        "label": {"category": "shipping", "priority": "high", "requires_human": True},
    },
    {
        "id": 7,
        "subject": "Update account email",
        "description": "How do I change the email address on my account?",
        "label": {"category": "account_access", "priority": "low", "requires_human": False},
    },
    {
        "id": 8,
        "subject": "Incorrect delivery charge",
        "description": "I was charged 12 GBP shipping even though the page said free delivery.",
        "label": {"category": "billing", "priority": "medium", "requires_human": False},
    },
    {
        "id": 10,
        "subject": "Question about pricing",
        "description": "Do you offer discounts for annual subscriptions?",
        "label": {"category": "general_question", "priority": "low", "requires_human": False},
    },
    {
        "id": 11,
        "subject": "Broken after two days",
        "description": "The headphones stopped working after only two days.",
        "label": {"category": "product_issue", "priority": "medium", "requires_human": False},
    },
]

