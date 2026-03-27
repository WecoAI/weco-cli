# Zendesk-Style Triage Exemplars (Reference Only)

Use this file as optimizer-visible reference guidance (for example via `--additional-instructions`).
These exemplars are intentionally separate from `optimization_tickets.py`.

## Output schema

Return JSON with:

- `category`: `account_access | billing | shipping | refund | product_issue | general_question`
- `priority`: `low | medium | high`
- `requires_human`: `true | false`

## Decision heuristics

- `requires_human=true` when the issue implies a dispute, urgent operational intervention, or manual case handling.
- `priority=high` for account lockout/access blocking issues, payment disputes, and urgent shipping problems.
- `priority=low` for informational questions without immediate risk.
- Distinguish `billing` vs `refund`:
  - `billing` = charge/payment/invoice/subscription-state issue
  - `refund` = refund request or refund status timeline

## Exemplar tickets

```json
[
  {
    "id": "x001",
    "subject": "Locked out after too many login attempts",
    "description": "I entered the wrong password too many times and now I can't access my account.",
    "label": {"category": "account_access", "priority": "high", "requires_human": false}
  },
  {
    "id": "x002",
    "subject": "Unexpected yearly renewal charge",
    "description": "I thought auto-renew was off, but I was charged for another year this morning.",
    "label": {"category": "billing", "priority": "high", "requires_human": true}
  },
  {
    "id": "x003",
    "subject": "Refund still pending",
    "description": "I got confirmation that my return was accepted, but the refund hasn't posted yet.",
    "label": {"category": "refund", "priority": "medium", "requires_human": true}
  },
  {
    "id": "x004",
    "subject": "Courier says address not found",
    "description": "Tracking says the driver couldn't locate my address and the package is on hold.",
    "label": {"category": "shipping", "priority": "high", "requires_human": true}
  },
  {
    "id": "x005",
    "subject": "Do students get discounts?",
    "description": "Just checking whether there are education discounts or promo plans.",
    "label": {"category": "general_question", "priority": "low", "requires_human": false}
  },
  {
    "id": "x006",
    "subject": "Mobile app freezes on checkout",
    "description": "Whenever I tap Pay, the app hangs and I have to force close it.",
    "label": {"category": "product_issue", "priority": "medium", "requires_human": false}
  }
]
```

## Short reminders

- Prefer consistency over creativity.
- If uncertain between nearby categories, choose the one most directly implied by user intent.
- Keep `requires_human` conservative: only set true when likely manual intervention is needed.
