"""Synthetic support reply-drafting cases for the LangSmith example.

This benchmark is intentionally small and synthetic. It is modeled on real
support workflows, but optimized for speed, reproducibility, and clear
evaluation.
"""

CASES = [
    {
        "case_id": "c001",
        "subject": "Charged twice this month",
        "message": "I think I was billed twice for the same subscription this month. What should I do?",
        "ideal_reply": (
            "I'm sorry about the duplicate charge. Please check the billing dashboard to confirm the "
            "duplicate. Once confirmed, duplicate charges are automatically refunded within 3-5 "
            "business days. If the refund has not appeared after 5 business days, contact "
            "billing@acmecloud.com so the billing team can help."
        ),
        "gold_doc_ids": ["double_charge"],
        "should_escalate": False,
    },
    {
        "case_id": "c002",
        "subject": "How do I upgrade to Pro?",
        "message": "Can I move from Starter to Pro today? What changes if I do?",
        "ideal_reply": (
            "Yes, you can upgrade from Starter to Pro at any time in Settings > Billing. The change "
            "takes effect immediately and is prorated for the current billing cycle. Pro increases "
            "your API allowance to 100K calls, supports up to 10 team members, and includes priority "
            "support."
        ),
        "gold_doc_ids": ["plan_upgrade"],
        "should_escalate": False,
    },
    {
        "case_id": "c003",
        "subject": "Charge after cancellation",
        "message": "I cancelled last week but a charge still showed up today. Can you fix this?",
        "ideal_reply": (
            "Cancellations take effect immediately, but a pending charge may still appear if it was "
            "queued before the cancellation was processed. If the charge appeared after you "
            "cancelled, you can request a courtesy refund for the post-cancellation charge by "
            "emailing billing@acmecloud.com with your account details."
        ),
        "gold_doc_ids": ["cancellation_charge"],
        "should_escalate": True,
    },
    {
        "case_id": "c004",
        "subject": "Which cards do you accept?",
        "message": "Can I pay with Amex? Also do you support invoices for larger companies?",
        "ideal_reply": (
            "Yes. We accept Visa, Mastercard, American Express, and ACH bank transfers in the US. "
            "If you're on an Enterprise plan, corporate invoicing with Net-30 terms is also "
            "available. You can update payment methods in Settings > Billing > Payment Methods."
        ),
        "gold_doc_ids": ["payment_methods"],
        "should_escalate": False,
    },
    {
        "case_id": "c005",
        "subject": "Can I get a refund mid-cycle?",
        "message": "I'm on a monthly plan and may downgrade. Do you refund the unused part as cash?",
        "ideal_reply": (
            "Monthly plans are not refundable, but you can cancel anytime to stop future charges. If "
            "you downgrade mid-cycle, the prorated difference is issued as account credit rather than "
            "a cash refund. Annual plans can be refunded within 14 days of purchase."
        ),
        "gold_doc_ids": ["proration_refund"],
        "should_escalate": False,
    },
    {
        "case_id": "c006",
        "subject": "Seeing 429 errors",
        "message": "My API calls suddenly started returning 429. Any idea why this is happening?",
        "ideal_reply": (
            "A 429 means you hit your rate limit. On Starter the limit is 100 requests per minute, "
            "and on Pro it's 1,000 requests per minute. Check the Retry-After header and review "
            "Usage > API Calls in the dashboard. If you need a higher limit, upgrading your plan can "
            "increase your allowance."
        ),
        "gold_doc_ids": ["rate_limiting"],
        "should_escalate": False,
    },
    {
        "case_id": "c007",
        "subject": "Rotate API key without downtime",
        "message": "What's the safest way to rotate my API key without breaking production traffic?",
        "ideal_reply": (
            "Create a new API key in Settings > API Keys, update your application to use the new key, "
            "verify it works, and then revoke the old key. Old keys remain active until you revoke "
            "them, so you can rotate without downtime. You can have up to 5 keys per project."
        ),
        "gold_doc_ids": ["api_key_rotation"],
        "should_escalate": False,
    },
    {
        "case_id": "c008",
        "subject": "Webhook stopped firing",
        "message": "My webhook endpoint stopped receiving events yesterday. Where should I look first?",
        "ideal_reply": (
            "Start in Settings > Webhooks > Delivery Log to see the failure details. Common causes are "
            "a non-2xx response, a timeout over 30 seconds, or an SSL certificate error. Deliveries "
            "are retried 3 times with exponential backoff at 1 minute, 5 minutes, and 30 minutes."
        ),
        "gold_doc_ids": ["webhook_troubleshooting"],
        "should_escalate": False,
    },
    {
        "case_id": "c009",
        "subject": "CORS issue in browser",
        "message": "I'm calling your API from my frontend and getting blocked by CORS. How do I fix it?",
        "ideal_reply": (
            "By default the API does not allow browser-based requests. Go to Settings > API > CORS "
            "Origins and add your domain, for example https://app.example.com. Changes take effect "
            "within 60 seconds. For development you can use a wildcard, but for production we "
            "recommend specific domains."
        ),
        "gold_doc_ids": ["cors_errors"],
        "should_escalate": False,
    },
    {
        "case_id": "c010",
        "subject": "Need higher API limit",
        "message": "How can I increase my API rate limit this week for a product launch?",
        "ideal_reply": (
            "Upgrading from Starter to Pro immediately increases the limit from 100 to 1,000 requests "
            "per minute. Pro and Enterprise plans can also enable temporary burst allowances of up to "
            "2x the plan limit for 5 minutes in Settings > API > Rate Limiting. Enterprise customers "
            "can request custom limits through their account manager."
        ),
        "gold_doc_ids": ["rate_limit_increase"],
        "should_escalate": False,
    },
    {
        "case_id": "c011",
        "subject": "Invite read-only teammate",
        "message": "I need to add a coworker so they can view the dashboard but not change anything.",
        "ideal_reply": (
            "Go to Settings > Team > Invite and enter their email address. Choose the Viewer role for "
            "read-only dashboard access. The invitation is valid for 7 days. If you need to change "
            "their permissions later, you can switch them to Developer or Admin."
        ),
        "gold_doc_ids": ["team_member_access"],
        "should_escalate": False,
    },
    {
        "case_id": "c012",
        "subject": "Password reset email missing",
        "message": "I requested a password reset but never got the email. What should I do next?",
        "ideal_reply": (
            "First check your spam or junk folder and confirm you're using the correct account email. "
            "If you already requested a reset, wait 5 minutes before trying again. If the email still "
            "doesn't arrive, contact support with the account email for manual verification. If your "
            "organization uses SSO, you need to reset through your identity provider instead."
        ),
        "gold_doc_ids": ["password_reset"],
        "should_escalate": False,
    },
    {
        "case_id": "c013",
        "subject": "Transfer org ownership",
        "message": "Can I transfer ownership of our organization to another teammate?",
        "ideal_reply": (
            "Yes. Go to Settings > Organization > Transfer Ownership. The new owner must already be an "
            "Admin, and they have 48 hours to accept. The transfer requires re-authentication, and "
            "after it is accepted you are downgraded to Admin."
        ),
        "gold_doc_ids": ["ownership_transfer"],
        "should_escalate": False,
    },
    {
        "case_id": "c014",
        "subject": "Need SSO",
        "message": "We're evaluating enterprise options. Do you support SSO and what do we need to set it up?",
        "ideal_reply": (
            "Yes. SSO is available on the Enterprise plan and supports SAML 2.0 and OIDC. Setup is in "
            "Settings > Security > SSO. You'll need your identity provider metadata URL or XML and "
            "attribute mappings for email and name."
        ),
        "gold_doc_ids": ["sso_setup"],
        "should_escalate": False,
    },
    {
        "case_id": "c015",
        "subject": "Delete my account",
        "message": "I want to permanently delete my account and all data. How does that work?",
        "ideal_reply": (
            "Go to Settings > Account > Delete Account. Before deleting, you may want to export your "
            "data in JSON or CSV from Settings > Account > Export Data. After you submit the request, "
            "you get a confirmation email with a 72-hour cancellation window. All data is permanently "
            "removed within 30 days."
        ),
        "gold_doc_ids": ["account_deletion"],
        "should_escalate": False,
    },
    {
        "case_id": "c016",
        "subject": "Does the API support batches?",
        "message": "Can I submit a large batch instead of sending requests one at a time?",
        "ideal_reply": (
            "Yes. The /v1/batch endpoint accepts up to 1,000 items per request on Pro and Enterprise "
            "plans. Jobs are processed asynchronously, results can be returned by webhook or polling, "
            "and batch status is available at /v1/batch/{batch_id}/status."
        ),
        "gold_doc_ids": ["batch_processing"],
        "should_escalate": False,
    },
    {
        "case_id": "c017",
        "subject": "Usage threshold alerts",
        "message": "Can I get alerts when my usage gets close to the plan limit?",
        "ideal_reply": (
            "Yes. Configure alerts in Settings > Billing > Alerts. Available thresholds are 50%, 75%, "
            "90%, and 100% of your plan limit. Alerts are emailed to all Admins and the billing "
            "contact, and Enterprise plans also support webhook notifications."
        ),
        "gold_doc_ids": ["usage_alerts"],
        "should_escalate": False,
    },
    {
        "case_id": "c018",
        "subject": "Need CSV export",
        "message": "Can I export our account data as CSV, and what does that include?",
        "ideal_reply": (
            "Yes. Go to Settings > Account > Export Data and choose CSV or JSON. Exports include usage "
            "logs, API call history, team members, and billing history for the last 12 months. If the "
            "export is larger than 1GB, you'll receive a download link by email."
        ),
        "gold_doc_ids": ["data_export"],
        "should_escalate": False,
    },
    {
        "case_id": "c019",
        "subject": "Workspace vs project",
        "message": "Can you explain the difference between a workspace and a project in Acme Cloud?",
        "ideal_reply": (
            "A workspace is the top-level container tied to a billing account. Projects live inside a "
            "workspace and isolate API keys, usage tracking, and team permissions. Workspace settings "
            "like billing and SSO apply across projects, while webhooks and API keys are "
            "project-specific."
        ),
        "gold_doc_ids": ["workspaces_vs_projects"],
        "should_escalate": False,
    },
    {
        "case_id": "c020",
        "subject": "Need on-prem deployment",
        "message": "We're looking for an on-prem deployment. Is that something you offer?",
        "ideal_reply": (
            "We do not offer a traditional on-prem deployment. For data residency needs, we support "
            "region selection at the workspace level and a dedicated VPC option on the Enterprise "
            "plan. We also support BYOK encryption for data at rest. Contact sales@acmecloud.com for "
            "dedicated VPC pricing."
        ),
        "gold_doc_ids": ["on_premise"],
        "should_escalate": True,
    },
]

OPTIMIZATION_CASE_IDS = [
    "c001",
    "c002",
    "c003",
    "c005",
    "c006",
    "c007",
    "c009",
    "c011",
    "c012",
    "c014",
    "c016",
    "c018",
]

VALIDATION_CASE_IDS = [
    "c004",
    "c008",
    "c010",
    "c013",
    "c015",
    "c017",
    "c019",
    "c020",
]

