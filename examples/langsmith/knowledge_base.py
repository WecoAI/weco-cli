"""Knowledge base for the Acme Cloud reply-drafting benchmark."""

SUPPORT_DOCS = [
    {
        "doc_id": "double_charge",
        "title": "Duplicate charges",
        "keywords": ["charged twice", "duplicate charge", "refund", "billing"],
        "body": (
            "Acme Cloud Billing Policy: If a customer reports a duplicate charge, "
            "verify the charge in the billing dashboard. Duplicate charges are "
            "automatically refunded within 3-5 business days once confirmed. If the "
            "refund has not appeared after 5 business days, escalate to the billing "
            "team at billing@acmecloud.com. Do not ask the customer to contact their "
            "bank first."
        ),
    },
    {
        "doc_id": "plan_upgrade",
        "title": "Upgrade plans",
        "keywords": ["upgrade", "starter", "pro", "billing"],
        "body": (
            "Acme Cloud Plans: Starter ($29/mo, 10K API calls, 2 team members), "
            "Pro ($99/mo, 100K API calls, 10 team members, priority support), "
            "Enterprise (custom pricing, unlimited calls, SSO, dedicated account manager). "
            "Upgrades take effect immediately and are prorated for the current billing "
            "cycle. Downgrades take effect at the next billing cycle."
        ),
    },
    {
        "doc_id": "cancellation_charge",
        "title": "Cancellation charges",
        "keywords": ["cancelled", "cancellation", "charged again", "refund"],
        "body": (
            "Acme Cloud Cancellation Policy: Cancellations are effective immediately. "
            "Any pending charges at the time of cancellation are for usage in the "
            "current billing period and are not refundable. If a charge appears after "
            "cancellation, it was likely queued before the cancellation was processed. "
            "The customer can request a courtesy refund for post-cancellation charges "
            "by emailing billing@acmecloud.com."
        ),
    },
    {
        "doc_id": "payment_methods",
        "title": "Payment methods",
        "keywords": ["payment", "visa", "mastercard", "amex", "invoice"],
        "body": (
            "Acme Cloud accepts Visa, Mastercard, American Express, and ACH bank "
            "transfers (US only). Customers can add or update payment methods in "
            "Settings > Billing > Payment Methods. Corporate invoicing is available "
            "on the Enterprise plan with Net-30 terms."
        ),
    },
    {
        "doc_id": "proration_refund",
        "title": "Refund policy",
        "keywords": ["refund", "downgrade", "annual", "monthly", "credit"],
        "body": (
            "Acme Cloud Refund Policy: Refunds are available for annual plans within "
            "14 days of purchase. Monthly plans are not refundable but customers can "
            "cancel anytime to stop future charges. Prorated refunds for mid-cycle "
            "downgrades are issued as account credit, not cash refunds."
        ),
    },
    {
        "doc_id": "rate_limiting",
        "title": "429 rate limits",
        "keywords": ["429", "rate limit", "retry-after", "usage"],
        "body": (
            "Acme Cloud Rate Limits: Starter plan: 100 requests/minute. Pro plan: "
            "1,000 requests/minute. Enterprise: custom limits. When rate limited, the "
            "API returns HTTP 429 with a Retry-After header. Customers can monitor "
            "usage in the dashboard under Usage > API Calls. Rate limit increases "
            "require a plan upgrade or contacting sales for Enterprise customers."
        ),
    },
    {
        "doc_id": "api_key_rotation",
        "title": "API key rotation",
        "keywords": ["api key", "rotate", "downtime", "project"],
        "body": (
            "Acme Cloud API Key Management: Customers can create up to 5 API keys "
            "per project. To rotate without downtime: 1) Create a new key in "
            "Settings > API Keys, 2) Update your application to use the new key, "
            "3) Verify the new key works, 4) Revoke the old key. Old keys remain "
            "active until explicitly revoked. There is no automatic expiration."
        ),
    },
    {
        "doc_id": "webhook_troubleshooting",
        "title": "Webhook troubleshooting",
        "keywords": ["webhook", "delivery log", "ssl", "timeout"],
        "body": (
            "Acme Cloud Webhooks: Webhook deliveries are retried 3 times with "
            "exponential backoff (1min, 5min, 30min). Failed deliveries are logged "
            "in Settings > Webhooks > Delivery Log. Common causes of failure: "
            "endpoint returned non-2xx status, endpoint timed out (>30s), SSL "
            "certificate error. Webhook signing secrets can be regenerated in "
            "Settings > Webhooks without changing the endpoint URL."
        ),
    },
    {
        "doc_id": "cors_errors",
        "title": "CORS setup",
        "keywords": ["cors", "frontend", "browser", "origin"],
        "body": (
            "Acme Cloud CORS Configuration: By default, the API does not allow "
            "browser-based requests. To enable CORS: go to Settings > API > CORS "
            "Origins and add your domain (e.g., https://app.example.com). Wildcard "
            "(*) is supported for development but not recommended for production. "
            "Changes take effect within 60 seconds. If issues persist, check that "
            "the request includes the correct Origin header."
        ),
    },
    {
        "doc_id": "rate_limit_increase",
        "title": "Rate limit upgrades",
        "keywords": ["increase rate limit", "burst", "launch", "upgrade"],
        "body": (
            "Acme Cloud Rate Limit Upgrades: Starter to Pro upgrade increases the "
            "limit from 100 to 1,000 req/min immediately. Enterprise customers can "
            "request custom limits by contacting their account manager. Temporary "
            "burst allowances (up to 2x the plan limit for 5 minutes) are available "
            "on Pro and Enterprise plans. Enable burst mode in Settings > API > "
            "Rate Limiting."
        ),
    },
    {
        "doc_id": "team_member_access",
        "title": "Team member roles",
        "keywords": ["viewer", "team", "invite", "read-only"],
        "body": (
            "Acme Cloud Team Management: Roles available: Owner (full access), "
            "Admin (manage members, billing, settings), Developer (API keys, "
            "projects, read logs), Viewer (read-only dashboard access). Invite "
            "members in Settings > Team > Invite. New members receive an email "
            "invitation valid for 7 days. Members can belong to multiple projects "
            "with different roles per project."
        ),
    },
    {
        "doc_id": "password_reset",
        "title": "Password reset",
        "keywords": ["password reset", "expired link", "spam", "sso"],
        "body": (
            "Acme Cloud Password Reset: Users can reset their password at "
            "https://app.acmecloud.com/reset-password. Reset emails are sent within "
            "1 minute. If the email doesn't arrive: check spam/junk folders, verify "
            "the email address matches the account, try requesting again after 5 "
            "minutes. If the issue persists, contact support with the account email "
            "for manual verification. SSO users do not have an Acme Cloud password "
            "and should reset through their identity provider."
        ),
    },
    {
        "doc_id": "ownership_transfer",
        "title": "Ownership transfer",
        "keywords": ["ownership", "organization", "admin", "transfer"],
        "body": (
            "Acme Cloud Organization Transfer: Organization ownership can be "
            "transferred to any existing Admin member. Go to Settings > Organization "
            "> Transfer Ownership. The new owner must accept the transfer within 48 "
            "hours. The previous owner is downgraded to Admin. Transfer requires "
            "re-authentication and cannot be undone without contacting support."
        ),
    },
    {
        "doc_id": "sso_setup",
        "title": "SSO setup",
        "keywords": ["sso", "saml", "oidc", "enterprise"],
        "body": (
            "Acme Cloud SSO: SSO is available on the Enterprise plan. Supported "
            "protocols: SAML 2.0 and OIDC. Setup: Settings > Security > SSO. "
            "Required information: IdP metadata URL or XML, attribute mapping for "
            "email and name. Once enabled, all team members must sign in through the "
            "IdP. Existing password-based sessions are invalidated within 24 hours. "
            "SSO can be enforced (no password fallback) or optional."
        ),
    },
    {
        "doc_id": "account_deletion",
        "title": "Account deletion",
        "keywords": ["delete account", "data deletion", "export", "cancel"],
        "body": (
            "Acme Cloud Account Deletion: Users can request account deletion in "
            "Settings > Account > Delete Account. Deletion is permanent and removes "
            "all data including projects, API keys, and usage history within 30 days. "
            "A confirmation email is sent with a 72-hour cancellation window. Active "
            "subscriptions must be cancelled before deletion. Data export (JSON or "
            "CSV) is available before deletion in Settings > Account > Export Data."
        ),
    },
    {
        "doc_id": "batch_processing",
        "title": "Batch API",
        "keywords": ["batch", "asynchronous", "webhook", "polling"],
        "body": (
            "Acme Cloud Batch API: The /v1/batch endpoint accepts up to 1,000 items "
            "per request. Each item is processed asynchronously and results are "
            "returned via webhook or polling. Batch jobs are queued and typically "
            "complete within 5 minutes for standard workloads. Monitor batch status "
            "at /v1/batch/{batch_id}/status. Failed items are retried once "
            "automatically. Available on Pro and Enterprise plans only."
        ),
    },
    {
        "doc_id": "usage_alerts",
        "title": "Usage alerts",
        "keywords": ["usage alerts", "threshold", "billing", "admins"],
        "body": (
            "Acme Cloud Usage Alerts: Configure alerts in Settings > Billing > "
            "Alerts. Available thresholds: 50%, 75%, 90%, and 100% of plan limit. "
            "Alerts are sent via email to all Admins and the billing contact. "
            "Webhook notifications are available on Enterprise plans. Usage resets "
            "on the 1st of each month at 00:00 UTC."
        ),
    },
    {
        "doc_id": "data_export",
        "title": "Data export",
        "keywords": ["export", "csv", "json", "12 months"],
        "body": (
            "Acme Cloud Data Export: Export data in Settings > Account > Export Data. "
            "Supported formats: CSV and JSON. Exports include: usage logs, API call "
            "history, team member list, and billing history. Large exports (>1GB) are "
            "delivered as a download link sent via email. Exports are available for "
            "the last 12 months of data. Real-time streaming export is available on "
            "Enterprise plans via the /v1/export endpoint."
        ),
    },
    {
        "doc_id": "workspaces_vs_projects",
        "title": "Workspaces vs projects",
        "keywords": ["workspace", "project", "billing", "permissions"],
        "body": (
            "Acme Cloud Workspaces and Projects: A Workspace is a top-level "
            "container tied to a billing account. Each workspace can contain multiple "
            "Projects. Projects isolate API keys, usage tracking, and team "
            "permissions. Example: a company (workspace) with separate projects for "
            "'Production', 'Staging', and 'Internal Tools'. Workspace settings "
            "(billing, SSO) apply to all projects. Project settings (API keys, "
            "webhooks) are project-specific."
        ),
    },
    {
        "doc_id": "on_premise",
        "title": "Deployment options",
        "keywords": ["on-prem", "self-hosted", "vpc", "byok", "region"],
        "body": (
            "Acme Cloud Deployment Options: Acme Cloud is a cloud-hosted SaaS "
            "product. On-premise deployment is not available. For customers with data "
            "residency requirements, we offer: 1) Region selection (US, EU, APAC) at "
            "the workspace level, 2) A dedicated VPC option on the Enterprise plan "
            "where your data is isolated in a single-tenant environment, 3) BYOK "
            "(Bring Your Own Key) encryption for data at rest. Contact "
            "sales@acmecloud.com for dedicated VPC pricing."
        ),
    },
]

DOCS_BY_ID = {doc["doc_id"]: doc for doc in SUPPORT_DOCS}
ARTICLES = {doc["doc_id"]: doc["body"] for doc in SUPPORT_DOCS}
