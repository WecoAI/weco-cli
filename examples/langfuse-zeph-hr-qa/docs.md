# ZephHR Product Documentation

## 1. Employee Onboarding

### Standard Onboarding Process

New employees must complete onboarding within **14 calendar days** of their start date. The onboarding checklist includes:

1. Personal information verification
2. Tax form submission (W-4 for US employees, equivalent local forms for international)
3. Benefits enrollment
4. Direct deposit setup
5. Policy acknowledgment signatures
6. Manager introduction meeting

Onboarding is **self-service** through the employee portal. Managers can track completion status but **cannot** fill out forms on behalf of the employee.

### Contractor Onboarding

Independent contractors follow a **separate onboarding flow** and are **not eligible** for benefits enrollment or direct deposit through ZephHR. Contractors must submit a W-9 (US) or W-8BEN (international) instead of a W-4. Contractor onboarding must be completed within **7 calendar days**.

### Re-hire Onboarding

Employees rehired within **12 months** of their termination date may use a streamlined re-hire flow that preserves their previous tax elections. Employees rehired **after 12 months** must complete full onboarding as a new hire.

---

## 2. Payroll

### Pay Schedule

- **Salaried employees**: Paid semi-monthly on the **1st and 15th** of each month.
- **Hourly employees**: Paid **bi-weekly** on Fridays.
- If a payday falls on a weekend or public holiday, payment is issued on the **preceding business day**.

### Payroll Cutoff

Timesheet submissions for hourly employees must be approved by the employee's **direct manager** by **5:00 PM local time on Monday** of the pay week. Late submissions are processed in the **next pay cycle** — no exceptions.

### Off-Cycle Payments

Off-cycle payments (bonuses, corrections, termination payouts) may be requested by an **HR admin** and require **VP-level approval**. Off-cycle runs are processed within **3 business days** of approval.

### Payroll Tax Jurisdictions

ZephHR automatically calculates federal and state/provincial taxes for **US and Canadian** employees. For employees in other jurisdictions, payroll tax calculations must be configured manually by an HR admin with the **Payroll Configuration** permission.

---

## 3. Benefits

### Eligibility

Full-time employees (30+ hours/week) are eligible for benefits starting on the **first day of the month following 60 days of employment**. Part-time employees (20–29 hours/week) are eligible for **dental and vision only**.

Employees who drop below 30 hours/week for **3 consecutive months** are reclassified as part-time and lose medical benefits at the **end of the third month**. They retain dental and vision.

### Open Enrollment

Open enrollment runs annually from **November 1–15**. Changes made during open enrollment take effect **January 1** of the following year.

Outside of open enrollment, benefits changes are permitted **only** for qualifying life events (marriage, birth/adoption, loss of other coverage, divorce). Employees must submit a qualifying life event request within **30 days** of the event.

### Benefits Edit Permissions

- **Employees** can view their own elections and update dependents during open enrollment or a qualifying life event.
- **Managers** have **read-only** access to their direct reports' enrollment status — they **cannot** edit benefits elections.
- **HR admins** can edit any employee's elections at any time but must attach a written justification.
- **Brokers** (external) can view plan summaries but **cannot** access individual employee elections.

### COBRA

Terminated employees (voluntary or involuntary) are offered COBRA continuation within **14 days** of their termination date. COBRA coverage extends for up to **18 months**. COBRA administration is handled by the third-party vendor HealthBridge, not directly through ZephHR.

---

## 4. Time Off

### PTO Accrual

- **0–2 years tenure**: 15 days/year (accrues at 1.25 days/month)
- **3–5 years tenure**: 20 days/year (accrues at 1.67 days/month)
- **6+ years tenure**: 25 days/year (accrues at 2.08 days/month)

PTO accrual is capped at **1.5x the annual entitlement**. Once the cap is reached, accrual pauses until the balance drops below the cap. Unused PTO **does not** pay out at termination unless required by state/provincial law.

### Request and Approval

- Requests of **1–3 days** require approval from the employee's **direct manager**.
- Requests of **4+ days** require approval from **both** the direct manager **and** the department head.
- Requests must be submitted at least **5 business days** in advance for planned absences. Sick leave is exempt from this notice requirement.

### Blackout Periods

Departments may declare **blackout periods** (max 4 weeks/year) during which PTO requests are automatically denied. Blackout periods must be announced at least **30 days** in advance. Exceptions require **VP-level** approval.

### Sick Leave

Sick leave is a **separate bank** from PTO. Full-time employees receive **10 sick days/year**, which reset on January 1 and **do not roll over**. Part-time employees receive **5 sick days/year**. Sick leave beyond 3 consecutive days requires a doctor's note uploaded through the portal.

---

## 5. Permissions and Roles

### Role Hierarchy

| Role | Scope |
|------|-------|
| **Employee** | View/edit own profile, submit time, request PTO, view own pay stubs |
| **Manager** | Everything above + approve direct reports' time/PTO, view (not edit) reports' benefits status, run team reports |
| **HR Admin** | Everything above + edit any employee record, run payroll, configure benefits, manage onboarding, create off-cycle payments |
| **System Admin** | Everything above + manage integrations, configure SSO/SAML, edit role permissions, access audit logs |

### Permission Boundaries

- Managers **cannot** view or edit compensation details for their reports — only HR admins can.
- HR admins **cannot** modify their own compensation or benefits elections — another HR admin must make the change.
- System admins can grant roles but **cannot** grant the System Admin role to themselves — this requires another System Admin.
- All permission changes are recorded in the **audit log** and retained for **7 years**.

---

## 6. Support & SLAs

### Support Tiers

| Priority | Response Time | Resolution Target | Examples |
|----------|--------------|-------------------|----------|
| **P1 – Critical** | 1 hour | 4 hours | Payroll not processing, entire system down, data breach |
| **P2 – High** | 4 hours | 1 business day | Benefits enrollment errors, SSO failures, report generation broken |
| **P3 – Medium** | 1 business day | 3 business days | UI bugs, non-blocking feature issues, import errors |
| **P4 – Low** | 2 business days | 5 business days | Feature requests, cosmetic issues, documentation questions |

### Escalation Path

1. **Tier 1**: Customer support agent (initial response)
2. **Tier 2**: Product specialist (domain-specific issues)
3. **Tier 3**: Engineering team (bugs, infrastructure)
4. **Executive escalation**: VP of Customer Success (requested by customer or after SLA breach)

Customers may request escalation at any time, but the support team determines appropriate tier assignment based on issue complexity.

### Support Hours

Standard support is available **Monday–Friday, 6 AM–8 PM Pacific Time**. P1 critical issues have **24/7 on-call coverage**. Weekend and holiday support for P2–P4 is available only on the **Enterprise** plan.

---

## 7. Pricing & Plans

### Plan Tiers

| Feature | Starter | Professional | Enterprise |
|---------|---------|-------------|------------|
| **Price** | $6/employee/month | $12/employee/month | Custom pricing |
| **Minimum seats** | 10 | 10 | 50 |
| **Payroll** | US only | US + Canada | Global |
| **Benefits admin** | Not included | Included | Included |
| **Time tracking** | Basic (clock in/out) | Advanced (projects, tasks) | Advanced + GPS |
| **Custom reports** | 5 saved reports | Unlimited | Unlimited + API access |
| **Integrations** | Slack, Email | + Accounting (QuickBooks, Xero) | + Custom API, HRIS sync |
| **SSO/SAML** | Not available | Available (add-on $2/user/mo) | Included |
| **Support** | P3/P4 email only | P2–P4, business hours | P1–P4, 24/7 for P1 |
| **Dedicated CSM** | No | No | Yes |
| **Audit log retention** | 1 year | 3 years | 7 years |

### Billing

All plans are billed **annually**. Monthly billing is available at a **20% surcharge**. Seat counts are reconciled **quarterly** — if headcount increases mid-quarter, the prorated difference is invoiced at the next reconciliation.

### Contract Terms

- **Starter and Professional**: 1-year minimum commitment, auto-renews unless canceled 30 days before renewal.
- **Enterprise**: Custom term length (typically 2–3 years), includes an annual business review.
- Downgrading from Professional to Starter mid-contract forfeits access to Professional features immediately — **no prorated refund**.

---

## 8. Product Features

### Payroll Module

Automates pay calculations, tax withholdings, and direct deposits. Supports multi-state (US) and multi-province (Canada) tax compliance. Generates year-end tax forms (W-2, T4) automatically. Payroll preview reports are available **2 business days** before each pay date.

### Onboarding Module

Configurable onboarding checklists with automated task assignment, document collection (e-signatures via DocuBridge integration), and new-hire provisioning workflows. Supports **conditional tasks** — e.g., tasks triggered only for remote employees or specific departments.

### Benefits Management Module

Carrier connectivity for medical, dental, vision, life, and disability plans. Automated eligibility tracking and ACA compliance reporting. Supports **multi-carrier** configurations. EDI feeds to carriers are sent **nightly** at 11 PM Pacific.

### Time & Attendance Module

Clock in/out via web, mobile, or kiosk mode. Geofencing available on **Enterprise** plan only. Overtime calculations follow **FLSA rules** by default; custom overtime rules can be configured by HR admins. Timesheet approval workflows support up to **3 levels** of approval.

### Reporting & Analytics

Pre-built reports for headcount, turnover, compensation analysis, and benefits utilization. Custom report builder available on **Professional and Enterprise** plans. Reports can be **scheduled** for automatic delivery via email (daily, weekly, monthly). Data export in CSV and PDF formats.

### Integrations

- **Starter**: Slack notifications, email alerts
- **Professional**: + QuickBooks, Xero (accounting sync)
- **Enterprise**: + REST API access, custom webhooks, HRIS data sync with Workday/SAP SuccessFactors

API rate limits: **100 requests/minute** for Professional, **500 requests/minute** for Enterprise. Starter plan has **no API access**.
