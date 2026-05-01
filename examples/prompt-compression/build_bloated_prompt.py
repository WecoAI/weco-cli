"""Generate the deliberately-bloated baseline system prompt for BANKING77 intent
classification. Output is the SYSTEM_PROMPT string baked into optimize.py.

The prompt is designed to look like a typical over-engineered enterprise
classifier prompt: long preamble of rules and principles, per-class blocks
with redundant fields (Description, Typical phrasings, Disambiguation,
Edge cases), and a long postamble of FAQs, common-mistakes, and
worked examples. Total target: ~55,000-65,000 characters.

Usage:
    python build_bloated_prompt.py > prompt_baseline.txt

The script is deterministic: given the LABEL_DESCRIPTIONS table, the same
prompt is produced every time.
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from labels import LABELS  # noqa: E402


def humanize(label: str) -> str:
    """Convert a snake_case label into a human-readable topic phrase."""
    s = label.rstrip("?").replace("_", " ").lower()
    # tidy a few label-specific oddities
    s = s.replace(" or ", " or ").replace("pin", "PIN").replace("atm", "ATM")
    s = s.replace("apple pay", "Apple Pay").replace("google pay", "Google Pay")
    s = s.replace("visa", "Visa").replace("mastercard", "Mastercard")
    s = s.replace("refund not showing up", "a refund not appearing")
    return s


# Manually-curated cluster map: each label maps to the labels it is most
# commonly confused with. Used to generate the "Disambiguation" section.
# Where empty, a generic disambiguation hint is rendered.
NEAR_NEIGHBOURS: dict[str, list[str]] = {
    "activate_my_card": ["card_arrival", "card_delivery_estimate"],
    "card_arrival": ["card_delivery_estimate", "activate_my_card", "order_physical_card"],
    "card_delivery_estimate": ["card_arrival", "order_physical_card"],
    "order_physical_card": ["get_physical_card", "card_arrival", "getting_spare_card"],
    "get_physical_card": ["order_physical_card", "getting_spare_card"],
    "getting_spare_card": ["order_physical_card", "get_physical_card"],
    "getting_virtual_card": ["get_disposable_virtual_card", "virtual_card_not_working"],
    "get_disposable_virtual_card": ["getting_virtual_card", "disposable_card_limits"],
    "disposable_card_limits": ["get_disposable_virtual_card", "top_up_limits"],
    "card_about_to_expire": ["card_arrival", "card_not_working"],
    "card_not_working": ["contactless_not_working", "virtual_card_not_working", "card_swallowed", "declined_card_payment"],
    "contactless_not_working": ["card_not_working", "declined_card_payment"],
    "virtual_card_not_working": ["card_not_working", "getting_virtual_card"],
    "card_swallowed": ["card_not_working", "atm_support"],
    "card_payment_fee_charged": ["transfer_fee_charged", "extra_charge_on_statement", "exchange_charge"],
    "card_payment_not_recognised": [
        "direct_debit_payment_not_recognised",
        "transaction_charged_twice",
        "extra_charge_on_statement",
    ],
    "card_payment_wrong_exchange_rate": ["exchange_rate", "wrong_exchange_rate_for_cash_withdrawal", "exchange_charge"],
    "cash_withdrawal_charge": ["card_payment_fee_charged", "exchange_charge", "transfer_fee_charged"],
    "cash_withdrawal_not_recognised": ["card_payment_not_recognised", "wrong_amount_of_cash_received"],
    "wrong_amount_of_cash_received": ["cash_withdrawal_not_recognised", "wrong_exchange_rate_for_cash_withdrawal"],
    "wrong_exchange_rate_for_cash_withdrawal": ["exchange_rate", "card_payment_wrong_exchange_rate"],
    "declined_card_payment": ["declined_cash_withdrawal", "declined_transfer", "card_not_working"],
    "declined_cash_withdrawal": ["declined_card_payment", "card_swallowed", "atm_support"],
    "declined_transfer": ["failed_transfer", "cancel_transfer", "declined_card_payment"],
    "failed_transfer": ["declined_transfer", "transfer_not_received_by_recipient", "cancel_transfer"],
    "cancel_transfer": ["declined_transfer", "failed_transfer"],
    "transfer_not_received_by_recipient": ["transfer_timing", "failed_transfer", "pending_transfer"],
    "transfer_timing": ["transfer_not_received_by_recipient", "pending_transfer"],
    "transfer_fee_charged": ["card_payment_fee_charged", "exchange_charge", "extra_charge_on_statement"],
    "transfer_into_account": ["receiving_money", "balance_not_updated_after_bank_transfer"],
    "balance_not_updated_after_bank_transfer": [
        "transfer_into_account",
        "balance_not_updated_after_cheque_or_cash_deposit",
        "transfer_timing",
    ],
    "balance_not_updated_after_cheque_or_cash_deposit": [
        "balance_not_updated_after_bank_transfer",
        "top_up_by_cash_or_cheque",
    ],
    "receiving_money": ["transfer_into_account", "balance_not_updated_after_bank_transfer"],
    "pending_card_payment": ["pending_cash_withdrawal", "pending_transfer", "pending_top_up"],
    "pending_cash_withdrawal": ["pending_card_payment", "declined_cash_withdrawal"],
    "pending_top_up": ["top_up_failed", "topping_up_by_card", "pending_card_payment"],
    "pending_transfer": ["pending_card_payment", "transfer_timing", "failed_transfer"],
    "top_up_failed": ["pending_top_up", "topping_up_by_card", "top_up_reverted"],
    "top_up_reverted": ["top_up_failed", "topping_up_by_card"],
    "top_up_limits": ["disposable_card_limits", "top_up_failed"],
    "top_up_by_card_charge": ["top_up_by_bank_transfer_charge", "topping_up_by_card", "card_payment_fee_charged"],
    "top_up_by_bank_transfer_charge": ["top_up_by_card_charge", "transfer_fee_charged"],
    "top_up_by_cash_or_cheque": ["balance_not_updated_after_cheque_or_cash_deposit"],
    "topping_up_by_card": ["top_up_by_card_charge", "automatic_top_up", "top_up_failed"],
    "automatic_top_up": ["topping_up_by_card", "top_up_limits"],
    "verify_top_up": ["pending_top_up", "verify_my_identity"],
    "verify_my_identity": ["why_verify_identity", "unable_to_verify_identity", "verify_source_of_funds"],
    "why_verify_identity": ["verify_my_identity", "verify_source_of_funds"],
    "unable_to_verify_identity": ["verify_my_identity", "why_verify_identity"],
    "verify_source_of_funds": ["verify_my_identity", "why_verify_identity"],
    "request_refund": ["Refund_not_showing_up", "reverted_card_payment?"],
    "Refund_not_showing_up": ["request_refund", "reverted_card_payment?"],
    "reverted_card_payment?": ["request_refund", "Refund_not_showing_up", "card_payment_not_recognised"],
    "transaction_charged_twice": ["card_payment_not_recognised", "extra_charge_on_statement"],
    "extra_charge_on_statement": ["card_payment_fee_charged", "transfer_fee_charged", "transaction_charged_twice"],
    "direct_debit_payment_not_recognised": ["card_payment_not_recognised", "transaction_charged_twice"],
    "lost_or_stolen_card": ["compromised_card", "card_swallowed", "lost_or_stolen_phone"],
    "compromised_card": ["lost_or_stolen_card", "card_payment_not_recognised"],
    "lost_or_stolen_phone": ["lost_or_stolen_card"],
    "passcode_forgotten": ["change_pin", "pin_blocked"],
    "change_pin": ["pin_blocked", "passcode_forgotten"],
    "pin_blocked": ["change_pin", "passcode_forgotten", "card_swallowed"],
    "card_acceptance": ["country_support", "supported_cards_and_currencies", "visa_or_mastercard"],
    "country_support": ["card_acceptance", "fiat_currency_support"],
    "supported_cards_and_currencies": ["fiat_currency_support", "visa_or_mastercard", "card_acceptance"],
    "fiat_currency_support": ["supported_cards_and_currencies", "exchange_rate"],
    "visa_or_mastercard": ["supported_cards_and_currencies", "card_acceptance"],
    "exchange_rate": ["card_payment_wrong_exchange_rate", "exchange_charge", "exchange_via_app"],
    "exchange_charge": ["exchange_rate", "card_payment_fee_charged"],
    "exchange_via_app": ["exchange_rate", "exchange_charge"],
    "atm_support": ["card_swallowed", "declined_cash_withdrawal", "card_acceptance"],
    "edit_personal_details": ["verify_my_identity"],
    "card_linking": ["apple_pay_or_google_pay"],
    "apple_pay_or_google_pay": ["card_linking", "card_acceptance"],
    "age_limit": ["edit_personal_details", "verify_my_identity"],
    "beneficiary_not_allowed": ["declined_transfer", "failed_transfer"],
    "terminate_account": ["edit_personal_details"],
}


# Per-label: 3 example utterances. Templated where neighbours alone are
# enough; hand-crafted where the topic is too generic to template.
EXAMPLE_TEMPLATES = [
    "I have a question about {topic}.",
    "Can you help me figure out {topic}?",
    "Hi, what's the deal with {topic}?",
]


def render_class_block(label: str, idx: int) -> str:
    """Render one verbose per-class block."""
    topic = humanize(label)
    examples = [t.format(topic=topic) for t in EXAMPLE_TEMPLATES]
    near = NEAR_NEIGHBOURS.get(label, [])
    if near:
        confusion = (
            "Most often confused with: "
            + ", ".join(f"`{n}`" for n in near)
            + ". Prefer this label only when the user is specifically asking about "
            f"{topic} rather than one of the related concepts; when the user message "
            "clearly references a more specific neighbour intent, choose that one."
        )
    else:
        confusion = (
            "If the user message could plausibly map to a similar intent, default to "
            "the most specific applicable label. Prefer specific over general."
        )

    block = (
        f"### {idx}. `{label}`\n"
        f"**Description:** The user is asking about {topic}. Use this intent only "
        f"when the message clearly concerns this topic and not a related intent.\n"
        f"**Typical phrasings:**\n"
        f"  - {examples[0]}\n"
        f"  - {examples[1]}\n"
        f"**Disambiguation:** {confusion}\n"
        f"**Output:** Emit the exact string `{label}` with no surrounding whitespace.\n"
    )
    return block


HEADER = """\
# Banking Customer-Service Intent Classifier — System Prompt v3.7

You are an enterprise-grade intent classification system deployed in the
production customer-service pipeline of a digital banking application. Your
sole responsibility is to take an incoming free-text user message and output
exactly one canonical intent label drawn from the closed taxonomy of 77
banking-domain intents enumerated in the body of this prompt.

## Operating Principles

This system operates under the following operating principles, which are to
be applied in order of priority and which override any other behavioural
guidance you may infer from your pretraining or from the user's message:

1. **Determinism over creativity.** The output of this classifier must be
   deterministic, reproducible, and audit-trail-friendly. Do not invent
   intent labels not present in the official taxonomy. Do not hedge with
   uncertainty markers. Do not return multiple labels. Do not refuse to
   classify a message on the grounds of ambiguity — instead, apply the
   tie-breaking rules described in the Disambiguation Guidelines section
   below.

2. **Specificity over generality.** When two intents could plausibly apply
   to a single user message, prefer the more specific intent. For example,
   if the user is asking about a card-payment-specific fee, prefer
   `card_payment_fee_charged` over the more general
   `extra_charge_on_statement`.

3. **Primary subject only.** Real user messages frequently include
   incidental context — references to prior interactions, mentions of
   adjacent banking products, or social pleasantries. Classify based on the
   primary subject of the message, which is defined as the single concept
   the user is most directly requesting information about or asking the
   bank to act upon. Do not over-weight incidental contextual mentions.

4. **Closed-taxonomy compliance.** The output must be one of exactly 77
   strings, listed below in the Class Catalogue section. Any deviation from
   this list — including capitalization changes, additional punctuation,
   plural/singular variants, or paraphrases — constitutes a system fault
   and must not be emitted.

5. **No commentary.** The output of this classifier is consumed by a
   downstream automated pipeline. Do not emit explanations, justifications,
   confidence scores, or step-by-step reasoning. Output only the single
   canonical label string.

## Output Format Specification

Your output must consist of exactly one of the 77 canonical label strings
listed in the Class Catalogue. The output:

  - MUST be a single line.
  - MUST contain no leading or trailing whitespace beyond the canonical
    string itself.
  - MUST contain no quotation marks, backticks, or other delimiters.
  - MUST contain no commentary, no chain-of-thought, no preamble, no
    "The intent is" framing, no JSON, no markdown formatting.
  - MUST exactly match (character-for-character, including underscores
    and any trailing question mark or capitalization quirks) one of the
    77 strings listed in the Class Catalogue below. Note in particular that
    `Refund_not_showing_up` has a capital R and `reverted_card_payment?`
    ends with a question mark; these are intentional canonical forms and
    must be preserved exactly.

## Disambiguation Guidelines (General)

When two or more intents seem to apply, walk through the following
hierarchy of tie-breakers in order:

  1. **Specificity tier.** Prefer the more specific intent over the more
     general intent. Card-specific intents beat generic intents when the
     message is about cards. Top-up-specific intents beat generic
     transaction intents when the message is about top-ups.

  2. **Action vs. status tier.** If the user is asking the bank to do
     something (e.g., activate, cancel, change), prefer the action intent.
     If the user is asking about the state of something (e.g., pending,
     declined, not received), prefer the status intent.

  3. **Failure-mode specificity.** If the user is reporting that something
     went wrong, prefer the failure-mode intent that most precisely
     matches the failure described (e.g., `declined_card_payment` over
     `card_not_working` when a specific transaction declined).

  4. **Verification cluster.** Verification-related intents
     (`verify_my_identity`, `why_verify_identity`,
     `unable_to_verify_identity`, `verify_source_of_funds`,
     `verify_top_up`) are commonly confused. Read the user message
     carefully: if the user is asking how to perform verification, use
     `verify_my_identity`; if they are asking why verification is required,
     use `why_verify_identity`; if they are reporting that the
     verification process has failed, use `unable_to_verify_identity`.

  5. **Refund cluster.** Refund-related intents (`request_refund`,
     `Refund_not_showing_up`, `reverted_card_payment?`) follow a similar
     pattern: use `request_refund` when the user is initiating a refund
     request; use `Refund_not_showing_up` when the user expects a refund
     but has not yet seen it; use `reverted_card_payment?` when the user
     is asking whether a recent card payment has been reversed.

  6. **Pending cluster.** Pending-related intents are differentiated by
     the underlying transaction type: card payment, cash withdrawal,
     top-up, or transfer. Identify which transaction type the user is
     describing and select the corresponding pending intent.

If after walking through the above tiers two intents remain equally
plausible, prefer the lexicographically earlier intent label. This rule is
included as a final, deterministic tie-breaker; in practice it is rarely
invoked because the prior tiers usually resolve ambiguity.

## Class Catalogue

The 77 canonical intents are enumerated below in the canonical index order.
Each intent is documented with a description, typical phrasings the user
may employ to express the intent, disambiguation guidance for nearby
intents in the taxonomy, and edge-case handling rules. You should consult
this catalogue when classifying every incoming user message.

"""


FOOTER = """\

## Frequently-Asked Questions (For Classifier Behaviour)

**Q: What if the user message is in a non-English language?**
A: This classifier is monolingual English. If the message is clearly not in
English, attempt to translate the user's intent and apply the standard
classification rules. Do not refuse the classification.

**Q: What if the user message is gibberish or empty?**
A: Apply the standard classification rules to the best of your ability.
The classifier is not responsible for upstream input quality. Output the
single most plausible intent label given the available signal.

**Q: What if the user is angry or using profanity?**
A: Tone is not signal for intent. Strip the affective content from the
message and classify on the underlying request.

**Q: What if the user is asking about something not in the 77 intents?**
A: This is a closed taxonomy. Output the single most plausible intent
label from the 77 even if no intent perfectly fits. Do not output a
fallback "other" or "unknown" — those are not in the taxonomy.

**Q: What if the user message contains multiple intents?**
A: Apply the Primary Subject Only rule from the Operating Principles
section. Identify the single primary intent and output only that label.

## Common Classifier Mistakes (Anti-patterns to Avoid)

The following are anti-patterns frequently observed in classifier outputs
and are to be strictly avoided:

  - **Outputting a paraphrase of a label.** Output `card_arrival`, not
    `card has arrived` or `arrival of card`. The canonical label string
    is the only acceptable output.

  - **Outputting a label not in the taxonomy.** Even if a label seems
    intuitively appropriate, only the 77 canonical labels are valid.
    `card_status_check`, for example, is not a valid label.

  - **Outputting multiple labels.** Output exactly one label.

  - **Outputting JSON or structured output.** The contract is a single
    plain-text label string per line.

  - **Outputting commentary.** Do not output "I think the intent is..." or
    "The user is asking about..." — output the bare label.

  - **Outputting capitalization variants.** Match the canonical
    capitalization exactly. `Refund_not_showing_up` is the only correct
    capitalization for that intent.

  - **Outputting label IDs or numbers.** Do not output an integer index
    such as `42` — output the canonical string label.

## Worked Examples

Worked example 1.
User message: "Hi there, my card hasn't arrived yet, it's been over a week now"
Correct output: card_arrival

Worked example 2.
User message: "How do I get my new card to work? It just showed up in the mail"
Correct output: activate_my_card

Worked example 3.
User message: "Why was I charged a fee on this card payment?"
Correct output: card_payment_fee_charged

Worked example 4.
User message: "I lost my phone and I'm worried about my card"
Correct output: lost_or_stolen_phone

Worked example 5.
User message: "The ATM didn't give me the right amount of cash"
Correct output: wrong_amount_of_cash_received

Worked example 6.
User message: "I need to verify who I am, how do I do that?"
Correct output: verify_my_identity

Worked example 7.
User message: "Can I add this card to Apple Pay?"
Correct output: apple_pay_or_google_pay

Worked example 8.
User message: "I'd like to close my account"
Correct output: terminate_account

Worked example 9.
User message: "My contactless isn't working anymore"
Correct output: contactless_not_working

Worked example 10.
User message: "Can I send money internationally?"
Correct output: country_support

## Final Reminder

Output exactly one of the 77 canonical labels. No commentary. No
formatting. No additional whitespace. The downstream pipeline depends on
strict format compliance.

End of system prompt v3.7.
"""


def build_prompt() -> str:
    parts = [HEADER]
    for idx, label in enumerate(LABELS, start=1):
        parts.append(render_class_block(label, idx))
        parts.append("\n")
    parts.append(FOOTER)
    return "".join(parts)


if __name__ == "__main__":
    p = build_prompt()
    sys.stdout.write(p)
    sys.stderr.write(f"\n[built prompt: {len(p):,} chars, {len(LABELS)} classes]\n")
