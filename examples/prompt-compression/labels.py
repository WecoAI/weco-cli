"""BANKING77 intent labels.

The PolyAI/banking77 dataset (https://huggingface.co/datasets/PolyAI/banking77)
defines 77 fine-grained banking customer-service intents. The labels are
exposed by the HF dataset's ClassLabel feature in this exact index order;
do NOT reorder.
"""

LABELS = [
    "activate_my_card",
    "age_limit",
    "apple_pay_or_google_pay",
    "atm_support",
    "automatic_top_up",
    "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit",
    "beneficiary_not_allowed",
    "cancel_transfer",
    "card_about_to_expire",
    "card_acceptance",
    "card_arrival",
    "card_delivery_estimate",
    "card_linking",
    "card_not_working",
    "card_payment_fee_charged",
    "card_payment_not_recognised",
    "card_payment_wrong_exchange_rate",
    "card_swallowed",
    "cash_withdrawal_charge",
    "cash_withdrawal_not_recognised",
    "change_pin",
    "compromised_card",
    "contactless_not_working",
    "country_support",
    "declined_card_payment",
    "declined_cash_withdrawal",
    "declined_transfer",
    "direct_debit_payment_not_recognised",
    "disposable_card_limits",
    "edit_personal_details",
    "exchange_charge",
    "exchange_rate",
    "exchange_via_app",
    "extra_charge_on_statement",
    "failed_transfer",
    "fiat_currency_support",
    "get_disposable_virtual_card",
    "get_physical_card",
    "getting_spare_card",
    "getting_virtual_card",
    "lost_or_stolen_card",
    "lost_or_stolen_phone",
    "order_physical_card",
    "passcode_forgotten",
    "pending_card_payment",
    "pending_cash_withdrawal",
    "pending_top_up",
    "pending_transfer",
    "pin_blocked",
    "receiving_money",
    "Refund_not_showing_up",
    "request_refund",
    "reverted_card_payment?",
    "supported_cards_and_currencies",
    "terminate_account",
    "top_up_by_bank_transfer_charge",
    "top_up_by_card_charge",
    "top_up_by_cash_or_cheque",
    "top_up_failed",
    "top_up_limits",
    "top_up_reverted",
    "topping_up_by_card",
    "transaction_charged_twice",
    "transfer_fee_charged",
    "transfer_into_account",
    "transfer_not_received_by_recipient",
    "transfer_timing",
    "unable_to_verify_identity",
    "verify_my_identity",
    "verify_source_of_funds",
    "verify_top_up",
    "virtual_card_not_working",
    "visa_or_mastercard",
    "why_verify_identity",
    "wrong_amount_of_cash_received",
    "wrong_exchange_rate_for_cash_withdrawal",
]

assert len(LABELS) == 77, f"Expected 77 labels, got {len(LABELS)}"
assert len(set(LABELS)) == 77, "Duplicate labels detected"


def normalize(s: str) -> str:
    """Canonicalize a string for label matching."""
    return s.strip().lower().replace(" ", "_").replace("-", "_").rstrip("?").rstrip(".")


_NORMALIZED_LABELS = {normalize(L): L for L in LABELS}


def parse_predicted_label(model_output: str) -> str | None:
    """Extract a canonical label from the model's free-text response.

    Strategy:
      1. Strict match: the entire response normalizes to a known label.
      2. First-line match: the first line normalizes to a known label.
      3. Substring scan: scan for any normalized label appearing in the text;
         return the longest match (most specific).
    Returns None if no label can be recovered.
    """
    text = (model_output or "").strip()
    if not text:
        return None

    norm_full = normalize(text)
    if norm_full in _NORMALIZED_LABELS:
        return _NORMALIZED_LABELS[norm_full]

    first_line = normalize(text.splitlines()[0])
    if first_line in _NORMALIZED_LABELS:
        return _NORMALIZED_LABELS[first_line]

    candidates = []
    norm_text = "_" + normalize(text) + "_"
    for norm, canonical in _NORMALIZED_LABELS.items():
        if "_" + norm + "_" in norm_text or norm in norm_text:
            candidates.append(canonical)
    if candidates:
        return max(candidates, key=len)

    return None


def verify_against_huggingface() -> None:
    """Sanity-check our LABELS against the live HF dataset's ClassLabel."""
    from datasets import load_dataset  # noqa: WPS433

    ds = load_dataset("PolyAI/banking77", split="test", trust_remote_code=True)
    hf_labels = ds.features["label"].names
    if hf_labels != LABELS:
        raise AssertionError(f"Label mismatch with PolyAI/banking77.\n  HF: {hf_labels}\n  ours: {LABELS}")
    print("OK: 77 labels match PolyAI/banking77 ClassLabel order.")


if __name__ == "__main__":
    verify_against_huggingface()
