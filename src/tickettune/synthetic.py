"""Deterministic, privacy-safe source generation for qualification review.

This module creates candidate data; it does not review, approve, qualify, train,
download, or deploy anything. Its output remains subject to two independent
human reviews and the separate qualification gate.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .data import validate_examples
from .prompts import SYSTEM_PROMPT
from .schemas import (
    CATEGORY_LABELS,
    PRIORITY_LABELS,
    SENTIMENT_LABELS,
    Category,
    Priority,
    Sentiment,
    TicketExample,
    TriageOutput,
)

QUALIFIED_SYNTHETIC_RECORD_COUNT = 1_120
GENERATOR_ID = "TicketTune deterministic qualified corpus generator v1"


@dataclass(frozen=True, slots=True)
class SyntheticCorpusArtifact:
    """Content identity for one exclusively created synthetic corpus."""

    path: Path
    sha256: str
    size_bytes: int
    record_count: int


@dataclass(frozen=True, slots=True)
class _Scenario:
    user_issue: str
    response_subject: str
    next_action: str
    placeholders: tuple[str, ...]


_CATEGORY_CODES: dict[Category, str] = {
    "account_access": "ACC",
    "billing": "BIL",
    "bug": "BUG",
    "cancellation": "CAN",
    "feature_request": "FEA",
    "shipping": "SHP",
    "security": "SEC",
}
_PRIORITY_CODES: dict[Priority, str] = {
    "low": "LOW",
    "medium": "MED",
    "high": "HIG",
    "urgent": "URG",
}
_SENTIMENT_CODES: dict[Sentiment, str] = {
    "positive": "POS",
    "neutral": "NEU",
    "frustrated": "FRU",
    "angry": "ANG",
    "worried": "WOR",
}

_SENTIMENT_CUES: dict[Sentiment, str] = {
    "positive": "I appreciate the support so far.",
    "neutral": "For context, I am reporting this factually.",
    "frustrated": "I am frustrated by this repeated friction.",
    "angry": "I am angry that this happened.",
    "worried": "I am worried about the impact.",
}
_PRIORITY_CUES: dict[Priority, str] = {
    "low": "This is routine, informational, and nothing is blocked.",
    "medium": "This is a non-blocking degradation, and a workaround is available.",
    "high": "This is a material blocker or time-sensitive risk that must be reviewed today.",
    "urgent": "This is an active outage, data-loss risk, or launch blocker right now.",
}

_SCENARIOS: dict[Category, tuple[_Scenario, ...]] = {
    "account_access": (
        _Scenario(
            "I cannot sign in to [ACCOUNT_ID] after completing a password reset.",
            "the post-reset sign-in issue",
            "verify_password_reset",
            ("[ACCOUNT_ID]",),
        ),
        _Scenario(
            "The verification code for [ACCOUNT_ID] is rejected before it expires.",
            "the rejected verification code",
            "review_verification_code",
            ("[ACCOUNT_ID]",),
        ),
        _Scenario(
            "Single sign-on loops back to the login screen for [ACCOUNT_ID].",
            "the single sign-on loop",
            "inspect_sso_login_loop",
            ("[ACCOUNT_ID]",),
        ),
        _Scenario(
            "The recovery message for [ACCOUNT_ID] never reaches the registered inbox.",
            "the missing recovery message",
            "review_recovery_delivery",
            ("[ACCOUNT_ID]",),
        ),
        _Scenario(
            "[ACCOUNT_ID] remains locked after the documented waiting period.",
            "the persistent account lock",
            "review_account_lock",
            ("[ACCOUNT_ID]",),
        ),
        _Scenario(
            "The active session for [ACCOUNT_ID] ends immediately after authentication.",
            "the immediate session expiry",
            "inspect_session_expiry",
            ("[ACCOUNT_ID]",),
        ),
        _Scenario(
            "The profile for [ACCOUNT_ID] unexpectedly reports that access is disabled.",
            "the disabled access status",
            "review_access_status",
            ("[ACCOUNT_ID]",),
        ),
        _Scenario(
            "A registered passkey is no longer recognized for [ACCOUNT_ID].",
            "the unrecognized passkey",
            "review_passkey_registration",
            ("[ACCOUNT_ID]",),
        ),
    ),
    "billing": (
        _Scenario(
            "[INVOICE_ID] shows the same subscription charge twice.",
            "the duplicate subscription charge",
            "review_duplicate_charge",
            ("[INVOICE_ID]",),
        ),
        _Scenario(
            "The approved refund linked to [PAYMENT_ID] has not appeared.",
            "the missing approved refund",
            "trace_approved_refund",
            ("[PAYMENT_ID]",),
        ),
        _Scenario(
            "The total on [INVOICE_ID] differs from the agreed plan price.",
            "the invoice total mismatch",
            "review_invoice_total",
            ("[INVOICE_ID]",),
        ),
        _Scenario(
            "A valid payment attempt for [ACCOUNT_ID] is repeatedly declined.",
            "the repeated payment decline",
            "review_payment_decline",
            ("[ACCOUNT_ID]",),
        ),
        _Scenario(
            "The tax line on [INVOICE_ID] does not match the billing profile.",
            "the tax-line discrepancy",
            "review_invoice_tax",
            ("[INVOICE_ID]",),
        ),
        _Scenario(
            "[ACCOUNT_ID] was billed at a different renewal price than the notice stated.",
            "the renewal-price discrepancy",
            "review_renewal_price",
            ("[ACCOUNT_ID]",),
        ),
        _Scenario(
            "The approved service credit is missing from [INVOICE_ID].",
            "the missing service credit",
            "apply_service_credit_review",
            ("[INVOICE_ID]",),
        ),
        _Scenario(
            "The receipt for [PAYMENT_ID] is unavailable in the billing portal.",
            "the unavailable payment receipt",
            "restore_payment_receipt",
            ("[PAYMENT_ID]",),
        ),
    ),
    "bug": (
        _Scenario(
            "The application closes unexpectedly on [DEVICE_ID] when I open settings.",
            "the settings crash",
            "triage_settings_crash",
            ("[DEVICE_ID]",),
        ),
        _Scenario(
            "The save control does nothing for draft [DOCUMENT_ID].",
            "the nonresponsive save control",
            "triage_save_failure",
            ("[DOCUMENT_ID]",),
        ),
        _Scenario(
            "Recent changes to [DOCUMENT_ID] do not synchronize across devices.",
            "the synchronization failure",
            "triage_sync_failure",
            ("[DOCUMENT_ID]",),
        ),
        _Scenario(
            "An upload for [FILE_ID] remains frozen before completion.",
            "the frozen upload",
            "triage_upload_freeze",
            ("[FILE_ID]",),
        ),
        _Scenario(
            "Report [REPORT_ID] calculates a total that conflicts with its visible rows.",
            "the incorrect report total",
            "triage_report_total",
            ("[REPORT_ID]",),
        ),
        _Scenario(
            "The notification preference for [ACCOUNT_ID] reverts after saving.",
            "the reverting notification preference",
            "triage_preference_revert",
            ("[ACCOUNT_ID]",),
        ),
        _Scenario(
            "Search returns a blank page for known item [ITEM_ID].",
            "the blank search result",
            "triage_blank_search",
            ("[ITEM_ID]",),
        ),
        _Scenario(
            "The exported file for [REPORT_ID] cannot be opened after download.",
            "the unreadable exported file",
            "triage_export_file",
            ("[REPORT_ID]",),
        ),
    ),
    "cancellation": (
        _Scenario(
            "I need subscription [SUBSCRIPTION_ID] canceled before its next renewal.",
            "the subscription cancellation",
            "review_subscription_cancel",
            ("[SUBSCRIPTION_ID]",),
        ),
        _Scenario(
            "Please stop order [ORDER_ID] before fulfillment begins.",
            "the order cancellation",
            "review_order_cancel",
            ("[ORDER_ID]",),
        ),
        _Scenario(
            "I want trial [SUBSCRIPTION_ID] ended without conversion to a paid plan.",
            "the trial cancellation",
            "review_trial_cancel",
            ("[SUBSCRIPTION_ID]",),
        ),
        _Scenario(
            "Please cancel scheduled service [SERVICE_ID] before the appointment window.",
            "the scheduled-service cancellation",
            "review_service_cancel",
            ("[SERVICE_ID]",),
        ),
        _Scenario(
            "Turn off automatic renewal for [SUBSCRIPTION_ID] at the end of this term.",
            "the renewal opt-out",
            "review_renewal_opt_out",
            ("[SUBSCRIPTION_ID]",),
        ),
        _Scenario(
            "I need booking [BOOKING_ID] canceled under the stated policy.",
            "the booking cancellation",
            "review_booking_cancel",
            ("[BOOKING_ID]",),
        ),
        _Scenario(
            "Remove paid add-on [ADDON_ID] from [ACCOUNT_ID] before the next bill.",
            "the paid add-on cancellation",
            "review_addon_cancel",
            ("[ADDON_ID]", "[ACCOUNT_ID]"),
        ),
        _Scenario(
            "I am requesting closure of [ACCOUNT_ID] under the account policy.",
            "the account closure request",
            "review_account_closure",
            ("[ACCOUNT_ID]",),
        ),
    ),
    "feature_request": (
        _Scenario(
            "Please add bulk export for records in [WORKSPACE_ID].",
            "the bulk-export request",
            "review_bulk_export_request",
            ("[WORKSPACE_ID]",),
        ),
        _Scenario(
            "I would like a dark appearance option for [ACCOUNT_ID].",
            "the appearance-option request",
            "review_dark_mode_request",
            ("[ACCOUNT_ID]",),
        ),
        _Scenario(
            "Please add custom team roles to [WORKSPACE_ID].",
            "the custom-role request",
            "review_custom_roles_request",
            ("[WORKSPACE_ID]",),
        ),
        _Scenario(
            "I need scheduled delivery for report [REPORT_ID].",
            "the scheduled-report request",
            "review_scheduled_report_request",
            ("[REPORT_ID]",),
        ),
        _Scenario(
            "Please add a native integration for [INTEGRATION_ID].",
            "the native-integration request",
            "review_integration_request",
            ("[INTEGRATION_ID]",),
        ),
        _Scenario(
            "I would like saved filters in [WORKSPACE_ID].",
            "the saved-filter request",
            "review_saved_filter_request",
            ("[WORKSPACE_ID]",),
        ),
        _Scenario(
            "Please add webhook events for workflow [WORKFLOW_ID].",
            "the webhook-event request",
            "review_webhook_request",
            ("[WORKFLOW_ID]",),
        ),
        _Scenario(
            "I need additional keyboard navigation in view [VIEW_ID].",
            "the keyboard-navigation request",
            "review_keyboard_access_request",
            ("[VIEW_ID]",),
        ),
    ),
    "shipping": (
        _Scenario(
            "Order [ORDER_ID] is past its delivery window.",
            "the late delivery",
            "trace_late_order",
            ("[ORDER_ID]",),
        ),
        _Scenario(
            "Tracking for [TRACKING_ID] has not updated at the last facility.",
            "the stalled tracking record",
            "trace_stalled_shipment",
            ("[TRACKING_ID]",),
        ),
        _Scenario(
            "Package [ORDER_ID] contains a different item than the packing list.",
            "the incorrect shipped item",
            "review_wrong_item",
            ("[ORDER_ID]",),
        ),
        _Scenario(
            "Item [ITEM_ID] arrived damaged in shipment [TRACKING_ID].",
            "the damaged shipment",
            "review_damaged_shipment",
            ("[ITEM_ID]", "[TRACKING_ID]"),
        ),
        _Scenario(
            "The delivery address for [ORDER_ID] needs correction before dispatch.",
            "the delivery-address correction",
            "review_address_correction",
            ("[ORDER_ID]",),
        ),
        _Scenario(
            "Only part of order [ORDER_ID] appears in the shipment record.",
            "the partial shipment",
            "trace_partial_shipment",
            ("[ORDER_ID]",),
        ),
        _Scenario(
            "[TRACKING_ID] says delivered, but the package is not at the approved location.",
            "the missing delivered package",
            "review_delivery_location",
            ("[TRACKING_ID]",),
        ),
        _Scenario(
            "Shipment [TRACKING_ID] is held for an unexplained customs requirement.",
            "the customs hold",
            "review_customs_hold",
            ("[TRACKING_ID]",),
        ),
    ),
    "security": (
        _Scenario(
            "[ACCOUNT_ID] shows a sign-in that I do not recognize.",
            "the unrecognized sign-in",
            "secure_unrecognized_login",
            ("[ACCOUNT_ID]",),
        ),
        _Scenario(
            "A message referencing [ACCOUNT_ID] asks me to disclose authentication details.",
            "the suspected phishing message",
            "review_phishing_message",
            ("[ACCOUNT_ID]",),
        ),
        _Scenario(
            "Access token [TOKEN_ID] may have been exposed in an unintended location.",
            "the potentially exposed token",
            "revoke_exposed_token",
            ("[TOKEN_ID]",),
        ),
        _Scenario(
            "A password-reset event for [ACCOUNT_ID] was not initiated by me.",
            "the unrecognized password reset",
            "secure_password_reset",
            ("[ACCOUNT_ID]",),
        ),
        _Scenario(
            "API credential [API_KEY_ID] appears in an unauthorized audit event.",
            "the unauthorized credential event",
            "revoke_api_credential",
            ("[API_KEY_ID]",),
        ),
        _Scenario(
            "Device [DEVICE_ID] with an active [ACCOUNT_ID] session is missing.",
            "the missing signed-in device",
            "secure_missing_device",
            ("[DEVICE_ID]", "[ACCOUNT_ID]"),
        ),
        _Scenario(
            "Security alert [ALERT_ID] reports suspicious software activity.",
            "the suspicious software alert",
            "review_security_alert",
            ("[ALERT_ID]",),
        ),
        _Scenario(
            "[ACCOUNT_ID] has profile changes and sessions that I did not authorize.",
            "the possible account takeover",
            "lock_and_review_account",
            ("[ACCOUNT_ID]",),
        ),
    ),
}


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _build_example(
    category: Category,
    priority: Priority,
    sentiment: Sentiment,
    scenario: _Scenario,
    variant: int,
) -> TicketExample:
    expected = TriageOutput(
        category=category,
        priority=priority,
        sentiment=sentiment,
        response=(
            f"I will review {scenario.response_subject} and route the next safe step without "
            "exposing private details."
        ),
        next_action=scenario.next_action,
    )
    return TicketExample.model_validate(
        {
            "id": (
                f"TTQ-{_CATEGORY_CODES[category]}-{_PRIORITY_CODES[priority]}-"
                f"{_SENTIMENT_CODES[sentiment]}-{variant:02d}"
            ),
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"{_SENTIMENT_CUES[sentiment]} {scenario.user_issue} "
                        f"{_PRIORITY_CUES[priority]}"
                    ),
                },
                {
                    "role": "assistant",
                    "content": _canonical_json(expected.model_dump(mode="json")),
                },
            ],
            "expected": expected.model_dump(mode="json"),
            "provenance": {
                "source": "synthetic",
                "created_by": GENERATOR_ID,
                "license": "CC0-1.0",
                "contains_real_customer_data": False,
            },
            "pii_placeholders": list(scenario.placeholders),
        },
        strict=True,
    )


def build_qualified_synthetic_examples() -> tuple[TicketExample, ...]:
    """Build the fixed 7 x 4 x 5 x 8 qualification-candidate grid."""

    examples = tuple(
        _build_example(category, priority, sentiment, scenario, variant)
        for category in CATEGORY_LABELS
        for priority in PRIORITY_LABELS
        for sentiment in SENTIMENT_LABELS
        for variant, scenario in enumerate(_SCENARIOS[category], 1)
    )
    if len(examples) != QUALIFIED_SYNTHETIC_RECORD_COUNT:
        raise RuntimeError("qualified synthetic corpus grid is incomplete")
    validate_examples(examples)
    return examples


def canonical_qualified_synthetic_bytes() -> bytes:
    """Return canonical newline-terminated JSONL for the fixed candidate grid."""

    return "".join(
        f"{_canonical_json(example.model_dump(mode='json'))}\n"
        for example in build_qualified_synthetic_examples()
    ).encode("utf-8")


def _reject_symlink_components(path: Path) -> Path:
    absolute = Path(os.path.abspath(path.expanduser()))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"synthetic corpus path must not contain symlinks: {current}")
    return absolute


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_qualified_synthetic_corpus(path: Path) -> SyntheticCorpusArtifact:
    """Exclusively create one canonical corpus without following symlinks."""

    payload = canonical_qualified_synthetic_bytes()
    destination = _reject_symlink_components(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination = _reject_symlink_components(destination)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError:
            raise FileExistsError(
                f"refusing to overwrite synthetic corpus: {destination}"
            ) from None
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return SyntheticCorpusArtifact(
        path=destination,
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        record_count=QUALIFIED_SYNTHETIC_RECORD_COUNT,
    )


__all__ = [
    "GENERATOR_ID",
    "QUALIFIED_SYNTHETIC_RECORD_COUNT",
    "SyntheticCorpusArtifact",
    "build_qualified_synthetic_examples",
    "canonical_qualified_synthetic_bytes",
    "write_qualified_synthetic_corpus",
]
