"""Deterministic text-shape features from EmailEvidence (Phase C).

All counts are simple case-insensitive keyword/character counts over
subject + first plain-text body part. They are pattern signals for the
technical model — NOT confirmed social-engineering indicators, and never
raw email content (only numbers cross the feature boundary).
"""

from __future__ import annotations

from app.schemas.email import EmailEvidence
from app.schemas.features import TextFeatures

_URGENCY_KEYWORDS = (
    "urgent",
    "immediately",
    "act now",
    "final notice",
    "asap",
    "time sensitive",
    "expires today",
    "last warning",
    "action required",
)

_MONEY_KEYWORDS = (
    "invoice",
    "payment",
    "wire transfer",
    "bank details",
    "payment overdue",
    "remittance",
    "amount due",
    "account number",
)

_CREDENTIAL_KEYWORDS = (
    "password",
    "verify your account",
    "confirm your identity",
    "login",
    "sign in",
    "credentials",
    "reset your password",
    "security alert",
    "unusual sign",
)


def _keyword_count(haystack: str, keywords: tuple[str, ...]) -> int:
    """Case-insensitive total occurrences of the keywords in haystack."""

    if not haystack:
        return 0
    lowered = haystack.casefold()
    return sum(lowered.count(keyword) for keyword in keywords)


def extract_text_features(evidence: EmailEvidence) -> TextFeatures:
    """Extract text features; never raises, never touches the network."""

    subject = evidence.message.subject or ""
    body_plain = evidence.body.plain_text or ""
    html = evidence.body.html or ""

    combined_text = f"{subject}\n{body_plain}" if subject or body_plain else ""

    return TextFeatures(
        subject_length=len(subject),
        subject_exclamation_count=subject.count("!"),
        subject_urgency_signal=(
            1 if any(k in subject.casefold() for k in _URGENCY_KEYWORDS) else 0
        ),
        subject_money_signal=(
            1 if any(k in subject.casefold() for k in _MONEY_KEYWORDS) else 0
        ),
        body_plain_length=len(body_plain),
        body_html_present=1 if html else 0,
        body_html_length=len(html),
        credential_signal_count=_keyword_count(combined_text, _CREDENTIAL_KEYWORDS),
        payment_signal_count=_keyword_count(combined_text, _MONEY_KEYWORDS),
        urgency_signal_count=_keyword_count(combined_text, _URGENCY_KEYWORDS),
    )


__all__ = ["extract_text_features"]
