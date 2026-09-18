"""Deterministic header-shape features from EmailEvidence (Phase C).

Mismatch flags are set to 1 only when BOTH sides of the comparison are
determinable from the evidence; when either side is missing the flag stays 0
and the accompanying presence/absence features carry that information.
Domain parsing reuses the Step 4 normalization utilities.
"""

from __future__ import annotations

from app.intelligence.indicator_extractor import normalize_domain
from app.schemas.email import EmailEvidence
from app.schemas.features import HeaderFeatures


def _mailbox_domain(address: str | None) -> str | None:
    """Normalized domain of a mailbox string, or None when not determinable."""

    if not address or "@" not in address:
        return None
    _, _, domain = address.rpartition("@")
    return normalize_domain(domain)


def _message_id_domain(message_id: str | None) -> str | None:
    """Domain between the angle brackets of a Message-ID, or None."""

    if not message_id:
        return None
    stripped = message_id.strip().strip("<>").strip()
    if "@" not in stripped:
        return None
    _, _, domain = stripped.rpartition("@")
    return normalize_domain(domain)


def _return_path_domain(return_path: str | None) -> str | None:
    """Domain of the Return-Path value, or None."""

    if not return_path:
        return None
    return _mailbox_domain(return_path.strip().strip("<>").strip())


def extract_header_features(evidence: EmailEvidence) -> HeaderFeatures:
    """Extract header features; never raises, never touches the network."""

    sender_domain = _mailbox_domain(evidence.sender.address if evidence.sender else None)

    reply_to_addresses = [a.address for a in evidence.message.reply_to if a.address]
    reply_to_domain = _mailbox_domain(reply_to_addresses[0]) if reply_to_addresses else None

    return_path_domain = _return_path_domain(evidence.message.return_path)
    message_id_domain = _message_id_domain(evidence.message.message_id)

    sender_missing = sender_domain is None

    def _mismatch(other: str | None) -> int:
        if sender_missing or other is None:
            return 0
        return 1 if other != sender_domain else 0

    return HeaderFeatures(
        received_hop_count=len(evidence.received_chain or evidence.headers.received),
        parser_defect_count=len(evidence.parser_defects),
        sender_reply_to_domain_mismatch=_mismatch(reply_to_domain),
        return_path_sender_mismatch=_mismatch(return_path_domain),
        message_id_domain_mismatch=_mismatch(message_id_domain),
        message_id_missing=1 if not evidence.message.message_id else 0,
        date_missing=1 if not evidence.message.date else 0,
        reply_to_present=1 if reply_to_addresses else 0,
        user_agent_present=1 if (evidence.headers.user_agent or evidence.headers.x_mailer) else 0,
        mime_multipart=1 if evidence.mime.is_multipart else 0,
    )


__all__ = ["extract_header_features"]
