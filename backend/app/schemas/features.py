"""Pydantic models for the Step 5 deterministic feature vector.

Every feature has a stable name, a fixed order (``FEATURE_NAMES``), and a
documented meaning. Features are derived ONLY from the Step 1–4 evidence
objects — feature extraction never performs network requests and never
includes raw email body text.

Sentinel convention: ``-1.0`` means "not determinable from the evidence"
(for example no RDAP registration date), and is documented per field. All
other values are counts in ``[0, inf)``, ratios in ``[0, 1]``, or booleans
(0.0/1.0).
"""

from __future__ import annotations

import typing
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class EvidenceModel(BaseModel):
    """Base model that keeps the exported evidence shape predictable."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class TextFeatures(EvidenceModel):
    """Deterministic text-shape features from EmailEvidence.

    Keyword counts are pattern signals for the model, NOT confirmed attack
    indicators.
    """

    subject_length: int = Field(default=0, ge=0, description="Character length of the Subject header (0 when missing).")
    subject_exclamation_count: int = Field(default=0, ge=0, description="Number of '!' characters in the subject.")
    subject_urgency_signal: Literal[0, 1] = 0
    subject_money_signal: Literal[0, 1] = 0
    body_plain_length: int = Field(default=0, ge=0, description="Character length of the first plain-text body part (0 when missing).")
    body_html_present: Literal[0, 1] = 0
    body_html_length: int = Field(default=0, ge=0, description="Character length of the raw HTML body part (0 when missing).")
    credential_signal_count: int = Field(default=0, ge=0, description="Count of credential-themed keywords in subject+plain body (password, verify your account, ...).")
    payment_signal_count: int = Field(default=0, ge=0, description="Count of payment-themed keywords (invoice, wire transfer, bank details, ...).")
    urgency_signal_count: int = Field(default=0, ge=0, description="Count of urgency-themed keywords (urgent, immediately, act now, ...).")


class HeaderFeatures(EvidenceModel):
    """Deterministic header-shape features from EmailEvidence."""

    received_hop_count: int = Field(default=0, ge=0, description="Number of Received headers in the parsed chain.")
    parser_defect_count: int = Field(default=0, ge=0, description="Number of parser defects reported by Step 1.")
    sender_reply_to_domain_mismatch: Literal[0, 1] = 0
    return_path_sender_mismatch: Literal[0, 1] = 0
    message_id_domain_mismatch: Literal[0, 1] = 0
    message_id_missing: Literal[0, 1] = 0
    date_missing: Literal[0, 1] = 0
    reply_to_present: Literal[0, 1] = 0
    user_agent_present: Literal[0, 1] = 0
    mime_multipart: Literal[0, 1] = 0


class AuthFeatures(EvidenceModel):
    """Categorical authentication-result encodings from AuthenticationEvidence.

    One-hot style: at most one flag per protocol is 1; all-zero means the
    result was unknown/unavailable (never guessed). Rspamd's ``score`` is a
    Step 2 email-security signal used as model input; it is not the project
    risk score.
    """

    spf_pass: Literal[0, 1] = 0
    spf_fail: Literal[0, 1] = 0
    spf_softfail: Literal[0, 1] = 0
    spf_neutral_or_none: Literal[0, 1] = 0
    spf_error_or_unknown: Literal[0, 1] = 0
    dkim_pass: Literal[0, 1] = 0
    dkim_fail: Literal[0, 1] = 0
    dkim_other: Literal[0, 1] = 0
    dmarc_pass: Literal[0, 1] = 0
    dmarc_fail: Literal[0, 1] = 0
    dmarc_other: Literal[0, 1] = 0
    rspamd_available: Literal[0, 1] = 0
    rspamd_score: float = Field(default=0.0, description="Rspamd scan score; 0.0 when the scan was unavailable.")


class UrlFeatures(EvidenceModel):
    """Deterministic URL-shape features from EmailEvidence (optionally Step 4)."""

    url_count: int = Field(default=0, ge=0)
    unique_url_host_count: int = Field(default=0, ge=0)
    https_ratio: float = Field(default=0.0, ge=0.0, le=1.0, description="Share of URLs using https (0.0 when there are no URLs).")
    ip_url_count: int = Field(default=0, ge=0, description="URLs whose host is a literal IP address (Step 4 classification when available).")
    suspicious_port_count: int = Field(default=0, ge=0, description="URLs with an explicit non-default port for their scheme.")
    max_url_path_length: int = Field(default=0, ge=0)
    unique_url_registered_domain_count: int = Field(default=0, ge=0, description="Distinct registered domains across URLs (-1-capable inputs yield 0).")
    url_registered_domain_mismatch_count: int = Field(default=0, ge=0, description="URLs whose registered domain differs from the sender domain (only counted when both sides are determinable).")


class IntelligenceFeatures(EvidenceModel):
    """Availability/shape features derived from IntelligenceEvidence.

    These describe what the Step 4 enrichment observed (or failed to
    observe); they never encode a reputation verdict.
    """

    public_ip_count: int = Field(default=0, ge=0)
    non_public_ip_count: int = Field(default=0, ge=0, description="Private/loopback/reserved/special-range IPs.")
    dns_success_count: int = Field(default=0, ge=0)
    dns_unavailable_count: int = Field(default=0, ge=0, description="Names whose DNS lookup ended in unavailable/timeout/error.")
    rdap_success_count: int = Field(default=0, ge=0)
    geoip_available_count: int = Field(default=0, ge=0)
    min_domain_age_days: float = Field(default=-1.0, description="Youngest RDAP registration age in days relative to the email Date header; -1.0 when not determinable.")


class FeatureVector(EvidenceModel):
    """The complete deterministic feature set consumed by the technical model.

    Serialization order is fixed by ``FEATURE_NAMES``; the model pipeline
    relies on that order for its column layout.
    """

    schema_version: str = "1.0"
    text: TextFeatures = Field(default_factory=TextFeatures)
    header: HeaderFeatures = Field(default_factory=HeaderFeatures)
    auth: AuthFeatures = Field(default_factory=AuthFeatures)
    url: UrlFeatures = Field(default_factory=UrlFeatures)
    intelligence: IntelligenceFeatures = Field(default_factory=IntelligenceFeatures)


def feature_names() -> tuple[str, ...]:
    """Stable, ordered feature names (dotted form ``section.field``).

    This tuple is the contract between feature extraction and the trained
    technical model. Changing order or adding/removing names invalidates
    existing model artifacts; the training pipeline records it in the
    artifact metadata.
    """

    return tuple(_iter_feature_names(FeatureVector()))


def _is_numeric_field(annotation: object) -> bool:
    """True for int/float/bool fields and integer-valued Literal fields."""

    if annotation in (int, float, bool):
        return True
    if typing.get_origin(annotation) is typing.Literal:
        return all(isinstance(arg, (int, float)) for arg in typing.get_args(annotation))
    return False


def _iter_feature_names(model: BaseModel, prefix: str = "") -> list[str]:
    """Depth-first dotted feature names in declared field order."""

    names: list[str] = []
    for field_name, field_info in type(model).model_fields.items():
        value = getattr(model, field_name)
        full_name = f"{prefix}{field_name}"
        if isinstance(value, BaseModel):
            names.extend(_iter_feature_names(value, prefix=f"{full_name}."))
        elif _is_numeric_field(field_info.annotation):
            names.append(full_name)
    return names
