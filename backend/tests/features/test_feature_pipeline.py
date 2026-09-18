"""Step 5 feature tests: determinism, edge cases, and contract stability."""

from __future__ import annotations

import pytest

from app.features.feature_fusion import build_feature_vector, feature_names
from app.schemas.email import EmailEvidence
from app.schemas.features import FeatureVector
from app.schemas.intelligence import IntelligenceEvidence

from ai_fixtures.helpers import (
    intelligence_with_domain_ages,
    make_indicator,
    zero_vector,
)


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


def test_feature_contract_is_stable_and_dotted():
    names = feature_names()
    assert len(names) == 48
    assert len(set(names)) == 48  # no duplicates
    assert all("." in name for name in names)
    # Literal[0,1] flags must be part of the contract too:
    assert names[:3] == (
        "text.subject_length",
        "text.subject_exclamation_count",
        "text.subject_urgency_signal",
    )
    assert "text.body_plain_length" in names
    assert "auth.spf_pass" in names
    assert "intelligence.min_domain_age_days" in names


def test_feature_vector_defaults_are_neutral():
    vector = zero_vector()
    assert vector.text.subject_length == 0
    assert vector.auth.spf_pass == 0
    assert vector.url.https_ratio == 0.0
    assert vector.intelligence.min_domain_age_days == -1.0


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_feature_extraction_is_deterministic(evidence):
    first = build_feature_vector(evidence)
    second = build_feature_vector(evidence)
    assert first.model_dump_json() == second.model_dump_json()


def test_feature_extraction_ignores_repeated_calls_with_intelligence(evidence):
    intelligence = IntelligenceEvidence()
    first = build_feature_vector(evidence, None, intelligence)
    second = build_feature_vector(evidence, None, intelligence)
    assert first.model_dump_json() == second.model_dump_json()


# ---------------------------------------------------------------------------
# Missing / malformed inputs
# ---------------------------------------------------------------------------


def test_empty_evidence_yields_neutral_vector():
    evidence = EmailEvidence.model_construct(
        message=__import__("app.schemas.email", fromlist=["MessageMetadata"]).MessageMetadata(),
        body=__import__("app.schemas.email", fromlist=["EmailBody"]).EmailBody(),
        headers=__import__("app.schemas.email", fromlist=["HeaderSummary"]).HeaderSummary(),
        recipients=__import__("app.schemas.email", fromlist=["RecipientGroups"]).RecipientGroups(),
        mime=__import__("app.schemas.email", fromlist=["MimeInformation"]).MimeInformation(),
    )
    vector = build_feature_vector(evidence)
    assert vector.text.body_plain_length == 0
    assert vector.header.received_hop_count == 0
    assert vector.url.url_count == 0
    assert vector.auth.spf_pass == 0
    assert vector.intelligence.public_ip_count == 0


def test_missing_authentication_and_intelligence_use_defaults(evidence):
    vector = build_feature_vector(evidence, None, None)
    assert vector.auth == FeatureVector().auth
    assert vector.intelligence == FeatureVector().intelligence


def test_malformed_section_does_not_break_others(evidence, monkeypatch):
    import app.features.feature_fusion as feature_fusion

    def _boom(_evidence):
        raise RuntimeError("boom")

    monkeypatch.setattr(feature_fusion, "extract_text_features", _boom)
    vector = build_feature_vector(evidence)
    assert vector.text == FeatureVector().text  # section isolated
    assert vector.header.received_hop_count >= 0  # other sections still work


# ---------------------------------------------------------------------------
# Text / header behavior
# ---------------------------------------------------------------------------


def test_text_features_count_keywords_and_lengths(evidence):
    vector = build_feature_vector(evidence)
    assert vector.text.subject_length == len(evidence.message.subject or "")
    assert vector.text.credential_signal_count >= 0
    assert vector.text.urgency_signal_count >= 0


def test_header_mismatch_flags_require_both_sides(evidence):
    vector = build_feature_vector(evidence)
    # The synthetic sample has a consistent sender domain; mismatches stay 0
    # when the counterpart is missing.
    if evidence.message.reply_to:
        assert vector.header.reply_to_present == 1
    else:
        assert vector.header.reply_to_present == 0
        assert vector.header.sender_reply_to_domain_mismatch == 0


# ---------------------------------------------------------------------------
# URL edge cases
# ---------------------------------------------------------------------------


def test_url_features_no_urls(evidence):
    stripped = evidence.model_copy(deep=True)
    stripped.urls = []
    vector = build_feature_vector(stripped)
    assert vector.url.url_count == 0
    assert vector.url.https_ratio == 0.0
    assert vector.url.ip_url_count == 0


def test_url_features_ip_literal_and_suspicious_port(evidence):
    from app.schemas.email import URLIndicator

    stripped = evidence.model_copy(deep=True)
    stripped.urls = [
        URLIndicator(url="http://192.0.2.10/login", source="plain"),
        URLIndicator(url="http://example.test:8080/pay", source="plain"),
        URLIndicator(url="https://example.test/", source="html"),
    ]
    vector = build_feature_vector(stripped)
    assert vector.url.url_count == 3
    assert vector.url.ip_url_count == 1
    assert vector.url.suspicious_port_count == 1
    assert vector.url.https_ratio == pytest.approx(1 / 3)
    assert vector.url.unique_url_registered_domain_count >= 1


def test_url_features_use_step4_registered_domain_when_available(evidence):
    from app.schemas.email import URLIndicator
    from app.schemas.intelligence import IndicatorSource, URLIntelligence

    stripped = evidence.model_copy(deep=True)
    url = "https://sub.example.test/path"
    stripped.urls = [URLIndicator(url=url, source="plain")]
    intel = URLIntelligence(
        indicator=__import__(
            "app.schemas.intelligence", fromlist=["Indicator"]
        ).Indicator(
            type="url",
            value=url,
            source=IndicatorSource(kind="url", location="body.plain"),
        ),
        scheme="https",
        hostname="sub.example.test",
        registered_domain="example.test",
    )
    vector = build_feature_vector(stripped, url_intelligence=[intel])
    assert vector.url.unique_url_registered_domain_count == 1


# ---------------------------------------------------------------------------
# Intelligence features
# ---------------------------------------------------------------------------


def test_intelligence_features_counts_and_sentinel(evidence):
    vector = build_feature_vector(evidence, None, IntelligenceEvidence())
    assert vector.intelligence.public_ip_count == 0
    assert vector.intelligence.dns_success_count == 0
    assert vector.intelligence.min_domain_age_days == -1.0


def test_intelligence_features_domain_age_relative_to_email_date(evidence):
    from datetime import datetime, timedelta, timezone

    from app.features.intelligence_features import extract_intelligence_features

    intelligence = intelligence_with_domain_ages([30.0, 100.0])
    now = datetime.now(timezone.utc)
    email_date = (now - timedelta(days=60)).isoformat()
    features = extract_intelligence_features(intelligence, email_date=email_date)  # type: ignore[arg-type]
    # Ages relative to the email date: the 30-day-old domain was registered
    # 30 days AFTER the email date (age = -30); the 100-day-old domain gives
    # +40. The minimum is -30 — a negative age is preserved, not clamped.
    assert features.min_domain_age_days == pytest.approx(-30.0, abs=0.5)
    assert features.rdap_success_count == 2


def test_intelligence_features_malformed_registration_date_is_sentinel(evidence):
    from app.schemas.intelligence import (
        DNSIntelligence,
        DomainIntelligence,
        IntelligenceEvidence,
        RDAPIntelligence,
    )

    rdap = RDAPIntelligence(
        indicator=make_indicator("domain", "example.test"),
        status="success",
        registration_date="not-a-date",
    )
    intelligence = IntelligenceEvidence(
        domains=[
            DomainIntelligence(
                indicator=make_indicator("domain", "example.test"),
                normalized_domain="example.test",
                dns=DNSIntelligence(
                    indicator=make_indicator("domain", "example.test"),
                    status="success",
                ),
                rdap=rdap,
            )
        ]
    )
    from app.features.intelligence_features import extract_intelligence_features

    features = extract_intelligence_features(intelligence)
    assert features.min_domain_age_days == -1.0
    assert features.rdap_success_count == 1
