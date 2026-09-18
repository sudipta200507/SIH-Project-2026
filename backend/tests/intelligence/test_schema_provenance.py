"""Tests 20-21 of the Step 4 plan: schema validation and provenance."""

from __future__ import annotations

import io

import pytest

from app.intelligence.orchestrator import (
    IntelligenceSettings,
    build_intelligence_evidence,
)
from app.schemas.intelligence import (
    DNSIntelligence,
    DomainIntelligence,
    GeoLocation,
    Indicator,
    IndicatorSource,
    IntelligenceEvidence,
    IPIntelligence,
    RDAPIntelligence,
    URLIntelligence,
)

from fixtures import make_indicator, sample_evidence


def _source(kind: str = "sender", location: str = "sender") -> dict:
    return {"kind": kind, "location": location}


# ---------------------------------------------------------------------------
# 20. intelligence schema validation
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValueError):
            IndicatorSource(kind="sender", location="sender", surprise="x")

    def test_indicator_requires_known_type(self) -> None:
        with pytest.raises(ValueError):
            Indicator(type="iban", value="x", source=_source())

    def test_indicator_requires_known_source_kind(self) -> None:
        with pytest.raises(ValueError):
            IndicatorSource(kind="gossip", location="x")

    def test_lookup_status_vocabulary_enforced(self) -> None:
        base = dict(indicator=make_indicator("domain", "example.test").model_dump())
        for good in ("success", "no_record", "unavailable", "timeout", "error"):
            DNSIntelligence(**base, status=good)
        with pytest.raises(ValueError):
            DNSIntelligence(**base, status="maybe")

    def test_ip_intelligence_requires_valid_payload(self) -> None:
        with pytest.raises(ValueError):
            IPIntelligence(
                indicator=_source(),  # wrong type on purpose
                ip_version=4,
                classification="public",
                is_global=True,
            )

    def test_geoip_reason_vocabulary_enforced(self) -> None:
        geo = GeoLocation(indicator=make_indicator("ipv4", "8.8.8.8"), available=False, reason="not_configured")
        assert geo.reason == "not_configured"
        with pytest.raises(ValueError):
            GeoLocation(indicator=make_indicator("ipv4", "8.8.8.8"), available=False, reason="because")

    def test_rdap_error_kind_vocabulary_enforced(self) -> None:
        base = dict(indicator=make_indicator("domain", "example.test").model_dump(), status="unavailable")
        RDAPIntelligence(**base, error_kind="http_error")
        RDAPIntelligence(**base, error_kind="no_registry")
        with pytest.raises(ValueError):
            RDAPIntelligence(**base, error_kind="exploded")

    def test_url_port_range_enforced(self) -> None:
        base = dict(
            indicator=make_indicator("url", "http://x.test/").model_dump(),
            scheme="http",
        )
        URLIntelligence(**base, port=65535)
        with pytest.raises(ValueError):
            URLIntelligence(**base, port=70000)

    def test_intelligence_evidence_defaults(self) -> None:
        evidence = IntelligenceEvidence()
        assert evidence.schema_version == "1.0"
        assert evidence.indicators == []
        assert evidence.counts == {}

    def test_full_evidence_roundtrip(self) -> None:
        result = build_intelligence_evidence(
            sample_evidence(),
            settings=IntelligenceSettings(perform_dns=False, perform_rdap=False),
        )
        dumped = result.model_dump_json()
        reparsed = IntelligenceEvidence.model_validate_json(dumped)
        assert reparsed == result

    def test_no_scoring_fields_anywhere(self) -> None:
        result = build_intelligence_evidence(
            sample_evidence(),
            settings=IntelligenceSettings(perform_dns=False, perform_rdap=False),
        )
        exported = result.model_dump_json().casefold()
        for forbidden in ("risk_score", "threat_type", "confidence", "ai_prediction", "verdict", "is_malicious"):
            assert forbidden not in exported


# ---------------------------------------------------------------------------
# 21. provenance preservation
# ---------------------------------------------------------------------------


class TestProvenancePreservation:
    def test_received_ip_provenance_end_to_end(self) -> None:
        result = build_intelligence_evidence(
            sample_evidence(),
            settings=IntelligenceSettings(perform_dns=False, perform_rdap=False),
        )
        ip_entries = [i for i in result.ips if i.indicator.value == "192.0.2.10"]
        assert len(ip_entries) == 1
        assert ip_entries[0].indicator.source.kind == "received_header"
        assert ip_entries[0].indicator.source.location == "received_header[0]"
        # The same provenance travels with the geolocation evidence.
        assert ip_entries[0].geolocation is not None
        assert ip_entries[0].geolocation.indicator.source.location == "received_header[0]"

    def test_url_provenance_keeps_body_source(self) -> None:
        result = build_intelligence_evidence(
            sample_evidence(),
            settings=IntelligenceSettings(perform_dns=False, perform_rdap=False),
        )
        locations = sorted(u.indicator.source.location for u in result.urls)
        assert locations == ["body.html", "body.plain"]

    def test_domain_provenance_from_sender(self) -> None:
        result = build_intelligence_evidence(
            sample_evidence(),
            settings=IntelligenceSettings(perform_dns=False, perform_rdap=False),
        )
        domains = [d for d in result.domains if d.normalized_domain == "example.test"]
        assert len(domains) == 1
        assert domains[0].indicator.source.kind == "sender"

    def test_source_kinds_are_stable_identifiers(self) -> None:
        from app.intelligence.indicator_extractor import extract_indicators

        indicators = extract_indicators(sample_evidence())
        allowed = {"received_header", "header_field", "sender", "reply_to", "return_path", "message_id", "url"}
        for indicator in indicators:
            assert indicator.source.kind in allowed
            assert indicator.source.location
