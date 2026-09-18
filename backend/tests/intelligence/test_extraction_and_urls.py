"""Tests 1-8 of the Step 4 plan: indicator extraction and URL parsing."""

from __future__ import annotations

import pytest

from app.extractor.email_parser import extract_email_from_bytes
from app.intelligence.indicator_extractor import (
    classify_ip,
    extract_domains_from_headers,
    extract_indicators,
    extract_received_ips,
    is_hostname,
    normalize_domain,
    registered_domain_of,
)
from app.intelligence.url_intelligence import parse_url, parse_url_indicators

from fixtures import make_indicator  # noqa: F401


def _received_evidence(received: list[str]):
    from app.extractor.email_parser import extract_email_from_bytes

    raw = b"From: Alice <alice@example.test>\r\n\r\nbody\r\n"
    evidence = extract_email_from_bytes(raw)
    # Replace the (empty) Received chain of the minimal fixture email with
    # the values under test; the schema stays identical.
    evidence.received_chain.clear()
    evidence.received_chain.extend(received)
    evidence.headers.received.clear()
    evidence.headers.received.extend(received)
    return evidence


# ---------------------------------------------------------------------------
# 1. IPv4 extraction
# ---------------------------------------------------------------------------


class TestIPv4Extraction:
    def test_received_header_ipv4_with_provenance(self) -> None:
        evidence = _received_evidence(
            ["from relay.example.test (relay.example.test [192.0.2.10]) by mx.test"]
        )
        ips = extract_received_ips(evidence)
        assert len(ips) == 1
        assert ips[0].type == "ipv4"
        assert ips[0].value == "192.0.2.10"
        assert ips[0].source.kind == "received_header"
        assert ips[0].source.location == "received_header[0]"

    def test_second_received_header_gets_position_one(self) -> None:
        evidence = _received_evidence(
            [
                "from a.test (a.test [203.0.113.5]) by b.test",
                "from c.test (c.test [198.51.100.7]) by d.test",
            ]
        )
        values = {ip.value: ip.source.location for ip in extract_received_ips(evidence)}
        assert values == {
            "203.0.113.5": "received_header[0]",
            "198.51.100.7": "received_header[1]",
        }

    def test_invalid_and_private_strings_not_treated_as_ips(self) -> None:
        evidence = _received_evidence(
            ["from x.test ([999.1.1.1]) (id 345) (version 1.2.3) by y.test"]
        )
        assert extract_received_ips(evidence) == []

    def test_bracketed_ipv6_in_received(self) -> None:
        evidence = _received_evidence(["from v6.test ([2001:db8::25]) by mx.test"])
        ips = extract_received_ips(evidence)
        assert [ip.value for ip in ips] == ["2001:db8::25"]
        assert ips[0].type == "ipv6"

    def test_duplicate_ip_extracted_once(self) -> None:
        evidence = _received_evidence(
            ["from a.test ([192.0.2.1] helo=[192.0.2.1]) by b.test"]
        )
        assert len(extract_received_ips(evidence)) == 1


# ---------------------------------------------------------------------------
# 2. IPv6 extraction (bare token path)
# ---------------------------------------------------------------------------


class TestIPv6Extraction:
    def test_bare_full_form_ipv6(self) -> None:
        evidence = _received_evidence(["from v6.test (2001:0db8:85a3:0000:0000:8a2e:0370:7334) by mx"])
        values = [ip.value for ip in extract_received_ips(evidence)]
        assert "2001:0db8:85a3:0000:0000:8a2e:0370:7334" in values

    def test_bare_compressed_ipv6(self) -> None:
        evidence = _received_evidence(["from v6.test (2606:4700::1111) by mx"])
        values = [ip.value for ip in extract_received_ips(evidence)]
        assert "2606:4700::1111" in values

    def test_prose_time_tokens_not_ips(self) -> None:
        evidence = _received_evidence(["from mx.test; Tue, 01 Apr 2025 10:30:00 +0000"])
        assert extract_received_ips(evidence) == []


# ---------------------------------------------------------------------------
# 3+4. private / public IP classification
# ---------------------------------------------------------------------------


class TestIpClassification:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("192.0.2.10", "reserved"),  # TEST-NET-1 documentation range
            ("10.0.0.5", "private"),
            ("172.16.1.1", "private"),
            ("192.168.1.1", "private"),
            ("127.0.0.1", "loopback"),
            ("8.8.8.8", "public"),
            ("169.254.1.1", "link_local"),
            ("224.0.0.1", "multicast"),
            ("100.64.0.1", "public"),  # CGNAT: shared, not RFC1918-private
            ("::1", "loopback"),
            ("fe80::1", "link_local"),
            ("fd00::5", "private"),
            ("2001:db8::1", "reserved"),
            ("2606:4700::1111", "public"),
            ("::ffff:10.0.0.1", "private"),  # IPv4-mapped treated as IPv4
        ],
    )
    def test_classification(self, value: str, expected: str) -> None:
        assert classify_ip(value)[0] == expected

    def test_invalid_value_is_none(self) -> None:
        assert classify_ip("not-an-ip") is None
        assert classify_ip("999.1.1.1") is None

    def test_ip_intelligence_reports_version_and_global(self) -> None:
        from app.intelligence.ip_intelligence import build_ip_intelligence

        intel = build_ip_intelligence(make_indicator("ipv4", "8.8.8.8"))
        assert intel.ip_version == 4
        assert intel.classification == "public"
        assert intel.is_global is True

    def test_ipv4_mapped_normalized_to_ipv4(self) -> None:
        from app.intelligence.ip_intelligence import build_ip_intelligence

        intel = build_ip_intelligence(make_indicator("ipv6", "::ffff:8.8.8.8"))
        assert intel.ip_version == 4
        assert intel.classification == "public"


# ---------------------------------------------------------------------------
# 5. Received-header provenance
# ---------------------------------------------------------------------------


class TestReceivedProvenance:
    def test_provenance_survives_dedup_first_wins(self) -> None:
        evidence = _received_evidence(
            [
                "from edge.test ([203.0.113.9]) by mid.test",
                "from mid.test ([203.0.113.9]) by final.test",
            ]
        )
        indicators = extract_indicators(evidence)
        ip_indicators = [i for i in indicators if i.value == "203.0.113.9"]
        assert len(ip_indicators) == 1
        assert ip_indicators[0].source.location == "received_header[0]"

    def test_no_origin_or_attacker_claims_anywhere(self) -> None:
        evidence = _received_evidence(["from x.test ([203.0.113.9]) by y.test"])
        text = extract_indicators(evidence).__repr__().casefold()
        assert "attacker" not in text
        assert "originating" not in text


# ---------------------------------------------------------------------------
# 6. URL parsing
# ---------------------------------------------------------------------------


class TestUrlParsing:
    def test_full_parse(self) -> None:
        parsed = parse_url("https://mail.example.co.uk:8443/a/b?x=1#frag")
        assert parsed.scheme == "https"
        assert parsed.hostname == "mail.example.co.uk"
        assert parsed.port == 8443
        assert parsed.path == "/a/b"
        assert parsed.query == "x=1"
        assert parsed.fragment == "frag"
        assert parsed.normalized_hostname == "mail.example.co.uk"
        # Conservative PSL: multi-label public suffixes yield None.
        assert parsed.registered_domain is None

    def test_two_label_registered_domain(self) -> None:
        parsed = parse_url("http://example.test/login")
        assert parsed.registered_domain == "example.test"

    def test_defaults_and_case_and_trailing_dot(self) -> None:
        parsed = parse_url("http://WWW.Example.TEST./p")
        # ``hostname`` preserves urlsplit's form; ``normalized_hostname`` is
        # the cleaned lookup-ready form (trailing dot stripped).
        assert parsed.hostname == "www.example.test."
        assert parsed.normalized_hostname == "www.example.test"
        assert parsed.registered_domain == "example.test"
        assert parsed.port is None  # implicit 80 stays implicit

    def test_explicit_default_port_reported(self) -> None:
        parsed = parse_url("http://example.test:80/p")
        assert parsed.port == 80

    def test_ip_literal_host_detected(self) -> None:
        v4 = parse_url("http://192.0.2.9/x")
        assert v4.ip_in_host == 4
        v6 = parse_url("http://[2001:db8::1]/x")
        assert v6.ip_in_host == 6
        assert v6.hostname == "2001:db8::1"

    def test_parse_never_visits_urls(self) -> None:
        # Parsing an unusual URL must not raise or perform any network work.
        parsed = parse_url("https://example.test/@evil?next=http://other.test")
        assert parsed.hostname == "example.test"

    def test_parse_url_indicators_preserves_both_sources(self, sample) -> None:
        parsed = parse_url_indicators(sample.urls)
        locations = [p.indicator.source.location for p in parsed]
        assert locations == ["body.plain", "body.html"]


# ---------------------------------------------------------------------------
# 7. hostname normalization
# ---------------------------------------------------------------------------


class TestHostnameNormalization:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("EXAMPLE.COM", "example.com"),
            ("  www.Example.COM.  ", "www.example.com"),
            ("mail.example.co.uk", "mail.example.co.uk"),
        ],
    )
    def test_valid_normalization(self, raw: str, expected: str) -> None:
        assert normalize_domain(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        [
            "localhost",
            "singlelabel",
            "x..y",
            "-a.com",
            "a-.com",
            "bad_.com",
            "192.0.2.10",
            "",
            "a" * 64 + ".com",
            "ex ample.com",
            "exämple.com",
        ],
    )
    def test_invalid_hostnames_rejected(self, raw: str) -> None:
        assert normalize_domain(raw) is None

    def test_is_hostname_boundaries(self) -> None:
        assert is_hostname("a-b.example.test") is True
        assert is_hostname("example.test") is True
        assert is_hostname("xn--nxasmq6b.example") is True
        assert is_hostname("-leading.example.test") is False

    def test_registered_domain_conservative(self) -> None:
        assert registered_domain_of("example.test") == "example.test"
        assert registered_domain_of("mail.example.test") == "example.test"
        # Known compound public suffixes are refused instead of guessed wrong.
        assert registered_domain_of("a.b.example.co.uk") is None
        assert registered_domain_of("example.co.uk") is None


# ---------------------------------------------------------------------------
# 8. domain deduplication
# ---------------------------------------------------------------------------


class TestDomainDeduplication:
    def test_same_domain_across_fields_deduplicated(self, sample) -> None:
        # sender and Reply-To both use example.test in the synthetic sample.
        indicators = [i for i in extract_indicators(sample) if i.type == "domain"]
        values = [i.value for i in indicators]
        assert values.count("example.test") == 1

    def test_first_source_wins(self) -> None:
        from app.schemas.email import EmailEvidence

        raw = (
            b"From: A <shared@example.test>\r\n"
            b"Reply-To: B <shared@example.test>\r\n"
            b"Return-Path: <shared@example.test>\r\n\r\nbody\r\n"
        )
        evidence = extract_email_from_bytes(raw)
        domains = [i for i in extract_indicators(evidence) if i.type == "domain"]
        assert len(domains) == 1
        assert domains[0].source.kind == "sender"

    def test_header_allowlist_extraction(self) -> None:
        indicators = extract_domains_from_headers(
            {"List-Unsubscribe": ["<mailto:leave@example.test>"], "X-Weird": ["a@evil.test"]}
        )
        values = [i.value for i in indicators]
        assert values == ["example.test"]

    def test_message_id_domain_extracted(self) -> None:
        indicators = extract_domains_from_headers(
            {"Message-ID": ["<step1-fixture@example.test>"]}
        )
        assert [i.value for i in indicators] == ["example.test"]

    def test_url_hostname_becomes_domain_in_orchestrator(self, sample) -> None:
        # example.test appears as sender domain AND URL host: one domain entry.
        from app.intelligence.orchestrator import (
            IntelligenceSettings,
            build_intelligence_evidence,
        )

        result = build_intelligence_evidence(
            sample,
            settings=IntelligenceSettings(perform_dns=False, perform_rdap=False),
        )
        domain_values = [d.normalized_domain for d in result.domains]
        assert domain_values.count("example.test") == 1
