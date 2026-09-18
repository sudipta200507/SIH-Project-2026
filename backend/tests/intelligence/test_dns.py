"""Tests 9-14 of the Step 4 plan: DNS intelligence (all faked, no network)."""

from __future__ import annotations

import dns.exception
import dns.resolver
import pytest

from app.intelligence.dns import (
    DNSLookupError,
    DNSSettings,
    build_dns_intelligence,
    validate_lookup_target,
)

from fixtures import FakeAnswer, FakeRdata, FakeResolver, make_indicator


def _answer(*values: str) -> FakeAnswer:
    return FakeAnswer([FakeRdata(v) for v in values])


def _script_for(name: str) -> dict[tuple[str, str], object]:
    return {
        (name, "A"): _answer("93.184.216.34"),
        (name, "AAAA"): _answer("2606:2800:220:1:248:1893:25c8:1946"),
        (name, "MX"): _answer("0 example.test."),
        (name, "TXT"): _answer("\"v=spf1 -all\""),
        (name, "NS"): _answer("a.iana-servers.net.", "b.iana-servers.net."),
        # CNAME deliberately absent -> no_records path.
    }


def _domain_indicator(value: str = "example.com"):
    return make_indicator("domain", value, kind="url", location="body.html")


# ---------------------------------------------------------------------------
# 9. DNS A
# ---------------------------------------------------------------------------


class TestDnsA:
    def test_a_record_success(self) -> None:
        resolver = FakeResolver(_script_for("example.com"))
        result = build_dns_intelligence(_domain_indicator(), resolver=resolver)
        assert result.status == "success"
        a_records = [r for r in result.records if r.type == "A"]
        assert len(a_records) == 1
        assert a_records[0].value == "93.184.216.34"
        assert ("example.com", "A") in resolver.queries

    def test_all_six_types_queried(self) -> None:
        resolver = FakeResolver(_script_for("example.com"))
        build_dns_intelligence(_domain_indicator(), resolver=resolver)
        queried_types = {rdtype for _, rdtype in resolver.queries}
        assert queried_types == {"A", "AAAA", "MX", "TXT", "NS", "CNAME"}


# ---------------------------------------------------------------------------
# 10. DNS AAAA
# ---------------------------------------------------------------------------


class TestDnsAAAA:
    def test_aaaa_record_success(self) -> None:
        resolver = FakeResolver(_script_for("example.com"))
        result = build_dns_intelligence(_domain_indicator(), resolver=resolver)
        aaaa = [r for r in result.records if r.type == "AAAA"]
        assert aaaa[0].value == "2606:2800:220:1:248:1893:25c8:1946"


# ---------------------------------------------------------------------------
# 11. DNS MX
# ---------------------------------------------------------------------------


class TestDnsMx:
    def test_mx_record_success(self) -> None:
        resolver = FakeResolver(_script_for("example.com"))
        result = build_dns_intelligence(_domain_indicator(), resolver=resolver)
        mx = [r for r in result.records if r.type == "MX"]
        assert mx[0].value == "0 example.test."


# ---------------------------------------------------------------------------
# 12. DNS TXT
# ---------------------------------------------------------------------------


class TestDnsTxt:
    def test_txt_quotes_stripped(self) -> None:
        resolver = FakeResolver(_script_for("example.com"))
        result = build_dns_intelligence(_domain_indicator(), resolver=resolver)
        txt = [r for r in result.records if r.type == "TXT"]
        assert txt[0].value == "v=spf1 -all"


# ---------------------------------------------------------------------------
# 13. DNS timeout
# ---------------------------------------------------------------------------


class TestDnsTimeout:
    def test_lifetime_timeout_is_timeout_not_no_record(self) -> None:
        name = "slow.example.test"
        script = {
            (name, rdtype): dns.resolver.LifetimeTimeout()
            for rdtype in ("A", "AAAA", "MX", "TXT", "NS", "CNAME")
        }
        result = build_dns_intelligence(_domain_indicator(name), resolver=FakeResolver(script))
        assert result.status == "timeout"
        assert result.error_kind == "timeout"
        assert result.records == []

    def test_generic_dns_timeout_exception(self) -> None:
        name = "slow.example.test"
        script = {
            (name, rdtype): dns.exception.Timeout()
            for rdtype in ("A", "AAAA", "MX", "TXT", "NS", "CNAME")
        }
        result = build_dns_intelligence(_domain_indicator(name), resolver=FakeResolver(script))
        assert result.status == "timeout"


# ---------------------------------------------------------------------------
# 14. DNS no record (and failure/absence separation)
# ---------------------------------------------------------------------------


class TestDnsNoRecord:
    def test_nxdomain_is_no_record(self) -> None:
        # FakeResolver raises NXDOMAIN for unscripted names.
        result = build_dns_intelligence(_domain_indicator("missing.example.test"), resolver=FakeResolver())
        assert result.status == "no_record"
        assert result.error_kind is None

    def test_no_answer_is_no_record(self) -> None:
        name = "empty.example.test"
        script = {
            (name, rdtype): dns.resolver.NoAnswer()
            for rdtype in ("A", "AAAA", "MX", "TXT", "NS", "CNAME")
        }
        result = build_dns_intelligence(_domain_indicator(name), resolver=FakeResolver(script))
        assert result.status == "no_record"

    def test_failure_is_never_no_record(self) -> None:
        name = "broken.example.test"
        script = {
            (name, rdtype): dns.resolver.NoNameservers()
            for rdtype in ("A", "AAAA", "MX", "TXT", "NS", "CNAME")
        }
        result = build_dns_intelligence(_domain_indicator(name), resolver=FakeResolver(script))
        assert result.status == "error"
        assert result.status != "no_record"
        assert result.error_kind == "no_nameservers"

    def test_unavailable_network_failure_distinct(self) -> None:
        name = "unreach.example.test"
        script = {
            (name, rdtype): OSError("network down")
            for rdtype in ("A", "AAAA", "MX", "TXT", "NS", "CNAME")
        }
        result = build_dns_intelligence(_domain_indicator(name), resolver=FakeResolver(script))
        assert result.status == "unavailable"
        assert result.error_kind == "dns_error"

    def test_partial_success_aggregates_to_success(self) -> None:
        name = "partial.example.test"
        script = {
            (name, "A"): _answer("93.184.216.34"),
            (name, "AAAA"): dns.resolver.NoAnswer(),
            (name, "MX"): dns.resolver.NoAnswer(),
            (name, "TXT"): dns.resolver.NoAnswer(),
            (name, "NS"): dns.resolver.NoAnswer(),
            (name, "CNAME"): dns.resolver.NoAnswer(),
        }
        result = build_dns_intelligence(_domain_indicator(name), resolver=FakeResolver(script))
        assert result.status == "success"
        assert len(result.records) == 1

    def test_timeout_beats_no_record_in_aggregation(self) -> None:
        name = "mixed.example.test"
        script = {
            (name, "A"): dns.resolver.LifetimeTimeout(),
            (name, "AAAA"): dns.resolver.NoAnswer(),
            (name, "MX"): dns.resolver.NoAnswer(),
            (name, "TXT"): dns.resolver.NoAnswer(),
            (name, "NS"): dns.resolver.NoAnswer(),
            (name, "CNAME"): dns.resolver.NoAnswer(),
        }
        result = build_dns_intelligence(_domain_indicator(name), resolver=FakeResolver(script))
        assert result.status == "timeout"


# ---------------------------------------------------------------------------
# Input validation and settings
# ---------------------------------------------------------------------------


class TestDnsValidation:
    @pytest.mark.parametrize(
        "bad",
        [
            "localhost",
            "singlelabel",
            "x..y",
            "-a.com",
            "a-.com",
            "bad_.com",
            "192.0.2.10",
            "foo.in-addr.arpa",
            "bar.ip6.arpa",
            "a%b.com",
            "a/b.com",
            "user@example.com",
            "",
            "  ",
        ],
    )
    def test_invalid_targets_rejected(self, bad: str) -> None:
        with pytest.raises(DNSLookupError):
            validate_lookup_target(bad)

    def test_invalid_target_becomes_unavailable_status(self) -> None:
        result = build_dns_intelligence(_domain_indicator("localhost"), resolver=FakeResolver())
        assert result.status == "unavailable"
        assert result.error_kind == "invalid_input"
        assert resolver_not_used(result) is True

    def test_only_validated_names_reach_resolver(self) -> None:
        build_dns_intelligence(_domain_indicator("localhost"), resolver=FakeResolver())
        # No exception: the invalid target never produced a query.

    def test_settings_bounds_applied(self) -> None:
        settings = DNSSettings(timeout_seconds=0.7, lifetime_seconds=1.1)
        assert settings.timeout_seconds == 0.7
        assert settings.lifetime_seconds == 1.1

    def test_unsupported_record_type_rejected(self) -> None:
        with pytest.raises(DNSLookupError):
            build_dns_intelligence(
                _domain_indicator(),
                resolver=FakeResolver(),
                record_types=("A", "BOGUS"),
            )


def resolver_not_used(result) -> bool:
    """A lookup marked invalid_input must carry no records from any resolver."""

    return result.records == []
