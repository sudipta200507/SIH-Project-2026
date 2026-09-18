"""Live integration tests (controlled): real DNS and RDAP lookups only.

Scope rules for this file:

- only the RFC-safe documentation/IANA domains ``example.com``,
  ``example.net``, and ``iana.org`` are queried;
- no suspicious URLs, no crawling, no email content ever leaves the machine;
- every test skips honestly (with the observed reason) when the live service
  is unreachable instead of fabricating results.

These tests intentionally do NOT run in the normal offline unit suite: they
are marked ``live`` and deselected by default. Run them explicitly:

    python -m pytest tests/intelligence -m live -q
"""

from __future__ import annotations

import os

import pytest

import dns.resolver

from app.intelligence.dns import build_dns_intelligence
from app.intelligence.rdap import build_rdap_intelligence

from fixtures import make_indicator

pytestmark = [
    pytest.mark.live,
    # The suite's default run must stay hermetic; opt in with -m live.
]

LIVE_DOMAIN = os.environ.get("INTELLIGENCE_LIVE_DOMAIN", "example.com")


def _dns_reachable() -> bool:
    """Quick, bounded check that a resolver answers at all."""

    try:
        resolver = dns.resolver.Resolver(configure=True)
        resolver.timeout = 2.0
        resolver.lifetime = 4.0
        resolver.resolve(LIVE_DOMAIN, "A")
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Live DNS
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _dns_reachable(), reason="No working DNS resolver available")
class TestLiveDns:
    def test_dns_a_lookup_example_com(self) -> None:
        result = build_dns_intelligence(
            make_indicator("domain", LIVE_DOMAIN, kind="url", location="body.plain"),
            record_types=("A",),
        )
        assert result.status == "success"
        a_records = [r for r in result.records if r.type == "A"]
        assert a_records, "example.com must resolve at least one A record"

    def test_dns_ns_lookup_example_com(self) -> None:
        result = build_dns_intelligence(
            make_indicator("domain", LIVE_DOMAIN, kind="url", location="body.plain"),
            record_types=("NS",),
        )
        assert result.status == "success"
        assert result.records

    def test_dns_nxdomain_is_no_record(self) -> None:
        result = build_dns_intelligence(
            make_indicator("domain", "this-domain-cannot-exist-4815162342.test", kind="url", location="body.plain"),
            record_types=("A",),
        )
        assert result.status == "no_record"


# ---------------------------------------------------------------------------
# Live RDAP (example.com is registered via IANA's RDAP service)
# ---------------------------------------------------------------------------


def _rdap_reachable() -> bool:
    try:
        import httpx

        response = httpx.get("https://rdap.org/domain/example.com", timeout=5.0)
        return response.status_code in {200, 404}
    except Exception:
        return False


@pytest.mark.skipif(not _rdap_reachable(), reason="RDAP service is unreachable")
class TestLiveRdap:
    def test_rdap_lookup_example_com(self) -> None:
        result = build_rdap_intelligence(
            make_indicator("domain", LIVE_DOMAIN, kind="url", location="body.plain")
        )
        # example.com is IANA-reserved; both a success and a registry-level
        # refusal are honest outcomes. Only an exception would be a bug.
        assert result.status in {"success", "no_record", "unavailable"}
        if result.status == "success":
            assert result.domain_name == "example.com"
            assert result.registrar or result.nameservers or result.statuses

    def test_rdap_invalid_tld_reports_no_registry(self) -> None:
        result = build_rdap_intelligence(
            make_indicator("domain", "host.invalidtld-4815162342", kind="url", location="body.plain")
        )
        assert result.status == "unavailable"
        assert result.error_kind in {"no_registry", "http_error"}


# ---------------------------------------------------------------------------
# Live GeoIP: honesty contract only (never fabricates availability)
# ---------------------------------------------------------------------------


class TestLiveGeoIPBehavior:
    def test_geoip_off_reports_not_configured_honestly(self) -> None:
        from app.intelligence.geolocation import GeoIPSettings, build_geolocation

        result = build_geolocation(
            make_indicator("ipv4", "192.0.2.10", kind="received_header", location="received_header[0]"),
            settings=GeoIPSettings(mode="off"),
        )
        assert result.available is False
        assert result.reason == "not_configured"

    @pytest.mark.skipif(
        not os.environ.get("MAXMIND_DB_PATH"),
        reason="MAXMIND_DB_PATH is not configured on this machine",
    )
    def test_geoip_configured_local_lookup(self) -> None:
        from app.intelligence.geolocation import build_geolocation, load_geoip_settings

        result = build_geolocation(
            make_indicator("ipv4", "8.8.8.8", kind="received_header", location="received_header[0]"),
            settings=load_geoip_settings(),
        )
        # With a real GeoLite2 database configured, 8.8.8.8 must produce
        # either data or an honest lookup failure — never invented values.
        if result.available:
            assert result.country is None or isinstance(result.country, str)
