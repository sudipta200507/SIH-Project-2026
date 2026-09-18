"""Step 4 orchestrator: EmailEvidence → indicators → DNS/RDAP/GeoIP → evidence.

The orchestrator composes the intelligence modules and enforces the Step 4
behavior contract:

- indicator extraction is pure local parsing (no network);
- lookups run only for validated indicators, each bounded by explicit
  timeouts and per-run caps;
- a failing provider degrades to an explicit status; it never destroys the
  email analysis;
- all provenance (``IndicatorSource``) is preserved end to end.
"""

from __future__ import annotations

from dataclasses import dataclass

import dns.resolver

from app.intelligence.dns import DNSSettings, build_dns_intelligence
from app.intelligence.domain_intelligence import build_domain_intelligence
from app.intelligence.geolocation import (
    GeoIPSettings,
    build_geolocation,
    load_geoip_settings,
)
from app.intelligence.indicator_extractor import (
    Indicator,
    extract_indicators,
    normalize_domain,
)
from app.intelligence.ip_intelligence import build_ip_intelligence
from app.intelligence.rdap import RDAPSettings, build_rdap_intelligence
from app.intelligence.url_intelligence import parse_url_indicators
from app.schemas.email import EmailEvidence
from app.schemas.intelligence import GeoLocation, IntelligenceEvidence

# Re-exported for callers that want one import surface.
__all__ = [
    "IntelligenceSettings",
    "build_dns_intelligence",
    "build_geolocation",
    "build_intelligence_evidence",
    "build_ip_intelligence",
    "build_rdap_intelligence",
    "extract_indicators",
]


@dataclass(frozen=True, slots=True)
class IntelligenceSettings:
    """Bounds and toggles for one orchestrated intelligence run."""

    dns: DNSSettings = DNSSettings()
    rdap: RDAPSettings = RDAPSettings()
    geoip: GeoIPSettings | None = None
    max_ips: int = 32
    max_domains: int = 32
    max_urls: int = 64
    # Provider toggles (tests and offline prototypes may disable either one).
    perform_dns: bool = True
    perform_rdap: bool = True


def _bounded(values: list[Indicator], limit: int) -> list[Indicator]:
    return values[:limit]


def _enrich_ip(
    indicator: Indicator,
    geoip_settings: GeoIPSettings,
    city_reader: object | None,
    asn_reader: object | None,
):
    """IP classification plus approximate geolocation; both fail soft."""

    intel = build_ip_intelligence(indicator)
    try:
        intel.geolocation = build_geolocation(
            indicator,
            settings=geoip_settings,
            city_reader=city_reader,
            asn_reader=asn_reader,
        )
    except Exception:
        intel.geolocation = GeoLocation(
            indicator=indicator, available=False, reason="lookup_failed"
        )
    return intel


def _correlate_url_hosts(
    url_intel: list,
    domain_intel: list,
    *,
    active: IntelligenceSettings,
    resolver: dns.resolver.Resolver | None,
    rdap_transport: object | None,
    rdap_bootstrap: dict[str, list[str]] | None,
) -> None:
    """Ensure each deduplicated URL hostname has domain evidence attached."""

    known = {entry.normalized_domain for entry in domain_intel}
    for url_evidence in url_intel:
        host = url_evidence.normalized_hostname
        if not host or url_evidence.ip_in_host is not None:
            continue
        normalized_host = normalize_domain(host)
        if normalized_host is None or normalized_host in known:
            continue
        try:
            domain_intel.append(
                build_domain_intelligence(
                    Indicator(
                        type="domain",
                        value=normalized_host,
                        source={
                            "kind": "url",
                            "location": url_evidence.indicator.source.location,
                        },
                    ),
                    dns_settings=active.dns,
                    rdap_settings=active.rdap,
                    resolver=resolver,
                    rdap_transport=rdap_transport,
                    rdap_bootstrap=rdap_bootstrap,
                    perform_dns=active.perform_dns,
                    perform_rdap=active.perform_rdap,
                )
            )
            known.add(normalized_host)
        except Exception:
            continue


def build_intelligence_evidence(
    evidence: EmailEvidence,
    *,
    settings: IntelligenceSettings | None = None,
    resolver: dns.resolver.Resolver | None = None,
    rdap_transport: object | None = None,
    rdap_bootstrap: dict[str, list[str]] | None = None,
    geoip_city_reader: object | None = None,
    geoip_asn_reader: object | None = None,
) -> IntelligenceEvidence:
    """Build the Step 4 evidence object for one normalized email.

    All external lookups are injectable for tests; production calls use the
    real bounded providers. No exception from a provider is allowed to escape:
    a provider failure degrades that provider's status only.
    """

    active = settings or IntelligenceSettings()
    geoip_settings = active.geoip if active.geoip is not None else load_geoip_settings()

    indicators = extract_indicators(evidence)
    url_intel = parse_url_indicators(evidence.urls)[: active.max_urls]

    ip_intel: list = []
    ip_indicators = _bounded(
        [i for i in indicators if i.type in {"ipv4", "ipv6"}], active.max_ips
    )
    for indicator in ip_indicators:
        try:
            ip_intel.append(
                _enrich_ip(indicator, geoip_settings, geoip_city_reader, geoip_asn_reader)
            )
        except ValueError:
            continue

    domain_intel: list = []
    domain_indicators = _bounded(
        [i for i in indicators if i.type == "domain"], active.max_domains
    )
    for indicator in domain_indicators:
        try:
            domain_intel.append(
                build_domain_intelligence(
                    indicator,
                    dns_settings=active.dns,
                    rdap_settings=active.rdap,
                    resolver=resolver,
                    rdap_transport=rdap_transport,
                    rdap_bootstrap=rdap_bootstrap,
                    perform_dns=active.perform_dns,
                    perform_rdap=active.perform_rdap,
                )
            )
        except Exception:
            continue

    # Correlation: deduplicated URL hostnames join the domain evidence set.
    _correlate_url_hosts(
        url_intel,
        domain_intel,
        active=active,
        resolver=resolver,
        rdap_transport=rdap_transport,
        rdap_bootstrap=rdap_bootstrap,
    )

    counts = {
        "indicators": len(indicators),
        "urls": len(url_intel),
        "ips": len(ip_intel),
        "domains": len(domain_intel),
    }

    return IntelligenceEvidence(
        indicators=indicators,
        urls=url_intel,
        ips=ip_intel,
        domains=domain_intel,
        counts=counts,
    )
