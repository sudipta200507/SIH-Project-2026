"""Domain intelligence: normalized domain + DNS + RDAP evidence.

The combination is deliberately evidence-only: no threat verdict, no
"malicious domain" claim, no scoring. DNS and RDAP failures degrade to their
explicit status values and never raise into the caller.
"""

from __future__ import annotations

import dns.resolver

from app.intelligence.dns import DNSSettings, build_dns_intelligence
from app.intelligence.indicator_extractor import (
    Indicator,
    is_ip_literal,
    normalize_domain,
)
from app.intelligence.rdap import RDAPSettings, build_rdap_intelligence
from app.schemas.intelligence import DomainIntelligence


def build_domain_intelligence(
    indicator: Indicator,
    *,
    dns_settings: DNSSettings | None = None,
    rdap_settings: RDAPSettings | None = None,
    resolver: dns.resolver.Resolver | None = None,
    rdap_transport: object | None = None,
    rdap_bootstrap: dict[str, list[str]] | None = None,
    perform_dns: bool = True,
    perform_rdap: bool = True,
) -> DomainIntelligence:
    """Enrich one validated domain indicator with DNS and RDAP evidence.

    ``perform_dns``/``perform_rdap`` allow callers (and tests) to disable one
    provider independently; both fail soft by design.
    """

    normalized = normalize_domain(indicator.value)
    if normalized is None:
        raise ValueError(f"Domain intelligence requires a valid domain: {indicator.value!r}")

    disguised = is_ip_literal(normalized)

    dns_evidence = None
    if perform_dns:
        dns_evidence = build_dns_intelligence(
            indicator, settings=dns_settings, resolver=resolver
        )

    rdap_evidence = None
    if perform_rdap:
        rdap_evidence = build_rdap_intelligence(
            indicator,
            settings=rdap_settings,
            transport=rdap_transport,  # type: ignore[arg-type]
            bootstrap=rdap_bootstrap,
        )

    return DomainIntelligence(
        indicator=indicator,
        normalized_domain=normalized,
        is_ip_in_disguise=disguised,
        dns=dns_evidence,
        rdap=rdap_evidence,
    )


__all__ = ["build_domain_intelligence"]
