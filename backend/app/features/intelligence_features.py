"""Availability/shape features from IntelligenceEvidence (Phase C).

These features describe what the Step 4 enrichment observed — or explicitly
failed to observe (timeouts, unconfigured GeoIP). They never encode a
reputation verdict, and a lookup failure is never treated as "no record".
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.schemas.features import IntelligenceFeatures
from app.schemas.intelligence import IntelligenceEvidence


def _parse_date(value: str | None) -> datetime | None:
    """Parse an RDAP/HTTP date; None when missing or malformed."""

    if not value:
        return None
    try:
        text = value.strip().replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def extract_intelligence_features(
    intelligence: IntelligenceEvidence | None,
    *,
    email_date: str | None = None,
) -> IntelligenceFeatures:
    """Extract intelligence features; missing evidence yields a neutral vector."""

    if intelligence is None:
        return IntelligenceFeatures()

    public = 0
    non_public = 0
    for ip in intelligence.ips:
        if ip.classification == "public":
            public += 1
        else:
            non_public += 1

    dns_success = 0
    dns_unavailable = 0
    for dns in intelligence.domains:
        if dns.dns is None:
            continue
        if dns.dns.status == "success":
            dns_success += 1
        elif dns.dns.status in {"unavailable", "timeout", "error"}:
            dns_unavailable += 1

    rdap_success = sum(
        1 for d in intelligence.domains if d.rdap is not None and d.rdap.status == "success"
    )

    geoip_available = sum(
        1 for ip in intelligence.ips if ip.geolocation is not None and ip.geolocation.available
    )

    # Youngest RDAP registration age, relative to the email Date header when
    # parseable (UTC fallback otherwise). -1.0 when not determinable — never
    # guessed. Negative ages (email dated before registration) are preserved;
    # only the sentinel itself is special.
    min_age_days = -1.0
    found_any = False
    reference = _parse_date(email_date) or datetime.now(timezone.utc)
    for domain in intelligence.domains:
        registration = domain.rdap.registration_date if domain.rdap else None
        parsed = _parse_date(registration)
        if parsed is None:
            continue
        age_days = (reference - parsed).total_seconds() / 86400.0
        if not found_any or age_days < min_age_days:
            min_age_days = age_days
            found_any = True

    return IntelligenceFeatures(
        public_ip_count=public,
        non_public_ip_count=non_public,
        dns_success_count=dns_success,
        dns_unavailable_count=dns_unavailable,
        rdap_success_count=rdap_success,
        geoip_available_count=geoip_available,
        min_domain_age_days=min_age_days,
    )


__all__ = ["extract_intelligence_features"]
