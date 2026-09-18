"""IP intelligence: normalize and classify IPs locally.

Focused first implementation: normalization and special-range classification
only. No external reputation services (VirusTotal, AbuseIPDB, Shodan,
GreyNoise, ...) are contacted; those decisions belong to a later phase.
"""

from __future__ import annotations

from ipaddress import ip_address

from app.intelligence.indicator_extractor import classify_ip
from app.schemas.intelligence import GeoLocation, Indicator, IPIntelligence


def build_ip_intelligence(
    indicator: Indicator,
    *,
    geolocation: GeoLocation | None = None,
) -> IPIntelligence:
    """Classify one validated IPv4/IPv6 indicator.

    Raises :class:`ValueError` when the indicator is not a valid IP literal —
    callers pass indicators produced by the validated extractor.
    ``geolocation`` attaches already-built GeoIP evidence (built separately by
    the geolocation module so each provider stays independently testable).
    """

    value = indicator.value
    if indicator.type not in {"ipv4", "ipv6"}:
        raise ValueError(f"Indicator type {indicator.type!r} is not an IP indicator.")
    classified = classify_ip(value)
    if classified is None:
        raise ValueError(f"Invalid IP literal: {value!r}")
    classification, version = classified

    address = ip_address(value)
    if address.version == 6 and address.ipv4_mapped:
        # Report the mapped IPv4 form; the original stays available via raw.
        normalized = address.ipv4_mapped.compressed
    else:
        normalized = address.compressed

    return IPIntelligence(
        indicator=indicator,
        ip_version=version,  # type: ignore[arg-type]
        classification=classification,  # type: ignore[arg-type]
        is_global=ip_address(normalized).is_global,
        geolocation=geolocation,
    )


__all__ = ["build_ip_intelligence"]
