"""URL intelligence: parse Step 1 URLs into structured information.

This module performs **parsing only**. It never visits URLs, follows
redirects, downloads content, or issues any network request. All inputs are
the URL strings already extracted locally by Step 1.
"""

from __future__ import annotations

from urllib.parse import urlsplit

from app.intelligence.indicator_extractor import (
    is_ip_literal,
    parse_url_indicator,
    registered_domain_of,
)
from app.schemas.email import URLIndicator
from app.schemas.intelligence import URLIntelligence

_HTTP_DEFAULT_PORTS = {"http": 80, "https": 443}


def _strip_zone_id(hostname: str) -> str | None:
    """Normalize a hostname: strip IPv6 zone suffixes, trim brackets/edge dots."""

    cleaned = hostname.strip()
    if cleaned.startswith("[") and cleaned.endswith("]"):
        cleaned = cleaned[1:-1]
    if "%" in cleaned:
        cleaned = cleaned.split("%", 1)[0]
    cleaned = cleaned.strip().strip(".").casefold()
    return cleaned or None


def parse_url(
    url: str,
    *,
    kind: str = "url",
    location: str = "body",
) -> URLIntelligence:
    """Parse one URL string into structured intelligence without network access."""

    indicator = parse_url_indicator(url, kind=kind, location=location)

    try:
        parts = urlsplit(url)
        hostname = parts.hostname
    except ValueError as error:
        return URLIntelligence(
            indicator=indicator,
            scheme=url.split(":", 1)[0].casefold() if ":" in url else "",
            parse_error=f"url_parsing_failed: {type(error).__name__}",
        )

    scheme = parts.scheme.casefold()
    normalized_hostname = _strip_zone_id(hostname) if hostname else None
    ip_version: int | None = None
    if normalized_hostname and is_ip_literal(normalized_hostname):
        ip_version = 6 if ":" in normalized_hostname else 4
    if normalized_hostname and ip_version is None and ":" in normalized_hostname:
        ip_version = 6

    try:
        port = parts.port
    except ValueError as error:
        return URLIntelligence(
            indicator=indicator,
            scheme=scheme,
            hostname=hostname,
            normalized_hostname=normalized_hostname,
            ip_in_host=ip_version,  # type: ignore[arg-type]
            parse_error=f"port_parsing_failed: {type(error).__name__}",
        )

    default_port = _HTTP_DEFAULT_PORTS.get(scheme)
    explicit_port = port if port is not None else None
    # Keep well-known defaults implicit: the port was not present in the URL.
    effective_port = explicit_port if (explicit_port is not None or default_port is None) else None

    registered_domain = (
        registered_domain_of(normalized_hostname)
        if normalized_hostname and ip_version is None
        else None
    )

    return URLIntelligence(
        indicator=indicator,
        scheme=scheme,
        hostname=hostname,
        port=effective_port,
        path=parts.path or None,
        query=parts.query or None,
        fragment=parts.fragment or None,
        normalized_hostname=normalized_hostname,
        registered_domain=registered_domain,
        ip_in_host=ip_version,  # type: ignore[arg-type]
    )


def parse_url_indicators(
    url_indicators: list[URLIndicator],
) -> list[URLIntelligence]:
    """Parse the URL occurrences already extracted by Step 1.

    The input list may contain duplicate URLs (one per body source); each
    occurrence is preserved because provenance differs. Callers that need
    deduplication use the orchestrator.
    """

    parsed: list[URLIntelligence] = []
    for url_indicator in url_indicators:
        parsed.append(
            parse_url(url_indicator.url, kind="url", location=f"body.{url_indicator.source}")
        )
    return parsed


__all__ = ["parse_url", "parse_url_indicators"]
