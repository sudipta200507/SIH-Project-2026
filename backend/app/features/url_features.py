"""Deterministic URL-shape features (Phase C).

Inputs come from Step 1 URL extraction and, when available, Step 4 URL
decomposition (registered domains, IP-in-host). URLs are never visited and
no network request is made; only already-extracted strings are parsed.
"""

from __future__ import annotations

from urllib.parse import urlparse

from app.intelligence.indicator_extractor import classify_ip, registered_domain_of
from app.schemas.email import EmailEvidence
from app.schemas.features import UrlFeatures

_DEFAULT_PORTS = {"http": 80, "https": 443}


def _explicit_port(scheme: str, port: int | None) -> bool:
    """True when a URL declares a non-default port for its scheme."""

    if port is None:
        return False
    return _DEFAULT_PORTS.get(scheme, port) != port


def extract_url_features(
    evidence: EmailEvidence,
    *,
    url_intelligence: list | None = None,
) -> UrlFeatures:
    """Extract URL features; never raises, never touches the network.

    ``url_intelligence`` (Step 4 ``URLIntelligence`` objects) is optional:
    when supplied, registered domains and IP-in-host come from the already
    computed decomposition; otherwise a conservative local heuristic is used.
    """

    urls = evidence.urls
    if not urls:
        return UrlFeatures()

    # Optional Step 4 lookup keyed by the original URL string.
    intel_by_url: dict[str, object] = {}
    if url_intelligence:
        intel_by_url = {item.indicator.value: item for item in url_intelligence}

    https_count = 0
    ip_url_count = 0
    suspicious_port_count = 0
    max_path_length = 0
    hostnames: list[str] = []
    registered_domains: set[str] = set()
    sender_registered: str | None = None
    if evidence.sender and evidence.sender.address and "@" in evidence.sender.address:
        sender_registered = registered_domain_of(
            evidence.sender.address.rpartition("@")[2]
        )

    mismatch_count = 0

    for url_indicator in urls:
        raw = url_indicator.url
        parsed = urlparse(raw)
        scheme = (parsed.scheme or "").casefold()
        hostname = (parsed.hostname or "").casefold()

        if scheme == "https":
            https_count += 1

        if parsed.port is not None and _explicit_port(scheme, parsed.port):
            suspicious_port_count += 1

        if parsed.path:
            max_path_length = max(max_path_length, len(parsed.path))

        if hostname:
            hostnames.append(hostname)

        # Step 4 knowledge wins when present; else local heuristics.
        intel = intel_by_url.get(raw)
        ip_version = None
        registered: str | None = None
        if intel is not None:
            ip_version = getattr(intel, "ip_in_host", None)
            registered = getattr(intel, "registered_domain", None)
        if ip_version is None and hostname:
            if classify_ip(hostname) is not None:
                ip_version = 4 if "." in hostname else 6
        if registered is None and hostname and "." in hostname and not hostname.startswith("["):
            registered = registered_domain_of(hostname)

        if ip_version is not None:
            ip_url_count += 1
        if registered:
            registered_domains.add(registered)
            if sender_registered is not None and registered != sender_registered:
                mismatch_count += 1

    total = len(urls)
    return UrlFeatures(
        url_count=total,
        unique_url_host_count=len(set(hostnames)),
        https_ratio=https_count / total,
        ip_url_count=ip_url_count,
        suspicious_port_count=suspicious_port_count,
        max_url_path_length=max_path_length,
        unique_url_registered_domain_count=len(registered_domains),
        url_registered_domain_mismatch_count=mismatch_count,
    )


__all__ = ["extract_url_features"]
