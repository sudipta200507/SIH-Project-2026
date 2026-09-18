"""Extraction of validated candidate indicators from Step 1 EmailEvidence.

Extraction is local parsing and validation only:

- no DNS queries, no RDAP queries, no HTTP requests;
- strings are accepted only after structured validation (``ipaddress``,
  ``urlsplit``, RFC 1123/5321 checks);
- every accepted indicator keeps its provenance (``IndicatorSource``) so later
  forensic phases can reason about where it came from;
- an IP inside the Received chain is a *candidate* IP: this module never
  claims originating-sender attribution.

Domain extraction is split into a reusable core (``iter_email_domains`` and
helpers such as ``extract_domains_from_headers``) so the orchestrator can reuse
it for correlation with DNS/RDAP results.
"""

from __future__ import annotations

import re
from ipaddress import (
    AddressValueError,
    IPv4Address,
    IPv6Address,
    ip_address,
    ip_network,
)
from urllib.parse import urlsplit

from app.schemas.email import EmailEvidence
from app.schemas.intelligence import Indicator, IndicatorSource

# Header fields that may legitimately carry hostnames, domains, or addresses.
# A fixed allowlist prevents unbounded scanning of arbitrary header fields.
INDICATOR_HEADER_FIELDS: tuple[str, ...] = (
    "From",
    "Reply-To",
    "Return-Path",
    "Message-ID",
    "List-Unsubscribe",
    "List-Subscribe",
    "List-Post",
    "List-ID",
    "Authentication-Results",
    "Received-SPF",
)

# RFC 1123 hostname label constraints.
_HOSTNAME_LABEL = re.compile(r"^(?!-)[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?$", re.IGNORECASE)
_TLD_SHAPE = re.compile(r"^[A-Z]{2,63}$", re.IGNORECASE)
# Mailbox token: roughly RFC 5321 dot-atom, kept intentionally conservative.
_ADDR_TOKEN = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
# Bracketed IPv4 literal as used in Received comments and address literals.
_IPV4_BRACKET = re.compile(r"(?:\d{1,3}\.){3}\d{1,3}")
# IPv6 candidates inside brackets (RFC 5321 address literals).
_IPV6_BRACKET = re.compile(r"\[([0-9A-Fa-f:]*:[0-9A-Fa-f:]+(?:%[0-9A-Za-z]+)?)\]")
# Whitespace/punctuation delimiters used to isolate bare IPv6 tokens.
_TOKEN_SPLIT = re.compile(r"[\s,;<>()]+")

# Domains that exist only for documentation/testing (RFC 2606, RFC 6761/6762).
_SPECIAL_USE_DOMAINS = frozenset(
    {
        "example.com",
        "example.net",
        "example.org",
        "example.test",
        "invalid",
        "test",
        "localhost",
        "localhost.localdomain",
    }
)


class IndicatorExtractionError(Exception):
    """Raised for programmer-facing misuse of the extraction API."""


def _source(kind: str, location: str, field: str | None = None) -> IndicatorSource:
    return IndicatorSource(kind=kind, location=location, field=field)  # type: ignore[arg-type]


# RFC 1918 privacy ranges (the only ranges reported as "private").
_RFC1918_NETWORKS = (
    ip_network("10.0.0.0/8"),
    ip_network("172.16.0.0/12"),
    ip_network("192.168.0.0/16"),
)
# IPv6 unique-local addresses (the RFC 4193 analog of RFC 1918).
_IPV6_UNIQUE_LOCAL = ip_network("fc00::/7")


def classify_ip(value: str) -> tuple[str, int] | None:
    """Canonical IP validation/classification; (classification, version) or None.

    ``private`` is reserved for RFC 1918 (IPv4) and unique-local (IPv6)
    ranges. Other IANA special-purpose ranges (documentation, benchmarking,
    CGNAT, ...) report as ``reserved`` so forensic output can distinguish
    "internal network" from "special-purpose address space".
    """

    try:
        address = ip_address(value)
    except ValueError:
        return None
    if isinstance(address, IPv6Address) and address.ipv4_mapped:
        # ::ffff:x.x.x.x is an IPv4 address in disguise; treat it as IPv4.
        return classify_ip(str(address.ipv4_mapped))
    if address.is_loopback:
        classification = "loopback"
    elif address.is_link_local:
        classification = "link_local"
    elif address.is_multicast:
        classification = "multicast"
    elif address.version == 4 and any(address in net for net in _RFC1918_NETWORKS):
        classification = "private"
    elif address.version == 6 and address in _IPV6_UNIQUE_LOCAL:
        classification = "private"
    elif address.is_private or address.is_reserved or address.is_unspecified:
        classification = "reserved"
    else:
        classification = "public"
    return classification, address.version


def _dedupe(indicators: list[Indicator]) -> list[Indicator]:
    """Keep the first occurrence of each (type, value) pair, dropping duplicates."""
    seen: set[tuple[str, str]] = set()
    unique: list[Indicator] = []
    for indicator in indicators:
        key = (indicator.type, indicator.value)
        if key in seen:
            continue
        seen.add(key)
        unique.append(indicator)
    return unique


def _make_indicator(kind: str, location: str, type_: str, value: str, raw: str | None = None,
                    field: str | None = None) -> Indicator:
    return Indicator(
        type=type_,  # type: ignore[arg-type]
        value=value,
        raw=raw,
        source=_source(kind, location, field),
    )


# ---------------------------------------------------------------------------
# IP extraction
# ---------------------------------------------------------------------------


def is_ip_literal(value: str) -> bool:
    """Return True when value is a syntactically valid IPv4/IPv6 literal."""
    return classify_ip(value) is not None


def extract_received_ips(evidence: EmailEvidence) -> list[Indicator]:
    """Extract candidate IPs from each Received header with provenance.

    Order is the header order in ``received_chain`` (top of the message
    first). Each IP is a *candidate*; no originating-IP claim is made.
    """

    indicators: list[Indicator] = []
    for position, header in enumerate(evidence.received_chain):
        seen_in_header: set[str] = set()
        for match in _IPV4_BRACKET.finditer(header):
            token = match.group(0)
            if token in seen_in_header or classify_ip(token) is None:
                continue
            seen_in_header.add(token)
            indicators.append(
                _make_indicator("received_header", f"received_header[{position}]", "ipv4", token)
            )
        # Bracketed IPv6 address literals.
        for match in _IPV6_BRACKET.finditer(header):
            token = match.group(1).strip()
            if token in seen_in_header or classify_ip(token) is None:
                continue
            seen_in_header.add(token)
            indicators.append(
                _make_indicator("received_header", f"received_header[{position}]", "ipv6", token)
            )
        # Bare IPv6 tokens (validated individually; nothing is trusted blindly).
        for token in _TOKEN_SPLIT.split(header):
            if ":" not in token:
                continue
            candidate = token.strip("[]")
            if candidate in seen_in_header or classify_ip(candidate) is None:
                continue
            seen_in_header.add(candidate)
            indicators.append(
                _make_indicator(
                    "received_header", f"received_header[{position}]", "ipv6", candidate
                )
            )
    return indicators


def extract_ip_indicators(evidence: EmailEvidence) -> list[Indicator]:
    """Extract IPs from the Received chain (the only structured IP source)."""

    return _dedupe(extract_received_ips(evidence))


# ---------------------------------------------------------------------------
# Hostname and domain validation helpers
# ---------------------------------------------------------------------------


def is_hostname(value: str) -> bool:
    """RFC 1123 hostname check: labels + alphabetic TLD shape."""
    if not value or len(value) > 253 or value != value.strip() or not value.isascii():
        return False
    labels = value.split(".")
    if len(labels) < 2:
        return False
    for label in labels:
        if not _HOSTNAME_LABEL.match(label):
            return False
    return _TLD_SHAPE.match(labels[-1]) is not None


def normalize_domain(value: str) -> str | None:
    """Normalize one candidate domain string; None when not a valid domain."""
    candidate = value.strip().strip(".").casefold()
    if not candidate or not candidate.isascii():
        return None
    if not is_hostname(candidate):
        return None
    return candidate


def is_domain(value: str) -> bool:
    """True when value is a normalized valid domain name."""
    return normalize_domain(value) is not None


# Common multi-label public suffixes. Without a Public Suffix List library,
# the registrable-domain heuristic below refuses these instead of guessing.
_COMPOUND_SUFFIXES = frozenset(
    {
        "co.uk", "org.uk", "ac.uk", "gov.uk", "me.uk", "net.uk", "ltd.uk", "plc.uk",
        "com.au", "net.au", "org.au", "edu.au", "gov.au",
        "co.jp", "ne.jp", "or.jp", "ac.jp",
        "com.br", "net.br", "org.br",
        "co.in", "net.in", "org.in",
        "co.nz", "net.nz", "org.nz",
        "com.mx", "com.ar", "com.tr", "com.cn", "com.tw", "com.hk", "com.sg",
        "co.za", "com.ua", "co.kr",
    }
)


def registered_domain_of(hostname: str) -> str | None:
    """Registered/domain portion when safely determinable, else None.

    No Public Suffix List library is configured in this step, so the rule is:
    the last two labels are returned unless they are a known compound public
    suffix (e.g. ``co.uk``); in that case None is returned rather than a
    wrong answer. A PSL-based refinement belongs to a later phase.
    """

    normalized = normalize_domain(hostname)
    if normalized is None:
        return None
    labels = normalized.split(".")
    if len(labels) < 2:
        return None
    last_two = ".".join(labels[-2:])
    if last_two in _COMPOUND_SUFFIXES:
        return None
    return last_two


def _mail_domain(value: str) -> str | None:
    """Domain part of a mailbox (last @); underscores allowed in the local part."""
    if value.count("@") != 1:
        return None
    _, _, domain = value.partition("@")
    return normalize_domain(domain)


# ---------------------------------------------------------------------------
# Domain extraction from email fields
# ---------------------------------------------------------------------------


def _mailbox_domains(values: list[str] | None, kind: str, location: str,
                     field: str) -> list[Indicator]:
    indicators: list[Indicator] = []
    for value in values or []:
        if "@" not in value:
            continue
        domain = _mail_domain(value)
        if domain is None:
            continue
        raw = domain
        normalized = normalize_domain(raw)
        if normalized is None:
            continue
        indicators.append(
            _make_indicator(kind, location, "domain", normalized, raw=raw, field=field)
        )
    return indicators


def _angle_addr_domain(value: str, kind: str, location: str, field: str) -> Indicator | None:
    if "<" in value and ">" in value:
        inner = value.split("<", 1)[1].split(">", 1)[0]
    else:
        inner = value
    domain = _mail_domain(inner.strip())
    if domain is None:
        return None
    return _make_indicator(kind, location, "domain", domain, field=field)


def _strip_angle(value: str) -> str:
    if "<" in value and ">" in value:
        return value.split("<", 1)[1].split(">", 1)[0]
    return value


def extract_domains_from_headers(
    raw_headers: dict[str, list[str]] | None,
) -> list[Indicator]:
    """Extract domain indicators from the allowlisted header fields."""

    indicators: list[Indicator] = []
    headers = raw_headers or {}
    for field in INDICATOR_HEADER_FIELDS:
        for value in headers.get(field, []):
            location = f"header:{field}"
            if field in {"From", "Reply-To", "Return-Path"}:
                indicator = _angle_addr_domain(value, "header_field", location, field)
                if indicator is not None:
                    indicators.append(indicator)
            elif field == "Message-ID":
                stripped = value.strip().strip("<>").strip()
                domain = _mail_domain(stripped) if stripped.count("@") == 1 else None
                if domain is not None:
                    indicators.append(
                        _make_indicator("header_field", location, "domain", domain, field=field)
                    )
            else:
                # Free-text fields: conservative mailbox token scan.
                for match in _ADDR_TOKEN.finditer(value):
                    domain = _mail_domain(match.group(0))
                    if domain is not None:
                        indicators.append(
                            _make_indicator(
                                "header_field", location, "domain", domain, field=field
                            )
                        )
    return indicators


def extract_domain_indicators(evidence: EmailEvidence) -> list[Indicator]:
    """Extract deduplicated domain indicators from structured email fields."""

    indicators: list[Indicator] = []
    if evidence.sender and evidence.sender.address:
        domain = _mail_domain(evidence.sender.address)
        if domain:
            indicators.append(
                _make_indicator("sender", "sender", "domain", domain, field="From")
            )
    indicators.extend(
        _mailbox_domains(
            [address.address for address in evidence.message.reply_to if address.address],
            "reply_to",
            "reply_to",
            "Reply-To",
        )
    )
    if evidence.message.return_path:
        indicator = _angle_addr_domain(
            evidence.message.return_path, "return_path", "return_path", "Return-Path"
        )
        if indicator:
            indicators.append(indicator)
    indicators.extend(extract_domains_from_headers(evidence.headers.raw))
    return _dedupe(indicators)


def iter_email_domains(evidence: EmailEvidence) -> list[str]:
    """Normalized domains from sender/reply-to/return-path/headers (no URLs).

    Order follows first occurrence in the email. Duplicates removed.
    """

    domains: list[str] = []
    for indicator in extract_domain_indicators(evidence):
        if indicator.type == "domain" and indicator.value not in domains:
            domains.append(indicator.value)
    return domains


# ---------------------------------------------------------------------------
# URL parsing (parsing only; URLs are never visited)
# ---------------------------------------------------------------------------


def parse_url_indicator(
    url: str,
    *,
    kind: str = "url",
    location: str = "body",
    field: str | None = None,
) -> Indicator:
    """Build one validated URL indicator from an already-extracted URL string."""

    if not isinstance(url, str) or not url:
        raise IndicatorExtractionError("A URL indicator requires a non-empty string.")
    scheme = urlsplit(url).scheme.casefold()
    if scheme not in {"http", "https"}:
        raise IndicatorExtractionError("Only http/https URL indicators are accepted.")
    return _make_indicator(kind, location, "url", url, field=field)


def extract_url_indicators(evidence: EmailEvidence) -> list[Indicator]:
    """Build URL indicators from the URLs already extracted by Step 1."""

    indicators: list[Indicator] = []
    for url_indicator in evidence.urls:
        try:
            indicators.append(
                parse_url_indicator(
                    url_indicator.url,
                    kind="url",
                    location=f"body.{url_indicator.source}",
                )
            )
        except IndicatorExtractionError:
            continue
    return indicators


# ---------------------------------------------------------------------------
# Total extraction entry point
# ---------------------------------------------------------------------------


def extract_indicators(evidence: EmailEvidence) -> list[Indicator]:
    """Extract all deduplicated indicators from normalized Step 1 evidence."""

    indicators: list[Indicator] = []
    indicators.extend(extract_ip_indicators(evidence))
    indicators.extend(extract_domain_indicators(evidence))
    indicators.extend(extract_url_indicators(evidence))
    return _dedupe(indicators)


# Re-exports for convenient, convention-following imports elsewhere.
__all__ = [
    "INDICATOR_HEADER_FIELDS",
    "Indicator",
    "IndicatorExtractionError",
    "IndicatorSource",
    "classify_ip",
    "extract_domain_indicators",
    "extract_domains_from_headers",
    "extract_indicators",
    "extract_ip_indicators",
    "extract_received_ips",
    "extract_url_indicators",
    "is_domain",
    "is_hostname",
    "is_ip_literal",
    "iter_email_domains",
    "normalize_domain",
    "parse_url_indicator",
    "registered_domain_of",
]
