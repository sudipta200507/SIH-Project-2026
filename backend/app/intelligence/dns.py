"""DNS intelligence via dnspython with explicit lookup states.

Safety properties:

- only validated hostnames/domains are ever resolved (see
  :func:`app.intelligence.indicator_extractor.is_hostname`);
- one bounded resolver with an explicit timeout and lifetime — no unbounded
  retry behavior;
- six record types maximum per name, each individually exception-safe;
- failure states (``timeout``/``error``/``unavailable``) are never collapsed
  into ``no_record``: a failed query and a confirmed-empty answer are
  different outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass

import dns.exception
import dns.resolver
from dns.rdatatype import from_text as rdtype_from_text

from app.intelligence.indicator_extractor import is_hostname
from app.schemas.intelligence import DNSIntelligence, DNSRecord, Indicator

RECORD_TYPES: tuple[str, ...] = ("A", "AAAA", "MX", "TXT", "NS", "CNAME")

DEFAULT_TIMEOUT_SECONDS = 3.0
DEFAULT_LIFETIME_SECONDS = 6.0


class DNSLookupError(Exception):
    """Raised for invalid inputs or misconfiguration of the DNS module."""


@dataclass(frozen=True, slots=True)
class DNSSettings:
    """Bounded DNS resolver configuration."""

    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    lifetime_seconds: float = DEFAULT_LIFETIME_SECONDS
    nameservers: tuple[str, ...] | None = None


def _build_resolver(settings: DNSSettings) -> dns.resolver.Resolver:
    """One resolver with explicit timeout and lifetime (no unbounded retries)."""

    resolver = dns.resolver.Resolver(configure=True)
    if settings.nameservers:
        resolver.nameservers = list(settings.nameservers)
    resolver.timeout = settings.timeout_seconds
    resolver.lifetime = settings.lifetime_seconds
    return resolver


def _record_value(rdata: object) -> str:
    """Render one answer record as plain text without parsing comments."""

    try:
        text = rdata.to_text()  # type: ignore[attr-defined]
    except Exception:
        text = str(rdata)
    return text.strip().strip('"')


def _query_one(
    resolver: dns.resolver.Resolver, name: str, record_type: str
) -> tuple[str, list[DNSRecord], str | None]:
    """Query one record type; return (status, records, error_kind).

    Statuses follow the shared vocabulary; ``no_records`` marks a successful
    answer whose answer section is empty and aggregates to ``no_record``.
    """

    try:
        answer = resolver.resolve(name, rdtype_from_text(record_type))
    except dns.resolver.NXDOMAIN:
        return "no_record", [], None
    except dns.resolver.NoAnswer:
        return "no_records", [], None
    except dns.resolver.NoNameservers:
        return "error", [], "no_nameservers"
    except (dns.resolver.LifetimeTimeout, dns.exception.Timeout):
        return "timeout", [], "timeout"
    except OSError:
        # Network-level failure (no route, no resolver reachable, ...).
        return "unavailable", [], "dns_error"
    except dns.exception.DNSException:
        return "error", [], "dns_error"
    except Exception:
        # Defensive: a provider bug degrades to an explicit error status and
        # must never propagate into (or destroy) the email analysis.
        return "error", [], "dns_error"

    records = [
        DNSRecord(type=record_type, value=_record_value(rdata))  # type: ignore[arg-type]
        for rdata in answer
    ]
    if records:
        return "success", records, None
    return "no_records", [], None


def _aggregate_status(
    per_type: dict[str, str],
    per_type_error_kinds: dict[str, str | None],
) -> tuple[str, str | None]:
    """Fold per-type statuses into one explicit aggregate (status, error_kind).

    Precedence: any success wins; otherwise timeout > unavailable > error;
    only confirmed-empty answers aggregate to ``no_record``. ``error_kind``
    preserves which concrete failure occurred instead of erasing it.
    """

    statuses = list(per_type.values())
    kinds = [kind for kind in per_type_error_kinds.values() if kind]
    if any(status == "success" for status in statuses):
        return "success", None
    if any(status == "timeout" for status in statuses):
        return "timeout", "timeout"
    if any(status == "unavailable" for status in statuses):
        return "unavailable", "dns_error"
    if any(status == "error" for status in statuses):
        # Keep the most specific concrete failure seen (no_nameservers first).
        if "no_nameservers" in kinds:
            return "error", "no_nameservers"
        return "error", "dns_error"
    # Only confirmed-empty answers remain: the name exists with no records.
    return "no_record", None


def validate_lookup_target(name: str) -> str:
    """Return the normalized lookup target or raise :class:`DNSLookupError`."""

    if not isinstance(name, str):
        raise DNSLookupError("DNS lookup target must be a string.")
    candidate = name.strip().strip(".").casefold()
    if candidate.endswith(".in-addr.arpa") or candidate.endswith(".ip6.arpa"):
        raise DNSLookupError("Reverse-DNS names are not accepted lookup targets.")
    if "%" in candidate or "/" in candidate or "@" in candidate:
        raise DNSLookupError("DNS lookup target contains forbidden characters.")
    if not is_hostname(candidate):
        raise DNSLookupError("DNS lookup target must be a validated hostname or domain.")
    return candidate


def build_dns_intelligence(
    indicator: Indicator,
    *,
    settings: DNSSettings | None = None,
    resolver: dns.resolver.Resolver | None = None,
    record_types: tuple[str, ...] = RECORD_TYPES,
) -> DNSIntelligence:
    """Resolve one validated domain/hostname indicator across record types.

    ``resolver`` may be injected for tests; otherwise a bounded resolver is
    built from ``settings``. Only validated names are ever sent to a resolver.
    """

    try:
        name = validate_lookup_target(indicator.value)
    except DNSLookupError as error:
        return DNSIntelligence(
            indicator=indicator,
            status="unavailable",
            error_kind="invalid_input",
            message=str(error),
        )

    unknown = [rt for rt in record_types if rt not in RECORD_TYPES]
    if unknown:
        raise DNSLookupError(f"Unsupported record types: {', '.join(unknown)}")

    active_settings = settings or DNSSettings()
    active_resolver = resolver or _build_resolver(active_settings)

    per_type_status: dict[str, str] = {}
    per_type_error_kinds: dict[str, str | None] = {}
    records: list[DNSRecord] = []
    for record_type in record_types:
        status, type_records, error_kind = _query_one(active_resolver, name, record_type)
        per_type_status[record_type] = status
        per_type_error_kinds[record_type] = error_kind
        records.extend(type_records)

    aggregate, error_kind = _aggregate_status(per_type_status, per_type_error_kinds)
    return DNSIntelligence(
        indicator=indicator,
        status=aggregate,  # type: ignore[arg-type]
        records=records,
        error_kind=error_kind,  # type: ignore[arg-type]
    )


__all__ = [
    "DNSLookupError",
    "DNSSettings",
    "RECORD_TYPES",
    "build_dns_intelligence",
    "validate_lookup_target",
]
