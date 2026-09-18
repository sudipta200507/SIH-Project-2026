"""RDAP domain intelligence using the IANA bootstrap registry.

Safety and correctness properties:

- RDAP only: no WHOIS scraping, no arbitrary HTTP fetching. Requests go to
  endpoints discovered from the IANA RDAP bootstrap file
  (``https://data.iana.org/rdap/dns.json``) or to an explicitly configured
  endpoint;
- the outgoing URL is always built from ``<endpoint> + /domain/<validated
  domain>`` — user-supplied URLs never become requests;
- explicit timeouts and controlled handling of HTTP errors and rate limits
  (HTTP 429 and Retry-After);
- privacy-safe output: registrar and dates and roles only. Registrant names,
  e-mail addresses, and phone numbers are never copied into evidence;
- RDAP metadata is registration evidence only — it never proves anything
  about who controls the domain or whether it is malicious.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from app.intelligence.indicator_extractor import normalize_domain
from app.schemas.intelligence import Indicator, RDAPIntelligence

IANA_BOOTSTRAP_URL = "https://data.iana.org/rdap/dns.json"
DEFAULT_TIMEOUT_SECONDS = 10.0

class RDAPLookupError(Exception):
    """Raised for invalid inputs or misconfiguration of the RDAP module.

    ``kind`` carries the machine-readable failure category so the outer
    handler can report an explicit ``error_kind`` instead of collapsing every
    bootstrap/registry problem into a generic HTTP error.
    """

    def __init__(self, message: str, *, kind: str = "http_error") -> None:
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True, slots=True)
class RDAPSettings:
    """Bounded RDAP client configuration."""

    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    bootstrap_url: str = IANA_BOOTSTRAP_URL
    # Explicit override (e.g. a registry-specific RDAP service for offline
    # testing). When set, the bootstrap lookup is skipped entirely.
    endpoint_override: str | None = None


def _validate_target(domain: str) -> str:
    normalized = normalize_domain(domain) if isinstance(domain, str) else None
    if normalized is None:
        raise RDAPLookupError("RDAP target must be a validated domain name.")
    return normalized


def _fetch_bootstrap(settings: RDAPSettings, client: httpx.Client) -> dict[str, list[str]]:
    """Return a TLD -> RDAP endpoint list map from the IANA bootstrap file."""

    try:
        response = client.get(settings.bootstrap_url)
    except httpx.TimeoutException as error:
        raise RDAPLookupError("RDAP bootstrap lookup timed out.", kind="timeout") from error
    except httpx.TransportError as error:
        raise RDAPLookupError(
            f"RDAP bootstrap is unreachable: {type(error).__name__}.", kind="http_error"
        ) from error
    if response.status_code != 200:
        raise RDAPLookupError(
            f"RDAP bootstrap returned HTTP {response.status_code}.", kind="http_error"
        )
    try:
        payload = response.json()
    except ValueError as error:
        raise RDAPLookupError(
            "RDAP bootstrap response is not valid JSON.", kind="invalid_response"
        ) from error

    services = payload.get("services") if isinstance(payload, dict) else None
    if not isinstance(services, list):
        raise RDAPLookupError(
            "RDAP bootstrap response has an unexpected shape.", kind="invalid_response"
        )

    mapping: dict[str, list[str]] = {}
    for entry in services:
        if not isinstance(entry, list) or len(entry) != 2:
            continue
        tlds, endpoints = entry
        if not isinstance(tlds, list) or not isinstance(endpoints, list):
            continue
        for tld in tlds:
            if isinstance(tld, str) and endpoints:
                mapping[tld.casefold().lstrip(".")] = [
                    str(endpoint).rstrip("/") for endpoint in endpoints if isinstance(endpoint, str)
                ]
    return mapping


def _endpoint_for_domain(
    domain: str,
    settings: RDAPSettings,
    client: httpx.Client,
) -> str:
    if settings.endpoint_override:
        return settings.endpoint_override.rstrip("/")
    key = (settings.bootstrap_url, settings.timeout_seconds)
    if key in _BOOTSTRAP_CACHE:
        mapping = _BOOTSTRAP_CACHE[key]
    else:
        mapping = _fetch_bootstrap(settings, client)
        _BOOTSTRAP_CACHE[key] = mapping
    tld = domain.rsplit(".", 1)[-1]
    endpoints = mapping.get(tld)
    if not endpoints:
        raise RDAPLookupError(
            f"No RDAP endpoint is registered for the .{tld} registry.",
            kind="no_registry",
        )
    return endpoints[0].rstrip("/")


def _first_date(events: Any, *actions: str) -> str | None:
    """First RFC 3339 date for one of the given RDAP event actions."""

    if not isinstance(events, list):
        return None
    for event in events:
        if not isinstance(event, dict):
            continue
        if str(event.get("eventAction", "")).casefold() in actions:
            value = event.get("eventDate")
            if isinstance(value, str):
                return value
    return None


def _entity_roles(entities: Any) -> list[str]:
    """Collect entity roles while dropping all contact/PII details."""

    roles: list[str] = []
    if not isinstance(entities, list):
        return roles
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        for role in entity.get("roles") or []:
            if isinstance(role, str) and role not in roles:
                roles.append(role)
    return roles


def _registrar_name(entities: Any) -> str | None:
    """Registrar name only (role=registrar); never registrant contact data."""

    if not isinstance(entities, list):
        return None
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        roles = {str(role).casefold() for role in entity.get("roles") or []}
        if "registrar" not in roles:
            continue
        vcard = entity.get("vcardArray")
        if isinstance(vcard, list) and len(vcard) == 2 and isinstance(vcard[1], list):
            for entry in vcard[1]:
                if (
                    isinstance(entry, list)
                    and len(entry) >= 3
                    and str(entry[0]).casefold() == "fn"
                    and isinstance(entry[3], str)
                ):
                    return entry[3]
    return None


def _nameservers(objects: Any) -> list[str]:
    names: list[str] = []
    if not isinstance(objects, list):
        return names
    for item in objects:
        if isinstance(item, dict) and isinstance(item.get("ldhName"), str):
            name = item["ldhName"].strip().strip(".").casefold()
            if name and name not in names:
                names.append(name)
    return names


def _statuses(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(value) for value in values if isinstance(value, str)]


def _parse_domain_payload(payload: Any) -> dict[str, Any]:
    """Extract the privacy-safe subset of an RDAP domain response."""

    if not isinstance(payload, dict):
        raise RDAPLookupError("RDAP response is not a JSON object.")

    domain_name: str | None = None
    ldh = payload.get("ldhName")
    unicode_name = payload.get("unicodeName")
    if isinstance(ldh, str) and ldh:
        domain_name = ldh.strip().strip(".").casefold()
    elif isinstance(unicode_name, str) and unicode_name:
        domain_name = unicode_name.strip().strip(".")
    if domain_name is None:
        raise RDAPLookupError("RDAP response does not describe a domain object.")

    events = payload.get("events")
    return {
        "domain_name": domain_name,
        "registrar": _registrar_name(payload.get("entities")),
        "registration_date": _first_date(events, "registration", "created"),
        "expiration_date": _first_date(events, "expiration"),
        "last_changed_date": _first_date(events, "last changed", "last update of rdap database"),
        "nameservers": _nameservers(payload.get("nameservers")),
        "statuses": _statuses(payload.get("status")),
        "entity_roles": _entity_roles(payload.get("entities")),
    }


def _classify_http_error(status_code: int) -> str:
    if status_code == 404:
        return "no_record"
    if status_code == 429:
        return "rate_limited"
    return "http_error"


# Per-process bootstrap cache. The IANA file changes rarely; caching it keeps
# a multi-domain analysis to one bootstrap request. clear_bootstrap_cache()
# exists for tests.
_BOOTSTRAP_CACHE: dict[tuple[str, float], dict[str, list[str]]] = {}


def clear_bootstrap_cache() -> None:
    """Reset the in-process RDAP bootstrap cache."""

    _BOOTSTRAP_CACHE.clear()


def build_rdap_intelligence(
    indicator: Indicator,
    *,
    settings: RDAPSettings | None = None,
    transport: httpx.BaseTransport | None = None,
    bootstrap: dict[str, list[str]] | None = None,
) -> RDAPIntelligence:
    """Query RDAP for one validated domain indicator.

    ``transport`` and ``bootstrap`` are injection points for tests; production
    calls use the IANA bootstrap and the default httpx transport.
    """

    active_settings = settings or RDAPSettings()
    try:
        domain = _validate_target(indicator.value)
    except RDAPLookupError as error:
        return RDAPIntelligence(
            indicator=indicator,
            status="unavailable",
            error_kind="invalid_input",
            message=str(error),
        )

    try:
        with httpx.Client(
            timeout=active_settings.timeout_seconds,
            follow_redirects=False,
            transport=transport,
        ) as client:
            if bootstrap is not None:
                tld = domain.rsplit(".", 1)[-1]
                endpoints = bootstrap.get(tld)
                if not endpoints:
                    return RDAPIntelligence(
                        indicator=indicator,
                        status="unavailable",
                        error_kind="no_registry",
                        message=f"No RDAP endpoint is registered for the .{tld} registry.",
                    )
                endpoint = endpoints[0].rstrip("/")
            else:
                endpoint = _endpoint_for_domain(domain, active_settings, client)

            url = f"{endpoint}/domain/{domain}"
            try:
                response = client.get(url)
            except httpx.TimeoutException as error:
                return RDAPIntelligence(
                    indicator=indicator,
                    status="timeout",
                    endpoint=url,
                    error_kind="timeout",
                    message="The RDAP query timed out.",
                )
            except httpx.TransportError as error:
                return RDAPIntelligence(
                    indicator=indicator,
                    status="unavailable",
                    endpoint=url,
                    error_kind="http_error",
                    message=f"The RDAP endpoint is unreachable: {type(error).__name__}.",
                )

            if response.status_code != 200:
                kind = _classify_http_error(response.status_code)
                if kind == "no_record":
                    return RDAPIntelligence(
                        indicator=indicator,
                        status="no_record",
                        endpoint=url,
                        message="The registry has no RDAP record for this domain.",
                    )
                retry_after = response.headers.get("retry-after")
                message = f"The RDAP endpoint returned HTTP {response.status_code}."
                if kind == "rate_limited" and retry_after:
                    message += f" Retry-After: {retry_after}"
                return RDAPIntelligence(
                    indicator=indicator,
                    status="unavailable",
                    endpoint=url,
                    error_kind="http_error",
                    message=message,
                )

            try:
                payload = response.json()
            except ValueError as error:
                return RDAPIntelligence(
                    indicator=indicator,
                    status="unavailable",
                    endpoint=url,
                    error_kind="invalid_response",
                    message="The RDAP endpoint returned invalid JSON.",
                )

            try:
                parsed = _parse_domain_payload(payload)
            except RDAPLookupError as error:
                return RDAPIntelligence(
                    indicator=indicator,
                    status="unavailable",
                    endpoint=url,
                    error_kind="invalid_response",
                    message=str(error),
                )

            return RDAPIntelligence(
                indicator=indicator,
                status="success",
                endpoint=url,
                domain_name=parsed["domain_name"],
                registrar=parsed["registrar"],
                registration_date=parsed["registration_date"],
                expiration_date=parsed["expiration_date"],
                last_changed_date=parsed["last_changed_date"],
                nameservers=parsed["nameservers"],
                statuses=parsed["statuses"],
                entity_roles=parsed["entity_roles"],
            )
    except RDAPLookupError as error:
        return RDAPIntelligence(
            indicator=indicator,
            status="unavailable",
            error_kind=error.kind,  # type: ignore[arg-type]
            message=str(error),
        )


__all__ = [
    "IANA_BOOTSTRAP_URL",
    "RDAPLookupError",
    "RDAPSettings",
    "build_rdap_intelligence",
    "clear_bootstrap_cache",
]
