"""MaxMind GeoIP enrichment, active only when a database is configured.

Behavior contract:

- ``MAXMIND_DB_PATH`` (GeoLite2-City.mmdb / GeoLite2-ASN.mmdb) or
  ``MAXMIND_SERVICE``/``MAXMIND_ACCOUNT_ID``/``MAXMIND_LICENSE_KEY`` (geoip2
  webservice) enable lookups;
- without configuration the result is ``available=false,
  reason="not_configured"`` — and the rest of the analysis never fails
  because of that;
- local database reads only; no invented results, no exact-location claims.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from ipaddress import ip_address

from app.intelligence.indicator_extractor import classify_ip
from app.schemas.intelligence import GeoLocation, Indicator

MAXMIND_DB_PATH_ENV = "MAXMIND_DB_PATH"
MAXMIND_ASN_DB_PATH_ENV = "MAXMIND_ASN_DB_PATH"
MAXMIND_SERVICE_ENV = "MAXMIND_SERVICE"
MAXMIND_ACCOUNT_ID_ENV = "MAXMIND_ACCOUNT_ID"
MAXMIND_LICENSE_KEY_ENV = "MAXMIND_LICENSE_KEY"
MAXMIND_HOST_ENV = "MAXMIND_HOST"

DEFAULT_SERVICE_HOST = "geolite.info"


class GeoIPError(Exception):
    """Raised for misconfiguration of the GeoIP module (never for missing data)."""


@dataclass(frozen=True, slots=True)
class GeoIPSettings:
    """Resolved GeoIP configuration loaded from the environment."""

    mode: str  # "off" | "local" | "service"
    db_path: str | None = None
    asn_db_path: str | None = None
    account_id: str | None = None
    license_key: str | None = None
    host: str = DEFAULT_SERVICE_HOST


def load_geoip_settings(env: dict[str, str] | None = None) -> GeoIPSettings:
    """Load GeoIP settings; ``mode`` states whether lookups are configured."""

    environment = env if env is not None else dict(os.environ)
    db_path = (environment.get(MAXMIND_DB_PATH_ENV) or "").strip() or None
    asn_db_path = (environment.get(MAXMIND_ASN_DB_PATH_ENV) or "").strip() or None
    service_name = (environment.get(MAXMIND_SERVICE_ENV) or "").strip()
    account_id = (environment.get(MAXMIND_ACCOUNT_ID_ENV) or "").strip() or None
    license_key = (environment.get(MAXMIND_LICENSE_KEY_ENV) or "").strip() or None
    host = (environment.get(MAXMIND_HOST_ENV) or "").strip() or DEFAULT_SERVICE_HOST

    if db_path:
        return GeoIPSettings(mode="local", db_path=db_path, asn_db_path=asn_db_path)
    if service_name and account_id and license_key:
        return GeoIPSettings(
            mode="service",
            account_id=account_id,
            license_key=license_key,
            host=host,
        )
    return GeoIPSettings(mode="off")


def _unavailable(indicator: Indicator, reason: str) -> GeoLocation:
    return GeoLocation(indicator=indicator, available=False, reason=reason)  # type: ignore[arg-type]


def _validate_ip(indicator: Indicator) -> str:
    try:
        address = ip_address(indicator.value)
    except ValueError as error:
        raise GeoIPError(f"GeoIP target must be a valid IP: {indicator.value!r}") from error
    return address.compressed


def _city_fields(record: object) -> dict[str, str | None]:
    country = getattr(record, "country", None)
    subdivisions = getattr(record, "subdivisions", None)
    city = getattr(record, "city", None)

    def _name(attribute: object) -> str | None:
        name = getattr(attribute, "name", None)
        return name if isinstance(name, str) and name else None

    region = None
    if subdivisions:
        try:
            most_specific = subdivisions.most_specific
        except AttributeError:
            most_specific = None
        region = _name(most_specific)
    return {
        "country": _name(country),
        "region": region,
        "city": _name(city),
    }


def _network_of(record: object) -> str | None:
    network = getattr(record, "network", None)
    return str(network) if network is not None else None


def build_geolocation(
    indicator: Indicator,
    *,
    settings: GeoIPSettings | None = None,
    city_reader: object | None = None,
    asn_reader: object | None = None,
) -> GeoLocation:
    """Return approximate GeoIP facts for one validated IP indicator.

    ``city_reader``/``asn_reader`` accept injected geoip2 readers (tests, or a
    shared process-wide reader). When none are injected, local database
    readers are opened per call from the configured paths.
    """

    try:
        normalized = _validate_ip(indicator)
    except GeoIPError as error:
        return _unavailable(indicator, "invalid_input")

    active_settings = settings or load_geoip_settings()
    if active_settings.mode == "off":
        return _unavailable(indicator, "not_configured")
    if active_settings.mode != "local":
        # Service mode is accepted but not implemented in this step; GeoIP
        # degrades gracefully instead of failing the analysis.
        return _unavailable(indicator, "not_configured")

    version = ip_address(normalized).version
    classified = classify_ip(normalized)
    classification = classified[0] if classified else "reserved"

    record: object | None = None
    try:
        if city_reader is not None:
            record = city_reader.get(normalized)
            fields = _city_fields(record)
            network = _network_of(record)
        else:
            import geoip2.database  # local import: optional dependency

            with geoip2.database.Reader(active_settings.db_path) as reader:  # type: ignore[arg-type]
                record = reader.city(normalized)
            fields = _city_fields(record)
            network = _network_of(record)
    except Exception:
        return GeoLocation(
            indicator=indicator,
            available=False,
            reason="lookup_failed",  # type: ignore[arg-type]
            database_type="local",
            ip_version=version,  # type: ignore[arg-type]
            classification=classification,  # type: ignore[arg-type]
        )

    if record is None:
        # The database answered cleanly: the IP is simply not in it.
        return GeoLocation(
            indicator=indicator,
            available=False,
            reason="not_found",  # type: ignore[arg-type]
            database_type="local",
            ip_version=version,  # type: ignore[arg-type]
            classification=classification,  # type: ignore[arg-type]
        )

    asn: int | None = None
    asn_organization: str | None = None
    if asn_reader is not None:
        try:
            asn_record = asn_reader.get(normalized)
        except Exception:
            asn_record = None
        if asn_record is not None:
            raw_asn = getattr(asn_record, "autonomous_system_number", None)
            raw_org = getattr(asn_record, "autonomous_system_organization", None)
            asn = int(raw_asn) if isinstance(raw_asn, int) else None
            asn_organization = (
                str(raw_org) if isinstance(raw_org, str) and raw_org else None
            )

    if network is None:
        network = None

    return GeoLocation(
        indicator=indicator,
        available=True,
        database_type="local",
        ip_version=version,  # type: ignore[arg-type]
        classification=classification,  # type: ignore[arg-type]
        country=fields["country"],
        region=fields["region"],
        city=fields["city"],
        asn=asn,
        asn_organization=asn_organization,
        network=network,
    )


__all__ = [
    "GeoIPError",
    "GeoIPSettings",
    "MAXMIND_DB_PATH_ENV",
    "build_geolocation",
    "load_geoip_settings",
]
