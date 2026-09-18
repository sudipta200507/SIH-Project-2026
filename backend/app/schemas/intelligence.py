"""Pydantic models for Step 4 infrastructure intelligence evidence.

Every external lookup carries an explicit ``status``. The vocabulary keeps two
concepts strictly separate:

- ``unavailable`` / ``timeout`` / ``error`` — the lookup itself did not answer
  (failure to query is never reported as "no record");
- ``no_record`` — the lookup answered and confirmed nothing exists.

The layer produces infrastructure evidence only. It never assigns a threat
verdict, risk score, or confidence value; those belong to the later AI phase.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Shared status vocabularies (failure is never conflated with absence).
# ---------------------------------------------------------------------------

LookupStatus = Literal["success", "no_record", "unavailable", "timeout", "error"]

IpVersion = Literal[4, 6]

IpClassification = Literal[
    "public",
    "private",
    "loopback",
    "link_local",
    "reserved",
    "multicast",
    "unspecified",
]

SourceKind = Literal[
    "received_header",
    "header_field",
    "sender",
    "reply_to",
    "return_path",
    "message_id",
    "url",
]


class EvidenceModel(BaseModel):
    """Base model that keeps the exported evidence shape predictable."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class IndicatorSource(EvidenceModel):
    """Provenance of one indicator: where it came from in the original email.

    ``location`` is a stable, human-readable reference into the Step 1
    evidence (for example ``received_header[2]`` or ``body.html``). It is
    never a verdict about the indicator itself.
    """

    kind: SourceKind
    location: str
    field: str | None = None


class Indicator(EvidenceModel):
    """One validated candidate indicator with its origin provenance.

    ``value`` is the normalized form; ``raw`` preserves the original string
    when it differed from the normalized value. Extraction says nothing about
    attacker attribution: an IP from the Received chain is a candidate IP only.
    """

    type: Literal["ipv4", "ipv6", "hostname", "domain", "url"]
    value: str
    raw: str | None = None
    source: IndicatorSource


class GeoLocation(EvidenceModel):
    """Approximate GeoIP facts for one IP; never a physical-location claim."""

    indicator: Indicator
    available: bool
    reason: (
        Literal["not_configured", "lookup_failed", "not_found", "invalid_input"]
        | None
    ) = None
    database_type: Literal["local", "service"] | None = None
    ip_version: IpVersion | None = None
    classification: IpClassification | None = None
    country: str | None = None
    region: str | None = None
    city: str | None = None
    asn: int | None = Field(default=None, ge=0)
    asn_organization: str | None = None
    network: str | None = None


class IPIntelligence(EvidenceModel):
    """Normalized IP facts: classification plus optional approximate geolocation.

    Classification is a local property of the address; ``geolocation`` carries
    MaxMind facts only when GeoIP is configured. Neither is a reputation
    verdict.
    """

    indicator: Indicator
    ip_version: IpVersion
    classification: IpClassification
    is_global: bool
    geolocation: GeoLocation | None = None


class DNSRecord(EvidenceModel):
    """One DNS answer record, stored as plain text without comment parsing."""

    type: Literal["A", "AAAA", "MX", "TXT", "NS", "CNAME"]
    value: str


class DNSIntelligence(EvidenceModel):
    """Explicit-state DNS evidence for one validated name."""

    indicator: Indicator
    status: LookupStatus
    records: list[DNSRecord] = Field(default_factory=list)
    error_kind: (
        Literal["timeout", "no_nameservers", "dns_error", "lifecycle_error", "invalid_input"]
        | None
    ) = None
    message: str | None = None


class RDAPIntelligence(EvidenceModel):
    """Explicit-state RDAP registration metadata (privacy-safe subset)."""

    indicator: Indicator
    status: LookupStatus
    endpoint: str | None = None
    domain_name: str | None = None
    registrar: str | None = None
    registration_date: str | None = None
    expiration_date: str | None = None
    last_changed_date: str | None = None
    nameservers: list[str] = Field(default_factory=list)
    statuses: list[str] = Field(default_factory=list)
    entity_roles: list[str] = Field(default_factory=list)
    error_kind: (
        Literal["http_error", "timeout", "invalid_response", "invalid_input", "no_registry"]
        | None
    ) = None
    message: str | None = None


class URLIntelligence(EvidenceModel):
    """Structured URL decomposition (parsing only — URLs are never visited)."""

    indicator: Indicator
    scheme: str
    hostname: str | None = None
    port: int | None = Field(default=None, ge=0, le=65535)
    path: str | None = None
    query: str | None = None
    fragment: str | None = None
    normalized_hostname: str | None = None
    registered_domain: str | None = None
    ip_in_host: IpVersion | None = None
    parse_error: str | None = None


class DomainIntelligence(EvidenceModel):
    """Combined normalized domain + DNS + RDAP evidence; never a verdict."""

    indicator: Indicator
    normalized_domain: str
    is_ip_in_disguise: bool = False
    dns: DNSIntelligence | None = None
    rdap: RDAPIntelligence | None = None


class IntelligenceEvidence(EvidenceModel):
    """Step 4 output: infrastructure evidence with provenance, never a verdict.

    ``counts`` is a summary of deduplicated indicator volumes so the combined
    API response can state what was enriched without exposing any scoring.
    """

    schema_version: Literal["1.0"] = "1.0"
    indicators: list[Indicator] = Field(default_factory=list)
    urls: list[URLIntelligence] = Field(default_factory=list)
    ips: list[IPIntelligence] = Field(default_factory=list)
    domains: list[DomainIntelligence] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)
