"""Shared fixtures and fakes for the Step 4 intelligence tests.

External services (DNS resolvers, RDAP endpoints, GeoIP readers) are always
faked in unit tests; no unit test touches the network. This module is named
``fixtures`` (not ``helpers``) to avoid a module-name clash with
``tests/api/helpers.py`` under pytest's flat test layout.
"""

from __future__ import annotations

from pathlib import Path

from app.extractor.email_parser import extract_email_from_bytes
from app.schemas.email import EmailEvidence
from app.schemas.intelligence import Indicator, IndicatorSource

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SAMPLE_PATH = PROJECT_ROOT / "samples" / "safe" / "step1_synthetic.eml"


def sample_evidence() -> EmailEvidence:
    """Normalized Step 1 evidence for the safe synthetic sample."""

    return extract_email_from_bytes(SAMPLE_PATH.read_bytes(), filename="step1_synthetic.eml")


def make_indicator(
    type_: str,
    value: str,
    *,
    kind: str = "sender",
    location: str = "sender",
) -> Indicator:
    """Build a minimal valid indicator for module-level tests."""

    return Indicator(
        type=type_,  # type: ignore[arg-type]
        value=value,
        source=IndicatorSource(kind=kind, location=location),  # type: ignore[arg-type]
    )


class FakeResolver:
    """Scriptable stand-in for dns.resolver.Resolver.

    Script keys use the textual record type ("A", "AAAA", ...). dnspython
    passes ``rdtype`` as an int/enum, so it is normalized before lookup.
    """

    def __init__(self, script: dict[tuple[str, str], object] | None = None) -> None:
        self.script = script or {}
        self.queries: list[tuple[str, str]] = []

    def resolve(self, name: str, rdtype: object) -> object:
        import dns.rdatatype

        if isinstance(rdtype, int):
            rdtype_text = dns.rdatatype.to_text(rdtype).upper()
        else:
            rdtype_text = str(rdtype).upper()
        self.queries.append((name, rdtype_text))
        result = self.script.get((name, rdtype_text))
        if isinstance(result, Exception):
            raise result
        if result is None:
            import dns.resolver

            raise dns.resolver.NXDOMAIN()
        return result


class FakeAnswer:
    """Minimal dnspython answer: iterable of rdata objects."""

    def __init__(self, values: list[str]) -> None:
        self._values = values

    def __iter__(self):
        return iter(self._values)


class FakeRdata:
    """Minimal rdata object with to_text()."""

    def __init__(self, text: str) -> None:
        self._text = text

    def to_text(self) -> str:
        return self._text


class FakeTransport:
    """Scriptable httpx transport for RDAP tests (no real network).

    Implements the transport context-manager protocol httpx.Client expects.
    """

    def __init__(self, handler) -> None:
        self._handler = handler
        self.requests: list[str] = []

    def handle_request(self, request):
        self.requests.append(str(request.url))
        return self._handler(request)

    def __enter__(self) -> "FakeTransport":
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def close(self) -> None:
        return None


class FakeCityReader:
    """Scriptable geoip2-like city reader."""

    def __init__(self, records: dict[str, object] | None = None) -> None:
        self.records = records or {}
        self.queries: list[str] = []

    def get(self, ip: str) -> object | None:
        self.queries.append(ip)
        return self.records.get(ip)


class FakeAsnReader:
    """Scriptable geoip2-like ASN reader."""

    def __init__(self, records: dict[str, object] | None = None) -> None:
        self.records = records or {}
        self.queries: list[str] = []

    def get(self, ip: str) -> object | None:
        self.queries.append(ip)
        return self.records.get(ip)


class FakeRecord:
    """Minimal geoip2 city record stand-in."""

    def __init__(
        self,
        *,
        country: str | None = None,
        region: str | None = None,
        city: str | None = None,
        network: str | None = None,
    ) -> None:
        self.country = _NameHolder(country)
        self.city = _NameHolder(city)

        class _Subs:
            def __init__(self, name: str | None) -> None:
                self.most_specific = _NameHolder(name)

        self.subdivisions = _Subs(region)
        self.network = network


class _NameHolder:
    def __init__(self, name: str | None) -> None:
        self.name = name


class FakeAsnRecord:
    """Minimal geoip2 ASN record stand-in."""

    def __init__(self, asn: int | None = None, organization: str | None = None) -> None:
        self.autonomous_system_number = asn
        self.autonomous_system_organization = organization


def rdap_domain_payload(**overrides: object) -> dict:
    """A realistic privacy-safe RDAP domain response (IANA reserved shape)."""

    payload = {
        "objectClassName": "domain",
        "ldhName": "EXAMPLE.COM",
        "status": ["client transfer prohibited"],
        "events": [
            {"eventAction": "registration", "eventDate": "1995-08-14T04:00:00Z"},
            {"eventAction": "expiration", "eventDate": "2027-08-13T04:00:00Z"},
            {"eventAction": "last changed", "eventDate": "2025-08-13T04:00:00Z"},
        ],
        "nameservers": [
            {"ldhName": "a.iana-servers.net"},
            {"ldhName": "B.IANA-SERVERS.NET"},
        ],
        "entities": [
            {
                "roles": ["registrar"],
                "vcardArray": [
                    "vcard",
                    [
                        ["version", {}, "text", "4.0"],
                        ["fn", {}, "text", "RESERVED-Internet Assigned Numbers Authority"],
                    ],
                ],
            },
            {
                # Registrant contact data must never leak into evidence.
                "roles": ["registrant"],
                "vcardArray": [
                    "vcard",
                    [
                        ["fn", {}, "text", "REDACTED FOR PRIVACY"],
                        ["email", {}, "text", "hidden@example.com"],
                        ["tel", {}, "text", "+1.5555555555"],
                    ],
                ],
            },
        ],
    }
    payload.update(overrides)
    return payload


__all__ = [
    "FakeAnswer",
    "FakeAsnReader",
    "FakeAsnRecord",
    "FakeCityReader",
    "FakeRdata",
    "FakeResolver",
    "FakeTransport",
    "SAMPLE_PATH",
    "make_indicator",
    "rdap_domain_payload",
    "sample_evidence",
]
