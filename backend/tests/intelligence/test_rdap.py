"""Tests 15-17 of the Step 4 plan: RDAP intelligence (fully mocked HTTP)."""

from __future__ import annotations

import json

import httpx
import pytest

from app.intelligence.rdap import (
    IANA_BOOTSTRAP_URL,
    RDAPLookupError,
    RDAPSettings,
    build_rdap_intelligence,
)

from fixtures import FakeTransport, make_indicator, rdap_domain_payload

BOOTSTRAP = {  # injected bootstrap (unit tests never fetch the real one)
    "com": ["https://rdap.verisign.com/com/v1"],
    "test": ["https://rdap.example.test/v1"],
}


def _domain_indicator(value: str = "example.com"):
    return make_indicator("domain", value, kind="url", location="body.html")


def _json_response(status_code: int, payload: object, headers: dict | None = None):
    import io

    content = json.dumps(payload).encode()
    return httpx.Response(
        status_code,
        content=content,
        headers=headers or {},
        request=httpx.Request("GET", "http://rdap.test/"),
    )


def _transport(handler) -> FakeTransport:
    return FakeTransport(handler)


def _ok_domain_handler(payload: object):
    def handler(request):
        url = str(request.url)
        if url == IANA_BOOTSTRAP_URL:
            return _json_response(
                200,
                {"services": [[["com", "test"], ["https://rdap.verisign.com/com/v1"]]]},
            )
        return _json_response(200, payload)

    return handler


# ---------------------------------------------------------------------------
# 15. RDAP success with mocked response
# ---------------------------------------------------------------------------


class TestRdapSuccess:
    def test_success_extracts_registration_metadata(self) -> None:
        payload = rdap_domain_payload()
        result = build_rdap_intelligence(
            _domain_indicator(),
            transport=_transport(_ok_domain_handler(payload)),
            bootstrap=BOOTSTRAP,
        )
        assert result.status == "success"
        assert result.domain_name == "example.com"
        assert result.registrar == "RESERVED-Internet Assigned Numbers Authority"
        assert result.registration_date == "1995-08-14T04:00:00Z"
        assert result.expiration_date == "2027-08-13T04:00:00Z"
        assert result.last_changed_date == "2025-08-13T04:00:00Z"
        assert result.nameservers == ["a.iana-servers.net", "b.iana-servers.net"]
        assert result.statuses == ["client transfer prohibited"]
        assert set(result.entity_roles) == {"registrar", "registrant"}

    def test_request_goes_to_known_endpoint_with_validated_domain(self) -> None:
        seen_urls: list[str] = []

        def handler(request):
            seen_urls.append(str(request.url))
            return _json_response(200, rdap_domain_payload())

        build_rdap_intelligence(
            _domain_indicator("Example.COM."),
            transport=_transport(handler),
            bootstrap=BOOTSTRAP,
        )
        domain_calls = [u for u in seen_urls if "/domain/" in u]
        assert domain_calls == ["https://rdap.verisign.com/com/v1/domain/example.com"]

    def test_registrant_pii_never_leaks_into_evidence(self) -> None:
        result = build_rdap_intelligence(
            _domain_indicator(),
            transport=_transport(_ok_domain_handler(rdap_domain_payload())),
            bootstrap=BOOTSTRAP,
        )
        exported = result.model_dump_json().casefold()
        assert "hidden@example.com" not in exported
        assert "redacted for privacy" not in exported
        assert "+1.5555555555" not in exported

    def test_no_verdict_language_in_evidence(self) -> None:
        result = build_rdap_intelligence(
            _domain_indicator(),
            transport=_transport(_ok_domain_handler(rdap_domain_payload())),
            bootstrap=BOOTSTRAP,
        )
        exported = result.model_dump_json().casefold()
        for word in ("malicious", "threat", "risk", "verdict", "score"):
            assert word not in exported

    def test_missing_optional_fields_tolerated(self) -> None:
        payload = {"objectClassName": "domain", "ldhName": "EXAMPLE.COM"}
        result = build_rdap_intelligence(
            _domain_indicator(),
            transport=_transport(_ok_domain_handler(payload)),
            bootstrap=BOOTSTRAP,
        )
        assert result.status == "success"
        assert result.registrar is None
        assert result.nameservers == []


# ---------------------------------------------------------------------------
# 16. RDAP HTTP failure
# ---------------------------------------------------------------------------


class TestRdapHttpFailure:
    def test_404_is_no_record(self) -> None:
        def handler(request):
            if str(request.url) == IANA_BOOTSTRAP_URL:
                return _json_response(200, {"services": [[["com"], ["https://rdap.verisign.com/com/v1"]]]})
            return _json_response(404, {"errorCode": 404})

        result = build_rdap_intelligence(
            _domain_indicator(), transport=_transport(handler), bootstrap=BOOTSTRAP
        )
        assert result.status == "no_record"
        assert result.error_kind is None

    def test_429_rate_limit_records_retry_after(self) -> None:
        def handler(request):
            return _json_response(429, {"errorCode": 429}, headers={"Retry-After": "60"})

        result = build_rdap_intelligence(
            _domain_indicator(), transport=_transport(handler), bootstrap=BOOTSTRAP
        )
        assert result.status == "unavailable"
        assert result.error_kind == "http_error"
        assert "429" in (result.message or "")
        assert "60" in (result.message or "")

    def test_500_is_http_error_not_no_record(self) -> None:
        def handler(request):
            return _json_response(500, "boom")

        result = build_rdap_intelligence(
            _domain_indicator(), transport=_transport(handler), bootstrap=BOOTSTRAP
        )
        assert result.status == "unavailable"
        assert result.error_kind == "http_error"
        assert result.status != "no_record"

    def test_transport_error_is_unavailable(self) -> None:
        def handler(request):
            raise httpx.ConnectError("connection refused")

        result = build_rdap_intelligence(
            _domain_indicator(), transport=_transport(handler), bootstrap=BOOTSTRAP
        )
        assert result.status == "unavailable"
        assert result.error_kind == "http_error"

    def test_invalid_json_is_invalid_response(self) -> None:
        def handler(request):
            return httpx.Response(
                200,
                content=b"not json at all",
                request=httpx.Request("GET", "http://rdap.test/"),
            )

        result = build_rdap_intelligence(
            _domain_indicator(), transport=_transport(handler), bootstrap=BOOTSTRAP
        )
        assert result.status == "unavailable"
        assert result.error_kind == "invalid_response"

    def test_non_domain_object_is_invalid_response(self) -> None:
        def handler(request):
            return _json_response(200, {"objectClassName": "nameserver"})

        result = build_rdap_intelligence(
            _domain_indicator(), transport=_transport(handler), bootstrap=BOOTSTRAP
        )
        assert result.status == "unavailable"
        assert result.error_kind == "invalid_response"

    def test_unknown_tld_is_no_registry(self) -> None:
        result = build_rdap_intelligence(
            _domain_indicator("example.unregistertld"),
            transport=_transport(lambda request: _json_response(200, {})),
            bootstrap=BOOTSTRAP,
        )
        assert result.status == "unavailable"
        assert result.error_kind == "no_registry"

    def test_bootstrap_failure_is_unavailable(self) -> None:
        def handler(request):
            if str(request.url) == IANA_BOOTSTRAP_URL:
                raise httpx.ConnectError("dns failure")
            return _json_response(200, rdap_domain_payload())

        result = build_rdap_intelligence(
            _domain_indicator(), transport=_transport(handler)
        )
        assert result.status == "unavailable"
        assert result.error_kind == "http_error"


# ---------------------------------------------------------------------------
# 17. RDAP timeout
# ---------------------------------------------------------------------------


class TestRdapTimeout:
    def test_timeout_status(self) -> None:
        def handler(request):
            raise httpx.ReadTimeout("timed out")

        result = build_rdap_intelligence(
            _domain_indicator(), transport=_transport(handler), bootstrap=BOOTSTRAP
        )
        assert result.status == "timeout"
        assert result.error_kind == "timeout"

    def test_settings_timeout_value_respected(self) -> None:
        settings = RDAPSettings(timeout_seconds=2.5)
        assert settings.timeout_seconds == 2.5


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestRdapValidation:
    def test_invalid_domain_is_unavailable_invalid_input(self) -> None:
        result = build_rdap_intelligence(
            make_indicator("domain", "not_a_domain"), bootstrap=BOOTSTRAP
        )
        assert result.status == "unavailable"
        assert result.error_kind == "invalid_input"

    def test_error_kind_vocabulary(self) -> None:
        assert RDAPLookupError("x", kind="no_registry").kind == "no_registry"
