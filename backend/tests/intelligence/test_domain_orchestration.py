"""Domain intelligence combination tests (DNS + RDAP, mocked) and orchestrator correlation."""

from __future__ import annotations

import dns.resolver

from app.intelligence.dns import DNSSettings
from app.intelligence.domain_intelligence import build_domain_intelligence
from app.intelligence.orchestrator import (
    IntelligenceSettings,
    build_intelligence_evidence,
)

from fixtures import (
    FakeAnswer,
    FakeRdata,
    FakeResolver,
    make_indicator,
    rdap_domain_payload,
)


def _answer(*values: str) -> FakeAnswer:
    return FakeAnswer([FakeRdata(v) for v in values])


def _script_for(name: str) -> dict[tuple[str, str], object]:
    return {
        (name, "A"): _answer("93.184.216.34"),
        (name, "AAAA"): _answer("2606:2800:220:1:248:1893:25c8:1946"),
        (name, "MX"): _answer("0 mail.example.test."),
        (name, "TXT"): _answer("\"v=spf1 -all\""),
        (name, "NS"): _answer("ns1.example.test."),
        (name, "CNAME"): dns.resolver.NoAnswer(),
    }


# ---------------------------------------------------------------------------
# Domain intelligence (module 11 of the plan)
# ---------------------------------------------------------------------------


class TestDomainIntelligence:
    def test_combines_dns_and_rdap(self, monkeypatch) -> None:
        from app.intelligence import domain_intelligence as module

        monkeypatch.setattr(
            module,
            "build_rdap_intelligence",
            lambda indicator, **kwargs: _fake_rdap(indicator, "success"),
        )
        result = build_domain_intelligence(
            make_indicator("domain", "example.com"),
            resolver=FakeResolver(_script_for("example.com")),
        )
        assert result.normalized_domain == "example.com"
        assert result.dns.status == "success"
        assert result.rdap.status == "success"
        assert result.is_ip_in_disguise is False

    def test_dns_failure_keeps_rdap_and_vice_versa(self, monkeypatch) -> None:
        from app.intelligence import domain_intelligence as module

        monkeypatch.setattr(
            module,
            "build_rdap_intelligence",
            lambda indicator, **kwargs: _fake_rdap(indicator, "timeout"),
        )
        broken = {
            ("example.com", rdtype): dns.resolver.LifetimeTimeout()
            for rdtype in ("A", "AAAA", "MX", "TXT", "NS", "CNAME")
        }
        result = build_domain_intelligence(
            make_indicator("domain", "example.com"),
            resolver=FakeResolver(broken),
        )
        assert result.dns.status == "timeout"
        assert result.rdap.status == "timeout"

    def test_providers_can_be_disabled_independently(self, monkeypatch) -> None:
        from app.intelligence import domain_intelligence as module

        called = {"rdap": False}

        def _rdap_spy(indicator, **kwargs):
            called["rdap"] = True
            return _fake_rdap(indicator, "success")

        monkeypatch.setattr(module, "build_rdap_intelligence", _rdap_spy)
        result = build_domain_intelligence(
            make_indicator("domain", "example.com"),
            resolver=FakeResolver(_script_for("example.com")),
            perform_rdap=False,
        )
        assert called["rdap"] is False
        assert result.rdap is None
        assert result.dns.status == "success"

    def test_no_threat_verdict_ever_produced(self, monkeypatch) -> None:
        from app.intelligence import domain_intelligence as module

        monkeypatch.setattr(
            module,
            "build_rdap_intelligence",
            lambda indicator, **kwargs: _fake_rdap(indicator, "success"),
        )
        result = build_domain_intelligence(
            make_indicator("domain", "example.com"),
            resolver=FakeResolver(_script_for("example.com")),
        )
        exported = result.model_dump_json().casefold()
        for word in ("malicious", "threat", "risk", "score", "verdict", "safe", "unsafe"):
            assert word not in exported

    def test_invalid_domain_indicator_rejected(self) -> None:
        import pytest

        with pytest.raises(ValueError):
            build_domain_intelligence(make_indicator("domain", "not_a_domain"), perform_rdap=False)


def _fake_rdap(indicator, status: str):
    from app.schemas.intelligence import RDAPIntelligence

    return RDAPIntelligence(
        indicator=indicator,
        status=status,  # type: ignore[arg-type]
        registrar="EXAMPLE REGISTRAR" if status == "success" else None,
    )


# ---------------------------------------------------------------------------
# Orchestrator-level behavior
# ---------------------------------------------------------------------------


class TestOrchestrator:
    def test_counts_reflect_deduplicated_indicators(self, sample) -> None:
        result = build_intelligence_evidence(
            sample,
            settings=IntelligenceSettings(perform_dns=False, perform_rdap=False),
        )
        assert result.counts["indicators"] == len(result.indicators)
        assert result.counts["urls"] == len(result.urls)
        assert result.counts["ips"] == len(result.ips)
        assert result.counts["domains"] == len(result.domains)
        assert result.counts == {"indicators": 5, "urls": 2, "ips": 2, "domains": 1}

    def test_per_run_caps_enforced(self, sample) -> None:
        result = build_intelligence_evidence(
            sample,
            settings=IntelligenceSettings(
                perform_dns=False, perform_rdap=False, max_urls=1, max_domains=1
            ),
        )
        assert len(result.urls) == 1
        assert len(result.domains) == 1

    def test_full_run_with_fakes_produces_complete_picture(self, sample) -> None:
        result = build_intelligence_evidence(
            sample,
            settings=IntelligenceSettings(),
            resolver=FakeResolver(_script_for("example.test")),
            rdap_bootstrap={"test": ["https://rdap.iana.test/v1"]},
        )
        domain = result.domains[0]
        assert domain.normalized_domain == "example.test"
        # DNS answers from the injected script; RDAP fails against the fake
        # endpoint, which is exactly the graceful-degradation contract.
        assert domain.dns.status == "success"
        assert domain.dns.error_kind is None
        assert domain.rdap.status in {"unavailable", "timeout", "no_record"}
        assert domain.rdap.status != "success"

    def test_settings_defaults_enable_providers(self) -> None:
        settings = IntelligenceSettings()
        assert settings.perform_dns is True
        assert settings.perform_rdap is True
        assert settings.max_ips == 32
        assert settings.max_domains == 32
        assert settings.max_urls == 64
