"""Tests 18-19 of the Step 4 plan: GeoIP availability, mocking, degradation."""

from __future__ import annotations

import pytest

from app.intelligence.geolocation import (
    GeoIPSettings,
    build_geolocation,
    load_geoip_settings,
)

from fixtures import FakeAsnReader, FakeAsnRecord, FakeCityReader, FakeRecord, make_indicator


def _ip_indicator(value: str = "8.8.8.8"):
    return make_indicator("ipv4", value, kind="received_header", location="received_header[0]")


# ---------------------------------------------------------------------------
# 18. GeoIP unavailable
# ---------------------------------------------------------------------------


class TestGeoIPUnavailable:
    def test_no_configuration_reports_not_configured(self) -> None:
        result = build_geolocation(_ip_indicator(), settings=GeoIPSettings(mode="off"))
        assert result.available is False
        assert result.reason == "not_configured"
        assert result.country is None

    def test_default_environment_without_maxmind_vars(self, monkeypatch) -> None:
        for name in (
            "MAXMIND_DB_PATH",
            "MAXMIND_ASN_DB_PATH",
            "MAXMIND_SERVICE",
            "MAXMIND_ACCOUNT_ID",
            "MAXMIND_LICENSE_KEY",
        ):
            monkeypatch.delenv(name, raising=False)
        settings = load_geoip_settings()
        assert settings.mode == "off"
        result = build_geolocation(_ip_indicator(), settings=settings)
        assert result.available is False
        assert result.reason == "not_configured"

    def test_service_mode_degrades_gracefully(self) -> None:
        # Service mode is accepted by configuration but not implemented in
        # this step: it must degrade, never fail the analysis.
        settings = GeoIPSettings(mode="service", account_id="a", license_key="k")
        result = build_geolocation(_ip_indicator(), settings=settings)
        assert result.available is False
        assert result.reason == "not_configured"

    def test_invalid_ip_indicator(self) -> None:
        result = build_geolocation(make_indicator("ipv4", "999.999.1.1"), settings=GeoIPSettings(mode="local", db_path="x"))
        assert result.available is False
        assert result.reason == "invalid_input"

    def test_reader_error_is_lookup_failed_not_crash(self) -> None:
        class BrokenReader:
            def get(self, ip: str) -> object:
                raise RuntimeError("corrupt database")

        result = build_geolocation(
            _ip_indicator(),
            settings=GeoIPSettings(mode="local", db_path="/tmp/x.mmdb"),
            city_reader=BrokenReader(),
        )
        assert result.available is False
        assert result.reason == "lookup_failed"

    def test_missing_database_file_is_lookup_failed(self, tmp_path) -> None:
        result = build_geolocation(
            _ip_indicator(),
            settings=GeoIPSettings(mode="local", db_path=str(tmp_path / "missing.mmdb")),
        )
        assert result.available is False
        assert result.reason == "lookup_failed"


# ---------------------------------------------------------------------------
# 19. GeoIP configured with mocked database/service
# ---------------------------------------------------------------------------


class TestGeoIPConfigured:
    def test_city_and_asn_data_extracted(self) -> None:
        city = FakeCityReader(
            {"8.8.8.8": FakeRecord(country="United States", region="Virginia", city="Ashburn", network="8.8.8.0/24")}
        )
        asn = FakeAsnReader({"8.8.8.8": FakeAsnRecord(asn=15169, organization="GOOGLE")})
        result = build_geolocation(
            _ip_indicator(),
            settings=GeoIPSettings(mode="local", db_path="unused"),
            city_reader=city,
            asn_reader=asn,
        )
        assert result.available is True
        assert result.database_type == "local"
        assert result.country == "United States"
        assert result.region == "Virginia"
        assert result.city == "Ashburn"
        assert result.asn == 15169
        assert result.asn_organization == "GOOGLE"
        assert result.network == "8.8.8.0/24"
        assert city.queries == ["8.8.8.8"]
        assert asn.queries == ["8.8.8.8"]

    def test_ip_not_in_database_is_not_found(self) -> None:
        result = build_geolocation(
            _ip_indicator("203.0.113.77"),
            settings=GeoIPSettings(mode="local", db_path="unused"),
            city_reader=FakeCityReader(),
        )
        assert result.available is False
        assert result.reason == "not_found"

    def test_unknown_country_values_stay_none(self) -> None:
        result = build_geolocation(
            _ip_indicator(),
            settings=GeoIPSettings(mode="local", db_path="unused"),
            city_reader=FakeCityReader({"8.8.8.8": FakeRecord()}),
        )
        assert result.available is True
        assert result.country is None
        assert result.region is None
        assert result.city is None

    def test_no_exact_location_claim_in_evidence(self) -> None:
        result = build_geolocation(
            _ip_indicator(),
            settings=GeoIPSettings(mode="local", db_path="unused"),
            city_reader=FakeCityReader(
                {"8.8.8.8": FakeRecord(country="United States", city="Ashburn")}
            ),
        )
        exported = result.model_dump_json().casefold()
        assert "latitude" not in exported
        assert "longitude" not in exported
        assert "exact" not in exported

    def test_asn_reader_failure_does_not_break_city_result(self) -> None:
        class BrokenAsn:
            def get(self, ip: str) -> object:
                raise RuntimeError("asn db corrupt")

        result = build_geolocation(
            _ip_indicator(),
            settings=GeoIPSettings(mode="local", db_path="unused"),
            city_reader=FakeCityReader({"8.8.8.8": FakeRecord(country="United States")}),
            asn_reader=BrokenAsn(),
        )
        assert result.available is True
        assert result.country == "United States"
        assert result.asn is None

    def test_environment_loaded_local_mode(self, monkeypatch, tmp_path) -> None:
        db = tmp_path / "GeoLite2-City.mmdb"
        db.write_bytes(b"x")
        monkeypatch.setenv("MAXMIND_DB_PATH", str(db))
        settings = load_geoip_settings()
        assert settings.mode == "local"
        assert settings.db_path == str(db)
