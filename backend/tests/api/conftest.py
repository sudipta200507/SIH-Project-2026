"""Shared fixtures for the Step 3 API tests.

Rspamd is always faked in unit tests via FastAPI dependency overrides; one
integration test file uses the real Dockerized container and skips otherwise.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api import dependencies
from app.core.config import IntelligenceEnvSettings
from app.main import app as fastapi_app

from helpers import SAMPLE_EML, FakeRspamdClient, valid_analysis  # noqa: F401


__all__ = ["SAMPLE_EML", "FakeRspamdClient", "valid_analysis"]


@pytest.fixture(autouse=True)
def offline_intelligence():
    """Keep Step 4 lookups out of the API unit tests.

    The intelligence providers (DNS, RDAP) are real network clients, so every
    API test runs with them disabled unless it overrides settings itself.
    """

    fastapi_app.dependency_overrides[dependencies.get_intelligence_settings] = lambda: (
        IntelligenceEnvSettings(perform_dns=False, perform_rdap=False)
    )
    yield
    fastapi_app.dependency_overrides.pop(dependencies.get_intelligence_settings, None)


@pytest.fixture()
def client() -> TestClient:
    return TestClient(fastapi_app)


@pytest.fixture()
def fake_rspamd():
    """Install a FakeRspamdClient and expose it for assertions."""

    fake = FakeRspamdClient(analysis=valid_analysis())
    fastapi_app.dependency_overrides[dependencies.get_rspamd_client] = lambda: fake
    yield fake
    fastapi_app.dependency_overrides.pop(dependencies.get_rspamd_client, None)


@pytest.fixture()
def failing_rspamd():
    def install(error: Exception) -> FakeRspamdClient:
        fake = FakeRspamdClient(error=error)
        fastapi_app.dependency_overrides[dependencies.get_rspamd_client] = lambda: fake
        return fake

    yield install
    fastapi_app.dependency_overrides.pop(dependencies.get_rspamd_client, None)


@pytest.fixture()
def small_upload_limit():
    """Shrink the upload limit so oversized-file tests stay fast."""

    fastapi_app.dependency_overrides[dependencies.get_max_upload_bytes] = lambda: 1024
    yield
    fastapi_app.dependency_overrides.pop(dependencies.get_max_upload_bytes, None)


# Fixture wiring only; helpers live in helpers.py.
