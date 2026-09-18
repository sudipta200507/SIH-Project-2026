"""Tests for GET /health, GET /health/rspamd, and the OpenAPI docs."""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import app as fastapi_app


def test_health_reports_api_liveness_without_rspamd(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["service"] == "forentisai-api"


def test_health_does_not_leak_environment_or_paths(client: TestClient) -> None:
    response = client.get("/health")
    text = response.text.casefold()

    assert "rspamd_url" not in text
    assert "environment" not in text
    assert "path" not in text


def test_rspamd_health_reports_available(monkeypatch: pytest.MonkeyPatch) -> None:
    # Rspamd answers GET /ping with HTTP 200 and the body "pong".
    monkeypatch.setattr(
        httpx, "get", lambda *args, **kwargs: httpx.Response(200, text="pong\r\n")
    )
    client = TestClient(fastapi_app)

    response = client.get("/health/rspamd")

    assert response.status_code == 200
    assert response.json()["rspamd_available"] is True


def test_rspamd_health_rejects_unexpected_ping_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        httpx, "get", lambda *args, **kwargs: httpx.Response(200, text="unexpected")
    )
    client = TestClient(fastapi_app)

    response = client.get("/health/rspamd")

    assert response.status_code == 200
    assert response.json()["rspamd_available"] is False


def test_rspamd_health_reports_unavailable_without_contacting_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_transport_error(*args: object, **kwargs: object) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "get", raise_transport_error)
    client = TestClient(fastapi_app)

    response = client.get("/health/rspamd")

    assert response.status_code == 200
    payload = response.json()
    assert payload["rspamd_available"] is False
    assert payload["detail"]


def test_openapi_document_lists_analyze_email(client: TestClient) -> None:
    openapi = client.get("/openapi.json")
    assert openapi.status_code == 200
    assert "/analyze-email" in openapi.json()["paths"]
    assert "post" in openapi.json()["paths"]["/analyze-email"]


def test_docs_page_is_served(client: TestClient) -> None:
    response = client.get("/docs")
    assert response.status_code == 200
    assert "swagger" in response.text.casefold()
