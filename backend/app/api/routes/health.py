"""Health-check routes: API liveness and Rspamd dependency availability."""

from __future__ import annotations

import httpx
from fastapi import APIRouter, Depends

from app.api.dependencies import get_rspamd_settings
from app.api.schemas import HealthResponse, RspamdHealthResponse
from app.core.config import RspamdSettings

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Report that the API process itself is alive.

    This endpoint never contacts Rspamd, so it always answers while the API
    process runs. Dependency availability is reported separately at
    ``GET /health/rspamd`` so the two concerns stay clearly distinguishable.
    """

    return HealthResponse()


@router.get("/health/rspamd", response_model=RspamdHealthResponse)
def get_rspamd_health(
    settings: RspamdSettings = Depends(get_rspamd_settings),
) -> RspamdHealthResponse:
    """Probe the Rspamd normal worker ``/ping`` endpoint.

    This is dependency status, not API liveness; the API can be healthy while
    Rspamd is down, and this endpoint then reports ``rspamd_available: false``.
    """

    ping_url = f"{settings.url}/ping"
    try:
        response = httpx.get(ping_url, timeout=5.0)
    except httpx.TransportError:
        return RspamdHealthResponse(
            rspamd_available=False,
            detail="Rspamd is unreachable at the configured URL.",
        )
    # Rspamd answers GET /ping with HTTP 200 and the body "pong".
    if response.status_code == 200 and response.text.strip().casefold() == "pong":
        return RspamdHealthResponse(rspamd_available=True)
    return RspamdHealthResponse(
        rspamd_available=False,
        detail="Rspamd responded but did not report a healthy ping.",
    )
