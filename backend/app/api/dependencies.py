"""FastAPI dependency wiring for configuration and the Rspamd client."""

from __future__ import annotations

from functools import lru_cache

from fastapi import Depends, Request

from app.api.errors import api_error
from app.authentication.rspamd_client import RspamdClient
from app.core.config import (
    ApiSettings,
    AISettings,
    IntelligenceEnvSettings,
    RspamdSettings,
    load_ai_settings,
    load_api_settings,
    load_intelligence_env_settings,
    load_rspamd_settings,
)


@lru_cache(maxsize=1)
def _cached_api_settings() -> ApiSettings:
    return load_api_settings()


@lru_cache(maxsize=1)
def _cached_rspamd_settings() -> RspamdSettings:
    return load_rspamd_settings()


@lru_cache(maxsize=1)
def _cached_intelligence_settings() -> IntelligenceEnvSettings:
    return load_intelligence_env_settings()


@lru_cache(maxsize=1)
def _cached_ai_settings() -> AISettings:
    return load_ai_settings()


def get_api_settings() -> ApiSettings:
    """Return process-wide API settings (cached; refresh via ``restart``)."""

    return _cached_api_settings()


def get_rspamd_settings() -> RspamdSettings:
    """Return process-wide Rspamd settings (cached)."""

    return _cached_rspamd_settings()


def get_rspamd_client(
    settings: RspamdSettings = Depends(get_rspamd_settings),
) -> RspamdClient:
    """Provide a stateless Rspamd client bound to the configured settings."""

    return RspamdClient(settings)


def get_intelligence_settings() -> IntelligenceEnvSettings:
    """Return process-wide intelligence bounds (cached)."""

    return _cached_intelligence_settings()


def get_ai_settings() -> AISettings:
    """Return process-wide AI pipeline settings (cached)."""

    return _cached_ai_settings()


def get_max_upload_bytes() -> int:
    """Expose the single Step 1 upload limit to route handlers."""

    from app.core.constants import DEFAULT_MAX_UPLOAD_SIZE_BYTES

    return DEFAULT_MAX_UPLOAD_SIZE_BYTES


def raise_api_error(request: Request, code: str) -> None:
    """Raise a controlled :class:`ApiError` (helper for routes)."""

    raise api_error(code)
