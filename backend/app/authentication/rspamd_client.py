"""HTTP client for Rspamd's ``/checkv2`` message scanning endpoint.

Step 2 boundary:
- the request body is the ORIGINAL raw email bytes, never reconstructed or
  modified before scanning;
- responses are validated and returned as structured data;
- connection failures, timeouts, and malformed replies are controlled errors;
- nothing is logged, and error messages never contain email content.

The endpoint and reply format follow https://docs.rspamd.com/developers/protocol/.
"""

from __future__ import annotations

from typing import Any

import httpx

from app.core.config import RspamdSettings, load_rspamd_settings
from app.schemas.authentication import RspamdAnalysis, RspamdSymbol


class RspamdError(Exception):
    """Base error with a safe, machine-readable code (no email content)."""

    code = "rspamd_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class RspamdConnectionError(RspamdError):
    """Rspamd is unreachable, stopped, or returned an HTTP-level failure."""

    code = "rspamd_unavailable"


class RspamdTimeoutError(RspamdError):
    """The scan request exceeded the configured timeout."""

    code = "rspamd_timeout"


class RspamdInvalidResponseError(RspamdError):
    """Rspamd replied with something that is not a valid scan response."""

    code = "rspamd_invalid_response"


class RspamdInvalidEmailError(RspamdError):
    """The supplied email content cannot be scanned at all."""

    code = "rspamd_invalid_email"


def parse_scan_response(payload: Any) -> RspamdAnalysis:
    """Validate a decoded Rspamd JSON reply and return structured analysis."""

    if not isinstance(payload, dict):
        raise RspamdInvalidResponseError("Rspamd response was not a JSON object.")

    action = payload.get("action")
    if not isinstance(action, str) or not action.strip():
        raise RspamdInvalidResponseError("Rspamd response is missing a valid 'action'.")

    try:
        score = float(payload["score"])  # type: ignore[arg-type]
        required_score = float(payload["required_score"])  # type: ignore[arg-type]
    except (KeyError, TypeError, ValueError) as error:
        raise RspamdInvalidResponseError(
            "Rspamd response is missing valid 'score'/'required_score' fields."
        ) from error

    symbols_raw = payload.get("symbols")
    if symbols_raw is None:
        symbols_raw = {}
    if not isinstance(symbols_raw, dict):
        raise RspamdInvalidResponseError("Rspamd 'symbols' field has an unexpected shape.")

    symbols: dict[str, RspamdSymbol] = {}
    for name, details in symbols_raw.items():
        if isinstance(details, dict):
            raw_score = details.get("score")
            score_value = float(raw_score) if isinstance(raw_score, (int, float)) else 0.0
            raw_options = details.get("options") or []
            options = (
                [str(option) for option in raw_options] if isinstance(raw_options, list) else []
            )
        elif isinstance(details, (int, float)):
            score_value = float(details)
            options = []
        else:
            score_value = 0.0
            options = []
        symbols[str(name)] = RspamdSymbol(score=score_value, options=options)

    message_id = payload.get("message-id")
    return RspamdAnalysis(
        action=action.strip(),
        score=score,
        required_score=required_score,
        symbols=symbols,
        message_id=str(message_id) if message_id else None,
        scanned=not bool(payload.get("is_skipped", False)),
    )


class RspamdClient:
    """Small stateless client around Rspamd's normal worker HTTP service."""

    def __init__(
        self,
        settings: RspamdSettings | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._settings = settings or load_rspamd_settings()
        self._transport = transport

    @property
    def settings(self) -> RspamdSettings:
        return self._settings

    def scan_bytes(self, raw_email: bytes) -> RspamdAnalysis:
        """Scan the ORIGINAL raw email bytes without modifying them."""

        if not raw_email:
            raise RspamdInvalidEmailError("Email content is empty; nothing to scan.")

        url = f"{self._settings.url}{self._settings.scan_path}"
        try:
            with httpx.Client(
                timeout=self._settings.timeout_seconds,
                transport=self._transport,
            ) as client:
                response = client.post(url, content=raw_email)
        except httpx.TimeoutException as error:
            raise RspamdTimeoutError(
                f"Rspamd scan timed out after {self._settings.timeout_seconds:g}s at {url}."
            ) from error
        except httpx.TransportError as error:
            raise RspamdConnectionError(
                f"Rspamd is unreachable at {url}: {type(error).__name__}."
            ) from error

        if response.status_code != 200:
            raise RspamdConnectionError(
                f"Rspamd returned HTTP {response.status_code} at {url}."
            )

        try:
            payload = response.json()
        except ValueError as error:
            raise RspamdInvalidResponseError(
                "Rspamd returned a response body that is not valid JSON."
            ) from error

        return parse_scan_response(payload)
