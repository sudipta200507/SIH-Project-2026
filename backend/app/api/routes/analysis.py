"""Analysis routes: the combined extraction + authentication + intelligence endpoint."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile

from app.ai.orchestrator import build_ai_analysis
from app.api import errors
from app.api.dependencies import (
    get_ai_settings,
    get_intelligence_settings,
    get_max_upload_bytes,
    get_rspamd_client,
)
from app.api.schemas import AnalysisResponse
from app.authentication.dmarc import build_authentication_evidence
from app.authentication.rspamd_client import (
    RspamdAnalysis,
    RspamdClient,
    RspamdError,
    RspamdInvalidEmailError,
    RspamdInvalidResponseError,
    RspamdTimeoutError,
)
from app.core.config import AISettings, IntelligenceEnvSettings
from app.schemas.ai import AIAnalysis
from app.extractor.email_parser import (
    EmailExtractionError,
    extract_email_from_bytes,
)
from app.intelligence.orchestrator import (
    IntelligenceSettings,
    build_intelligence_evidence,
)
from app.schemas.intelligence import IntelligenceEvidence

router = APIRouter(prefix="/analyze-email", tags=["analysis"])


def _validate_extension(filename: str | None) -> None:
    """Reject non-.eml uploads using the Step 1 strategy.

    The filename is treated as display metadata only; it is never used in any
    filesystem, shell, or network operation.
    """

    name = (filename or "").replace("\\", "/").rsplit("/", maxsplit=1)[-1]
    suffix = Path(name).suffix.casefold()
    if suffix != ".eml":
        raise errors.api_error("unsupported_format")


async def _read_upload(upload: UploadFile, max_upload_bytes: int) -> bytes:
    """Read the uploaded part while enforcing the single Step 1 file limit."""

    buffer = bytearray()
    while True:
        chunk = await upload.read(1024 * 1024)
        if not chunk:
            break
        buffer.extend(chunk)
        if len(buffer) > max_upload_bytes:
            raise errors.api_error("file_too_large")
    return bytes(buffer)


def _perform_rspamd_scan(client: RspamdClient, raw_email: bytes) -> RspamdAnalysis:
    """Scan the ORIGINAL bytes; controlled Rspamd failures become ApiErrors."""

    try:
        return client.scan_bytes(raw_email)
    except RspamdInvalidEmailError as error:
        raise errors.api_error("invalid_email") from error
    except RspamdTimeoutError as error:
        raise errors.api_error("rspamd_timeout") from error
    except RspamdInvalidResponseError as error:
        raise errors.api_error("rspamd_invalid_response") from error
    except RspamdError as error:
        raise errors.api_error("rspamd_unavailable") from error


def _intelligence_settings_from_env(
    env_settings: IntelligenceEnvSettings,
) -> IntelligenceSettings:
    """Translate cached env settings into one orchestrator settings object."""

    from app.intelligence.dns import DNSSettings
    from app.intelligence.rdap import RDAPSettings

    return IntelligenceSettings(
        dns=DNSSettings(
            timeout_seconds=env_settings.dns_timeout_seconds,
            lifetime_seconds=env_settings.dns_lifetime_seconds,
        ),
        rdap=RDAPSettings(timeout_seconds=env_settings.rdap_timeout_seconds),
        max_ips=env_settings.max_ips,
        max_domains=env_settings.max_domains,
        max_urls=env_settings.max_urls,
        perform_dns=env_settings.perform_dns,
        perform_rdap=env_settings.perform_rdap,
    )


@router.post("", response_model=AnalysisResponse, name="analyze_email")
async def analyze_email(
    upload: UploadFile = File(..., description="An original .eml file (multipart/form-data)."),
    client: RspamdClient = Depends(get_rspamd_client),
    max_upload_bytes: int = Depends(get_max_upload_bytes),
    intelligence_env: IntelligenceEnvSettings = Depends(get_intelligence_settings),
    ai_settings: "AISettings" = Depends(get_ai_settings),
) -> AnalysisResponse:
    """Analyze one original .eml upload and return combined evidence.

    The uploaded bytes are passed byte-for-byte to four independent
    pipelines: Step 1 extraction (``EmailEvidence``), Step 2 Rspamd
    authentication (``AuthenticationEvidence``), Step 4 infrastructure
    intelligence (``IntelligenceEvidence``), and Step 5 AI model analysis
    (``AIAnalysis`` — model evidence only, never a verdict). The email is
    never reconstructed, modified, or persisted. The upload limit is
    25 MiB (the Step 1 limit; there is no second, conflicting API limit).

    Intelligence and AI providers degrade gracefully: a DNS timeout, an
    unconfigured GeoIP database, or a missing AI model becomes an explicit
    status inside the respective section while ``email`` and
    ``authentication`` remain fully valid.
    """

    _validate_extension(upload.filename)

    raw_email = await _read_upload(upload, max_upload_bytes)
    if not raw_email:
        raise errors.api_error("empty_file")
    if len(raw_email) > max_upload_bytes:
        raise errors.api_error("file_too_large")

    try:
        evidence = extract_email_from_bytes(
            raw_email,
            filename=upload.filename or "uploaded.eml",
            max_size_bytes=max_upload_bytes,
        )
    except EmailExtractionError as error:
        raise errors.map_exception_to_api_error(error) from error

    analysis = _perform_rspamd_scan(client, raw_email)
    authentication = build_authentication_evidence(analysis)

    # Step 4 runs after authentication and never aborts the response: every
    # provider failure degrades to an explicit status inside IntelligenceEvidence.
    try:
        intelligence = build_intelligence_evidence(
            evidence,
            settings=_intelligence_settings_from_env(intelligence_env),
        )
    except Exception:
        # A total orchestrator failure still must not destroy the analysis.
        intelligence = IntelligenceEvidence()

    # Step 5 runs last: AI model failures are isolated to the ``ai`` section
    # and can never destroy the Step 1-4 evidence.
    try:
        ai_analysis = build_ai_analysis(
            evidence,
            authentication,
            intelligence,
            settings=ai_settings,
        )
    except Exception:
        # Defensive: build_ai_analysis is total, but a failure here must
        # still not destroy the analysis.
        ai_analysis = AIAnalysis()

    return AnalysisResponse(
        email=evidence,
        authentication=authentication,
        intelligence=intelligence,
        ai=ai_analysis,
    )
