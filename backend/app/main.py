"""ForentisAI FastAPI application (Steps 3-5).

The API composes the existing Step 1 extractor, Step 2 authentication,
Step 4 intelligence, and Step 5 AI model-evidence pipelines. It never
produces a final risk score, threat verdict, forensic report, or any
persistence; those belong to later project phases.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import errors
from app.api.routes import analysis, health, reports
from app.core.constants import MAX_ANALYZE_REQUEST_BYTES

APP_TITLE = "ForentisAI API"
APP_DESCRIPTION = (
    "Email analysis API: upload an original .eml file and receive normalized "
    "email evidence (Step 1), authentication evidence from Rspamd (Step 2), "
    "infrastructure intelligence (Step 4), and AI model evidence (Step 5). "
    "All results are evidence only - they never constitute a safety verdict "
    "or a risk score."
)
APP_VERSION = "0.1.0"


def create_app() -> FastAPI:
    """Build the configured FastAPI application."""

    from app.api.dependencies import get_api_settings

    app = FastAPI(
        title=APP_TITLE,
        description=APP_DESCRIPTION,
        version=APP_VERSION,
    )

    # CORS for local development. Origins come from the environment and the
    # list is never unrestricted; credentials are deliberately not enabled so
    # a wildcard misconfiguration cannot silently become dangerous.
    api_settings = get_api_settings()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(api_settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # Routers.
    app.include_router(health.router)
    app.include_router(analysis.router)
    app.include_router(reports.router)

    # Transport-level guard against oversized multipart requests. The exact
    # Step 1 file limit is enforced in the route while reading the upload
    # part; this guard only bounds the total request size with a small
    # margin for multipart overhead. No second conflicting file limit exists.
    @app.middleware("http")
    async def limit_request_body(request: Request, call_next):
        content_length = request.headers.get("content-length")
        if request.method == "POST" and content_length and content_length.isdigit():
            if int(content_length) > MAX_ANALYZE_REQUEST_BYTES:
                return errors.error_response("file_too_large")
        return await call_next(request)

    # Controlled error responses for domain and unexpected failures. Handlers
    # never leak stack traces, email content, or environment details.
    @app.exception_handler(errors.ApiError)
    async def handle_api_error(request: Request, exc: errors.ApiError) -> JSONResponse:
        return exc.to_response()

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        return errors.error_response("internal_error")

    return app


app = create_app()
