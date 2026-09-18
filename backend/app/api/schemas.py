"""Pydantic models for the combined API responses (Steps 1-5)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.ai import AIAnalysis
from app.schemas.authentication import AuthenticationEvidence
from app.schemas.email import EmailEvidence
from app.schemas.intelligence import IntelligenceEvidence


class AnalysisResponse(BaseModel):
    """Combined Step 1 + Step 2 + Step 4 + Step 5 analysis result.

    ``email``, ``authentication``, and ``intelligence`` reuse the existing
    evidence schemas verbatim; ``ai`` carries Step 5 MODEL EVIDENCE only —
    no risk score, threat verdict, or forensic report exists at this phase.
    AI model failures never abort the request: they degrade to explicit
    ``unavailable``/``error`` states inside ``ai``.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: str = "1.0"
    email: EmailEvidence
    authentication: AuthenticationEvidence
    intelligence: IntelligenceEvidence = Field(default_factory=IntelligenceEvidence)
    ai: AIAnalysis = Field(default_factory=AIAnalysis)


class HealthResponse(BaseModel):
    """Liveness response for the API process itself."""

    model_config = ConfigDict(extra="forbid")

    status: str = "ok"
    service: str = "forentisai-api"


class RspamdHealthResponse(BaseModel):
    """Rspamd dependency availability, clearly separate from API liveness."""

    model_config = ConfigDict(extra="forbid")

    rspamd_available: bool
    detail: str | None = Field(default=None)
