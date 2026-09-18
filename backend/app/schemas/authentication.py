"""Pydantic models for Step 2 authentication and security evidence."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class EvidenceModel(BaseModel):
    """Base model that keeps the exported evidence shape predictable."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# ---------------------------------------------------------------------------
# Shared normalization terms (RFC 8601 style result vocabularies).
# "unknown" always means: no authentication evaluation was available at all.
# ---------------------------------------------------------------------------

SpfResult = Literal[
    "pass",
    "fail",
    "softfail",
    "neutral",
    "none",
    "temperror",
    "permerror",
    "unknown",
]

DkimResult = Literal[
    "pass",
    "fail",
    "temperror",
    "permerror",
    "none",
    "unknown",
]

DmarcResult = Literal[
    "pass",
    "fail",
    "quarantine",
    "reject",
    "none",
    "unknown",
]

Availability = Literal["available", "unavailable"]


class SpfEvidence(EvidenceModel):
    """SPF evidence.

    SPF only authorizes the sending IP for a domain. A ``pass`` result is NOT a
    safety statement about the message or its sender's trustworthiness.
    """

    result: SpfResult = "unknown"
    domain: str | None = None
    explanation: str | None = None
    symbols: list[str] = Field(default_factory=list)
    available: Availability = "unavailable"


class DkimEvidence(EvidenceModel):
    """DKIM evidence.

    The presence of a DKIM-Signature header does not imply a valid signature;
    ``result`` is populated only from an actual verification outcome.
    """

    result: DkimResult = "unknown"
    domain: str | None = None
    selector: str | None = None
    symbols: list[str] = Field(default_factory=list)
    available: Availability = "unavailable"


class DmarcEvidence(EvidenceModel):
    """DMARC evidence.

    Alignment is recorded only when Rspamd actually reported it; missing
    alignment information stays ``None`` and is never guessed.
    """

    result: DmarcResult = "unknown"
    domain: str | None = None
    policy: str | None = None
    spf_aligned: bool | None = None
    dkim_aligned: bool | None = None
    spf_contribution: SpfEvidence | None = None
    dkim_contribution: DkimEvidence | None = None
    symbols: list[str] = Field(default_factory=list)
    available: Availability = "unavailable"


class RspamdSymbol(EvidenceModel):
    """One Rspamd symbol with its score and any string options."""

    score: float = 0.0
    options: list[str] = Field(default_factory=list)


class RspamdAnalysis(EvidenceModel):
    """Rspamd scan output.

    ``score`` is an email-security signal from Rspamd. It is never the final
    ForentisAI risk score; that belongs to a later AI phase.
    """

    action: str | None = None
    score: float | None = None
    required_score: float | None = None
    symbols: dict[str, RspamdSymbol] = Field(default_factory=dict)
    message_id: str | None = None
    scanned: bool = False
    error: str | None = None


class AuthenticationEvidence(EvidenceModel):
    """Step 2 authentication/security evidence — deliberately NOT a verdict.

    SPF/DKIM/DMARC PASS never means "safe": legitimate accounts can be
    compromised and phishing can be correctly authenticated. This object is
    evidence for later correlation phases, nothing more.
    """

    schema_version: Literal["1.0"] = "1.0"
    spf: SpfEvidence = Field(default_factory=SpfEvidence)
    dkim: DkimEvidence = Field(default_factory=DkimEvidence)
    dmarc: DmarcEvidence = Field(default_factory=DmarcEvidence)
    rspamd: RspamdAnalysis = Field(default_factory=RspamdAnalysis)
