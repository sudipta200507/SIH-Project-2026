"""Categorical authentication features from AuthenticationEvidence (Phase C).

One-hot style encodings: at most one flag per protocol is 1, and an
all-zero protocol row means the result was unknown/unavailable — never a
guessed value. Rspamd's score is consumed as a Step 2 security signal; it
is not the project risk score.
"""

from __future__ import annotations

from app.schemas.authentication import AuthenticationEvidence
from app.schemas.features import AuthFeatures


def extract_auth_features(evidence: AuthenticationEvidence | None) -> AuthFeatures:
    """Extract auth features; missing/None evidence yields all-unknown rows."""

    if evidence is None:
        return AuthFeatures()

    spf = evidence.spf.result if evidence.spf.available == "available" else "unknown"
    dkim = evidence.dkim.result if evidence.dkim.available == "available" else "unknown"
    dmarc = evidence.dmarc.result if evidence.dmarc.available == "available" else "unknown"

    rspamd_score = 0.0
    if evidence.rspamd.scanned and evidence.rspamd.score is not None:
        rspamd_score = float(evidence.rspamd.score)

    return AuthFeatures(
        spf_pass=1 if spf == "pass" else 0,
        spf_fail=1 if spf == "fail" else 0,
        spf_softfail=1 if spf == "softfail" else 0,
        spf_neutral_or_none=1 if spf in {"neutral", "none"} else 0,
        spf_error_or_unknown=1 if spf in {"temperror", "permerror", "unknown"} else 0,
        dkim_pass=1 if dkim == "pass" else 0,
        dkim_fail=1 if dkim == "fail" else 0,
        dkim_other=1 if dkim in {"temperror", "permerror", "none", "unknown"} else 0,
        dmarc_pass=1 if dmarc == "pass" else 0,
        dmarc_fail=1 if dmarc == "fail" else 0,
        dmarc_other=1 if dmarc in {"quarantine", "reject", "none", "unknown"} else 0,
        rspamd_available=1 if evidence.rspamd.scanned and not evidence.rspamd.error else 0,
        rspamd_score=rspamd_score,
    )


__all__ = ["extract_auth_features"]
