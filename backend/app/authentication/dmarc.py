"""DMARC evidence normalization from Rspamd scan output.

Step 2 boundary: DMARC results come from Rspamd's ``DMARC_*`` symbols. This
module never invents alignment information — ``spf_aligned`` and
``dkim_aligned`` are set only from explicit Rspamd options, and remain
``None`` when Rspamd did not report them.
"""

from __future__ import annotations

from app.authentication.dkim import normalize_dkim
from app.authentication.spf import normalize_spf
from app.schemas.authentication import (
    AuthenticationEvidence,
    DmarcEvidence,
    DmarcResult,
    RspamdAnalysis,
)

# Rspamd policy symbols (conf/scores.d/policies_group.conf) mapped to DMARC
# outcome vocabulary. ``DMARC_POLICY_ALLOW`` (and ``..._WITH_FAILURES``) mean
# the message passed DMARC; ``DMARC_POLICY_SOFTFAIL`` means it failed under a
# quarantine-oriented policy; the REJECT/QUARANTINE symbols mean the message
# failed DMARC whose policy requests that specific disposition.
_SYMBOL_TO_RESULT: dict[str, DmarcResult] = {
    "DMARC_POLICY_ALLOW": "pass",
    "DMARC_POLICY_ALLOW_WITH_FAILURES": "pass",
    "DMARC_POLICY_REJECT": "reject",
    "DMARC_POLICY_QUARANTINE": "quarantine",
    "DMARC_POLICY_SOFTFAIL": "fail",
    "DMARC_NA": "none",
}

# Policy strings reported alongside DMARC results when present.
_POLICY_HINTS: dict[str, str] = {
    "DMARC_POLICY_REJECT": "reject",
    "DMARC_POLICY_QUARANTINE": "quarantine",
    "DMARC_POLICY_SOFTFAIL": "quarantine",
    "DMARC_POLICY_ALLOW": "none",
    "DMARC_POLICY_ALLOW_WITH_FAILURES": "none",
    "DMARC_NA": "none",
}

_ALIGNMENT_TRUE = {"true", "yes", "align", "aligned", "pass"}
_ALIGNMENT_FALSE = {"false", "no", "noalign", "not_aligned", "fail"}


def _alignment_value(token: str) -> bool | None:
    lowered = token.casefold()
    if lowered in _ALIGNMENT_TRUE:
        return True
    if lowered in _ALIGNMENT_FALSE:
        return False
    return None


def _extract_alignment(options: list[str]) -> tuple[bool | None, bool | None]:
    """Extract explicit SPF/DKIM alignment flags from symbol options.

    Recognized shapes include ``spf_aligned=true``, ``dkim_aligned=false``,
    ``spf:align``, ``dkim:noalign``. Missing information stays ``None``.
    """

    spf_aligned: bool | None = None
    dkim_aligned: bool | None = None
    for option in options:
        tokens = option.replace(";", " ").replace(",", " ").split()
        for token in tokens:
            lowered = token.casefold()
            if lowered.startswith("spf_aligned=") or lowered.startswith("spf:"):
                value = lowered.split("=", 1)[-1].split(":", 1)[-1]
                spf_aligned = _alignment_value(value)
            elif lowered.startswith("dkim_aligned=") or lowered.startswith("dkim:"):
                value = lowered.split("=", 1)[-1].split(":", 1)[-1]
                dkim_aligned = _alignment_value(value)
    return spf_aligned, dkim_aligned


def _extract_domain(options: list[str]) -> str | None:
    """Best-effort extraction of the DMARC evaluated domain from options."""

    for option in options:
        for token in option.replace(";", " ").replace(",", " ").split():
            lowered = token.casefold()
            candidate = None
            if lowered.startswith("domain="):
                candidate = token.split("=", 1)[1]
            elif "." in token and not lowered.startswith(("d=", "s=", "ip=")):
                candidate = token
            if candidate and "." in candidate:
                return candidate.strip().strip(",")
    return None


def normalize_dmarc(analysis: RspamdAnalysis) -> DmarcEvidence:
    """Derive DMARC evidence from the Rspamd symbols of one scan."""

    symbols: list[str] = []
    results: list[DmarcResult] = []
    domain: str | None = None
    policy: str | None = None
    spf_aligned: bool | None = None
    dkim_aligned: bool | None = None

    for name, symbol in analysis.symbols.items():
        if not name.startswith("DMARC"):
            continue
        symbols.append(name)
        result = _SYMBOL_TO_RESULT.get(name)
        if result is not None:
            results.append(result)
        policy = policy or _POLICY_HINTS.get(name)
        option_domain = _extract_domain(symbol.options)
        domain = domain or option_domain
        option_spf, option_dkim = _extract_alignment(symbol.options)
        spf_aligned = spf_aligned if spf_aligned is not None else option_spf
        dkim_aligned = dkim_aligned if dkim_aligned is not None else option_dkim

    result = results[0] if results else "unknown"

    return DmarcEvidence(
        result=result,
        domain=domain,
        policy=policy,
        spf_aligned=spf_aligned,
        dkim_aligned=dkim_aligned,
        symbols=symbols,
        available="available" if symbols else "unavailable",
    )


def build_authentication_evidence(analysis: RspamdAnalysis | None = None) -> AuthenticationEvidence:
    """Aggregate SPF, DKIM, DMARC, and Rspamd output into Step 2 evidence.

    The result is authentication/security evidence only. It carries no risk
    score, threat verdict, or safety statement: authenticated phishing and
    compromised legitimate senders remain possible regardless of any ``pass``
    result here.
    """

    if analysis is None:
        return AuthenticationEvidence()

    return AuthenticationEvidence(
        spf=normalize_spf(analysis),
        dkim=normalize_dkim(analysis),
        dmarc=normalize_dmarc(analysis),
        rspamd=analysis,
    )
