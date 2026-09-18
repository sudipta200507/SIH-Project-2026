"""SPF evidence normalization from Rspamd scan output.

Step 2 boundary: this module normalizes what the authentication analysis
already determined. It never performs its own SPF/DNS verification and never
labels a ``pass`` result as "safe" — SPF only authorizes the sending IP for a
domain.
"""

from __future__ import annotations

from app.schemas.authentication import RspamdAnalysis, SpfEvidence, SpfResult

# Rspamd policy symbols (conf/scores.d/policies_group.conf) mapped to RFC 7208
# result vocabulary. ``R_SPF_PLUSALL`` is reported as ``fail``-adjacent context
# only when no result symbol is present; on its own it means the sender's SPF
# record authorizes every IP (+all), which is recorded as ``unknown`` here
# because the message-level check result is not available.
_SYMBOL_TO_RESULT: dict[str, SpfResult] = {
    "R_SPF_ALLOW": "pass",
    "R_SPF_FAIL": "fail",
    "R_SPF_SOFTFAIL": "softfail",
    "R_SPF_NEUTRAL": "neutral",
    "R_SPF_NA": "none",
    "R_SPF_DNSFAIL": "temperror",
    "R_SPF_PERMFAIL": "permerror",
}

# Most significant first, used only when Rspamd unexpectedly reports several
# SPF result symbols for one message.
_RESULT_PRIORITY: tuple[SpfResult, ...] = (
    "fail",
    "permerror",
    "temperror",
    "softfail",
    "neutral",
    "pass",
    "none",
)


def _result_priority(result: SpfResult) -> int:
    try:
        return _RESULT_PRIORITY.index(result)
    except ValueError:
        return len(_RESULT_PRIORITY)


def normalize_spf(analysis: RspamdAnalysis) -> SpfEvidence:
    """Derive SPF evidence from the Rspamd symbols of one scan."""

    symbols: list[str] = []
    results: list[SpfResult] = []
    explanation: str | None = None

    for name, symbol in analysis.symbols.items():
        if not name.startswith("R_SPF_"):
            continue
        symbols.append(name)
        result = _SYMBOL_TO_RESULT.get(name)
        if result is not None:
            results.append(result)
        # Rspamd puts check details (e.g. the evaluated identity) into symbol
        # options. They are preserved verbatim as explanatory evidence.
        if symbol.options and explanation is None:
            explanation = "; ".join(symbol.options)

    if results:
        result = min(results, key=_result_priority)
    elif "R_SPF_PLUSALL" in symbols:
        result = "unknown"
    else:
        result = "unknown"

    return SpfEvidence(
        result=result,
        domain=None,
        symbols=symbols,
        available="available" if symbols else "unavailable",
        explanation=explanation,
    )
