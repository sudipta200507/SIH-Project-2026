"""DKIM evidence normalization from Rspamd scan output.

Step 2 boundary: a DKIM-Signature header existing proves nothing — only an
actual verification outcome (Rspamd ``R_DKIM_*`` symbols) sets ``result``.
This module never performs signature verification itself and never treats a
present signature as a valid one.
"""

from __future__ import annotations

from app.schemas.authentication import DkimEvidence, DkimResult, RspamdAnalysis

# Rspamd policy symbols (conf/scores.d/policies_group.conf) mapped to RFC 6376
# style result vocabulary.
_SYMBOL_TO_RESULT: dict[str, DkimResult] = {
    "R_DKIM_ALLOW": "pass",
    "R_DKIM_REJECT": "fail",
    "R_DKIM_TEMPFAIL": "temperror",
    "R_DKIM_PERMFAIL": "permerror",
    "R_DKIM_NA": "none",
}

# Most significant first, used only when several DKIM results are reported
# (multiple signatures): a pass is reported if any signature passed, while a
# failure is still surfaced when no signature passed.
_RESULT_PRIORITY: tuple[DkimResult, ...] = (
    "pass",
    "fail",
    "permerror",
    "temperror",
    "none",
)


def _result_priority(result: DkimResult) -> int:
    return _RESULT_PRIORITY.index(result)


def _split_option_parts(option: str) -> list[str]:
    """Split one option string into ``key:value``-like parts."""

    parts: list[str] = []
    for token in option.replace(";", " ").replace(",", " ").split():
        parts.append(token)
        if ":" in token:
            parts.extend(token.split(":"))
    return parts


def _extract_signature_details(options: list[str]) -> tuple[str | None, str | None]:
    """Best-effort extraction of signing domain and selector from options.

    Rspamd reports DKIM details in symbol options; recognized shapes include
    ``d:example.com``, ``s:selector``, ``domain=example.com`` and
    ``selector=sel``. Anything unrecognized is left as ``None`` rather than
    guessed.
    """

    domain: str | None = None
    selector: str | None = None
    for option in options:
        parts = _split_option_parts(option.lower())
        for index, part in enumerate(parts):
            if domain is None and part in {"d", "domain"} and index + 1 < len(parts):
                candidate = parts[index + 1].strip().strip(",")
                if candidate and "." in candidate:
                    domain = candidate
            if selector is None and part in {"s", "selector"} and index + 1 < len(parts):
                candidate = parts[index + 1].strip().strip(",")
                if candidate and "." not in candidate:
                    selector = candidate
            if part.startswith("d=") and domain is None:
                candidate = part[2:].strip()
                if candidate and "." in candidate:
                    domain = candidate
            if part.startswith("s=") and selector is None:
                candidate = part[2:].strip()
                if candidate and "." not in candidate:
                    selector = candidate
    return domain, selector


def normalize_dkim(analysis: RspamdAnalysis) -> DkimEvidence:
    """Derive DKIM evidence from the Rspamd symbols of one scan."""

    symbols: list[str] = []
    results: list[DkimResult] = []
    domain: str | None = None
    selector: str | None = None

    for name, symbol in analysis.symbols.items():
        if not name.startswith("R_DKIM_"):
            continue
        symbols.append(name)
        result = _SYMBOL_TO_RESULT.get(name)
        if result is not None:
            results.append(result)
        option_domain, option_selector = _extract_signature_details(symbol.options)
        domain = domain or option_domain
        selector = selector or option_selector

    # ``R_DKIM_ALIGNED`` adds alignment context but is not itself a
    # verification result, so it is only recorded as a symbol.
    result = min(results, key=_result_priority) if results else "unknown"

    return DkimEvidence(
        result=result,
        domain=domain,
        selector=selector,
        symbols=symbols,
        available="available" if symbols else "unavailable",
    )
