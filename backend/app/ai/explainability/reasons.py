"""Human-readable textual reasons (Phase H).

Aggregates MODEL SIGNALS into short deterministic strings for later
reporting. These are supporting signals — explicitly NOT explanations of
transformer internals and NOT security verdicts. Wording consistently uses
"model indicator" / "signal" / "pattern" rather than "confirmed".
"""

from __future__ import annotations

from app.schemas.ai import FusionResult, NLPResult, TechnicalMLResult

_MAX_REASONS = 12


def build_textual_reasons(
    nlp: NLPResult,
    technical: TechnicalMLResult,
    fusion: FusionResult,
) -> list[str]:
    """Deterministic human-readable model signals for one analysis."""

    reasons: list[str] = []

    if nlp.available:
        reasons.append(
            f"NLP model ({nlp.model_name}) predicted '{nlp.predicted_class}' "
            f"with confidence {nlp.confidence:.2f}."
            if nlp.confidence is not None
            else f"NLP model ({nlp.model_name}) produced no usable class."
        )
        for indicator in nlp.indicators:
            reasons.append(f"NLP model indicator '{indicator.name}': {indicator.detail}")
    elif nlp.reason == "model_not_trained":
        reasons.append(
            "NLP model unavailable: no fine-tuned artifact exists. The base "
            "DeBERTa model is not a phishing classifier and was not used."
        )
    elif nlp.reason:
        reasons.append(f"NLP model unavailable ({nlp.reason}); no NLP evidence included.")

    if technical.available:
        reasons.append(
            f"Technical model ({technical.model_name}) predicted "
            f"'{technical.predicted_class}' with confidence "
            f"{technical.confidence:.2f}."
            if technical.confidence is not None
            else f"Technical model ({technical.model_name}) produced no usable class."
        )
        for indicator in technical.indicators:
            reasons.append(
                f"Technical model indicator '{indicator.name}': {indicator.detail}"
            )
    elif technical.reason:
        reasons.append(
            f"Technical model unavailable ({technical.reason}); no technical "
            "evidence included."
        )

    if fusion.available:
        if fusion.message:
            reasons.append(f"Fusion note: {fusion.message}")
    elif fusion.reason:
        reasons.append("Fusion unavailable: no model evidence could be combined.")

    return reasons[:_MAX_REASONS]


__all__ = ["build_textual_reasons"]
