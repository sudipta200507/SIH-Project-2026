"""NLP inference (Phase E) and explainable indicators (Phase F).

Runs the fine-tuned DeBERTa-v3 classifier locally and derives deterministic
keyword-pattern indicators from the prepared model input text.

Terminology contract (Phase F): indicators are "detected signals" /
"model indicators" / "patterns" — NEVER "confirmed phishing" or similar.
Keyword hits are pattern evidence for later phases, not verdicts.

Failures never become benign predictions:
- no fine-tuned artifact → ``unavailable`` with ``model_not_found`` or
  ``model_not_trained``;
- no usable text → ``unavailable`` with ``empty_text``;
- runtime inference failure → ``error``.
"""

from __future__ import annotations

from app.ai.nlp.model import NLPModelBundle, NLPModelLoadError
from app.schemas.ai import ModelIndicator, NLPResult

# Phase F keyword lexicons. Counts and hits are pattern signals only.
_INDICATOR_LEXICONS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "urgency_language",
        "Urgency/pressure pattern detected in the message text (model indicator, not a verdict).",
        ("urgent", "immediately", "act now", "final notice", "asap", "expires today", "last warning", "action required"),
    ),
    (
        "credential_request_pattern",
        "Credential-request pattern detected (password/login/verify signals; model indicator only).",
        ("password", "verify your account", "confirm your identity", "login", "sign in", "credentials", "reset your password"),
    ),
    (
        "payment_request_pattern",
        "Payment/wire-transfer request pattern detected (model indicator only).",
        ("invoice", "payment", "wire transfer", "bank details", "remittance", "amount due", "gift card"),
    ),
    (
        "account_verification_pattern",
        "Account-verification pattern detected (model indicator only).",
        ("verify your account", "account suspended", "account locked", "confirm your identity", "validate your account", "unusual activity"),
    ),
    (
        "threat_pressure_pattern",
        "Threat/pressure language pattern detected (model indicator only).",
        ("legal action", "lawsuit", "account will be closed", "suspended permanently", "report you", "final warning"),
    ),
    (
        "authority_claim_pattern",
        "Authority-claim pattern detected (bank/government/IT-support style claims; model indicator only).",
        ("irs", "hmrc", "bank officer", "security team", "it department", "administrator", "support team"),
    ),
    (
        "beca_request_pattern",
        "Business-email-compromise-style request pattern detected (model indicator only).",
        ("confidential", "this request is confidential", "discreet", "as discussed", "do not forward", "while traveling", "off hours"),
    ),
)

_MAX_NLP_INDICATORS = 10


def _detect_indicators(text: str) -> list[ModelIndicator]:
    """Deterministic keyword-pattern indicators over the prepared text."""

    lowered = text.casefold()
    indicators: list[ModelIndicator] = []
    for name, detail, keywords in _INDICATOR_LEXICONS:
        count = sum(lowered.count(keyword) for keyword in keywords)
        if count > 0:
            indicators.append(
                ModelIndicator(name=name, detail=detail, weight=float(min(count, 10)))
            )
    indicators.sort(key=lambda item: (-item.weight, item.name))
    return indicators[:_MAX_NLP_INDICATORS]


def _probabilities_dict(labels: tuple[str, ...], probabilities) -> dict[str, float]:
    return {str(label): float(prob) for label, prob in zip(labels, probabilities)}


def predict_nlp(text: str | None, bundle: NLPModelBundle | None) -> NLPResult:
    """Run NLP inference; total function never raises.

    ``text`` must be the prepared, length-bounded model input (see
    ``app.ai.nlp.tokenizer.prepare_model_text``).
    """

    if bundle is None:
        return NLPResult(
            available=False,
            status="unavailable",
            reason="model_not_found",
            message="No fine-tuned NLP model artifact is configured or loadable.",
        )

    if not text or not text.strip():
        return NLPResult(
            available=False,
            status="unavailable",
            reason="empty_text",
            message="No usable text was extracted from the email for NLP analysis.",
        )

    try:
        import torch

        tokenizer = bundle.tokenizer
        model = bundle.model
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=bundle.max_length_tokens,
        )
        with torch.no_grad():
            logits = model(**encoded).logits
        probabilities = torch.softmax(logits, dim=-1)[0].tolist()
    except Exception as error:  # noqa: BLE001 - inference failure is isolated
        return NLPResult(
            available=False,
            status="error",
            reason="invalid_input",
            message=f"NLP model inference failed: {type(error).__name__}",
        )

    labels = bundle.class_labels
    prob_dict = _probabilities_dict(labels, probabilities)
    suspicious = prob_dict.get("suspicious")
    benign = prob_dict.get("benign")

    if suspicious is None:
        predicted = "unknown"
        confidence = None
    else:
        predicted = "suspicious" if (benign is None or suspicious >= benign) else "benign"
        confidence = max(prob_dict.values()) if prob_dict else None

    indicators = _detect_indicators(text)

    return NLPResult(
        available=True,
        status="available",
        model_name=bundle.model_name,
        model_kind="fine_tuned",
        predicted_class=predicted,  # type: ignore[arg-type]
        probabilities=prob_dict,
        confidence=confidence,
        indicators=indicators,
        text_chars=len(text),
        message=None,
    )


def load_error_to_result(error: NLPModelLoadError) -> NLPResult:
    """Convert a model-load failure into a controlled NLP result."""

    mapping = {
        "model_not_found": "model_not_found",
        "model_not_trained": "model_not_trained",
        "incompatible_model": "incompatible_model",
        "dependency_unavailable": "dependency_unavailable",
    }
    reason = mapping.get(error.reason, "model_not_found")
    messages = {
        "model_not_trained": (
            "NLP artifact is a base/foreign model without ForentisAI "
            "fine-tune metadata; it is NOT a phishing classifier and is "
            "refused for prediction. Fine-tune via ai/training/train_transformer.py."
        ),
    }
    return NLPResult(
        available=False,
        status="unavailable",
        reason=reason,  # type: ignore[arg-type]
        message=messages.get(reason, f"NLP model could not be loaded: {error.reason}"),
    )


__all__ = ["predict_nlp", "load_error_to_result"]
