"""Technical ML inference (Phase D).

Converts a :class:`FeatureVector` into a :class:`TechnicalMLResult` with an
explicit state. Failures never become benign predictions:

- missing / incompatible artifact → ``unavailable`` with ``reason``;
- malformed feature vector → ``unavailable`` with
  ``feature_extraction_failed``;
- unexpected prediction failure → ``error``.
"""

from __future__ import annotations

import numpy as np

from app.ai.technical_ml.model import (
    TechnicalModelBundle,
    TechnicalModelLoadError,
)
from app.schemas.ai import ModelIndicator, TechnicalMLResult
from app.schemas.features import FeatureVector, feature_names

# Human-readable detail for the strongest suspicious-leaning contributions.
# Used only as indicator text; values themselves come from the model.
_TOP_INDICATOR_LIMIT = 5


def vector_to_array(vector: FeatureVector) -> np.ndarray:
    """Serialize a FeatureVector into the contracted 2-D float array.

    Column order is exactly :func:`feature_names`. Raises ``ValueError`` on
    any value that cannot become a finite float (bools and ints are fine).
    """

    names = feature_names()
    values: list[float] = []

    for name in names:
        section_name, field_name = name.split(".", maxsplit=1)
        section = getattr(vector, section_name)
        raw = getattr(section, field_name)
        value = float(raw)
        if not np.isfinite(value):
            raise ValueError(f"non-finite feature value for {name}")
        values.append(value)

    return np.asarray([values], dtype=np.float64)


def _probabilities_dict(
    classes: list[str], probabilities: np.ndarray
) -> dict[str, float]:
    return {str(cls): float(prob) for cls, prob in zip(classes, probabilities)}


def predict_technical(
    vector: FeatureVector,
    bundle: TechnicalModelBundle | None,
) -> TechnicalMLResult:
    """Run technical-ML inference; total function never raises."""

    if bundle is None:
        return TechnicalMLResult(
            available=False,
            status="unavailable",
            reason="model_not_found",
            message="No technical model artifact was provided (or it failed to load).",
        )

    try:
        array = vector_to_array(vector)
    except Exception as error:  # noqa: BLE001 - malformed input is isolated
        return TechnicalMLResult(
            available=False,
            status="unavailable",
            reason="feature_extraction_failed",
            message=f"Feature vector could not be serialized: {error}",
        )

    try:
        probabilities = bundle.pipeline.predict_proba(array)[0]
        classes = [str(cls) for cls in bundle.pipeline.classes_]
    except Exception as error:  # noqa: BLE001 - prediction failure is isolated
        return TechnicalMLResult(
            available=False,
            status="error",
            reason="incompatible_model",
            message=f"Technical model prediction failed: {error}",
        )

    prob_dict = _probabilities_dict(classes, np.asarray(probabilities, dtype=float))
    suspicious = prob_dict.get("suspicious")
    benign = prob_dict.get("benign")

    if suspicious is None:
        predicted = "unknown"
        confidence = None
    else:
        predicted = "suspicious" if (benign is None or suspicious >= benign) else "benign"
        confidence = max(prob_dict.values()) if prob_dict else None

    return TechnicalMLResult(
        available=True,
        status="available",
        model_name=bundle.metadata.model_name,
        predicted_class=predicted,  # type: ignore[arg-type]
        probabilities=prob_dict,
        confidence=confidence,
        indicators=_top_indicators(vector, prob_dict.get("suspicious", 0.0)),
        message=None,
    )


def _top_indicators(
    vector: FeatureVector, suspicious_probability: float
) -> list[ModelIndicator]:
    """Deterministic model-signal list for the technical result.

    Signals are described as model evidence only — never "confirmed"-
    style language. The base rate note is included when the model itself
    leans suspicious so consumers do not over-read a single probability.
    """

    indicators: list[ModelIndicator] = []

    if suspicious_probability >= 0.5:
        indicators.append(
            ModelIndicator(
                name="technical_model_suspicious_lean",
                detail=(
                    "Technical model probability mass leans toward the "
                    f"suspicious class ({suspicious_probability:.2f}); this is a "
                    "model signal only."
                ),
                weight=round(suspicious_probability, 4),
            )
        )
    else:
        indicators.append(
            ModelIndicator(
                name="technical_model_benign_lean",
                detail=(
                    "Technical model probability mass leans toward the benign "
                    f"class ({1.0 - suspicious_probability:.2f}); this is a model "
                    "signal, not a safety guarantee."
                ),
                weight=round(1.0 - suspicious_probability, 4),
            )
        )

    return indicators[:_TOP_INDICATOR_LIMIT]


__all__ = ["load_error_to_result", "predict_technical", "vector_to_array"]


# Exported for orchestrator convenience: converts a load error to a result.
def load_error_to_result(error: TechnicalModelLoadError) -> TechnicalMLResult:
    """Convert a model-load failure into a controlled unavailable result."""

    mapping = {
        "model_not_found": "model_not_found",
        "malformed_model": "incompatible_model",
        "incompatible_model": "incompatible_model",
    }
    reason = mapping.get(error.reason, "model_not_found")
    return TechnicalMLResult(
        available=False,
        status="unavailable",
        reason=reason,  # type: ignore[arg-type]
        message=f"Technical model could not be loaded: {error.reason}",
    )
