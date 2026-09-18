"""Explainability for the TECHNICAL model (Phase H).

SHAP is applied only where it is technically appropriate: the classical
tree-ensemble model (``TreeExplainer``). For a linear model a documented
coefficient-based fallback is used instead. For the transformer NLP model,
SHAP is deliberately NOT used: token attribution would require a separate
validated implementation, and pretending SHAP applies there would produce
fabricated explanations.

Failure behavior is total: any unsupported model, missing SHAP install, or
explainer error yields an explicit unavailable state with a reason — never
fabricated contributions and never an exception crossing the API boundary.

The output distinguishes MODEL PREDICTION (the probabilities already in
``TechnicalMLResult``) from SUPPORTING SIGNALS (the contributions here).
A contribution is attribution relative to the suspicious class: positive
pushed toward suspicious, negative away from it. It is not a statement
that the email is malicious or safe.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.schemas.ai import FeatureContribution
from app.schemas.features import feature_names

_MAX_CONTRIBUTIONS = 10


@dataclass(frozen=True, slots=True)
class TechnicalExplanation:
    """Result of explaining one technical-model prediction."""

    contributions: list[FeatureContribution] = field(default_factory=list)
    shap_available: bool = False
    reason: str | None = None
    message: str | None = None


def _classes_of(bundle) -> list[str]:
    return [str(cls) for cls in bundle.pipeline.classes_]


def _suspicious_index(bundle) -> int | None:
    classes = _classes_of(bundle)
    try:
        return classes.index("suspicious")
    except ValueError:
        return None


def _explain_tree(bundle, array: np.ndarray, class_index: int) -> list[FeatureContribution]:
    import shap

    classifier = bundle.pipeline.named_steps.get("classifier")
    if classifier is None:
        raise TypeError("pipeline has no 'classifier' step")
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(array)

    if isinstance(shap_values, list):
        values = np.asarray(shap_values[class_index])[0]
    else:
        array_values = np.asarray(shap_values)
        if array_values.ndim == 3:  # (n_samples, n_features, n_classes)
            values = array_values[0, :, class_index]
        elif array_values.ndim == 2:  # (n_samples, n_features)
            values = array_values[0]
        else:
            raise TypeError(f"unexpected shap_values shape {array_values.shape}")

    return _to_contributions(values, array[0])


def _explain_linear(bundle, array: np.ndarray) -> list[FeatureContribution]:
    """Documented fallback for linear models: coefficient × feature value.

    This is a standard linear-model attribution approximation, clearly not
    SHAP; ``shap_available`` stays False for this path.
    """

    classifier = bundle.pipeline.named_steps.get("classifier")
    classes = _classes_of(bundle)
    coefficients = np.asarray(classifier.coef_)
    if coefficients.ndim == 2:
        # Binary sklearn linear models expose a single row for classes_[1]
        # (the positive class); multi-class exposes one row per class.
        if coefficients.shape[0] == 1:
            coefficients = coefficients[0]
        else:
            class_index = classes.index("suspicious") if "suspicious" in classes else -1
            coefficients = coefficients[class_index]
    return _to_contributions(coefficients * array[0], array[0])


def _to_contributions(values: np.ndarray, feature_values: np.ndarray) -> list[FeatureContribution]:
    names = list(feature_names())
    contributions: list[FeatureContribution] = []
    for index, raw_value in enumerate(np.asarray(values).ravel()):
        if index >= len(names):
            break
        contribution = float(raw_value)
        if not np.isfinite(contribution):
            continue
        contributions.append(
            FeatureContribution(
                feature=names[index],
                value=float(feature_values[index]),
                contribution=round(contribution, 6),
                direction="toward" if contribution >= 0 else "away",
            )
        )
    contributions.sort(key=lambda item: (-abs(item.contribution), item.feature))
    return contributions[:_MAX_CONTRIBUTIONS]


def explain_technical_prediction(bundle, array: np.ndarray) -> TechnicalExplanation:
    """Explain one technical prediction; total function never raises.

    ``array`` is the 2-D feature array produced by
    ``app.ai.technical_ml.inference.vector_to_array``.
    """

    try:
        class_index = _suspicious_index(bundle)
        if class_index is None:
            return TechnicalExplanation(
                reason="unsupported_model",
                message="Model classes do not include 'suspicious'; nothing to attribute.",
            )

        classifier = bundle.pipeline.named_steps.get("classifier")
        classifier_class_name = type(classifier).__name__

        if "Forest" in classifier_class_name or "Boosting" in classifier_class_name:
            contributions = _explain_tree(bundle, array, class_index)
            return TechnicalExplanation(
                contributions=contributions,
                shap_available=True,
                message="TreeExplainer SHAP values relative to the suspicious class.",
            )

        if "LogisticRegression" in classifier_class_name:
            contributions = _explain_linear(bundle, array)
            return TechnicalExplanation(
                contributions=contributions,
                shap_available=False,
                reason="unsupported_model",
                message=(
                    "Linear fallback attribution (coefficient × value) used; "
                    "SHAP TreeExplainer does not apply to this model type."
                ),
            )

        return TechnicalExplanation(
            reason="unsupported_model",
            message=f"Model type {classifier_class_name} has no validated explainer.",
        )
    except ImportError:
        return TechnicalExplanation(
            reason="dependency_unavailable",
            message="SHAP is not installed; technical explanations are unavailable.",
        )
    except Exception as error:  # noqa: BLE001 - any explainer failure is isolated
        return TechnicalExplanation(
            reason="explanation_failed",
            message=f"SHAP explanation failed: {type(error).__name__}",
        )


__all__ = ["TechnicalExplanation", "explain_technical_prediction"]
