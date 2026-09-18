"""Fusion (Phase G) and explainability (Phase H) tests."""

from __future__ import annotations

import numpy as np
import pytest

from app.ai.explainability.shap_explainer import explain_technical_prediction
from app.ai.fusion.classifier import fuse
from app.schemas.ai import NLPResult, TechnicalMLResult

from ai_fixtures.helpers import (
    make_linear_bundle,
    make_nlp_result,
    make_technical_bundle,
    make_technical_result,
    zero_vector,
)

from app.ai.technical_ml.inference import vector_to_array


# ---------------------------------------------------------------------------
# Fusion availability matrix
# ---------------------------------------------------------------------------


def test_fusion_both_available():
    nlp = make_nlp_result(suspicious=0.9, benign=0.1)
    tech = make_technical_result(suspicious=0.5, benign=0.5)
    fusion = fuse(nlp, tech)
    assert fusion.available is True
    assert fusion.probability_suspicious == pytest.approx(0.7)  # (0.9+0.5)/2
    assert fusion.predicted_class == "suspicious"
    assert [signal.contributed for signal in fusion.signals] == [True, True]


def test_fusion_nlp_only():
    nlp = make_nlp_result(suspicious=0.8, benign=0.2)
    tech = make_technical_result(available=False)
    fusion = fuse(nlp, tech)
    assert fusion.available is True
    assert fusion.probability_suspicious == pytest.approx(0.8)
    assert [signal.contributed for signal in fusion.signals] == [True, False]
    assert "Only the NLP component" in (fusion.message or "")


def test_fusion_technical_only():
    nlp = make_nlp_result(available=False)
    tech = make_technical_result(suspicious=0.3, benign=0.7)
    fusion = fuse(nlp, tech)
    assert fusion.available is True
    assert fusion.probability_suspicious == pytest.approx(0.3)
    assert fusion.predicted_class == "benign"
    assert [signal.contributed for signal in fusion.signals] == [False, True]


def test_fusion_both_unavailable_is_never_benign():
    nlp = make_nlp_result(available=False)
    tech = make_technical_result(available=False)
    fusion = fuse(nlp, tech)
    assert fusion.available is False
    assert fusion.status == "unavailable"
    assert fusion.predicted_class == "unknown"
    assert fusion.probability_suspicious is None
    assert fusion.probabilities == {}


def test_fusion_conflicting_predictions_low_confidence_note():
    nlp = make_nlp_result(suspicious=0.95, benign=0.05)
    tech = make_technical_result(suspicious=0.05, benign=0.95)
    fusion = fuse(nlp, tech)
    assert fusion.available is True
    assert fusion.probability_suspicious == pytest.approx(0.5)
    # Confidence is exactly at the boundary; a disagreement note exists.
    assert fusion.confidence == pytest.approx(0.5)
    assert "disagree" in (fusion.message or "").lower()


def test_fusion_probability_normalization_bounds():
    nlp = make_nlp_result(suspicious=1.0, benign=0.0)
    tech = make_technical_result(suspicious=0.0, benign=1.0)
    fusion = fuse(nlp, tech)
    assert 0.0 <= fusion.probability_suspicious <= 1.0
    assert sum(fusion.probabilities.values()) == pytest.approx(1.0)


def test_fusion_custom_weights():
    nlp = make_nlp_result(suspicious=0.6, benign=0.4)
    tech = make_technical_result(suspicious=0.2, benign=0.8)
    fusion = fuse(nlp, tech, nlp_weight=0.75, technical_weight=0.25)
    assert fusion.probability_suspicious == pytest.approx(0.75 * 0.6 + 0.25 * 0.2)


def test_fusion_is_deterministic():
    nlp = make_nlp_result(suspicious=0.7, benign=0.3)
    tech = make_technical_result(suspicious=0.4, benign=0.6)
    assert fuse(nlp, tech).model_dump_json() == fuse(nlp, tech).model_dump_json()


# ---------------------------------------------------------------------------
# Explainability — SHAP tree path (real SHAP, tiny forest)
# ---------------------------------------------------------------------------


def _fitted_forest_bundle():
    from app.ai.technical_ml.model import build_pipeline

    rng = np.random.default_rng(42)
    X = rng.normal(size=(60, len(vector_to_array(zero_vector())[0])))
    y = np.where(X[:, 0] > 0, "suspicious", "benign")
    pipeline = build_pipeline(n_estimators=10)
    pipeline.fit(X, y)
    return make_technical_bundle(pipeline)


def test_shap_tree_explainer_produces_contributions():
    bundle = _fitted_forest_bundle()
    array = vector_to_array(zero_vector())
    explanation = explain_technical_prediction(bundle, array)
    assert explanation.shap_available is True
    assert len(explanation.contributions) > 0
    for contribution in explanation.contributions:
        assert contribution.direction in {"toward", "away"}
        assert "." in contribution.feature  # dotted contract name
    # Sorted by descending magnitude:
    magnitudes = [abs(c.contribution) for c in explanation.contributions]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_shap_contributions_sorted_by_magnitude():
    bundle = _fitted_forest_bundle()
    array = vector_to_array(zero_vector())
    explanation = explain_technical_prediction(bundle, array)
    magnitudes = [abs(c.contribution) for c in explanation.contributions]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_shap_linear_fallback_marks_shap_unavailable():
    bundle = make_linear_bundle()
    array = vector_to_array(zero_vector())
    explanation = explain_technical_prediction(bundle, array)
    assert explanation.shap_available is False
    assert explanation.reason == "unsupported_model"
    assert len(explanation.contributions) > 0  # coefficient-based fallback


def test_shap_unsupported_model_type():
    """A classifier with no validated explainer yields unsupported_model."""

    class MysteryEstimator:
        classes_ = ["benign", "suspicious"]

    class _StubPipeline:
        classes_ = ["benign", "suspicious"]  # pipeline-level classes_
        named_steps = {"classifier": MysteryEstimator()}

    class _StubBundle:
        pipeline = _StubPipeline()

    array = vector_to_array(zero_vector())
    explanation = explain_technical_prediction(_StubBundle(), array)
    assert explanation.shap_available is False
    assert explanation.reason == "unsupported_model"


def test_shap_missing_library_degrades(monkeypatch):
    bundle = _fitted_forest_bundle()
    array = vector_to_array(zero_vector())
    import builtins

    real_import = builtins.__import__

    def _no_shap(name, *args, **kwargs):
        if name == "shap":
            raise ImportError("shap not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_shap)
    explanation = explain_technical_prediction(bundle, array)
    assert explanation.shap_available is False
    assert explanation.reason == "dependency_unavailable"


def test_shap_failure_is_isolated(monkeypatch):
    bundle = _fitted_forest_bundle()
    array = vector_to_array(zero_vector())

    class _Exploder:
        def __init__(self, *_args, **_kwargs):
            pass

        def shap_values(self, *_args, **_kwargs):
            raise RuntimeError("explainer exploded")

    import shap as real_shap

    monkeypatch.setattr(
        real_shap, "TreeExplainer", _Exploder, raising=True
    )
    explanation = explain_technical_prediction(bundle, array)
    assert explanation.shap_available is False
    assert explanation.reason == "explanation_failed"


def test_explanation_never_proves_maliciousness():
    """Attributions are model signals; wording must not claim verdicts."""

    bundle = _fitted_forest_bundle()
    array = vector_to_array(zero_vector())
    explanation = explain_technical_prediction(bundle, array)
    text = (explanation.message or "").lower()
    assert "confirmed" not in text
    assert "malicious" not in text
