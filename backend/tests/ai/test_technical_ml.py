"""Technical ML tests (Phase D): inference, artifacts, failure modes."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from app.ai.technical_ml.inference import load_error_to_result, predict_technical, vector_to_array
from app.ai.technical_ml.model import (
    ARTIFACT_FORMAT_VERSION,
    TechnicalModelLoadError,
    build_pipeline,
    load_technical_model,
    save_technical_model,
)
from app.schemas.ai import TechnicalMLResult
from app.schemas.features import FeatureVector

from ai_fixtures.helpers import make_technical_bundle, save_bundle, zero_vector


# ---------------------------------------------------------------------------
# Feature serialization
# ---------------------------------------------------------------------------


def test_vector_to_array_shape_and_order():
    from app.schemas.features import feature_names as names

    array = vector_to_array(zero_vector())
    assert array.shape == (1, len(names()))
    assert array.dtype == np.float64


def test_vector_to_array_rejects_non_finite():
    vector = zero_vector()
    vector.auth.rspamd_score = float("inf")
    with pytest.raises(ValueError):
        vector_to_array(vector)


def test_vector_to_array_values_match_features():
    vector = zero_vector()
    vector.text.subject_length = 42
    vector.auth.rspamd_score = -1.5
    array = vector_to_array(vector)
    assert 42.0 in array[0]
    assert -1.5 in array[0]


# ---------------------------------------------------------------------------
# Prediction behavior
# ---------------------------------------------------------------------------


def _fitted_bundle():
    """Bundle with a tiny fitted forest (deterministic)."""

    rng = np.random.default_rng(42)
    names_count = len(vector_to_array(zero_vector())[0])
    X = rng.normal(size=(60, names_count))
    y = (X[:, 0] > 0).astype(int)
    labels = np.where(y == 1, "suspicious", "benign")
    pipeline = build_pipeline(n_estimators=10)
    pipeline.fit(X, labels)
    return make_technical_bundle(pipeline)


def test_predict_technical_missing_bundle_is_unavailable():
    result = predict_technical(zero_vector(), None)
    assert result.available is False
    assert result.status == "unavailable"
    assert result.reason == "model_not_found"
    assert result.predicted_class == "unknown"
    assert result.probabilities == {}


def test_predict_technical_valid_prediction_and_probabilities():
    result = predict_technical(zero_vector(), _fitted_bundle())
    assert result.available is True
    assert result.status == "available"
    assert result.predicted_class in {"benign", "suspicious"}
    assert set(result.probabilities) == {"benign", "suspicious"}
    assert sum(result.probabilities.values()) == pytest.approx(1.0, abs=1e-6)
    assert result.confidence is not None and 0.0 <= result.confidence <= 1.0


def test_predict_technical_is_deterministic():
    bundle = _fitted_bundle()
    first = predict_technical(zero_vector(), bundle)
    second = predict_technical(zero_vector(), bundle)
    assert first.model_dump_json() == second.model_dump_json()


def test_predict_technical_malformed_features_isolated(monkeypatch):
    bundle = _fitted_bundle()

    def _boom(_vector):
        raise ValueError("bad vector")

    monkeypatch.setattr(
        "app.ai.technical_ml.inference.vector_to_array", _boom
    )
    result = predict_technical(zero_vector(), bundle)
    assert result.available is False
    assert result.reason == "feature_extraction_failed"


# ---------------------------------------------------------------------------
# Artifact save/load validation
# ---------------------------------------------------------------------------


def test_load_missing_model(tmp_path):
    with pytest.raises(TechnicalModelLoadError) as excinfo:
        load_technical_model(tmp_path / "missing.joblib")
    assert excinfo.value.reason == "model_not_found"


def test_load_malformed_model(tmp_path):
    path = tmp_path / "bad.joblib"
    path.write_bytes(b"this is not a joblib file")
    with pytest.raises(TechnicalModelLoadError) as excinfo:
        load_technical_model(path)
    assert excinfo.value.reason == "malformed_model"


def test_load_wrong_object_type(tmp_path):
    import joblib

    path = tmp_path / "wrong.joblib"
    joblib.dump({"not": "a bundle"}, path)
    with pytest.raises(TechnicalModelLoadError) as excinfo:
        load_technical_model(path)
    assert excinfo.value.reason == "incompatible_model"


def test_load_incompatible_feature_contract(tmp_path):
    bundle = make_technical_bundle(
        _fitted_bundle().pipeline,
        feature_names_override=("old.feature_a", "old.feature_b"),
    )
    path = save_bundle(bundle, tmp_path / "old_contract.joblib")
    with pytest.raises(TechnicalModelLoadError) as excinfo:
        load_technical_model(path)
    assert excinfo.value.reason == "incompatible_model"


def test_load_incompatible_format_version(tmp_path):
    from app.ai.technical_ml.model import ModelMetadata, TechnicalModelBundle
    from app.schemas.features import feature_names

    pipeline = build_pipeline(n_estimators=5)
    bundle = TechnicalModelBundle(
        pipeline=pipeline,
        metadata=ModelMetadata(
            model_name="old",
            class_labels=("benign", "suspicious"),
            feature_names=tuple(feature_names()),
            trained_at="2020-01-01T00:00:00+00:00",
            dataset_description="old",
            training_notes="old",
            artifact_format_version=ARTIFACT_FORMAT_VERSION + 1,
        ),
    )
    path = save_bundle(bundle, tmp_path / "old_version.joblib")
    with pytest.raises(TechnicalModelLoadError) as excinfo:
        load_technical_model(path)
    assert excinfo.value.reason == "incompatible_model"


def test_roundtrip_save_load(tmp_path):
    bundle = _fitted_bundle()
    path = save_bundle(bundle, tmp_path / "model.joblib")
    loaded = load_technical_model(path)
    assert loaded.metadata.model_name == bundle.metadata.model_name
    result_a = predict_technical(zero_vector(), bundle)
    result_b = predict_technical(zero_vector(), loaded)
    assert result_a.model_dump_json() == result_b.model_dump_json()


def test_load_error_to_result_mapping():
    result = load_error_to_result(TechnicalModelLoadError("model_not_found"))
    assert result.reason == "model_not_found"
    result = load_error_to_result(TechnicalModelLoadError("malformed_model"))
    assert result.reason == "incompatible_model"


# ---------------------------------------------------------------------------
# Linear model path (used by the SHAP fallback tests too)
# ---------------------------------------------------------------------------


def test_linear_pipeline_prediction():
    from ai_fixtures.helpers import make_linear_bundle

    bundle = make_linear_bundle()
    result = predict_technical(zero_vector(), bundle)
    assert result.available is True
    assert result.predicted_class in {"benign", "suspicious"}


def test_unavailable_result_never_carries_probabilities():
    unavailable = TechnicalMLResult(
        available=False, status="unavailable", reason="model_not_found"
    )
    assert unavailable.probabilities == {}
    assert unavailable.predicted_class == "unknown"
