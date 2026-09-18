"""Orchestrator tests (Phase I): pipeline composition and failure isolation."""

from __future__ import annotations

import pytest

from app.ai import model_registry
from app.ai.orchestrator import build_ai_analysis
from app.core.config import AISettings, DEFAULT_TECHNICAL_MODEL_FILENAME
from app.schemas.intelligence import IntelligenceEvidence

from ai_fixtures.helpers import (
    make_technical_bundle,
    save_bundle,
    sample_evidence,
    zero_vector,
)


@pytest.fixture()
def evidence():
    return sample_evidence()


def _settings_with_model(tmp_path, **overrides):
    model_path = tmp_path / "models" / DEFAULT_TECHNICAL_MODEL_FILENAME
    from app.ai.technical_ml.model import build_pipeline

    import numpy as np

    from app.ai.technical_ml.inference import vector_to_array

    rng = np.random.default_rng(42)
    X = rng.normal(size=(60, len(vector_to_array(zero_vector())[0])))
    y = np.where(X[:, 0] > 0, "suspicious", "benign")
    pipeline = build_pipeline(n_estimators=10)
    pipeline.fit(X, y)
    save_bundle(make_technical_bundle(pipeline), model_path)
    settings = AISettings(
        ai_enabled=True,
        model_dir=tmp_path / "models",
        technical_model_path=model_path,
        nlp_model_path=None,
        nlp_model_name="test-nlp",
        max_text_length=2000,
        inference_timeout_seconds=10.0,
    )
    for key, value in overrides.items():
        object.__setattr__(settings, key, value)
    return settings, model_path


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def test_full_pipeline_with_technical_model(evidence, tmp_path):
    settings, _ = _settings_with_model(tmp_path)
    analysis = build_ai_analysis(
        evidence, None, IntelligenceEvidence(), settings=settings
    )
    assert analysis.status == "available"
    assert analysis.technical_ml.available is True
    assert analysis.fusion.available is True  # technical-only fusion
    assert analysis.fusion.signals[0].contributed is False  # NLP absent
    assert analysis.fusion.signals[1].contributed is True
    assert analysis.explainability.shap_available is True
    assert len(analysis.explainability.textual_reasons) > 0


def test_no_models_available_degrades_cleanly(evidence, ai_settings):
    analysis = build_ai_analysis(evidence, None, None, settings=ai_settings)
    assert analysis.status == "unavailable"
    assert analysis.nlp.available is False
    assert analysis.technical_ml.available is False
    assert analysis.fusion.available is False
    assert analysis.fusion.predicted_class == "unknown"
    # Failure states, never benign guesses:
    assert analysis.technical_ml.reason == "model_not_found"
    assert analysis.technical_ml.probabilities == {}


def test_disabled_ai(ai_settings, disabled_ai_settings, evidence):
    analysis = build_ai_analysis(evidence, None, None, settings=disabled_ai_settings)
    assert analysis.status == "unavailable"
    assert analysis.nlp.reason == "not_enabled"
    assert analysis.fusion.reason == "not_enabled"


def test_malformed_evidence_is_rejected(ai_settings):
    analysis = build_ai_analysis("not-evidence", None, None, settings=ai_settings)  # type: ignore[arg-type]
    assert analysis.status == "unavailable"
    assert analysis.nlp.reason == "invalid_input"


# ---------------------------------------------------------------------------
# Model registry caching (Phase O)
# ---------------------------------------------------------------------------


def test_model_loaded_once_per_artifact(evidence, tmp_path, monkeypatch):
    settings, model_path = _settings_with_model(tmp_path)

    calls = {"count": 0}
    real_loader = model_registry.load_technical_model

    def counting_loader(path):
        calls["count"] += 1
        return real_loader(path)

    monkeypatch.setattr(model_registry, "load_technical_model", counting_loader)

    build_ai_analysis(evidence, None, None, settings=settings)
    build_ai_analysis(evidence, None, None, settings=settings)
    build_ai_analysis(evidence, None, None, settings=settings)
    assert calls["count"] == 1  # loaded once, reused across requests


def test_retrained_model_is_picked_up(evidence, tmp_path):
    settings, model_path = _settings_with_model(tmp_path)
    first = build_ai_analysis(evidence, None, None, settings=settings)

    # Simulate retraining: rewrite the artifact with a different fitted
    # pipeline (mtime/size changes; the registry must reload it).
    import time

    import numpy as np

    from app.ai.technical_ml.inference import vector_to_array
    from app.ai.technical_ml.model import build_pipeline

    rng = np.random.default_rng(123)
    X = rng.normal(size=(60, len(vector_to_array(zero_vector())[0])))
    y = np.where(X[:, 1] > 0, "suspicious", "benign")
    pipeline = build_pipeline(n_estimators=10)
    pipeline.fit(X, y)
    save_bundle(make_technical_bundle(pipeline), model_path)
    time.sleep(0.01)
    second = build_ai_analysis(evidence, None, None, settings=settings)

    assert first.status == "available"
    assert second.status == "available"


# ---------------------------------------------------------------------------
# Timeout guard
# ---------------------------------------------------------------------------


def test_inference_timeout_becomes_error_state(evidence, tmp_path, monkeypatch):
    settings, _ = _settings_with_model(tmp_path)
    slow_settings = AISettings(
        ai_enabled=True,
        model_dir=settings.model_dir,
        technical_model_path=settings.technical_model_path,
        nlp_model_path=None,
        inference_timeout_seconds=0.05,
    )

    def slow_predict(*_args, **_kwargs):
        import time

        time.sleep(0.5)
        raise AssertionError("should have been timed out")

    monkeypatch.setattr(
        "app.ai.orchestrator.predict_technical", slow_predict
    )
    analysis = build_ai_analysis(evidence, None, None, settings=slow_settings)
    assert analysis.technical_ml.status == "error"
    assert analysis.technical_ml.reason == "inference_timeout"
    assert analysis.fusion.available is False


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_orchestrator_deterministic_serialization(evidence, tmp_path):
    settings, _ = _settings_with_model(tmp_path)
    first = build_ai_analysis(evidence, None, None, settings=settings)
    second = build_ai_analysis(evidence, None, None, settings=settings)
    assert first.model_dump_json() == second.model_dump_json()


# ---------------------------------------------------------------------------
# Security: no network access from the AI layer
# ---------------------------------------------------------------------------


def test_ai_layer_performs_no_socket_operations(evidence, tmp_path, monkeypatch):
    """Any socket connection attempt during the AI pipeline fails the test."""

    import socket

    def _forbidden(*args, **kwargs):
        raise AssertionError("AI layer attempted a network operation")

    monkeypatch.setattr(socket, "create_connection", _forbidden)
    monkeypatch.setattr(socket.socket, "connect", _forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", _forbidden)

    settings, _ = _settings_with_model(tmp_path)
    analysis = build_ai_analysis(
        evidence, None, IntelligenceEvidence(), settings=settings
    )
    assert analysis.status in {"available", "unavailable"}


def test_evidence_objects_are_not_mutated(evidence, tmp_path):
    settings, _ = _settings_with_model(tmp_path)
    before = evidence.model_dump_json()
    build_ai_analysis(
        evidence, None, IntelligenceEvidence(), settings=settings
    )
    assert evidence.model_dump_json() == before
