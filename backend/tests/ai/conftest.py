"""Shared fixtures for Step 5 AI tests: settings builders, cache isolation."""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_fixtures.helpers import PROJECT_ROOT, SAMPLE_PATH  # noqa: F401
from app.ai import model_registry
from app.core.config import AISettings, DEFAULT_TECHNICAL_MODEL_FILENAME


@pytest.fixture(autouse=True)
def _isolate_model_caches():
    """Keep the process-wide model registry clean between tests."""

    model_registry.clear_caches()
    yield
    model_registry.clear_caches()


@pytest.fixture()
def ai_settings(tmp_path: Path) -> AISettings:
    """Default AI settings: enabled, no real model installed anywhere."""

    return AISettings(
        ai_enabled=True,
        model_dir=tmp_path / "models",
        technical_model_path=tmp_path / "models" / DEFAULT_TECHNICAL_MODEL_FILENAME,
        nlp_model_path=None,
        nlp_model_name="test-nlp-model",
        max_text_length=2000,
        inference_timeout_seconds=10.0,
    )


@pytest.fixture()
def disabled_ai_settings(tmp_path: Path) -> AISettings:
    return AISettings(
        ai_enabled=False,
        model_dir=tmp_path / "models",
        technical_model_path=tmp_path / "models" / DEFAULT_TECHNICAL_MODEL_FILENAME,
        nlp_model_path=None,
    )
