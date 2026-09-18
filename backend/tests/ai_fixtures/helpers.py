"""Shared fakes and builders for the Step 5 AI tests.

No unit test in tests/features/ or tests/ai/ touches the network: model
artifacts are created in tmp_path via the real training helpers, NLP bundles
are always fakes, and orchestrator tests run with stubbed registry loads.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from app.ai.technical_ml.model import (
    ARTIFACT_FORMAT_VERSION,
    ModelMetadata,
    TechnicalModelBundle,
    build_pipeline,
)
from app.schemas.ai import NLPResult, TechnicalMLResult
from app.schemas.features import FeatureVector, feature_names

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SAMPLE_PATH = PROJECT_ROOT / "samples" / "safe" / "step1_synthetic.eml"


def sample_evidence():
    """Normalized Step 1 evidence for the safe synthetic sample."""

    from app.extractor.email_parser import extract_email_from_bytes

    return extract_email_from_bytes(SAMPLE_PATH.read_bytes(), filename="step1_synthetic.eml")


def zero_vector() -> FeatureVector:
    """All-default feature vector (everything neutral/zero)."""

    return FeatureVector()


def make_technical_bundle(
    pipeline: Pipeline | None = None,
    *,
    feature_names_override: tuple[str, ...] | None = None,
    format_version: int = ARTIFACT_FORMAT_VERSION,
) -> TechnicalModelBundle:
    """A small deterministic bundle for tests (fixture-like, not production)."""

    if pipeline is None:
        pipeline = build_pipeline(n_estimators=10)
    metadata = ModelMetadata(
        model_name="test-technical-model",
        class_labels=("benign", "suspicious"),
        feature_names=feature_names_override or tuple(feature_names()),
        trained_at="2026-01-01T00:00:00+00:00",
        dataset_description="unit-test fixture",
        training_notes="test only",
        artifact_format_version=format_version,
    )
    return TechnicalModelBundle(pipeline=pipeline, metadata=metadata)


def make_linear_bundle() -> TechnicalModelBundle:
    """Deterministic linear bundle (for the SHAP linear-fallback path)."""

    names = tuple(feature_names())
    rng = np.random.default_rng(7)
    classifier = LogisticRegression()
    X = rng.normal(size=(40, len(names)))
    y = np.where((X[:, 0] + X[:, 1]) > 0, "suspicious", "benign")
    classifier.fit(X, y)
    return make_technical_bundle(Pipeline([("classifier", classifier)]))


def save_bundle(bundle: TechnicalModelBundle, path: Path) -> Path:
    from app.ai.technical_ml.model import save_technical_model

    save_technical_model(bundle, path)
    return path


def make_nlp_result(
    *, available: bool = True, suspicious: float = 0.8, benign: float = 0.2
) -> NLPResult:
    if not available:
        return NLPResult(available=False, status="unavailable", reason="model_not_found")
    return NLPResult(
        available=True,
        status="available",
        model_name="test-nlp",
        model_kind="fine_tuned",
        predicted_class="suspicious" if suspicious >= benign else "benign",
        probabilities={"suspicious": suspicious, "benign": benign},
        confidence=max(suspicious, benign),
    )


def make_technical_result(
    *, available: bool = True, suspicious: float = 0.8, benign: float = 0.2
) -> TechnicalMLResult:
    if not available:
        return TechnicalMLResult(
            available=False, status="unavailable", reason="model_not_found"
        )
    return TechnicalMLResult(
        available=True,
        status="available",
        model_name="test-technical",
        predicted_class="suspicious" if suspicious >= benign else "benign",
        probabilities={"suspicious": suspicious, "benign": benign},
        confidence=max(suspicious, benign),
    )


def make_indicator(type_: str, value: str) -> object:
    """Minimal valid Step 4 indicator (mirrors tests/intelligence/fixtures)."""

    from app.schemas.intelligence import Indicator, IndicatorSource

    return Indicator(
        type=type_,  # type: ignore[arg-type]
        value=value,
        source=IndicatorSource(kind="url", location="body.plain"),  # type: ignore[arg-type]
    )


def intelligence_with_domain_ages(ages_days: list[float]) -> object:
    """Build a minimal IntelligenceEvidence with RDAP registration ages.

    Ages are computed relative to 'now'; the email date passed by the caller
    should also be relative to 'now' for the assertion to hold.
    """

    from app.schemas.intelligence import (
        DNSIntelligence,
        DomainIntelligence,
        IntelligenceEvidence,
        RDAPIntelligence,
    )

    _mk = make_indicator

    now = datetime.now(timezone.utc)
    domains = []
    for age in ages_days:
        registered = now - timedelta(days=age)
        rdap = RDAPIntelligence(
            indicator=_mk("domain", "example.test"),
            status="success",
            registration_date=registered.isoformat(),
        )
        domains.append(
            DomainIntelligence(
                indicator=_mk("domain", "example.test"),
                normalized_domain="example.test",
                dns=DNSIntelligence(
                    indicator=_mk("domain", "example.test"), status="no_record"
                ),
                rdap=rdap,
            )
        )
    return IntelligenceEvidence(domains=domains)
