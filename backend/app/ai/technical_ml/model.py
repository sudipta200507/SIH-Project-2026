"""Technical ML model artifact (Phase D).

Architecture choice — ``RandomForestClassifier``:

- The feature vector mixes binary flags, small counts, ratios, and an
  unbounded score on very different scales. Tree ensembles are scale-
  invariant, so no fragile preprocessing is needed.
- Trees capture non-linear interactions (e.g. ``spf_fail`` AND
  ``credential_signal_count``) that a linear model would miss.
- Feature importances provide a built-in, honest fallback explanation when
  SHAP is unavailable.

The exported artifact is a :class:`TechnicalModelBundle` containing the
fitted sklearn ``Pipeline`` and metadata that ties it to the exact feature
contract it was trained on. ``load_technical_model`` refuses artifacts whose
recorded feature names do not match the current :func:`feature_names`
contract — an incompatible model becomes a controlled ``unavailable`` state
at inference time, never a wrong prediction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from app.schemas.features import feature_names

ARTIFACT_FORMAT_VERSION = 2


class TechnicalModelLoadError(Exception):
    """Raised when a model artifact cannot be loaded or is incompatible."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    """Provenance of a trained technical model (honesty fields included)."""

    model_name: str
    class_labels: tuple[str, ...]
    feature_names: tuple[str, ...]
    trained_at: str
    dataset_description: str
    training_notes: str
    artifact_format_version: int = ARTIFACT_FORMAT_VERSION


@dataclass(frozen=True, slots=True)
class TechnicalModelBundle:
    """A trained pipeline plus the metadata needed to use it safely."""

    pipeline: Pipeline
    metadata: ModelMetadata
    extra: dict = field(default_factory=dict)


def build_pipeline(n_estimators: int = 200, random_state: int = 42) -> Pipeline:
    """Build the (unfitted) sklearn pipeline used for training and export.

    ``random_state`` is fixed so that training and inference behavior are
    deterministic for identical inputs and artifacts.
    """

    return Pipeline(
        steps=[
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    random_state=random_state,
                    n_jobs=1,  # deterministic ordering of tree building
                ),
            )
        ]
    )


def save_technical_model(bundle: TechnicalModelBundle, path: Path) -> None:
    """Persist a bundle with joblib, creating parent directories."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_technical_model(path: Path) -> TechnicalModelBundle:
    """Load and validate a bundle.

    Raises :class:`TechnicalModelLoadError` (never arbitrary exceptions) when
    the file is missing, unreadable, malformed, or was trained against a
    different feature contract. The caller maps every failure to a
    controlled ``unavailable`` inference state.
    """

    path = Path(path)
    if not path.is_file():
        raise TechnicalModelLoadError("model_not_found")
    try:
        bundle = joblib.load(path)
    except Exception as error:  # noqa: BLE001 - any unpickling failure is a load failure
        raise TechnicalModelLoadError("malformed_model") from error

    if not isinstance(bundle, TechnicalModelBundle):
        raise TechnicalModelLoadError("incompatible_model")
    if bundle.metadata.artifact_format_version != ARTIFACT_FORMAT_VERSION:
        raise TechnicalModelLoadError("incompatible_model")

    current_names = feature_names()
    if tuple(bundle.metadata.feature_names) != current_names:
        raise TechnicalModelLoadError("incompatible_model")

    if not hasattr(bundle.pipeline, "predict_proba"):
        raise TechnicalModelLoadError("incompatible_model")

    return bundle


__all__ = [
    "ARTIFACT_FORMAT_VERSION",
    "ModelMetadata",
    "TechnicalModelBundle",
    "TechnicalModelLoadError",
    "build_pipeline",
    "load_technical_model",
    "save_technical_model",
]
