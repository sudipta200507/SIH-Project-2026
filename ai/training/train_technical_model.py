#!/usr/bin/env python
"""Train the technical (classical ML) model — Phase D training pipeline.

HONESTY CONTRACT
================
This script refuses to train on real data that has not been supplied by the
project. It supports two modes:

``--use-fixture``
    Trains on the tiny, clearly-labeled synthetic FIXTURE dataset
    (``ai/datasets/processed/technical_fixture.json``) that exists ONLY to
    validate the training/saving/loading/inference pipeline. The resulting
    artifact is written to ``ai/models/technical_model_fixture.joblib`` and
    its metadata states ``dataset_description="pipeline validation fixture
    (synthetic)"``. Its performance numbers are NOT real-world accuracy and
    must never be reported as such.

default mode (no fixture)
    Requires a real labeled dataset in the layout documented by
    ``ai/datasets/README.md``. If none exists, the script exits with a
    clear message instead of fabricating anything.

Usage (from the ``backend`` directory, venv active):

    python ../ai/training/train_technical_model.py --use-fixture
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Path bootstrap: allow running as a script from the repository.
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[2] / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import argparse
import json
from datetime import datetime, timezone

from app.ai.technical_ml.model import (
    ModelMetadata,
    TechnicalModelBundle,
    build_pipeline,
    save_technical_model,
)
from app.schemas.features import FeatureVector, feature_names

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = PROJECT_ROOT / "ai" / "datasets" / "processed" / "technical_fixture.json"
FIXTURE_MODEL_PATH = PROJECT_ROOT / "ai" / "models" / "technical_model_fixture.joblib"
PRODUCTION_MODEL_PATH = PROJECT_ROOT / "ai" / "models" / "technical_model.joblib"

PRODUCTION_DATASET_HINT = (
    "Place a labeled dataset (feature vectors + labels) under "
    "ai/datasets/external/ and register it in ai/datasets/README.md."
)


def _load_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise SystemExit(f"{path} must contain a non-empty 'rows' list.")
    return rows


def _rows_to_xy(rows: list[dict]) -> tuple[list[list[float]], list[str]]:
    names = list(feature_names())
    X: list[list[float]] = []
    y: list[str] = []
    for index, row in enumerate(rows):
        fv = row.get("features")
        label = row.get("label")
        if not isinstance(fv, dict) or label not in {"suspicious", "benign"}:
            raise SystemExit(f"Row {index} is malformed (needs features + label).")
        values: list[float] = []
        for name in names:
            section, field_name = name.split(".", maxsplit=1)
            section_data = fv.get(section)
            if not isinstance(section_data, dict) or field_name not in section_data:
                raise SystemExit(f"Row {index} is missing feature {name}.")
            value = float(section_data[field_name])
            if value != value or value in (float("inf"), float("-inf")):
                raise SystemExit(f"Row {index} feature {name} is not finite.")
            values.append(value)
        X.append(values)
        y.append(str(label))
    return X, y


def train(
    X: list[list[float]],
    y: list[str],
    *,
    dataset_description: str,
    training_notes: str,
) -> TechnicalModelBundle:
    """Fit the pipeline and wrap it in a metadata-carrying bundle."""

    pipeline = build_pipeline()
    pipeline.fit(X, y)
    metadata = ModelMetadata(
        model_name="forentisai-technical-rf-v1",
        class_labels=tuple(sorted(set(y))),
        feature_names=tuple(feature_names()),
        trained_at=datetime.now(timezone.utc).isoformat(),
        dataset_description=dataset_description,
        training_notes=training_notes,
    )
    return TechnicalModelBundle(pipeline=pipeline, metadata=metadata)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--use-fixture",
        action="store_true",
        help="Train on the tiny synthetic pipeline-validation fixture.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to a real labeled JSON dataset (default mode).",
    )
    args = parser.parse_args()

    if args.use_fixture:
        if not FIXTURE_PATH.is_file():
            raise SystemExit(
                "Fixture dataset not found. Generate it first:\n"
                "  python ../ai/training/generate_technical_fixture.py"
            )
        rows = _load_rows(FIXTURE_PATH)
        dataset_description = "pipeline validation fixture (synthetic)"
        training_notes = (
            "TRAINED ON A SYNTHETIC FIXTURE. Exists only to validate the "
            "training/serving pipeline end-to-end. NOT production accuracy; "
            "NOT a real-world phishing detector. Requires retraining on a "
            "legitimate labeled dataset before any real use."
        )
        output_path = FIXTURE_MODEL_PATH
    else:
        dataset_path = args.dataset
        if dataset_path is None:
            raise SystemExit(
                "No real labeled dataset is configured.\n" + PRODUCTION_DATASET_HINT
            )
        if not dataset_path.is_file():
            raise SystemExit(f"Dataset not found: {dataset_path}")
        rows = _load_rows(dataset_path)
        dataset_description = f"project-supplied dataset: {dataset_path.name}"
        training_notes = (
            "Trained via ai/training/train_technical_model.py on a "
            "project-supplied labeled dataset. See ai/datasets/README.md for "
            "the documented source and labeling methodology."
        )
        output_path = PRODUCTION_MODEL_PATH

    X, y = _rows_to_xy(rows)

    if len(set(y)) < 2:
        raise SystemExit(
            "Training data must contain both classes (suspicious and benign); "
            "refusing to train a degenerate model."
        )

    bundle = train(
        X,
        y,
        dataset_description=dataset_description,
        training_notes=training_notes,
    )
    save_technical_model(bundle, output_path)

    # Holdout evaluation is reported for the fixture honestly: it validates
    # mechanics only, never real-world accuracy.
    holdout_fraction = 0.2 if args.use_fixture else 0.2
    if len(rows) >= 10:
        split = int(len(X) * (1 - holdout_fraction))
        X_train, y_train = X[:split], y[:split]
        X_hold, y_hold = X[split:], y[split:]
        eval_bundle = train(
            X_train,
            y_train,
            dataset_description=dataset_description,
            training_notes=training_notes,
        )
        accuracy = float(eval_bundle.pipeline.score(X_hold, y_hold))
        print(f"[honesty] holdout accuracy on {dataset_description}: {accuracy:.3f}")
        print(
            "[honesty] fixture accuracy is a PIPELINE CHECK ONLY — "
            "it is not real-world model performance."
            if args.use_fixture
            else "[honesty] holdout accuracy is indicative only; see "
            "ai/datasets/README.md for dataset provenance."
        )

    print(f"Saved technical model bundle to {output_path}")
    print(f"Classes: {bundle.metadata.class_labels}")
    print(f"Features: {len(bundle.metadata.feature_names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
