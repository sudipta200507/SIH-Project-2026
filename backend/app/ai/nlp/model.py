"""NLP model artifact handling (Phase E).

CRITICAL HONESTY RULE
=====================
The base pretrained ``microsoft/deberta-v3-base`` model is NOT a phishing
classifier. Only a project-specific FINE-TUNED artifact produced by
``ai/training/train_transformer.py`` may produce predictions here.

Rules enforced by this module:

- A valid fine-tuned artifact must carry the ForentisAI metadata marker
  (``FORENTISAI_NLP_MARKER``) inside its config ``model_type``-independent
  ``finetuned_from``-style custom attribute written by the training script.
  An artifact without the marker is treated as a base model and is NEVER
  used for prediction.
- Loading is strictly ``local_files_only=True``: inference must never
  download models during an API request (Phase O). A missing artifact is a
  controlled ``unavailable`` state, not a download trigger.
- Every failure mode maps to a :class:`NLPModelLoadError` reason that the
  inference layer converts into an explicit result state.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

FORENTISAI_NLP_MARKER = "forentisai_nlp_finetuned_v1"
DEFAULT_CLASS_LABELS: tuple[str, ...] = ("benign", "suspicious")


class NLPModelLoadError(Exception):
    """Raised when the NLP artifact cannot be loaded or is not fine-tuned."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class NLPModelBundle:
    """A loaded fine-tuned tokenizer+model pair plus its metadata."""

    tokenizer: object
    model: object
    model_name: str
    class_labels: tuple[str, ...]
    max_length_tokens: int


def _read_marker(model_path: Path) -> str | None:
    """Read the ForentisAI fine-tune marker from config.json, if present."""

    config_path = model_path / "config.json"
    if not config_path.is_file():
        return None
    try:
        import json

        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - malformed config is a load failure
        return None
    marker = config.get("_forentisai_finetune_marker")
    return marker if isinstance(marker, str) else None


def _read_class_labels(model_path: Path) -> tuple[str, ...] | None:
    """Read id->label mappings written by the training script."""

    config_path = model_path / "config.json"
    if not config_path.is_file():
        return None
    try:
        import json

        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    id2label = config.get("id2label")
    if not isinstance(id2label, dict) or not id2label:
        return None
    try:
        labels = tuple(str(id2label[key]) for key in sorted(id2label, key=int))
    except (KeyError, TypeError, ValueError):
        return None
    return labels


def load_nlp_model(
    model_path: Path,
    *,
    model_name: str = "unknown",
    max_length_tokens: int = 512,
) -> NLPModelBundle:
    """Load a fine-tuned artifact locally; never downloads.

    Raises :class:`NLPModelLoadError` with one of: ``model_not_found``,
    ``dependency_unavailable``, ``model_not_trained`` (no ForentisAI
    marker → base/unmarked model), ``incompatible_model``.
    """

    model_path = Path(model_path)
    if not model_path.is_dir():
        raise NLPModelLoadError("model_not_found")

    marker = _read_marker(model_path)
    if marker is None:
        # No marker: this is a base or foreign model. It is NOT a phishing
        # classifier, so we refuse to predict with it.
        raise NLPModelLoadError("model_not_trained")
    if marker != FORENTISAI_NLP_MARKER:
        raise NLPModelLoadError("incompatible_model")

    labels = _read_class_labels(model_path)
    if labels is None or set(labels) != set(DEFAULT_CLASS_LABELS):
        raise NLPModelLoadError("incompatible_model")

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as error:  # noqa: BLE001 - missing/broken dependency
        raise NLPModelLoadError("dependency_unavailable") from error

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path), local_files_only=True
        )
    except Exception as error:  # noqa: BLE001 - corrupted artifact, missing weights
        raise NLPModelLoadError("model_not_found") from error

    model.eval()  # type: ignore[union-attr]

    return NLPModelBundle(
        tokenizer=tokenizer,
        model=model,
        model_name=model_name,
        class_labels=labels,
        max_length_tokens=max_length_tokens,
    )


__all__ = [
    "DEFAULT_CLASS_LABELS",
    "FORENTISAI_NLP_MARKER",
    "NLPModelBundle",
    "NLPModelLoadError",
    "load_nlp_model",
]
