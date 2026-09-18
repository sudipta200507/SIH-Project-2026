"""Feature fusion: assemble the complete deterministic FeatureVector.

The orchestrator (Phase I) calls :func:`build_feature_vector` with the
already-computed Step 1–4 evidence objects. Feature extraction never
re-parses the original email, never performs network requests, and never
embeds raw email text.
"""

from __future__ import annotations

from app.schemas.authentication import AuthenticationEvidence
from app.schemas.email import EmailEvidence
from app.schemas.features import FeatureVector, feature_names
from app.schemas.intelligence import IntelligenceEvidence

from app.features.auth_features import extract_auth_features
from app.features.header_features import extract_header_features
from app.features.intelligence_features import extract_intelligence_features
from app.features.text_features import extract_text_features
from app.features.url_features import extract_url_features

__all__ = ["build_feature_vector", "feature_names"]


def build_feature_vector(
    evidence: EmailEvidence,
    authentication: AuthenticationEvidence | None = None,
    intelligence: IntelligenceEvidence | None = None,
    *,
    url_intelligence: list | None = None,
) -> FeatureVector:
    """Assemble the deterministic feature vector from Steps 1–4 evidence.

    Every sub-extractor is exception-safe; a failure in one section yields
    that section's neutral defaults rather than propagating an error.
    """

    vector = FeatureVector()

    try:
        vector.text = extract_text_features(evidence)
    except Exception:  # noqa: BLE001 - isolation by design
        pass
    try:
        vector.header = extract_header_features(evidence)
    except Exception:  # noqa: BLE001
        pass
    if authentication is not None:
        try:
            vector.auth = extract_auth_features(authentication)
        except Exception:  # noqa: BLE001
            pass
    try:
        vector.url = extract_url_features(evidence, url_intelligence=url_intelligence)
    except Exception:  # noqa: BLE001
        pass
    if intelligence is not None:
        try:
            vector.intelligence = extract_intelligence_features(
                intelligence, email_date=evidence.message.date
            )
        except Exception:  # noqa: BLE001
            pass

    return vector
