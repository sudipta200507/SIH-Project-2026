#!/usr/bin/env python
"""Generate the synthetic technical-model FIXTURE dataset.

The fixture exists ONLY to validate the training/saving/loading/inference
pipeline mechanically. Rows are generated from two simple, fully synthetic
rules (clearly separable by design):

- ``benign`` rows: authenticated mail, no URL/credential/payment signals,
  consistent sender/return-path/message-id domains;
- ``suspicious`` rows: failed authentication, credential/payment/urgency
  keywords, IP-literal URLs, mismatched sender/return-path domains.

The class-separating rules are deliberately artificial. Accuracy measured on
this fixture is a PIPELINE CHECK ONLY and is never real-world performance.

Usage (from the ``backend`` directory, venv active):

    python ../ai/training/generate_technical_fixture.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[2] / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import json
import random

from app.schemas.features import FeatureVector, feature_names

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "ai" / "datasets" / "processed" / "technical_fixture.json"
ROWS_PER_CLASS = 60


def _benign_vector(rng: random.Random) -> FeatureVector:
    return FeatureVector.model_validate(
        {
            "text": {
                "subject_length": rng.randint(20, 70),
                "subject_exclamation_count": 0,
                "subject_urgency_signal": 0,
                "subject_money_signal": 0,
                "body_plain_length": rng.randint(200, 4000),
                "body_html_present": rng.choice([0, 1]),
                "body_html_length": rng.randint(0, 20000),
                "credential_signal_count": 0,
                "payment_signal_count": rng.choice([0, 1]),
                "urgency_signal_count": rng.choice([0, 0, 1]),
            },
            "header": {
                "received_hop_count": rng.randint(2, 6),
                "parser_defect_count": 0,
                "sender_reply_to_domain_mismatch": 0,
                "return_path_sender_mismatch": 0,
                "message_id_domain_mismatch": 0,
                "message_id_missing": 0,
                "date_missing": 0,
                "reply_to_present": rng.choice([0, 1]),
                "user_agent_present": 1,
                "mime_multipart": rng.choice([0, 1]),
            },
            "auth": {
                "spf_pass": 1,
                "dkim_pass": 1,
                "dmarc_pass": 1,
                "rspamd_available": 1,
                "rspamd_score": round(rng.uniform(-3.0, 1.0), 2),
            },
            "url": {
                "url_count": rng.randint(0, 2),
                "unique_url_host_count": rng.randint(0, 2),
                "https_ratio": 1.0,
                "ip_url_count": 0,
                "suspicious_port_count": 0,
                "max_url_path_length": rng.randint(0, 40),
                "unique_url_registered_domain_count": rng.randint(0, 1),
                "url_registered_domain_mismatch_count": 0,
            },
            "intelligence": {
                "public_ip_count": rng.randint(1, 4),
                "non_public_ip_count": 0,
                "dns_success_count": rng.randint(1, 3),
                "dns_unavailable_count": 0,
                "rdap_success_count": rng.randint(1, 2),
                "geoip_available_count": 0,
                "min_domain_age_days": rng.uniform(365, 4000),
            },
        }
    )


def _suspicious_vector(rng: random.Random) -> FeatureVector:
    return FeatureVector.model_validate(
        {
            "text": {
                "subject_length": rng.randint(40, 120),
                "subject_exclamation_count": rng.randint(1, 5),
                "subject_urgency_signal": 1,
                "subject_money_signal": rng.choice([0, 1]),
                "body_plain_length": rng.randint(300, 3000),
                "body_html_present": 1,
                "body_html_length": rng.randint(5000, 60000),
                "credential_signal_count": rng.randint(1, 4),
                "payment_signal_count": rng.randint(1, 3),
                "urgency_signal_count": rng.randint(1, 5),
            },
            "header": {
                "received_hop_count": rng.randint(1, 8),
                "parser_defect_count": rng.choice([0, 1, 2]),
                "sender_reply_to_domain_mismatch": rng.choice([0, 1, 1]),
                "return_path_sender_mismatch": rng.choice([0, 1, 1]),
                "message_id_domain_mismatch": rng.choice([0, 1]),
                "message_id_missing": 0,
                "date_missing": 0,
                "reply_to_present": 1,
                "user_agent_present": rng.choice([0, 1]),
                "mime_multipart": 1,
            },
            "auth": {
                "spf_fail": 1,
                "dkim_fail": 1,
                "dmarc_fail": 1,
                "rspamd_available": 1,
                "rspamd_score": round(rng.uniform(4.0, 12.0), 2),
            },
            "url": {
                "url_count": rng.randint(1, 5),
                "unique_url_host_count": rng.randint(1, 4),
                "https_ratio": round(rng.uniform(0.0, 0.5), 2),
                "ip_url_count": rng.choice([0, 1, 2]),
                "suspicious_port_count": rng.choice([0, 1]),
                "max_url_path_length": rng.randint(20, 200),
                "unique_url_registered_domain_count": rng.randint(1, 3),
                "url_registered_domain_mismatch_count": rng.choice([0, 1, 2]),
            },
            "intelligence": {
                "public_ip_count": rng.randint(0, 3),
                "non_public_ip_count": rng.choice([0, 1]),
                "dns_success_count": rng.randint(0, 2),
                "dns_unavailable_count": rng.choice([0, 1]),
                "rdap_success_count": rng.choice([0, 1]),
                "geoip_available_count": 0,
                "min_domain_age_days": rng.uniform(1, 60),
            },
        }
    )


def main() -> int:
    rng = random.Random(42)  # deterministic fixture generation
    rows = []
    for _ in range(ROWS_PER_CLASS):
        rows.append({"features": _benign_vector(rng).model_dump(), "label": "benign"})
    for _ in range(ROWS_PER_CLASS):
        rows.append(
            {"features": _suspicious_vector(rng).model_dump(), "label": "suspicious"}
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": (
            "SYNTHETIC FIXTURE for technical-model pipeline validation only. "
            "Generated deterministically by generate_technical_fixture.py. "
            "NOT a real dataset; accuracy on it is not real-world performance."
        ),
        "feature_names": list(feature_names()),
        "rows": rows,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} synthetic fixture rows to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
