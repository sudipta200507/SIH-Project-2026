"""Integration test against the real Dockerized Rspamd service.

Skipped automatically when Rspamd is not reachable (for example when the
container is stopped), so `pytest` still passes on machines without Docker.
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest

from app.authentication.dmarc import build_authentication_evidence
from app.authentication.rspamd_client import (
    RspamdClient,
    RspamdError,
    RspamdSettings,
)
from app.schemas.authentication import AuthenticationEvidence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SAMPLE_PATH = PROJECT_ROOT / "samples" / "safe" / "step1_synthetic.eml"


def _rspamd_reachable(base_url: str) -> bool:
    try:
        ping = httpx.get(f"{base_url}/ping", timeout=2.0)
    except httpx.TransportError:
        return False
    return ping.status_code == 200


RSPAMD_URL = os.environ.get("RSPAMD_URL", "http://127.0.0.1:11333")

pytestmark = pytest.mark.skipif(
    not _rspamd_reachable(RSPAMD_URL),
    reason="Rspamd container is not running (start it with: docker compose up -d rspamd)",
)


def test_live_rspamd_scans_original_synthetic_email() -> None:
    raw_email = SAMPLE_PATH.read_bytes()

    client = RspamdClient(RspamdSettings(url=RSPAMD_URL, timeout_seconds=30.0))
    analysis = client.scan_bytes(raw_email)

    # Structured response with the documented core fields.
    assert analysis.scanned is True
    assert analysis.action
    assert analysis.score is not None
    assert analysis.required_score is not None

    # The synthesized fixture carries no resolvable SPF/DKIM/DMARC evidence.
    # Whatever Rspamd reports must map into the controlled vocabulary without
    # ever being invented as a pass or a verdict.
    evidence: AuthenticationEvidence = build_authentication_evidence(analysis)
    assert evidence.rspamd.message_id or evidence.rspamd.action
    if evidence.spf.available == "unavailable":
        assert evidence.spf.result == "unknown"
    else:
        assert evidence.spf.result in {
            "pass", "fail", "softfail", "neutral", "none", "temperror", "permerror"
        }
    assert evidence.dkim.result in {
        "pass", "fail", "temperror", "permerror", "none", "unknown"
    }
    assert evidence.dmarc.result in {
        "pass", "fail", "quarantine", "reject", "none", "unknown"
    }

    # Evidence serializes cleanly for later phases.
    assert isinstance(evidence.model_dump(mode="json"), dict)


def test_live_rspamd_rejects_garbage_without_crashing() -> None:
    client = RspamdClient(RspamdSettings(url=RSPAMD_URL, timeout_seconds=30.0))
    try:
        analysis = client.scan_bytes(b"\x00\x01\x02 not an email at all \xff")
    except RspamdError:
        return  # a controlled error is acceptable for binary junk
    assert analysis.action  # or an RspamdError: never an uncontrolled crash
