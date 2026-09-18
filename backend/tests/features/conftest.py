"""Shared fixtures for Step 5 feature tests."""

from __future__ import annotations

import pytest

from ai_fixtures.helpers import sample_evidence  # noqa: F401


@pytest.fixture()
def evidence():
    """Step 1 evidence for the synthetic sample."""

    return sample_evidence()
