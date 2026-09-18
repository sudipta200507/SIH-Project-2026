"""Fixtures for the Step 4 intelligence tests (fakes only, no network)."""

from __future__ import annotations

import pytest

from app.intelligence.rdap import clear_bootstrap_cache


@pytest.fixture(autouse=True)
def _fresh_bootstrap_cache():
    """Keep RDAP bootstrap cache state out of the unit tests."""

    clear_bootstrap_cache()
    yield
    clear_bootstrap_cache()


@pytest.fixture()
def sample():
    """Normalized Step 1 evidence for the safe synthetic sample."""

    from fixtures import sample_evidence

    return sample_evidence()
