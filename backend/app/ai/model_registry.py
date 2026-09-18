"""Process-wide model registry with cached loading (Phase O).

Models are loaded ONCE per artifact version and reused across requests:

- the cache key includes the artifact path plus its size/mtime, so a
  retrained model is picked up without a process restart while identical
  artifacts are never re-loaded per request;
- load FAILURES are cached too, so a missing artifact does not cause a
  filesystem probe on every request;
- loading is strictly local — inference never downloads model weights;
- no training or heavy work happens at import time or API startup.

This module is intentionally the ONLY place that loads model artifacts for
inference, so per-request loading can be verified by construction.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

from app.ai.nlp.model import NLPModelBundle, NLPModelLoadError, load_nlp_model
from app.ai.technical_ml.model import (
    TechnicalModelBundle,
    TechnicalModelLoadError,
    load_technical_model,
)

_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class _CacheEntry:
    version: tuple  # (path, size, mtime_ns) — None when the artifact was absent
    bundle: object | None
    error_reason: str | None


_technical_cache: dict[str, _CacheEntry] = {}
_nlp_cache: dict[str, _CacheEntry] = {}


def _artifact_version(path: Path) -> tuple | None:
    """Cache-busting version of an artifact, or None when it does not exist."""

    try:
        stat = path.stat()
    except OSError:
        return None
    return (str(path), stat.st_size, stat.st_mtime_ns)


def _cached_load(
    cache: dict[str, _CacheEntry],
    cache_key: str,
    artifact_path: Path,
    loader,
) -> tuple[object | None, str | None]:
    """Shared cached-load logic; returns (bundle|None, error_reason|None)."""

    version = _artifact_version(artifact_path)
    with _LOCK:
        entry = cache.get(cache_key)
        if entry is not None and entry.version == version:
            return entry.bundle, entry.error_reason

    bundle: object | None = None
    error_reason: str | None = None
    if version is None:
        error_reason = "model_not_found"
    else:
        try:
            bundle = loader(artifact_path)
        except (TechnicalModelLoadError, NLPModelLoadError) as error:
            error_reason = error.reason
        except Exception as error:  # noqa: BLE001 - defensive: never propagate
            error_reason = getattr(error, "reason", None) or "model_not_found"

    with _LOCK:
        cache[cache_key] = _CacheEntry(
            version=version, bundle=bundle, error_reason=error_reason
        )
    return bundle, error_reason


def get_technical_bundle(path: Path) -> tuple[TechnicalModelBundle | None, str | None]:
    """Cached technical-model load: (bundle, error_reason)."""

    key = f"technical::{path}"
    bundle, reason = _cached_load(
        _technical_cache, key, path, load_technical_model
    )
    return bundle, reason


def get_nlp_bundle(
    path: Path,
    *,
    model_name: str,
    max_length_tokens: int = 512,
) -> tuple[NLPModelBundle | None, str | None]:
    """Cached NLP-model load: (bundle, error_reason)."""

    key = f"nlp::{path}::{model_name}::{max_length_tokens}"
    loader = lambda p: load_nlp_model(  # noqa: E731 - small local adapter
        p, model_name=model_name, max_length_tokens=max_length_tokens
    )
    bundle, reason = _cached_load(_nlp_cache, key, path, loader)
    return bundle, reason


def clear_caches() -> None:
    """Test helper: drop all cached bundles (not used during inference)."""

    with _LOCK:
        _technical_cache.clear()
        _nlp_cache.clear()


__all__ = [
    "clear_caches",
    "get_nlp_bundle",
    "get_technical_bundle",
]
