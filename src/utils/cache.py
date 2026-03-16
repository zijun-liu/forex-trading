from __future__ import annotations

from pathlib import Path
from typing import Any

import diskcache

_ROOT = Path(__file__).resolve().parents[2]
_CACHE_DIR = _ROOT / ".cache"

_cache: diskcache.Cache | None = None


def get_cache() -> diskcache.Cache:
    global _cache
    if _cache is None:
        _cache = diskcache.Cache(str(_CACHE_DIR))
    return _cache


_MISSING = object()


def cached_get(key: str, fn, ttl_seconds: int = 3600) -> Any:
    """Return cached value for *key*, or call *fn()* and cache the result."""
    cache = get_cache()
    val = cache.get(key, default=_MISSING)
    if val is not _MISSING:
        return val
    val = fn()
    cache.set(key, val, expire=ttl_seconds)
    return val
