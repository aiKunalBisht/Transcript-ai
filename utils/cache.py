# cache.py
# Caching layer — prevents recomputing identical transcripts
# Zero dependencies — pure Python, MD5 hash key, JSON storage

import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timedelta

CACHE_DIR  = Path("cache")
CACHE_TTL  = timedelta(hours=24)   # cache expires after 24 hours


def _cache_key(transcript: str, language: str) -> str:
    """MD5 hash of transcript + language = cache key."""
    content = f"{language}::{transcript.strip()}"
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def get_cached(transcript: str, language: str) -> dict | None:
    """
    Returns cached result if it exists and is fresh.
    Returns None if cache miss or expired.
    """
    key  = _cache_key(transcript, language)
    path = CACHE_DIR / f"{key}.json"

    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            cached = json.load(f)

        cached_at = datetime.fromisoformat(cached.get("_cached_at", "2000-01-01"))
        if datetime.now() - cached_at > CACHE_TTL:
            path.unlink()   # delete expired cache
            return None

        cached["_from_cache"] = True
        return cached

    except Exception:
        return None


def set_cache(transcript: str, language: str, result: dict):
    """Stores result in cache with timestamp."""
    CACHE_DIR.mkdir(exist_ok=True)
    key  = _cache_key(transcript, language)
    path = CACHE_DIR / f"{key}.json"

    to_store = {**result, "_cached_at": datetime.now().isoformat()}
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(to_store, f, ensure_ascii=False, indent=2)
    except Exception:
        pass  # cache write failure is non-fatal


def clear_cache():
    """Clears all cached results."""
    if CACHE_DIR.exists():
        for f in CACHE_DIR.glob("*.json"):
            f.unlink()


def get_cache_stats() -> dict:
    """Returns cache statistics."""
    if not CACHE_DIR.exists():
        return {"entries": 0, "size_kb": 0}
    files = list(CACHE_DIR.glob("*.json"))
    size  = sum(f.stat().st_size for f in files)
    return {
        "entries":  len(files),
        "size_kb":  round(size / 1024, 1),
        "ttl_hours": CACHE_TTL.total_seconds() / 3600
    }


if __name__ == "__main__":
    import json
    transcript = "Tanaka: Good morning. Let's review Q3."
    result = {"summary": ["Q3 reviewed"], "action_items": [], "_provider": "test"}

    set_cache(transcript, "en", result)
    cached = get_cached(transcript, "en")
    print("Cache hit:", cached is not None)
    print("From cache flag:", cached.get("_from_cache"))
    print("Stats:", get_cache_stats())