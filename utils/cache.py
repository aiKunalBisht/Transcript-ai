# utils/cache.py — v2.0
# MD5 content-addressable cache — per-user namespaced.
#
# v2.0 changes (multi-user safety):
#   - user_id parameter added to _cache_key(), get_cached(), set_cache()
#   - Each user gets their own subdirectory: cache/users/{safe_user_id}/
#   - Anonymous / unauthenticated requests use "anonymous" namespace
#     (shared, same behaviour as v1 — safe because anon users are read-only
#      on HF Spaces; no personal data in their transcripts)
#   - filelock added to set_cache() — prevents corruption under workers=2+
#   - get_user_cache_stats() added — per-user entry count for dashboard
#
# DSA: O(n) hash per key, O(1) file stat for hit/miss check.
# No DB, no migrations — just files. At 65 users × 30 days = 1,950 files,
# each ~8KB → 15MB total. SQLite makes sense above ~10,000 entries.

import hashlib
import json
import os
import re
from pathlib import Path
from datetime import datetime, timedelta

try:
    import filelock as _fl
    _FILELOCK_AVAILABLE = True
except ImportError:
    _FILELOCK_AVAILABLE = False

CACHE_DIR  = Path("cache")
USER_DIR   = CACHE_DIR / "users"
ANON_DIR   = CACHE_DIR / "anonymous"
CACHE_TTL  = timedelta(hours=24)


def _safe_uid(user_id: str) -> str:
    """
    Sanitise user_id for use as a directory name.
    Google sub IDs look like '1234567890' — safe already.
    Email-based IDs may have @ and . — replace with underscore.
    Max 64 chars to stay well within filesystem limits.
    """
    return re.sub(r"[^a-zA-Z0-9_-]", "_", user_id)[:64]


def _user_cache_dir(user_id: str | None) -> Path:
    """
    Returns the cache directory for this user.
    None / empty string → anonymous shared cache (backwards-compatible).
    """
    if not user_id:
        return ANON_DIR
    return USER_DIR / _safe_uid(user_id)


def _cache_key(transcript: str, language: str) -> str:
    """MD5 hash of transcript + language. Same as v1 — no user_id in hash."""
    content = f"{language}::{transcript.strip()}"
    return hashlib.md5(content.encode("utf-8")).hexdigest()


# ── Public API ─────────────────────────────────────────────────────────────────

def get_cached(transcript: str, language: str,
               user_id: str | None = None) -> dict | None:
    """
    Returns cached result if it exists and is fresh, else None.

    user_id=None  → anonymous cache (shared, same as v1 behaviour)
    user_id="xyz" → private per-user cache — no cross-user contamination
    """
    key     = _cache_key(transcript, language)
    cache_d = _user_cache_dir(user_id)
    path    = cache_d / f"{key}.json"

    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            cached = json.load(f)

        cached_at = datetime.fromisoformat(cached.get("_cached_at", "2000-01-01"))
        if datetime.now() - cached_at > CACHE_TTL:
            path.unlink(missing_ok=True)
            return None

        cached["_from_cache"] = True
        return cached

    except Exception:
        return None


def set_cache(transcript: str, language: str, result: dict,
              user_id: str | None = None) -> None:
    """
    Stores result in the correct per-user (or anonymous) cache directory.

    Uses filelock when available to prevent write corruption under
    concurrent workers (workers=2 / Gunicorn).
    """
    cache_d = _user_cache_dir(user_id)
    cache_d.mkdir(parents=True, exist_ok=True)

    key  = _cache_key(transcript, language)
    path = cache_d / f"{key}.json"
    to_store = {**result, "_cached_at": datetime.now().isoformat()}

    def _write():
        with open(path, "w", encoding="utf-8") as f:
            json.dump(to_store, f, ensure_ascii=False, indent=2)

    try:
        if _FILELOCK_AVAILABLE:
            lock = _fl.FileLock(str(path) + ".lock", timeout=5)
            with lock:
                _write()
        else:
            _write()   # no lock — acceptable on workers=1
    except Exception:
        pass  # cache write failure is always non-fatal


def clear_user_cache(user_id: str) -> int:
    """Deletes all cached entries for a specific user. Returns count deleted."""
    cache_d = _user_cache_dir(user_id)
    if not cache_d.exists():
        return 0
    count = 0
    for f in cache_d.glob("*.json"):
        f.unlink(missing_ok=True)
        count += 1
    return count


def clear_cache() -> None:
    """Clears ALL cached results for all users (admin / maintenance use only)."""
    for d in [ANON_DIR, USER_DIR]:
        if d.exists():
            for f in d.rglob("*.json"):
                f.unlink(missing_ok=True)


def get_cache_stats(user_id: str | None = None) -> dict:
    """
    Returns cache statistics.
    user_id=None → global stats (all users combined).
    user_id=str  → stats for that specific user only.
    """
    if user_id:
        cache_d = _user_cache_dir(user_id)
        files   = list(cache_d.glob("*.json")) if cache_d.exists() else []
    else:
        # Global: anonymous + all user subdirectories
        files = []
        if ANON_DIR.exists():
            files += list(ANON_DIR.glob("*.json"))
        if USER_DIR.exists():
            files += list(USER_DIR.rglob("*.json"))

    size = sum(f.stat().st_size for f in files if f.exists())
    return {
        "entries":   len(files),
        "size_kb":   round(size / 1024, 1),
        "ttl_hours": CACHE_TTL.total_seconds() / 3600,
        "available": True,
    }


if __name__ == "__main__":
    # Self-test
    t = "Tanaka: Good morning. Let's review Q3."
    r = {"summary": ["Q3 reviewed"], "action_items": [], "_provider": "test"}

    set_cache(t, "en", r, user_id="user_abc")
    hit = get_cached(t, "en", user_id="user_abc")
    print("Per-user hit:         ", hit is not None)
    print("From cache flag:      ", hit.get("_from_cache"))

    miss = get_cached(t, "en", user_id="user_xyz")   # different user
    print("Cross-user miss:      ", miss is None)     # must be True

    anon_miss = get_cached(t, "en", user_id=None)     # anonymous
    print("Anon miss (no anon cache):", anon_miss is None)

    print("Stats (user_abc):     ", get_cache_stats("user_abc"))
    print("Stats (global):       ", get_cache_stats())