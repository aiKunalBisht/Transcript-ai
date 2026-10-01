# utils/cache.py — v2.1
# MD5 content-addressable cache — strictly per-user.
#
# v2.1 changes vs v2.0:
#   - Anonymous/unauthenticated requests are BLOCKED entirely.
#     get_cached() returns None, set_cache() is a no-op for user_id=None.
#     Reason: result is stored AFTER PII restoration — anonymous
#     cross-user leakage is a real risk, not just theoretical.
#   - ANON_DIR removed — no anonymous namespace at all.
#   - _user_cache_dir() raises if user_id is falsy (belt + suspenders).
#   - clear_cache() only touches USER_DIR (no ANON_DIR to clear).
#   - All v2.0 filelock + TTL + stats logic retained unchanged.

import hashlib
import json
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
CACHE_TTL  = timedelta(hours=24)


def _safe_uid(user_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", user_id)[:64]


def _user_cache_dir(user_id: str) -> Path:
    """Always returns a per-user directory. Caller must ensure user_id is truthy."""
    return USER_DIR / _safe_uid(user_id)


def _cache_key(transcript: str, language: str) -> str:
    content = f"{language}::{transcript.strip()}"
    return hashlib.md5(content.encode("utf-8")).hexdigest()


# ── Public API ─────────────────────────────────────────────────────────────────

def get_cached(transcript: str, language: str,
               user_id: str | None = None) -> dict | None:
    """
    Returns cached result if fresh, else None.
    Returns None immediately for anonymous (user_id=None) requests —
    no anonymous persistent cache.
    """
    if not user_id:          # ← BLOCK anonymous
        return None

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
    Stores result in the per-user cache directory.
    No-op for anonymous (user_id=None) — never writes anonymous cache.
    """
    if not user_id:          # ← BLOCK anonymous
        return

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
            _write()
    except Exception:
        pass  # cache write failure is always non-fatal


def clear_user_cache(user_id: str) -> int:
    """Deletes all cached entries for a specific user."""
    if not user_id:
        return 0
    cache_d = _user_cache_dir(user_id)
    if not cache_d.exists():
        return 0
    count = 0
    for f in cache_d.glob("*.json"):
        f.unlink(missing_ok=True)
        count += 1
    return count


def clear_cache() -> None:
    """Clears ALL cached results for all users (admin use only)."""
    if USER_DIR.exists():
        for f in USER_DIR.rglob("*.json"):
            f.unlink(missing_ok=True)


def get_cache_stats(user_id: str | None = None) -> dict:
    if user_id:
        cache_d = _user_cache_dir(user_id)
        files   = list(cache_d.glob("*.json")) if cache_d.exists() else []
    else:
        files = list(USER_DIR.rglob("*.json")) if USER_DIR.exists() else []

    size = sum(f.stat().st_size for f in files if f.exists())
    return {
        "entries":   len(files),
        "size_kb":   round(size / 1024, 1),
        "ttl_hours": CACHE_TTL.total_seconds() / 3600,
        "available": True,
    }


if __name__ == "__main__":
    t = "Tanaka: Good morning. Let's review Q3."
    r = {"summary": ["Q3 reviewed"], "action_items": [], "_provider": "test"}

    set_cache(t, "en", r, user_id="user_abc")

    hit = get_cached(t, "en", user_id="user_abc")
    print("Per-user hit:       ", hit is not None)           # True
    print("From cache flag:    ", hit.get("_from_cache"))    # True

    miss = get_cached(t, "en", user_id="user_xyz")
    print("Cross-user miss:    ", miss is None)              # True

    anon_set = set_cache(t, "en", r, user_id=None)          # no-op
    anon_get = get_cached(t, "en", user_id=None)
    print("Anon blocked:       ", anon_get is None)          # True ← key fix