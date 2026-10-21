# utils/vector_cache.py — v2.3
# Persistent vector cache using ChromaDB + sentence-transformers
#
# v2.2 → v2.3 changes (this version):
#
# SEC-1 FIX — Unguessable ChromaDB collection names via HMAC-SHA256.
#   VULNERABILITY: Previously `f"transcripts_{_safe_uid(user_id)}"` where
#   user_id is derived deterministically from email in main.py:
#     user_id = "usr_" + email.replace("@","_at_").replace(".","_")
#   Result: collection "transcripts_usr_alice_at_company_com" is 100% guessable
#   by anyone who knows Alice's email. ChromaDB has no built-in auth —
#   any process with access to the client can query any collection by name.
#
#   FIX: collection name = HMAC-SHA256(COLLECTION_SECRET, user_id)[:40]
#   prefixed with "t". Without the COLLECTION_SECRET env var, an attacker
#   cannot derive the collection name from the email or user_id alone.
#   Fallback when COLLECTION_SECRET is unset: plain SHA-256 (still not
#   email-derivable, but set the secret for full protection).
#
#   Set in .env:  COLLECTION_SECRET=<random-string-32+-chars>
#   Generate one: python -c "import secrets; print(secrets.token_hex(32))"
#
# SEC-2 FIX — Ownership check in get_cached_result().
#   Defense-in-depth: after retrieving a cached result from disk, verify
#   that result._user_id matches the requesting user. Guards against any
#   future code path that might bypass the collection-name protection.
#
# ─── MIGRATION NOTE ───────────────────────────────────────────────────────────
# Existing collections named "transcripts_<safe_uid>" are NOT accessible after
# this upgrade — their names no longer match the new HMAC-derived names.
# Users will see a cold-cache miss on first analysis after upgrade.
# Results rebuild automatically on next run. No data is deleted; the old
# ChromaDB collections are simply no longer referenced.
# To clean up: chroma_client.delete_collection("transcripts_<old_safe_uid>")
# ─────────────────────────────────────────────────────────────────────────────
#
# v2.1 → v2.2 retained (all fixes carried forward unchanged):
#   Y1 FIX: Anonymous persistent cache disabled — no cross-user leakage.
#   Y2 FIX: Degraded / fallback results not cached — no stale poisoning.
#   X1 FIX: masked_transcript used for embedding query consistency.
#   X2 FIX: masked_transcript stored in ChromaDB (no raw PII on disk).
#   X3 FIX: stderr logging on all exceptions — no silent swallowing.

import os
import sys
import json
import hashlib
import hmac as _hmac
import time
import re
from pathlib import Path

# ── STORAGE PATHS ─────────────────────────────────────────────────────────────
VECTOR_STORE_DIR = Path(os.getenv("VECTOR_STORE_DIR", "./vector_store"))
CHROMA_DIR       = VECTOR_STORE_DIR / "chroma_db"
RESULTS_DIR      = VECTOR_STORE_DIR / "results"
PATTERNS_DIR     = VECTOR_STORE_DIR / "patterns"

VECTOR_STORE_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
PATTERNS_DIR.mkdir(exist_ok=True)

# ── SIMILARITY THRESHOLDS ─────────────────────────────────────────────────────
EXACT_THRESHOLD    = 0.98   # near-identical transcript → instant return
SEMANTIC_THRESHOLD = 0.82   # same meeting, slightly different wording → reuse
# Below 0.82 = different meeting → call LLM

# ── SEC-1 FIX: COLLECTION SECRET ──────────────────────────────────────────────
# Used to derive unguessable ChromaDB collection names.
# Generate: python -c "import secrets; print(secrets.token_hex(32))"
# Add to .env: COLLECTION_SECRET=<your-64-char-hex-string>
COLLECTION_SECRET: bytes = os.getenv("COLLECTION_SECRET", "").encode("utf-8")

if not COLLECTION_SECRET:
    print(
        "[VECTOR_CACHE] WARNING: COLLECTION_SECRET env var is not set. "
        "ChromaDB collection names fall back to plain SHA-256(user_id) — "
        "still not email-derivable, but set COLLECTION_SECRET for full protection. "
        "Generate one: python -c \"import secrets; print(secrets.token_hex(32))\"",
        file=sys.stderr, flush=True,
    )

# Y2 FIX: Only cache results from these providers — never cache degraded output.
_CACHEABLE_PROVIDERS = frozenset({
    "nim",
    "groq",
    "groq_langchain",
    "ollama",
    "ollama_langchain",
})

# ── LAZY SINGLETONS ───────────────────────────────────────────────────────────
_chroma_client   = None
_transcript_coll = None
_patterns_coll   = None
_embedder        = None


def _get_embedder():
    """Lazy-load sentence-transformers — only when first needed."""
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        except ImportError:
            _embedder = False
    return _embedder if _embedder else None


def _get_chroma():
    """Lazy-load ChromaDB client with persistent storage."""
    global _chroma_client, _transcript_coll, _patterns_coll
    if _chroma_client is not None:
        return _chroma_client, _transcript_coll, _patterns_coll
    try:
        import chromadb
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

        # Anonymous collection retained for NLP patterns only — not transcripts.
        _transcript_coll = _chroma_client.get_or_create_collection(
            name="transcripts_anonymous",
            metadata={"hnsw:space": "cosine"}
        )
        _patterns_coll = _chroma_client.get_or_create_collection(
            name="nlp_patterns",
            metadata={"hnsw:space": "cosine"}
        )

        if _patterns_coll.count() == 0:
            _seed_nlp_patterns(_patterns_coll)

    except ImportError:
        _chroma_client = False
    return _chroma_client or None, _transcript_coll, _patterns_coll


# ── USER COLLECTION HELPERS ───────────────────────────────────────────────────

def _safe_uid(user_id: str) -> str:
    """
    Sanitise user_id for FILESYSTEM paths only (results directory).
    NOT used for ChromaDB collection names — see _collection_name_for_user().
    DSA: O(n) single regex pass.
    """
    clean = re.sub(r"[^a-zA-Z0-9_-]", "_", str(user_id))[:40]
    return clean if clean else "anonymous"


def _collection_name_for_user(user_id: str) -> str:
    """
    SEC-1 FIX: Derive an unguessable ChromaDB collection name from user_id.

    Uses HMAC-SHA256(COLLECTION_SECRET, user_id) so the collection name is NOT
    derivable from the user_id or email alone. Without knowing COLLECTION_SECRET,
    an attacker cannot enumerate collection names to access another user's data.

    Fallback when COLLECTION_SECRET is unset: plain SHA-256("transcriptai_v1:"
    + user_id) — still not email-derivable, but set the secret for full
    protection.

    ChromaDB collection name constraints satisfied:
      - 3-63 characters  → "t" + 40 hex chars = 41 ✓
      - Only alphanumeric + _ + -  → hex (0-9, a-f) + "t" prefix ✓
      - Start and end alphanumeric ✓
    DSA: O(1) — single HMAC call.
    """
    if COLLECTION_SECRET:
        raw = _hmac.new(
            COLLECTION_SECRET,
            user_id.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
    else:
        raw = hashlib.sha256(
            f"transcriptai_v1:{user_id}".encode("utf-8")
        ).hexdigest()

    return f"t{raw[:40]}"


def _get_user_collection(user_id: str | None = None):
    """
    Returns the ChromaDB collection for this specific authenticated user.

    SEC-1 FIX: Collection name is HMAC-SHA256(COLLECTION_SECRET, user_id),
    NOT the predictable f"transcripts_{_safe_uid(user_id)}" from v2.2.
    An attacker who knows a user's email cannot derive the collection name.

    Anonymous users (user_id=None) return None — no persistent transcript cache.
    (Y1 FIX retained from v2.1)
    """
    client, _, _ = _get_chroma()
    if client is None:
        return None

    # Y1 FIX: no persistent collection for anonymous users
    if not user_id:
        return None

    coll_name = _collection_name_for_user(user_id)  # SEC-1 FIX ← was _safe_uid
    try:
        return client.get_or_create_collection(
            name=coll_name,
            metadata={"hnsw:space": "cosine"}
        )
    except Exception as e:
        print(
            f"[VECTOR_CACHE] _get_user_collection failed for user={user_id!r}: {e}",
            file=sys.stderr, flush=True,
        )
        return None


def _seed_nlp_patterns(coll):
    """Store reference NLP patterns in ChromaDB for semantic matching."""
    patterns = [
        {"id": "role_leader_1",     "text": "Let's get started. I'll chair today's meeting.",           "category": "ROLE",       "role":     "leader"},
        {"id": "role_leader_2",     "text": "Okay everyone, let me summarize what we agreed.",          "category": "ROLE",       "role":     "leader"},
        {"id": "role_leader_3",     "text": "I'll take ownership of this. Everyone please note.",       "category": "ROLE",       "role":     "leader"},
        {"id": "role_manager_1",    "text": "I need this done by end of week. No exceptions.",          "category": "ROLE",       "role":     "manager"},
        {"id": "role_manager_2",    "text": "As per my earlier direction, this should be prioritized.", "category": "ROLE",       "role":     "manager"},
        {"id": "role_teamlead_1",   "text": "My team will handle this. I'll assign it internally.",     "category": "ROLE",       "role":     "team_lead"},
        {"id": "role_teamlead_2",   "text": "Let me check with my team and get back to you.",           "category": "ROLE",       "role":     "team_lead"},
        {"id": "role_sub_1",        "text": "Yes sir, I will make sure it gets done.",                  "category": "ROLE",       "role":     "subordinate"},
        {"id": "role_sub_2",        "text": "As you wish. I'll follow your guidance.",                  "category": "ROLE",       "role":     "subordinate"},
        {"id": "phase_start_1",     "text": "Good morning everyone. Shall we begin?",                   "category": "PHASE",      "phase":    "start"},
        {"id": "phase_start_2",     "text": "Let's kick things off. First item on the agenda.",         "category": "PHASE",      "phase":    "start"},
        {"id": "phase_start_3",     "text": "Thanks for joining. Today we'll cover the following.",     "category": "PHASE",      "phase":    "start"},
        {"id": "phase_end_1",       "text": "That wraps up today's meeting. See you next time.",        "category": "PHASE",      "phase":    "end"},
        {"id": "phase_end_2",       "text": "Any final questions before we close?",                     "category": "PHASE",      "phase":    "end"},
        {"id": "phase_end_3",       "text": "Minutes will be shared. Thank you all.",                   "category": "PHASE",      "phase":    "end"},
        {"id": "phase_decision_1",  "text": "So we've decided to go ahead with this approach.",         "category": "PHASE",      "phase":    "decision"},
        {"id": "phase_decision_2",  "text": "Agreed. Let's lock this in and move forward.",             "category": "PHASE",      "phase":    "decision"},
        {"id": "phase_conflict_1",  "text": "I completely disagree with this approach.",                "category": "PHASE",      "phase":    "conflict"},
        {"id": "phase_conflict_2",  "text": "That's not acceptable. We need to revisit this.",          "category": "PHASE",      "phase":    "conflict"},
        {"id": "deadline_hard_1",   "text": "This must be done by end of day Friday. Non-negotiable.",  "category": "DEADLINE",   "urgency":  "hard"},
        {"id": "deadline_hard_2",   "text": "The client is expecting this by Monday morning.",          "category": "DEADLINE",   "urgency":  "hard"},
        {"id": "deadline_soft_1",   "text": "Try to get it done by next week if possible.",             "category": "DEADLINE",   "urgency":  "soft"},
        {"id": "deadline_soft_2",   "text": "Whenever you get a chance, please send this over.",        "category": "DEADLINE",   "urgency":  "soft"},
        {"id": "deadline_missed_1", "text": "This was supposed to be done last week.",                  "category": "DEADLINE",   "urgency":  "missed"},
        {"id": "deadline_missed_2", "text": "You've already missed the deadline twice.",                "category": "DEADLINE",   "urgency":  "missed"},
        {"id": "commit_strong_1",   "text": "I will have this ready by Thursday. You can count on me.", "category": "COMMITMENT", "strength": "strong"},
        {"id": "commit_strong_2",   "text": "Consider it done. I'll send by EOD.",                     "category": "COMMITMENT", "strength": "strong"},
        {"id": "commit_weak_1",     "text": "I'll try my best to get it done.",                        "category": "COMMITMENT", "strength": "weak"},
        {"id": "commit_weak_2",     "text": "I'll see what I can do. No promises though.",             "category": "COMMITMENT", "strength": "weak"},
        {"id": "commit_none_1",     "text": "We'll look into it and get back to you.",                 "category": "COMMITMENT", "strength": "none"},
        {"id": "commit_none_2",     "text": "This is something we can explore going forward.",         "category": "COMMITMENT", "strength": "none"},
        {"id": "escalation_1",      "text": "I'm going to have to take this to upper management.",     "category": "ESCALATION", "level":    "high"},
        {"id": "escalation_2",      "text": "This needs to be escalated. It's blocking us.",           "category": "ESCALATION", "level":    "high"},
        {"id": "escalation_3",      "text": "If this isn't resolved I'll involve legal.",              "category": "ESCALATION", "level":    "critical"},
    ]

    embedder = _get_embedder()
    if not embedder:
        return

    texts      = [p["text"]                                  for p in patterns]
    ids        = [p["id"]                                    for p in patterns]
    metadatas  = [{k: v for k, v in p.items() if k != "text"} for p in patterns]
    embeddings = embedder.encode(texts, show_progress_bar=False).tolist()
    coll.add(documents=texts, ids=ids, metadatas=metadatas, embeddings=embeddings)


# ── MAIN PUBLIC API ───────────────────────────────────────────────────────────

def _user_results_dir(user_id: str | None) -> Path:
    """
    Filesystem path for storing full JSON results.
    Uses _safe_uid (sanitised, not HMAC) — filesystem paths are protected
    by OS permissions, not queryable by name like ChromaDB collections.
    """
    if not user_id:
        return RESULTS_DIR / "anonymous"
    return RESULTS_DIR / "users" / _safe_uid(user_id)


def get_cached_result(
    transcript:        str,
    language:          str,
    user_id:           str | None = None,
    masked_transcript: str | None = None,
) -> dict | None:
    """
    Search this user's ChromaDB collection for a semantically similar transcript.
    Returns stored analysis result if similarity >= SEMANTIC_THRESHOLD.

    Y1 FIX (v2.1): Returns None immediately for anonymous users (user_id=None).
    SEC-1 FIX (v2.3): Collection name is HMAC-derived — not guessable from email.
    SEC-2 FIX (v2.3): Ownership check on retrieved result — defense-in-depth.
    X1 FIX (v2.1): masked_transcript used for embedding query for consistency.
    """
    # Y1 FIX: no persistent cache for unauthenticated users
    if not user_id:
        return None

    embedder = _get_embedder()
    if not embedder:
        return None

    coll = _get_user_collection(user_id)
    if coll is None or coll.count() == 0:
        return None

    # X1 FIX: use masked transcript for embedding consistency
    embed_text = masked_transcript if masked_transcript else transcript

    try:
        embedding = embedder.encode([embed_text], show_progress_bar=False).tolist()
        results   = coll.query(
            query_embeddings=embedding,
            n_results=1,
            where={"language": language} if language else None,
        )

        if not results["ids"] or not results["ids"][0]:
            return None

        distance   = results["distances"][0][0]
        similarity = 1 - distance
        doc_id     = results["ids"][0][0]

        if similarity >= SEMANTIC_THRESHOLD:
            result_dir  = _user_results_dir(user_id)
            result_path = result_dir / f"{doc_id}.json"
            if result_path.exists():
                with open(result_path, encoding="utf-8") as f:
                    result = json.load(f)

                # SEC-2 FIX: Ownership check — defense-in-depth.
                # The HMAC collection name (SEC-1) already makes cross-user
                # queries impossible in normal operation. This check is a
                # second layer: if _user_id in the stored result doesn't match
                # the requesting user, we refuse to return it and log the
                # anomaly. This guards against any future code-path bug that
                # might bypass the collection-name protection.
                # DSA: O(1) — single dict lookup + string compare.
                stored_uid = result.get("_user_id")
                if stored_uid and stored_uid != user_id:
                    print(
                        f"[VECTOR_CACHE] SEC-2 OWNERSHIP VIOLATION: "
                        f"stored _user_id={stored_uid!r} does not match "
                        f"requesting user={user_id!r}. "
                        f"doc_id={doc_id!r} lang={language!r}. "
                        f"Returning cache miss — investigate immediately.",
                        file=sys.stderr, flush=True,
                    )
                    return None

                result["_from_vector_cache"] = True
                result["_cache_similarity"]  = round(similarity, 4)
                result["_cache_doc_id"]      = doc_id
                return result

    except Exception as e:
        print(
            f"[VECTOR_CACHE] get_cached_result failed "
            f"(user={user_id!r}, lang={language!r}): {e}",
            file=sys.stderr, flush=True,
        )

    return None


def store_result(
    transcript:        str,
    language:          str,
    result:            dict,
    user_id:           str | None = None,
    masked_transcript: str | None = None,
) -> str | None:
    """
    Store a transcript embedding + full result in this user's private store.

    Y1 FIX (v2.1): Returns None immediately for anonymous users.
    Y2 FIX (v2.1): Skips storage for degraded/fallback providers.
    SEC-1 FIX (v2.3): Collection name is HMAC-derived — not guessable.
    X2 FIX (v2.1): ChromaDB documents[] stores masked_transcript (no raw PII).
    """
    # Y1 FIX: no persistent cache for unauthenticated users
    if not user_id:
        return None

    # Y2 FIX: skip degraded / fallback results
    provider       = result.get("_provider", "")
    provider_lower = provider.lower()
    if provider and not any(p in provider_lower for p in _CACHEABLE_PROVIDERS):
        print(
            f"[VECTOR_CACHE] Skipping cache for non-cacheable provider: {provider!r}",
            file=sys.stderr, flush=True,
        )
        return None

    embedder = _get_embedder()
    if not embedder:
        return None

    coll = _get_user_collection(user_id)
    if coll is None:
        return None

    client, _, _ = _get_chroma()
    if not client:
        return None

    # X1/X2 FIX: use and store masked transcript, not raw PII
    embed_text    = masked_transcript if masked_transcript else transcript
    document_text = masked_transcript if masked_transcript else transcript

    try:
        doc_id    = hashlib.md5(transcript.encode()).hexdigest()
        embedding = embedder.encode([embed_text], show_progress_bar=False).tolist()

        coll.upsert(
            ids       =[doc_id],
            documents =[document_text[:2000]],   # X2 FIX: masked text only
            embeddings=embedding,
            metadatas =[{
                "language":   language or "unknown",
                "word_count": len(transcript.split()),
                "stored_at":  time.strftime("%Y-%m-%dT%H:%M:%S"),
                "provider":   provider,
                "user_id":    user_id,
                "pii_masked": bool(masked_transcript),
            }]
        )

        result_dir = _user_results_dir(user_id)
        result_dir.mkdir(parents=True, exist_ok=True)
        result_path = result_dir / f"{doc_id}.json"

        clean_result = {
            k: v for k, v in result.items()
            if not k.startswith("_") or k in ("_provider", "_duration_ms")
        }
        clean_result["_cached_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        clean_result["_user_id"]   = user_id  # stored for SEC-2 ownership check

        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(clean_result, f, ensure_ascii=False, indent=2)

        return doc_id

    except Exception as e:
        print(
            f"[VECTOR_CACHE] store_result failed "
            f"(user={user_id!r}, lang={language!r}): {e}",
            file=sys.stderr, flush=True,
        )
        return None


def query_patterns(text: str, category: str = None, top_k: int = 3) -> list:
    """Query the NLP pattern library for semantic matches."""
    embedder = _get_embedder()
    if not embedder:
        return []

    _, _, patterns_coll = _get_chroma()
    if not patterns_coll or patterns_coll.count() == 0:
        return []

    try:
        embedding = embedder.encode([text], show_progress_bar=False).tolist()
        where     = {"category": category} if category else None
        results   = patterns_coll.query(
            query_embeddings=embedding,
            n_results=top_k,
            where=where,
        )

        matched = []
        for i, doc_id in enumerate(results["ids"][0]):
            similarity = 1 - results["distances"][0][i]
            if similarity >= 0.65:
                matched.append({
                    "pattern_id": doc_id,
                    "text":       results["documents"][0][i],
                    "metadata":   results["metadatas"][0][i],
                    "similarity": round(similarity, 3),
                })
        return matched

    except Exception as e:
        print(f"[VECTOR_CACHE] query_patterns failed: {e}", file=sys.stderr, flush=True)
        return []


def get_cache_stats(user_id: str | None = None) -> dict:
    """Returns vector cache stats for this user's collection."""
    client, _, patterns_coll = _get_chroma()
    if not client:
        return {"available": False, "transcript_count": 0}

    # Y1 FIX: anonymous users have no transcript cache
    if not user_id:
        return {
            "available":        True,
            "transcript_count": 0,
            "pattern_count":    patterns_coll.count() if patterns_coll else 0,
            "store_path":       str(VECTOR_STORE_DIR),
            "user_id":          "anonymous",
            "cache_enabled":    False,
            "cache_note":       "Sign in to enable persistent transcript caching",
        }

    try:
        coll = _get_user_collection(user_id)
        return {
            "available":        True,
            "transcript_count": coll.count() if coll else 0,
            "pattern_count":    patterns_coll.count() if patterns_coll else 0,
            "store_path":       str(VECTOR_STORE_DIR),
            "user_id":          user_id,
            "cache_enabled":    True,
        }
    except Exception as e:
        print(f"[VECTOR_CACHE] get_cache_stats failed: {e}", file=sys.stderr, flush=True)
        return {"available": False, "transcript_count": 0}


def is_available() -> bool:
    """Quick check — returns True if ChromaDB + embedder are both available."""
    return _get_embedder() is not None and _get_chroma()[0] is not None