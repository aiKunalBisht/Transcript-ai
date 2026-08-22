# vector_cache.py — v2.1
# Persistent vector cache using ChromaDB + sentence-transformers
#
# v2.0 → v2.1 changes:
#
# X1 FIX: get_cached_result() now accepts masked_transcript parameter.
#         Embedding is computed on masked_transcript when provided, falling
#         back to raw transcript if masking was unavailable. This ensures
#         cache lookups are consistent with how documents were stored (X2).
#
# X2 FIX: store_result() now accepts masked_transcript parameter.
#         ChromaDB documents[] field now stores masked_transcript[:2000]
#         instead of raw transcript[:2000]. Raw PII (names, emails, phones)
#         no longer written to ChromaDB plaintext storage on disk.
#         doc_id still computed from raw transcript MD5 for backward
#         compatibility — existing cache entries continue to resolve.
#         Embedding also computed on masked_transcript for consistency with X1.
#
# X3 FIX: Bare `except Exception: pass` replaced with stderr logging in
#         both get_cached_result() and store_result(). Silent failures were
#         hiding ChromaDB corruption, disk-full errors, and collection
#         schema mismatches. Now all cache errors are logged with context.
#
# Retained from v2.0:
#   - Per-user ChromaDB collections: transcripts_{safe_user_id}
#   - Anonymous requests use "transcripts_anonymous" (shared, as before)
#   - _get_user_collection(user_id) replaces the global _transcript_coll
#   - Global singleton _chroma_client still shared (ChromaDB supports this)
#   - NLP patterns collection remains global (not user-specific)
#
# Storage: ./vector_store/chroma_db/ (persists across restarts)
# Each user's embeddings are isolated — no cross-user semantic leakage.

import os
import sys
import json
import hashlib
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
SEMANTIC_THRESHOLD = 0.92   # same meeting, slightly different wording → return
# Below 0.92 = different meeting → call Groq

# ── LAZY SINGLETONS ───────────────────────────────────────────────────────────
_chroma_client     = None
_transcript_coll   = None
_patterns_coll     = None
_embedder          = None


def _get_embedder():
    """Lazy-load sentence-transformers — only when first needed."""
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
            # all-MiniLM-L6-v2: 80MB, 384-dim, ~30ms per embedding
            _embedder = SentenceTransformer("all-MiniLM-L6-v2")
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


def _safe_uid(user_id: str) -> str:
    """Sanitise user_id for ChromaDB collection name."""
    clean = re.sub(r"[^a-zA-Z0-9_-]", "_", str(user_id))[:40]
    return clean if clean else "anonymous"


def _get_user_collection(user_id: str | None = None):
    """
    Returns the ChromaDB collection for this specific user.
    user_id=None  → shared anonymous collection (v1 behaviour)
    user_id="xyz" → isolated collection "transcripts_{safe_id}"
    """
    client, anon_coll, _ = _get_chroma()
    if client is None:
        return None

    if not user_id:
        return anon_coll

    coll_name = f"transcripts_{_safe_uid(user_id)}"
    try:
        return client.get_or_create_collection(
            name=coll_name,
            metadata={"hnsw:space": "cosine"}
        )
    except Exception as e:
        print(
            f"[VECTOR_CACHE] _get_user_collection failed for user={user_id!r}: {e}",
            file=sys.stderr, flush=True
        )
        return anon_coll


def _seed_nlp_patterns(coll):
    """Store reference NLP patterns in ChromaDB for semantic matching."""
    patterns = [
        # ── MEETING ROLES ──────────────────────────────────────────────────────
        {"id": "role_leader_1",    "text": "Let's get started. I'll chair today's meeting.",          "category": "ROLE", "role": "leader"},
        {"id": "role_leader_2",    "text": "Okay everyone, let me summarize what we agreed.",         "category": "ROLE", "role": "leader"},
        {"id": "role_leader_3",    "text": "I'll take ownership of this. Everyone please note.",      "category": "ROLE", "role": "leader"},
        {"id": "role_manager_1",   "text": "I need this done by end of week. No exceptions.",         "category": "ROLE", "role": "manager"},
        {"id": "role_manager_2",   "text": "As per my earlier direction, this should be prioritized.", "category": "ROLE", "role": "manager"},
        {"id": "role_teamlead_1",  "text": "My team will handle this. I'll assign it internally.",    "category": "ROLE", "role": "team_lead"},
        {"id": "role_teamlead_2",  "text": "Let me check with my team and get back to you.",          "category": "ROLE", "role": "team_lead"},
        {"id": "role_subordinate_1","text": "Yes sir, I will make sure it gets done.",                "category": "ROLE", "role": "subordinate"},
        {"id": "role_subordinate_2","text": "As you wish. I'll follow your guidance.",                "category": "ROLE", "role": "subordinate"},
        # ── MEETING PHASES ─────────────────────────────────────────────────────
        {"id": "phase_start_1",    "text": "Good morning everyone. Shall we begin?",                  "category": "PHASE", "phase": "start"},
        {"id": "phase_start_2",    "text": "Let's kick things off. First item on the agenda.",        "category": "PHASE", "phase": "start"},
        {"id": "phase_start_3",    "text": "Thanks for joining. Today we'll cover the following.",    "category": "PHASE", "phase": "start"},
        {"id": "phase_end_1",      "text": "That wraps up today's meeting. See you next time.",       "category": "PHASE", "phase": "end"},
        {"id": "phase_end_2",      "text": "Any final questions before we close?",                    "category": "PHASE", "phase": "end"},
        {"id": "phase_end_3",      "text": "Minutes will be shared. Thank you all.",                  "category": "PHASE", "phase": "end"},
        {"id": "phase_decision_1", "text": "So we've decided to go ahead with this approach.",        "category": "PHASE", "phase": "decision"},
        {"id": "phase_decision_2", "text": "Agreed. Let's lock this in and move forward.",            "category": "PHASE", "phase": "decision"},
        {"id": "phase_conflict_1", "text": "I completely disagree with this approach.",               "category": "PHASE", "phase": "conflict"},
        {"id": "phase_conflict_2", "text": "That's not acceptable. We need to revisit this.",         "category": "PHASE", "phase": "conflict"},
        # ── DEADLINES ──────────────────────────────────────────────────────────
        {"id": "deadline_hard_1",  "text": "This must be done by end of day Friday. Non-negotiable.", "category": "DEADLINE", "urgency": "hard"},
        {"id": "deadline_hard_2",  "text": "The client is expecting this by Monday morning.",         "category": "DEADLINE", "urgency": "hard"},
        {"id": "deadline_soft_1",  "text": "Try to get it done by next week if possible.",            "category": "DEADLINE", "urgency": "soft"},
        {"id": "deadline_soft_2",  "text": "Whenever you get a chance, please send this over.",       "category": "DEADLINE", "urgency": "soft"},
        {"id": "deadline_missed_1","text": "This was supposed to be done last week.",                 "category": "DEADLINE", "urgency": "missed"},
        {"id": "deadline_missed_2","text": "You've already missed the deadline twice.",               "category": "DEADLINE", "urgency": "missed"},
        # ── COMMITMENTS ───────────────────────────────────────────────────────
        {"id": "commit_strong_1",  "text": "I will have this ready by Thursday. You can count on me.","category": "COMMITMENT", "strength": "strong"},
        {"id": "commit_strong_2",  "text": "Consider it done. I'll send by EOD.",                    "category": "COMMITMENT", "strength": "strong"},
        {"id": "commit_weak_1",    "text": "I'll try my best to get it done.",                       "category": "COMMITMENT", "strength": "weak"},
        {"id": "commit_weak_2",    "text": "I'll see what I can do. No promises though.",            "category": "COMMITMENT", "strength": "weak"},
        {"id": "commit_none_1",    "text": "We'll look into it and get back to you.",                "category": "COMMITMENT", "strength": "none"},
        {"id": "commit_none_2",    "text": "This is something we can explore going forward.",        "category": "COMMITMENT", "strength": "none"},
        # ── ESCALATION ────────────────────────────────────────────────────────
        {"id": "escalation_1",     "text": "I'm going to have to take this to upper management.",    "category": "ESCALATION", "level": "high"},
        {"id": "escalation_2",     "text": "This needs to be escalated. It's blocking us.",          "category": "ESCALATION", "level": "high"},
        {"id": "escalation_3",     "text": "If this isn't resolved I'll involve legal.",             "category": "ESCALATION", "level": "critical"},
    ]

    embedder = _get_embedder()
    if not embedder:
        return

    texts      = [p["text"]     for p in patterns]
    ids        = [p["id"]       for p in patterns]
    metadatas  = [{k: v for k, v in p.items() if k != "text"} for p in patterns]
    embeddings = embedder.encode(texts, show_progress_bar=False).tolist()
    coll.add(documents=texts, ids=ids, metadatas=metadatas, embeddings=embeddings)


# ── MAIN PUBLIC API ───────────────────────────────────────────────────────────

def _user_results_dir(user_id: str | None) -> Path:
    """
    Returns the directory where full result JSONs are stored for this user.
    vector_store/results/anonymous/{doc_id}.json   ← shared / no auth
    vector_store/results/users/{safe_uid}/{doc_id}.json  ← per-user
    """
    if not user_id:
        return RESULTS_DIR / "anonymous"
    return RESULTS_DIR / "users" / _safe_uid(user_id)


def get_cached_result(
    transcript:        str,
    language:          str,
    user_id:           str | None = None,
    masked_transcript: str | None = None,   # X1 FIX: use for embedding lookup
) -> dict | None:
    """
    Search this user's ChromaDB collection for a semantically similar transcript.
    Returns stored analysis result if similarity >= SEMANTIC_THRESHOLD.

    X1 FIX: masked_transcript is used for the embedding query when provided.
    This ensures the lookup embedding is computed on the same text representation
    that was used when the document was originally stored (store_result X2 FIX).
    Falls back to raw transcript if masked_transcript is not available,
    maintaining backward compatibility with entries stored before v2.1.

    DSA: HNSW approximate nearest-neighbour O(log n).
    """
    embedder = _get_embedder()
    if not embedder:
        return None

    coll = _get_user_collection(user_id)
    if coll is None or coll.count() == 0:
        return None

    # X1 FIX: prefer masked text for embedding — consistent with store_result
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
                with open(result_path, "r", encoding="utf-8") as f:
                    result = json.load(f)
                result["_from_vector_cache"] = True
                result["_cache_similarity"]  = round(similarity, 4)
                result["_cache_doc_id"]      = doc_id
                return result

    # X3 FIX: log cache read failures — don't swallow silently
    except Exception as e:
        print(
            f"[VECTOR_CACHE] get_cached_result failed "
            f"(user={user_id!r}, lang={language!r}): {e}",
            file=sys.stderr, flush=True
        )

    return None


def store_result(
    transcript:        str,
    language:          str,
    result:            dict,
    user_id:           str | None = None,
    masked_transcript: str | None = None,   # X2 FIX: store this, not raw
) -> str | None:
    """
    Store a transcript embedding + full result in this user's private store.

    X2 FIX: ChromaDB documents[] field now stores masked_transcript[:2000]
    instead of raw transcript[:2000]. This ensures names, emails, and phone
    numbers are NOT written as plaintext to the ChromaDB SQLite files on disk.

    doc_id is still derived from the raw transcript MD5 for backward
    compatibility — existing cache entries resolve with the same key.

    Embedding is computed on masked_transcript (when available) to match
    the query embedding computed in get_cached_result (X1 FIX). Consistent
    embedding space = correct similarity scores.

    If masked_transcript is not provided (masking unavailable), falls back
    to raw transcript for both embedding and document storage — no regression.
    """
    embedder = _get_embedder()
    if not embedder:
        return None

    coll = _get_user_collection(user_id)
    if coll is None:
        return None

    client, _, _ = _get_chroma()
    if not client:
        return None

    # X2 FIX: prefer masked text for embedding + document storage
    embed_text    = masked_transcript if masked_transcript else transcript
    document_text = masked_transcript if masked_transcript else transcript

    try:
        # doc_id on RAW text MD5 — backward compatible with pre-v2.1 entries
        doc_id    = hashlib.md5(transcript.encode()).hexdigest()
        # X2 FIX: embedding on masked text — consistent with get_cached_result
        embedding = embedder.encode([embed_text], show_progress_bar=False).tolist()

        word_count = len(transcript.split())
        lang_label = language or "unknown"

        # X2 FIX: document_text is masked — no raw PII written to ChromaDB disk
        coll.upsert(
            ids        =[doc_id],
            documents  =[document_text[:2000]],   # ← masked, not raw
            embeddings =embedding,
            metadatas  =[{
                "language":   lang_label,
                "word_count": word_count,
                "stored_at":  time.strftime("%Y-%m-%dT%H:%M:%S"),
                "provider":   result.get("_provider", "unknown"),
                "user_id":    user_id or "anonymous",
                "pii_masked": bool(masked_transcript),   # auditable flag
            }]
        )

        # Full result JSON stored in user's results directory (no size limit)
        result_dir = _user_results_dir(user_id)
        result_dir.mkdir(parents=True, exist_ok=True)
        result_path = result_dir / f"{doc_id}.json"

        clean_result = {
            k: v for k, v in result.items()
            if not k.startswith("_") or k in ("_provider", "_duration_ms")
        }
        clean_result["_cached_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        clean_result["_user_id"]   = user_id or "anonymous"

        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(clean_result, f, ensure_ascii=False, indent=2)

        return doc_id

    # X3 FIX: log store failures — don't swallow silently
    except Exception as e:
        print(
            f"[VECTOR_CACHE] store_result failed "
            f"(user={user_id!r}, lang={language!r}): {e}",
            file=sys.stderr, flush=True
        )
        return None


def query_patterns(text: str, category: str = None, top_k: int = 3) -> list:
    """
    Query the NLP pattern library for semantic matches.
    Used to identify meeting roles, phases, deadlines, commitments.
    """
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

    try:
        coll = _get_user_collection(user_id)
        return {
            "available":        True,
            "transcript_count": coll.count() if coll else 0,
            "pattern_count":    patterns_coll.count() if patterns_coll else 0,
            "store_path":       str(VECTOR_STORE_DIR),
            "user_id":          user_id or "anonymous",
        }
    except Exception as e:
        print(f"[VECTOR_CACHE] get_cache_stats failed: {e}", file=sys.stderr, flush=True)
        return {"available": False, "transcript_count": 0}


def is_available() -> bool:
    """Quick check — returns True if ChromaDB + embedder both available."""
    return _get_embedder() is not None and _get_chroma()[0] is not None