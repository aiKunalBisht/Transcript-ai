# vector_cache.py — v1
# Persistent vector cache using ChromaDB + sentence-transformers
#
# Architecture:
#   - Every analyzed transcript is stored as a vector embedding
#   - On new transcript: semantic similarity search first
#   - If similarity > 0.92: return stored result instantly (no Groq call)
#   - If not found: call Groq, store result for future
#
# Storage: ./vector_store/ (persists across restarts, reloads, everything)
# No login required — global shared collection
# Sample transcript cached on first run, instant forever after

import os
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
            _embedder = False   # sentence-transformers not installed
    return _embedder if _embedder else None


def _get_chroma():
    """Lazy-load ChromaDB client with persistent storage."""
    global _chroma_client, _transcript_coll, _patterns_coll
    if _chroma_client is not None:
        return _chroma_client, _transcript_coll, _patterns_coll
    try:
        import chromadb
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

        # Main transcript collection
        _transcript_coll = _chroma_client.get_or_create_collection(
            name="transcripts",
            metadata={"hnsw:space": "cosine"}
        )

        # NLP pattern reference library
        _patterns_coll = _chroma_client.get_or_create_collection(
            name="nlp_patterns",
            metadata={"hnsw:space": "cosine"}
        )

        # Seed NLP patterns if empty
        if _patterns_coll.count() == 0:
            _seed_nlp_patterns(_patterns_coll)

    except ImportError:
        _chroma_client = False
    return _chroma_client or None, _transcript_coll, _patterns_coll


def _seed_nlp_patterns(coll):
    """
    Store reference NLP patterns in ChromaDB.
    These are used for semantic matching against transcript segments
    to identify roles, meeting phases, and structural events.
    """
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
    metadatas  = [{k:v for k,v in p.items() if k != "text"} for p in patterns]
    embeddings = embedder.encode(texts, show_progress_bar=False).tolist()

    coll.add(documents=texts, ids=ids, metadatas=metadatas, embeddings=embeddings)


# ── MAIN PUBLIC API ───────────────────────────────────────────────────────────

def get_cached_result(transcript: str, language: str) -> dict | None:
    """
    Search ChromaDB for a semantically similar transcript.
    Returns stored analysis result if similarity >= threshold.
    Returns None if no match found (caller should run Groq).
    """
    embedder = _get_embedder()
    if not embedder:
        return None

    client, coll, _ = _get_chroma()
    if not client or coll.count() == 0:
        return None

    try:
        embedding = embedder.encode([transcript], show_progress_bar=False).tolist()
        results   = coll.query(
            query_embeddings=embedding,
            n_results=1,
            where={"language": language} if language else None,
        )

        if not results["ids"] or not results["ids"][0]:
            return None

        distance   = results["distances"][0][0]
        similarity = 1 - distance   # cosine distance → similarity
        doc_id     = results["ids"][0][0]

        if similarity >= SEMANTIC_THRESHOLD:
            # Load full result from results store
            result_path = RESULTS_DIR / f"{doc_id}.json"
            if result_path.exists():
                with open(result_path) as f:
                    result = json.load(f)
                result["_from_vector_cache"] = True
                result["_cache_similarity"]  = round(similarity, 4)
                result["_cache_doc_id"]      = doc_id
                return result

    except Exception:
        pass

    return None


def store_result(transcript: str, language: str, result: dict) -> str | None:
    """
    Store a transcript and its analysis result in ChromaDB.
    Returns the document ID on success, None on failure.
    """
    embedder = _get_embedder()
    if not embedder:
        return None

    client, coll, _ = _get_chroma()
    if not client:
        return None

    try:
        # Use MD5 as stable document ID
        doc_id    = hashlib.md5(transcript.encode()).hexdigest()
        embedding = embedder.encode([transcript], show_progress_bar=False).tolist()

        word_count = len(transcript.split())
        lang_label = language or "unknown"

        # Store embedding + metadata in ChromaDB
        # Upsert so re-analyzing same transcript updates the result
        coll.upsert(
            ids        =[doc_id],
            documents  =[transcript[:2000]],  # ChromaDB doc limit
            embeddings =embedding,
            metadatas  =[{
                "language":   lang_label,
                "word_count": word_count,
                "stored_at":  time.strftime("%Y-%m-%dT%H:%M:%S"),
                "provider":   result.get("_provider","unknown"),
            }]
        )

        # Store full result JSON separately (no size limit)
        result_path = RESULTS_DIR / f"{doc_id}.json"
        clean_result = {k: v for k, v in result.items()
                        if not k.startswith("_") or k in ("_provider","_duration_ms")}
        clean_result["_cached_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

        with open(result_path, "w") as f:
            json.dump(clean_result, f, ensure_ascii=False, indent=2)

        return doc_id

    except Exception:
        return None


def query_patterns(text: str, category: str = None, top_k: int = 3) -> list:
    """
    Query the NLP pattern library for semantic matches.
    Used to identify meeting roles, phases, deadlines, commitments.

    Returns list of matched patterns with similarity scores.
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
            if similarity >= 0.65:   # only return meaningful matches
                matched.append({
                    "pattern_id":  doc_id,
                    "text":        results["documents"][0][i],
                    "metadata":    results["metadatas"][0][i],
                    "similarity":  round(similarity, 3),
                })
        return matched

    except Exception:
        return []


def get_cache_stats() -> dict:
    """Returns stats about the vector cache — used in Trends tab."""
    client, coll, patterns_coll = _get_chroma()
    if not client:
        return {"available": False, "transcript_count": 0}

    try:
        return {
            "available":        True,
            "transcript_count": coll.count() if coll else 0,
            "pattern_count":    patterns_coll.count() if patterns_coll else 0,
            "store_path":       str(VECTOR_STORE_DIR),
        }
    except Exception:
        return {"available": False, "transcript_count": 0}


def is_available() -> bool:
    """Quick check — returns True if ChromaDB + embedder both available."""
    return _get_embedder() is not None and _get_chroma()[0] is not None