# meeting_store.py
# ChromaDB-based meeting storage and semantic search
#
# Why ChromaDB over FAISS:
#   FAISS  — fast in-memory similarity search, no metadata filtering,
#             no persistence without manual serialization
#   ChromaDB — persistent SQLite backend, built-in metadata filtering
#             (filter by language, date, risk level), simpler API
#             for this use case where we need "find meetings where
#             soft rejection risk was HIGH in the last 30 days"
#
# This enables:
#   1. Cross-meeting semantic search ("find all meetings where budget was discussed")
#   2. Trend retrieval ("meetings with HIGH soft rejection risk this month")
#   3. RAG context for follow-up Q&A on past meetings

from datetime import datetime
from pathlib import Path

# ChromaDB — graceful import
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

# Sentence transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    _embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    _embed_model = None

CHROMA_DIR = Path("chroma_db")
_client     = None
_collection = None


def _get_collection():
    """Lazy init ChromaDB client and collection."""
    global _client, _collection
    if _collection is not None:
        return _collection

    if not CHROMADB_AVAILABLE:
        return None

    CHROMA_DIR.mkdir(exist_ok=True)
    _client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False)
    )
    _collection = _client.get_or_create_collection(
        name="meetings",
        metadata={"hnsw:space": "cosine"}
    )
    return _collection


def _embed(text: str) -> list:
    """Generate embedding for a text using sentence-transformers."""
    if not EMBEDDINGS_AVAILABLE or _embed_model is None:
        # Fallback: simple character hash as pseudo-embedding
        # Not useful for similarity but keeps the interface consistent
        return [float(ord(c)) / 1000 for c in text[:384]]
    return _embed_model.encode(text).tolist()


def store_meeting(
    meeting_id: str,
    transcript: str,
    result: dict,
    language: str,
) -> bool:
    """
    Store a meeting analysis in ChromaDB for future semantic search.

    Each meeting stored with:
    - Embedding of the transcript (for similarity search)
    - Metadata: language, date, risk levels, speaker count
    - Document: full transcript text
    """
    collection = _get_collection()
    if collection is None:
        return False

    try:
        soft_risk  = result.get("soft_rejections", {}).get("risk_level", "NONE")
        halluc_risk= result.get("verification",    {}).get("risk_label",  "UNKNOWN")
        keigo      = result.get("japan_insights",  {}).get("keigo_level", "unknown")
        speakers   = len(result.get("speakers", []))
        actions    = len(result.get("action_items", []))
        summary_text = " ".join(result.get("summary", []))

        # Store transcript + summary as searchable document
        doc_text   = f"{transcript}\n\nSUMMARY: {summary_text}"
        embedding  = _embed(doc_text)

        collection.upsert(
            ids=[meeting_id],
            embeddings=[embedding],
            documents=[doc_text],
            metadatas=[{
                "meeting_id":    meeting_id,
                "date":          datetime.now().isoformat(),
                "language":      language,
                "soft_risk":     soft_risk,
                "halluc_risk":   halluc_risk,
                "keigo_level":   keigo,
                "speaker_count": speakers,
                "action_count":  actions,
                "char_length":   len(transcript),
            }]
        )
        return True

    except Exception as e:
        print(f"ChromaDB store error: {e}")
        return False


def search_meetings(
    query: str,
    n_results: int = 5,
    filter_language: str = None,
    filter_risk: str = None,
) -> list:
    """
    Semantic search across stored meetings.

    Args:
        query:           Natural language query e.g. "budget discussion"
        n_results:       Number of results to return
        filter_language: Only search ja/en/mixed/hi meetings
        filter_risk:     Only return meetings with this soft rejection risk

    Returns list of dicts with transcript excerpt, metadata, distance score.

    ChromaDB advantage over FAISS here: metadata filtering
    e.g. "find meetings with HIGH soft rejection risk in Japanese"
    FAISS would require post-filter; ChromaDB does it at query time.
    """
    collection = _get_collection()
    if collection is None:
        return []

    try:
        query_embedding = _embed(query)
        where_filter    = {}
        if filter_language:
            where_filter["language"] = {"$eq": filter_language}
        if filter_risk:
            where_filter["soft_risk"] = {"$eq": filter_risk}

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, collection.count() or 1),
            where=where_filter if where_filter else None,
            include=["documents", "metadatas", "distances"]
        )

        output = []
        for i in range(len(results["ids"][0])):
            doc   = results["documents"][0][i]
            meta  = results["metadatas"][0][i]
            dist  = results["distances"][0][i]
            # Extract just the transcript part (before SUMMARY:)
            transcript_excerpt = doc.split("SUMMARY:")[0][:300].strip()
            output.append({
                "meeting_id":   meta.get("meeting_id"),
                "date":         meta.get("date", ""),
                "language":     meta.get("language", ""),
                "soft_risk":    meta.get("soft_risk", ""),
                "keigo_level":  meta.get("keigo_level", ""),
                "excerpt":      transcript_excerpt,
                "similarity":   round(1 - dist, 3),
            })
        return output

    except Exception as e:
        print(f"ChromaDB search error: {e}")
        return []


def get_meeting_count() -> int:
    """Returns total meetings stored in ChromaDB."""
    collection = _get_collection()
    if collection is None:
        return 0
    try:
        return collection.count()
    except Exception:
        return 0


def get_stats() -> dict:
    """Returns ChromaDB storage statistics."""
    collection = _get_collection()
    if collection is None:
        return {
            "available": False,
            "reason": "pip install chromadb to enable meeting storage"
        }
    return {
        "available":      True,
        "total_meetings": collection.count(),
        "storage_path":   str(CHROMA_DIR.absolute()),
        "embeddings":     EMBEDDINGS_AVAILABLE,
        "embed_model":    "paraphrase-multilingual-MiniLM-L12-v2" if EMBEDDINGS_AVAILABLE else "fallback",
    }


if __name__ == "__main__":
    import json, uuid

    print(f"ChromaDB available: {CHROMADB_AVAILABLE}")
    print(f"Embeddings available: {EMBEDDINGS_AVAILABLE}")
    print(f"Stats: {get_stats()}")

    if CHROMADB_AVAILABLE:
        # Store a test meeting
        test_id = str(uuid.uuid4())[:8]
        ok = store_meeting(
            meeting_id=test_id,
            transcript="田中: Q3の予算について検討いたします。難しいかもしれません。",
            result={
                "summary": ["Budget discussion for Q3.", "Soft rejection detected."],
                "action_items": [],
                "speakers": [{"name": "田中"}],
                "japan_insights": {"keigo_level": "high"},
                "soft_rejections": {"risk_level": "HIGH"},
                "verification": {"risk_label": "LOW"},
            },
            language="ja"
        )
        print(f"\nStored test meeting: {ok}")
        print(f"Total meetings: {get_meeting_count()}")

        # Search
        results = search_meetings("budget discussion risk")
        print(f"\nSearch results for 'budget discussion risk': {len(results)}")
        for r in results:
            print(f"  {r['meeting_id']} | {r['language']} | risk:{r['soft_risk']} | sim:{r['similarity']}")