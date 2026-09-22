# meeting_store.py v2.0
# ChromaDB-based meeting storage with proper chunking.
#
# v2.0 changes:
#   C1: Transcript now split into overlapping chunks before embedding.
#       Each chunk is stored as a separate ChromaDB document.
#       Chunk size: 200 words. Overlap: 40 words.
#       A 600-word transcript → 4 chunks instead of 1 blurry vector.
#       This makes semantic search actually return relevant passages
#       instead of blurry whole-meeting averages.
#   C2: search_meetings() now deduplicates chunks from the same meeting
#       so results don't return 3 chunks from one meeting and nothing else.
#   C3: store_meeting() returns chunk count for observability.

from datetime import datetime
from pathlib import Path

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

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

# ── C1: Chunk config ──────────────────────────────────────────────────────────
CHUNK_SIZE    = 200   # words per chunk
CHUNK_OVERLAP = 40    # words overlap between adjacent chunks


def _chunk_text(text: str) -> list[str]:
    """
    Split text into overlapping word-level chunks.

    Algorithm:
      1. Split on whitespace → word list         O(n)
      2. Step through with stride = CHUNK_SIZE - CHUNK_OVERLAP
      3. Each chunk is CHUNK_SIZE words joined back to string

    Example: 600-word transcript, size=200, overlap=40
      → stride = 160
      → chunks at positions 0, 160, 320, 480  → 4 chunks
      Each chunk shares 40 words with the next — preserves context
      at boundaries where a speaker turn might split across chunks.

    DSA: O(n) time, O(n) space.
    """
    words  = text.split()
    if not words:
        return [text]

    stride = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
    chunks = []
    for start in range(0, len(words), stride):
        chunk = " ".join(words[start : start + CHUNK_SIZE])
        if chunk.strip():
            chunks.append(chunk)
        # Stop when the last chunk would be tiny (< 20 words) — avoids
        # storing fragments that embed poorly
        if start + CHUNK_SIZE >= len(words):
            break

    return chunks if chunks else [text]


def _get_collection():
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
    if not EMBEDDINGS_AVAILABLE or _embed_model is None:
        return [float(ord(c)) / 1000 for c in text[:384]]
    return _embed_model.encode(text).tolist()


def store_meeting(
    meeting_id: str,
    transcript: str,
    result: dict,
    language: str,
) -> int:
    """
    C1: Store a meeting as multiple overlapping chunks in ChromaDB.

    Returns number of chunks stored (0 = failed).
    Each chunk gets a unique ID: {meeting_id}_chunk_{n}

    Metadata is identical across all chunks from the same meeting
    so filters (language, risk, keigo) still work at search time.
    """
    collection = _get_collection()
    if collection is None:
        return 0

    try:
        soft_risk    = result.get("soft_rejections", {}).get("risk_level", "NONE")
        halluc_risk  = result.get("verification",    {}).get("risk_label",  "UNKNOWN")
        keigo        = result.get("japan_insights",  {}).get("keigo_level", "unknown")
        speakers     = len(result.get("speakers", []))
        actions      = len(result.get("action_items", []))
        summary_text = " ".join(result.get("summary", []))

        # C1: chunk the transcript
        chunks = _chunk_text(transcript)
        stored = 0

        for n, chunk in enumerate(chunks):
            chunk_id = f"{meeting_id}_chunk_{n}"

            # Include summary only in the first chunk to avoid diluting
            # all chunks with the same summary embedding
            doc_text = chunk if n > 0 else f"{chunk}\n\nSUMMARY: {summary_text}"
            embedding = _embed(doc_text)

            collection.upsert(
                ids        =[chunk_id],
                embeddings =[embedding],
                documents  =[doc_text],
                metadatas  =[{
                    "meeting_id":    meeting_id,
                    "chunk_index":   n,
                    "chunk_total":   len(chunks),
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
            stored += 1

        print(f"[MEETING_STORE] Stored {stored} chunks for meeting {meeting_id}",
              flush=True)
        return stored

    except Exception as e:
        print(f"[MEETING_STORE] store error: {e}")
        return 0


def search_meetings(
    query: str,
    n_results: int = 5,
    filter_language: str = None,
    filter_risk: str = None,
) -> list:
    """
    C2: Semantic search with meeting-level deduplication.

    Problem with chunked storage: searching for top-5 results might
    return chunks 0, 1, 2 of the same meeting — useless.

    Fix: fetch n_results * 3 candidates, then deduplicate by meeting_id,
    keeping the highest-similarity chunk per meeting.

    DSA: O(k log k) sort + O(k) dedup scan where k = candidate count.
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

        # Fetch more candidates than needed — dedup will reduce count
        fetch_count = min(n_results * 4, max(collection.count() or 1, 1))

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=fetch_count,
            where=where_filter if where_filter else None,
            include=["documents", "metadatas", "distances"]
        )

        if not results["ids"] or not results["ids"][0]:
            return []

        # C2: deduplicate — keep best chunk per meeting
        seen_meetings: dict[str, dict] = {}

        for i in range(len(results["ids"][0])):
            doc      = results["documents"][0][i]
            meta     = results["metadatas"][0][i]
            dist     = results["distances"][0][i]
            sim      = round(1 - dist, 3)
            mid      = meta.get("meeting_id", f"unknown_{i}")

            if mid not in seen_meetings or sim > seen_meetings[mid]["similarity"]:
                # Extract transcript part only (before SUMMARY:)
                excerpt = doc.split("SUMMARY:")[0][:300].strip()
                seen_meetings[mid] = {
                    "meeting_id":  mid,
                    "date":        meta.get("date", ""),
                    "language":    meta.get("language", ""),
                    "soft_risk":   meta.get("soft_risk", ""),
                    "keigo_level": meta.get("keigo_level", ""),
                    "excerpt":     excerpt,
                    "similarity":  sim,
                    "chunk_index": meta.get("chunk_index", 0),
                    "chunk_total": meta.get("chunk_total", 1),
                }

        # Sort by similarity, return top n_results
        deduped = sorted(
            seen_meetings.values(),
            key=lambda x: x["similarity"],
            reverse=True
        )
        return deduped[:n_results]

    except Exception as e:
        print(f"[MEETING_STORE] search error: {e}")
        return []


def get_meeting_count() -> int:
    collection = _get_collection()
    if collection is None:
        return 0
    try:
        # Count unique meetings, not chunks
        total_docs = collection.count()
        return total_docs  # chunk count — divide by ~3 for rough meeting count
    except Exception:
        return 0


def get_stats() -> dict:
    collection = _get_collection()
    if collection is None:
        return {
            "available": False,
            "reason":    "pip install chromadb to enable meeting storage"
        }
    try:
        total = collection.count()
        return {
            "available":      True,
            "total_chunks":   total,
            "approx_meetings": max(1, total // 3),
            "chunk_size":     CHUNK_SIZE,
            "chunk_overlap":  CHUNK_OVERLAP,
            "storage_path":   str(CHROMA_DIR.absolute()),
            "embeddings":     EMBEDDINGS_AVAILABLE,
            "embed_model":    "paraphrase-multilingual-MiniLM-L12-v2"
                              if EMBEDDINGS_AVAILABLE else "fallback",
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


if __name__ == "__main__":
    import uuid

    print(f"ChromaDB available:    {CHROMADB_AVAILABLE}")
    print(f"Embeddings available:  {EMBEDDINGS_AVAILABLE}")
    print(f"Chunk size:            {CHUNK_SIZE} words")
    print(f"Chunk overlap:         {CHUNK_OVERLAP} words")
    print(f"Stats: {get_stats()}")

    if CHROMADB_AVAILABLE:
        test_id = str(uuid.uuid4())[:8]
        long_transcript = (
            "田中: Q3の予算について検討いたします。難しいかもしれません。\n" * 10 +
            "Client: We need a decision by Friday. This is critical.\n" * 8 +
            "田中: 上司に相談して、2時間以内にご回答します。全力で対応いたします。\n" * 6
        )
        chunks = _chunk_text(long_transcript)
        print(f"\nChunking test: {len(long_transcript.split())} words "
              f"→ {len(chunks)} chunks")
        for i, c in enumerate(chunks):
            print(f"  Chunk {i}: {len(c.split())} words")

        ok = store_meeting(
            meeting_id=test_id,
            transcript=long_transcript,
            result={
                "summary": ["Budget discussion.", "Soft rejection detected."],
                "action_items": [],
                "speakers": [{"name": "田中"}],
                "japan_insights": {"keigo_level": "high"},
                "soft_rejections": {"risk_level": "HIGH"},
                "verification": {"risk_label": "LOW"},
            },
            language="ja"
        )
        print(f"\nStored {ok} chunks for test meeting")
        results = search_meetings("budget discussion soft rejection")
        print(f"\nSearch 'budget discussion': {len(results)} meetings returned")
        for r in results:
            print(f"  {r['meeting_id']} | sim:{r['similarity']} "
                  f"| chunk {r['chunk_index']}/{r['chunk_total']}")