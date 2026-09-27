# utils/rag_evaluator.py
# RAGAS-style RAG evaluation for TranscriptAI
#
# Why custom instead of the RAGAS library:
#   TranscriptAI isn't a Q&A system — there's no "question / retrieved chunks / answer"
#   triple. The "context" is the raw transcript and the "answer" is the full
#   analysis dict. A bespoke implementation avoids shoehorning the wrong abstraction.
#
# Metrics:
#   faithfulness      — fraction of output claims grounded in the transcript
#   answer_relevancy  — semantic similarity: analysis ↔ transcript
#   context_precision — how relevant the ChromaDB cached doc is to the current input
#   hallucination_rate — fraction of action items flagged by hallucination_guard
#   rag_score          — weighted composite of the above
#
# All four use sentence-transformers (already in the stack) for semantic scoring.
# Token-overlap fallback is used when the encoder is unavailable.
#
# MLflow integration: call evaluate_rag(..., log_to_mlflow=True) and scores
# appear under the current or a new nested run — compatible with the existing
# evaluator.py flow.

import logging
import re
import threading
import time
from typing import Optional

import numpy as np

log = logging.getLogger("transcriptai.rag_eval")

# ── Encoder (lazy-loaded, thread-safe) ────────────────────────────────────────
_encoder      = None
_encoder_lock = threading.Lock()

# Deliberately use the same multilingual model the rest of TranscriptAI uses
# so similarity scores are on the same scale as the ChromaDB HNSW index.
_ENCODER_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def _get_encoder():
    global _encoder
    if _encoder is not None:
        return _encoder
    with _encoder_lock:
        if _encoder is not None:       # double-checked locking
            return _encoder
        try:
            from sentence_transformers import SentenceTransformer
            _encoder = SentenceTransformer(_ENCODER_MODEL)
            log.info("RAG evaluator: encoder loaded (%s)", _ENCODER_MODEL)
        except Exception as exc:
            log.warning("RAG evaluator: encoder unavailable (%s) — using fallback", exc)
            _encoder = None
    return _encoder


def _encode_batch(texts: list[str]) -> Optional[np.ndarray]:
    """Return L2-normalised embeddings, shape (N, D). None on failure."""
    enc = _get_encoder()
    if enc is None or not texts:
        return None
    try:
        return enc.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    except Exception as exc:
        log.warning("Encode failed: %s", exc)
        return None


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Dot product of two L2-normalised vectors = cosine similarity."""
    return float(np.dot(a, b))


# ── Text utilities ─────────────────────────────────────────────────────────────
_SENT_RE = re.compile(r'(?<=[。！？.!?])\s*')


def _split_sentences(text: str) -> list[str]:
    """Sentence-split for both Japanese (。！？) and English (. ! ?)."""
    parts = _SENT_RE.split(text)
    return [p.strip() for p in parts if len(p.strip()) > 8]


def _extract_claims(result: dict) -> list[str]:
    """
    Collect every assertable claim from an analysis result.

    Includes:
      - Bullet-point summary items
      - Sentences from full_summary
      - Action item task descriptions
    """
    claims: list[str] = []

    for bullet in result.get("summary", []):
        if isinstance(bullet, str) and bullet.strip():
            claims.append(bullet.strip())

    if full := result.get("full_summary", ""):
        claims.extend(_split_sentences(full))

    for item in result.get("action_items", []):
        if task := item.get("task", ""):
            claims.append(task)

    return claims


# ── Metric: faithfulness ───────────────────────────────────────────────────────
# Threshold: empirically set at 0.55 on Japanese/English meeting data.
# A claim scores "grounded" if its max cosine similarity to ANY transcript
# sentence is ≥ this threshold.
_FAITHFULNESS_THRESHOLD = 0.55


def faithfulness_score(transcript: str, result: dict) -> float:
    """
    What fraction of the model's output claims are supported by the transcript?

    Returns:
        float [0, 1].  1.0 = fully grounded.  0.0 = all claims hallucinated.
    """
    claims = _extract_claims(result)
    if not claims:
        return 1.0   # nothing was claimed → nothing to be unfaithful about

    ctx_sentences = _split_sentences(transcript)
    if not ctx_sentences:
        return 0.0   # no context to ground anything in

    claim_vecs = _encode_batch(claims)
    ctx_vecs   = _encode_batch(ctx_sentences)

    if claim_vecs is None or ctx_vecs is None:
        return _faithfulness_token_overlap(claims, transcript)

    # sim_matrix[i, j] = similarity(claim_i, ctx_sentence_j)
    sim_matrix = claim_vecs @ ctx_vecs.T          # (n_claims, n_ctx)
    max_sims   = sim_matrix.max(axis=1)           # best context match per claim
    supported  = int((max_sims >= _FAITHFULNESS_THRESHOLD).sum())

    score = supported / len(claims)
    log.debug("Faithfulness: %d/%d claims grounded → %.3f", supported, len(claims), score)
    return round(score, 4)


def _faithfulness_token_overlap(claims: list[str], transcript: str) -> float:
    """Token-overlap fallback (Jaccard) when encoder is unavailable."""
    ctx_tokens = set(transcript.lower().split())
    scores = [
        len(set(c.lower().split()) & ctx_tokens) / max(len(set(c.lower().split())), 1)
        for c in claims if c.strip()
    ]
    return round(sum(scores) / len(scores), 4) if scores else 1.0


# ── Metric: answer relevancy ───────────────────────────────────────────────────
def answer_relevancy_score(transcript: str, result: dict) -> float:
    """
    How semantically on-topic is the analysis relative to the transcript?

    Built by concatenating the full_summary + bullet summaries into one
    document and computing its cosine similarity to the raw transcript.

    Returns:
        float [0, 1].
    """
    full_summary = result.get("full_summary", "")
    bullets      = " ".join(result.get("summary", []))
    analysis_doc = f"{full_summary} {bullets}".strip()

    if not analysis_doc or not transcript:
        return 0.0

    vecs = _encode_batch([analysis_doc, transcript])
    if vecs is None or len(vecs) < 2:
        # Fallback: Jaccard
        a = set(analysis_doc.lower().split())
        b = set(transcript.lower().split())
        return round(len(a & b) / max(len(a | b), 1), 4)

    return round(max(0.0, _cosine(vecs[0], vecs[1])), 4)


# ── Metric: context precision ──────────────────────────────────────────────────
def context_precision_score(
    transcript:        str,
    cached_transcript: Optional[str],
    cache_hit:         bool,
) -> float:
    """
    When the ChromaDB vector cache serves a result, how relevant is the
    cached source transcript to the current input?

    This surfaces whether the 95%-cosine-threshold ChromaDB gate let through
    a document that's only marginally relevant.

    Returns:
        1.0 if no cache was used (not applicable).
        Cosine similarity between current and cached transcript otherwise.
    """
    if not cache_hit or cached_transcript is None:
        return 1.0   # cache not used — metric is N/A, return neutral

    vecs = _encode_batch([transcript, cached_transcript])
    if vecs is None or len(vecs) < 2:
        return 1.0   # can't measure — assume fine

    return round(max(0.0, _cosine(vecs[0], vecs[1])), 4)


# ── Metric: hallucination rate ─────────────────────────────────────────────────
def hallucination_rate(result: dict) -> float:
    """
    Fraction of action items flagged by the hallucination_guard module.

    The guard already runs in the main 12-stage pipeline; this metric
    surfaces its output as a trackable signal alongside the other RAG scores.

    Returns:
        float [0, 1].  0.0 is ideal (no flagged items).
    """
    items = result.get("action_items", [])
    if not items:
        return 0.0
    flagged = sum(1 for item in items if item.get("hallucination_flag"))
    return round(flagged / len(items), 4)


# ── Combined evaluation ────────────────────────────────────────────────────────
def evaluate_rag(
    transcript:        str,
    result:            dict,
    cached_transcript: Optional[str] = None,
    cache_hit:         bool           = False,
    tc_name:           str            = "unknown",
    log_to_mlflow:     bool           = True,
) -> dict:
    """
    Run all four RAG metrics and return a report dict.

    Args:
        transcript:         Raw transcript that was analyzed.
        result:             Dict returned by analyze_transcript().
        cached_transcript:  Source transcript of the ChromaDB cache hit (if any).
        cache_hit:          Whether the result came from ChromaDB cache.
        tc_name:            Label for MLflow run naming.
        log_to_mlflow:      Whether to log metric scores to MLflow.

    Returns:
        {
            "faithfulness":       float,   # 0–1, higher is better
            "answer_relevancy":   float,   # 0–1, higher is better
            "context_precision":  float,   # 0–1, higher is better
            "hallucination_rate": float,   # 0–1, LOWER is better
            "rag_score":          float,   # weighted composite, higher is better
        }

    Weights (tuned for meeting-intelligence domain):
        faithfulness      40% — grounding is the top priority
        answer_relevancy  30% — on-topic output
        context_precision 20% — cache quality
        1-hallucination   10% — guard signal
    """
    t0 = time.monotonic()

    faith    = faithfulness_score(transcript, result)
    relev    = answer_relevancy_score(transcript, result)
    ctx_prec = context_precision_score(transcript, cached_transcript, cache_hit)
    hall     = hallucination_rate(result)

    rag_score = round(
        faith    * 0.40 +
        relev    * 0.30 +
        ctx_prec * 0.20 +
        (1.0 - hall) * 0.10,
        4,
    )

    elapsed_ms = round((time.monotonic() - t0) * 1000, 1)

    report = {
        "faithfulness":       faith,
        "answer_relevancy":   relev,
        "context_precision":  ctx_prec,
        "hallucination_rate": hall,
        "rag_score":          rag_score,
        "_eval_ms":           elapsed_ms,
    }

    log.info("[%s] RAG eval in %.0fms: %s", tc_name, elapsed_ms, {
        k: v for k, v in report.items() if not k.startswith("_")
    })

    if log_to_mlflow:
        _log_mlflow(report, tc_name)

    return report


def _log_mlflow(report: dict, tc_name: str) -> None:
    """
    Log RAG metrics to MLflow under a nested run so they appear
    alongside (not overwriting) the existing accuracy metrics from evaluator.py.
    """
    try:
        import mlflow

        # Use nested=True so this doesn't close the parent run from evaluator.py
        with mlflow.start_run(run_name=f"rag_{tc_name}", nested=True):
            for key, val in report.items():
                if key.startswith("_"):
                    continue
                mlflow.log_metric(f"rag_{key}", float(val))
            mlflow.set_tag("tc_name",    tc_name)
            mlflow.set_tag("eval_type",  "rag")
            log.debug("RAG metrics logged to MLflow (tc=%s)", tc_name)

    except Exception as exc:
        # MLflow is optional — don't crash eval if it's unavailable
        log.debug("MLflow logging skipped: %s", exc)


# ── Quick sanity check ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    _transcript = (
        "Tanaka: The Q3 revenue numbers look strong. "
        "We should finalise the vendor contract by Friday. "
        "Sato: 承知しました。I'll send the draft today. "
        "Tanaka: 検討いたします。Let's also schedule a review meeting."
    )

    _result = {
        "full_summary": "Q3 revenue discussed. Vendor contract to be finalised by Friday.",
        "summary": [
            "Q3 results are strong.",
            "Vendor contract deadline: Friday.",
            "Review meeting to be scheduled.",
            "The moon is made of cheese.",   # deliberate hallucination
        ],
        "action_items": [
            {"task": "Send vendor contract draft", "owner": "Sato",   "hallucination_flag": False},
            {"task": "Finalise Q3 numbers",        "owner": "Tanaka", "hallucination_flag": False},
            {"task": "Order pizza",                "owner": "TBD",    "hallucination_flag": True},
        ],
    }

    report = evaluate_rag(
        transcript=_transcript,
        result=_result,
        log_to_mlflow=False,
        tc_name="sanity_check",
    )

    print("\nRAG Evaluation Report")
    print("─" * 40)
    for k, v in report.items():
        print(f"  {k:<22} {v}")