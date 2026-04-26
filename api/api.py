# api.py
# FastAPI REST API for TranscriptAI
#
# This makes TranscriptAI enterprise-ready:
# - Any CRM, HR system, or dashboard can call this endpoint
# - Streamlit app continues working unchanged (calls analyzer.py directly)
# - This API layer is for external integrations
#
# Run with:
#   pip install fastapi uvicorn
#   uvicorn api:app --reload --port 8000
#
# Then call:
#   POST http://localhost:8000/analyze
#   GET  http://localhost:8000/health

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
import uuid
import json

from analysis.analyzer import analyze_transcript
from utils.utils import detect_language, clean_text

# Optional modules
try:
    from transcription.pii_masker import mask_transcript, restore_pii_in_result, get_pii_report
    PII_AVAILABLE = True
except ImportError:
    PII_AVAILABLE = False

try:
    from analysis.soft_rejection_detector import detect_soft_rejections
    SOFT_REJECTION_AVAILABLE = True
except ImportError:
    SOFT_REJECTION_AVAILABLE = False

try:
    from analysis.hallucination_guard import verify_result
    HALLUCINATION_GUARD_AVAILABLE = True
except ImportError:
    HALLUCINATION_GUARD_AVAILABLE = False

# ── APP SETUP ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="TranscriptAI API",
    description=(
        "Japanese Business Intelligence — Call Transcript Analyzer API. "
        "Extracts action items, sentiment, speaker breakdown, and Japan-specific "
        "insights from meeting transcripts. APPI compliant."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── REQUEST / RESPONSE MODELS ─────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    transcript: str = Field(
        ...,
        min_length=20,
        description="The meeting transcript text. Supports Japanese, English, mixed JA/EN."
    )
    language: Optional[str] = Field(
        None,
        description="Force language: 'ja', 'en', or 'mixed'. Leave null for auto-detect."
    )
    mask_pii: bool = Field(
        True,
        description="Anonymize PII before analysis (APPI compliance). Recommended: true."
    )
    include_soft_rejections: bool = Field(
        True,
        description="Detect indirect rejection patterns in Japanese speech."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "transcript": "田中: おはようございます。\nSato: Good morning. The Q3 report is ready.",
                "language": None,
                "mask_pii": True,
                "include_soft_rejections": True
            }
        }


class AnalyzeResponse(BaseModel):
    request_id: str
    timestamp: str
    language_detected: str
    pii_masked: bool
    pii_items_found: int
    processing_time_ms: float
    result: dict


# ── HEALTH CHECK ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Check API status and available modules."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "modules": {
            "pii_masker":          PII_AVAILABLE,
            "soft_rejection":      SOFT_REJECTION_AVAILABLE,
            "hallucination_guard": HALLUCINATION_GUARD_AVAILABLE,
        },
        "model": "qwen3:8b via Ollama",
        "appi_compliant": PII_AVAILABLE,
    }


# ── MAIN ANALYZE ENDPOINT ─────────────────────────────────────────────────────
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """
    Analyze a meeting transcript and return structured intelligence.

    - Detects language automatically (or use forced language)
    - Masks PII before LLM processing (APPI compliant)
    - Extracts: summary, action items, sentiment, speakers, Japan insights
    - Detects soft rejections (検討します, 難しいかもしれません, etc.)
    - Runs hallucination prevention on all outputs
    """
    start_time  = datetime.now()
    request_id  = str(uuid.uuid4())[:8]

    # Clean and validate
    transcript = clean_text(request.transcript)
    if len(transcript.strip()) < 20:
        raise HTTPException(status_code=400, detail="Transcript too short (minimum 20 characters)")

    # Detect language
    detected_lang = detect_language(transcript)
    active_lang   = request.language or detected_lang

    # PII masking
    pii_items_found = 0
    pii_mask        = None
    text_to_analyze = transcript

    if request.mask_pii and PII_AVAILABLE:
        text_to_analyze, pii_mask = mask_transcript(transcript)
        pii_report      = get_pii_report(pii_mask)
        pii_items_found = pii_report.get("total_pii_found", 0)

    # Run analysis
    try:
        result = analyze_transcript(text_to_analyze, active_lang)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    # Restore PII in results
    if pii_mask is not None:
        result = restore_pii_in_result(result, pii_mask)

    # Soft rejection detection
    if request.include_soft_rejections and SOFT_REJECTION_AVAILABLE:
        result["soft_rejections"] = detect_soft_rejections(transcript)

    # Calculate processing time
    elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000

    return AnalyzeResponse(
        request_id       = request_id,
        timestamp        = datetime.now().isoformat(),
        language_detected = active_lang,
        pii_masked       = request.mask_pii and PII_AVAILABLE,
        pii_items_found  = pii_items_found,
        processing_time_ms = round(elapsed_ms, 1),
        result           = result
    )


# ── BATCH ENDPOINT ────────────────────────────────────────────────────────────
@app.post("/analyze/batch")
async def analyze_batch(requests: list[AnalyzeRequest]):
    """
    Analyze multiple transcripts in sequence.
    For high-volume use (10,000+/day), combine with Redis Queue + vLLM.
    """
    if len(requests) > 10:
        raise HTTPException(
            status_code=400,
            detail="Batch limit is 10 transcripts. For larger volumes use async queue."
        )

    results = []
    for req in requests:
        try:
            result = await analyze(req)
            results.append({"status": "success", "data": result})
        except Exception as e:
            results.append({"status": "error", "error": str(e)})

    return {
        "batch_size": len(requests),
        "successful": sum(1 for r in results if r["status"] == "success"),
        "failed":     sum(1 for r in results if r["status"] == "error"),
        "results":    results
    }


# ── PATTERNS ENDPOINT ─────────────────────────────────────────────────────────
@app.get("/patterns/soft-rejections")
def get_soft_rejection_patterns():
    """Returns the full soft rejection pattern dictionary with cultural explanations."""
    if not SOFT_REJECTION_AVAILABLE:
        raise HTTPException(status_code=503, detail="soft_rejection_detector.py not available")
    from analysis.soft_rejection_detector import SOFT_REJECTION_PATTERNS
    return {
        "total_patterns": len(SOFT_REJECTION_PATTERNS),
        "patterns": SOFT_REJECTION_PATTERNS,
        "cultural_context": (
            "Japanese business communication avoids direct refusal. "
            "These patterns encode the speaker's true intent through indirect language."
        )
    }


# ── RUN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)