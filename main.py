"""
main.py - TranscriptAI v3.0
FastAPI server replacing Streamlit (app.py).
Run: uvicorn main:app --reload --port 7860

v3.1 correction: the original port of this file assumed a class-based
AudioProcessor with a .transcribe() method. The real interface (used in
app.py) is function-based: transcribe_audio(bytes, filename) -> dict,
format_transcript_with_timestamps(segments) -> str. Fixed here.

Also replicates app.py's actual two-step flow: upload/paste fills the
transcript box first (/transcribe), then a single "Analyze" button runs
the NLP pipeline (/analyze-text) — not a combined one-click upload+analyze.

v3.2 fixes:
  - BUG 1/2: Static files & templates now use absolute paths (Path(__file__).parent)
              so they resolve correctly regardless of CWD on HF Spaces.
              StaticFiles mount is guarded so a missing /static dir doesn't crash the app.
  - BUG 3:   api.api sub-app import is wrapped in try/except so a missing module
              doesn't prevent the main app from starting.
  - BUG 4/5: PPTX export validates required keys and wraps build_pptx in try/except
              so broken exports return a readable JSON error instead of a corrupt file.
  - BUG 6:   Jinja2Templates uses absolute path (same fix as StaticFiles).
  - BUG 7:   restore_pii_in_result is now gated on both PII_AVAILABLE and pii_mask.
  - BUG 8:   Second detect_hindi_patterns import renamed to avoid silent overwrite.
  - BUG 9:   Gijiroku export validates that format_gijiroku returns a non-None value.
  - BUG 10:  /transcribe checks GROQ_API_KEY before attempting audio transcription.
  - BUG 11:  _get_cache_stats logs exceptions instead of swallowing them silently.
  - BUG 12:  CORS: allow_credentials=True is incompatible with allow_origins=["*"];
              switched to allow_credentials=False.
"""
import asyncio, io, json as _json, logging, os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from analysis.analyzer import analyze_transcript
from utils import detect_language, clean_text, parse_uploaded_file
from utils.html_renderer import build_results_html

logger = logging.getLogger(__name__)

# ── Absolute base directory (works regardless of CWD on HF Spaces) ───────────
BASE_DIR = Path(__file__).parent

# ── Optional modules — same guard pattern as app.py ───────────────────────────
try:
    from transcription.pii_masker import mask_transcript, restore_pii_in_result, get_pii_report
    PII_AVAILABLE = True
except ImportError:
    PII_AVAILABLE = False

try:
    from transcription.audio_processor import (
        transcribe_audio, format_transcript_with_timestamps, MAX_FILE_SIZE_MB
    )
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    MAX_FILE_SIZE_MB = 25

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

try:
    from exporters.pptx_builder import build_pptx
    PPTX_AVAILABLE = True
except ImportError:
    PPTX_AVAILABLE = False

try:
    from agents.gijiroku_formatter import format_gijiroku
    GIJIROKU_AVAILABLE = True
except ImportError:
    GIJIROKU_AVAILABLE = False

try:
    from agents.slide_architect import build_slide_structure
    SLIDE_ARCHITECT_AVAILABLE = True
except ImportError:
    SLIDE_ARCHITECT_AVAILABLE = False

# ── get_features — ported exactly from app.py's import/fallback chain ────────
try:
    from utils.language_intelligence import get_features, detect_hindi_patterns
    LANGUAGE_INTEL_AVAILABLE = True
except ImportError:
    LANGUAGE_INTEL_AVAILABLE = False

try:
    from analysis.english_analyzer import detect_english_patterns
    ENGLISH_NLP_AVAILABLE = True
except ImportError:
    ENGLISH_NLP_AVAILABLE = False

try:
    # BUG 8 FIX: renamed to _detect_hindi_nlp to avoid silently overwriting
    # the detect_hindi_patterns already imported from utils.language_intelligence above.
    from analysis.hindi_analyzer import detect_hindi_patterns as _detect_hindi_nlp
    HINDI_NLP_AVAILABLE = True
except ImportError:
    HINDI_NLP_AVAILABLE = False
    LANGUAGE_INTEL_AVAILABLE = False
    def get_features(lang):
        has_ja = lang in ("ja", "mixed")
        return {
            "show_japan_insights": has_ja,
            "show_hindi_insights": lang == "hi",
            "show_english_insights": lang == "en",
            "show_bilingual_insights": lang == "mixed" and not has_ja,
            "show_code_switch": has_ja,
            "insight_tab_label": (
                "🔍 Communication Intelligence" if has_ja else
                "💬 English Analysis"           if lang == "en" else
                "🗣️ Hindi Analysis"             if lang == "hi" else
                "🌐 Insights"
            ),
            "insight_tab_enabled": True,
        }

AUDIO_EXT = {".mp3", ".wav", ".m4a", ".mp4", ".ogg", ".webm"}
TEXT_EXT  = {".txt", ".vtt", ".json"}

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="TranscriptAI", version="3.0.0", docs_url="/docs")

# BUG 12 FIX: allow_credentials=True is spec-invalid with allow_origins=["*"].
# Browsers reject credentialed preflight when origin is a wildcard.
# Switched to allow_credentials=False.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# BUG 1 & 2 FIX: use absolute paths so CWD on HF Spaces doesn't matter.
# Guard the mount so a missing static/ dir doesn't crash the whole app on startup.
_static_dir = BASE_DIR / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")
else:
    logger.warning(
        "static/ directory not found at %s — CSS/JS will not be served. "
        "Create the directory or check your repo structure.", _static_dir
    )

# BUG 6 FIX: absolute path for templates as well.
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# BUG 3 FIX: wrap sub-app import so a missing api/api.py doesn't kill the main app.
try:
    from api.api import app as _rest_api
    app.mount("/api", _rest_api)
except ImportError as _api_err:
    logger.warning("api.api sub-app not available (%s) — /api routes will 404.", _api_err)


# ── Shared: vector cache stats for the sidebar ────────────────────────────────
def _get_cache_stats():
    try:
        from utils.vector_cache import get_cache_stats
        vc = get_cache_stats()
        return vc if vc.get("available") else None
    except Exception as exc:
        # BUG 11 FIX: log instead of silently swallowing — helps surface deeper import failures.
        logger.warning("vector cache stats unavailable: %s", exc)
        return None


# ── Pages ─────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"cache_stats": _get_cache_stats()})


@app.get("/export", response_class=HTMLResponse)
async def export_page(request: Request):
    return templates.TemplateResponse(request, "export.html", {
        "pptx_available":     PPTX_AVAILABLE,
        "gijiroku_available": GIJIROKU_AVAILABLE,
        "cache_stats":        _get_cache_stats(),
    })


# ── /transcribe — step 1: file in, transcript text out (no analysis yet) ─────
class _FileShim:
    """Minimal shim so utils.parse_uploaded_file (written for Streamlit's
    UploadedFile) works unchanged against FastAPI's UploadFile bytes."""
    def __init__(self, filename: str, data: bytes):
        self.name = filename
        self._data = data
    def getvalue(self): return self._data
    def read(self):     return self._data


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """
    Mirrors app.py's upload handling exactly:
    - audio (.mp3/.wav/.m4a/.mp4/.webm/.ogg) -> transcribe_audio(), then
      format_transcript_with_timestamps() if segments are available
    - text (.txt/.vtt/.json) -> parse_uploaded_file()
    Returns the transcript text only. The caller fills the textarea with
    it; analysis happens separately via /analyze-text.
    """
    filename = file.filename or ""
    ext = Path(filename).suffix.lower()
    content = await file.read()

    if ext in AUDIO_EXT:
        if not AUDIO_AVAILABLE:
            return JSONResponse({"success": False, "error": "Audio transcription module unavailable."})

        # BUG 10 FIX: surface a clear message when the API key is missing rather
        # than letting transcribe_audio raise an opaque AuthenticationError.
        if not os.getenv("GROQ_API_KEY"):
            return JSONResponse({
                "success": False,
                "error": "GROQ_API_KEY is not set — audio transcription is unavailable on this deployment.",
            })

        size_mb = len(content) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            return JSONResponse({"success": False, "error": f"File too large ({size_mb:.1f} MB). Max: {MAX_FILE_SIZE_MB} MB"})
        try:
            res = await asyncio.to_thread(transcribe_audio, content, filename)
        except Exception as exc:
            return JSONResponse({"success": False, "error": str(exc)})
        if not res.get("success"):
            return JSONResponse({"success": False, "error": res.get("error", "Transcription failed")})
        seg  = format_transcript_with_timestamps(res.get("segments", []))
        text = seg or res.get("text", "")
        return JSONResponse({
            "success": True, "transcript": text,
            "meta": {
                "duration": res.get("duration", 0),
                "language": res.get("language", "?"),
                "provider": res.get("provider", ""),
            },
        })

    if ext in TEXT_EXT:
        try:
            shim = _FileShim(filename, content)
            parsed = parse_uploaded_file(shim)
            return JSONResponse({"success": True, "transcript": parsed, "meta": {"chars": len(parsed)}})
        except Exception as exc:
            return JSONResponse({"success": False, "error": str(exc)})

    return JSONResponse({"success": False, "error": f"Unsupported file type: {ext}"})


# ── /analyze-text — step 2: run the NLP pipeline on whatever's in the box ────
@app.post("/analyze-text", response_class=HTMLResponse)
async def analyze_text_route(
    transcript: str           = Form(...),
    language:   Optional[str] = Form(None),
    mask_pii:   bool          = Form(True),
):
    if len(transcript.strip()) < 20:
        return HTMLResponse(content=_err("Transcript too short (min 20 chars)."), status_code=400)
    try:
        cleaned = clean_text(transcript)
        detected_lang = language or detect_language(cleaned)

        pii_report = pii_mask = None
        text_to_analyze = cleaned
        if mask_pii and PII_AVAILABLE:
            text_to_analyze, pii_mask = mask_transcript(cleaned)
            pii_report = get_pii_report(pii_mask)

        result = await asyncio.to_thread(analyze_transcript, text_to_analyze, detected_lang)

        # BUG 7 FIX: gate restore on PII_AVAILABLE *and* pii_mask to avoid
        # calling an unavailable function if the logic path changes.
        if PII_AVAILABLE and pii_mask is not None:
            result = restore_pii_in_result(result, pii_mask)

        if SOFT_REJECTION_AVAILABLE:
            result["soft_rejections"] = detect_soft_rejections(cleaned)

        result["_detected_language"] = detected_lang

        features = get_features(detected_lang)
        html = build_results_html(result, detected_lang, features, pii_report)
        tag = ('<div id="tai-result-data" style="display:none">' +
               _json.dumps(result, ensure_ascii=False) + '</div>')
        return HTMLResponse(content=html + tag)

    except Exception as exc:
        return HTMLResponse(content=_err(str(exc)), status_code=500)


# ── Export ────────────────────────────────────────────────────────────────────
@app.post("/export/pptx")
async def export_pptx(request: Request):
    if not PPTX_AVAILABLE:
        raise HTTPException(503, "PPTX builder not available")
    body = await request.json()

    # BUG 4 FIX: validate the payload has a "result" key with meaningful content
    # before passing it to build_pptx.  Return a clear 422 instead of a corrupt file.
    result_data = body.get("result", body)
    if not isinstance(result_data, dict) or not result_data:
        raise HTTPException(422, "Request body must contain a non-empty 'result' object.")

    # BUG 5 FIX: wrap build_pptx so any exception returns a readable JSON error.
    try:
        pptx_bytes = await asyncio.to_thread(build_pptx, result_data)
    except Exception as exc:
        logger.exception("build_pptx failed")
        return JSONResponse({"error": f"PPTX generation failed: {exc}"}, status_code=500)

    return StreamingResponse(
        io.BytesIO(pptx_bytes),
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        headers={"Content-Disposition": "attachment; filename=meeting_report.pptx"},
    )


@app.post("/export/gijiroku")
async def export_gijiroku(request: Request):
    if not GIJIROKU_AVAILABLE:
        raise HTTPException(503, "Gijiroku not available")
    body = await request.json()
    result_data = body.get("result", body)

    # BUG 9 FIX: validate format_gijiroku returns a usable value.
    try:
        text = await asyncio.to_thread(format_gijiroku, result_data)
    except Exception as exc:
        logger.exception("format_gijiroku failed")
        return JSONResponse({"error": f"Gijiroku formatting failed: {exc}"}, status_code=500)

    if text is None:
        return JSONResponse({"error": "format_gijiroku returned no output."}, status_code=500)

    return JSONResponse({"gijiroku": text})


@app.post("/export/markdown")
async def export_markdown(request: Request):
    body = await request.json()
    r    = body.get("result", body)
    lines = ["# Meeting Analysis\n"]
    if r.get("full_summary"): lines += ["## Overview\n", r["full_summary"], "\n"]
    if r.get("summary"):      lines += ["## Key Points\n"] + [f"- {b}\n" for b in r["summary"]]
    if r.get("action_items"):
        lines += ["\n## Action Items\n"]
        for i in r["action_items"]:
            flag = " \u26a0" if i.get("hallucination_flag") else ""
            lines.append(f"- **{i.get('task','')}**{flag}  \n"
                         f"  Owner: {i.get('owner','TBD')}  Deadline: {i.get('deadline','TBD')}\n")
    md = "".join(lines)
    return StreamingResponse(
        io.BytesIO(md.encode("utf-8")), media_type="text/markdown",
        headers={"Content-Disposition": "attachment; filename=meeting_notes.md"},
    )


@app.post("/export/json")
async def export_json_route(request: Request):
    body = await request.json()
    raw  = _json.dumps(body.get("result", body), ensure_ascii=False, indent=2).encode("utf-8")
    return StreamingResponse(
        io.BytesIO(raw), media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=analysis.json"},
    )


@app.post("/export/txt")
async def export_txt_route(request: Request):
    """Plain-text export — same content as Markdown, no markdown syntax."""
    body = await request.json()
    r    = body.get("result", body)
    lines = ["MEETING ANALYSIS", "=" * 40, ""]
    if r.get("full_summary"):
        lines += ["OVERVIEW", "-" * 20, r["full_summary"], ""]
    if r.get("summary"):
        lines += ["KEY POINTS", "-" * 20]
        lines += [f"{i}. {b}" for i, b in enumerate(r["summary"], 1)]
        lines.append("")
    if r.get("action_items"):
        lines += ["ACTION ITEMS", "-" * 20]
        for i in r["action_items"]:
            flag = " [FLAGGED]" if i.get("hallucination_flag") else ""
            lines.append(f"- {i.get('task','')}{flag}")
            lines.append(f"    Owner: {i.get('owner','TBD')}   Deadline: {i.get('deadline','TBD')}")
        lines.append("")
    if r.get("sentiment"):
        lines += ["SENTIMENT", "-" * 20]
        lines += [f"- {s.get('speaker','')}: {s.get('score','').upper()}" for s in r["sentiment"]]
        lines.append("")
    if r.get("speakers"):
        lines += ["SPEAKERS", "-" * 20]
        for spk in r["speakers"]:
            lines.append(f"- {spk.get('name','')}: {spk.get('talk_time_pct',0)}% ({spk.get('tone','')})")
        lines.append("")
    txt = "\n".join(lines)
    return StreamingResponse(
        io.BytesIO(txt.encode("utf-8")), media_type="text/plain",
        headers={"Content-Disposition": "attachment; filename=meeting_notes.txt"},
    )


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "healthy", "version": "3.0.0", "mode": "fastapi+jinja2+htmx",
        "provider": "groq" if os.getenv("GROQ_API_KEY") else "mock",
        "appi_compliant": PII_AVAILABLE,
        "static_dir_exists": _static_dir.exists(),
        "modules": {
            "audio":              AUDIO_AVAILABLE,
            "pii_masker":         PII_AVAILABLE,
            "soft_rejection":     SOFT_REJECTION_AVAILABLE,
            "hallucination":      HALLUCINATION_GUARD_AVAILABLE,
            "pptx":               PPTX_AVAILABLE,
            "gijiroku":           GIJIROKU_AVAILABLE,
            "slide_architect":    SLIDE_ARCHITECT_AVAILABLE,
            "language_intel":     LANGUAGE_INTEL_AVAILABLE,
        },
    }


# ── Helpers ───────────────────────────────────────────────────────────────────
def _err(msg: str) -> str:
    return (f'<div style="background:var(--red-bg);border-left:3px solid var(--red);'
            f'border-radius:0 10px 10px 0;padding:14px 18px;color:#3C2416;margin-top:12px">'
            f'<b style="color:var(--red)">⚠ Error</b><br>{msg}</div>')


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0",
                port=int(os.getenv("PORT", 7860)),
                reload=os.getenv("ENV") == "development",
                workers=1)