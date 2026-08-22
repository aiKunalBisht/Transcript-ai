"""
main.py - TranscriptAI v3.1
FastAPI server. Run: uvicorn main:app --reload --port 7860

v3.1 fixes (June 2026):
  - Removed duplicate @app.post("/export/cultural-insights") route (caused 500 on gijiroku too)
  - Added ensure_speaker_labels() call before analyze_transcript() for unlabeled transcripts
  - Sentiment prompt now scores communicative register, not emotional valence
  - Health score capped at 22 for explicit contract termination meetings
  - CRITICAL risk level added to soft_rejection_detector output
"""
import asyncio, io, json as _json, os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
load_dotenv()

# ── Optional Google OAuth ─────────────────────────────────────────────────────
AUTH_ENABLED = os.getenv("AUTH_ENABLED", "0") == "1"
_AUTH_AVAILABLE = False

if AUTH_ENABLED:
    try:
        from starlette.middleware.sessions import SessionMiddleware
        from authlib.integrations.starlette_client import OAuth
        _AUTH_AVAILABLE = True
    except ImportError:
        print("[TRANSCRIPT_AI] authlib not installed — auth disabled. "
              "pip install authlib itsdangerous", flush=True)
        AUTH_ENABLED = False


def _setup_auth(app):
    """Attach session middleware and OAuth client to the app if auth is enabled."""
    if not AUTH_ENABLED or not _AUTH_AVAILABLE:
        return None
    secret = os.getenv("SESSION_SECRET", "change-me-in-production")
    app.add_middleware(SessionMiddleware, secret_key=secret,
                       max_age=60 * 60 * 24 * 30)
    oauth = OAuth()
    oauth.register(
        name="google",
        client_id=os.getenv("GOOGLE_CLIENT_ID", ""),
        client_secret=os.getenv("GOOGLE_CLIENT_SECRET", ""),
        server_metadata_url=(
            "https://accounts.google.com/.well-known/openid-configuration"
        ),
        client_kwargs={"scope": "openid email profile"},
    )
    return oauth


def get_current_user(request: Request) -> str | None:
    """
    Returns the authenticated user_id (Google sub) for this request.
    Returns None when auth is disabled or the user is not logged in.
    """
    if not AUTH_ENABLED:
        return None
    return request.session.get("user_id")

from analysis.analyzer import analyze_transcript
from utils import detect_language, clean_text, parse_uploaded_file
from utils.html_renderer import build_results_html

# ── Optional modules ──────────────────────────────────────────────────────────
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
    from agents.slide_architect import SlideArchitectAgent
    SLIDE_ARCHITECT_AVAILABLE = True
except ImportError:
    SLIDE_ARCHITECT_AVAILABLE = False

try:
    from agents.cultural_insights_formatter import format_cultural_insights
    CULTURAL_INSIGHTS_AVAILABLE = True
except ImportError:
    CULTURAL_INSIGHTS_AVAILABLE = False

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
    from analysis.hindi_analyzer import detect_hindi_patterns as detect_hindi_nlp
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
                "🗣️ Hindi Analysis"        if lang == "hi" else
                "🌐 Insights"
            ),
            "insight_tab_enabled": True,
        }

# ── Evaluation module ──────────────────────────────────────────────────────────
try:
    from utils.evaluator import evaluate, MLFLOW_AVAILABLE
    from tests.test_data import TEST_CASES
    from utils.html_renderer import build_evaluation_html
    EVAL_AVAILABLE = True
except ImportError:
    EVAL_AVAILABLE = False
    MLFLOW_AVAILABLE = False
    TEST_CASES = []

AUDIO_EXT = {".mp3", ".wav", ".m4a", ".mp4", ".ogg", ".webm"}
TEXT_EXT  = {".txt", ".vtt", ".json"}

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="TranscriptAI", version="3.2.0", docs_url="/docs")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

_oauth = _setup_auth(app)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

from api.api import app as _rest_api
app.mount("/api", _rest_api)


# ── Speaker label detection (unlabeled transcript fallback) ───────────────────
import re as _re

_SPEAKER_PATTERNS = [
    r"^\*?\*?[\w\s\u3000-\u9fff]{1,40}\*?\*?\s*[：:]\s*\S",
    r"^\[[\w\s\u3000-\u9fff]{1,40}\]\s*[：:]\s*\S",
    r"^【[\w\s\u3000-\u9fff]{1,40}】\s*[：:]\s*\S",
]

def _has_speaker_labels(text: str) -> bool:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    hits = sum(
        1 for line in lines[:40]
        if any(_re.match(p, line) for p in _SPEAKER_PATTERNS)
    )
    return hits >= 2

def _strip_markdown_bold(text: str) -> str:
    # Remove markdown bold markers from speaker labels before analysis.
    # '**Japanese Director:** text' becomes 'Japanese Director: text'
    cleaned = []
    for line in text.split("\n"):
        stripped = line
        if stripped.startswith("**"):
            stripped = stripped[2:]
        stripped = stripped.replace("**", "")
        cleaned.append(stripped)
    return "\n".join(cleaned)

def _ensure_speaker_labels(text: str):
    """Return (processed_text, was_unlabeled)."""
    text = _strip_markdown_bold(text)
    if _has_speaker_labels(text):
        return text, False
    paragraphs = [p.strip() for p in _re.split(r"\n{2,}", text) if p.strip()]
    if not paragraphs:
        paragraphs = [l.strip() for l in text.split("\n") if l.strip()]
    labeled = [f"Speaker {i+1}: {p}" for i, p in enumerate(paragraphs)]
    return "\n".join(labeled), True


# ── Cache stats ───────────────────────────────────────────────────────────────
def _get_cache_stats(user_id: str | None = None) -> dict | None:
    try:
        from utils.vector_cache import get_cache_stats
        vc = get_cache_stats(user_id=user_id)
        return vc if vc.get("available") else None
    except Exception:
        return None


# ── Pages ─────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {
        "cache_stats":    _get_cache_stats(get_current_user(request)),
        "current_user":   get_current_user(request),
        "auth_enabled":   AUTH_ENABLED,
    })


# ── Auth routes (only active when AUTH_ENABLED=1) ─────────────────────────────

@app.get("/auth/login")
async def auth_login(request: Request):
    """Redirect user to Google consent screen."""
    if not AUTH_ENABLED or not _oauth:
        return JSONResponse({"error": "Auth not enabled — set AUTH_ENABLED=1"}, 400)
    redirect_uri = request.url_for("auth_callback")
    return await _oauth.google.authorize_redirect(request, redirect_uri)


@app.get("/auth/callback", name="auth_callback")
async def auth_callback(request: Request):
    """
    Google redirects here after the user grants consent.
    Exchange code → token → decode user info → store in session.
    """
    if not AUTH_ENABLED or not _oauth:
        return RedirectResponse("/")
    try:
        token     = await _oauth.google.authorize_access_token(request)
        user_info = token["userinfo"]
        request.session["user_id"] = user_info["sub"]
        request.session["email"]   = user_info.get("email", "")
        request.session["name"]    = user_info.get("name", "")
        await _upsert_user(
            user_id = user_info["sub"],
            email   = user_info.get("email", ""),
            name    = user_info.get("name", ""),
        )
    except Exception as exc:
        print(f"[AUTH] callback error: {exc}", flush=True)
    return RedirectResponse("/")


@app.get("/auth/logout")
async def auth_logout(request: Request):
    """Clear session — user is logged out."""
    request.session.clear()
    return RedirectResponse("/")


@app.get("/auth/me")
async def auth_me(request: Request):
    """Returns current user info — used by frontend to show name/avatar."""
    user_id = get_current_user(request)
    if not user_id:
        return JSONResponse({"authenticated": False})
    return JSONResponse({
        "authenticated": True,
        "user_id":       user_id,
        "email":         request.session.get("email", ""),
        "name":          request.session.get("name", ""),
    })


@app.get("/export", response_class=HTMLResponse)
async def export_page(request: Request):
    return templates.TemplateResponse(request, "export.html", {
        "pptx_available":               PPTX_AVAILABLE,
        "gijiroku_available":           GIJIROKU_AVAILABLE,
        "cultural_insights_available":  CULTURAL_INSIGHTS_AVAILABLE,
        "cache_stats":                  _get_cache_stats(get_current_user(request)),
    })


@app.get("/evaluate", response_class=HTMLResponse)
async def evaluate_page(request: Request):
    return templates.TemplateResponse(request, "evaluate.html", {
        "eval_available":   EVAL_AVAILABLE,
        "test_case_count":  len(TEST_CASES) if EVAL_AVAILABLE else 0,
        "mlflow_available": MLFLOW_AVAILABLE,
        "cache_stats":      _get_cache_stats(get_current_user(request)),
    })


@app.post("/evaluate/run", response_class=HTMLResponse)
async def evaluate_run():
    if not EVAL_AVAILABLE:
        return HTMLResponse(
            content=_err("Evaluation module unavailable — check utils/evaluator.py and tests/test_data.py."),
            status_code=500,
        )
    if not TEST_CASES:
        return HTMLResponse(
            content=_err("No ground-truth test cases found in tests/test_data.py."),
            status_code=400,
        )

    async def _run_one(tc: dict) -> dict:
        prediction = await asyncio.to_thread(
            analyze_transcript, tc["transcript"], tc["language"], bypass_cache=True
        )
        report = await asyncio.to_thread(
            evaluate, prediction, tc["ground_truth"], tc["transcript"],
            tc_name=tc["name"], provider=prediction.get("_provider", "unknown"),
        )
        return {
            "tc_id":       tc["id"],
            "tc_name":     tc["name"],
            "provider":    prediction.get("_provider", "unknown"),
            "duration_ms": prediction.get("_duration_ms", 0),
            "report":      report,
        }

    try:
        reports = await asyncio.gather(*[_run_one(tc) for tc in TEST_CASES])
    except Exception as exc:
        return HTMLResponse(content=_err(f"Evaluation run failed: {exc}"), status_code=500)

    html = build_evaluation_html(list(reports), MLFLOW_AVAILABLE)
    return HTMLResponse(content=html)


# ── /transcribe ───────────────────────────────────────────────────────────────
class _FileShim:
    def __init__(self, filename: str, data: bytes):
        self.name = filename
        self._data = data
    def getvalue(self): return self._data
    def read(self):     return self._data


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    filename = file.filename or ""
    ext = Path(filename).suffix.lower()
    content = await file.read()

    if ext in AUDIO_EXT:
        if not AUDIO_AVAILABLE:
            return JSONResponse({"success": False, "error": "Audio transcription module unavailable."})
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


# ── /analyze-text ─────────────────────────────────────────────────────────────
@app.post("/analyze-text", response_class=HTMLResponse)
async def analyze_text_route(
    request:    Request,
    transcript: str           = Form(...),
    language:   Optional[str] = Form(None),
    mask_pii:   bool          = Form(True),
):
    if len(transcript.strip()) < 20:
        return HTMLResponse(content=_err("Transcript too short (min 20 chars)."), status_code=400)
    try:
        cleaned = clean_text(transcript)
        detected_lang = language or detect_language(cleaned)

        cleaned, was_unlabeled = _ensure_speaker_labels(cleaned)

        pii_report = pii_mask = None
        text_to_analyze = cleaned
        if mask_pii and PII_AVAILABLE:
            text_to_analyze, pii_mask = mask_transcript(cleaned)
            pii_report = get_pii_report(pii_mask)

        result = await asyncio.to_thread(
            analyze_transcript, text_to_analyze, detected_lang,
            user_id=get_current_user(request)
        )

        if pii_mask is not None:
            result = restore_pii_in_result(result, pii_mask)
        if SOFT_REJECTION_AVAILABLE:
            result["soft_rejections"] = detect_soft_rejections(cleaned)

        result["_detected_language"]    = detected_lang
        result["_unlabeled_transcript"] = was_unlabeled

        features = get_features(detected_lang)
        html = build_results_html(result, detected_lang, features, pii_report)
        tag = ('<div id="tai-result-data" style="display:none">' +
               _json.dumps(result, ensure_ascii=False) + '</div>')
        return HTMLResponse(content=html + tag)

    except Exception as exc:
        return HTMLResponse(content=_err(str(exc)), status_code=500)


# ── Export routes ─────────────────────────────────────────────────────────────
@app.post("/export/pptx")
async def export_pptx(request: Request):
    if not PPTX_AVAILABLE or not SLIDE_ARCHITECT_AVAILABLE:
        raise HTTPException(503, "PPTX builder not available")
    body   = await request.json()
    result = body.get("result", body)

    from analysis.analyzer import _get_groq_key
    agent = SlideArchitectAgent(groq_api_key=_get_groq_key())
    plan  = await asyncio.to_thread(
        agent.plan, result, result.get("_detected_language", "en")
    )
    pptx_bytes = await asyncio.to_thread(build_pptx, plan)
    return StreamingResponse(
        io.BytesIO(pptx_bytes),
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        headers={"Content-Disposition": "attachment; filename=meeting_report.pptx"},
    )


@app.post("/export/cultural-insights")
async def export_cultural_insights(request: Request):
    if not CULTURAL_INSIGHTS_AVAILABLE:
        raise HTTPException(503, "Cultural insights formatter not available")
    body = await request.json()
    try:
        text = await asyncio.to_thread(
            format_cultural_insights, body.get("result", body), True
        )
        return JSONResponse({"cultural_insights": text})
    except Exception as exc:
        return JSONResponse({"error": str(exc)[:500]}, status_code=500)


@app.post("/export/gijiroku")
async def export_gijiroku(request: Request):
    if not GIJIROKU_AVAILABLE:
        raise HTTPException(503, "Gijiroku formatter not available")
    body = await request.json()
    try:
        text = await asyncio.to_thread(
            format_gijiroku, body.get("result", body), True
        )
        return JSONResponse({"gijiroku": text})
    except Exception as exc:
        return JSONResponse({"error": str(exc)[:500]}, status_code=500)


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
            flag = " ⚠" if i.get("hallucination_flag") else ""
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
        "status": "healthy", "version": "3.2.0",
        "provider": "groq" if os.getenv("GROQ_API_KEY") else "mock",
        "appi_compliant": PII_AVAILABLE,
        "modules": {
            "audio":              AUDIO_AVAILABLE,
            "pii_masker":         PII_AVAILABLE,
            "soft_rejection":     SOFT_REJECTION_AVAILABLE,
            "hallucination":      HALLUCINATION_GUARD_AVAILABLE,
            "pptx":               PPTX_AVAILABLE,
            "gijiroku":           GIJIROKU_AVAILABLE,
            "cultural_insights":  CULTURAL_INSIGHTS_AVAILABLE,
            "slide_architect":    SLIDE_ARCHITECT_AVAILABLE,
            "language_intel":     LANGUAGE_INTEL_AVAILABLE,
            "evaluation":         EVAL_AVAILABLE,
        },
    }


# ── SEO: robots.txt ───────────────────────────────────────────────────────────
# FIX: FastAPI does not auto-serve files at the project root.
# Without this route every crawler gets a 404 on /robots.txt.
@app.get("/robots.txt", response_class=HTMLResponse)
async def robots_txt():
    return HTMLResponse(open("robots.txt").read(), media_type="text/plain")


# ── SEO: sitemap.xml ──────────────────────────────────────────────────────────
# Submit this to Google Search Console after deploy:
#   https://kunalthebeast-transcriptai.hf.space/sitemap.xml
@app.get("/sitemap.xml", response_class=HTMLResponse)
async def sitemap():
    from datetime import date
    today = date.today().isoformat()
    base  = "https://kunalthebeast-transcriptai.hf.space"
    pages = [
        ("",          "daily",  "1.0"),
        ("/export",   "weekly", "0.7"),
        ("/evaluate", "weekly", "0.6"),
    ]
    urls = "\n".join(
        f"""  <url>
    <loc>{base}{path}</loc>
    <lastmod>{today}</lastmod>
    <changefreq>{freq}</changefreq>
    <priority>{priority}</priority>
  </url>"""
        for path, freq, priority in pages
    )
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
{urls}
</urlset>"""
    return HTMLResponse(content=xml, media_type="application/xml")


# ── SEO: manifest.json (PWA metadata — helps Google classify the app) ─────────
@app.get("/manifest.json")
async def manifest():
    return JSONResponse({
        "name":             "TranscriptAI",
        "short_name":       "TranscriptAI",
        "description":      "Japanese & Multilingual Meeting Intelligence AI — detects nemawashi, keigo, soft rejections, and meeting outcomes.",
        "start_url":        "/",
        "display":          "standalone",
        "background_color": "#FDF8F5",
        "theme_color":      "#D96080",
        "categories":       ["business", "productivity"],
        "lang":             "en",
        "keywords":         "Japanese meeting, nemawashi, soft rejection, keigo, transcript analysis, multilingual AI",
    })


# ── Helpers ───────────────────────────────────────────────────────────────────
def _err(msg: str) -> str:
    return (f'<div style="background:var(--red-bg);border-left:3px solid var(--red);' 
            f'border-radius:0 10px 10px 0;padding:14px 18px;color:#3C2416;margin-top:12px">' 
            f'<b style="color:var(--red)">⚠ Error</b><br>{msg}</div>')


# ── User store — aiosqlite (non-blocking) ─────────────────────────────────────
# C3 fix: replaced blocking sqlite3 with aiosqlite.
try:
    import aiosqlite as _aiosqlite
    _AIOSQLITE = True
except ImportError:
    import sqlite3 as _sqlite3
    _AIOSQLITE = False

_DB_PATH = Path("users.db")


def _init_user_db():
    """Create users table on first startup."""
    try:
        import sqlite3
        con = sqlite3.connect(_DB_PATH)
        con.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id    TEXT PRIMARY KEY,
                email      TEXT UNIQUE NOT NULL,
                name       TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                last_seen  TEXT,
                total_analyses INTEGER DEFAULT 0
            )
        """)
        con.commit()
        con.close()
    except Exception as exc:
        print(f"[DB] init_user_db failed: {exc}", flush=True)


async def _upsert_user(user_id: str, email: str, name: str) -> None:
    """Insert or update a user record — fully async, never blocks the event loop."""
    now = __import__("datetime").datetime.utcnow().isoformat()
    sql = """
        INSERT INTO users (user_id, email, name, created_at, last_seen)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET last_seen=excluded.last_seen
    """
    try:
        if _AIOSQLITE:
            async with _aiosqlite.connect(_DB_PATH) as db:
                await db.execute(sql, (user_id, email, name, now, now))
                await db.commit()
        else:
            await asyncio.to_thread(_sync_upsert, user_id, email, name, now, sql)
    except Exception as exc:
        print(f"[DB] upsert_user failed: {exc}", flush=True)


def _sync_upsert(user_id: str, email: str, name: str, now: str, sql: str) -> None:
    """Sync fallback for _upsert_user when aiosqlite is unavailable."""
    import sqlite3
    con = sqlite3.connect(_DB_PATH)
    con.execute(sql, (user_id, email, name, now, now))
    con.commit()
    con.close()


_init_user_db()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0",
                port=int(os.getenv("PORT", 7860)),
                reload=os.getenv("ENV") == "development",
                workers=1)