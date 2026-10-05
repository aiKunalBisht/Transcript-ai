# main.py — TranscriptAI v3.5
# FastAPI server. Run: uvicorn main:app --reload --port 7860
#
# REQUIRES: api/async_processor.py v2.1 (user_id + requesting_user params).
#
# v3.5 fixes applied over v3.4:
#
#   FIX 3 — _setup_auth no longer crashes when SESSION_SECRET is missing
#     and AUTH_ENABLED=0. In v3.4 a missing SESSION_SECRET always raised
#     RuntimeError at startup, even for local dev runs that don't need auth.
#     Now only raises when AUTH_ENABLED=1 actually requires a secret.
#     get_current_user() already has a try/except fallback, so running
#     without SessionMiddleware is safe.
#
#   FIX 4 — evaluate_run no longer passes bypass_cache=True to
#     analyze_transcript. That kwarg is not in the analyzer.py signature,
#     causing a TypeError on every /evaluate/run call.
#
#   FIX 7 — _err() now HTML-escapes exception messages before embedding
#     them in the response. Raw exception strings could contain angle
#     brackets from tracebacks or user-controlled input, causing XSS.
#
#   FIX 9 — robots_txt route wraps open() in try/except. A missing
#     robots.txt raised FileNotFoundError → HTTP 500.

import asyncio
import html as _html        # FIX 7: for _err() escaping
import io
import json as _json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import bcrypt as _bcrypt
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
load_dotenv()

# ── Rate limiting ──────────────────────────────────────────────────────────────
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# ── Optional Google OAuth ──────────────────────────────────────────────────────
AUTH_ENABLED    = os.getenv("AUTH_ENABLED", "0") == "1"
_AUTH_AVAILABLE = False

try:
    from authlib.integrations.starlette_client import OAuth
    _AUTH_AVAILABLE = True
except ImportError:
    if AUTH_ENABLED:
        print("[TRANSCRIPT_AI] authlib not installed — Google auth disabled. "
              "pip install authlib itsdangerous", flush=True)
    AUTH_ENABLED = False


def _setup_auth(app):
    secret = os.getenv("SESSION_SECRET")
    if not secret:
        if AUTH_ENABLED:
            # FIX 3: only crash when auth is actually required
            raise RuntimeError(
                "SESSION_SECRET env variable is not set. "
                "Add it to your HF Space secrets (Settings → Variables and secrets). "
                "Use any random string of 32+ characters."
            )
        # FIX 3: AUTH_ENABLED=0 → sessions not needed.
        # get_current_user() has a try/except fallback, so skipping
        # SessionMiddleware here is safe — it just returns None for all users.
        return None

    try:
        from starlette.middleware.sessions import SessionMiddleware
        app.add_middleware(SessionMiddleware, secret_key=secret,
                           max_age=60 * 60 * 24 * 30)
    except Exception as e:
        print(f"[TRANSCRIPT_AI] SessionMiddleware error: {e}", flush=True)

    if not AUTH_ENABLED or not _AUTH_AVAILABLE:
        return None
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
    try:
        return request.session.get("user_id")
    except Exception:
        return None


from analysis.analyzer import analyze_transcript
from utils import detect_language, clean_text, parse_uploaded_file
from utils.html_renderer import build_results_html

# ── Firebase user storage ──────────────────────────────────────────────────────
try:
    from utils.firebase_client import (
        upsert_user_firebase       as _upsert_user_fb,
        get_user_by_email_firebase as _get_user_by_email_fb,
    )
    _FIREBASE_AVAILABLE = True
except ImportError:
    _FIREBASE_AVAILABLE = False
    print("[TRANSCRIPT_AI] firebase_client not found — user storage disabled.", flush=True)


async def _upsert_user(user_id: str, email: str, name: str,
                       password_hash: str = "") -> None:
    if _FIREBASE_AVAILABLE:
        await _upsert_user_fb(user_id, email, name, password_hash)


async def _get_user_by_email(email: str) -> dict | None:
    if _FIREBASE_AVAILABLE:
        return await _get_user_by_email_fb(email)
    return None


# ── Optional modules ───────────────────────────────────────────────────────────
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
            "show_japan_insights":     has_ja,
            "show_hindi_insights":     lang == "hi",
            "show_english_insights":   lang == "en",
            "show_bilingual_insights": lang == "mixed" and not has_ja,
            "show_code_switch":        has_ja,
            "insight_tab_label": (
                "🔍 Communication Intelligence" if has_ja else
                "💬 English Analysis"           if lang == "en" else
                "🗣️ Hindi Analysis"             if lang == "hi" else
                "🌐 Insights"
            ),
            "insight_tab_enabled": True,
        }

# ── Async job processor ────────────────────────────────────────────────────────
try:
    from api.async_processor import (
        submit_job,
        get_job_status,
        get_job_result,
        get_queue_stats,
    )
    ASYNC_PROCESSOR_AVAILABLE = True
except ImportError:
    ASYNC_PROCESSOR_AVAILABLE = False

# ── Evaluation ─────────────────────────────────────────────────────────────────
try:
    from utils.evaluator import evaluate, MLFLOW_AVAILABLE
    from utils.html_renderer import build_evaluation_html
    from utils.scenario_loader import (
        load_scenarios       as _load_scenarios,
        get_scenario_summary as _get_scenario_summary,
    )
    EVAL_AVAILABLE = True
except ImportError:
    EVAL_AVAILABLE   = False
    MLFLOW_AVAILABLE = False
    def _load_scenarios()       -> list: return []
    def _get_scenario_summary() -> dict: return {"total": 0, "by_lang": {}, "by_outcome": {}, "by_risk": {}}

# ── RAG evaluation ─────────────────────────────────────────────────────────────
try:
    from utils.rag_evaluator import evaluate_rag
    RAG_EVAL_AVAILABLE = True
except ImportError:
    RAG_EVAL_AVAILABLE = False

AUDIO_EXT = {".mp3", ".wav", ".m4a", ".mp4", ".ogg", ".webm"}
TEXT_EXT  = {".txt", ".vtt", ".json"}


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    if ASYNC_PROCESSOR_AVAILABLE:
        from api.async_processor import startup as _async_startup
        _async_startup()
    yield
    if ASYNC_PROCESSOR_AVAILABLE:
        from api.async_processor import shutdown as _async_shutdown
        _async_shutdown()


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="TranscriptAI",
    version="3.5",
    docs_url="/docs",
    lifespan=lifespan,
)

_ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "https://kunalthebeast-transcriptai.hf.space,http://localhost:7860,http://127.0.0.1:7860"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Rate limiter ───────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

_oauth = _setup_auth(app)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ── Speaker label detection ────────────────────────────────────────────────────
import re as _re

_SPEAKER_PATTERNS = [
    r"^\*?\*?[\w\s\u3000-\u9fff]{1,40}\*?\*?\s*[：:]\s*\S",
    r"^\[[\w\s\u3000-\u9fff]{1,40}\]\s*[：:]\s*\S",
    r"^【[\w\s\u3000-\u9fff]{1,40}】\s*[：:]\s*\S",
]

def _has_speaker_labels(text: str) -> bool:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    hits  = sum(
        1 for line in lines[:40]
        if any(_re.match(p, line) for p in _SPEAKER_PATTERNS)
    )
    return hits >= 2

def _strip_markdown_bold(text: str) -> str:
    cleaned = []
    for line in text.split("\n"):
        stripped = line
        if stripped.startswith("**"):
            stripped = stripped[2:]
        stripped = stripped.replace("**", "")
        cleaned.append(stripped)
    return "\n".join(cleaned)

def _ensure_speaker_labels(text: str):
    text = _strip_markdown_bold(text)
    if _has_speaker_labels(text):
        return text, False
    paragraphs = [p.strip() for p in _re.split(r"\n{2,}", text) if p.strip()]
    if not paragraphs:
        paragraphs = [l.strip() for l in text.split("\n") if l.strip()]
    labeled = [f"Speaker {i+1}: {p}" for i, p in enumerate(paragraphs)]
    return "\n".join(labeled), True


# ── Cache stats ────────────────────────────────────────────────────────────────
def _get_cache_stats(user_id: str | None = None) -> dict | None:
    try:
        from utils.vector_cache import get_cache_stats
        vc = get_cache_stats(user_id=user_id)
        return vc if vc.get("available") else None
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Pages
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {
        "cache_stats":  _get_cache_stats(get_current_user(request)),
        "current_user": get_current_user(request),
        "auth_enabled": AUTH_ENABLED,
    })


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    if get_current_user(request):
        return RedirectResponse("/")
    return templates.TemplateResponse(request, "login.html", {
        "request":      request,
        "auth_enabled": AUTH_ENABLED,
        "current_user": get_current_user(request),
        "error":        request.query_params.get("error", ""),
        "mode":         request.query_params.get("mode", "login"),
    })


@app.post("/login", response_class=HTMLResponse)
async def login_submit(
    request:  Request,
    email:    str = Form(default=""),
    password: str = Form(default=""),
    name:     str = Form(default=""),
    mode:     str = Form(default="login"),
):
    clean_email = email.strip().lower()

    if not clean_email or "@" not in clean_email:
        return templates.TemplateResponse(request, "login.html", {
            "request": request, "auth_enabled": AUTH_ENABLED,
            "current_user": None, "mode": mode,
            "error": "Please provide a valid email address.",
        })
    if not password or len(password) < 6:
        return templates.TemplateResponse(request, "login.html", {
            "request": request, "auth_enabled": AUTH_ENABLED,
            "current_user": None, "mode": mode,
            "error": "Password must be at least 6 characters.",
        })

    user_id = "usr_" + clean_email.replace("@", "_at_").replace(".", "_")

    if mode == "signup":
        if not _FIREBASE_AVAILABLE:
            return templates.TemplateResponse(request, "login.html", {
                "request": request, "auth_enabled": AUTH_ENABLED,
                "current_user": None, "mode": mode,
                "error": "Account storage unavailable. Try Google sign-in instead.",
            })
        existing = await _get_user_by_email(clean_email)
        if existing:
            return templates.TemplateResponse(request, "login.html", {
                "request": request, "auth_enabled": AUTH_ENABLED,
                "current_user": None, "mode": "login",
                "error": "An account with this email already exists. Please sign in.",
            })
        display_name = name.strip() or clean_email.split("@")[0].replace(".", " ").title()
        pw_hash = _bcrypt.hashpw(password.encode(), _bcrypt.gensalt()).decode()
        await _upsert_user(user_id=user_id, email=clean_email,
                           name=display_name, password_hash=pw_hash)
    else:
        if _FIREBASE_AVAILABLE:
            user = await _get_user_by_email(clean_email)
            if not user:
                return templates.TemplateResponse(request, "login.html", {
                    "request": request, "auth_enabled": AUTH_ENABLED,
                    "current_user": None, "mode": mode,
                    "error": "No account found with this email. Please sign up first.",
                })
            stored_hash = user.get("password_hash", "")
            if not stored_hash or not _bcrypt.checkpw(
                    password.encode(), stored_hash.encode()):
                return templates.TemplateResponse(request, "login.html", {
                    "request": request, "auth_enabled": AUTH_ENABLED,
                    "current_user": None, "mode": mode,
                    "error": "Incorrect password.",
                })
            display_name = user.get("name", clean_email.split("@")[0].title())
            user_id      = user.get("user_id", user_id)
        else:
            display_name = clean_email.split("@")[0].title()

    try:
        request.session["user_id"] = user_id
        request.session["email"]   = clean_email
        request.session["name"]    = display_name
    except Exception as exc:
        print(f"[AUTH] Session write failed: {exc}", flush=True)

    return RedirectResponse("/", status_code=303)


# ── Auth routes ────────────────────────────────────────────────────────────────
@app.get("/auth/login")
async def auth_login(request: Request):
    if not AUTH_ENABLED or not _oauth:
        return JSONResponse({"error": "Auth not enabled — set AUTH_ENABLED=1"}, 400)
    redirect_uri = request.url_for("auth_callback")
    return await _oauth.google.authorize_redirect(request, redirect_uri)


@app.get("/auth/callback", name="auth_callback")
async def auth_callback(request: Request):
    if not AUTH_ENABLED or not _oauth:
        return RedirectResponse("/")
    try:
        token     = await _oauth.google.authorize_access_token(request)
        user_info = token["userinfo"]
        request.session["user_id"] = user_info["sub"]
        request.session["email"]   = user_info.get("email", "")
        request.session["name"]    = user_info.get("name", "")
        await _upsert_user(
            user_id=user_info["sub"],
            email=user_info.get("email", ""),
            name=user_info.get("name", ""),
        )
    except Exception as exc:
        print(f"[AUTH] callback error: {exc}", flush=True)
        return RedirectResponse("/?error=1")
    return RedirectResponse("/")


@app.get("/auth/logout")
async def auth_logout(request: Request):
    try:
        request.session.clear()
    except Exception:
        pass
    return RedirectResponse("/")


@app.get("/auth/me")
async def auth_me(request: Request):
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
        "pptx_available":              PPTX_AVAILABLE,
        "gijiroku_available":          GIJIROKU_AVAILABLE,
        "cultural_insights_available": CULTURAL_INSIGHTS_AVAILABLE,
        "cache_stats":                 _get_cache_stats(get_current_user(request)),
        "current_user":                get_current_user(request),
        "auth_enabled":                AUTH_ENABLED,
    })


@app.get("/evaluate", response_class=HTMLResponse)
async def evaluate_page(request: Request):
    summary = _get_scenario_summary() if EVAL_AVAILABLE else {
        "total": 0, "by_lang": {}, "by_outcome": {}, "by_risk": {}
    }
    return templates.TemplateResponse(request, "evaluate.html", {
        "eval_available":    EVAL_AVAILABLE,
        "test_case_count":   summary["total"],
        "scenario_summary":  summary,
        "mlflow_available":  MLFLOW_AVAILABLE,
        "cache_stats":       _get_cache_stats(get_current_user(request)),
        "current_user":      get_current_user(request),
        "auth_enabled":      AUTH_ENABLED,
    })


@app.post("/evaluate/run", response_class=HTMLResponse)
@limiter.limit("3/minute")
async def evaluate_run(request: Request):
    if not EVAL_AVAILABLE:
        return HTMLResponse(content=_err("Evaluation module unavailable."), status_code=500)

    scenarios = _load_scenarios()
    if not scenarios:
        return HTMLResponse(
            content=_err(
                "No scenario files found in tests/scenarios/. "
                "Add *.json files with id, language, transcript, expected keys."
            ),
            status_code=400,
        )

    user_id = get_current_user(request)

    async def _run_one(sc: dict) -> dict:
        transcript = sc.get("transcript", "")
        language   = sc.get("language", "en")
        expected   = sc.get("expected", {})

        # FIX 4: removed bypass_cache=True — analyze_transcript doesn't accept
        # that kwarg in its current signature, causing TypeError on every eval run.
        # Per-user cache invalidation is handled inside the analyzer via user_id.
        prediction = await asyncio.to_thread(
            analyze_transcript, transcript, language,
            user_id=user_id,
        )

        report = await asyncio.to_thread(
            evaluate, prediction, expected, transcript,
            tc_name=sc["id"],
            provider=prediction.get("_provider", "unknown"),
        )

        if RAG_EVAL_AVAILABLE:
            try:
                rag_report = await asyncio.to_thread(
                    evaluate_rag,
                    transcript, prediction, None, False,
                    sc["id"], MLFLOW_AVAILABLE,
                )
                report["rag_metrics"] = rag_report
            except Exception as exc:
                report["rag_metrics"] = {"error": str(exc)}

        return {
            "tc_id":       sc["id"],
            "tc_name":     sc.get("id", ""),
            "language":    language,
            "provider":    prediction.get("_provider", "unknown"),
            "duration_ms": prediction.get("_duration_ms", 0),
            "expected":    expected,
            "actual": {
                "deal_outcome": prediction.get("deal_outcome"),
                "risk_level":   prediction.get("risk_level"),
            },
            "report": report,
        }

    try:
        reports = await asyncio.gather(*[_run_one(sc) for sc in scenarios])
    except Exception as exc:
        return HTMLResponse(content=_err(f"Evaluation failed: {exc}"), status_code=500)

    html = build_evaluation_html(list(reports), MLFLOW_AVAILABLE)
    return HTMLResponse(content=html)


# ══════════════════════════════════════════════════════════════════════════════
# Job queue routes
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/jobs/submit")
@limiter.limit("10/minute")
async def jobs_submit(
    request:    Request,
    transcript: str           = Form(...),
    language:   Optional[str] = Form(None),
):
    """
    Submit a transcript for async analysis.
    Returns job_id immediately — non-blocking.
    Poll /jobs/{job_id} until status == "done".
    """
    if not ASYNC_PROCESSOR_AVAILABLE:
        return JSONResponse({"error": "Async processor not available"}, status_code=503)
    if len(transcript.strip()) < 20:
        return JSONResponse({"error": "Transcript too short (min 20 chars)"}, status_code=400)

    cleaned       = clean_text(transcript)
    detected_lang = language or detect_language(cleaned)
    cleaned, _    = _ensure_speaker_labels(cleaned)

    user_id = get_current_user(request)

    try:
        job_id = submit_job(cleaned, detected_lang, user_id=user_id)
    except RuntimeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=503)

    return JSONResponse({
        "job_id":   job_id,
        "status":   "queued",
        "poll_url": f"/jobs/{job_id}",
    })


@app.get("/jobs/{job_id}")
async def jobs_status(job_id: str, request: Request):
    """
    Poll job status and retrieve result when done.

    Cross-user reads return 404 — job existence is not revealed to other users.

    Returns:
        status: "queued" | "running" | "done" | "failed"
        result: analysis dict (only when status == "done")
        error:  message (only when status == "failed")
    """
    if not ASYNC_PROCESSOR_AVAILABLE:
        return JSONResponse({"error": "Async processor not available"}, status_code=503)

    user_id = get_current_user(request)
    status  = get_job_status(job_id, requesting_user=user_id)
    if "error" in status:
        return JSONResponse(status, status_code=404)

    response: dict = {
        "job_id":      job_id,
        "status":      status["status"],
        "duration_ms": status.get("duration_ms"),
    }

    if status["status"] == "done":
        try:
            result = get_job_result(
                job_id, timeout_sec=0, requesting_user=user_id
            )

            if SOFT_REJECTION_AVAILABLE:
                result["soft_rejections"] = detect_soft_rejections(
                    result.get("_cleaned_transcript", "")
                )

            detected_lang = result.get("_detected_language", "en")
            features      = get_features(detected_lang)
            pii_report    = result.get("_pii_report")

            html = build_results_html(result, detected_lang, features, pii_report)
            response["result"] = result
            response["html"]   = html
        except Exception as exc:
            response["status"] = "failed"
            response["error"]  = str(exc)

    elif status["status"] == "failed":
        response["error"] = status.get("error", "Unknown error")

    return JSONResponse(response)


@app.get("/jobs/queue/stats")
async def jobs_queue_stats():
    """Returns current job queue statistics."""
    if not ASYNC_PROCESSOR_AVAILABLE:
        return JSONResponse({"error": "Async processor not available"}, status_code=503)
    return JSONResponse(get_queue_stats())


# ══════════════════════════════════════════════════════════════════════════════
# /transcribe
# ══════════════════════════════════════════════════════════════════════════════

class _FileShim:
    def __init__(self, filename: str, data: bytes):
        self.name  = filename
        self._data = data
    def getvalue(self): return self._data
    def read(self):     return self._data


@app.post("/transcribe")
@limiter.limit("5/minute")
async def transcribe(request: Request, file: UploadFile = File(...)):
    filename = file.filename or ""
    ext      = Path(filename).suffix.lower()
    content  = await file.read()

    if ext in AUDIO_EXT:
        if not AUDIO_AVAILABLE:
            return JSONResponse({"success": False, "error": "Audio transcription module unavailable."})
        size_mb = len(content) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            return JSONResponse({
                "success": False,
                "error":   f"File too large ({size_mb:.1f} MB). Max: {MAX_FILE_SIZE_MB} MB",
            })
        try:
            res = await asyncio.to_thread(transcribe_audio, content, filename)
        except Exception as exc:
            return JSONResponse({"success": False, "error": str(exc)})
        if not res.get("success"):
            return JSONResponse({"success": False, "error": res.get("error", "Transcription failed")})
        seg  = format_transcript_with_timestamps(res.get("segments", []))
        text = seg or res.get("text", "")
        return JSONResponse({
            "success": True,
            "transcript": text,
            "meta": {
                "duration": res.get("duration", 0),
                "language": res.get("language", "?"),
                "provider": res.get("provider", ""),
            },
        })

    if ext in TEXT_EXT:
        try:
            shim   = _FileShim(filename, content)
            parsed = parse_uploaded_file(shim)
            return JSONResponse({"success": True, "transcript": parsed,
                                 "meta": {"chars": len(parsed)}})
        except Exception as exc:
            return JSONResponse({"success": False, "error": str(exc)})

    return JSONResponse({"success": False, "error": f"Unsupported file type: {ext}"})


# ══════════════════════════════════════════════════════════════════════════════
# /analyze-text
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/analyze-text", response_class=HTMLResponse)
@limiter.limit("3/minute")
async def analyze_text_route(
    request:    Request,
    transcript: str           = Form(...),
    language:   Optional[str] = Form(None),
):
    if len(transcript.strip()) < 20:
        return HTMLResponse(content=_err("Transcript too short (min 20 chars)."), status_code=400)
    try:
        cleaned       = clean_text(transcript)
        detected_lang = language or detect_language(cleaned)
        cleaned, was_unlabeled = _ensure_speaker_labels(cleaned)

        result = await asyncio.to_thread(
            analyze_transcript, cleaned, detected_lang,
            user_id=get_current_user(request),
        )

        if SOFT_REJECTION_AVAILABLE:
            result["soft_rejections"] = detect_soft_rejections(cleaned)

        result["_detected_language"]    = detected_lang
        result["_unlabeled_transcript"] = was_unlabeled

        features   = get_features(detected_lang)
        pii_report = result.get("_pii_report", None)

        html = build_results_html(result, detected_lang, features, pii_report)
        # FIX: use <script type="application/json"> instead of <div>.
        # A <div> breaks when the result JSON contains </div> (e.g. in summaries
        # of web meetings or transcripts quoting HTML). The browser's HTML parser
        # closes the div early, truncating the JSON so JSON.parse() throws inside
        # captureResult's silent catch — lastResult stays null, export page shows
        # "run analysis first" even though results rendered fine.
        # <script type="application/json"> is not parsed as HTML so </div> is safe;
        # only </script> needs escaping, which is extremely rare in meeting content.
        _rjson = _json.dumps(result, ensure_ascii=False).replace('</script>', r'<\/script>')
        tag    = f'<script id="tai-result-data" type="application/json">{_rjson}</script>'
        return HTMLResponse(content=html + tag)

    except Exception as exc:
        print(f"[ANALYZE ERROR] {exc}", flush=True)
        import traceback; traceback.print_exc()
        return HTMLResponse(content=_err(str(exc)), status_code=500)


# ══════════════════════════════════════════════════════════════════════════════
# Export routes
# ══════════════════════════════════════════════════════════════════════════════

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
    body  = await request.json()
    r     = body.get("result", body)
    lines = ["# Meeting Analysis\n"]
    if r.get("full_summary"): lines += ["## Overview\n", r["full_summary"], "\n"]
    if r.get("summary"):      lines += ["## Key Points\n"] + [f"- {b}\n" for b in r["summary"]]
    if r.get("action_items"):
        lines += ["\n## Action Items\n"]
        for i in r["action_items"]:
            flag = " ⚠" if i.get("hallucination_flag") else ""
            lines.append(
                f"- **{i.get('task','')}**{flag}  \n"
                f"  Owner: {i.get('owner','TBD')}  Deadline: {i.get('deadline','TBD')}\n"
            )
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
    body  = await request.json()
    r     = body.get("result", body)
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
        lines += [f"- {s.get('speaker','')}: {s.get('score','').upper()}"
                  for s in r["sentiment"]]
        lines.append("")
    if r.get("speakers"):
        lines += ["SPEAKERS", "-" * 20]
        for spk in r["speakers"]:
            lines.append(
                f"- {spk.get('name','')}: {spk.get('talk_time_pct',0)}% ({spk.get('tone','')})"
            )
        lines.append("")
    txt = "\n".join(lines)
    return StreamingResponse(
        io.BytesIO(txt.encode("utf-8")), media_type="text/plain",
        headers={"Content-Disposition": "attachment; filename=meeting_notes.txt"},
    )


# ══════════════════════════════════════════════════════════════════════════════
# Health / SEO
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    scenario_count = _get_scenario_summary().get("total", 0) if EVAL_AVAILABLE else 0
    return {
        "status":           "healthy",
        "version":          "3.5",
        "provider":         os.getenv("TRANSCRIPT_AI_PROVIDER", "auto"),
        "nim_configured":   bool(os.getenv("NIM_API_KEY")),
        "groq_configured":  bool(os.getenv("GROQ_API_KEY")),
        "pii_masking":      PII_AVAILABLE,
        "auth_enabled":     AUTH_ENABLED,
        "firebase":         _FIREBASE_AVAILABLE,
        "scenario_count":   scenario_count,
        "modules": {
            "audio":             AUDIO_AVAILABLE,
            "pii_masker":        PII_AVAILABLE,
            "soft_rejection":    SOFT_REJECTION_AVAILABLE,
            "hallucination":     HALLUCINATION_GUARD_AVAILABLE,
            "pptx":              PPTX_AVAILABLE,
            "gijiroku":          GIJIROKU_AVAILABLE,
            "cultural_insights": CULTURAL_INSIGHTS_AVAILABLE,
            "slide_architect":   SLIDE_ARCHITECT_AVAILABLE,
            "language_intel":    LANGUAGE_INTEL_AVAILABLE,
            "evaluation":        EVAL_AVAILABLE,
            "rag_evaluation":    RAG_EVAL_AVAILABLE,
            "async_processor":   ASYNC_PROCESSOR_AVAILABLE,
        },
    }


@app.get("/robots.txt", response_class=HTMLResponse)
async def robots_txt():
    # FIX 9: guard open() — missing file was raising FileNotFoundError → HTTP 500
    try:
        content = open("robots.txt").read()
    except FileNotFoundError:
        content = "User-agent: *\nAllow: /\n"
    return HTMLResponse(content, media_type="text/plain")


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


@app.get("/manifest.json")
async def manifest():
    return JSONResponse({
        "name":             "TranscriptAI",
        "short_name":       "TranscriptAI",
        "description":      "Japanese & Multilingual Meeting Intelligence AI.",
        "start_url":        "/",
        "display":          "standalone",
        "background_color": "#FDF8F5",
        "theme_color":      "#D96080",
        "categories":       ["business", "productivity"],
        "lang":             "en",
    })


# ── Helpers ────────────────────────────────────────────────────────────────────
def _err(msg: str) -> str:
    # FIX 7: HTML-escape msg before embedding.
    # Raw exception strings can contain angle brackets from tracebacks or
    # user-controlled transcript content, which previously rendered as live HTML (XSS).
    safe_msg = _html.escape(str(msg))
    return (
        f'<div style="background:var(--red-bg);border-left:3px solid var(--red);'
        f'border-radius:0 10px 10px 0;padding:14px 18px;color:#3C2416;margin-top:12px">'
        f'<b style="color:var(--red)">⚠ Error</b><br>{safe_msg}</div>'
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 7860)),
        reload=os.getenv("ENV") == "development",
        workers=1,
    )