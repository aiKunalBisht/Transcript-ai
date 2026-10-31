# main.py — TranscriptAI v3.8
# FastAPI server. Run: uvicorn main:app --reload --port 7860
#
# v3.7 fixes over v3.6:
#
#   FIX-E — Logging never initialised.
#     Every logger.info/warning/error call throughout the codebase produced
#     nothing in production because no handler was ever registered. Replaced
#     print(stderr) with logger calls in v9.0 but forgot to configure the
#     logging system. _configure_logging() called at module startup now:
#       - StreamHandler to stdout with timestamp + level + name + message
#       - INFO in production, DEBUG in development/local environments
#       - Noisy third-party loggers (chromadb, httpx, mlflow) quieted to WARNING
#       - force=True ensures configuration even when uvicorn pre-configures root
#
#   FIX-A — asyncio.to_thread(analyze_transcript, ...) replaced with
#     await analyze_transcript(...) in /analyze-text and /evaluate/run.
#     analyze_transcript() is now async def (analyzer.py v9.0). Passing an
#     async def to asyncio.to_thread does not await it — it returns a coroutine
#     object. The first dict access on the "result" then raises:
#       TypeError: 'coroutine' object is not subscriptable
#     Both /analyze-text and /evaluate/run were completely broken with v3.6
#     against analyzer.py v9.0. Direct await is the correct call pattern.
#
#   FIX-B — Rate limiter key function replaced with _get_real_ip().
#     get_remote_address() returns the proxy IP on Hugging Face Spaces, not the
#     client IP. Every request from every user appears to come from the same
#     address. The 3/minute limit therefore applied globally: the third request
#     from any user triggered 429 for all users for a full minute.
#     _get_real_ip() reads X-Forwarded-For first, falling back to the socket IP.
#     The first entry in X-Forwarded-For is the original client, which is correct
#     for single-proxy deployments (HF Spaces). For multi-hop proxies, adjust
#     the index.
#
#   FIX-C — export_pptx no longer imports _get_groq_key from analyzer.py.
#     Importing a private function from another module as a production dependency
#     fails silently at runtime if the function is renamed or refactored.
#     The API key is read directly from the environment, which is the canonical
#     source and does not couple this route to analyzer internals.
#
#   FIX-D — _last_result_cache documented and guarded for single-worker use.
#     The cache is an in-memory dict. With multiple gunicorn workers each worker
#     holds its own dict: a user whose analysis ran on worker 2 calls
#     /api/last-result and hits worker 3, receiving 404. A startup guard now
#     logs a warning when WORKERS > 1 so the problem is visible in logs.
#     Production fix: replace with Redis (see comment at _last_result_cache).
#
# v3.6 fixes retained unchanged:
#   SEC-1 — cache keyed by user_id only (no IP fallback for anonymous).
#   SEC-2 — /api/last-result requires authentication.
#   FIX 3 — _setup_auth safe when SESSION_SECRET missing + AUTH_ENABLED=0.
#   FIX 4 — evaluate_run no longer passes bypass_cache=True.
#   FIX 7 — _err() HTML-escapes exception messages (XSS fix).
#   FIX 9 — robots_txt wraps open() in try/except.

import asyncio
import html as _html
import io
import json as _json
import logging
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


def _configure_logging() -> None:
    """
    FIX-E: Initialise the logging system at startup.

    Previously every logger.info/warning/error call in the entire codebase
    (analyzer.py, main.py, pii_masker.py, sentiment_engine.py) produced
    nothing because no handler was ever registered. basicConfig() was never
    called, so the root logger had no handlers, and all records were silently
    discarded.

    Called at module level so it fires before any import-time logger calls.
    force=True overrides any partial configuration uvicorn may have applied.
    """
    _env       = os.getenv("ENVIRONMENT", "production").lower()
    _log_level = (
        logging.DEBUG if _env in ("development", "local", "dev")
        else logging.INFO
    )
    logging.basicConfig(
        level   = _log_level,
        format  = "%(asctime)s | %(levelname)-8s | %(name)-35s | %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
        stream  = __import__("sys").stdout,
        force   = True,   # override any pre-existing root handler config
    )
    # Quiet noisy third-party loggers that spam at INFO/DEBUG
    for _noisy in (
        "uvicorn.access",
        "httpx", "httpcore",
        "chromadb", "chromadb.telemetry",
        "mlflow", "mlflow.tracking",
        "urllib3", "asyncio",
    ):
        logging.getLogger(_noisy).setLevel(logging.WARNING)


_configure_logging()   # FIX-E: must run before any module-level logger calls

logger = logging.getLogger("transcriptai.main")

# ── Rate limiting ──────────────────────────────────────────────────────────────
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded


# FIX-B: Proper client IP resolution for reverse-proxy deployments.
# On Hugging Face Spaces (and most PaaS platforms), requests arrive through a
# reverse proxy. get_remote_address() returns the proxy's IP, not the client's.
# X-Forwarded-For contains a comma-separated list; the first entry is the
# original client IP added by the outermost proxy.
# For multi-hop proxy chains where you control only the innermost hop, use
# index -1 (last entry you can trust). For HF Spaces single-proxy, index 0
# is correct.
def _get_real_ip(request: Request) -> str:
    """
    FIX-B: Return the real client IP, respecting X-Forwarded-For.
    Falls back to socket address when the header is absent (direct connections,
    local dev). DSA: O(1) header lookup + O(k) split where k = proxy hops.
    """
    forwarded_for = request.headers.get("X-Forwarded-For", "").strip()
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    cf_ip = request.headers.get("CF-Connecting-IP", "").strip()
    if cf_ip:
        return cf_ip
    return get_remote_address(request)


# ── Optional Google OAuth ──────────────────────────────────────────────────────
AUTH_ENABLED    = os.getenv("AUTH_ENABLED", "0") == "1"
_AUTH_AVAILABLE = False

try:
    from authlib.integrations.starlette_client import OAuth
    _AUTH_AVAILABLE = True
except ImportError:
    if AUTH_ENABLED:
        logger.warning(
            "[TRANSCRIPT_AI] authlib not installed — Google auth disabled. "
            "pip install authlib itsdangerous"
        )
    AUTH_ENABLED = False


def _setup_auth(app):
    secret = os.getenv("SESSION_SECRET")
    if not secret:
        if AUTH_ENABLED:
            raise RuntimeError(
                "SESSION_SECRET env variable is not set. "
                "Add it to your HF Space secrets (Settings → Variables and secrets). "
                "Use any random string of 32+ characters."
            )
        return None

    try:
        from starlette.middleware.sessions import SessionMiddleware
        app.add_middleware(
            SessionMiddleware, secret_key=secret, max_age=60 * 60 * 24 * 30
        )
    except Exception as e:
        logger.error(f"[TRANSCRIPT_AI] SessionMiddleware error: {e}")

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


# FIX-A: analyze_transcript is now async def (analyzer.py v9.0).
# Import it directly — it is awaited, never passed to asyncio.to_thread.
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
    logger.warning("[TRANSCRIPT_AI] firebase_client not found — user storage disabled.")


async def _upsert_user(
    user_id: str, email: str, name: str, password_hash: str = ""
) -> None:
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
    AUDIO_AVAILABLE  = False
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
    HINDI_NLP_AVAILABLE      = False
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
    def _get_scenario_summary() -> dict: return {
        "total": 0, "by_lang": {}, "by_outcome": {}, "by_risk": {}
    }

# ── RAG evaluation ─────────────────────────────────────────────────────────────
try:
    from utils.rag_evaluator import evaluate_rag
    RAG_EVAL_AVAILABLE = True
except ImportError:
    RAG_EVAL_AVAILABLE = False

AUDIO_EXT = {".mp3", ".wav", ".m4a", ".mp4", ".ogg", ".webm"}
TEXT_EXT  = {".txt", ".vtt", ".json"}

# ── Server-side last-result cache ──────────────────────────────────────────────
# FIX-D: This is an in-memory dict — it is NOT safe for multi-worker deployments.
# With gunicorn -w 4, each worker has its own copy. A user whose analysis ran on
# worker 2 will hit a 404 on /api/last-result if the request goes to worker 3.
#
# For single-worker deployments (HF Spaces default, uvicorn without -w):
# this works correctly.
#
# For multi-worker production: replace with Redis.
#   import redis.asyncio as redis
#   _redis = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
#   await _redis.set(f"last_result:{user_id}", json.dumps(result), ex=3600)
#   raw = await _redis.get(f"last_result:{user_id}")
#
# SEC-FIX v3.6: keyed by user_id only — never by IP. Anonymous users rely on
# the <script id="tai-result-data"> tag embedded in the /analyze-text response.
_last_result_cache: dict[str, dict] = {}

_WORKER_COUNT = int(os.getenv("WEB_CONCURRENCY", os.getenv("WORKERS", "1")))


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # FIX-D: Warn at startup when running multi-worker without Redis.
    if _WORKER_COUNT > 1:
        logger.warning(
            f"[TRANSCRIPT_AI] Running with {_WORKER_COUNT} workers and in-memory "
            f"_last_result_cache. /api/last-result will return 404 for ~{(_WORKER_COUNT-1)/_WORKER_COUNT*100:.0f}% "
            f"of authenticated export requests. Set REDIS_URL and replace the cache."
        )
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
    version="3.8",
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

# FIX-B: Use _get_real_ip so the rate limiter keys by actual client IP,
# not the shared HF Spaces / reverse-proxy IP.
limiter = Limiter(key_func=_get_real_ip)
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
        display_name = (
            name.strip() or
            clean_email.split("@")[0].replace(".", " ").title()
        )
        pw_hash = _bcrypt.hashpw(password.encode(), _bcrypt.gensalt()).decode()
        await _upsert_user(
            user_id=user_id, email=clean_email,
            name=display_name, password_hash=pw_hash,
        )
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
                password.encode(), stored_hash.encode()
            ):
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
        logger.error(f"[AUTH] Session write failed: {exc}")

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
        logger.error(f"[AUTH] callback error: {exc}")
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
    _cache_key   = get_current_user(request)
    _last_result = _last_result_cache.get(_cache_key) if _cache_key else None
    _result_json = (
        _json.dumps(_last_result, ensure_ascii=False).replace(
            "</script>", r"<\/script>"
        )
        if _last_result else "null"
    )
    return templates.TemplateResponse(request, "export.html", {
        "pptx_available":              PPTX_AVAILABLE,
        "gijiroku_available":          GIJIROKU_AVAILABLE,
        "cultural_insights_available": CULTURAL_INSIGHTS_AVAILABLE,
        "cache_stats":                 _get_cache_stats(get_current_user(request)),
        "current_user":                get_current_user(request),
        "auth_enabled":                AUTH_ENABLED,
        "server_result_json":          _result_json,
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
        return HTMLResponse(
            content=_err("Evaluation module unavailable."), status_code=500
        )

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

        # FIX-A: analyze_transcript is async def — await it directly.
        # asyncio.to_thread would run the coroutine without awaiting it.
        prediction = await analyze_transcript(
            transcript, language, user_id=user_id
        )

        # evaluate() and evaluate_rag() remain synchronous — to_thread is correct.
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
        return HTMLResponse(
            content=_err(f"Evaluation failed: {exc}"), status_code=500
        )

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
    if not ASYNC_PROCESSOR_AVAILABLE:
        return JSONResponse({"error": "Async processor not available"}, status_code=503)
    if len(transcript.strip()) < 20:
        return JSONResponse(
            {"error": "Transcript too short (min 20 chars)"}, status_code=400
        )

    cleaned       = clean_text(transcript)
    detected_lang = language or detect_language(cleaned)
    cleaned, _    = _ensure_speaker_labels(cleaned)
    user_id       = get_current_user(request)

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
            html          = build_results_html(result, detected_lang, features, pii_report)
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
            return JSONResponse({
                "success": False,
                "error":   "Audio transcription module unavailable.",
            })
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
            return JSONResponse({
                "success": False,
                "error":   res.get("error", "Transcription failed"),
            })
        seg  = format_transcript_with_timestamps(res.get("segments", []))
        text = seg or res.get("text", "")
        return JSONResponse({
            "success":    True,
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
            return JSONResponse({
                "success": True, "transcript": parsed,
                "meta": {"chars": len(parsed)},
            })
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
        return HTMLResponse(
            content=_err("Transcript too short (min 20 chars)."), status_code=400
        )
    try:
        cleaned       = clean_text(transcript)
        detected_lang = language or detect_language(cleaned)
        cleaned, was_unlabeled = _ensure_speaker_labels(cleaned)

        # FIX-A: analyze_transcript is now async def (analyzer.py v9.0).
        # Direct await — do NOT wrap in asyncio.to_thread.
        # asyncio.to_thread(async_fn, ...) does not await the coroutine;
        # it returns a coroutine object, making all subsequent dict access fail.
        result = await analyze_transcript(
            cleaned,
            detected_lang,
            user_id=get_current_user(request),
        )

        if SOFT_REJECTION_AVAILABLE:
            result["soft_rejections"] = detect_soft_rejections(cleaned)

        result["_detected_language"]    = detected_lang
        result["_unlabeled_transcript"] = was_unlabeled
        result.setdefault(
            "has_japanese_content",
            bool(_re.search(r"[\u3040-\u9fff\u4e00-\u9fff]", cleaned)),
        )

        features   = get_features(detected_lang)
        pii_report = result.get("_pii_report", None)

        # SEC-FIX v3.6 (retained): cache only for authenticated users.
        # FIX-D: in-memory dict is single-worker only — see note at declaration.
        _cache_key = get_current_user(request)
        if _cache_key:
            _last_result_cache[_cache_key] = result

        html   = build_results_html(result, detected_lang, features, pii_report)
        _rjson = _json.dumps(result, ensure_ascii=False).replace(
            "</script>", r"<\/script>"
        )
        tag = (
            f'<script id="tai-result-data" type="application/json">'
            f'{_rjson}</script>'
        )
        return HTMLResponse(content=html + tag)

    except Exception as exc:
        logger.error(f"[ANALYZE ERROR] {exc}", exc_info=True)
        return HTMLResponse(content=_err(str(exc)), status_code=500)


# ── Server-side last-result retrieval ─────────────────────────────────────────
@app.get("/api/last-result")
async def api_last_result(request: Request):
    """
    Returns the most recent analysis result for the authenticated user.
    SEC-FIX v3.6: Requires authentication.
    FIX-D: In-memory — unreliable with multiple workers. See _last_result_cache.
    """
    cache_key = get_current_user(request)
    if not cache_key:
        return JSONResponse(
            {"error": "Authentication required to retrieve cached results."},
            status_code=401,
        )
    result = _last_result_cache.get(cache_key)
    if not result:
        return JSONResponse(
            {"error": "No result cached — run an analysis first."},
            status_code=404,
        )
    return JSONResponse(result)


# ══════════════════════════════════════════════════════════════════════════════
# Export routes
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/export/pptx")
async def export_pptx(request: Request):
    if not PPTX_AVAILABLE or not SLIDE_ARCHITECT_AVAILABLE:
        raise HTTPException(503, "PPTX builder not available")
    body   = await request.json()
    result = body.get("result", body)

    # FIX-C: Do not import _get_groq_key from analyzer.py.
    # _get_groq_key is a private function — coupling a route to it makes the
    # export endpoint break silently if analyzer internals are refactored.
    # The API key is read directly from the environment, which is the source
    # of truth for all modules.
    api_key = (
        os.getenv("NIM_API_KEY", "").strip()
        or os.getenv("GROQ_API_KEY", "").strip()
        or os.getenv("GROQ_API_KEY_2", "").strip()
    )
    agent      = SlideArchitectAgent(groq_api_key=api_key)
    plan       = await asyncio.to_thread(
        agent.plan, result, result.get("_detected_language", "en")
    )
    pptx_bytes = await asyncio.to_thread(build_pptx, plan)
    return StreamingResponse(
        io.BytesIO(pptx_bytes),
        media_type=(
            "application/vnd.openxmlformats-officedocument"
            ".presentationml.presentation"
        ),
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
    if r.get("full_summary"):
        lines += ["## Overview\n", r["full_summary"], "\n"]
    if r.get("summary"):
        lines += ["## Key Points\n"] + [f"- {b}\n" for b in r["summary"]]
    if r.get("action_items"):
        lines += ["\n## Action Items\n"]
        for i in r["action_items"]:
            flag = " ⚠" if i.get("hallucination_flag") else ""
            # Uses "task" field — aligned with current analyzer output.
            # See analysis_prompt.py fix for schema alignment (task vs task_en).
            task = i.get("task") or i.get("task_en") or ""
            lines.append(
                f"- **{task}**{flag}  \n"
                f"  Owner: {i.get('owner','TBD')}  "
                f"Deadline: {i.get('deadline','TBD')}\n"
            )
    md = "".join(lines)
    return StreamingResponse(
        io.BytesIO(md.encode("utf-8")),
        media_type="text/markdown",
        headers={"Content-Disposition": "attachment; filename=meeting_notes.md"},
    )


@app.post("/export/json")
async def export_json_route(request: Request):
    body = await request.json()
    raw  = _json.dumps(
        body.get("result", body), ensure_ascii=False, indent=2
    ).encode("utf-8")
    return StreamingResponse(
        io.BytesIO(raw),
        media_type="application/json",
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
            task = i.get("task") or i.get("task_en") or ""
            lines.append(f"- {task}{flag}")
            lines.append(
                f"    Owner: {i.get('owner','TBD')}   "
                f"Deadline: {i.get('deadline','TBD')}"
            )
        lines.append("")
    if r.get("sentiment"):
        lines += ["SENTIMENT", "-" * 20]
        lines += [
            f"- {s.get('speaker','')}: {s.get('score','').upper()}"
            for s in r["sentiment"]
        ]
        lines.append("")
    if r.get("speakers"):
        lines += ["SPEAKERS", "-" * 20]
        for spk in r["speakers"]:
            lines.append(
                f"- {spk.get('name','')}: "
                f"{spk.get('talk_time_pct',0)}% ({spk.get('tone','')})"
            )
        lines.append("")
    txt = "\n".join(lines)
    return StreamingResponse(
        io.BytesIO(txt.encode("utf-8")),
        media_type="text/plain",
        headers={"Content-Disposition": "attachment; filename=meeting_notes.txt"},
    )


# ══════════════════════════════════════════════════════════════════════════════
# Health / SEO
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    scenario_count = (
        _get_scenario_summary().get("total", 0) if EVAL_AVAILABLE else 0
    )
    return {
        "status":           "healthy",
        "version":          "3.8",
        "provider":         os.getenv("TRANSCRIPT_AI_PROVIDER", "auto"),
        "nim_configured":   bool(os.getenv("NIM_API_KEY")),
        "groq_configured":  bool(os.getenv("GROQ_API_KEY")),
        "pii_masking":      PII_AVAILABLE,
        "auth_enabled":     AUTH_ENABLED,
        "firebase":         _FIREBASE_AVAILABLE,
        "scenario_count":   scenario_count,
        "worker_count":     _WORKER_COUNT,
        "cache_mode":       "in_memory" if _WORKER_COUNT == 1 else "in_memory_UNSAFE",
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