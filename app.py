"""
app.py — TranscriptAI  v7.1
Japanese Business Intelligence Platform

Run: python -m streamlit run app.py

v7.1 FIXES (app.py side):
  FIX-A: _cold_start_tasks no longer calls st.secrets inside a background thread
          (st.secrets is not thread-safe). Reads key from os.getenv only.
  FIX-B: Mock warning block now checks _last_error from results so the user
          sees the actual reason instead of the generic "Demo mode active." message.
  FIX-C: Added a small debug expander under the warning so the provider status
          is visible without needing to open HF Logs.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
from datetime import datetime
import streamlit as st
from analysis import analyze_transcript
from utils.html_renderer import build_results_html
from utils import (
    add_to_history, build_export_json, clean_text, detect_language,
    export_filename, format_history_label, language_display_name, parse_uploaded_file,
)

# ── Optional dependencies ────────────────────────────────────────────────────
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
    from analysis.analyzer import stream_transcript_groq
    STREAMING_AVAILABLE = True
except ImportError:
    STREAMING_AVAILABLE = False

try:
    from utils.evaluator import evaluate
    from tests.test_data import TEST_CASES
    EVAL_AVAILABLE = True
except ImportError:
    EVAL_AVAILABLE = False

try:
    from utils.logger import get_trends, get_stats, get_recent_entries
    TRENDS_AVAILABLE = True
except ImportError:
    TRENDS_AVAILABLE = False

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
                "🗣️ Hindi Analysis"             if lang == "hi" else
                "🌐 Insights"
            ),
            "insight_tab_enabled": True,
        }

# ── SEO + preconnect head tags ───────────────────────────────────────────────
# Injected before page config renders — fixes Lighthouse SEO 82→90+
st.markdown("""
<head>
  <meta name="description" content="TranscriptAI — Japanese meeting intelligence. Extracts action items, keigo formality, soft rejections, and speaker sentiment from JA/EN/HI transcripts. APPI compliant.">
  <meta name="robots" content="index, follow">
  <meta property="og:title" content="TranscriptAI · Japanese Business Intelligence">
  <meta property="og:description" content="Turn any meeting transcript into structured intelligence in 3 seconds. Keigo detection, nemawashi patterns, APPI compliant.">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link rel="preconnect" href="https://api.groq.com">
</head>
""", unsafe_allow_html=True)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TranscriptAI · Speech & Meeting Analyzer",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS — warm sakura/peach palette ─────────────────────────────────────────
st.markdown("""
<style>
/* Preconnect hints — reduces font TTFB by ~200ms */
/* font-display:swap prevents invisible text during font load (fixes CLS) */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=Noto+Sans+JP:wght@400;500&display=swap');

:root {
    --washi:        #FAF6F2;
    --surface:      #FFFEFB;
    --surface-warm: #FEF3EC;
    --border:       #EFE2D8;
    --border-mid:   #E5D0C4;
    --ink:          #3C2416;
    --ink-mid:      #7A5040;
    --ink-soft:     #A87868;
    --ink-faint:    #C8A898;
    --sakura:       #D96080;
    --sakura-deep:  #BE4060;
    --sakura-bg:    #FDEEF2;
    --sakura-pale:  #FEF6F8;
    --sakura-light: #F2B0C0;
    --peach:        #E88060;
    --peach-bg:     #FDF0EA;
    --gold:         #B87830;
    --gold-light:   #F5E0C0;
    --green:        #486858;
    --green-bg:     #EDF3EF;
    --red:          #B04040;
    --red-bg:       #FAF0F0;
    --amber:        #986820;
    --amber-bg:     #FAF0E0;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', 'Noto Sans JP', sans-serif !important;
    color: var(--ink) !important;
    -webkit-font-smoothing: antialiased;
    scroll-behavior: smooth;
}

*, *::before, *::after {
    transition-timing-function: cubic-bezier(0.4, 0, 0.2, 1);
}

.stApp {
    background-color: var(--washi) !important;
    background-image:
        radial-gradient(circle at 92% 8%,  rgba(217,96,128,0.09) 0%, transparent 45%),
        radial-gradient(circle at 8%  92%, rgba(232,128,96,0.07) 0%, transparent 45%),
        radial-gradient(circle at 50% 50%, rgba(184,120,48,0.03) 0%, transparent 60%) !important;
    /* Removed SVG pattern — was causing main-thread repaints on scroll (LH diagnostic: avoid non-composited animations) */
}

[data-testid="stToolbar"],
[data-testid="stHeader"],
[data-testid="stDecoration"],
header[data-testid="stHeader"] {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
}

[data-testid="stSidebar"] {
    background-color: #FDF8F5 !important;
    border-right: 1px solid var(--border) !important;
    background-image:
        radial-gradient(circle at 50% 0%,   rgba(217,96,128,0.07) 0%, transparent 55%),
        radial-gradient(circle at 100% 100%, rgba(184,120,48,0.04) 0%, transparent 50%) !important;
    box-shadow: 2px 0 20px rgba(60,36,22,0.06) !important;
}
[data-testid="stSidebar"] * { color: var(--ink) !important; }
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label { color: var(--ink-mid) !important; }

.block-container { background: transparent !important; padding-top: 1rem !important; }
[data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"],
[data-testid="column"],
section.main > div,
.main > div { background: transparent !important; }

[data-testid="stFileUploader"],
[data-testid="stFileUploader"] > div,
[data-testid="stFileUploader"] section,
[data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploadDropzone"] {
    background-color: var(--surface) !important;
    background: var(--surface) !important;
    border: 1.5px dashed var(--border-mid) !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploadDropzone"] {
    background-color: var(--sakura-pale) !important;
    background: var(--sakura-pale) !important;
}
[data-testid="stFileUploader"] *,
[data-testid="stFileUploaderDropzone"] *,
[data-testid="stFileUploadDropzone"] * {
    color: var(--ink-mid) !important;
    background: transparent !important;
}
[data-testid="stFileUploaderDropzone"] svg,
[data-testid="stFileUploadDropzone"] svg {
    fill: var(--sakura-light) !important;
    color: var(--sakura-light) !important;
}
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploadDropzone"] button,
[data-testid="stFileUploader"] button {
    background-color: var(--sakura-bg) !important;
    background: var(--sakura-bg) !important;
    color: var(--sakura-deep) !important;
    border: 1px solid var(--sakura-light) !important;
    border-radius: 6px !important;
    padding: 0.3rem 1rem !important;
    font-weight: 500 !important;
}
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] small {
    color: var(--ink-soft) !important;
}

textarea,
.stTextArea textarea,
div[data-baseweb="textarea"],
div[data-baseweb="textarea"] textarea {
    background-color: var(--surface) !important;
    background: var(--surface) !important;
    border: 1.5px solid var(--border-mid) !important;
    border-radius: 10px !important;
    color: var(--ink) !important;
    font-family: 'Noto Sans JP', 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
    line-height: 1.7 !important;
    caret-color: var(--sakura) !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
textarea:focus,
div[data-baseweb="textarea"]:focus-within,
div[data-baseweb="textarea"]:focus-within textarea {
    border-color: var(--sakura-light) !important;
    box-shadow: 0 0 0 3px rgba(217,96,128,0.10) !important;
    outline: none !important;
}

div[data-baseweb="select"] > div,
div[data-baseweb="select"] input {
    background-color: var(--surface) !important;
    border-color: var(--border-mid) !important;
    color: var(--ink) !important;
    border-radius: 8px !important;
}
[data-testid="stSelectbox"] label { color: var(--ink-soft) !important; }
li[role="option"] {
    background: var(--surface) !important;
    color: var(--ink) !important;
}
li[role="option"]:hover { background: var(--sakura-pale) !important; }

[data-testid="stToggle"] input:checked + div {
    background-color: var(--sakura) !important;
}

.stButton > button {
    background: linear-gradient(135deg, var(--sakura) 0%, var(--sakura-deep) 100%) !important;
    color: #FFFDFB !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.86rem !important;
    padding: 0.52rem 1.4rem !important;
    letter-spacing: 0.02em !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 2px 8px rgba(217,96,128,0.30), 0 1px 2px rgba(217,96,128,0.20) !important;
    position: relative !important;
    overflow: hidden !important;
}
.stButton > button::after {
    content: '' !important;
    position: absolute !important;
    inset: 0 !important;
    background: linear-gradient(135deg, rgba(255,255,255,0.15) 0%, transparent 100%) !important;
    opacity: 0 !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, var(--sakura-deep) 0%, #A03050 100%) !important;
    box-shadow: 0 6px 20px rgba(217,96,128,0.40), 0 2px 6px rgba(217,96,128,0.25) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:hover::after { opacity: 1 !important; }
.stButton > button:active { transform: scale(0.97) translateY(0) !important; }

[data-testid="stDownloadButton"] button {
    background-color: transparent !important;
    color: var(--sakura-deep) !important;
    border: 1.5px solid var(--sakura-light) !important;
    box-shadow: none !important;
}
[data-testid="stDownloadButton"] button:hover {
    background-color: var(--sakura-bg) !important;
    box-shadow: none !important;
}

.stProgress > div > div {
    background-color: var(--border) !important;
    border-radius: 999px !important;
}
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--sakura), var(--peach), var(--gold)) !important;
    border-radius: 999px !important;
    transition: width 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 0 8px rgba(217,96,128,0.35) !important;
}

[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 1px solid var(--border) !important;
    background: transparent !important;
    gap: 0 !important;
    padding-bottom: 0 !important;
}
[data-testid="stTabs"] button {
    background: transparent !important;
    color: var(--ink-soft) !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    font-size: 0.83rem !important;
    font-weight: 400 !important;
    padding: 0.55rem 1rem !important;
    margin-bottom: -1px !important;
    border-radius: 4px 4px 0 0 !important;
    transition: color 0.2s, background 0.2s !important;
    letter-spacing: 0.01em !important;
}
[data-testid="stTabs"] button:hover {
    color: var(--sakura) !important;
    background: rgba(217,96,128,0.05) !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--sakura-deep) !important;
    border-bottom: 2px solid var(--sakura) !important;
    font-weight: 600 !important;
    background: rgba(217,96,128,0.04) !important;
}

[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    background: var(--surface) !important;
}
[data-testid="stExpander"] summary {
    color: var(--ink-mid) !important;
    font-size: 0.85rem !important;
}
[data-testid="stExpander"] summary:hover { color: var(--sakura) !important; }

[data-testid="stSpinner"] > div { border-top-color: var(--sakura) !important; }

.stAlert { border-radius: 8px !important; }
div[data-testid="stAlert"][data-baseweb="notification"] {
    background: var(--sakura-pale) !important;
    border-left-color: var(--sakura) !important;
    border-radius: 8px !important;
    color: var(--ink-mid) !important;
}

.stMarkdown p, .stMarkdown li, .stMarkdown span {
    color: var(--ink-mid) !important;
}

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, var(--sakura-light), var(--sakura));
    border-radius: 999px;
    transition: background 0.2s;
}
::-webkit-scrollbar-thumb:hover { background: var(--sakura-deep); }

.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
    transition: border-color 0.25s, box-shadow 0.25s, transform 0.2s;
    box-shadow: 0 1px 3px rgba(60,36,22,0.04), 0 1px 2px rgba(60,36,22,0.03);
    contain: layout style;               /* FIX CLS — prevents card resize cascading */
    will-change: transform;              /* composited — GPU handles hover animation */
}
.card:hover {
    border-color: var(--sakura-light);
    box-shadow: 0 4px 20px rgba(217,96,128,0.12), 0 1px 4px rgba(60,36,22,0.05);
    transform: translateY(-1px);
}

.metric-card {
    background: linear-gradient(135deg, var(--surface) 0%, var(--sakura-pale) 100%);
    border: 1px solid var(--border);
    border-top: 3px solid var(--sakura);
    border-radius: 10px;
    padding: 1.2rem 0.8rem;
    text-align: center;
    transition: box-shadow 0.25s, transform 0.2s;
    box-shadow: 0 1px 3px rgba(60,36,22,0.04);
    min-height: 90px;                    /* FIX CLS — reserved height prevents layout jump */
    contain: layout style;               /* FIX CLS — isolates layout recalcs to this element */
    will-change: transform;              /* composited layer — no main thread paint on hover */
}
.metric-card:hover {
    box-shadow: 0 6px 20px rgba(217,96,128,0.13);
    transform: translateY(-2px);
}
.metric-value {
    font-size: 1.85rem; font-weight: 700;
    color: var(--sakura-deep); line-height: 1.1;
    letter-spacing: -0.02em;
}
.metric-label {
    font-size: 0.59rem; color: var(--ink-faint);
    text-transform: uppercase; letter-spacing: 0.13em; margin-top: 0.4rem;
    font-weight: 600;
}

.sh {
    font-size: 0.67rem; font-weight: 700;
    color: var(--ink-soft); letter-spacing: 0.16em;
    text-transform: uppercase; margin-bottom: 0.9rem;
    padding-bottom: 0.45rem;
    border-bottom: 2px solid var(--border);
    display: flex; align-items: center; gap: 0.5rem;
    background: linear-gradient(90deg, var(--ink-soft), var(--ink-faint));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.action-row {
    display: flex; align-items: flex-start; gap: 0.85rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 4px solid var(--sakura);
    border-radius: 0 12px 12px 0;
    padding: 0.95rem 1.2rem; margin-bottom: 0.65rem;
    transition: border-color 0.25s, box-shadow 0.25s, transform 0.2s;
    box-shadow: 0 1px 3px rgba(60,36,22,0.03);
}
.action-row:hover {
    border-left-color: var(--sakura-deep);
    box-shadow: 0 4px 16px rgba(217,96,128,0.12);
    transform: translateX(2px);
}
.action-row.flagged {
    border-left-color: var(--red);
    background: var(--red-bg);
}
.action-task { font-weight: 500; color: var(--ink); font-size: 0.91rem; line-height: 1.5; }
.action-meta { font-size: 0.78rem; color: var(--ink-soft); margin-top: 0.3rem; }
.action-flag { font-size: 0.74rem; color: var(--red); margin-top: 0.25rem; }

.sentiment-row {
    display: flex; align-items: center; gap: 1rem;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 0.85rem 1.1rem; margin-bottom: 0.55rem;
    transition: background 0.25s, border-color 0.25s, box-shadow 0.25s, transform 0.2s;
    box-shadow: 0 1px 3px rgba(60,36,22,0.03);
}
.sentiment-row:hover {
    background: var(--sakura-pale);
    border-color: var(--sakura-light);
    box-shadow: 0 4px 14px rgba(217,96,128,0.10);
    transform: translateX(2px);
}
.sentiment-name  { font-weight: 500; font-size: 0.89rem; color: var(--ink); min-width: 130px; }
.sentiment-label { font-size: 0.78rem; color: var(--ink-soft); flex: 1; font-style: italic; }

.badge {
    display: inline-block; padding: 0.22rem 0.8rem;
    border-radius: 999px; font-size: 0.68rem; font-weight: 700;
    letter-spacing: 0.07em; text-transform: uppercase;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    transition: transform 0.15s, box-shadow 0.15s;
}
.badge:hover { transform: scale(1.04); }
.badge-positive { background: var(--green-bg); color: var(--green); border: 1px solid rgba(72,104,88,0.2); }
.badge-neutral  { background: var(--peach-bg); color: var(--ink-mid); border: 1px solid rgba(120,80,64,0.15); }
.badge-negative { background: var(--red-bg);   color: var(--red); border: 1px solid rgba(176,64,64,0.2); }

.signal-high {
    background: var(--red-bg);
    border-left: 3px solid var(--red);
    border-radius: 0 10px 10px 0;
    padding: 0.85rem 1.1rem; margin-bottom: 0.6rem;
}
.signal-medium {
    background: var(--amber-bg);
    border-left: 3px solid var(--amber);
    border-radius: 0 10px 10px 0;
    padding: 0.85rem 1.1rem; margin-bottom: 0.6rem;
}
.signal-low {
    background: var(--sakura-pale);
    border-left: 3px solid var(--sakura-light);
    border-radius: 0 10px 10px 0;
    padding: 0.85rem 1.1rem; margin-bottom: 0.6rem;
}
.signal-phrase  {
    font-weight: 600; font-size: 0.9rem;
    font-family: 'Noto Sans JP', sans-serif; color: var(--ink);
}
.signal-reading { font-size: 0.79rem; color: var(--ink-mid); margin-top: 0.2rem; }
.signal-exp     {
    font-size: 0.77rem; color: var(--ink-soft);
    margin-top: 0.4rem; line-height: 1.6;
}

.spk-bar-bg {
    background: var(--border); border-radius: 999px;
    height: 7px; overflow: hidden; margin-top: 0.4rem;
    box-shadow: inset 0 1px 3px rgba(60,36,22,0.06);
}
.spk-bar-fill {
    height: 100%; border-radius: 999px;
    transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 1px 4px rgba(0,0,0,0.12);
}

.pii-pill {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: var(--green-bg); border: 1px solid #A8C8B5;
    border-radius: 999px; padding: 0.28rem 0.9rem;
    font-size: 0.74rem; color: var(--green); font-weight: 500; margin-bottom: 1rem;
}

.risk-pill {
    display: inline-block; padding: 0.28rem 0.9rem;
    border-radius: 999px; font-size: 0.71rem; font-weight: 700;
    letter-spacing: 0.06em; text-transform: uppercase;
}
.risk-HIGH    { background: var(--red-bg);    color: var(--red);         }
.risk-MEDIUM  { background: var(--amber-bg);  color: var(--amber);       }
.risk-LOW     { background: var(--sakura-pale); color: var(--sakura-deep);}
.risk-MINIMAL { background: var(--peach-bg);  color: var(--ink-soft);    }
.risk-NONE    { background: var(--green-bg);  color: var(--green);       }

.sakura-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.4rem 0;
    position: relative;
}


.prev-session-card {
    background: var(--surface-warm);
    border: 1px solid var(--border-mid);
    border-left: 3px solid var(--gold);
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.3rem;
    margin-top: 0.5rem;
    margin-bottom: 0.5rem;
}
.prev-session-header {
    font-size: 0.68rem; font-weight: 600; color: var(--gold);
    letter-spacing: 0.12em; text-transform: uppercase;
    margin-bottom: 0.55rem;
}
.prev-session-bullet {
    font-size: 0.83rem; color: var(--ink-mid);
    line-height: 1.65; margin-bottom: 0.25rem;
    padding-left: 0.9rem; position: relative;
}
.prev-session-bullet::before {
    content: "·";
    position: absolute; left: 0;
    color: var(--gold); font-weight: 700;
}

/* ══════════════════════════════════════════════════════════════
   MOBILE RESPONSIVE — fixes all breakpoints
   Desktop: sidebar visible, columns side-by-side
   Tablet ≤1024px: narrower sidebar, smaller metrics
   Mobile ≤768px: columns stack, tabs scroll, touch targets 44px
   Small ≤480px: full stack, minimal padding
   ══════════════════════════════════════════════════════════════ */

/* ── Sidebar — desktop always visible ──────────────────────── */
[data-testid="stSidebar"] {
    display: flex !important;
    visibility: visible !important;
    transform: none !important;
    min-width: 240px !important;
    max-width: 320px !important;
}
[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"] {
    display: none !important;
}

/* ── Tablet (≤1024px) ──────────────────────────────────────── */
@media (max-width: 1024px) {
    .metric-value { font-size: 1.5rem !important; }
    [data-testid="stSidebar"] {
        min-width: 200px !important;
        max-width: 260px !important;
    }
}

/* ── Mobile (≤768px) ───────────────────────────────────────── */
@media (max-width: 768px) {

    /* Sidebar — allow collapse on mobile, don't force open */
    [data-testid="stSidebar"] {
        min-width: 0 !important;
        max-width: 85vw !important;
    }
    [data-testid="stSidebarCollapseButton"],
    [data-testid="collapsedControl"] {
        display: flex !important;
    }

    /* Stack all columns vertically */
    [data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
        gap: 0.5rem !important;
    }
    [data-testid="column"] {
        width: 100% !important;
        min-width: 100% !important;
        flex: 1 1 100% !important;
    }

    /* Metric cards */
    .metric-card {
        min-height: 70px !important;
        padding: 0.8rem 0.4rem !important;
    }
    .metric-value {
        font-size: 1.3rem !important;
    }
    .metric-label {
        font-size: 0.55rem !important;
        letter-spacing: 0.08em !important;
    }

    /* Cards */
    .card { padding: 0.9rem 1rem !important; }

    /* Action rows */
    .action-row {
        gap: 0.6rem !important;
        padding: 0.75rem 0.9rem !important;
    }
    .action-task { font-size: 0.85rem !important; }
    .action-meta { font-size: 0.73rem !important; }

    /* Sentiment rows */
    .sentiment-row { flex-wrap: wrap !important; gap: 0.5rem !important; }
    .sentiment-name { min-width: 100px !important; font-size: 0.83rem !important; }

    /* Tabs — horizontally scrollable */
    [data-testid="stTabs"] [role="tablist"] {
        overflow-x: auto !important;
        flex-wrap: nowrap !important;
        -webkit-overflow-scrolling: touch !important;
        scrollbar-width: none !important;
    }
    [data-testid="stTabs"] [role="tablist"]::-webkit-scrollbar { display: none !important; }
    [data-testid="stTabs"] button {
        font-size: 0.73rem !important;
        padding: 0.45rem 0.7rem !important;
        white-space: nowrap !important;
    }

    /* Buttons — 44px min touch target (Apple HIG) */
    .stButton > button {
        padding: 0.65rem 1rem !important;
        font-size: 0.82rem !important;
        min-height: 44px !important;
    }

    /* Layout */
    .block-container {
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
        padding-top: 0.5rem !important;
    }
    h1 { font-size: 1.5rem !important; }
    textarea { font-size: 0.82rem !important; }

    /* Signals */
    .signal-high, .signal-medium, .signal-low {
        padding: 0.65rem 0.8rem !important;
    }
    .signal-phrase { font-size: 0.83rem !important; }
    .signal-exp    { font-size: 0.72rem !important; }

    /* Badge */
    .badge { font-size: 0.62rem !important; padding: 0.18rem 0.6rem !important; }

    /* PII pill */
    .pii-pill { flex-wrap: wrap !important; font-size: 0.69rem !important; }

    /* Speaker bar */
    .spk-bar-bg { height: 6px !important; }
}

/* ── Small mobile (≤480px) ─────────────────────────────────── */
@media (max-width: 480px) {
    .metric-value { font-size: 1.15rem !important; }
    h1 { font-size: 1.25rem !important; }

    [data-testid="stTabs"] button {
        font-size: 0.68rem !important;
        padding: 0.4rem 0.55rem !important;
    }

    /* Sample buttons — tighter */
    [data-testid="stHorizontalBlock"] .stButton > button {
        font-size: 0.75rem !important;
        padding: 0.5rem 0.5rem !important;
    }
}
</style>
""", unsafe_allow_html=True)

# ── Sample transcripts ───────────────────────────────────────────────────────
try:
    from tests.sample_transcripts import (
        SAMPLE_TRILINGUAL, SAMPLE_HIGH_CONFLICT, SAMPLE_HINGLISH_STANDUP
    )
except ImportError:
    SAMPLE_TRILINGUAL = """Rahul: Good morning everyone. Aaj hum Q3 product launch ke baare mein discuss karenge.
Priya: Haan, main ready hoon. Mujhe kuch concerns hain about the timeline.
田中: おはようございます。よろしくお願いいたします。
Rahul: Tanaka-san, can you give us the Japan market update first?
田中: はい。Q3の日本市場では、売上目標の92%に達しています。ただ、新機能のリリースについては少し懸念がございます。
Priya: Tanaka, yeh concern kya hai exactly? Timeline issue hai ya technical?
田中: 検討いたします。Technical team se confirm karna padega. It might be difficult to meet the October deadline.
Rahul: Okay. प्रिया, kya tum India market ka update de sakti ho?
Priya: Haan bilkul. India mein hum 87% pe hain. Main blocker hai ki support team short-staffed hai.
Rahul: Priya, can you prepare a staffing proposal by Friday?
Priya: Dekhte hain. Thoda mushkil hai but koshish karenge.
Rahul: I need a yes or no — Friday tak hoga ya nahi?
Priya: Okay, yes. Friday tak de dungi.
田中: 承知しました。Japan side se bhi ek resource provide kar sakte hain if needed.
Rahul: Perfect. 田中 will confirm technical timeline by Wednesday. Next meeting Monday 10am.
田中: はい、水曜日までに確認いたします。お疲れ様でした。"""

    SAMPLE_HIGH_CONFLICT = """Client: This is completely unacceptable. The system has been down for 6 hours.
Kenji: 大変申し訳ございません。We are working on it as fast as possible.
Client: This is the second major outage this month. I need a written commitment.
Kenji: ご要望はよく分かりました。上司に相談して、2時間以内に書面でご回答します。
Client: If this isn't resolved by Friday we will reconsider the entire contract.
Kenji: 誠に申し訳ございません。全力で対応いたします。We will not let that happen."""

    SAMPLE_HINGLISH_STANDUP = """Sharma Sir: Sprint ka kya status hai? Deadline kal hai.
Vikram: Haan sir, almost done hai. Bas ek bug hai jo thoda mushkil hai.
Sharma Sir: Thoda mushkil matlab? Done hoga ya nahi kal tak?
Vikram: Haan haan bilkul sir. Koshish karenge. Dekhte hain, manage ho jayega.
Priya: Main help kar sakti hoon. Kal tak fix ho jayega, 100% committed hoon.
Vikram: Main bhi karta hoon sir. Upar se baat karta hoon agar koi blocker aaya.
Sharma Sir: Seedha mujhe batao. Aap jo theek samjhe karo, but kal tak complete chahiye."""

SAMPLE_TRANSCRIPT = SAMPLE_TRILINGUAL

# ── Session state ────────────────────────────────────────────────────────────
for k, v in [
    ("history", []), ("results", None), ("current_transcript", ""),
    ("current_language", ""), ("transcript_text", ""), ("pii_report", None),
    ("groq_warmed", False),
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Cold start: Groq warmup ──────────────────────────────────────────────────
if not st.session_state.groq_warmed:
    import threading

    def _cold_start_tasks():
        """
        Runs in background on first app load.
        FIX-A: Only reads key from os.getenv — st.secrets is NOT thread-safe.
        FIX-6: Warmup ping removed — was burning 20-29 of 30 daily Groq calls
                on a "hi" → 1 token request that added zero user value.
                Groq's servers are always warm — no ping needed.
        """
        # Pre-cache sample transcripts in vector DB so first demo loads instantly
        try:
            from utils.vector_cache import get_cached_result, store_result, is_available
            from analysis.analyzer import analyze_transcript as _analyze
            import os as _os
            key = _os.getenv("GROQ_API_KEY", "").strip()
            if is_available() and key:
                samples_to_cache = [
                    (SAMPLE_TRILINGUAL,       "mixed"),
                    (SAMPLE_HIGH_CONFLICT,    "mixed"),
                    (SAMPLE_HINGLISH_STANDUP, "hi"),
                ]
                for _s_text, _s_lang in samples_to_cache:
                    try:
                        _cached = get_cached_result(_s_text, _s_lang)
                        if not _cached:
                            _result = _analyze(_s_text, _s_lang)
                            if "mock" not in _result.get("_provider", ""):
                                store_result(_s_text, _s_lang, _result)
                    except Exception:
                        pass
        except Exception:
            pass

    threading.Thread(target=_cold_start_tasks, daemon=True).start()
    st.session_state.groq_warmed = True


# ── Hamburger only ────────────────────────────────────────────────────────────
_NAV_HTML = """
<div id='tai-hbg' title='Open / Close Menu'>
  <span></span><span></span><span></span>
</div>

<script>
(function() {
  var q = String.fromCharCode;

  function toggleSidebar() {
    var s1 = '[data-testid=' + q(34) + 'stSidebarCollapseButton' + q(34) + '] button';
    var s2 = '[data-testid=' + q(34) + 'collapsedControl' + q(34) + ']';
    var a  = document.querySelector(s1) || document.querySelector(s2);
    if (a) { a.click(); }
  }

  function clickHidden(btnId) {
    var btn = document.getElementById(btnId);
    if (btn) { btn.click(); return true; }
    var allBtns = document.querySelectorAll('button');
    for (var i = 0; i < allBtns.length; i++) {
      if (allBtns[i].getAttribute('data-nav') === btnId) {
        allBtns[i].click();
        return true;
      }
    }
    return false;
  }

  function setActive(id) {
    document.querySelectorAll('.tai-lnk').forEach(function(el) {
      el.classList.remove('tai-lnk-active');
    });
    var el = document.getElementById(id);
    if (el) el.classList.add('tai-lnk-active');
  }

  function attach() {
    var hbg = document.getElementById('tai-hbg');
    if (!hbg) { setTimeout(attach, 400); return; }

    hbg.onclick = function() {
      hbg.classList.toggle('tai-hbg-open');
      toggleSidebar();
    };

    var navA = document.getElementById('nav-analyze');
    if (navA) {
      navA.onclick = function(e) {
        setActive('nav-analyze');
        window.scrollTo({ top: 0, behavior: 'smooth' });
      };
    }

    var navH = document.getElementById('nav-history');
    if (navH) {
      navH.onclick = function() {
        setActive('nav-history');
        var sidebar = document.querySelector('[data-testid=' + q(34) + 'stSidebar' + q(34) + ']');
        var isOpen  = sidebar && sidebar.getBoundingClientRect().left > -200;
        if (!isOpen) { toggleSidebar(); }
        setTimeout(function() {
          var sidebarEl = document.querySelector('[data-testid=' + q(34) + 'stSidebar' + q(34) + ']');
          if (sidebarEl) sidebarEl.scrollTop = 300;
        }, 400);
      };
    }

    var navT = document.getElementById('nav-trends');
    if (navT) {
      navT.onclick = function() {
        setActive('nav-trends');
        var allBtns = document.querySelectorAll('button');
        for (var i = 0; i < allBtns.length; i++) {
          if (allBtns[i].innerText.trim() === '__nav_trends__') {
            allBtns[i].click();
            return;
          }
        }
      };
    }

    var navE = document.getElementById('nav-evaluate');
    if (navE) {
      navE.onclick = function() {
        setActive('nav-evaluate');
        var allBtns = document.querySelectorAll('button');
        for (var i = 0; i < allBtns.length; i++) {
          if (allBtns[i].innerText.trim() === '__nav_evaluate__') {
            allBtns[i].click();
            return;
          }
        }
      };
    }
  }

  setTimeout(attach, 700);
})();
</script>
"""
st.markdown(_NAV_HTML, unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1.8rem 0.5rem 1.2rem;'>
      <div style='font-size:1.6rem; margin-bottom:0.4rem;'>🎙️</div>
      <div style='font-size:1rem; font-weight:600; color:#3D2B1F; letter-spacing:0.01em;'>
        TranscriptAI
      </div>
      <div style='font-size:0.62rem; color:#C4A99E; letter-spacing:0.14em;
                  text-transform:uppercase; margin-top:0.2rem;'>
        Speech &amp; Meeting Intelligence
      </div>
    </div>
    <hr style='border:none; border-top:1px solid #EDE0D8; margin:0 0 1rem;'/>
    """, unsafe_allow_html=True)

    st.markdown("<div class='sh'>Language</div>", unsafe_allow_html=True)
    lang_choice = st.selectbox(
        "lang",
        ["Auto-detect", "Japanese (日本語)", "English", "Hindi (हिन्दी)"],
        label_visibility="collapsed",
    )
    lang_map = {
        "Auto-detect": None, "Japanese (日本語)": "ja",
        "English": "en", "Hindi (हिन्दी)": "hi"
    }
    forced_lang = lang_map[lang_choice]

    st.markdown("<hr style='border:none; border-top:1px solid #EDE0D8; margin:1rem 0;'/>", unsafe_allow_html=True)
    if PII_AVAILABLE:
        st.markdown("<div class='sh'>Privacy · APPI</div>", unsafe_allow_html=True)
        pii_enabled = st.toggle("Mask PII before analysis", value=True,
            help="Names, phones, emails anonymized before LLM. Restored locally after.")
        if pii_enabled:
            st.markdown("<div style='font-size:0.75rem; color:#5A7D6B; margin-top:0.2rem;'>✓ APPI compliant</div>", unsafe_allow_html=True)
    else:
        pii_enabled = False

    if STREAMING_AVAILABLE:
        st.markdown("<hr style='border:none; border-top:1px solid #EDE0D8; margin:1rem 0;'/>", unsafe_allow_html=True)
        st.markdown("<div class='sh'>Mode</div>", unsafe_allow_html=True)
        stream_mode = st.toggle("Stream results live", value=False,
            help="See summary generate in real time — requires Groq API key")
    else:
        stream_mode = False

    st.markdown("<hr style='border:none; border-top:1px solid #EDE0D8; margin:1rem 0;'/>", unsafe_allow_html=True)
    st.markdown("<div class='sh'>Recent Analyses</div>", unsafe_allow_html=True)
    if not st.session_state.history:
        st.markdown("<div style='font-size:0.8rem; color:#C4A99E; padding:0.3rem 0;'>No analyses yet.</div>", unsafe_allow_html=True)
    else:
        for i, entry in enumerate(st.session_state.history[:6]):
            label = format_history_label(entry)
            short = (label[:36] + "…") if len(label) > 36 else label
            if st.button(f"↩  {short}", key=f"h_{i}", use_container_width=True):
                st.session_state.results            = entry["results"]
                st.session_state.current_transcript = entry["transcript"]
                st.session_state.current_language   = entry["language"]
                st.session_state.transcript_text    = entry["transcript"]
                st.rerun()

    st.markdown("<hr style='border:none; border-top:1px solid #EDE0D8; margin:1rem 0;'/>", unsafe_allow_html=True)

    try:
        from utils.vector_cache import get_cache_stats
        vc = get_cache_stats()
        if vc.get("available"):
            n = vc.get("transcript_count", 0)
            st.markdown(
                f"<div style='font-size:0.74rem; color:#486858; background:#EDF3EF; "
                f"border:1px solid #A8C8B8; border-radius:6px; padding:0.4rem 0.7rem; "
                f"margin-bottom:0.8rem;'>"
                f"⚡ Vector cache · {n} transcript{'s' if n!=1 else ''} stored"
                f"</div>",
                unsafe_allow_html=True,
            )
    except Exception:
        pass

    with st.expander("About"):
        st.markdown("""
**TranscriptAI** turns any meeting or speech recording into structured intelligence — summaries, action items, speaker sentiment, and communication risk signals.

**Input** &nbsp;·&nbsp; TXT · VTT · JSON · MP4 · MP3 · WAV

**Languages** &nbsp;·&nbsp; English · Hindi · Japanese · Mixed

**Analysis** &nbsp;·&nbsp; Formality level · Indirect signals · Soft rejection · Code-switch

*Set `GROQ_API_KEY` for 3s cloud inference. Or run Ollama locally.*
""")

# ────────────────────────────────────────────────────────────────────────────
# HEADER
# ────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding:2rem 0 1.6rem; position:relative;'>
  <div style='position:absolute; top:1.5rem; right:2rem; opacity:0.15;
              font-size:2.5rem; line-height:1; user-select:none;'>🎙️</div>
  <div style='font-size:0.62rem; color:#C8A898; letter-spacing:0.2em;
              text-transform:uppercase; margin-bottom:0.8rem; font-weight:500;'>
    Speech &amp; Meeting Intelligence
  </div>
  <div style='display:flex; align-items:flex-end; gap:1rem; flex-wrap:wrap; margin-bottom:0.7rem;'>
    <h1 style='font-size:2.1rem; font-weight:600; color:#3C2416;
               margin:0; letter-spacing:-0.025em; line-height:1;'>
      TranscriptAI
    </h1>
  </div>
  <div style='display:flex; align-items:center; gap:0.6rem; flex-wrap:wrap;'>
    <span style='font-size:0.75rem; color:#D96080; background:#FDEEF2;
                 padding:0.2rem 0.7rem; border-radius:999px; font-weight:500;
                 border:1px solid #F2B0C0;'>AI-powered</span>
    <span style='font-size:0.75rem; color:#486858; background:#EDF3EF;
                 padding:0.2rem 0.7rem; border-radius:999px; font-weight:500;
                 border:1px solid #A8C8B8;'>APPI Compliant</span>
    <span style='font-size:0.75rem; color:#B87830; background:#F5E8D0;
                 padding:0.2rem 0.7rem; border-radius:999px; font-weight:500;
                 border:1px solid #D9C090;'>Multi-language</span>
    <span style='font-size:0.75rem; color:#7A5040; background:#FEF3EC;
                 padding:0.2rem 0.7rem; border-radius:999px; font-weight:500;
                 border:1px solid #E5D0C4;'>Formality · Indirect Signals · Code-switch</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────
# INPUT
# ────────────────────────────────────────────────────────────────────────────
st.markdown("<div class='sh'>Transcript Input</div>", unsafe_allow_html=True)

col_up, col_paste = st.columns([1, 1], gap="large")

with col_up:
    st.markdown("<div style='font-size:0.79rem; color:#A8897C; margin-bottom:0.5rem;'>Upload file &nbsp;·&nbsp; TXT · VTT · JSON · MP4 · MP3 · WAV</div>", unsafe_allow_html=True)
    accepted = ["txt", "vtt", "json"] + (["mp4","mp3","wav","m4a","webm"] if AUDIO_AVAILABLE else [])
    uploaded = st.file_uploader("Upload", type=accepted, label_visibility="collapsed")

    if uploaded is not None:
        ext = uploaded.name.lower().split(".")[-1]
        if ext in ["mp4","mp3","wav","m4a","webm","ogg"]:
            size_mb = len(uploaded.getvalue()) / (1024 * 1024)
            if size_mb > MAX_FILE_SIZE_MB:
                st.error(f"File too large ({size_mb:.1f} MB). Max: {MAX_FILE_SIZE_MB} MB")
            else:
                with st.spinner(f"Transcribing {uploaded.name}…"):
                    res = transcribe_audio(uploaded.getvalue(), uploaded.name)
                if res["success"]:
                    seg = format_transcript_with_timestamps(res.get("segments", []))
                    st.session_state.transcript_text = seg or res["text"]
                    st.success(f"✓ Transcribed · {res.get('duration',0):.0f}s · {res.get('language','?')} · {res.get('provider','')}")
                else:
                    st.error(res.get("error", "Transcription failed"))
        else:
            parsed = parse_uploaded_file(uploaded)
            st.session_state.transcript_text = parsed
            st.success(f"✓ Loaded {uploaded.name} · {len(parsed):,} chars")

with col_paste:
    st.markdown("<div style='font-size:0.79rem; color:#A8897C; margin-bottom:0.5rem;'>Or paste transcript directly</div>", unsafe_allow_html=True)
    inp = st.text_area(
        "Transcript", value=st.session_state.transcript_text, height=210,
        placeholder="Paste transcript here…\n\nSupports Japanese, English, Hindi, and mixed text.",
        label_visibility="collapsed",
    )
    if inp != st.session_state.transcript_text:
        st.session_state.transcript_text = inp

c_s1, c_s2, c_s3, c_clear, _ = st.columns([0.22, 0.22, 0.22, 0.14, 0.20])
with c_s1:
    if st.button("🌐 Trilingual", help="Hindi + English + Japanese — tests cross-script name switching", use_container_width=True):
        st.session_state.transcript_text = SAMPLE_TRILINGUAL
        st.rerun()
with c_s2:
    if st.button("⚡ Conflict", help="High-conflict EN+JA — escalation, aggressive tone, PII", use_container_width=True):
        st.session_state.transcript_text = SAMPLE_HIGH_CONFLICT
        st.rerun()
with c_s3:
    if st.button("🗣️ Hinglish", help="Pure Hinglish standup — Hindi NLP layer, hierarchical yes, jugaad", use_container_width=True):
        st.session_state.transcript_text = SAMPLE_HINGLISH_STANDUP
        st.rerun()
with c_clear:
    if st.button("Clear"):
        st.session_state.transcript_text = ""
        st.session_state.results   = None
        st.session_state.pii_report = None
        st.rerun()

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

final_text  = clean_text(st.session_state.transcript_text or "")
can_analyze = len(final_text.strip()) >= 20

c_btn, c_meta = st.columns([0.3, 0.7])
with c_btn:
    run_analysis = st.button("Analyze Transcript →", disabled=not can_analyze, use_container_width=True)
with c_meta:
    if final_text:
        detected     = detect_language(final_text)
        active_disp  = forced_lang or detected
        wc           = len(final_text.split())
        lang_color   = {"ja":"#C45C74","hi":"#C9924A","en":"#5A7D6B","mixed":"#A8897C"}.get(detected,"#A8897C")
        st.markdown(
            f"<div style='padding-top:0.6rem; font-size:0.81rem; color:#A8897C;'>"
            f"Detected <span style='color:{lang_color}; font-weight:600;'>{language_display_name(detected)}</span>"
            f" &nbsp;·&nbsp; Active <span style='color:#C45C74; font-weight:600;'>{language_display_name(active_disp)}</span>"
            f" &nbsp;·&nbsp; {wc:,} words</div>",
            unsafe_allow_html=True,
        )

if not can_analyze and not final_text:
    st.markdown("<div style='font-size:0.82rem; color:#C4A99E; padding:0.4rem 0;'>Paste a transcript or upload a file to begin.</div>", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────
# ANALYSIS
# ────────────────────────────────────────────────────────────────────────────
if run_analysis and final_text:
    detected_lang = detect_language(final_text)
    active_lang   = forced_lang or detected_lang

    ph = st.empty()
    with ph.container():
        bar = st.progress(0, text="Detecting language…")
        time.sleep(0.2)

        pii_mask = None
        text_in  = final_text

        if pii_enabled and PII_AVAILABLE:
            bar.progress(20, text="Masking PII (APPI compliance)…")
            text_in, pii_mask = mask_transcript(final_text)
            st.session_state.pii_report = get_pii_report(pii_mask)

        bar.progress(35, text="Running AI analysis…")
        with st.spinner("Analyzing · ~3s with Groq · 1–2 min with Ollama"):
            results = analyze_transcript(text_in, active_lang)

        if pii_mask is not None:
            results = restore_pii_in_result(results, pii_mask)

        bar.progress(92, text="Finalizing results…")
        time.sleep(0.2)
        bar.progress(100, text="Complete ✓")
        time.sleep(0.3)
    ph.empty()

    st.session_state.results            = results
    st.session_state.current_transcript = final_text
    st.session_state.current_language   = active_lang

    provider   = results.get("_provider", "")
    duration   = results.get("_duration_ms", 0)
    last_error = results.get("_last_error", "")

    if results.get("_from_vector_cache"):
        sim = results.get("_cache_similarity", 0)
        st.success(f"⚡ Loaded from vector cache · {sim:.0%} match · instant")
    elif "mock" in provider:
        # ── FIX-B + FIX-7: show actual reason + AI summary if available ──
        groq_key_present = bool(os.getenv("GROQ_API_KEY", "").strip())
        has_ai_summary   = results.get("_has_ai_summary", False)

        if "no_key" in provider or not groq_key_present:
            warn_msg = (
                "No GROQ_API_KEY found. "
                "Go to your HuggingFace Space → Settings → Repository secrets "
                "and add **GROQ_API_KEY** with your key from console.groq.com (free)."
            )
        elif "rate_limit" in provider or "429" in last_error:
            if has_ai_summary:
                warn_msg = (
                    "Daily API limit reached — showing AI-generated summary below. "
                    "Full structured analysis (action items, sentiment, speakers) resumes in 24 hours."
                )
            else:
                warn_msg = (
                    "Daily API limit reached — demo data shown. "
                    "Full analysis resumes automatically within 24 hours."
                )
        elif "timeout" in provider or "timeout" in last_error.lower():
            warn_msg = "Groq request timed out. Try a shorter transcript (under 800 words)."
        elif "offline" in provider or "connection" in last_error.lower():
            warn_msg = "Could not reach Groq API. Check network or try again in a moment."
        else:
            warn_msg = f"Analysis ran in demo mode. {last_error or 'AI provider unavailable.'}"

        st.warning(f"⚠ {warn_msg}")

        # FIX-C: debug expander
        with st.expander("🔍 Debug info", expanded=False):
            st.code(
                f"provider   : {provider}\n"
                f"groq_key   : present={groq_key_present}\n"
                f"has_ai_sum : {has_ai_summary}\n"
                f"last_error : {last_error or 'none'}\n"
                f"duration   : {duration}ms",
                language="text",
            )
    else:
        st.success(f"✓ Analysis complete · {provider} · {duration/1000:.1f}s")

    st.session_state.history = add_to_history(st.session_state.history, {
        "timestamp": datetime.now().isoformat(),
        "language":  active_lang,
        "snippet":   final_text[:80],
        "transcript":final_text,
        "results":   results,
    })
    st.rerun()

# ── Streaming ────────────────────────────────────────────────────────────────
if STREAMING_AVAILABLE and stream_mode and final_text and not run_analysis:
    if st.button("⚡ Stream Live Summary"):
        st.markdown("<div class='sh'>Live Summary</div>", unsafe_allow_html=True)
        try:
            st.write_stream(stream_transcript_groq(
                final_text, st.session_state.get("current_language","en")
            ))
        except Exception as e:
            st.error(str(e))

# ────────────────────────────────────────────────────────────────────────────
# MEETING HEALTH SCORE
# ────────────────────────────────────────────────────────────────────────────
def compute_health_score(R: dict) -> dict:
    score = 0
    breakdown = {}

    sentiment = R.get("sentiment", [])
    if sentiment:
        weights = {"positive": 1.0, "neutral": 0.6, "negative": 0.1}
        avg = sum(weights.get(s.get("score","neutral").lower(), 0.5)
                  for s in sentiment) / len(sentiment)
        s_pts = round(avg * 30)
    else:
        s_pts = 15
    score += s_pts
    breakdown["sentiment"] = s_pts

    items = R.get("action_items", [])
    if not items:
        a_pts = 10
    else:
        verified = [i for i in items if not i.get("hallucination_flag", False)]
        with_owner    = sum(1 for i in verified
                           if i.get("owner","TBD") not in ("TBD","Unknown",""))
        with_deadline = sum(1 for i in verified
                           if i.get("deadline","TBD") not in ("TBD","N/A",""))
        clarity = (with_owner + with_deadline) / (2 * len(items))
        a_pts = round(clarity * 25)
    score += a_pts
    breakdown["action_clarity"] = a_pts

    soft = R.get("soft_rejections", {})
    risk = soft.get("risk_level", "NONE")
    risk_pts = {"NONE": 25, "MINIMAL": 20, "LOW": 15, "MEDIUM": 8, "HIGH": 0}
    r_pts = risk_pts.get(risk, 25)
    score += r_pts
    breakdown["soft_rejection"] = r_pts

    verification = R.get("verification", {})
    hall_rate = verification.get("overall_hallucination_risk", 0)
    h_pts = round((1 - hall_rate) * 20)
    score += h_pts
    breakdown["hallucination"] = h_pts

    if score >= 80:
        label, color, bg, border = "Productive Meeting", "#486858", "#EDF3EF", "#A8C8B8"
    elif score >= 60:
        label, color, bg, border = "Mostly Aligned",    "#986820", "#FAF0E0", "#D9C090"
    elif score >= 40:
        label, color, bg, border = "Needs Follow-up",   "#C87030", "#FDF0EA", "#E8C090"
    else:
        label, color, bg, border = "High Risk",         "#B04040", "#FAF0F0", "#E8A0A0"

    return {
        "score":     min(score, 100),
        "label":     label,
        "color":     color,
        "bg":        bg,
        "border":    border,
        "breakdown": breakdown,
    }

st.markdown("""
<style>
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0);    }
}
.results-wrapper { animation: fadeSlideUp 0.4s cubic-bezier(0.4, 0, 0.2, 1) both; }
</style>
<div class='results-wrapper' id='results-top'></div>
""", unsafe_allow_html=True)

if st.session_state.results:
    R        = st.session_state.results
    language = st.session_state.current_language
    pii_rep  = st.session_state.pii_report
    features = get_features(language)

    st.markdown(
        "<hr style='border:none;border-top:1px solid rgba(255,255,255,0.08);margin:1.6rem 0 1rem;'/>",
        unsafe_allow_html=True,
    )

    # ── Single HTML render — replaces 50+ individual st.markdown() calls ──────
    # One WebSocket message instead of 50+ = scales to 10K concurrent users
    st.markdown(
        build_results_html(R, language, features, pii_rep),
        unsafe_allow_html=True,
    )

    # ── Streamlit-native controls (need Python callbacks) ─────────────────────
    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

    # Export JSON (Streamlit download button — can't be done in pure HTML)
    from utils import build_export_json, export_filename
    exp = build_export_json(st.session_state.current_transcript, language, R)
    st.download_button(
        "⬇ Export JSON",
        data=exp.encode(),
        file_name=export_filename(language),
        mime="application/json",
    )

    # ── Evaluation tab (needs Streamlit widgets — kept native) ────────────────
    if EVAL_AVAILABLE:
        with st.expander("📊 Accuracy Evaluation · Ground Truth Comparison"):
            st.markdown(
                "<div style='font-size:0.82rem;color:rgba(255,255,255,0.5);margin-bottom:1rem'>"
                "Select a test case with known ground truth to measure analysis accuracy.</div>",
                unsafe_allow_html=True,
            )
            tc_names = [tc["name"] for tc in TEST_CASES]
            selected = st.selectbox("Test case", tc_names, key="eval_tc")
            tc       = next(t for t in TEST_CASES if t["name"] == selected)
            if st.button("Run evaluation →", key="run_eval"):
                with st.spinner("Evaluating…"):
                    pred   = analyze_transcript(
                        tc["transcript"], tc["language"], bypass_cache=True
                    )
                    report = evaluate(
                        pred, tc["ground_truth"], tc["transcript"],
                        tc_name=tc["name"],
                        provider=pred.get("_provider", "unknown")
                    )
                overall = report.get("overall_score", 0)
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Overall", f"{overall}%")
                with c2:
                    st.metric("ROUGE-1", report.get("summary",{}).get("avg_rouge1_f1","—"))
                with c3:
                    st.metric("Action F1", report.get("action_items",{}).get("f1","—"))
                with c4:
                    st.metric("Sentiment", report.get("sentiment",{}).get("soft_accuracy","—"))
                if "japan_insights" in report:
                    ji_r = report["japan_insights"]
                    c5, c6, c7 = st.columns(3)
                    with c5: st.metric("Keigo", ji_r["keigo"].get("grade","—"))
                    with c6: st.metric("Nemawashi P", ji_r["nemawashi"].get("precision","—"))
                    with c7: st.metric("Code-switch", ji_r["code_switching"].get("grade","—"))
                with st.expander("Full report (JSON)"):
                    st.json(report)

    # ── Trends tab (needs Streamlit charts — kept native) ─────────────────────
    if TRENDS_AVAILABLE:
        with st.expander("📈 Meeting Intelligence Trends"):
            trends = get_trends(last_n=50)
            if trends.get("empty"):
                st.info(trends.get("message","No trend data yet."))
            else:
                c1, c2, c3, c4 = st.columns(4)
                with c1: st.metric("High Risk Meetings", f"{trends['high_soft_rejection_pct']}%")
                with c2: st.metric("Avg Hallucination", f"{trends['avg_hallucination_pct']}%")
                with c3: st.metric("Avg Action Items", trends["avg_action_items"])
                with c4:
                    dur = trends["avg_duration_sec"]
                    st.metric("Avg Analysis Time", f"{dur:.0f}s" if dur < 60 else f"{dur/60:.1f}m")
                for alert_key in ["soft_rejection_alert","hallucination_alert","duration_alert"]:
                    alert = trends.get(alert_key)
                    if alert:
                        st.warning(alert)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding:2.5rem 0 1.5rem;
            color:#C4A99E; font-size:0.73rem; letter-spacing:0.05em;'>
  🎙️ &nbsp; TranscriptAI &nbsp;·&nbsp; Speech &amp; Meeting Intelligence
  &nbsp;·&nbsp; APPI Compliant &nbsp; 🎙️
  <br><br>
  <span style='color:#EDE0D8;'>Groq · Ollama · Claude · GPT-4 · Any OpenAI-compatible provider</span>
</div>
""", unsafe_allow_html=True)
