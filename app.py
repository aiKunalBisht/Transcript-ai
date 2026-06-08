"""
app.py — TranscriptAI  v7.1
Japanese Business Intelligence Platform

Run: python -m streamlit run app.py

v7.1 FIXES (app.py side):
  FIX-A: _cold_start_tasks no longer calls st.secrets inside a background thread
  FIX-B: Mock warning block now checks _last_error from results
  FIX-C: Added a small debug expander under the warning
  FIX-D: Moved build_results_html + helpers ABOVE the call site (NameError fix)
  FIX-E: Split footer into two st.markdown() calls (truncation fix from v2)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
from datetime import datetime
import streamlit as st
from analysis import analyze_transcript
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

# ── Page config — MUST be the first Streamlit command ────────────────────────
st.set_page_config(
    page_title="TranscriptAI · Speech & Meeting Analyzer",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── SEO + preconnect head tags ───────────────────────────────────────────────
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

# ── CSS — warm sakura/peach palette ─────────────────────────────────────────
st.markdown("""
<style>
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
    contain: layout style;
    will-change: transform;
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
    min-height: 90px;
    contain: layout style;
    will-change: transform;
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

@media (max-width: 1024px) {
    .metric-value { font-size: 1.5rem !important; }
    [data-testid="stSidebar"] {
        min-width: 200px !important;
        max-width: 260px !important;
    }
}

@media (max-width: 768px) {
    [data-testid="stSidebar"] {
        min-width: 0 !important;
        max-width: 85vw !important;
    }
    [data-testid="stSidebarCollapseButton"],
    [data-testid="collapsedControl"] {
        display: flex !important;
    }
    [data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
        gap: 0.5rem !important;
    }
    [data-testid="column"] {
        width: 100% !important;
        min-width: 100% !important;
        flex: 1 1 100% !important;
    }
    .metric-card {
        min-height: 70px !important;
        padding: 0.8rem 0.4rem !important;
    }
    .metric-value { font-size: 1.3rem !important; }
    .metric-label { font-size: 0.55rem !important; letter-spacing: 0.08em !important; }
    .card { padding: 0.9rem 1rem !important; }
    .action-row { gap: 0.6rem !important; padding: 0.75rem 0.9rem !important; }
    .action-task { font-size: 0.85rem !important; }
    .action-meta { font-size: 0.73rem !important; }
    .sentiment-row { flex-wrap: wrap !important; gap: 0.5rem !important; }
    .sentiment-name { min-width: 100px !important; font-size: 0.83rem !important; }
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
    .stButton > button {
        padding: 0.65rem 1rem !important;
        font-size: 0.82rem !important;
        min-height: 44px !important;
    }
    .block-container {
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
        padding-top: 0.5rem !important;
    }
    h1 { font-size: 1.5rem !important; }
    textarea { font-size: 0.82rem !important; }
    .signal-high, .signal-medium, .signal-low { padding: 0.65rem 0.8rem !important; }
    .signal-phrase { font-size: 0.83rem !important; }
    .signal-exp    { font-size: 0.72rem !important; }
    .badge { font-size: 0.62rem !important; padding: 0.18rem 0.6rem !important; }
    .pii-pill { flex-wrap: wrap !important; font-size: 0.69rem !important; }
    .spk-bar-bg { height: 6px !important; }
}

@media (max-width: 480px) {
    .metric-value { font-size: 1.15rem !important; }
    h1 { font-size: 1.25rem !important; }
    [data-testid="stTabs"] button {
        font-size: 0.68rem !important;
        padding: 0.4rem 0.55rem !important;
    }
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
        FIX-A: Only reads key from os.getenv — st.secrets is NOT thread-safe.
        """
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


# ── Hamburger nav ─────────────────────────────────────────────────────────────
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
  function attach() {
    var hbg = document.getElementById('tai-hbg');
    if (!hbg) { setTimeout(attach, 400); return; }
    hbg.onclick = function() {
      hbg.classList.toggle('tai-hbg-open');
      toggleSidebar();
    };
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
    st.session_state["analysis_result"]     = results       
    st.session_state["detected_language"]   = active_lang 


    provider   = results.get("_provider", "")
    duration   = results.get("_duration_ms", 0)
    last_error = results.get("_last_error", "")

    if results.get("_from_vector_cache"):
        sim = results.get("_cache_similarity", 0)
        st.success(f"⚡ Loaded from vector cache · {sim:.0%} match · instant")
    elif "mock" in provider:
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

# ════════════════════════════════════════════════════════════════════════════
# RESULTS RENDERER — helper functions defined HERE, before they are called
# ════════════════════════════════════════════════════════════════════════════

def _svg_donut(pct: int, color: str, size: int = 56) -> str:
    """SVG donut chart — GPU composited, zero JS."""
    r = (size - 8) // 2
    circ = 2 * 3.14159 * r
    dash = circ * pct / 100
    return (
        f"<svg width='{size}' height='{size}' viewBox='0 0 {size} {size}'>"
        f"<circle cx='{size//2}' cy='{size//2}' r='{r}' fill='none' "
        f"stroke='rgba(60,36,22,0.12)' stroke-width='6'/>"
        f"<circle cx='{size//2}' cy='{size//2}' r='{r}' fill='none' "
        f"stroke='{color}' stroke-width='6' stroke-linecap='round' "
        f"stroke-dasharray='{dash:.1f} {circ:.1f}' "
        f"transform='rotate(-90 {size//2} {size//2})'/>"
        f"<text x='50%' y='54%' text-anchor='middle' "
        f"font-size='13' font-weight='700' fill='{color}' font-family='Arial'>"
        f"{pct}%</text></svg>"
    )


def _avatar(name: str, color: str) -> str:
    initials = "".join(p[0].upper() for p in name.split()[:2]) or name[:2].upper()
    return (
        f"<div style='width:36px;height:36px;border-radius:50%;background:{color}22;"
        f"border:2px solid {color};display:flex;align-items:center;justify-content:center;"
        f"font-size:0.75rem;font-weight:700;color:{color};flex-shrink:0'>{initials}</div>"
    )


def _health_ring(score: int, color: str) -> str:
    r, size = 54, 120
    circ = 2 * 3.14159 * r
    dash = circ * score / 100
    label = ("Excellent" if score >= 80 else "Good" if score >= 60
             else "Fair" if score >= 40 else "At Risk")
    return (
        f"<div style='text-align:center'>"
        f"<svg width='{size}' height='{size}' viewBox='0 0 {size} {size}'>"
        f"<circle cx='60' cy='60' r='{r}' fill='none' stroke='rgba(60,36,22,0.10)' stroke-width='10'/>"
        f"<circle cx='60' cy='60' r='{r}' fill='none' stroke='{color}' stroke-width='10' "
        f"stroke-linecap='round' stroke-dasharray='{dash:.1f} {circ:.1f}' "
        f"transform='rotate(-90 60 60)' style='filter:drop-shadow(0 0 6px {color}88)'/>"
        f"<text x='50%' y='46%' text-anchor='middle' font-size='22' font-weight='800' "
        f"fill='#3C2416' font-family=Arial>{score}</text>"
        f"<text x='50%' y='62%' text-anchor='middle' font-size='10' fill='#A87868' "
        f"font-family=Arial>/ 100</text></svg>"
        f"<div style='font-size:0.7rem;font-weight:600;color:{color};letter-spacing:0.1em;"
        f"text-transform:uppercase;margin-top:2px'>{label}</div></div>"
    )


def build_results_html(R: dict, language: str, features: dict,
                       pii_rep: dict | None) -> str:
    """
    Builds the ENTIRE results view as a single HTML string.
    One st.markdown() call = one WebSocket message = scales to 10K users.
    """
    COLORS = ["#E8829A","#F4A07A","#C9924A","#5A7D6B","#A8897C","#7A5C50"]
    SENT_ICON  = {"positive":"🌸","neutral":"🌿","negative":"🍂"}

    ji       = R.get("japan_insights", {})
    speakers = sorted(R.get("speakers", []),
                      key=lambda s: s.get("talk_time_pct", 0), reverse=True)

    # ── Health score ──────────────────────────────────────────────────────────
    def _health():
        soft  = R.get("soft_rejections", {})
        risk  = soft.get("risk_level", "NONE")
        risk_pts = {"NONE":25,"MINIMAL":20,"LOW":15,"MEDIUM":8,"HIGH":0}
        sents = R.get("sentiment", [])
        w = {"positive":1.0,"neutral":0.6,"negative":0.1}
        s_pts = round((sum(w.get(s.get("score","neutral").lower(),0.5)
                           for s in sents)/len(sents)*30) if sents else 15)
        items = R.get("action_items", [])
        if not items:
            a_pts = 10
        else:
            ver = [i for i in items if not i.get("hallucination_flag")]
            wo  = sum(1 for i in ver if i.get("owner","TBD") not in ("TBD","Unknown",""))
            wd  = sum(1 for i in ver if i.get("deadline","TBD") not in ("TBD","N/A",""))
            a_pts = round((wo+wd)/(2*len(items))*25)
        r_pts = risk_pts.get(risk, 25)
        ver2  = R.get("verification", {})
        h_pts = round((1 - ver2.get("overall_hallucination_risk", 0)) * 20)
        score = min(s_pts + a_pts + r_pts + h_pts, 100)
        color = ("#2D9E6B" if score >= 80 else "#B87830" if score >= 60
                 else "#D96080" if score >= 40 else "#C84040")
        bd = [("Sentiment", s_pts, 30),("Action Clarity", a_pts, 25),
              ("Comm Risk", r_pts, 25),("AI Confidence", h_pts, 20)]
        bars = "".join(
            f"<div style='margin-bottom:8px'>"
            f"<div style='display:flex;justify-content:space-between;margin-bottom:3px'>"
            f"<span style='font-size:0.68rem;color:#7A5040'>{lb}</span>"
            f"<span style='font-size:0.68rem;color:{color};font-weight:600'>{pt}/{tot}</span></div>"
            f"<div style='height:5px;background:rgba(60,36,22,0.10);border-radius:999px'>"
            f"<div style='height:100%;width:{round(pt/tot*100)}%;background:{color};"
            f"border-radius:999px;box-shadow:0 0 6px {color}66'></div></div></div>"
            for lb, pt, tot in bd
        )
        return score, color, bars

    score, hc, hbars = _health()

    # ── Metric tiles ──────────────────────────────────────────────────────────
    spk_count = len(R.get("speakers", []))
    act_count = len(R.get("action_items", []))
    cs_val    = ji.get("code_switch_count","—") if features.get("show_code_switch") else "—"
    keigo_val = ji.get("keigo_level","—").title() if features.get("show_japan_insights") else language_display_name(language).split(" ",1)[-1]
    keigo_lbl = "Formality" if features.get("show_japan_insights") else "Language"

    def _tile(val, lbl, icon):
        return (
            f"<div class='tai-tile'>"
            f"<div class='tai-tile-icon'>{icon}</div>"
            f"<div class='tai-tile-val'>{val}</div>"
            f"<div class='tai-tile-lbl'>{lbl}</div>"
            f"</div>"
        )

    tiles = (
        _tile(spk_count, "Speakers", "🎤") +
        _tile(act_count, "Actions", "✅") +
        _tile(cs_val, "Code Switches", "🌐") +
        _tile(keigo_val, keigo_lbl, "🏯")
    )

    # ── PII pill ──────────────────────────────────────────────────────────────
    pii_html = ""
    if pii_rep and pii_rep.get("total_pii_found", 0) > 0:
        n = pii_rep["total_pii_found"]
        pii_html = (
            f"<div class='tai-pii-pill'>🔒 APPI — "
            f"{n} item{'s' if n!=1 else ''} anonymized before analysis</div>"
        )

    # ── Tab 1: Summary ────────────────────────────────────────────────────────
    full_sum = R.get("full_summary", "")
    bullets  = R.get("summary", [])
    sum_html = ""
    if full_sum:
        sum_html += (
            f"<div class='tai-summary-box'>"
            f"<div class='tai-summary-label'>📋 Meeting Overview</div>"
            f"<p style='margin:0;line-height:1.9;font-size:0.93rem;color:#3C2416'>"
            f"{full_sum}</p></div>"
        )
    sum_html += f"<div class='tai-section-label'>{len(bullets)} Key Points</div>"
    sum_html += "".join(
        f"<div class='tai-bullet-card'>"
        f"<span class='tai-bullet-num'>{i:02d}</span>"
        f"<span style='color:#3C2416;font-size:0.9rem;line-height:1.65'>{b}</span>"
        f"</div>"
        for i, b in enumerate(bullets, 1)
    )

    # ── Tab 2: Actions ────────────────────────────────────────────────────────
    items    = R.get("action_items", [])
    v_count  = sum(1 for i in items if not i.get("hallucination_flag"))
    f_count  = len(items) - v_count
    act_html = (
        f"<div class='tai-section-label'>{len(items)} Items · "
        f"<span style='color:#2D9E6B'>✓ {v_count} verified</span>"
        + (f" · <span style='color:#C84040'>⚑ {f_count} flagged</span>" if f_count else "")
        + "</div>"
    )
    act_html += "".join(
        (
            "<div class='tai-action-card" +
            (" tai-action-flagged" if i.get("hallucination_flag") else "") +
            "'>"
            "<div style='font-size:1.1rem;padding-top:2px'>" +
            ("⚑" if i.get("hallucination_flag") else "◆") +
            "</div>"
            "<div style='flex:1'>"
            "<div style='font-weight:600;color:#3C2416;font-size:0.9rem;margin-bottom:4px'>" +
            str(i.get("task","")) + "</div>"
            "<div style='font-size:0.76rem;color:#A87868'>"
            "Owner: <strong style='color:#7A5040'>" + str(i.get("owner","TBD")) + "</strong>"
            " &nbsp;·&nbsp; Deadline: <strong style='color:#7A5040'>" + str(i.get("deadline","TBD")) + "</strong>" +
            (f" &nbsp;·&nbsp; {i.get('confidence',0):.0%} confidence" if i.get("confidence") else "") +
            (f"<div style='color:#963030;font-size:0.72rem;margin-top:3px'>⚠ {i.get('flag_reason','')}</div>" if i.get("flag_reason") else "") +
            "</div></div></div>"
        )
        for i in items
    ) if items else "<div style='color:#A87868;font-size:0.85rem;padding:1rem 0'>No action items extracted.</div>"

    # ── Tab 3: Sentiment ──────────────────────────────────────────────────────
    sent_html = "<div class='tai-section-label'>Speaker Sentiment</div>"
    sent_html += "".join(
        (
            "<div class='tai-sent-row'>"
            "<span style='font-size:1.2rem'>" + SENT_ICON.get(s.get("score","neutral").lower(),"🌿") + "</span>"
            "<div style='flex:1'>"
            "<div style='font-weight:600;color:#3C2416;font-size:0.88rem'>" + str(s.get("speaker","")) + "</div>"
            "<div style='font-size:0.75rem;color:#A87868;font-style:italic;margin-top:1px'>" + str(s.get("label","")) + "</div>"
            "</div>"
            "<span class='tai-sent-badge tai-sent-" + s.get("score","neutral").lower() + "'>" +
            s.get("score","neutral").upper() + "</span>"
            "</div>"
        )
        for s in R.get("sentiment", [])
    )

    # ── Tab 4: Speakers ───────────────────────────────────────────────────────
    spk_html = "<div class='tai-section-label'>Talk Time Distribution</div>"
    for idx2, spk in enumerate(speakers):
        nm  = spk.get("name", f"Speaker {idx2+1}")
        pct = spk.get("talk_time_pct", 0)
        tone= spk.get("tone","—")
        col = COLORS[idx2 % len(COLORS)]
        spk_html += (
            f"<div class='tai-spk-row'>"
            f"{_avatar(nm, col)}"
            f"<div style='flex:1;min-width:0'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:6px'>"
            f"<span style='font-weight:600;color:#3C2416;font-size:0.88rem'>{nm}</span>"
            f"<span style='font-size:0.75rem;color:{col};font-weight:600'>{pct}%</span></div>"
            f"<div style='height:6px;background:rgba(60,36,22,0.10);border-radius:999px'>"
            f"<div style='height:100%;width:{pct}%;background:{col};border-radius:999px;"
            f"box-shadow:0 0 8px {col}66;transition:width 0.8s cubic-bezier(0.4,0,0.2,1)'></div></div>"
            f"<div style='font-size:0.7rem;color:#A87868;margin-top:4px'>{tone}</div>"
            f"</div>"
            f"{_svg_donut(pct, col, 52)}"
            f"</div>"
        )

    # ── Tab 5: Insights (Japan/Hindi) ─────────────────────────────────────────
    ins_html = ""
    if features.get("show_japan_insights"):
        keigo   = ji.get("keigo_level","—")
        k_src   = ji.get("keigo_source","llm")
        kc      = {"high":"#BE4060","medium":"#986820","low":"#A87868"}.get(keigo,"#7A5040")
        sigs    = ji.get("nemawashi_signals",[])
        soft    = R.get("soft_rejections",{})
        risk    = soft.get("risk_level","NONE") if soft else "NONE"
        rclr    = {"HIGH":"#963030","MEDIUM":"#986820","LOW":"#BE4060","MINIMAL":"#A87868","NONE":"#2D7A55"}.get(risk,"#2D7A55")
        cs_cnt  = ji.get("code_switch_count",0)

        ins_html += (
            "<div style='display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px'>"
            "<div class='tai-insight-chip'>"
            "<div style='font-size:0.6rem;color:#A87868;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:2px'>Keigo Register</div>"
            f"<div style='font-size:1.1rem;font-weight:700;color:{kc}'>{keigo.upper()}</div>"
            f"<div style='font-size:0.62rem;color:#C8A898'>via {k_src}</div>"
            "</div>"
            "<div class='tai-insight-chip'>"
            "<div style='font-size:0.6rem;color:#A87868;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:2px'>Rejection Risk</div>"
            f"<div style='font-size:1.1rem;font-weight:700;color:{rclr}'>{risk}</div>"
            f"<div style='font-size:0.62rem;color:#C8A898'>{soft.get('total_signals',0)} signals</div>"
            "</div>"
            "<div class='tai-insight-chip'>"
            "<div style='font-size:0.6rem;color:#A87868;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:2px'>Code Switches</div>"
            f"<div style='font-size:1.1rem;font-weight:700;color:#E88060'>{cs_cnt}</div>"
            "<div style='font-size:0.62rem;color:#C8A898'>language switches</div>"
            "</div>"
            "</div>"
        )

        if sigs:
            ins_html += f"<div class='tai-section-label'>Indirect Consensus Signals · {len(sigs)} detected</div>"
            ins_html += "".join(
                "<div class='tai-nemawashi-pill'>◆ " + s + "</div>" for s in sigs
            )

        if soft and soft.get("total_signals",0) > 0:
            ins_html += "<div class='tai-section-label' style='margin-top:16px'>Soft Rejection Analysis</div>"
            for sig in soft.get("high_signals",[]):
                ins_html += (
                    f"<div class='tai-sig-high'>"
                    f"<div style='font-weight:700;font-size:0.9rem'>🚨 {sig['phrase']}</div>"
                    f"<div style='font-size:0.76rem;color:#7A5040;margin-top:4px'>"
                    f"{sig['reading']} · {sig['speaker']} · {sig['confidence']:.0%}</div>"
                    f"<div style='font-size:0.75rem;color:#3C2416;margin-top:6px;line-height:1.5'>{sig['explanation']}</div>"
                    f"</div>"
                )
            for sig in soft.get("medium_signals",[]):
                ins_html += (
                    f"<div class='tai-sig-med'>"
                    f"<div style='font-weight:700;font-size:0.9rem'>⚠ {sig['phrase']}</div>"
                    f"<div style='font-size:0.76rem;color:#7A5040;margin-top:4px'>"
                    f"{sig['reading']} · {sig['speaker']} · {sig['confidence']:.0%}</div>"
                    f"<div style='font-size:0.75rem;color:#3C2416;margin-top:6px;line-height:1.5'>{sig['explanation']}</div>"
                    f"</div>"
                )
            ins_html += f"<div style='font-size:0.73rem;color:#A87868;font-style:italic;margin-top:8px'>{soft.get('cultural_note','')}</div>"
    else:
        ins_html = "<div style='color:#A87868;font-size:0.85rem;padding:1rem 0;line-height:1.7'>Cultural intelligence features apply to Japanese and Hindi transcripts.</div>"

    # ── Assemble full HTML ────────────────────────────────────────────────────
    return f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Noto+Sans+JP:wght@400;500;700&display=swap" rel="stylesheet">
<style>
:root {{
  --glass:     rgba(60,36,22,0.04);
  --glass-b:   rgba(60,36,22,0.10);
  --glass-h:   rgba(60,36,22,0.14);
  --accent:    #BE4060;
  --accent2:   #E88060;
  --green:     #2D7A55;
  --ink:       #3C2416;
  --ink-mid:   #7A5040;
  --ink-soft:  #A87868;
  --r:         14px;
}}
.tai-results {{
  font-family: 'Inter','Noto Sans JP',sans-serif;
  color: var(--ink);
  padding: 0 0 2rem;
}}
.tai-glass {{
  background: var(--glass);
  border: 1px solid var(--glass-b);
  border-radius: var(--r);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  transition: border-color 0.25s, box-shadow 0.25s;
  contain: layout style;
}}
.tai-glass:hover {{
  border-color: rgba(190,64,96,0.25);
  box-shadow: 0 8px 32px rgba(190,64,96,0.07);
}}
.tai-tiles {{
  display: grid;
  grid-template-columns: repeat(4,1fr);
  gap: 12px;
  margin-bottom: 16px;
}}
@media(max-width:768px) {{ .tai-tiles {{ grid-template-columns: repeat(2,1fr); }} }}
.tai-tile {{
  background: var(--glass);
  border: 1px solid var(--glass-b);
  border-radius: var(--r);
  padding: 16px 12px;
  text-align: center;
  transition: transform 0.2s, border-color 0.25s, box-shadow 0.25s;
  will-change: transform;
  contain: layout style;
  min-height: 90px;
}}
.tai-tile:hover {{
  transform: translateY(-3px);
  border-color: rgba(190,64,96,0.30);
  box-shadow: 0 12px 32px rgba(190,64,96,0.10);
}}
.tai-tile-icon {{ font-size: 1.3rem; margin-bottom: 4px; }}
.tai-tile-val  {{ font-size: 1.6rem; font-weight: 800; color: var(--accent); letter-spacing: -0.02em; line-height: 1; }}
.tai-tile-lbl  {{ font-size: 0.6rem; color: var(--ink-soft); text-transform: uppercase; letter-spacing: 0.12em; margin-top: 4px; font-weight: 600; }}
.tai-health {{
  display: grid;
  grid-template-columns: 140px 1fr;
  gap: 0;
  background: linear-gradient(135deg, rgba(232,130,154,0.08) 0%, rgba(244,160,122,0.05) 100%);
  border: 1px solid rgba(232,130,154,0.2);
  border-radius: var(--r);
  overflow: hidden;
  margin-bottom: 16px;
}}
@media(max-width:600px) {{ .tai-health {{ grid-template-columns: 1fr; }} }}
.tai-health-left  {{ padding: 24px 20px; display: flex; align-items: center; justify-content: center; border-right: 1px solid rgba(60,36,22,0.10); }}
.tai-health-right {{ padding: 20px 24px; }}
.tai-health-title {{ font-size: 0.6rem; font-weight: 700; color: var(--accent); letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 14px; }}
@media(max-width:600px) {{ .tai-health-left {{ border-right: none; border-bottom: 1px solid rgba(60,36,22,0.10); }} }}
.tai-tab-bar {{
  display: flex; gap: 4px;
  border-bottom: 1px solid rgba(60,36,22,0.12);
  margin-bottom: 16px;
  overflow-x: auto;
  scrollbar-width: none;
  -webkit-overflow-scrolling: touch;
}}
.tai-tab-bar::-webkit-scrollbar {{ display: none; }}
.tai-tab {{
  padding: 8px 16px;
  font-size: 0.8rem; font-weight: 500;
  color: var(--ink-soft); border: none; background: none;
  border-bottom: 2px solid transparent;
  cursor: pointer; white-space: nowrap;
  transition: color 0.2s, border-color 0.2s;
}}
.tai-tab:hover {{ color: var(--accent); }}
.tai-tab.active {{ color: var(--accent); border-bottom-color: var(--accent); font-weight: 600; }}
.tai-tab-content {{ display: none; animation: fadeIn 0.25s ease; }}
.tai-tab-content.active {{ display: block; }}
@keyframes fadeIn {{ from{{opacity:0;transform:translateY(6px)}} to{{opacity:1;transform:none}} }}
.tai-section-label {{
  font-size: 0.62rem; font-weight: 700; color: var(--ink-soft);
  letter-spacing: 0.16em; text-transform: uppercase;
  border-bottom: 1px solid rgba(60,36,22,0.12);
  padding-bottom: 8px; margin-bottom: 12px; margin-top: 8px;
}}
.tai-summary-box {{
  background: var(--glass);
  border: 1px solid var(--glass-b);
  border-left: 3px solid var(--accent);
  border-radius: 0 var(--r) var(--r) 0;
  padding: 18px 20px;
  margin-bottom: 16px;
  line-height: 1.85;
}}
.tai-summary-label {{
  font-size: 0.62rem; font-weight: 700; color: var(--accent);
  letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 10px;
}}
.tai-bullet-card {{
  display: flex; align-items: flex-start; gap: 12px;
  background: var(--glass);
  border: 1px solid var(--glass-b);
  border-radius: 10px;
  padding: 12px 16px;
  margin-bottom: 8px;
  transition: border-color 0.2s, transform 0.2s;
  will-change: transform;
}}
.tai-bullet-card:hover {{ border-color: rgba(232,130,154,0.3); transform: translateX(3px); }}
.tai-bullet-num {{
  font-size: 0.65rem; font-weight: 800; color: var(--accent);
  background: rgba(232,130,154,0.12); border-radius: 6px;
  padding: 2px 6px; flex-shrink: 0; margin-top: 2px; letter-spacing: 0.05em;
}}
.tai-action-card {{
  display: flex; gap: 12px; align-items: flex-start;
  background: var(--glass);
  border: 1px solid var(--glass-b);
  border-left: 3px solid var(--accent);
  border-radius: 0 10px 10px 0;
  padding: 14px 16px; margin-bottom: 10px;
  transition: transform 0.2s, border-color 0.2s;
  will-change: transform;
  color: var(--accent);
}}
.tai-action-card:hover {{ transform: translateX(4px); border-left-color: #963050; }}
.tai-action-flagged {{ border-left-color: #963030 !important; background: rgba(150,48,48,0.05) !important; color: #963030 !important; }}
.tai-sent-row {{
  display: flex; align-items: center; gap: 12px;
  background: var(--glass);
  border: 1px solid var(--glass-b);
  border-radius: 10px; padding: 12px 16px; margin-bottom: 8px;
  transition: border-color 0.2s, transform 0.2s;
  will-change: transform;
}}
.tai-sent-row:hover {{ border-color: rgba(232,130,154,0.25); transform: translateX(3px); }}
.tai-sent-badge {{
  font-size: 0.65rem; font-weight: 700; letter-spacing: 0.08em;
  padding: 4px 10px; border-radius: 999px;
}}
.tai-sent-positive {{ background: rgba(45,122,85,0.10); color: #2D7A55; border: 1px solid rgba(45,122,85,0.25); }}
.tai-sent-neutral  {{ background: rgba(120,80,64,0.08); color: #7A5040; border: 1px solid rgba(120,80,64,0.20); }}
.tai-sent-negative {{ background: rgba(176,64,64,0.10); color: #963030; border: 1px solid rgba(176,64,64,0.25); }}
.tai-spk-row {{
  display: flex; align-items: center; gap: 14px;
  background: var(--glass);
  border: 1px solid var(--glass-b);
  border-radius: 10px; padding: 14px 16px; margin-bottom: 10px;
  transition: border-color 0.2s;
}}
.tai-spk-row:hover {{ border-color: rgba(190,64,96,0.30); }}
.tai-insight-chip {{
  background: var(--glass);
  border: 1px solid var(--glass-b);
  border-radius: 10px; padding: 12px 16px;
  min-width: 100px; flex: 1;
}}
.tai-nemawashi-pill {{
  display: inline-block;
  background: rgba(190,64,96,0.08);
  border: 1px solid rgba(190,64,96,0.25);
  border-radius: 999px; padding: 5px 14px;
  font-size: 0.82rem; color: #BE4060;
  font-family: 'Noto Sans JP', sans-serif;
  margin: 0 6px 8px 0;
}}
.tai-sig-high {{
  background: rgba(176,64,64,0.07); border-left: 3px solid #963030;
  border-radius: 0 10px 10px 0; padding: 12px 16px; margin-bottom: 10px;
  color: #3C2416;
}}
.tai-sig-med {{
  background: rgba(152,104,32,0.07); border-left: 3px solid #986820;
  border-radius: 0 10px 10px 0; padding: 12px 16px; margin-bottom: 10px;
  color: #3C2416;
}}
.tai-pii-pill {{
  display: inline-flex; align-items: center; gap: 6px;
  background: rgba(45,122,85,0.10); border: 1px solid rgba(45,122,85,0.30);
  border-radius: 999px; padding: 5px 14px;
  font-size: 0.73rem; color: #2D7A55; font-weight: 500; margin-bottom: 14px;
}}
.tai-panel {{
  background: rgba(255,254,251,0.85);
  border: 1px solid rgba(60,36,22,0.10);
  border-radius: var(--r);
  padding: 20px;
}}
</style>

<div class="tai-results">

{pii_html}

<div class="tai-tiles">{tiles}</div>

<div class="tai-health">
  <div class="tai-health-left">
    {_health_ring(score, hc)}
  </div>
  <div class="tai-health-right">
    <div class="tai-health-title">Meeting Health Breakdown</div>
    {hbars}
  </div>
</div>

<div class="tai-tab-bar">
  <button class="tai-tab active" onclick="taiTab(this,'sum')">📝 Summary</button>
  <button class="tai-tab" onclick="taiTab(this,'act')">✅ Actions</button>
  <button class="tai-tab" onclick="taiTab(this,'sent')">🌸 Sentiment</button>
  <button class="tai-tab" onclick="taiTab(this,'spk')">🎤 Speakers</button>
  <button class="tai-tab" onclick="taiTab(this,'ins')">{features.get('insight_tab_label','🌐 Insights')}</button>
</div>

<div class="tai-panel">
  <div id="tai-sum" class="tai-tab-content active">{sum_html}</div>
  <div id="tai-act" class="tai-tab-content">{act_html}</div>
  <div id="tai-sent" class="tai-tab-content">{sent_html}</div>
  <div id="tai-spk" class="tai-tab-content">{spk_html}</div>
  <div id="tai-ins" class="tai-tab-content">{ins_html}</div>
</div>

</div>

<script>
function taiTab(btn, id) {{
  document.querySelectorAll('.tai-tab').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tai-tab-content').forEach(c => c.classList.remove('active'));
  btn.classList.add('active');
  var el = document.getElementById('tai-' + id);
  if(el) el.classList.add('active');
}}
</script>
"""


# ── Meeting health score (kept for backward-compat, renderer uses inline _health()) ──
def compute_health_score(R: dict) -> dict:
    sentiment = R.get("sentiment", [])
    if sentiment:
        weights = {"positive": 1.0, "neutral": 0.6, "negative": 0.1}
        avg = sum(weights.get(s.get("score","neutral").lower(), 0.5)
                  for s in sentiment) / len(sentiment)
        s_pts = round(avg * 30)
    else:
        s_pts = 15

    items = R.get("action_items", [])
    if not items:
        a_pts = 10
    else:
        verified      = [i for i in items if not i.get("hallucination_flag", False)]
        with_owner    = sum(1 for i in verified if i.get("owner","TBD") not in ("TBD","Unknown",""))
        with_deadline = sum(1 for i in verified if i.get("deadline","TBD") not in ("TBD","N/A",""))
        clarity = (with_owner + with_deadline) / (2 * len(items))
        a_pts = round(clarity * 25)

    soft     = R.get("soft_rejections", {})
    risk     = soft.get("risk_level", "NONE")
    risk_pts = {"NONE": 25, "MINIMAL": 20, "LOW": 15, "MEDIUM": 8, "HIGH": 0}
    r_pts    = risk_pts.get(risk, 25)

    verification = R.get("verification", {})
    hall_rate    = verification.get("overall_hallucination_risk", 0)
    h_pts        = round((1 - hall_rate) * 20)

    score = min(s_pts + a_pts + r_pts + h_pts, 100)

    if score >= 80:
        label, color, bg, border = "Productive Meeting", "#486858", "#EDF3EF", "#A8C8B8"
    elif score >= 60:
        label, color, bg, border = "Mostly Aligned",    "#986820", "#FAF0E0", "#D9C090"
    elif score >= 40:
        label, color, bg, border = "Needs Follow-up",   "#C87030", "#FDF0EA", "#E8C090"
    else:
        label, color, bg, border = "High Risk",         "#B04040", "#FAF0F0", "#E8A0A0"

    return {
        "score":     score,
        "label":     label,
        "color":     color,
        "bg":        bg,
        "border":    border,
        "breakdown": {
            "sentiment":      s_pts,
            "action_clarity": a_pts,
            "soft_rejection": r_pts,
            "hallucination":  h_pts,
        },
    }


# ────────────────────────────────────────────────────────────────────────────
# RESULTS DISPLAY
# ────────────────────────────────────────────────────────────────────────────
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
        "<hr style='border:none;border-top:1px solid rgba(60,36,22,0.10);margin:1.6rem 0 1rem;'/>",
        unsafe_allow_html=True,
    )

    st.markdown(
        build_results_html(R, language, features, pii_rep),
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

    from utils import build_export_json, export_filename
    exp = build_export_json(st.session_state.current_transcript, language, R)
    st.download_button(
        "⬇ Export JSON",
        data=exp.encode(),
        file_name=export_filename(language),
        mime="application/json",
    )

    if EVAL_AVAILABLE:
        with st.expander("📊 Accuracy Evaluation · Ground Truth Comparison"):
            st.markdown(
                "<div style='font-size:0.82rem;color:#7A5040;margin-bottom:1rem'>"
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
                with c1: st.metric("Overall", f"{overall}%")
                with c2: st.metric("ROUGE-1", report.get("summary",{}).get("avg_rouge1_f1","—"))
                with c3: st.metric("Action F1", report.get("action_items",{}).get("f1","—"))
                with c4: st.metric("Sentiment", report.get("sentiment",{}).get("soft_accuracy","—"))
                if "japan_insights" in report:
                    ji_r = report["japan_insights"]
                    c5, c6, c7 = st.columns(3)
                    with c5: st.metric("Keigo", ji_r["keigo"].get("grade","—"))
                    with c6: st.metric("Nemawashi P", ji_r["nemawashi"].get("precision","—"))
                    with c7: st.metric("Code-switch", ji_r["code_switching"].get("grade","—"))
                with st.expander("Full report (JSON)"):
                    st.json(report)

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

# ────────────────────────────────────────────────────────────────────────────
# FOOTER — split into two st.markdown() calls to avoid Streamlit truncation
# ────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.tai-footer { margin-top:3rem; border-top:1px solid #EDE0D8; padding:2.5rem 0 2rem; }
.tai-footer-card { background:linear-gradient(135deg,#FFFEFB 0%,#FEF6F8 100%); border:1px solid #EDE0D8; border-radius:16px; padding:2rem 2.5rem; max-width:860px; margin:0 auto; box-shadow:0 2px 16px rgba(217,96,128,0.06); }
.tai-footer-name  { font-size:1.25rem; font-weight:700; color:#3C2416; letter-spacing:-0.01em; margin-bottom:0.25rem; }
.tai-footer-title { font-size:0.78rem; color:#D96080; font-weight:500; letter-spacing:0.04em; margin-bottom:1rem; }
.tai-footer-bio   { font-size:0.82rem; color:#7A5040; line-height:1.75; margin-bottom:1.4rem; border-left:3px solid #F2B0C0; padding-left:1rem; }
.tai-footer-links { display:flex; flex-wrap:wrap; gap:10px; margin-bottom:1.4rem; }
.tai-footer-link  { display:inline-flex; align-items:center; gap:6px; padding:6px 14px; border-radius:999px; font-size:0.75rem; font-weight:500; text-decoration:none; border:1px solid; transition:box-shadow 0.2s,transform 0.2s; }
.tai-footer-link:hover { transform:translateY(-2px); box-shadow:0 4px 12px rgba(217,96,128,0.15); }
.tai-footer-link-gh   { background:#F8F4FF; color:#3C2416; border-color:#C8A8C8; }
.tai-footer-link-li   { background:#EFF6FF; color:#1A56A8; border-color:#93C5FD; }
.tai-footer-link-hf   { background:#FFF7ED; color:#C05A00; border-color:#FDB97B; }
.tai-footer-link-mail { background:#FEF6F8; color:#BE4060; border-color:#F2B0C0; }
.tai-footer-link-repo { background:#F0FDF4; color:#166534; border-color:#86EFAC; }
.tai-footer-divider   { border:none; border-top:1px solid #EDE0D8; margin:1.2rem 0; }
.tai-footer-stack { display:flex; flex-wrap:wrap; gap:7px; margin-bottom:1.2rem; }
.tai-footer-chip  { font-size:0.68rem; padding:3px 10px; border-radius:999px; background:#FEF3EC; color:#7A5040; border:1px solid #E5D0C4; font-weight:500; }
.tai-footer-bottom { display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px; font-size:0.71rem; color:#C8A898; }
.tai-footer-stat  { display:inline-flex; align-items:center; gap:5px; background:#FEF6F8; border:1px solid #F2B0C0; border-radius:999px; padding:3px 10px; font-size:0.7rem; color:#D96080; font-weight:600; }
@media (max-width:768px) { .tai-footer-card { padding:1.4rem 1.2rem; } .tai-footer-bottom { flex-direction:column; align-items:flex-start; } }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="tai-footer">
  <div class="tai-footer-card">
    <div class="tai-footer-name">Kunal Bisht</div>
    <div class="tai-footer-title">AI Engineer &middot; LLM Systems &amp; RAG Pipelines &middot; Multilingual NLP</div>
    <div class="tai-footer-bio">
      I build AI to turn real problems into actual solutions &mdash; not proof-of-concepts that never ship.
      My focus is NLP, RAG pipelines, and language systems: work where meaning matters and silent failures cost money.
      TranscriptAI started because I kept forgetting my meetings &mdash; it became a trilingual intelligence platform
      rebuilt five times until accuracy went from 22% to 93%.
    </div>
    <div class="tai-footer-links">
      <a class="tai-footer-link tai-footer-link-gh"   href="https://github.com/aiKunalBisht" target="_blank">GitHub</a>
      <a class="tai-footer-link tai-footer-link-li"   href="https://linkedin.com/in/kunalhere" target="_blank">LinkedIn</a>
      <a class="tai-footer-link tai-footer-link-hf"   href="https://huggingface.co/spaces/KunalTheBeast/TranscriptAI" target="_blank">Live Demo</a>
      <a class="tai-footer-link tai-footer-link-repo" href="https://github.com/aiKunalBisht/Transcript-ai" target="_blank">Source Code</a>
      <a class="tai-footer-link tai-footer-link-mail" href="mailto:kunalbisht909@gmail.com">kunalbisht909@gmail.com</a>
    </div>
    <div class="tai-footer-divider"></div>
    <div class="tai-footer-stack">
      <span class="tai-footer-chip">Python</span>
      <span class="tai-footer-chip">FastAPI</span>
      <span class="tai-footer-chip">LangChain</span>
      <span class="tai-footer-chip">ChromaDB</span>
      <span class="tai-footer-chip">Groq API</span>
      <span class="tai-footer-chip">Ollama</span>
      <span class="tai-footer-chip">MeCab</span>
      <span class="tai-footer-chip">HuggingFace</span>
      <span class="tai-footer-chip">MLflow</span>
      <span class="tai-footer-chip">Docker</span>
      <span class="tai-footer-chip">Streamlit</span>
      <span class="tai-footer-chip">RAG</span>
    </div>
    <div class="tai-footer-bottom">
      <div style="display:flex;gap:8px;flex-wrap:wrap;">
        <span class="tai-footer-stat">93.8% eval accuracy</span>
        <span class="tai-footer-stat">v5 &middot; 5 rebuilds</span>
        <span class="tai-footer-stat">JA &middot; HI &middot; EN &middot; Mixed</span>
        <span class="tai-footer-stat">APPI compliant</span>
      </div>
      <div>Bengaluru, Karnataka, India &middot; Open to Remote / Relocation</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)