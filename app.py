"""
app.py — TranscriptAI  v7.6
Japanese Business Intelligence Platform

FIXES v7.6:
  - Export banner extracted from Summary tab and made persistent across ALL tabs
  - Navbar transparent and polished styling
  - Top padding (~1cm) removed for flush alignment
  - Footer display guaranteed during background execution
"""
import sys
import os
from dotenv import load_dotenv
import pathlib
load_dotenv(dotenv_path=pathlib.Path(__file__).resolve().parent / ".env")

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

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TranscriptAI · Speech & Meeting Analyzer",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── SEO head tags ─────────────────────────────────────────────────────────────
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

# ── CSS ───────────────────────────────────────────────────────────────────────
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
    --purple:       #7D4E8A;
    --purple-bg:    #F5EEF8;
    --purple-border:#D0B0C8;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', 'Noto Sans JP', sans-serif !important;
    color: var(--ink) !important;
    -webkit-font-smoothing: antialiased;
    scroll-behavior: smooth;
}

/* ── Remove dead space at top (1 cm requested fix) ───────────────────────── */
.block-container {
    background: transparent !important;
    padding-top: 0rem !important; /* Reduced to remove space */
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
    max-width: 1200px !important;
}
[data-testid="stAppViewBlockContainer"] {
    padding-top: 0rem !important;
}
.stMainBlockContainer {
    padding-top: 0rem !important;
}
/* Streamlit injects a spacer div at the very top — zero it out */
.stApp > header + div,
.stApp > section > div:first-child {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

/* ── Sidebar ────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #FDF8F5 !important;
    border-right: 1px solid var(--border) !important;
    background-image:
        radial-gradient(circle at 50% 0%,   rgba(217,96,128,0.07) 0%, transparent 55%),
        radial-gradient(circle at 100% 100%, rgba(184,120,48,0.04) 0%, transparent 50%) !important;
    box-shadow: 2px 0 20px rgba(60,36,22,0.06) !important;
}
/* Hide ONLY the auto page-nav — JS physically removes it from DOM below */
[data-testid="stSidebarNav"],
[data-testid="stSidebarNavItems"],
[data-testid="stSidebarNavLink"] {
    display: none !important;
    height: 0 !important;
    overflow: hidden !important;
    visibility: hidden !important;
}
/* Hide the close button INSIDE the sidebar only.
   NEVER touch collapsedControl — that is the expand arrow on the main page.
   Without it, a collapsed sidebar has no way to reopen. */
[data-testid="stSidebarCollapseButton"] {
    display: none !important;
    visibility: hidden !important;
}
[data-testid="stSidebar"] * { color: var(--ink) !important; }
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label { color: var(--ink-mid) !important; }
/* ── App background ──────────────────────────────────────────────────────── */
.stApp {
    background-color: var(--washi) !important;
    background-image:
        radial-gradient(circle at 92% 8%,  rgba(217,96,128,0.09) 0%, transparent 45%),
        radial-gradient(circle at 8%  92%, rgba(232,128,96,0.07) 0%, transparent 45%),
        radial-gradient(circle at 50% 50%, rgba(184,120,48,0.03) 0%, transparent 60%) !important;
}

/* ── Hide Streamlit chrome ───────────────────────────────────────────────── */
[data-testid="stToolbar"],
[data-testid="stHeader"],
[data-testid="stDecoration"],
header[data-testid="stHeader"] {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
}

/* ── File uploader ───────────────────────────────────────────────────────── */
[data-testid="stFileUploader"],
[data-testid="stFileUploader"] > div,
[data-testid="stFileUploader"] section,
[data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploadDropzone"] {
    background-color: var(--surface) !important;
    border: 1.5px dashed var(--border-mid) !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploadDropzone"] {
    background-color: var(--sakura-pale) !important;
}
[data-testid="stFileUploader"] * { color: var(--ink-mid) !important; background: transparent !important; }
[data-testid="stFileUploaderDropzone"] svg { fill: var(--sakura-light) !important; }
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploader"] button {
    background-color: var(--sakura-bg) !important;
    color: var(--sakura-deep) !important;
    border: 1px solid var(--sakura-light) !important;
    border-radius: 6px !important;
    padding: 0.3rem 1rem !important;
    font-weight: 500 !important;
}

/* ── Textarea ────────────────────────────────────────────────────────────── */
textarea,
.stTextArea textarea,
div[data-baseweb="textarea"] textarea {
    background-color: var(--surface) !important;
    border: 1.5px solid var(--border-mid) !important;
    border-radius: 10px !important;
    color: var(--ink) !important;
    font-family: 'Noto Sans JP', 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
    line-height: 1.7 !important;
    caret-color: var(--sakura) !important;
}
textarea:focus, div[data-baseweb="textarea"]:focus-within textarea {
    border-color: var(--sakura-light) !important;
    box-shadow: 0 0 0 3px rgba(217,96,128,0.10) !important;
    outline: none !important;
}

/* ── Select ──────────────────────────────────────────────────────────────── */
div[data-baseweb="select"] > div { background-color: var(--surface) !important; border-color: var(--border-mid) !important; color: var(--ink) !important; border-radius: 8px !important; }
li[role="option"] { background: var(--surface) !important; color: var(--ink) !important; }
li[role="option"]:hover { background: var(--sakura-pale) !important; }
[data-testid="stToggle"] input:checked + div { background-color: var(--sakura) !important; }

/* ── Buttons ─────────────────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, var(--sakura) 0%, var(--sakura-deep) 100%) !important;
    color: #FFFDFB !important; border: none !important; border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important; font-weight: 600 !important;
    font-size: 0.86rem !important; padding: 0.52rem 1.4rem !important;
    transition: all 0.2s !important;
    box-shadow: 0 2px 8px rgba(217,96,128,0.30) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, var(--sakura-deep) 0%, #A03050 100%) !important;
    box-shadow: 0 6px 20px rgba(217,96,128,0.40) !important;
    transform: translateY(-1px) !important;
}
[data-testid="stDownloadButton"] button {
    background-color: transparent !important; color: var(--sakura-deep) !important;
    border: 1.5px solid var(--sakura-light) !important; box-shadow: none !important;
}
[data-testid="stDownloadButton"] button:hover { background-color: var(--sakura-bg) !important; }

/* ── Progress ────────────────────────────────────────────────────────────── */
.stProgress > div > div { background-color: var(--border) !important; border-radius: 999px !important; }
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--sakura), var(--peach), var(--gold)) !important;
    border-radius: 999px !important; box-shadow: 0 0 8px rgba(217,96,128,0.35) !important;
}

/* ── Streamlit native tabs (keep styled for fallback) ─────────────────────── */
[data-testid="stTabs"] [role="tablist"] { border-bottom: 1px solid var(--border) !important; background: transparent !important; }
[data-testid="stTabs"] button { background: transparent !important; color: var(--ink-soft) !important; border: none !important; border-bottom: 2px solid transparent !important; font-size: 0.83rem !important; font-weight: 400 !important; padding: 0.55rem 1rem !important; margin-bottom: -1px !important; border-radius: 4px 4px 0 0 !important; }
[data-testid="stTabs"] button:hover { color: var(--sakura) !important; }
[data-testid="stTabs"] button[aria-selected="true"] { color: var(--sakura-deep) !important; border-bottom: 2px solid var(--sakura) !important; font-weight: 600 !important; }

/* ── Expander ────────────────────────────────────────────────────────────── */
[data-testid="stExpander"] { border: 1px solid var(--border) !important; border-radius: 8px !important; background: var(--surface) !important; }
[data-testid="stExpander"] summary { color: var(--ink-mid) !important; font-size: 0.85rem !important; }

/* ── Misc ─────────────────────────────────────────────────────────────────── */
[data-testid="stSpinner"] > div { border-top-color: var(--sakura) !important; }
.stAlert { border-radius: 8px !important; }
.stMarkdown p, .stMarkdown li, .stMarkdown span { color: var(--ink-mid) !important; }
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-thumb { background: linear-gradient(180deg, var(--sakura-light), var(--sakura)); border-radius: 999px; }

/* ── Cards ───────────────────────────────────────────────────────────────── */
.card {
    background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
    padding: 1.2rem 1.4rem; margin-bottom: 0.8rem;
    transition: border-color 0.25s, box-shadow 0.25s, transform 0.2s;
    box-shadow: 0 1px 3px rgba(60,36,22,0.04);
}
.card:hover { border-color: var(--sakura-light); box-shadow: 0 4px 20px rgba(217,96,128,0.12); transform: translateY(-1px); }

.metric-card {
    background: linear-gradient(135deg, var(--surface) 0%, var(--sakura-pale) 100%);
    border: 1px solid var(--border); border-top: 3px solid var(--sakura);
    border-radius: 10px; padding: 1.2rem 0.8rem; text-align: center;
    transition: box-shadow 0.25s, transform 0.2s; box-shadow: 0 1px 3px rgba(60,36,22,0.04);
    min-height: 90px;
}
.metric-card:hover { box-shadow: 0 6px 20px rgba(217,96,128,0.13); transform: translateY(-2px); }
.metric-value { font-size: 1.85rem; font-weight: 700; color: var(--sakura-deep); line-height: 1.1; letter-spacing: -0.02em; }
.metric-label { font-size: 0.59rem; color: var(--ink-faint); text-transform: uppercase; letter-spacing: 0.13em; margin-top: 0.4rem; font-weight: 600; }

.sh {
    font-size: 0.67rem; font-weight: 700; color: var(--ink-soft); letter-spacing: 0.16em;
    text-transform: uppercase; margin-bottom: 0.9rem; padding-bottom: 0.45rem;
    border-bottom: 2px solid var(--border);
    display: flex; align-items: center; gap: 0.5rem;
}

/* ── Other component classes ─────────────────────────────────────────────── */
.action-row {
    display: flex; align-items: flex-start; gap: 0.85rem;
    background: var(--surface); border: 1px solid var(--border); border-left: 4px solid var(--sakura);
    border-radius: 0 12px 12px 0; padding: 0.95rem 1.2rem; margin-bottom: 0.65rem;
    transition: border-color 0.25s, box-shadow 0.25s, transform 0.2s;
}
.action-row:hover { border-left-color: var(--sakura-deep); box-shadow: 0 4px 16px rgba(217,96,128,0.12); transform: translateX(2px); }
.action-row.flagged { border-left-color: var(--red); background: var(--red-bg); }

.sentiment-row { display: flex; align-items: center; gap: 1rem; background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 0.85rem 1.1rem; margin-bottom: 0.55rem; transition: background 0.25s, border-color 0.25s, transform 0.2s; }
.sentiment-row:hover { background: var(--sakura-pale); border-color: var(--sakura-light); transform: translateX(2px); }

.badge { display: inline-block; padding: 0.22rem 0.8rem; border-radius: 999px; font-size: 0.68rem; font-weight: 700; letter-spacing: 0.07em; text-transform: uppercase; }
.badge-positive { background: var(--green-bg); color: var(--green); border: 1px solid rgba(72,104,88,0.2); }
.badge-neutral  { background: var(--peach-bg); color: var(--ink-mid); border: 1px solid rgba(120,80,64,0.15); }
.badge-negative { background: var(--red-bg);   color: var(--red); border: 1px solid rgba(176,64,64,0.2); }

.signal-high { background: var(--red-bg); border-left: 3px solid var(--red); border-radius: 0 10px 10px 0; padding: 0.85rem 1.1rem; margin-bottom: 0.6rem; }
.signal-medium { background: var(--amber-bg); border-left: 3px solid var(--amber); border-radius: 0 10px 10px 0; padding: 0.85rem 1.1rem; margin-bottom: 0.6rem; }
.signal-low { background: var(--sakura-pale); border-left: 3px solid var(--sakura-light); border-radius: 0 10px 10px 0; padding: 0.85rem 1.1rem; margin-bottom: 0.6rem; }
.signal-phrase { font-weight: 600; font-size: 0.9rem; font-family: 'Noto Sans JP', sans-serif; color: var(--ink); }
.signal-reading { font-size: 0.79rem; color: var(--ink-mid); margin-top: 0.2rem; }
.signal-exp { font-size: 0.77rem; color: var(--ink-soft); margin-top: 0.4rem; line-height: 1.6; }

.spk-bar-bg { background: var(--border); border-radius: 999px; height: 7px; overflow: hidden; margin-top: 0.4rem; }
.spk-bar-fill { height: 100%; border-radius: 999px; transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1); }

.pii-pill { display: inline-flex; align-items: center; gap: 0.4rem; background: var(--green-bg); border: 1px solid #A8C8B5; border-radius: 999px; padding: 0.28rem 0.9rem; font-size: 0.74rem; color: var(--green); font-weight: 500; margin-bottom: 1rem; }

.risk-pill { display: inline-block; padding: 0.28rem 0.9rem; border-radius: 999px; font-size: 0.71rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; }
.risk-HIGH    { background: var(--red-bg);    color: var(--red); }
.risk-MEDIUM  { background: var(--amber-bg);  color: var(--amber); }
.risk-LOW     { background: var(--sakura-pale); color: var(--sakura-deep); }
.risk-MINIMAL { background: var(--peach-bg);  color: var(--ink-soft); }
.risk-NONE    { background: var(--green-bg);  color: var(--green); }

.prev-session-card { background: var(--surface-warm); border: 1px solid var(--border-mid); border-left: 3px solid var(--gold); border-radius: 0 10px 10px 0; padding: 1rem 1.3rem; margin-top: 0.5rem; margin-bottom: 0.5rem; }
.prev-session-header { font-size: 0.68rem; font-weight: 600; color: var(--gold); letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 0.55rem; }
.prev-session-bullet { font-size: 0.83rem; color: var(--ink-mid); line-height: 1.65; margin-bottom: 0.25rem; padding-left: 0.9rem; position: relative; }
.prev-session-bullet::before { content: "·"; position: absolute; left: 0; color: var(--gold); font-weight: 700; }

/* ── Bilingual summary styles ─────────────────────────────────────────────── */
.tai-lang-label { display: inline-block; font-size: 0.62rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; padding: 2px 8px; border-radius: 999px; margin-right: 6px; margin-bottom: 6px; }
.tai-lang-ja { background: rgba(217,96,128,0.12); color: #BE4060; border: 1px solid rgba(217,96,128,0.3); }
.tai-lang-en { background: rgba(72,104,88,0.12); color: #2D7A55; border: 1px solid rgba(72,104,88,0.3); }
.tai-bilingual-block { border: 1px solid rgba(60,36,22,0.10); border-radius: 10px; padding: 12px 16px; margin-bottom: 10px; background: rgba(255,254,251,0.8); }
.tai-bilingual-ja { font-family: 'Noto Sans JP', sans-serif; font-size: 0.88rem; color: #3C2416; line-height: 1.8; margin-bottom: 6px; }
.tai-bilingual-en { font-size: 0.80rem; color: #7A5040; line-height: 1.65; font-style: italic; border-top: 1px dashed rgba(60,36,22,0.12); padding-top: 6px; margin-top: 4px; }

/* ── Summary box ─────────────────────────────────────────────────────────── */
.tai-summary-box { max-height: 220px !important; overflow-y: auto !important; scrollbar-width: thin !important; }
.tai-summary-box p { font-size: 0.85rem !important; line-height: 1.75 !important; }

/* ── Responsive ──────────────────────────────────────────────────────────── */
@media (max-width: 768px) {
    .block-container { padding-left: 0.75rem !important; padding-right: 0.75rem !important; }
    [data-testid="stHorizontalBlock"] { flex-direction: column !important; gap: 0.5rem !important; }
    [data-testid="column"] { width: 100% !important; min-width: 100% !important; flex: 1 1 100% !important; }
    .metric-card { min-height: 70px !important; padding: 0.8rem 0.4rem !important; }
    .metric-value { font-size: 1.3rem !important; }
    .stButton > button { padding: 0.65rem 1rem !important; min-height: 44px !important; }
    h1 { font-size: 1.5rem !important; }
}
@media (max-width: 480px) {
    .metric-value { font-size: 1.15rem !important; }
    h1 { font-size: 1.25rem !important; }
}

/* ═══════════════════════════════════════════════════════════════
   Results panel CSS 
   ═══════════════════════════════════════════════════════════════ */

:root {
  --glass:    rgba(60,36,22,0.04);
  --glass-b:  rgba(60,36,22,0.10);
  --r:        14px;
}

/* Metric tiles */
.tai-results { font-family:'DM Sans','Noto Sans JP',sans-serif; color:var(--ink); padding:0 0 2rem; }
.tai-tiles { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:16px; }
@media(max-width:768px) { .tai-tiles { grid-template-columns:repeat(2,1fr); } }
.tai-tile { background:var(--glass); border:1px solid var(--glass-b); border-radius:var(--r); padding:16px 12px; text-align:center; transition:transform 0.2s,border-color 0.25s; min-height:90px; }
.tai-tile:hover { transform:translateY(-3px); border-color:rgba(190,64,96,0.30); box-shadow:0 12px 32px rgba(190,64,96,0.10); }
.tai-tile-icon { font-size:1.3rem; margin-bottom:4px; }
.tai-tile-val  { font-size:1.6rem; font-weight:800; color:var(--sakura-deep); letter-spacing:-0.02em; line-height:1; }
.tai-tile-lbl  { font-size:0.6rem; color:var(--ink-soft); text-transform:uppercase; letter-spacing:0.12em; margin-top:4px; font-weight:600; }

/* ── New Responsive Health Layout ───────────────────────────────── */
.tai-health { 
    display:grid; 
    grid-template-columns: 240px 1fr; /* Metrics on left, content on right */
    gap: 20px;
    background: transparent;
    margin-bottom: 24px; 
}
@media(max-width:900px) { 
    .tai-health { grid-template-columns:1fr; } 
}

/* Left: Centered Metrics */
.tai-health-left  { 
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    align-content: start;
}
.tai-health-left .tai-tile {
    min-height: 80px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

/* Right: Main Content */
.tai-health-right { 
    background:var(--surface); 
    border:1px solid var(--glass-b); 
    border-radius:var(--r); 
    padding:24px; 
}
.tai-health-title { font-size:0.6rem; font-weight:700; color:var(--sakura-deep); letter-spacing:0.15em; text-transform:uppercase; margin-bottom:14px; }

/* ── Radio tab system ───────────────────────────────────── */
.tai-radio-tabs { display:contents; }
.tai-radio-tabs input[type="radio"] { display:none; }
.tai-tab-bar { display:flex; gap:4px; border-bottom:1px solid rgba(60,36,22,0.12); margin-bottom:0; overflow-x:auto; scrollbar-width:none; }
.tai-tab-bar::-webkit-scrollbar { display:none; }
.tai-tab-label {
  padding:10px 16px; font-size:0.8rem; font-weight:500; color:var(--ink-soft);
  background:none; border-bottom:2px solid transparent;
  cursor:pointer; white-space:nowrap; transition:color 0.2s,border-color 0.2s;
  margin-bottom:-1px; user-select:none; display:inline-block;
}
.tai-tab-label:hover { color:var(--sakura-deep); background:rgba(190,64,96,0.04); border-radius:4px 4px 0 0; }
.tai-panel { background:rgba(255,254,251,0.90); border:1px solid rgba(60,36,22,0.10); border-radius:0 0 var(--r) var(--r); padding:20px; }
.tai-tab-content { display:none; animation:tFadeIn 0.22s ease; }
@keyframes tFadeIn { from{opacity:0;transform:translateY(5px)} to{opacity:1;transform:none} }

#tai-radio-sum:checked  ~ .tai-panel #tai-sum  { display:block; }
#tai-radio-act:checked  ~ .tai-panel #tai-act  { display:block; }
#tai-radio-sent:checked ~ .tai-panel #tai-sent { display:block; }
#tai-radio-spk:checked  ~ .tai-panel #tai-spk  { display:block; }
#tai-radio-ins:checked  ~ .tai-panel #tai-ins  { display:block; }

#tai-radio-sum:checked  ~ .tai-tab-bar label[for="tai-radio-sum"],
#tai-radio-act:checked  ~ .tai-tab-bar label[for="tai-radio-act"],
#tai-radio-sent:checked ~ .tai-tab-bar label[for="tai-radio-sent"],
#tai-radio-spk:checked  ~ .tai-tab-bar label[for="tai-radio-spk"],
#tai-radio-ins:checked  ~ .tai-tab-bar label[for="tai-radio-ins"] {
  color:var(--sakura-deep); border-bottom-color:var(--sakura); font-weight:600;
  background:rgba(190,64,96,0.04);
}

/* Content components */
.tai-section-label { font-size:0.62rem; font-weight:700; color:var(--ink-soft); letter-spacing:0.16em; text-transform:uppercase; border-bottom:1px solid rgba(60,36,22,0.12); padding-bottom:8px; margin-bottom:12px; margin-top:8px; }
.tai-summary-box { background:var(--glass); border:1px solid var(--glass-b); border-left:3px solid var(--sakura-deep); border-radius:0 var(--r) var(--r) 0; padding:18px 20px; margin-bottom:16px; line-height:1.85; max-height:220px; overflow-y:auto; scrollbar-width:thin; }
.tai-summary-label { font-size:0.62rem; font-weight:700; color:var(--sakura-deep); letter-spacing:0.12em; text-transform:uppercase; margin-bottom:10px; }
.tai-bilingual-block { border:1px solid rgba(60,36,22,0.10); border-radius:10px; padding:12px 16px; margin-bottom:10px; background:rgba(255,254,251,0.8); }
.tai-bilingual-ja { font-family:'Noto Sans JP',sans-serif; font-size:0.88rem; color:#3C2416; line-height:1.8; margin-bottom:6px; }
.tai-bilingual-en { font-size:0.80rem; color:#7A5040; line-height:1.65; font-style:italic; border-top:1px dashed rgba(60,36,22,0.12); padding-top:6px; margin-top:4px; }
.tai-lang-label { display:inline-block; font-size:0.62rem; font-weight:700; letter-spacing:0.1em; text-transform:uppercase; padding:2px 8px; border-radius:999px; margin-right:6px; margin-bottom:6px; }
.tai-lang-ja { background:rgba(217,96,128,0.12); color:#BE4060; border:1px solid rgba(217,96,128,0.3); }
.tai-lang-en { background:rgba(72,104,88,0.12); color:#2D7A55; border:1px solid rgba(72,104,88,0.3); }
.tai-bullet-card { display:flex; align-items:flex-start; gap:12px; background:var(--glass); border:1px solid var(--glass-b); border-radius:10px; padding:12px 16px; margin-bottom:8px; transition:border-color 0.2s,transform 0.2s; }
.tai-bullet-card:hover { border-color:rgba(232,130,154,0.3); transform:translateX(3px); }
.tai-bullet-num { font-size:0.65rem; font-weight:800; color:var(--sakura-deep); background:rgba(232,130,154,0.12); border-radius:6px; padding:2px 6px; flex-shrink:0; margin-top:2px; }
.tai-action-card { display:flex; gap:12px; align-items:flex-start; background:var(--glass); border:1px solid var(--glass-b); border-left:3px solid var(--sakura-deep); border-radius:0 10px 10px 0; padding:14px 16px; margin-bottom:10px; transition:transform 0.2s; }
.tai-action-card:hover { transform:translateX(4px); }
.tai-action-card-icon { font-size:1.1rem; padding-top:2px; color:var(--sakura-deep); }
.tai-action-flagged { border-left-color:#963030 !important; background:rgba(150,48,48,0.05) !important; }
.tai-action-flagged .tai-action-card-icon { color:#963030; }
.tai-sent-row { display:flex; align-items:center; gap:12px; background:var(--glass); border:1px solid var(--glass-b); border-radius:10px; padding:12px 16px; margin-bottom:8px; transition:border-color 0.2s,transform 0.2s; }
.tai-sent-row:hover { border-color:rgba(232,130,154,0.25); transform:translateX(3px); }
.tai-sent-badge { font-size:0.65rem; font-weight:700; letter-spacing:0.08em; padding:4px 10px; border-radius:999px; }
.tai-sent-positive { background:rgba(45,122,85,0.10); color:#2D7A55; border:1px solid rgba(45,122,85,0.25); }

/* Gijiroku preview card inside summary tab */
.tai-gijiroku-card { border:1px solid #D0B0C8; border-radius:14px; overflow:hidden; background:#FDFAFF; margin-top:20px; }
.tai-gijiroku-header { background:linear-gradient(135deg,#7D4E8A,#A06CB5); padding:14px 18px; display:flex; justify-content:space-between; align-items:flex-start; }
.tai-gijiroku-section { font-size:0.58rem; font-weight:700; color:#7D4E8A; letter-spacing:0.15em; text-transform:uppercase; margin-bottom:8px; padding-bottom:5px; border-bottom:1px solid #D0B0C8; }
.tai-gijiroku-chip { display:inline-block; background:rgba(125,78,138,0.08); border:1px solid #D0B0C8; border-radius:999px; padding:3px 12px; font-size:0.72rem; color:#7D4E8A; margin:2px 4px 2px 0; }
.tai-gijiroku-table { width:100%; border-collapse:collapse; border:1px solid var(--border); border-radius:8px; overflow:hidden; }
.tai-gijiroku-table th { padding:7px 10px; font-size:0.6rem; font-weight:700; color:#7D4E8A; background:#F5EEF8; text-align:left; letter-spacing:0.08em; text-transform:uppercase; }
.tai-gijiroku-table td { padding:7px 10px; font-size:0.78rem; color:var(--ink); border-top:1px solid var(--border); }

</style>
""", unsafe_allow_html=True)

# ── JS: remove stSidebarNav gap on every Streamlit rerun ─────────────────────
# ONLY removes stSidebarNav. Never touches collapsedControl or stSidebarCollapseButton.
# Padding zeroed at levels 1-2 only — level 3 is the actual user content wrapper.
st.markdown("""
<script>
(function(){
    function _removeNav(){
        var nav = document.querySelector('[data-testid="stSidebarNav"]');
        if(nav && nav.parentNode){ nav.parentNode.removeChild(nav); }
        var sb = document.querySelector('section[data-testid="stSidebar"]');
        if(sb){
            var lv1 = sb.querySelector(':scope > div');
            if(lv1){
                lv1.style.setProperty('padding-top','0','important');
                lv1.style.setProperty('margin-top','0','important');
                var lv2 = lv1.querySelector(':scope > div');
                if(lv2){
                    lv2.style.setProperty('padding-top','0','important');
                    lv2.style.setProperty('margin-top','0','important');
                }
            }
        }
    }
    _removeNav();
    [80, 300, 800].forEach(function(t){ setTimeout(_removeNav, t); });
    new MutationObserver(_removeNav).observe(
        document.documentElement, { childList: true, subtree: true }
    );
})();
</script>
""", unsafe_allow_html=True)


# ── Sample transcripts ────────────────────────────────────────────────────────
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

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [
    ("history", []), ("results", None), ("current_transcript", ""),
    ("current_language", ""), ("transcript_text", ""), ("pii_report", None),
    ("groq_warmed", False), ("analysis_done", False)
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Cold start warmup ─────────────────────────────────────────────────────────
if not st.session_state.groq_warmed:
    import threading as _cs_thread
    def _cold_start_tasks():
        try:
            from utils.vector_cache import is_available, get_cache_stats
            if is_available():
                _ = get_cache_stats()
        except Exception:
            pass
    _cs_thread.Thread(target=_cold_start_tasks, daemon=True).start()
    st.session_state.groq_warmed = True

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:0.3rem 0.5rem 1.2rem;'>
      <div style='font-size:1.6rem; margin-bottom:0.1rem;'>🎙️</div>
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

    st.markdown("<div class='sh'>Navigation</div>", unsafe_allow_html=True)
    st.page_link("app.py", label="🎙️  Meeting Analysis", use_container_width=True)
    st.page_link("pages/Export_Documents.py", label="📄  Export Documents", use_container_width=True)
    st.markdown("<hr style='border:none; border-top:1px solid #EDE0D8; margin:1rem 0;'/>", unsafe_allow_html=True)

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
                st.session_state.analysis_done      = True
                st.rerun()

    st.markdown("<hr style='border:none; border-top:1px solid #EDE0D8; margin:1rem 0;'/>", unsafe_allow_html=True)

    try:
        from utils.vector_cache import get_cache_stats
        vc = get_cache_stats()
        if vc.get("available"):
            n = vc.get("transcript_count", 0)
            st.markdown(
                f"<div style='font-size:0.74rem; color:#486858; background:#EDF3EF; "
                f"border:1px solid #A8C8B8; border-radius:6px; padding:0.4rem 0.7rem; margin-bottom:0.8rem;'>"
                f"⚡ Vector cache · {n} transcript{'s' if n!=1 else ''} stored</div>",
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

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding:1.2rem 0 1.2rem; position:relative;'>
  <div style='position:absolute; top:1rem; right:2rem; opacity:0.15;
              font-size:2.5rem; line-height:1; user-select:none;'>🎙️</div>
  <div style='font-size:0.62rem; color:#C8A898; letter-spacing:0.2em;
              text-transform:uppercase; margin-bottom:0.6rem; font-weight:500;'>
    Speech &amp; Meeting Intelligence
  </div>
  <h1 style='font-size:2.1rem; font-weight:600; color:#3C2416;
             margin:0 0 0.6rem; letter-spacing:-0.025em; line-height:1;'>
    TranscriptAI
  </h1>
  <div style='display:flex; align-items:center; gap:0.6rem; flex-wrap:wrap;'>
    <span style='font-size:0.75rem; color:#D96080; background:#FDEEF2; padding:0.2rem 0.7rem; border-radius:999px; font-weight:500; border:1px solid #F2B0C0;'>AI-powered</span>
    <span style='font-size:0.75rem; color:#486858; background:#EDF3EF; padding:0.2rem 0.7rem; border-radius:999px; font-weight:500; border:1px solid #A8C8B8;'>APPI Compliant</span>
    <span style='font-size:0.75rem; color:#B87830; background:#F5E8D0; padding:0.2rem 0.7rem; border-radius:999px; font-weight:500; border:1px solid #D9C090;'>Multi-language</span>
    <span style='font-size:0.75rem; color:#7A5040; background:#FEF3EC; padding:0.2rem 0.7rem; border-radius:999px; font-weight:500; border:1px solid #E5D0C4;'>Formality · Indirect Signals · Code-switch</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────────────────────
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
    if st.button("🌐 Trilingual", help="Hindi + English + Japanese", use_container_width=True):
        st.session_state.transcript_text = SAMPLE_TRILINGUAL; st.rerun()
with c_s2:
    if st.button("⚡ Conflict", help="High-conflict EN+JA", use_container_width=True):
        st.session_state.transcript_text = SAMPLE_HIGH_CONFLICT; st.rerun()
with c_s3:
    if st.button("🗣️ Hinglish", help="Pure Hinglish standup", use_container_width=True):
        st.session_state.transcript_text = SAMPLE_HINGLISH_STANDUP; st.rerun()
with c_clear:
    if st.button("Clear"):
        st.session_state.transcript_text = ""
        st.session_state.results   = None
        st.session_state.pii_report = None
        st.session_state.analysis_done = False
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

# ── Analysis thread bus ───────────────────────────────────────────────────────
import threading as _threading
import builtins

if "_RESULT_BUS" not in st.session_state:
    if not hasattr(builtins, "_TAI_RESULT_BUS"):
        builtins._TAI_RESULT_BUS = {
            "running": False, "done": False,
            "result": None, "error": None,
            "lang": "", "start": 0.0,
        }
    st.session_state["_RESULT_BUS"] = builtins._TAI_RESULT_BUS

_BUS = st.session_state["_RESULT_BUS"]

def _run_analysis_bg(text_in, active_lang, pii_mask, pii_flag, bus):
    try:
        result = analyze_transcript(text_in, active_lang)
        if pii_mask is not None and pii_flag and PII_AVAILABLE:
            result = restore_pii_in_result(result, pii_mask)
        bus["result"] = result
        bus["error"]  = None
    except Exception as _e:
        bus["result"] = None
        bus["error"]  = str(_e)
    finally:
        bus["running"] = False
        bus["done"]    = True

if run_analysis and final_text and not _BUS["running"] and not _BUS["done"]:
    detected_lang = detect_language(final_text)
    active_lang   = forced_lang or detected_lang
    pii_mask = None
    text_in  = final_text
    if pii_enabled and PII_AVAILABLE:
        text_in, pii_mask = mask_transcript(final_text)
        st.session_state.pii_report = get_pii_report(pii_mask)
    _BUS["running"] = True; _BUS["done"] = False; _BUS["result"] = None
    _BUS["error"] = None; _BUS["lang"] = active_lang; _BUS["start"] = time.time()
    st.session_state.current_transcript = final_text
    st.session_state.current_language   = active_lang
    _threading.Thread(target=_run_analysis_bg, args=(text_in, active_lang, pii_mask, pii_enabled, _BUS), daemon=True).start()
    st.rerun()

# Logic to allow footer to render correctly during loading state
needs_rerun = False 

if _BUS["running"]:
    _elapsed = time.time() - _BUS.get("start", time.time())
    _pct = min(int(_elapsed / 6 * 88) + 5, 93)
    st.progress(_pct, text=f"Analyzing · {_elapsed:.0f}s · navigate freely ✓")
    st.caption("⚡ Running in background — result will appear shortly.")
    needs_rerun = True

if _BUS["done"] and _BUS["result"] is not None:
    results     = _BUS["result"]
    active_lang = _BUS["lang"]
    _BUS["done"] = False; _BUS["result"] = None; _BUS["running"] = False
    st.session_state.results              = results
    st.session_state["analysis_result"]   = results
    st.session_state["detected_language"] = active_lang
    st.session_state["analysis_done"]     = True
    
    provider   = results.get("_provider", "")
    duration   = results.get("_duration_ms", 0)
    last_error = results.get("_last_error", "")
    if results.get("_from_vector_cache"):
        st.success(f"⚡ Vector cache hit · {results.get('_cache_similarity',0):.0%} match · instant")
    elif "mock" in provider:
        groq_key_present = bool(os.getenv("GROQ_API_KEY", "").strip())
        if "no_key" in provider or not groq_key_present:
            st.warning("⚠ No GROQ_API_KEY found. Add it in Space Settings → Repository secrets.")
        elif "rate_limit" in provider or "429" in last_error:
            st.warning("⚠ Daily API limit reached — demo data shown. Resumes in 24h.")
        elif "timeout" in provider:
            st.warning("⚠ Groq timed out. Try a shorter transcript (under 800 words).")
        else:
            st.warning(f"⚠ Demo mode. {last_error or 'AI provider unavailable.'}")
    else:
        st.success(f"✓ Done · {provider} · {duration/1000:.1f}s")
    st.session_state.history = add_to_history(st.session_state.history, {
        "timestamp":  datetime.now().isoformat(),
        "language":   active_lang,
        "snippet":    final_text[:80],
        "transcript": st.session_state.current_transcript,
        "results":    results,
    })
    st.rerun()

if _BUS["done"] and _BUS["error"]:
    st.error(f"Analysis failed: {_BUS['error']}")
    _BUS["done"] = False; _BUS["error"] = None

if STREAMING_AVAILABLE and stream_mode and final_text and not run_analysis:
    if st.button("⚡ Stream Live Summary"):
        st.markdown("<div class='sh'>Live Summary</div>", unsafe_allow_html=True)
        try:
            st.write_stream(stream_transcript_groq(final_text, st.session_state.get("current_language","en")))
        except Exception as e:
            st.error(str(e))

# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS RENDERER
# ═══════════════════════════════════════════════════════════════════════════════

def _svg_donut(pct: int, color: str, size: int = 56) -> str:
    r = (size - 8) // 2
    circ = 2 * 3.14159 * r
    dash = circ * pct / 100
    return (
        f"<svg width='{size}' height='{size}' viewBox='0 0 {size} {size}'>"
        f"<circle cx='{size//2}' cy='{size//2}' r='{r}' fill='none' stroke='rgba(60,36,22,0.12)' stroke-width='6'/>"
        f"<circle cx='{size//2}' cy='{size//2}' r='{r}' fill='none' stroke='{color}' stroke-width='6' stroke-linecap='round' "
        f"stroke-dasharray='{dash:.1f} {circ:.1f}' transform='rotate(-90 {size//2} {size//2})'/>"
        f"<text x='50%' y='54%' text-anchor='middle' font-size='13' font-weight='700' fill='{color}' font-family='Arial'>{pct}%</text></svg>"
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
    label = ("Excellent" if score >= 80 else "Good" if score >= 60 else "Fair" if score >= 40 else "At Risk")
    return (
        f"<div style='text-align:center'>"
        f"<svg width='{size}' height='{size}' viewBox='0 0 {size} {size}'>"
        f"<circle cx='60' cy='60' r='{r}' fill='none' stroke='rgba(60,36,22,0.10)' stroke-width='10'/>"
        f"<circle cx='60' cy='60' r='{r}' fill='none' stroke='{color}' stroke-width='10' stroke-linecap='round' "
        f"stroke-dasharray='{dash:.1f} {circ:.1f}' transform='rotate(-90 60 60)' style='filter:drop-shadow(0 0 6px {color}88)'/>"
        f"<text x='50%' y='46%' text-anchor='middle' font-size='22' font-weight='800' fill='#3C2416' font-family=Arial>{score}</text>"
        f"<text x='50%' y='62%' text-anchor='middle' font-size='10' fill='#A87868' font-family=Arial>/ 100</text></svg>"
        f"<div style='font-size:0.7rem;font-weight:600;color:{color};letter-spacing:0.1em;text-transform:uppercase;margin-top:2px'>{label}</div></div>"
    )


def _build_gijiroku_preview(R: dict, language: str) -> str:
    def _clean_val(v):
        if isinstance(v, dict): return " ".join(str(val) for val in v.values() if val)
        if isinstance(v, list): return " ".join(str(val) for val in v if val)
        return str(v)

    try:
        from agents.gijiroku_formatter import GijirokulFormatter, render_markdown
        formatter = GijirokulFormatter()
        plan = formatter.format(analysis=R)

        attendee_chips = "".join(
            f"<span style='display:inline-block;background:rgba(125,78,138,0.10);border:1px solid #D0B0C8;"
            f"border-radius:999px;padding:3px 12px;font-size:0.72rem;color:#7D4E8A;margin:2px 4px 2px 0;'>"
            f"{_clean_val(s)}</span>"
            for s in plan.shussekisha[:6]
        )

        agenda_items = "".join(
            f"<div style='display:flex;align-items:flex-start;gap:8px;margin-bottom:6px;'>"
            f"<span style='background:#7D4E8A;color:#fff;border-radius:5px;padding:1px 7px;"
            f"font-size:0.6rem;font-weight:700;flex-shrink:0;margin-top:2px'>{i:02d}</span>"
            f"<span style='font-size:0.82rem;color:#3C2416;line-height:1.5;'>{_clean_val(item)}</span>"
            f"</div>"
            for i, item in enumerate(plan.gidai[:3], 1)
        )

        action_rows = "".join(
            f"<tr><td style='padding:5px 10px;font-size:0.75rem;color:#7A5040;border-bottom:1px solid #EFE2D8;'>{a.owner}</td>"
            f"<td style='padding:5px 10px;font-size:0.75rem;color:#3C2416;border-bottom:1px solid #EFE2D8;'>{a.task}"
            ("" if not a.flag else "<span style='color:#963030'> ⚠</span>") + "</td>"
            f"<td style='padding:5px 10px;font-size:0.75rem;color:#A87868;border-bottom:1px solid #EFE2D8;'>{a.deadline}</td></tr>"
            for a in plan.action_items[:4]
        )

        soft = R.get("soft_rejections", {}) or {}
        risk = soft.get("risk_level", "NONE")
        risk_colors = {"HIGH":"#963030","MEDIUM":"#986820","LOW":"#BE4060","MINIMAL":"#A87868","NONE":"#2D7A55"}
        risk_bgs    = {"HIGH":"#FAF0F0","MEDIUM":"#FAF0E0","LOW":"#FEF6F8","MINIMAL":"#FDF0EA","NONE":"#EDF3EF"}
        risk_clr = risk_colors.get(risk, "#2D7A55")
        risk_bg  = risk_bgs.get(risk, "#EDF3EF")

        tokki = ""
        if plan.tokki_jiko:
            tokki = (
                f"<div style='margin-top:12px;padding:8px 12px;background:#FAF0F0;"
                f"border-left:3px solid #963030;border-radius:0 8px 8px 0;font-size:0.75rem;color:#963030;'>"
                f"⚠ {plan.tokki_jiko}</div>"
            )

        return f"""
<div style='margin-top:20px;border:1px solid #D0B0C8;border-radius:14px;overflow:hidden;background:#FDFAFF;'>
  <div style='background:linear-gradient(135deg,#7D4E8A 0%,#A06CB5 100%);padding:14px 18px;display:flex;align-items:center;justify-content:space-between;'>
    <div>
      <div style='font-size:0.6rem;color:rgba(255,255,255,0.7);letter-spacing:0.15em;text-transform:uppercase;margin-bottom:3px;'>議事録 · Japanese Formal Business Minutes</div>
      <div style='font-size:0.95rem;font-weight:700;color:#fff;font-family:"Noto Sans JP",sans-serif;'>{plan.kaigi_mei}</div>
    </div>
    <div style='text-align:right;'>
      <div style='font-size:0.68rem;color:rgba(255,255,255,0.75);'>{plan.nichiji}</div>
      <div style='font-size:0.65rem;color:rgba(255,255,255,0.6);margin-top:2px;'>{plan.basho}</div>
    </div>
  </div>
  <div style='padding:16px 18px;'>
    <div style='font-size:0.6rem;font-weight:700;color:#7D4E8A;letter-spacing:0.15em;text-transform:uppercase;margin-bottom:7px;'>出席者 · Attendees</div>
    <div style='margin-bottom:14px;'>{attendee_chips}</div>
    <div style='display:grid;grid-template-columns:1fr auto;gap:16px;margin-bottom:14px;align-items:start;'>
      <div>
        <div style='font-size:0.6rem;font-weight:700;color:#7D4E8A;letter-spacing:0.15em;text-transform:uppercase;margin-bottom:8px;'>議題 · Agenda</div>
        {agenda_items}
      </div>
      <div style='min-width:100px;text-align:center;'>
        <div style='font-size:0.6rem;font-weight:700;color:#7D4E8A;letter-spacing:0.15em;text-transform:uppercase;margin-bottom:7px;'>リスク · Risk Level</div>
        <div style='background:{risk_bg};border:1px solid {risk_clr}33;border-radius:8px;padding:8px 12px;'>
          <div style='font-size:1rem;font-weight:800;color:{risk_clr};'>{risk}</div>
          <div style='font-size:0.6rem;color:#A87868;margin-top:2px;'>{soft.get("total_signals",0)} signals</div>
        </div>
      </div>
    </div>
    <div style='font-size:0.6rem;font-weight:700;color:#7D4E8A;letter-spacing:0.15em;text-transform:uppercase;margin-bottom:8px;'>アクションアイテム · Action Items</div>
    <div style='border:1px solid #EFE2D8;border-radius:8px;overflow:hidden;'>
      <table style='width:100%;border-collapse:collapse;'>
        <thead>
          <tr style='background:#F5EEF8;'>
            <th style='padding:6px 10px;font-size:0.6rem;font-weight:700;color:#7D4E8A;text-align:left;letter-spacing:0.08em;text-transform:uppercase;'>担当者 <span style="font-weight:500;color:#A87868;text-transform:none;letter-spacing:0;">Owner</span></th>
            <th style='padding:6px 10px;font-size:0.6rem;font-weight:700;color:#7D4E8A;text-align:left;letter-spacing:0.08em;text-transform:uppercase;'>タスク <span style="font-weight:500;color:#A87868;text-transform:none;letter-spacing:0;">Task</span></th>
            <th style='padding:6px 10px;font-size:0.6rem;font-weight:700;color:#7D4E8A;text-align:left;letter-spacing:0.08em;text-transform:uppercase;'>期限 <span style="font-weight:500;color:#A87868;text-transform:none;letter-spacing:0;">Deadline</span></th>
          </tr>
        </thead>
        <tbody>{action_rows}</tbody>
      </table>
    </div>
    {tokki}
    <div style='margin-top:12px;display:flex;justify-content:space-between;align-items:center;'>
      <div style='font-size:0.68rem;color:#A87868;'>次回予定 · Next Meeting: {plan.jikai_yotei}</div>
    </div>
  </div>
</div>"""
    except Exception:
        return ""


def build_results_html(R: dict, language: str, features: dict, pii_rep: dict | None) -> str:
    COLORS    = ["#E8829A","#F4A07A","#C9924A","#5A7D6B","#A8897C","#7A5C50"]
    SENT_ICON = {"positive":"🌸","neutral":"🌿","negative":"🍂"}

    ji       = R.get("japan_insights", {})
    speakers = sorted(R.get("speakers", []), key=lambda s: s.get("talk_time_pct", 0), reverse=True)

    def _health():
        soft    = R.get("soft_rejections", {})
        risk    = soft.get("risk_level", "NONE")
        risk_pts= {"NONE":25,"MINIMAL":20,"LOW":15,"MEDIUM":8,"HIGH":0}
        sents   = R.get("sentiment", [])
        w       = {"positive":1.0,"neutral":0.6,"negative":0.1}
        s_pts   = round((sum(w.get(s.get("score","neutral").lower(),0.5) for s in sents)/len(sents)*30) if sents else 15)
        items   = R.get("action_items", [])
        if not items:
            a_pts = 10
        else:
            ver = [i for i in items if not i.get("hallucination_flag")]
            wo  = sum(1 for i in ver if i.get("owner","TBD") not in ("TBD","Unknown",""))
            wd  = sum(1 for i in ver if i.get("deadline","TBD") not in ("TBD","N/A",""))
            a_pts = round((wo+wd)/(2*len(items))*25)
        r_pts   = risk_pts.get(risk, 25)
        ver2    = R.get("verification", {})
        h_pts   = round((1 - ver2.get("overall_hallucination_risk", 0)) * 20)
        score   = min(s_pts + a_pts + r_pts + h_pts, 100)
        color   = ("#2D9E6B" if score >= 80 else "#B87830" if score >= 60 else "#D96080" if score >= 40 else "#C84040")
        bd      = [("Sentiment",s_pts,30),("Action Clarity",a_pts,25),("Comm Risk",r_pts,25),("AI Confidence",h_pts,20)]
        bars    = "".join(
            f"<div style='margin-bottom:8px'>"
            f"<div style='display:flex;justify-content:space-between;margin-bottom:3px'>"
            f"<span style='font-size:0.68rem;color:#7A5040'>{lb}</span>"
            f"<span style='font-size:0.68rem;color:{color};font-weight:600'>{pt}/{tot}</span></div>"
            f"<div style='height:5px;background:rgba(60,36,22,0.10);border-radius:999px'>"
            f"<div style='height:100%;width:{round(pt/tot*100)}%;background:{color};border-radius:999px;'></div></div></div>"
            for lb, pt, tot in bd
        )
        return score, color, bars

    score, hc, hbars = _health()

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

    pii_html = ""
    if pii_rep and pii_rep.get("total_pii_found", 0) > 0:
        n = pii_rep["total_pii_found"]
        pii_html = (
            f"<div class='tai-pii-pill'>🔒 APPI — "
            f"{n} item{'s' if n!=1 else ''} anonymized before analysis</div>"
        )

    # ── Tab 1: Summary ────────────────────────────────────────────────────────
    def _clean_val(v):
        if isinstance(v, dict): return " ".join(str(val) for val in v.values() if val)
        if isinstance(v, list): return " ".join(str(val) for val in v if val)
        return str(v)

    full_sum    = _clean_val(R.get("full_summary", ""))
    bullets     = [_clean_val(b) for b in R.get("summary", [])]
    en_summary  = _clean_val(R.get("en_summary", "") or R.get("english_summary", ""))
    is_japanese = language in ("ja", "mixed")

    sum_html = ""
    if full_sum:
        if is_japanese and en_summary and en_summary.strip() != full_sum.strip():
            sum_html += (
                f"<div class='tai-summary-box'>"
                f"<div class='tai-summary-label'>📋 Meeting Overview</div>"
                f"<div class='tai-bilingual-block'>"
                f"<span class='tai-lang-label tai-lang-ja'>JA</span>"
                f"<div class='tai-bilingual-ja'>{full_sum}</div>"
                f"<span class='tai-lang-label tai-lang-en'>EN</span>"
                f"<div class='tai-bilingual-en'>{en_summary}</div>"
                f"</div>"
                f"</div>"
            )
        elif is_japanese and not en_summary:
            sum_html += (
                f"<div class='tai-summary-box'>"
                f"<div class='tai-summary-label'>📋 Meeting Overview</div>"
                f"<span class='tai-lang-label tai-lang-ja'>JA</span>"
                f"<p style='margin:0;line-height:1.9;font-size:0.85rem;font-family:Noto Sans JP,sans-serif;color:#3C2416'>{full_sum}</p>"
                f"</div>"
            )
        else:
            sum_html += (
                f"<div class='tai-summary-box'>"
                f"<div class='tai-summary-label'>📋 Meeting Overview</div>"
                f"<p style='margin:0;line-height:1.75;font-size:0.85rem;color:#3C2416'>{full_sum}</p>"
                f"</div>"
            )

    if bullets:
        sum_html += f"<div class='tai-section-label'>{len(bullets)} Key Points</div>"
        for i, b in enumerate(bullets, 1):
            has_cjk = any('\u4e00' <= c <= '\u9fff' or '\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff' for c in str(b))
            bullet_font = "font-family:'Noto Sans JP',sans-serif;" if has_cjk else ""
            sum_html += (
                f"<div class='tai-bullet-card'>"
                f"<span class='tai-bullet-num'>{i:02d}</span>"
                f"<span style='color:#3C2416;font-size:0.88rem;line-height:1.65;{bullet_font}'>{b}</span>"
                f"</div>"
            )
    elif not full_sum:
        sum_html += "<div style='color:#A87868;font-size:0.85rem;padding:1rem 0'>No summary extracted. Try a longer transcript.</div>"

    gijiroku_preview = _build_gijiroku_preview(R, language)
    if gijiroku_preview:
        sum_html += gijiroku_preview

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
            "<div style='font-weight:600;color:#3C2416;font-size:0.9rem;margin-bottom:4px'>" + str(i.get("task","")) + "</div>"
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
            "<span class='tai-sent-badge tai-sent-" + s.get("score","neutral").lower() + "'>" + s.get("score","neutral").upper() + "</span>"
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
            f"<div style='height:100%;width:{pct}%;background:{col};border-radius:999px;box-shadow:0 0 8px {col}66;'></div></div>"
            f"<div style='font-size:0.7rem;color:#A87868;margin-top:4px'>{tone}</div>"
            f"</div>"
            f"{_svg_donut(pct, col, 52)}"
            f"</div>"
        )

    # ── Tab 5: Insights ───────────────────────────────────────────────────────
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
            ins_html += "".join("<div class='tai-nemawashi-pill'>◆ " + s + "</div>" for s in sigs)

        if soft and soft.get("total_signals",0) > 0:
            ins_html += "<div class='tai-section-label' style='margin-top:16px'>Soft Rejection Analysis</div>"
            for sig in soft.get("high_signals",[]):
                ins_html += (
                    f"<div class='tai-sig-high'>"
                    f"<div style='font-weight:700;font-size:0.9rem'>🚨 {sig['phrase']}</div>"
                    f"<div style='font-size:0.76rem;color:#7A5040;margin-top:4px'>{sig['reading']} · {sig['speaker']} · {sig['confidence']:.0%}</div>"
                    f"<div style='font-size:0.75rem;color:#3C2416;margin-top:6px;line-height:1.5'>{sig['explanation']}</div>"
                    f"</div>"
                )
            for sig in soft.get("medium_signals",[]):
                ins_html += (
                    f"<div class='tai-sig-med'>"
                    f"<div style='font-weight:700;font-size:0.9rem'>⚠ {sig['phrase']}</div>"
                    f"<div style='font-size:0.76rem;color:#7A5040;margin-top:4px'>{sig['reading']} · {sig['speaker']} · {sig['confidence']:.0%}</div>"
                    f"<div style='font-size:0.75rem;color:#3C2416;margin-top:6px;line-height:1.5'>{sig['explanation']}</div>"
                    f"</div>"
                )
            ins_html += f"<div style='font-size:0.73rem;color:#A87868;font-style:italic;margin-top:8px'>{soft.get('cultural_note','')}</div>"
    else:
        ins_html = "<div style='color:#A87868;font-size:0.85rem;padding:1rem 0;line-height:1.7'>Cultural intelligence features apply to Japanese and Hindi transcripts.</div>"

    insight_label = features.get('insight_tab_label', '🌐 Insights')
    if not insight_label:
        insight_label = "インサイト" if language == "ja" else "Insights"

    # Persistent export banner appended to the end of the panel (visible across all tabs)
    export_banner = """
    <div style='margin-top:20px; padding:18px; background:linear-gradient(135deg, rgba(125,78,138,0.04), rgba(160,108,181,0.06)); border:1px solid #D0B0C8; border-radius:12px; display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:10px;'>
        <div>
            <div style='font-size:0.9rem; font-weight:700; color:#7D4E8A; margin-bottom:4px;'>✨ Analysis Complete</div>
            <div style='font-size:0.75rem; color:#A87868;'>Your meeting intelligence is ready. Export full documents directly to PPT, MD, or TXT.</div>
    """

    return (
        '<div class="tai-results">'
        + pii_html
        + '<div class="tai-tiles">' + tiles + '</div>'
        + '<div class="tai-health">'
        + '<div style="margin-bottom:16px;border:1px solid #D0B0C8;border-radius:12px;overflow:hidden;">'
        +   '<div style="background:linear-gradient(135deg,#7D4E8A,#A06CB5);padding:10px 16px;display:flex;align-items:center;justify-content:space-between;">'
        +     '<div style="font-size:0.72rem;font-weight:700;color:#fff;letter-spacing:0.1em;text-transform:uppercase;">🗾 議事録 Format · Japanese Business Minutes</div>'
        +     '<div style="font-size:0.65rem;color:rgba(255,255,255,0.7);">Standard enterprise document structure</div>'
        +   '</div>'
        +   '<div style="padding:16px;background:#FDFAFF;display:grid;grid-template-columns:repeat(5, 1fr);gap:8px;text-align:center;align-items:center;">'
        +     ''.join(f"<div style='padding:6px 4px;'><div style='font-size:0.78rem;font-weight:700;color:#7D4E8A;font-family:Noto Sans JP,sans-serif;'>{ja}</div><div style='font-size:0.6rem;color:#A87868;margin-top:2px;'>{en}</div></div>" for ja, en in [("会議名","Meeting name"),("出席者","Attendees"),("議題","Agenda"),("決定事項","Decisions"),("アクション","Action items")])
        +   '</div>'
        + '</div>'
        +   '<div class="tai-health-left">' + _health_ring(score, hc) + '</div>'
        +   '<div class="tai-health-right">'
        +     '<div class="tai-health-title">Meeting Health Breakdown</div>'
        +     hbars
        +   '</div>'
        + '</div>'
        + '<div class="tai-radio-tabs">'
        +   '<input type="radio" name="tai-tabs" id="tai-radio-sum" checked>'
        +   '<input type="radio" name="tai-tabs" id="tai-radio-act">'
        +   '<input type="radio" name="tai-tabs" id="tai-radio-sent">'
        +   '<input type="radio" name="tai-tabs" id="tai-radio-spk">'
        +   '<input type="radio" name="tai-tabs" id="tai-radio-ins">'
        +   '<div class="tai-tab-bar">'
        +     '<label class="tai-tab-label" for="tai-radio-sum">\U0001f4dd Summary</label>'
        +     '<label class="tai-tab-label" for="tai-radio-act">\u2705 Actions</label>'
        +     '<label class="tai-tab-label" for="tai-radio-sent">\U0001f338 Sentiment</label>'
        +     '<label class="tai-tab-label" for="tai-radio-spk">\U0001f3a4 Speakers</label>'
        +     '<label class="tai-tab-label" for="tai-radio-ins">' + (insight_label or "Insights") + '</label>'        +   '</div>'
        +   '<div class="tai-panel">'
        +     '<div id="tai-sum"  class="tai-tab-content">' + sum_html  + '</div>'
        +     '<div id="tai-act"  class="tai-tab-content">' + act_html  + '</div>'
        +     '<div id="tai-sent" class="tai-tab-content">' + sent_html + '</div>'
        +     '<div id="tai-spk"  class="tai-tab-content">' + spk_html  + '</div>'
        +     '<div id="tai-ins"  class="tai-tab-content">' + ins_html  + '</div>'
        +     export_banner
        +   '</div>'
        + '</div>'
        + '''<script>
(function(){
  var TAB_KEY = 'tai-active-tab';
  var ids = ['tai-radio-sum','tai-radio-act','tai-radio-sent','tai-radio-spk','tai-radio-ins'];
  var panels = ['tai-sum','tai-act','tai-sent','tai-spk','tai-ins'];
  function activateTab(radioId) {
    ids.forEach(function(id, idx) {
      var radio = document.getElementById(id);
      var panel = document.getElementById(panels[idx]);
      if (radio && panel) {
        if (id === radioId) {
          radio.checked = true;
          panel.style.display = 'block';
        } else {
          radio.checked = false;
          panel.style.display = 'none';
        }
      }
    });
    // Update label styles
    document.querySelectorAll('.tai-tab-label').forEach(function(lbl) {
      var forId = lbl.getAttribute('for');
      if (forId === radioId) {
        lbl.style.color = '#BE4060';
        lbl.style.borderBottomColor = '#D96080';
        lbl.style.fontWeight = '600';
        lbl.style.background = 'rgba(190,64,96,0.04)';
      } else {
        lbl.style.color = '';
        lbl.style.borderBottomColor = '';
        lbl.style.fontWeight = '';
        lbl.style.background = '';
      }
    });
    try { sessionStorage.setItem(TAB_KEY, radioId); } catch(e) {}
  }
  function init() {
    var saved = null;
    try { saved = sessionStorage.getItem(TAB_KEY); } catch(e) {}
    if (saved && ids.indexOf(saved) !== -1) {
      activateTab(saved);
    } else {
      activateTab('tai-radio-sum');
    }
    // Attach click handlers
    document.querySelectorAll('.tai-tab-label').forEach(function(lbl) {
      lbl.addEventListener('click', function(e) {
        e.preventDefault();
        e.stopPropagation();
        var targetId = lbl.getAttribute('for');
        activateTab(targetId);
      });
    });
    // Intercept radio changes from CSS (Streamlit resets)
    ids.forEach(function(id) {
      var radio = document.getElementById(id);
      if (radio) {
        radio.addEventListener('change', function() {
          if (radio.checked) activateTab(id);
        });
      }
    });
  }
  if (document.getElementById('tai-radio-sum')) { init(); }
  else { setTimeout(init, 100); }
})()
</script>'''
        + '</div>'
    )


def compute_health_score(R: dict) -> dict:
    sentiment = R.get("sentiment", [])
    if sentiment:
        weights = {"positive": 1.0, "neutral": 0.6, "negative": 0.1}
        avg = sum(weights.get(s.get("score","neutral").lower(), 0.5) for s in sentiment) / len(sentiment)
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
    soft  = R.get("soft_rejections", {})
    risk  = soft.get("risk_level", "NONE")
    r_pts = {"NONE":25,"MINIMAL":20,"LOW":15,"MEDIUM":8,"HIGH":0}.get(risk, 25)
    h_pts = round((1 - R.get("verification",{}).get("overall_hallucination_risk", 0)) * 20)
    score = min(s_pts + a_pts + r_pts + h_pts, 100)
    if score >= 80:   label, color, bg, border = "Productive Meeting", "#486858", "#EDF3EF", "#A8C8B8"
    elif score >= 60: label, color, bg, border = "Mostly Aligned",    "#986820", "#FAF0E0", "#D9C090"
    elif score >= 40: label, color, bg, border = "Needs Follow-up",   "#C87030", "#FDF0EA", "#E8C090"
    else:             label, color, bg, border = "High Risk",         "#B04040", "#FAF0F0", "#E8A0A0"
    return {"score":score,"label":label,"color":color,"bg":bg,"border":border}


# ── Results display ───────────────────────────────────────────────────────────
st.markdown("""
<style>
@keyframes fadeSlideUp { from{opacity:0;transform:translateY(16px)} to{opacity:1;transform:translateY(0)} }
.results-wrapper { animation: fadeSlideUp 0.4s cubic-bezier(0.4,0,0.2,1) both; }
</style>
<div class='results-wrapper' id='results-top'></div>
""", unsafe_allow_html=True)

if st.session_state.results:
    R        = st.session_state.results
    language = st.session_state.current_language
    pii_rep  = st.session_state.pii_report
    features = get_features(language)

    st.markdown(
        "<hr style='border:none;border-top:1px solid rgba(60,36,22,0.10);margin:1.2rem 0 0.8rem;'/>",
        unsafe_allow_html=True,
    )
    st.markdown(build_results_html(R, language, features, pii_rep), unsafe_allow_html=True)
    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

    if st.button("📄  Export Documents — PPT · 議事録 · MD · JSON →", key="go_export", use_container_width=True):
        st.switch_page("pages/Export_Documents.py")

    from utils import build_export_json, export_filename
    exp = build_export_json(st.session_state.current_transcript, language, R)
    st.download_button(
        "⬇ Export JSON", data=exp.encode(),
        file_name=export_filename(language), mime="application/json",
    )

    if EVAL_AVAILABLE:
        with st.expander("📊 Accuracy Evaluation · Ground Truth Comparison"):
            tc_names = [tc["name"] for tc in TEST_CASES]
            selected = st.selectbox("Test case", tc_names, key="eval_tc")
            tc       = next(t for t in TEST_CASES if t["name"] == selected)
            if st.button("Run evaluation →", key="run_eval"):
                with st.spinner("Evaluating…"):
                    pred   = analyze_transcript(tc["transcript"], tc["language"], bypass_cache=True)
                    report = evaluate(pred, tc["ground_truth"], tc["transcript"], tc_name=tc["name"], provider=pred.get("_provider","unknown"))
                overall = report.get("overall_score", 0)
                rouge1  = report.get("summary",{}).get("avg_rouge1_f1","—")
                act_f1  = report.get("action_items",{}).get("f1","—")
                sent_acc = report.get("sentiment",{}).get("soft_accuracy","—")
                ov_color = "#2D9E6B" if isinstance(overall, (int,float)) and overall >= 80 else "#B87830" if isinstance(overall, (int,float)) and overall >= 60 else "#C84040"
                st.markdown(f"""
                <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:12px 0;'>
                  <div style='background:linear-gradient(135deg,rgba(60,36,22,0.03),rgba(217,96,128,0.06));
                       border:1px solid rgba(60,36,22,0.10);border-top:3px solid {ov_color};
                       border-radius:12px;padding:16px 12px;text-align:center;'>
                    <div style='font-size:0.6rem;color:#A87868;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:6px;font-weight:600;'>🎯 Overall</div>
                    <div style='font-size:1.6rem;font-weight:800;color:{ov_color};line-height:1;'>{overall}%</div>
                    <div style='height:4px;background:rgba(60,36,22,0.08);border-radius:999px;margin-top:8px;'>
                      <div style='height:100%;width:{overall}%;background:{ov_color};border-radius:999px;'></div>
                    </div>
                  </div>
                  <div style='background:linear-gradient(135deg,rgba(60,36,22,0.03),rgba(72,104,88,0.06));
                       border:1px solid rgba(60,36,22,0.10);border-top:3px solid #486858;
                       border-radius:12px;padding:16px 12px;text-align:center;'>
                    <div style='font-size:0.6rem;color:#A87868;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:6px;font-weight:600;'>📊 ROUGE-1</div>
                    <div style='font-size:1.6rem;font-weight:800;color:#486858;line-height:1;'>{rouge1}</div>
                  </div>
                  <div style='background:linear-gradient(135deg,rgba(60,36,22,0.03),rgba(232,128,96,0.06));
                       border:1px solid rgba(60,36,22,0.10);border-top:3px solid #E88060;
                       border-radius:12px;padding:16px 12px;text-align:center;'>
                    <div style='font-size:0.6rem;color:#A87868;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:6px;font-weight:600;'>✅ Action F1</div>
                    <div style='font-size:1.6rem;font-weight:800;color:#E88060;line-height:1;'>{act_f1}</div>
                  </div>
                  <div style='background:linear-gradient(135deg,rgba(60,36,22,0.03),rgba(184,120,48,0.06));
                       border:1px solid rgba(60,36,22,0.10);border-top:3px solid #B87830;
                       border-radius:12px;padding:16px 12px;text-align:center;'>
                    <div style='font-size:0.6rem;color:#A87868;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:6px;font-weight:600;'>🌸 Sentiment</div>
                    <div style='font-size:1.6rem;font-weight:800;color:#B87830;line-height:1;'>{sent_acc}</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
                with st.expander("Full report (JSON)"): st.json(report)

    if TRENDS_AVAILABLE:
        with st.expander("📈 Meeting Intelligence Trends"):
            trends = get_trends(last_n=50)
            if trends.get("empty"):
                st.info(trends.get("message","No trend data yet."))
            else:
                hr_pct = trends['high_soft_rejection_pct']
                h_pct  = trends['avg_hallucination_pct']
                avg_ai = trends['avg_action_items']
                dur    = trends['avg_duration_sec']
                dur_s  = f"{dur:.0f}s" if dur < 60 else f"{dur/60:.1f}m"
                hr_clr = "#C84040" if isinstance(hr_pct, (int,float)) and hr_pct > 30 else "#B87830" if isinstance(hr_pct, (int,float)) and hr_pct > 15 else "#2D9E6B"
                h_clr  = "#C84040" if isinstance(h_pct, (int,float)) and h_pct > 20 else "#B87830" if isinstance(h_pct, (int,float)) and h_pct > 10 else "#2D9E6B"
                st.markdown(f"""
                <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:12px 0;'>
                  <div style='background:linear-gradient(135deg,rgba(60,36,22,0.03),rgba(200,64,64,0.06));
                       border:1px solid rgba(60,36,22,0.10);border-top:3px solid {hr_clr};
                       border-radius:12px;padding:16px 12px;text-align:center;transition:transform 0.2s;'>
                    <div style='font-size:0.6rem;color:#A87868;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:6px;font-weight:600;'>🚨 High Risk</div>
                    <div style='font-size:1.6rem;font-weight:800;color:{hr_clr};line-height:1;'>{hr_pct}%</div>
                    <div style='font-size:0.65rem;color:#A87868;margin-top:4px;'>of meetings</div>
                    <div style='height:4px;background:rgba(60,36,22,0.08);border-radius:999px;margin-top:6px;'>
                      <div style='height:100%;width:{min(hr_pct if isinstance(hr_pct,(int,float)) else 0, 100)}%;background:{hr_clr};border-radius:999px;'></div>
                    </div>
                  </div>
                  <div style='background:linear-gradient(135deg,rgba(60,36,22,0.03),rgba(184,120,48,0.06));
                       border:1px solid rgba(60,36,22,0.10);border-top:3px solid {h_clr};
                       border-radius:12px;padding:16px 12px;text-align:center;transition:transform 0.2s;'>
                    <div style='font-size:0.6rem;color:#A87868;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:6px;font-weight:600;'>🔍 Hallucination</div>
                    <div style='font-size:1.6rem;font-weight:800;color:{h_clr};line-height:1;'>{h_pct}%</div>
                    <div style='font-size:0.65rem;color:#A87868;margin-top:4px;'>avg rate</div>
                    <div style='height:4px;background:rgba(60,36,22,0.08);border-radius:999px;margin-top:6px;'>
                      <div style='height:100%;width:{min(h_pct if isinstance(h_pct,(int,float)) else 0, 100)}%;background:{h_clr};border-radius:999px;'></div>
                    </div>
                  </div>
                  <div style='background:linear-gradient(135deg,rgba(60,36,22,0.03),rgba(72,104,88,0.06));
                       border:1px solid rgba(60,36,22,0.10);border-top:3px solid #486858;
                       border-radius:12px;padding:16px 12px;text-align:center;transition:transform 0.2s;'>
                    <div style='font-size:0.6rem;color:#A87868;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:6px;font-weight:600;'>✅ Avg Actions</div>
                    <div style='font-size:1.6rem;font-weight:800;color:#486858;line-height:1;'>{avg_ai}</div>
                    <div style='font-size:0.65rem;color:#A87868;margin-top:4px;'>per meeting</div>
                  </div>
                  <div style='background:linear-gradient(135deg,rgba(60,36,22,0.03),rgba(217,96,128,0.06));
                       border:1px solid rgba(60,36,22,0.10);border-top:3px solid #D96080;
                       border-radius:12px;padding:16px 12px;text-align:center;transition:transform 0.2s;'>
                    <div style='font-size:0.6rem;color:#A87868;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:6px;font-weight:600;'>⚡ Avg Time</div>
                    <div style='font-size:1.6rem;font-weight:800;color:#D96080;line-height:1;'>{dur_s}</div>
                    <div style='font-size:0.65rem;color:#A87868;margin-top:4px;'>analysis speed</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
def render_footer():
    st.markdown("""
    <style>
    .tai-footer{margin-top:2rem;border-top:1px solid #EDE0D8;padding:1.8rem 0 1.4rem;}
    .tai-footer-card{background:linear-gradient(135deg,#FFFEFB 0%,#FEF6F8 100%);border:1px solid #EDE0D8;border-radius:16px;padding:1.6rem 2rem;max-width:860px;margin:0 auto;box-shadow:0 2px 16px rgba(217,96,128,0.06);}
    .tai-footer-name{font-size:1.1rem;font-weight:700;color:#3C2416;letter-spacing:-0.01em;margin-bottom:0.2rem;}
    .tai-footer-title{font-size:0.74rem;color:#D96080;font-weight:500;letter-spacing:0.04em;margin-bottom:0.8rem;}
    .tai-footer-bio{font-size:0.79rem;color:#7A5040;line-height:1.7;margin-bottom:1.1rem;border-left:3px solid #F2B0C0;padding-left:0.9rem;}
    .tai-footer-links{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:1rem;}
    .tai-footer-link-pill{display:inline-flex;align-items:center;gap:5px;padding:5px 12px;border-radius:999px;font-size:0.72rem;font-weight:500;text-decoration:none;border:1px solid;transition:box-shadow 0.2s,transform 0.2s;}
    .tai-footer-link-pill:hover{transform:translateY(-2px);box-shadow:0 4px 12px rgba(217,96,128,0.15);}
    .tai-footer-link-gh{background:#F8F4FF;color:#3C2416;border-color:#C8A8C8;}
    .tai-footer-link-li{background:#EFF6FF;color:#1A56A8;border-color:#93C5FD;}
    .tai-footer-link-hf{background:#FFF7ED;color:#C05A00;border-color:#FDB97B;}
    .tai-footer-link-mail{background:#FEF6F8;color:#BE4060;border-color:#F2B0C0;}
    .tai-footer-link-repo{background:#F0FDF4;color:#166534;border-color:#86EFAC;}
    .tai-footer-divider{border:none;border-top:1px solid #EDE0D8;margin:0.9rem 0;}
    .tai-footer-stack{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:0.9rem;}
    .tai-footer-chip{font-size:0.66rem;padding:2px 9px;border-radius:999px;background:#FEF3EC;color:#7A5040;border:1px solid #E5D0C4;font-weight:500;}
    .tai-footer-bottom{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px;font-size:0.69rem;color:#C8A898;}
    .tai-footer-stat{display:inline-flex;align-items:center;gap:4px;background:#FEF6F8;border:1px solid #F2B0C0;border-radius:999px;padding:2px 9px;font-size:0.68rem;color:#D96080;font-weight:600;}
    </style>
    <div class="tai-footer">
      <div class="tai-footer-card">
        <div class="tai-footer-name">Kunal Bisht</div>
        <div class="tai-footer-title">AI Engineer &middot; LLM Systems &amp; RAG Pipelines &middot; Multilingual NLP</div>
        <div class="tai-footer-bio">
          I build AI to turn real problems into actual solutions &mdash; not proof-of-concepts that never ship.
          TranscriptAI started because I kept forgetting my meetings &mdash; it became a trilingual intelligence platform
          rebuilt five times until accuracy went from 22% to 93%.
        </div>
        <div class="tai-footer-links">
          <a class="tai-footer-link-pill tai-footer-link-gh"   href="https://github.com/aiKunalBisht"   target="_blank">GitHub</a>
          <a class="tai-footer-link-pill tai-footer-link-li"   href="https://linkedin.com/in/kunalhere"  target="_blank">LinkedIn</a>
          <a class="tai-footer-link-pill tai-footer-link-hf"   href="https://huggingface.co/spaces/KunalTheBeast/TranscriptAI" target="_blank">Live Demo</a>
          <a class="tai-footer-link-pill tai-footer-link-repo" href="https://github.com/aiKunalBisht/Transcript-ai" target="_blank">Source Code</a>
          <a class="tai-footer-link-pill tai-footer-link-mail" href="mailto:kunalbisht909@gmail.com">kunalbisht909@gmail.com</a>
        </div>
        <div class="tai-footer-divider"></div>
        <div class="tai-footer-stack">
          <span class="tai-footer-chip">Python</span><span class="tai-footer-chip">FastAPI</span>
          <span class="tai-footer-chip">LangChain</span><span class="tai-footer-chip">ChromaDB</span>
          <span class="tai-footer-chip">Streamlit</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# Render footer at the very end so it shows up even during the sleep execution
render_footer()

if needs_rerun:
    time.sleep(1.5)
    st.rerun()