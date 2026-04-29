"""
app.py — TranscriptAI
Japanese Business Intelligence Platform

Run: python -m streamlit run app.py
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
    def get_features(lang):
        return {
            "show_japan_insights": lang in ("ja", "mixed"),
            "show_hindi_insights": lang == "hi",
            "show_code_switch": lang in ("ja", "mixed"),
            "insight_tab_label": "🔍 Communication Intelligence" if lang in ("ja","mixed") else "🌐 Insights",
            "insight_tab_enabled": lang != "en",
        }

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
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,300&family=Noto+Sans+JP:wght@300;400;500;700&display=swap');

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

/* ── Base reset ─────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'DM Sans', 'Noto Sans JP', sans-serif !important;
    color: var(--ink) !important;
    -webkit-font-smoothing: antialiased;
}

/* ── App background ─────────────────────────────────────────── */
.stApp {
    background-color: var(--washi) !important;
    background-image:
        radial-gradient(circle at 92% 8%, rgba(217,96,128,0.06) 0%, transparent 40%),
        radial-gradient(circle at 8% 92%, rgba(232,128,96,0.05) 0%, transparent 40%) !important;
}

/* ── Hide the deploy toolbar (black bar at top) ─────────────── */
[data-testid="stToolbar"],
[data-testid="stHeader"],
[data-testid="stDecoration"],
header[data-testid="stHeader"] {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
}

/* ── Sidebar ────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #FDF8F5 !important;
    border-right: 1px solid var(--border) !important;
    background-image: radial-gradient(circle at 50% 0%, rgba(217,96,128,0.05) 0%, transparent 60%) !important;
}
[data-testid="stSidebar"] * { color: var(--ink) !important; }
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label { color: var(--ink-mid) !important; }

/* ── Kill ALL dark backgrounds injected by Streamlit ───────── */
.block-container { background: transparent !important; padding-top: 1rem !important; }
[data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"],
[data-testid="column"],
section.main > div,
.main > div { background: transparent !important; }

/* ── File uploader — the biggest offender ──────────────────── */
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
/* Browse files button inside uploader */
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

/* ── Textarea ────────────────────────────────────────────────── */
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

/* ── Selectbox ───────────────────────────────────────────────── */
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

/* ── Toggle ──────────────────────────────────────────────────── */
[data-testid="stToggle"] input:checked + div {
    background-color: var(--sakura) !important;
}

/* ── Buttons — refined, not bubbly ──────────────────────────── */
.stButton > button {
    background-color: var(--sakura) !important;
    color: #FFFDFB !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.86rem !important;
    padding: 0.5rem 1.3rem !important;
    letter-spacing: 0.01em !important;
    transition: background 0.15s ease, box-shadow 0.15s ease !important;
    box-shadow: 0 1px 4px rgba(217,96,128,0.25) !important;
}
.stButton > button:hover {
    background-color: var(--sakura-deep) !important;
    box-shadow: 0 3px 10px rgba(217,96,128,0.30) !important;
}
.stButton > button:active { transform: scale(0.98) !important; }

/* Download button — outlined style */
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

/* ── Progress bar ────────────────────────────────────────────── */
.stProgress > div > div {
    background-color: var(--border) !important;
    border-radius: 999px !important;
}
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--sakura), var(--peach)) !important;
    border-radius: 999px !important;
    transition: width 0.3s ease !important;
}

/* ── Tabs ────────────────────────────────────────────────────── */
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
    border-radius: 0 !important;
    transition: color 0.15s !important;
}
[data-testid="stTabs"] button:hover {
    color: var(--sakura) !important;
    background: transparent !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--sakura-deep) !important;
    border-bottom: 2px solid var(--sakura) !important;
    font-weight: 600 !important;
    background: transparent !important;
}

/* ── Expander ────────────────────────────────────────────────── */
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

/* ── Spinner ─────────────────────────────────────────────────── */
[data-testid="stSpinner"] > div { border-top-color: var(--sakura) !important; }

/* ── Alerts ──────────────────────────────────────────────────── */
.stAlert { border-radius: 8px !important; }
div[data-testid="stAlert"][data-baseweb="notification"] {
    background: var(--sakura-pale) !important;
    border-left-color: var(--sakura) !important;
    border-radius: 8px !important;
    color: var(--ink-mid) !important;
}

/* ── Markdown text ───────────────────────────────────────────── */
.stMarkdown p, .stMarkdown li, .stMarkdown span {
    color: var(--ink-mid) !important;
}

/* ── Scrollbar ───────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--washi); }
::-webkit-scrollbar-thumb { background: var(--sakura-light); border-radius: 999px; }
::-webkit-scrollbar-thumb:hover { background: var(--sakura); }

/* ════════════════════════════════════════════════════════════════
   CUSTOM COMPONENTS
   ════════════════════════════════════════════════════════════════ */

/* Cards */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.card:hover {
    border-color: var(--sakura-light);
    box-shadow: 0 2px 12px rgba(217,96,128,0.08);
}

/* Metric cards */
.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-top: 3px solid var(--sakura);
    border-radius: 6px;
    padding: 1.1rem 0.8rem;
    text-align: center;
    transition: box-shadow 0.2s;
}
.metric-card:hover { box-shadow: 0 2px 10px rgba(217,96,128,0.10); }
.metric-value {
    font-size: 1.85rem; font-weight: 600;
    color: var(--sakura-deep); line-height: 1.1;
}
.metric-label {
    font-size: 0.59rem; color: var(--ink-faint);
    text-transform: uppercase; letter-spacing: 0.13em; margin-top: 0.4rem;
}

/* Section headers */
.sh {
    font-size: 0.67rem; font-weight: 600;
    color: var(--ink-soft); letter-spacing: 0.14em;
    text-transform: uppercase; margin-bottom: 0.9rem;
    padding-bottom: 0.45rem;
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 0.5rem;
}

/* Action items */
.action-row {
    display: flex; align-items: flex-start; gap: 0.85rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 4px solid var(--sakura);
    border-radius: 0 10px 10px 0;
    padding: 0.95rem 1.2rem; margin-bottom: 0.65rem;
    transition: border-color 0.15s, box-shadow 0.15s;
}
.action-row:hover {
    border-color: var(--sakura-light);
    box-shadow: 0 2px 8px rgba(217,96,128,0.08);
}
.action-row.flagged {
    border-left-color: var(--red);
    background: var(--red-bg);
}
.action-task { font-weight: 500; color: var(--ink); font-size: 0.91rem; line-height: 1.5; }
.action-meta { font-size: 0.78rem; color: var(--ink-soft); margin-top: 0.3rem; }
.action-flag { font-size: 0.74rem; color: var(--red); margin-top: 0.25rem; }

/* Sentiment rows */
.sentiment-row {
    display: flex; align-items: center; gap: 1rem;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 0.85rem 1.1rem; margin-bottom: 0.55rem;
    transition: background 0.15s, border-color 0.15s;
}
.sentiment-row:hover {
    background: var(--sakura-pale);
    border-color: var(--sakura-light);
}
.sentiment-name  { font-weight: 500; font-size: 0.89rem; color: var(--ink); min-width: 130px; }
.sentiment-label { font-size: 0.78rem; color: var(--ink-soft); flex: 1; font-style: italic; }

/* Badges */
.badge {
    display: inline-block; padding: 0.2rem 0.75rem;
    border-radius: 999px; font-size: 0.68rem; font-weight: 600;
    letter-spacing: 0.06em; text-transform: uppercase;
}
.badge-positive { background: var(--green-bg); color: var(--green); }
.badge-neutral  { background: var(--peach-bg); color: var(--ink-mid); }
.badge-negative { background: var(--red-bg);   color: var(--red); }

/* Cultural signal boxes */
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

/* Speaker bars */
.spk-bar-bg   {
    background: var(--border); border-radius: 999px;
    height: 7px; overflow: hidden; margin-top: 0.4rem;
}
.spk-bar-fill { height: 100%; border-radius: 999px; }

/* PII pill */
.pii-pill {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: var(--green-bg); border: 1px solid #A8C8B5;
    border-radius: 999px; padding: 0.28rem 0.9rem;
    font-size: 0.74rem; color: var(--green); font-weight: 500; margin-bottom: 1rem;
}

/* Risk pills */
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

/* Sakura divider - decorative */
.sakura-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.4rem 0;
    position: relative;
}

/* Previous session card */
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
</style>
""", unsafe_allow_html=True)

# ── Sample transcript ────────────────────────────────────────────────────────
SAMPLE_TRANSCRIPT = """田中: おはようございます。本日はお時間をいただきありがとうございます。
鈴木: こちらこそ、よろしくお願いいたします。
田中: Q4の進捗ですが、売上KPIは目標の98%に達しています。
鈴木: 順調ですね。ただ、新機能のリリーススケジュールについては少し懸念がございます。
田中: Yes, I understand. The release is April 1st — we may need a short buffer.
鈴木: 検討いたします。技術チームとも相談しますが、難しいかもしれません。前向きに対応したいと思います。
田中: Understood. では鈴木さんにサインオフをお願いできますか？
鈴木: 承知しました。来週月曜までに確認いたします。
田中: ありがとうございます。次に、サポートチームの増員についてですが。
鈴木: 確認してみます。サポートマニュアルの改訂も同時に進めた方が良いかもしれません。
田中: 同感です。鈴木さん、来週金曜までにドラフトをお願いできますか？
鈴木: かしこまりました。対応いたします。
田中: では次回は来週金曜15:00に。議事録は田中が担当します。
鈴木: 承知いたしました。お疲れ様でした。"""

# ── Session state ────────────────────────────────────────────────────────────
for k, v in [
    ("history", []), ("results", None), ("current_transcript", ""),
    ("current_language", ""), ("transcript_text", ""), ("pii_report", None),
]:
    if k not in st.session_state:
        st.session_state[k] = v

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

  <!-- Decorative accent -->
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

c_sample, c_clear, _ = st.columns([0.2, 0.14, 0.66])
with c_sample:
    if st.button("Load sample"):
        st.session_state.transcript_text = SAMPLE_TRANSCRIPT
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

    provider = results.get("_provider", "")
    duration = results.get("_duration_ms", 0)
    if "mock" in provider:
        msgs = {
            "no_key":  "No API key — add GROQ_API_KEY for real analysis.",
            "timeout": "Analysis timed out. Try a shorter transcript.",
            "offline": "Ollama offline. Start Ollama or add GROQ_API_KEY.",
        }
        reason = next((msgs[k] for k in msgs if k in provider), "Demo mode active.")
        st.warning(f"⚠ {reason}")
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
# RESULTS
# ────────────────────────────────────────────────────────────────────────────
if st.session_state.results:
    R        = st.session_state.results
    language = st.session_state.current_language
    pii_rep  = st.session_state.pii_report
    features = get_features(language)

    st.markdown("<hr style='border:none; border-top:1px solid #EDE0D8; margin:1.6rem 0 1rem;'/>", unsafe_allow_html=True)

    # PII pill
    if pii_rep and pii_rep.get("total_pii_found", 0) > 0:
        n = pii_rep["total_pii_found"]
        st.markdown(
            f"<div class='pii-pill'>🔒 APPI — {n} item{'s' if n!=1 else ''} anonymized before analysis</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div style='font-size:1.2rem; font-weight:600; color:#3D2B1F; margin-bottom:1rem;'>"
        "Analysis Results</div>",
        unsafe_allow_html=True,
    )

    # Stats row
    ji = R.get("japan_insights", {})
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{len(R.get('speakers',[]))}</div><div class='metric-label'>Speakers</div></div>", unsafe_allow_html=True)
    with m2:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{len(R.get('action_items',[]))}</div><div class='metric-label'>Action Items</div></div>", unsafe_allow_html=True)
    with m3:
        cs_val = ji.get("code_switch_count","—") if features.get("show_code_switch") else "—"
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{cs_val}</div><div class='metric-label'>Code Switches</div></div>", unsafe_allow_html=True)
    with m4:
        if features.get("show_japan_insights"):
            mv = ji.get("keigo_level","—").title()
            ml = "Formality"
        else:
            mv = language_display_name(language).split(" ",1)[-1]
            ml = "Language"
        st.markdown(f"<div class='metric-card'><div class='metric-value' style='font-size:1.2rem;'>{mv}</div><div class='metric-label'>{ml}</div></div>", unsafe_allow_html=True)

    st.markdown("<div style='height:0.7rem'></div>", unsafe_allow_html=True)

    exp = build_export_json(st.session_state.current_transcript, language, R)
    st.download_button(
        "Export JSON ↓", data=exp.encode(),
        file_name=export_filename(language), mime="application/json",
    )
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # ── Dynamic tabs ──────────────────────────────────────────────────────────
    tab_labels = ["📝  Summary", "✅  Actions", "🌸  Sentiment", "🎤  Speakers"]
    if features.get("insight_tab_enabled"):
        tab_labels.append(features.get("insight_tab_label", "🌐  Insights"))
    if EVAL_AVAILABLE:
        tab_labels.append("📊  Evaluation")
    if TRENDS_AVAILABLE:
        tab_labels.append("📈  Trends")

    tabs        = st.tabs(tab_labels)
    t_summary   = tabs[0]
    t_actions   = tabs[1]
    t_sentiment = tabs[2]
    t_speakers  = tabs[3]
    idx         = 4
    t_insights  = tabs[idx] if features.get("insight_tab_enabled") else None
    if features.get("insight_tab_enabled"): idx += 1
    t_eval      = tabs[idx] if EVAL_AVAILABLE else None
    if EVAL_AVAILABLE: idx += 1
    t_trends    = tabs[idx] if TRENDS_AVAILABLE else None

    # ── Summary ───────────────────────────────────────────────────────────────
    with t_summary:
        full_summary = R.get("full_summary", "")
        bullets      = R.get("summary", [])

        # ── Resolve previous session data ─────────────────────────────────────
        # history[0] = current analysis (just added), history[1] = previous
        history = st.session_state.history
        prev_data = None
        if len(history) >= 2:
            prev_entry   = history[1]
            prev_results = prev_entry.get("results", {})
            prev_full    = prev_results.get("full_summary", "")
            prev_bullets = prev_results.get("summary", [])
            prev_ts      = prev_entry.get("timestamp", "")
            prev_snippet = prev_entry.get("snippet", "")
            # Only show if there's actual content
            if prev_full or prev_bullets:
                try:
                    prev_label = datetime.fromisoformat(prev_ts).strftime("%b %d · %H:%M") if prev_ts else "Previous"
                except Exception:
                    prev_label = "Previous"
                prev_data = {
                    "full":    prev_full,
                    "bullets": prev_bullets[:3],   # cap at 3 for compact view
                    "label":   prev_label,
                    "snippet": prev_snippet,
                }

        # ── 1. Full narrative summary ──────────────────────────────────────────
        if full_summary:
            st.markdown("<div class='sh'>Meeting Overview</div>", unsafe_allow_html=True)
            st.markdown(
                f"""<div style='
                    background: var(--surface);
                    border: 1px solid var(--border);
                    border-left: 4px solid var(--sakura);
                    border-radius: 0 12px 12px 0;
                    padding: 1.3rem 1.6rem;
                    margin-bottom: 1.4rem;
                    line-height: 1.85;
                    font-size: 0.93rem;
                    color: #3C2416;
                '>{full_summary}</div>""",
                unsafe_allow_html=True,
            )

        # ── 2. Key bullet points (existing behaviour, unchanged) ───────────────
        st.markdown(
            f"<div class='sh'>{len(bullets)} Key Point{'s' if len(bullets) != 1 else ''}</div>",
            unsafe_allow_html=True,
        )
        for i, b in enumerate(bullets, 1):
            st.markdown(
                f"<div class='card'>"
                f"<span style='font-size:0.7rem; font-weight:700; color:#E8829A; "
                f"margin-right:0.7rem; letter-spacing:0.06em;'>{i:02d}</span>"
                f"<span style='color:#3D2B1F; font-size:0.91rem; line-height:1.6;'>{b}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # ── 3. Previous session summary ────────────────────────────────────────
        if prev_data:
            st.markdown(
                "<hr style='border:none; border-top:1px solid var(--border); margin:1.4rem 0 1rem;'/>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='sh'>Previous Session &nbsp;·&nbsp; {prev_data['label']}</div>",
                unsafe_allow_html=True,
            )

            # Show previous full summary if it exists, else fall back to bullets
            if prev_data["full"]:
                st.markdown(
                    f"""<div class='prev-session-card'>
                        <div class='prev-session-header'>📋 Previous Meeting Overview</div>
                        <div style='font-size:0.86rem; color:var(--ink-mid); line-height:1.8;'>
                            {prev_data["full"]}
                        </div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            elif prev_data["bullets"]:
                bullets_html = "".join(
                    f"<div class='prev-session-bullet'>{b}</div>"
                    for b in prev_data["bullets"]
                )
                st.markdown(
                    f"""<div class='prev-session-card'>
                        <div class='prev-session-header'>📋 Previous Key Points</div>
                        {bullets_html}
                        {"<div style='font-size:0.75rem; color:var(--ink-faint); margin-top:0.4rem;'>Showing first 3 — load from sidebar for full view</div>" if len(history[1].get("results",{}).get("summary",[])) > 3 else ""}
                    </div>""",
                    unsafe_allow_html=True,
                )

            if prev_data["snippet"]:
                st.markdown(
                    f"<div style='font-size:0.74rem; color:var(--ink-faint); "
                    f"margin-top:0.4rem; font-style:italic;'>"
                    f"Transcript: \"{prev_data['snippet'][:60]}...\"</div>",
                    unsafe_allow_html=True,
                )

    # ── Action Items ──────────────────────────────────────────────────────────
    with t_actions:
        items = R.get("action_items", [])
        v = sum(1 for i in items if not i.get("hallucination_flag"))
        f = len(items) - v
        f_str = f" &nbsp;·&nbsp; <span style='color:#C0514A;'>⚑ {f} flagged</span>" if f else ""
        st.markdown(
            f"<div class='sh'>{len(items)} Item{'s' if len(items)!=1 else ''}"
            f" &nbsp;·&nbsp; <span style='color:#5A7D6B; font-weight:600;'>✓ {v} verified</span>"
            f"{f_str}</div>",
            unsafe_allow_html=True,
        )
        if not items:
            st.markdown("<div style='color:#C4A99E; font-size:0.85rem;'>No action items extracted.</div>", unsafe_allow_html=True)
        for item in items:
            flagged  = item.get("hallucination_flag", False)
            conf     = item.get("confidence")
            reason   = item.get("flag_reason", "")
            cls      = "action-row flagged" if flagged else "action-row"
            conf_str = f" &nbsp;·&nbsp; {conf:.0%} confidence" if conf is not None else ""
            flag_str = f"<div class='action-flag'>⚠ {reason}</div>" if flagged else ""
            st.markdown(
                f"<div class='{cls}'>"
                f"<div style='font-size:1rem; color:{'#C0514A' if flagged else '#E8829A'}; padding-top:0.1rem;'>{'⚑' if flagged else '◆'}</div>"
                f"<div style='flex:1;'>"
                f"<div class='action-task'>{item.get('task','')}</div>"
                f"<div class='action-meta'>"
                f"Owner: <strong>{item.get('owner','TBD')}</strong>"
                f" &nbsp;·&nbsp; Deadline: <strong>{item.get('deadline','TBD')}</strong>"
                f"{conf_str}</div>"
                f"{flag_str}"
                f"</div></div>",
                unsafe_allow_html=True,
            )

    # ── Sentiment ─────────────────────────────────────────────────────────────
    with t_sentiment:
        st.markdown("<div class='sh'>Speaker Sentiment</div>", unsafe_allow_html=True)
        for s in R.get("sentiment", []):
            lbl   = s.get("score","neutral").lower()
            icon  = {"positive":"🌸","neutral":"🌿","negative":"🍂"}.get(lbl,"🌿")
            badge = f"badge-{lbl}" if lbl in ("positive","neutral","negative") else "badge-neutral"
            st.markdown(
                f"<div class='sentiment-row'>"
                f"<div class='sentiment-name'>{icon} {s.get('speaker','')}</div>"
                f"<span class='badge {badge}'>{lbl.upper()}</span>"
                f"<div class='sentiment-label'>{s.get('label','')}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── Speakers ──────────────────────────────────────────────────────────────
    with t_speakers:
        st.markdown("<div class='sh'>Talk Time Distribution</div>", unsafe_allow_html=True)
        colors = ["#E8829A","#F4A07A","#C9924A","#5A7D6B","#A8897C","#7A5C50"]
        sorted_speakers = sorted(R.get("speakers",[]), key=lambda s: s.get("talk_time_pct",0), reverse=True)
        for i, spk in enumerate(sorted_speakers):
            name  = spk.get("name", f"Speaker {i+1}")
            pct   = spk.get("talk_time_pct", 0)
            tone  = spk.get("tone","—")
            color = colors[i % len(colors)]
            c_l, c_r = st.columns([0.35, 0.65])
            with c_l:
                st.markdown(
                    f"<div class='card' style='padding:0.75rem 1rem;'>"
                    f"<div style='font-weight:500; font-size:0.89rem; color:#3D2B1F;'>🎤 {name}</div>"
                    f"<div style='font-size:0.74rem; color:#C4A99E; margin-top:0.2rem;'>{tone}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with c_r:
                st.markdown(
                    f"<div style='padding-top:0.9rem;'>"
                    f"<div style='font-size:0.77rem; color:#A8897C; margin-bottom:0.3rem;'>{pct}% talk time</div>"
                    f"<div class='spk-bar-bg'>"
                    f"<div class='spk-bar-fill' style='width:{pct}%; background:{color};'></div>"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )

    # ── Insights (Japan / Hindi / none) ───────────────────────────────────────
    if t_insights is not None:
        with t_insights:
            if features.get("show_japan_insights"):
                st.markdown("<div class='sh'>Communication Intelligence · Formality &amp; Signal Analysis</div>", unsafe_allow_html=True)

                # Keigo
                keigo  = ji.get("keigo_level","—")
                k_src  = ji.get("keigo_source","llm")
                k_color= {"high":"#C45C74","medium":"#B07D3A","low":"#A8897C"}.get(keigo,"#A8897C")
                st.markdown(
                    f"<div class='card'>"
                    f"<div class='sh' style='margin-bottom:0.5rem;'>Formality Register</div>"
                    f"<span style='font-size:1.3rem; font-weight:600; color:{k_color};'>{keigo.upper()}</span>"
                    f"<span style='font-size:0.74rem; color:#C4A99E; margin-left:0.6rem;'>via {k_src}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                # Nemawashi signals
                sigs = ji.get("nemawashi_signals",[])
                st.markdown(
                    f"<div style='font-size:0.79rem; font-weight:500; color:#7A5C50; margin:0.6rem 0 0.5rem;'>"
                    f"Indirect Consensus Signals &nbsp;·&nbsp; {len(sigs)} detected</div>",
                    unsafe_allow_html=True,
                )
                if sigs:
                    for sig in sigs:
                        st.markdown(
                            f"<div style='padding:0.55rem 0.9rem; background:#FDE8ED; "
                            f"border-left:2px solid #E8829A; margin-bottom:0.4rem; "
                            f"border-radius:0 6px 6px 0; font-family:\"Noto Sans JP\",sans-serif; "
                            f"font-size:0.87rem; color:#3D2B1F;'>◆ {sig}</div>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.markdown("<div style='color:#C4A99E; font-size:0.83rem;'>No nemawashi signals detected.</div>", unsafe_allow_html=True)

                # Code switching
                if features.get("show_code_switch"):
                    cs = ji.get("code_switch_count", 0)
                    st.markdown(
                        f"<div class='card' style='margin-top:0.6rem;'>"
                        f"<div class='sh' style='margin-bottom:0.4rem;'>Language Code-Switching</div>"
                        f"<span style='font-size:1.6rem; font-weight:600; color:#E8829A;'>{cs}</span>"
                        f"<span style='color:#C4A99E; font-size:0.81rem; margin-left:0.5rem;'>switches detected</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    if cs > 5:
                        st.info(f"High code-switching ({cs}×) — globally-oriented team or international client context.")

                # Soft rejection
                soft = R.get("soft_rejections",{})
                if soft and soft.get("total_signals",0) > 0:
                    risk = soft.get("risk_level","NONE")
                    st.markdown(
                        "<div style='font-size:0.79rem; font-weight:500; color:#7A5C50; margin:1rem 0 0.5rem;'>"
                        "Soft Rejection Analysis</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<div class='card'>"
                        f"<span class='risk-pill risk-{risk}'>{risk} RISK</span>"
                        f"<div style='color:#A8897C; font-size:0.82rem; margin-top:0.5rem;'>"
                        f"{soft.get('risk_summary','')}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    for sig in soft.get("high_signals",[]):
                        st.markdown(
                            f"<div class='signal-high'>"
                            f"<div class='signal-phrase'>🚨 {sig['phrase']}</div>"
                            f"<div class='signal-reading'>{sig['reading']} &nbsp;·&nbsp; Speaker: {sig['speaker']} &nbsp;·&nbsp; {sig['confidence']:.0%}</div>"
                            f"<div class='signal-exp'>{sig['explanation']}</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    for sig in soft.get("medium_signals",[]):
                        st.markdown(
                            f"<div class='signal-medium'>"
                            f"<div class='signal-phrase'>⚠ {sig['phrase']}</div>"
                            f"<div class='signal-reading'>{sig['reading']} &nbsp;·&nbsp; Speaker: {sig['speaker']} &nbsp;·&nbsp; {sig['confidence']:.0%}</div>"
                            f"<div class='signal-exp'>{sig['explanation']}</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    for sig in soft.get("low_signals",[]):
                        st.markdown(
                            f"<div class='signal-low'>"
                            f"<div class='signal-phrase'>◆ {sig['phrase']}</div>"
                            f"<div class='signal-reading'>{sig['reading']} &nbsp;·&nbsp; {sig['speaker']}</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    st.markdown(
                        f"<div style='font-size:0.74rem; color:#C4A99E; margin-top:0.5rem; font-style:italic;'>"
                        f"{soft.get('cultural_note','')}</div>",
                        unsafe_allow_html=True,
                    )

                # PII report
                if pii_rep and pii_rep.get("total_pii_found",0) > 0:
                    st.markdown(
                        "<div style='font-size:0.79rem; font-weight:500; color:#7A5C50; margin:1rem 0 0.5rem;'>"
                        "PII Masking Report</div>",
                        unsafe_allow_html=True,
                    )
                    by_cat = pii_rep.get("by_category",{})
                    if by_cat:
                        cols = st.columns(len(by_cat))
                        for idx,(cat,cnt) in enumerate(by_cat.items()):
                            with cols[idx]:
                                st.markdown(
                                    f"<div class='metric-card'>"
                                    f"<div class='metric-value' style='font-size:1.5rem;'>{cnt}</div>"
                                    f"<div class='metric-label'>{cat}</div>"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )

            elif features.get("show_hindi_insights"):
                st.markdown("<div class='sh'>Communication Intelligence · Indirect Signal Analysis</div>", unsafe_allow_html=True)
                if LANGUAGE_INTEL_AVAILABLE:
                    hi = detect_hindi_patterns(st.session_state.current_transcript)
                    risk = hi.get("risk_level","NONE")
                    st.markdown(
                        f"<div class='card'>"
                        f"<span class='risk-pill risk-{risk}'>{risk} RISK</span>"
                        f"<div style='color:#A8897C; font-size:0.83rem; margin-top:0.5rem;'>"
                        f"{hi.get('risk_summary','')}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    for sig in hi.get("detected",[]):
                        sev_cls = {"HIGH":"signal-high","MEDIUM":"signal-medium","LOW":"signal-low"}.get(sig["severity"],"signal-low")
                        st.markdown(
                            f"<div class='{sev_cls}'>"
                            f"<div class='signal-phrase'>{sig['phrase']}</div>"
                            f"<div class='signal-reading'>{sig['reading']} &nbsp;·&nbsp; {sig['confidence']:.0%} confidence</div>"
                            f"<div class='signal-exp'>{sig['explanation']}</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    st.markdown(
                        f"<div style='font-size:0.74rem; color:#C4A99E; margin-top:0.6rem; font-style:italic;'>"
                        f"{hi.get('cultural_note','')}</div>",
                        unsafe_allow_html=True,
                    )

            else:
                st.markdown(
                    "<div style='color:#C4A99E; font-size:0.85rem; padding:1.2rem 0; line-height:1.7;'>"
                    "Cultural intelligence features apply to Japanese and Hindi transcripts.<br>"
                    "Switch to a Japanese or Hindi transcript to see keigo, nemawashi,"
                    " and indirect signal analysis."
                    "</div>",
                    unsafe_allow_html=True,
                )

    # ── Evaluation ────────────────────────────────────────────────────────────
    if t_eval is not None:
        with t_eval:
            st.markdown("<div class='sh'>Accuracy Evaluation · Ground Truth Comparison</div>", unsafe_allow_html=True)
            st.markdown(
                "<div style='font-size:0.82rem; color:#A8897C; margin-bottom:1rem;'>"
                "Select a test case with known ground truth to measure analysis accuracy.</div>",
                unsafe_allow_html=True,
            )
            tc_names = [tc["name"] for tc in TEST_CASES]
            selected = st.selectbox("Test case", tc_names)
            tc       = next(t for t in TEST_CASES if t["name"] == selected)

            if st.button("Run evaluation →"):
                with st.spinner("Evaluating…"):
                    pred   = analyze_transcript(tc["transcript"], tc["language"])
                    report = evaluate(pred, tc["ground_truth"], tc["transcript"])

                overall = report.get("overall_score", 0)
                c1,c2,c3,c4 = st.columns(4)
                with c1: st.markdown(f"<div class='metric-card'><div class='metric-value'>{overall}%</div><div class='metric-label'>Overall</div></div>", unsafe_allow_html=True)
                with c2: st.markdown(f"<div class='metric-card'><div class='metric-value'>{report.get('summary',{}).get('avg_rouge1_f1','—')}</div><div class='metric-label'>ROUGE-1</div></div>", unsafe_allow_html=True)
                with c3: st.markdown(f"<div class='metric-card'><div class='metric-value'>{report.get('action_items',{}).get('f1','—')}</div><div class='metric-label'>Action F1</div></div>", unsafe_allow_html=True)
                with c4: st.markdown(f"<div class='metric-card'><div class='metric-value'>{report.get('sentiment',{}).get('accuracy','—')}</div><div class='metric-label'>Sentiment</div></div>", unsafe_allow_html=True)

                if "japan_insights" in report:
                    ji_r = report["japan_insights"]
                    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
                    st.markdown("<div class='sh'>Japan Intelligence Validation</div>", unsafe_allow_html=True)
                    ck,cn,cc = st.columns(3)
                    with ck:
                        kg = ji_r["keigo"]["grade"]
                        kc = "#5A7D6B" if kg=="PASS" else "#C0514A"
                        st.markdown(f"<div class='metric-card'><div class='metric-value' style='color:{kc}; font-size:1.3rem;'>{kg}</div><div class='metric-label'>Keigo</div></div>", unsafe_allow_html=True)
                    with cn:
                        st.markdown(f"<div class='metric-card'><div class='metric-value'>{ji_r['nemawashi']['precision']}</div><div class='metric-label'>Nemawashi</div></div>", unsafe_allow_html=True)
                    with cc:
                        csg = ji_r["code_switching"]["grade"]
                        cc2 = "#5A7D6B" if csg=="PASS" else "#C0514A"
                        st.markdown(f"<div class='metric-card'><div class='metric-value' style='color:{cc2}; font-size:1.3rem;'>{csg}</div><div class='metric-label'>Code-switch</div></div>", unsafe_allow_html=True)

                with st.expander("Full report (JSON)"):
                    st.json(report)

    # ── Trends ───────────────────────────────────────────────────────────────
    if t_trends is not None:
        with t_trends:
            st.markdown("<div class='sh'>Meeting Intelligence Trends</div>", unsafe_allow_html=True)
            trends = get_trends(last_n=50)

            if trends.get("empty"):
                st.markdown(
                    f"<div style='color:var(--ink-faint); font-size:0.85rem; padding:1rem 0;'>"
                    f"{trends.get('message','')}</div>",
                    unsafe_allow_html=True
                )
            else:
                total   = trends["total"]
                f_date  = trends["first_date"]
                l_date  = trends["last_date"]

                st.markdown(
                    f"<div style='font-size:0.8rem; color:var(--ink-soft); margin-bottom:1.2rem;'>"
                    f"{total} analyses &nbsp;·&nbsp; {f_date} → {l_date}</div>",
                    unsafe_allow_html=True
                )

                # Alerts — most important, shown first
                for alert_key in ["soft_rejection_alert","hallucination_alert","duration_alert"]:
                    alert = trends.get(alert_key)
                    if alert:
                        st.warning(alert)

                # Summary metric cards
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    sr_pct = trends["high_soft_rejection_pct"]
                    color  = "var(--red)" if sr_pct > 30 else "var(--sakura-deep)"
                    st.markdown(
                        f"<div class='metric-card'>"
                        f"<div class='metric-value' style='color:{color};'>{sr_pct}%</div>"
                        f"<div class='metric-label'>High Risk Meetings</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                with m2:
                    hr = trends["avg_hallucination_pct"]
                    color = "var(--red)" if hr > 25 else "var(--sakura-deep)"
                    st.markdown(
                        f"<div class='metric-card'>"
                        f"<div class='metric-value' style='color:{color};'>{hr}%</div>"
                        f"<div class='metric-label'>Avg Hallucination Rate</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                with m3:
                    st.markdown(
                        f"<div class='metric-card'>"
                        f"<div class='metric-value'>{trends['avg_action_items']}</div>"
                        f"<div class='metric-label'>Avg Action Items</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                with m4:
                    dur = trends["avg_duration_sec"]
                    color = "var(--red)" if dur > 60 else "var(--sakura-deep)"
                    dur_str = f"{dur:.0f}s" if dur < 60 else f"{dur/60:.1f}m"
                    st.markdown(
                        f"<div class='metric-card'>"
                        f"<div class='metric-value' style='color:{color};'>{dur_str}</div>"
                        f"<div class='metric-label'>Avg Analysis Time</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

                st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

                # Soft rejection trend — most valuable business signal
                st.markdown("<div class='sh'>🎭 Soft Rejection Risk Trend</div>", unsafe_allow_html=True)
                sr_trend   = trends["soft_rejection_trend"]
                sr_scores  = trends["soft_rejection_scores"]
                sr_labels  = ["None","Minimal","Low","Medium","High"]
                trend_icon = {"up":"↑ Increasing","down":"↓ Decreasing","stable":"→ Stable"}.get(sr_trend,"→")
                trend_color= {"up":"var(--red)","down":"var(--green)","stable":"var(--ink-soft)"}.get(sr_trend,"var(--ink-soft)")
                st.markdown(
                    f"<div style='font-size:0.8rem; color:{trend_color}; font-weight:600; margin-bottom:0.6rem;'>"
                    f"Trend: {trend_icon}</div>",
                    unsafe_allow_html=True
                )
                # Visual bar chart of soft rejection scores
                for i, (ts, score) in enumerate(zip(trends["timestamps"][-10:], sr_scores[-10:])):
                    bar_w   = max(score * 20, 2)
                    bar_col = ["var(--green-bg)","var(--green-bg)","var(--amber-bg)","var(--amber-bg)","var(--red-bg)"][min(score,4)]
                    bar_bdr = ["var(--green)","var(--green)","var(--amber)","var(--amber)","var(--red)"][min(score,4)]
                    label   = sr_labels[min(score,4)]
                    st.markdown(
                        f"<div style='display:flex; align-items:center; gap:0.8rem; margin-bottom:0.3rem; font-size:0.78rem;'>"
                        f"<span style='color:var(--ink-faint); width:80px; flex-shrink:0;'>{ts}</span>"
                        f"<div style='height:20px; width:{bar_w}%; background:{bar_col}; "
                        f"border-left:3px solid {bar_bdr}; border-radius:0 4px 4px 0;'></div>"
                        f"<span style='color:var(--ink-soft);'>{label}</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

                st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

                # Two-column distributions
                col_l, col_r = st.columns(2)

                with col_l:
                    st.markdown("<div class='sh'>📊 Formality Distribution</div>", unsafe_allow_html=True)
                    keigo_dist = trends.get("keigo_dist", {})
                    keigo_total = sum(keigo_dist.values()) or 1
                    keigo_colors = {
                        "high":    ("var(--sakura-deep)", "var(--sakura-pale)"),
                        "medium":  ("var(--amber)",       "var(--amber-bg)"),
                        "low":     ("var(--ink-soft)",    "var(--surface-warm)"),
                        "unknown": ("var(--ink-faint)",   "var(--surface)"),
                    }
                    for level, count in sorted(keigo_dist.items(), key=lambda x: -x[1]):
                        pct  = round(count / keigo_total * 100)
                        fg, bg = keigo_colors.get(level, ("var(--ink-soft)","var(--surface)"))
                        st.markdown(
                            f"<div style='display:flex; align-items:center; gap:0.7rem; margin-bottom:0.4rem;'>"
                            f"<span style='font-size:0.78rem; color:var(--ink-mid); width:60px;'>{level.title()}</span>"
                            f"<div style='flex:1; height:18px; background:{bg}; border-radius:999px; overflow:hidden;'>"
                            f"<div style='height:100%; width:{pct}%; background:{fg}; border-radius:999px;'></div></div>"
                            f"<span style='font-size:0.75rem; color:var(--ink-faint); width:36px; text-align:right;'>{count}</span>"
                            f"</div>",
                            unsafe_allow_html=True
                        )

                with col_r:
                    st.markdown("<div class='sh'>🌐 Language Distribution</div>", unsafe_allow_html=True)
                    lang_dist   = trends.get("language_dist", {})
                    lang_total  = sum(lang_dist.values()) or 1
                    lang_colors = {
                        "ja":    ("var(--sakura-deep)", "var(--sakura-pale)"),
                        "mixed": ("var(--gold)",        "var(--gold-light)"),
                        "en":    ("var(--green)",       "var(--green-bg)"),
                        "hi":    ("var(--peach)",       "var(--peach-bg)"),
                    }
                    for lang, count in sorted(lang_dist.items(), key=lambda x: -x[1]):
                        pct  = round(count / lang_total * 100)
                        fg, bg = lang_colors.get(lang, ("var(--ink-soft)","var(--surface)"))
                        lang_label = {"ja":"Japanese","mixed":"Mixed","en":"English","hi":"Hindi"}.get(lang, lang)
                        st.markdown(
                            f"<div style='display:flex; align-items:center; gap:0.7rem; margin-bottom:0.4rem;'>"
                            f"<span style='font-size:0.78rem; color:var(--ink-mid); width:70px;'>{lang_label}</span>"
                            f"<div style='flex:1; height:18px; background:{bg}; border-radius:999px; overflow:hidden;'>"
                            f"<div style='height:100%; width:{pct}%; background:{fg}; border-radius:999px;'></div></div>"
                            f"<span style='font-size:0.75rem; color:var(--ink-faint); width:36px; text-align:right;'>{count}</span>"
                            f"</div>",
                            unsafe_allow_html=True
                        )

                st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

                # Provider breakdown
                st.markdown("<div class='sh'>⚡ Provider Usage</div>", unsafe_allow_html=True)
                prov_dist  = trends.get("provider_dist", {})
                prov_total = sum(prov_dist.values()) or 1
                prov_cols  = st.columns(min(len(prov_dist), 4))
                prov_colors= {
                    "groq":   "var(--sakura)",
                    "ollama": "var(--peach)",
                    "mock":   "var(--ink-faint)",
                }
                for i, (prov, cnt) in enumerate(sorted(prov_dist.items(), key=lambda x: -x[1])):
                    pct   = round(cnt / prov_total * 100)
                    color = next((v for k,v in prov_colors.items() if k in prov), "var(--ink-soft)")
                    with prov_cols[i % len(prov_cols)]:
                        st.markdown(
                            f"<div class='metric-card' style='border-top-color:{color};'>"
                            f"<div class='metric-value' style='color:{color}; font-size:1.5rem;'>{pct}%</div>"
                            f"<div class='metric-label'>{prov}</div>"
                            f"<div style='font-size:0.7rem; color:var(--ink-faint); margin-top:0.2rem;'>{cnt} runs</div>"
                            f"</div>",
                            unsafe_allow_html=True
                        )

                # Last 10 action item counts — workload trend
                st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
                st.markdown("<div class='sh'>✅ Action Items per Meeting (last 10)</div>", unsafe_allow_html=True)
                ai_counts = trends["action_item_counts"][-10:]
                ts_labels = trends["timestamps"][-10:]
                max_ai    = max(ai_counts) if ai_counts else 1
                for ts, cnt in zip(ts_labels, ai_counts):
                    bar_w = round(cnt / max_ai * 100) if max_ai else 0
                    st.markdown(
                        f"<div style='display:flex; align-items:center; gap:0.8rem; margin-bottom:0.3rem; font-size:0.78rem;'>"
                        f"<span style='color:var(--ink-faint); width:80px; flex-shrink:0;'>{ts}</span>"
                        f"<div style='flex:1; height:16px; background:var(--sakura-pale); border-radius:999px; overflow:hidden;'>"
                        f"<div style='height:100%; width:{bar_w}%; background:var(--sakura); border-radius:999px;'></div></div>"
                        f"<span style='color:var(--ink-soft); width:24px; text-align:right;'>{cnt}</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )


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