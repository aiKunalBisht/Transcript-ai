"""
app.py
------
Main Streamlit application for TranscriptAI.
Japanese Business Intelligence — Call Transcript Analyzer.

HOW TO RUN (only one command needed):
    python -m streamlit run app.py

All other files (analyzer.py, utils.py, pii_masker.py, evaluator.py)
are imported automatically. You never run them directly.

EXCEPTION — to test a single file in isolation:
    python analyzer.py       ← tests AI connection + prints sample JSON
    python pii_masker.py     ← tests PII masking on a sample transcript
    python evaluator.py      ← runs full evaluation against test_data.py
"""

import json
import time
from datetime import datetime

import streamlit as st

from analyzer import analyze_transcript
from utils import (
    add_to_history,
    build_export_json,
    clean_text,
    detect_language,
    export_filename,
    format_history_label,
    language_display_name,
    parse_uploaded_file,
)

# PII masker — graceful import so app works even if file not yet added
try:
    from pii_masker import mask_transcript, restore_pii_in_result, get_pii_report
    PII_AVAILABLE = True
except ImportError:
    PII_AVAILABLE = False

# Evaluator — graceful import
try:
    from evaluator import evaluate
    from test_data import TEST_CASES
    EVAL_AVAILABLE = True
except ImportError:
    EVAL_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TranscriptAI — Japanese Business Intelligence",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Noto+Sans+JP:wght@300;400;500;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', 'Noto Sans JP', sans-serif; }
.stApp { background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%); min-height: 100vh; }
[data-testid="stSidebar"] { background: rgba(255,255,255,0.04); border-right: 1px solid rgba(255,255,255,0.08); backdrop-filter: blur(12px); }
.card { background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.10); border-radius: 16px; padding: 1.4rem 1.6rem; margin-bottom: 1rem; backdrop-filter: blur(8px); transition: border-color 0.2s ease; }
.card:hover { border-color: rgba(139,92,246,0.45); }
.metric-card { background: linear-gradient(135deg,rgba(139,92,246,0.15),rgba(59,130,246,0.15)); border: 1px solid rgba(139,92,246,0.30); border-radius: 12px; padding: 1rem 1.2rem; text-align: center; }
.metric-card .metric-value { font-size: 2rem; font-weight: 700; color: #a78bfa; }
.metric-card .metric-label { font-size: 0.78rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.08em; }
.badge { display: inline-block; padding: 0.25rem 0.75rem; border-radius: 999px; font-size: 0.78rem; font-weight: 600; letter-spacing: 0.04em; }
.badge-positive { background: rgba(34,197,94,0.20); color: #4ade80; border: 1px solid rgba(74,222,128,0.35); }
.badge-neutral  { background: rgba(148,163,184,0.15); color: #94a3b8; border: 1px solid rgba(148,163,184,0.30); }
.badge-negative { background: rgba(239,68,68,0.20); color: #f87171; border: 1px solid rgba(248,113,113,0.35); }
.section-header { font-size: 1.05rem; font-weight: 600; color: #c4b5fd; letter-spacing: 0.04em; margin-bottom: 0.6rem; padding-bottom: 0.4rem; border-bottom: 1px solid rgba(196,181,253,0.20); }
.highlight-box { background: linear-gradient(135deg,rgba(245,158,11,0.12),rgba(234,88,12,0.08)); border-left: 3px solid #f59e0b; border-radius: 0 8px 8px 0; padding: 0.75rem 1rem; margin-bottom: 0.6rem; color: #fde68a; font-size: 0.9rem; }
.action-item { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 0.9rem 1.1rem; margin-bottom: 0.6rem; border-left: 3px solid #8b5cf6; }
.action-item .task-text { color: #e2e8f0; font-size: 0.95rem; font-weight: 500; }
.action-item .meta-text { color: #94a3b8; font-size: 0.8rem; margin-top: 0.3rem; }
.speaker-bar-wrap { margin-bottom: 1rem; }
.speaker-bar-label { color: #e2e8f0; font-size: 0.9rem; font-weight: 500; margin-bottom: 0.25rem; }
.speaker-bar-bg { background: rgba(255,255,255,0.08); border-radius: 999px; height: 10px; overflow: hidden; }
.speaker-bar-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg,#8b5cf6,#3b82f6); }
[data-testid="stTabs"] button { color: #94a3b8 !important; font-weight: 500; border-radius: 8px 8px 0 0; padding: 0.5rem 1rem; }
[data-testid="stTabs"] button[aria-selected="true"] { color: #a78bfa !important; border-bottom: 2px solid #8b5cf6 !important; background: rgba(139,92,246,0.10) !important; }
.stButton > button { background: linear-gradient(135deg,#7c3aed,#4f46e5); color: white; border: none; border-radius: 10px; padding: 0.55rem 1.5rem; font-weight: 600; font-size: 0.95rem; transition: all 0.2s ease; box-shadow: 0 4px 15px rgba(124,58,237,0.35); }
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 6px 20px rgba(124,58,237,0.5); }
[data-testid="stFileUploader"] { border: 2px dashed rgba(139,92,246,0.35) !important; border-radius: 12px !important; background: rgba(139,92,246,0.05) !important; }
textarea { background: rgba(255,255,255,0.05) !important; border: 1px solid rgba(255,255,255,0.12) !important; border-radius: 10px !important; color: #e2e8f0 !important; font-family: 'Noto Sans JP','Inter',monospace !important; }
.stProgress > div > div { background: linear-gradient(90deg,#7c3aed,#3b82f6) !important; border-radius: 999px !important; }
.pii-badge { background: rgba(16,185,129,0.15); border: 1px solid rgba(16,185,129,0.35); border-radius: 8px; padding: 0.5rem 0.9rem; color: #34d399; font-size: 0.82rem; margin-bottom: 0.8rem; display: inline-block; }
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-thumb { background: rgba(139,92,246,0.4); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Sample transcript
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_TRANSCRIPT = """田中: おはようございます、田中です。本日はお時間をいただきありがとうございます。
鈴木: こちらこそ、よろしくお願いいたします。鈴木です。
田中: まず、Q4の進捗についてご報告させていただきます。売上KPIは現時点で目標の98%に達しており、ほぼ計画通りです。
鈴木: そうですね、順調に進んでいるようで安心しました。ただ、新機能のリリーススケジュールについては、少し懸念がございます。
田中: Yes, I understand your concern. The release is scheduled for April 1st, but we may need a buffer.
鈴木: 検討いたします。技術チームとも相談してみますが、難しいかもしれません。できれば前向きに対応したいと思います。
田中: Understood. では、リリース日を鈴木さんの方でサインオフをいただければ、我々は準備を進めます。
鈴木: 承知しました。来週の月曜日までに確認いたします。
田中: ありがとうございます。次に、顧客からのフィードバック対応についてですが、サポートチームの増員が必要だと考えています。
鈴木: そうですね、確認してみます。サポートマニュアルの改訂も同時に進めた方が良いかもしれません。
田中: 同感です。鈴木さん、マニュアルのドラフト作成をお願いできますか？来週の金曜日までにレビュー用に提出していただければ。
鈴木: かしこまりました。対応いたします。
田中: では、次回のミーティングは来週金曜日の15:00に設定しましょう。議事録は田中が担当します。
鈴木: 承知いたしました。本日はありがとうございました。
田中: こちらこそ、よろしくお願いいたします。それでは失礼いたします。
鈴木: 失礼いたします。"""

# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
for key, default in [
    ("history", []),
    ("results", None),
    ("current_transcript", ""),
    ("current_language", ""),
    ("transcript_text", ""),
    ("pii_report", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:1rem 0 0.5rem;'>
        <div style='font-size:2.5rem;'>🎙️</div>
        <div style='font-size:1.1rem; font-weight:700; color:#e2e8f0; margin-top:0.3rem;'>TranscriptAI</div>
        <div style='font-size:0.75rem; color:#94a3b8; margin-top:0.2rem;'>Japanese Business Intelligence</div>
    </div>
    <hr style='border-color:rgba(255,255,255,0.1); margin:1rem 0;'/>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>🌐 Language</div>", unsafe_allow_html=True)
    lang_choice = st.selectbox(
        "Language",
        options=["Auto-detect", "Japanese (日本語)", "English"],
        label_visibility="collapsed",
    )
    lang_map = {"Auto-detect": None, "Japanese (日本語)": "ja", "English": "en"}
    forced_lang = lang_map[lang_choice]

    # PII toggle
    st.markdown("<hr style='border-color:rgba(255,255,255,0.08); margin:1rem 0;'/>", unsafe_allow_html=True)
    if PII_AVAILABLE:
        st.markdown("<div class='section-header'>🔒 Privacy (APPI)</div>", unsafe_allow_html=True)
        pii_enabled = st.toggle("Mask PII before analysis", value=True,
                                help="Anonymizes names, phones, emails before sending to LLM. Required for APPI compliance.")
        if pii_enabled:
            st.markdown("<div style='color:#34d399; font-size:0.78rem;'>✅ APPI compliant mode ON</div>", unsafe_allow_html=True)
    else:
        pii_enabled = False

    st.markdown("<hr style='border-color:rgba(255,255,255,0.08); margin:1rem 0;'/>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>🕘 Recent Analyses</div>", unsafe_allow_html=True)

    if not st.session_state.history:
        st.markdown("<div style='color:#64748b; font-size:0.82rem; padding:0.5rem 0;'>No analyses yet.<br>Run your first analysis to see history here.</div>", unsafe_allow_html=True)
    else:
        for i, entry in enumerate(st.session_state.history):
            label = format_history_label(entry)
            if st.button(f"📄 {label[:45]}…" if len(label) > 45 else f"📄 {label}", key=f"hist_{i}", use_container_width=True):
                st.session_state.results = entry["results"]
                st.session_state.current_transcript = entry["transcript"]
                st.session_state.current_language = entry["language"]
                st.session_state.transcript_text = entry["transcript"]
                st.rerun()

    st.markdown("<hr style='border-color:rgba(255,255,255,0.08); margin:1rem 0;'/>", unsafe_allow_html=True)
    with st.expander("ℹ️ About"):
        st.markdown("""
**TranscriptAI** analyzes meeting transcripts with specialized support for Japanese business communication.

**Supported formats:** `.txt` `.vtt` `.json`

**Japan-specific features:**
- 敬語 (Keigo) register detection
- Nemawashi cue extraction
- Code-switching counter (JA↔EN)
- APPI-compliant PII masking

*Swap the LLM in `analyzer.py` — Claude, GPT-4, or Gemini.*
""")

# ─────────────────────────────────────────────────────────────────────────────
# Main header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding:2rem 0 1rem;'>
    <h1 style='font-size:2.4rem; font-weight:800;
               background:linear-gradient(135deg,#a78bfa,#60a5fa);
               -webkit-background-clip:text; -webkit-text-fill-color:transparent;
               background-clip:text; margin:0;'>
        🎙️ Call Transcript Analyzer
    </h1>
    <p style='color:#94a3b8; margin-top:0.5rem; font-size:1rem;'>
        AI-powered meeting intelligence · Japanese business culture optimized
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Input section
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>📄 Transcript Input</div>", unsafe_allow_html=True)

col_upload, col_paste = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown("<div style='color:#cbd5e1; font-size:0.85rem; margin-bottom:0.4rem;'>Upload a file</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload", type=["txt", "vtt", "json"], label_visibility="collapsed")
    if uploaded is not None:
        parsed = parse_uploaded_file(uploaded)
        st.session_state.transcript_text = parsed
        st.success(f"✅ Loaded **{uploaded.name}** · {len(parsed):,} chars")

with col_paste:
    st.markdown("<div style='color:#cbd5e1; font-size:0.85rem; margin-bottom:0.4rem;'>Or paste transcript</div>", unsafe_allow_html=True)
    transcript_input = st.text_area(
        "Paste",
        value=st.session_state.transcript_text,
        height=220,
        placeholder="Paste your transcript here…\n\nSupports Japanese, English, and mixed JA/EN text.",
        label_visibility="collapsed",
    )
    if transcript_input != st.session_state.transcript_text:
        st.session_state.transcript_text = transcript_input

col_btn_sample, col_btn_clear, col_spacer = st.columns([0.22, 0.15, 0.63])
with col_btn_sample:
    if st.button("📋 Load sample transcript"):
        st.session_state.transcript_text = SAMPLE_TRANSCRIPT
        st.rerun()
with col_btn_clear:
    if st.button("🗑️ Clear"):
        st.session_state.transcript_text = ""
        st.session_state.results = None
        st.session_state.pii_report = None
        st.rerun()

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

final_text = clean_text(st.session_state.transcript_text or "")
analyze_disabled = len(final_text.strip()) < 20

col_analyze, col_lang_display = st.columns([0.35, 0.65])
with col_analyze:
    run_analysis = st.button("🔍 Analyze Transcript", disabled=analyze_disabled, use_container_width=True)
with col_lang_display:
    if final_text:
        detected = detect_language(final_text)
        active_lang = forced_lang if forced_lang else detected
        word_count = len(final_text.split())
        st.markdown(
            f"<div style='padding-top:0.55rem; color:#94a3b8; font-size:0.87rem;'>"
            f"Detected: {language_display_name(detected)} &nbsp;|&nbsp; "
            f"Active: <span style='color:#a78bfa; font-weight:600;'>{language_display_name(active_lang)}</span>"
            f" &nbsp;|&nbsp; {word_count:,} words"
            f"</div>",
            unsafe_allow_html=True,
        )

if analyze_disabled and not final_text:
    st.info("Paste a transcript or upload a file, then click **Analyze Transcript**.")

# ─────────────────────────────────────────────────────────────────────────────
# Run analysis
# ─────────────────────────────────────────────────────────────────────────────
if run_analysis and final_text:
    detected_lang = detect_language(final_text)
    active_lang = forced_lang if forced_lang else detected_lang

    progress_placeholder = st.empty()
    with progress_placeholder.container():
        progress_bar = st.progress(0, text="🔍 Detecting language…")
        time.sleep(0.3)
        progress_bar.progress(15, text="🔒 Masking PII (APPI compliance)…" if pii_enabled and PII_AVAILABLE else "📊 Preparing transcript…")

        # ── PII MASKING ──
        pii_report = None
        text_to_analyze = final_text
        pii_mask = None

        if pii_enabled and PII_AVAILABLE:
            text_to_analyze, pii_mask = mask_transcript(final_text)
            pii_report = get_pii_report(pii_mask)
            st.session_state.pii_report = pii_report

        progress_bar.progress(35, text="🤖 Running AI analysis…")
        with st.spinner("Analyzing transcript · this may take 1–2 minutes locally…"):
            results = analyze_transcript(text_to_analyze, active_lang)

        # Restore real names in results
        # CRITICAL ORDER: restore PII BEFORE speaker normalization
        # Otherwise normalizer sees [NAME_2] and cannot resolve identities
        if pii_mask is not None:
            results = restore_pii_in_result(results, pii_mask)
            pii_mask = None  # mark as restored so we don't restore twice

        progress_bar.progress(85, text="🎨 Formatting results…")
        time.sleep(0.3)
        progress_bar.progress(100, text="✅ Analysis complete!")
        time.sleep(0.4)

    progress_placeholder.empty()

    st.session_state.results  = results
    st.session_state.current_transcript = final_text
    st.session_state.current_language   = active_lang

    # Fix 3: UX failure feedback
    provider = results.get("_provider", "unknown")
    error    = results.get("_last_error", "")
    duration = results.get("_duration_ms", 0)

    if "mock" in provider:
        if "no_key" in provider:
            st.warning("⚠️ No API key found — showing demo data. Add GROQ_API_KEY in Streamlit secrets for real analysis.")
        elif "timeout" in provider:
            st.warning("⚠️ Analysis timed out after retries — showing demo data. Try a shorter transcript.")
        elif "offline" in provider:
            st.warning("⚠️ AI model offline — showing demo data. Start Ollama or add GROQ_API_KEY.")
        else:
            st.info("ℹ️ Running in demo mode — showing sample data.")
    else:
        st.success(f"✅ Analysis complete · Provider: {provider} · {round(duration/1000, 1)}s")

    history_entry = {
        "timestamp": datetime.now().isoformat(),
        "language": active_lang,
        "snippet": final_text[:80],
        "transcript": final_text,
        "results": results,
    }
    st.session_state.history = add_to_history(st.session_state.history, history_entry)
    st.success("✅ Analysis complete!")
    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Results dashboard
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.results:
    results  = st.session_state.results
    language = st.session_state.current_language
    transcript = st.session_state.current_transcript
    pii_report = st.session_state.pii_report

    st.markdown("<hr style='border-color:rgba(255,255,255,0.08); margin:1.5rem 0 1rem;'/>", unsafe_allow_html=True)

    # PII badge
    if pii_report:
        n = pii_report.get("total_pii_found", 0)
        st.markdown(
            f"<div class='pii-badge'>🔒 APPI Compliant — {n} PII item{'s' if n != 1 else ''} masked before AI processing</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='font-size:1.5rem; font-weight:700; color:#e2e8f0; margin-bottom:0.8rem;'>📊 Analysis Results</div>", unsafe_allow_html=True)

    # Quick stats
    c1, c2, c3, c4 = st.columns(4)
    summary_count = len(results.get("summary", []))
    with c1:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{len(results.get('speakers', []))}</div><div class='metric-label'>Speakers</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{len(results.get('action_items', []))}</div><div class='metric-label'>Action Items</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{results.get('japan_insights', {}).get('code_switch_count', 0)}</div><div class='metric-label'>Code Switches</div></div>", unsafe_allow_html=True)
    with c4:
        lang_name = language_display_name(language).split(" ", 1)[-1]
        st.markdown(f"<div class='metric-card'><div class='metric-value' style='font-size:1.3rem;'>{lang_name}</div><div class='metric-label'>Language</div></div>", unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    export_json = build_export_json(transcript, language, results)
    st.download_button("⬇️ Export results as JSON", data=export_json.encode("utf-8"),
                       file_name=export_filename(language), mime="application/json")

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # ── Build tabs dynamically ──
    tab_labels = ["📝 Summary", "✅ Action Items", "😊 Sentiment", "🎤 Speakers", "🇯🇵 Japan Insights"]
    if EVAL_AVAILABLE:
        tab_labels.append("📊 Evaluation")

    tabs = st.tabs(tab_labels)
    tab_summary, tab_actions, tab_sentiment, tab_speakers, tab_japan = tabs[:5]
    tab_eval = tabs[5] if EVAL_AVAILABLE else None

    # ── Summary ──
    with tab_summary:
        bullets = results.get("summary", [])
        st.markdown(
            f"<div class='section-header'>Meeting Summary — {len(bullets)} key point{'s' if len(bullets) != 1 else ''}</div>",
            unsafe_allow_html=True
        )
        for i, bullet in enumerate(bullets, 1):
            st.markdown(
                f"<div class='card'>"
                f"<span style='color:#8b5cf6; font-weight:700; margin-right:0.5rem;'>{i}.</span>"
                f"<span style='color:#e2e8f0;'>{bullet}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── Action Items ──
    with tab_actions:
        items = results.get("action_items", [])
        verified_count = sum(1 for i in items if not i.get("hallucination_flag", False))
        flagged_count  = sum(1 for i in items if i.get("hallucination_flag", False))

        st.markdown(
            f"<div class='section-header'>Action Items &nbsp;"
            f"<span style='font-size:0.8rem; font-weight:400; color:#94a3b8;'>"
            f"✅ {verified_count} verified &nbsp;·&nbsp; "
            f"{'🚩 ' + str(flagged_count) + ' flagged' if flagged_count else '0 flagged'}"
            f"</span></div>",
            unsafe_allow_html=True
        )

        if not items:
            st.info("No action items extracted.")
        else:
            for item in items:
                is_flagged  = item.get("hallucination_flag", False)
                confidence  = item.get("confidence", None)
                flag_reason = item.get("flag_reason", "")
                border_color = "#ef4444" if is_flagged else "#8b5cf6"
                flag_icon    = "🚩" if is_flagged else "🔲"
                conf_str     = f" &nbsp;·&nbsp; 🎯 <strong>Confidence:</strong> {confidence:.0%}" if confidence is not None else ""

                st.markdown(
                    f"<div class='action-item' style='border-left-color:{border_color};'>"
                    f"<div class='task-text'>{flag_icon} {item.get('task','')}</div>"
                    f"<div class='meta-text'>"
                    f"👤 <strong>Owner:</strong> {item.get('owner','TBD')} &nbsp;·&nbsp; "
                    f"📅 <strong>Deadline:</strong> {item.get('deadline','TBD')}{conf_str}"
                    f"</div>"
                    f"{'<div style="color:#f87171;font-size:0.78rem;margin-top:0.3rem;">⚠️ ' + flag_reason + '</div>' if is_flagged else ''}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # ── Sentiment ──
    with tab_sentiment:
        st.markdown("<div class='section-header'>Speaker Sentiment</div>", unsafe_allow_html=True)
        for s in results.get("sentiment", []):
            label = s.get("score", "neutral").lower()
            icon  = {"positive": "😊", "neutral": "😐", "negative": "😟"}.get(label, "😐")
            badge_cls = f"badge-{label}" if label in ("positive","neutral","negative") else "badge-neutral"
            col_spk, col_badge, col_lbl = st.columns([0.35, 0.25, 0.40])
            with col_spk:
                st.markdown(f"<div class='card' style='padding:0.8rem 1rem; margin-bottom:0.4rem;'><span style='color:#e2e8f0; font-weight:600;'>{icon} {s.get('speaker','')}</span></div>", unsafe_allow_html=True)
            with col_badge:
                st.markdown(f"<div style='padding-top:0.55rem;'><span class='badge {badge_cls}'>{label.upper()}</span></div>", unsafe_allow_html=True)
            with col_lbl:
                st.markdown(f"<div style='padding-top:0.6rem; color:#94a3b8; font-size:0.85rem;'>{s.get('label','')}</div>", unsafe_allow_html=True)

    # ── Speakers ──
    with tab_speakers:
        st.markdown("<div class='section-header'>Speaker Breakdown</div>", unsafe_allow_html=True)
        colors = ["#8b5cf6","#3b82f6","#10b981","#f59e0b","#ef4444"]
        for i, spk in enumerate(results.get("speakers", [])):
            name  = spk.get("name", f"Speaker {i+1}")
            pct   = spk.get("talk_time_pct", 0)
            tone  = spk.get("tone", "—")
            color = colors[i % len(colors)]
            col_info, col_bar = st.columns([0.4, 0.6])
            with col_info:
                st.markdown(f"<div class='card'><div style='color:#e2e8f0; font-weight:600;'>🎤 {name}</div><div style='color:#94a3b8; font-size:0.82rem; margin-top:0.3rem;'>Tone: {tone}</div></div>", unsafe_allow_html=True)
            with col_bar:
                st.markdown(f"<div class='speaker-bar-wrap' style='padding-top:0.7rem;'><div class='speaker-bar-label'>{pct}% talk time</div><div class='speaker-bar-bg'><div class='speaker-bar-fill' style='width:{pct}%; background:linear-gradient(90deg,{color},{color}aa);'></div></div></div>", unsafe_allow_html=True)

    # ── Japan Insights ──
    with tab_japan:
        st.markdown("<div class='section-header'>🇯🇵 Japan Business Intelligence</div>", unsafe_allow_html=True)
        japan = results.get("japan_insights", {})

        st.markdown("<div style='color:#a78bfa; font-weight:600; font-size:0.9rem; margin:0.8rem 0 0.4rem;'>📜 Keigo Register (敬語レベル)</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='card'><span style='color:#fde68a;'>🏯 </span><span style='color:#e2e8f0;'>{japan.get('keigo_level','Not detected')}</span></div>", unsafe_allow_html=True)

        signals = japan.get("nemawashi_signals", [])
        st.markdown(f"<div style='color:#a78bfa; font-weight:600; font-size:0.9rem; margin:0.8rem 0 0.4rem;'>🌱 Nemawashi Signals (根回し) — {len(signals)} detected</div>", unsafe_allow_html=True)
        if signals:
            for sig in signals:
                st.markdown(f"<div class='highlight-box'>🔸 {sig}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='card'><span style='color:#94a3b8;'>No nemawashi signals detected.</span></div>", unsafe_allow_html=True)

        cs = japan.get("code_switch_count", 0)
        st.markdown("<div style='color:#a78bfa; font-weight:600; font-size:0.9rem; margin:0.8rem 0 0.4rem;'>🔀 Code-Switching (JA↔EN)</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='card'><span style='font-size:1.8rem; font-weight:700; color:#60a5fa;'>{cs}</span><span style='color:#94a3b8; font-size:0.9rem; margin-left:0.6rem;'>language switches detected mid-conversation</span></div>", unsafe_allow_html=True)
        if cs > 5:
            st.warning(f"⚡ High code-switching frequency ({cs} times) — indicates a globally-oriented team or international client.")

        # Soft rejection signals
        soft = results.get("soft_rejections", {})
        if soft and soft.get("total_signals", 0) > 0:
            risk = soft.get("risk_level", "NONE")
            risk_colors = {
                "HIGH":    "#ef4444",
                "MEDIUM":  "#f59e0b",
                "LOW":     "#60a5fa",
                "MINIMAL": "#94a3b8",
                "NONE":    "#4ade80"
            }
            risk_color = risk_colors.get(risk, "#94a3b8")

            st.markdown(
                f"<div style='color:#a78bfa; font-weight:600; font-size:0.9rem; margin:0.8rem 0 0.4rem;'>"
                f"🎭 Soft Rejection Analysis (間接的拒否)</div>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<div class='card'>"
                f"<span style='color:{risk_color}; font-weight:700; font-size:1rem;'>● {risk} RISK</span>"
                f"<div style='color:#94a3b8; font-size:0.85rem; margin-top:0.4rem;'>{soft.get('risk_summary','')}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

            for signal in soft.get("high_signals", []):
                st.markdown(
                    f"<div style='background:rgba(239,68,68,0.10); border-left:3px solid #ef4444; "
                    f"border-radius:0 8px 8px 0; padding:0.75rem 1rem; margin-bottom:0.5rem;'>"
                    f"<div style='color:#f87171; font-weight:600;'>🚨 {signal['phrase']} — {signal['reading']}</div>"
                    f"<div style='color:#94a3b8; font-size:0.82rem; margin-top:0.3rem;'>Speaker: {signal['speaker']} · Confidence: {signal['confidence']:.0%}</div>"
                    f"<div style='color:#cbd5e1; font-size:0.82rem; margin-top:0.3rem;'>{signal['explanation']}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

            for signal in soft.get("medium_signals", []):
                st.markdown(
                    f"<div style='background:rgba(245,158,11,0.10); border-left:3px solid #f59e0b; "
                    f"border-radius:0 8px 8px 0; padding:0.75rem 1rem; margin-bottom:0.5rem;'>"
                    f"<div style='color:#fde68a; font-weight:600;'>⚠️ {signal['phrase']} — {signal['reading']}</div>"
                    f"<div style='color:#94a3b8; font-size:0.82rem; margin-top:0.3rem;'>Speaker: {signal['speaker']} · Confidence: {signal['confidence']:.0%}</div>"
                    f"<div style='color:#cbd5e1; font-size:0.82rem; margin-top:0.3rem;'>{signal['explanation']}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

            for signal in soft.get("low_signals", []):
                st.markdown(
                    f"<div class='highlight-box'>"
                    f"💡 <strong>{signal['phrase']}</strong> — {signal['reading']} "
                    f"<span style='color:#94a3b8; font-size:0.8rem;'>(Speaker: {signal['speaker']})</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )

            st.markdown(
                f"<div style='color:#64748b; font-size:0.78rem; font-style:italic; margin-top:0.5rem;'>"
                f"💬 {soft.get('cultural_note','')}</div>",
                unsafe_allow_html=True
            )

        # PII detail
        if pii_report and pii_report.get("total_pii_found", 0) > 0:
            st.markdown("<div style='color:#a78bfa; font-weight:600; font-size:0.9rem; margin:0.8rem 0 0.4rem;'>🔒 PII Masking Report</div>", unsafe_allow_html=True)
            by_cat = pii_report.get("by_category", {})
            cols = st.columns(len(by_cat) if by_cat else 1)
            for idx, (cat, count) in enumerate(by_cat.items()):
                with cols[idx]:
                    st.markdown(f"<div class='metric-card'><div class='metric-value' style='font-size:1.5rem;'>{count}</div><div class='metric-label'>{cat}</div></div>", unsafe_allow_html=True)

    # ── Evaluation ──
    if tab_eval is not None:
        with tab_eval:
            st.markdown("<div class='section-header'>📊 AI Evaluation — How accurate is the analysis?</div>", unsafe_allow_html=True)
            st.markdown("<div style='color:#94a3b8; font-size:0.87rem; margin-bottom:1rem;'>Select a test case with known ground truth to measure accuracy.</div>", unsafe_allow_html=True)

            tc_names = [tc["name"] for tc in TEST_CASES]
            selected = st.selectbox("Test case", tc_names)
            tc = next(t for t in TEST_CASES if t["name"] == selected)

            if st.button("▶️ Run evaluation on this test case"):
                with st.spinner("Running evaluation…"):
                    pred = analyze_transcript(tc["transcript"], tc["language"])
                    report = evaluate(pred, tc["ground_truth"], tc["transcript"])

                overall = report.get("overall_score", 0)
                grade   = report.get("overall_grade", "—")

                col_score, col_rouge, col_f1, col_sent = st.columns(4)
                with col_score:
                    st.markdown(f"<div class='metric-card'><div class='metric-value'>{overall}%</div><div class='metric-label'>Overall score</div></div>", unsafe_allow_html=True)
                with col_rouge:
                    st.markdown(f"<div class='metric-card'><div class='metric-value'>{report['summary']['avg_rouge1_f1']}</div><div class='metric-label'>ROUGE-1 F1</div></div>", unsafe_allow_html=True)
                with col_f1:
                    st.markdown(f"<div class='metric-card'><div class='metric-value'>{report['action_items']['f1']}</div><div class='metric-label'>Action F1</div></div>", unsafe_allow_html=True)
                with col_sent:
                    st.markdown(f"<div class='metric-card'><div class='metric-value'>{report['sentiment']['accuracy']}</div><div class='metric-label'>Sentiment acc.</div></div>", unsafe_allow_html=True)

                if "japan_insights" in report:
                    ji = report["japan_insights"]
                    st.markdown("<div class='section-header' style='margin-top:1rem;'>Japan Intelligence Validation</div>", unsafe_allow_html=True)
                    col_k, col_n, col_cs = st.columns(3)
                    with col_k:
                        kg = ji["keigo"]["grade"]
                        color = "#4ade80" if kg == "PASS" else "#f87171"
                        st.markdown(f"<div class='card'><div style='color:{color}; font-weight:700;'>{kg}</div><div style='color:#94a3b8; font-size:0.82rem;'>Keigo detection</div></div>", unsafe_allow_html=True)
                    with col_n:
                        st.markdown(f"<div class='card'><div style='color:#a78bfa; font-weight:700;'>{ji['nemawashi']['precision']}</div><div style='color:#94a3b8; font-size:0.82rem;'>Nemawashi precision</div></div>", unsafe_allow_html=True)
                    with col_cs:
                        csg = ji["code_switching"]["grade"]
                        color = "#4ade80" if csg == "PASS" else "#f87171"
                        st.markdown(f"<div class='card'><div style='color:{color}; font-weight:700;'>{csg}</div><div style='color:#94a3b8; font-size:0.82rem;'>Code-switch count</div></div>", unsafe_allow_html=True)

                with st.expander("Full evaluation report (JSON)"):
                    st.json(report)

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding:2rem 0 1rem; color:#475569; font-size:0.78rem;'>
    TranscriptAI · Powered by Streamlit · Japanese Business Intelligence Platform<br>
    <span style='color:#334155;'>Swap in your preferred LLM in <code>analyzer.py</code> · Supports Claude, GPT-4, Gemini</span>
</div>
""", unsafe_allow_html=True)