"""
pages/Export_Documents.py — TranscriptAI Export Layer  v7.4 (Fortified against NoneType)

FIXES v7.4:
  FIX-1: Dead space at top — block-container padding reduced to 0.3rem,
          header padding reduced from 2rem to 0.8rem
  FIX-2: 議事録 cards (col5/col6) moved to Row 2 — immediately after PPT row
          Plain Text and JSON pushed to Row 3
  FIX-3: Japanese output redesigned — structured HTML card with sections,
          color-coded fields, table layout for action items (not single line)
  FIX-6: Sidebar — force-open CSS removed; Streamlit collapse/reopen works
  SECURITY FIX: Added strict 'or' fallbacks for all R.get dict calls to avoid NoneType errors.
"""
import sys, os
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(dotenv_path=_ROOT / ".env")

import streamlit as st
import json
from datetime import datetime

st.set_page_config(
    page_title="TranscriptAI · Export Documents",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Import builders ──────────────────────────────────────────────────────────
try:
    from agents.slide_architect import SlideArchitectAgent
    from exporters.pptx_builder import build_pptx
    PPTX_AVAILABLE = True
except Exception as _pptx_err:
    PPTX_AVAILABLE = False
    _pptx_err_msg  = str(_pptx_err)

try:
    from agents.gijiroku_formatter import GijirokulFormatter, render_markdown, render_text
    GIJIROKU_AVAILABLE = True
except Exception as _g_err:
    GIJIROKU_AVAILABLE = False
    _g_err_msg = str(_g_err)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=Noto+Sans+JP:wght@400;500&display=swap');
:root {
    --washi:#FAF6F2; --surface:#FFFEFB; --border:#EFE2D8; --border-mid:#E5D0C4;
    --ink:#3C2416; --ink-mid:#7A5040; --ink-soft:#A87868; --ink-faint:#C8A898;
    --sakura:#D96080; --sakura-deep:#BE4060; --sakura-bg:#FDEEF2;
    --sakura-pale:#FEF6F8; --sakura-light:#F2B0C0;
    --peach:#E88060; --peach-bg:#FDF0EA;
    --gold:#B87830; --gold-light:#F5E0C0;
    --green:#486858; --green-bg:#EDF3EF;
    --red:#B04040; --red-bg:#FAF0F0;
    --amber:#986820; --amber-bg:#FAF0E0;
    --purple:#7D4E8A; --purple-bg:#F5EEF8; --purple-border:#D0B0C8;
}
html,body,[class*="css"]{font-family:'DM Sans','Noto Sans JP',sans-serif!important;color:var(--ink)!important;-webkit-font-smoothing:antialiased;}
.stApp{background-color:var(--washi)!important;background-image:radial-gradient(circle at 92% 8%,rgba(217,96,128,0.09) 0%,transparent 45%),radial-gradient(circle at 8% 92%,rgba(232,128,96,0.07) 0%,transparent 45%)!important;}

/* ── FIX-1: Remove dead space at top ──────────────────────────────────── */
.block-container{background:transparent!important;padding-top:0.3rem!important;padding-left:1.5rem!important;padding-right:1.5rem!important;max-width:1200px!important;}
[data-testid="stAppViewBlockContainer"]{padding-top:0.3rem!important;}
.stMainBlockContainer{padding-top:0.3rem!important;}
.stApp > header + div,.stApp > section > div:first-child{padding-top:0!important;margin-top:0!important;}

/* ── FIX-6: Sidebar — let Streamlit control collapse ──────────────────── */
/* REMOVED: display:flex!important, transform:none!important, min-width overrides */
[data-testid="stToolbar"],[data-testid="stHeader"],[data-testid="stDecoration"],header[data-testid="stHeader"]{display:none!important;}
[data-testid="stSidebar"]{background-color:#FDF8F5!important;border-right:1px solid var(--border)!important;box-shadow:2px 0 20px rgba(60,36,22,0.06)!important;}
[data-testid="stSidebar"] *{color:var(--ink)!important;}

/* Collapse button — always accessible */
[data-testid="stSidebarCollapseButton"],[data-testid="collapsedControl"]{display:flex!important;visibility:visible!important;}
[data-testid="stSidebarCollapseButton"] button,[data-testid="collapsedControl"] button{background:transparent!important;border:none!important;color:var(--ink-soft)!important;}

/* Streamlit auto-nav — visible (restored) */
[data-testid="stSidebarNav"],
[data-testid="stSidebarNavItems"],
[data-testid="stSidebarNavLink"],
section[data-testid="stSidebar"] > div:first-child > div > ul,
section[data-testid="stSidebar"] nav {
    display: block !important;
    height: auto !important;
    overflow: visible !important;
    visibility: visible !important;
}

/* ── Buttons ─────────────────────────────────────────────────────────────── */
.stButton>button{background:linear-gradient(135deg,var(--sakura) 0%,var(--sakura-deep) 100%)!important;color:#FFFDFB!important;border:none!important;border-radius:8px!important;font-family:'DM Sans',sans-serif!important;font-weight:600!important;font-size:0.9rem!important;padding:0.7rem 1.6rem!important;transition:all 0.2s!important;box-shadow:0 2px 8px rgba(217,96,128,0.30)!important;}
.stButton>button:hover{background:linear-gradient(135deg,var(--sakura-deep) 0%,#A03050 100%)!important;box-shadow:0 6px 20px rgba(217,96,128,0.40)!important;transform:translateY(-1px)!important;}
[data-testid="stDownloadButton"] button{background-color:transparent!important;color:var(--sakura-deep)!important;border:1.5px solid var(--sakura-light)!important;box-shadow:none!important;font-size:0.95rem!important;padding:0.75rem 1.6rem!important;border-radius:10px!important;width:100%!important;transition:all 0.2s!important;font-weight:500!important;}
[data-testid="stDownloadButton"] button:hover{background-color:var(--sakura-bg)!important;border-color:var(--sakura)!important;transform:translateY(-2px)!important;box-shadow:0 4px 16px rgba(217,96,128,0.15)!important;}
[data-testid="stSpinner"]>div{border-top-color:var(--sakura)!important;}
.stAlert{border-radius:8px!important;}
::-webkit-scrollbar{width:4px;}::-webkit-scrollbar-thumb{background:linear-gradient(180deg,var(--sakura-light),var(--sakura));border-radius:999px;}

/* ── Export cards ─────────────────────────────────────────────────────────── */
.export-card{background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:2rem;margin-bottom:1.2rem;transition:border-color 0.25s,box-shadow 0.25s,transform 0.2s;box-shadow:0 1px 3px rgba(60,36,22,0.04);}
.export-card:hover{border-color:var(--sakura-light);box-shadow:0 6px 24px rgba(217,96,128,0.10);transform:translateY(-2px);}
.export-card-icon{font-size:2.2rem;margin-bottom:0.8rem;}
.export-card-title{font-size:1.05rem;font-weight:600;color:var(--ink);margin-bottom:0.35rem;}
.export-card-desc{font-size:0.81rem;color:var(--ink-soft);line-height:1.65;margin-bottom:1.2rem;}
.export-card-badge{display:inline-block;padding:0.2rem 0.7rem;border-radius:999px;font-size:0.68rem;font-weight:600;letter-spacing:0.06em;text-transform:uppercase;margin-bottom:1rem;}
.badge-ppt{background:var(--peach-bg);color:var(--peach);border:1px solid #F0C0A0;}
.badge-md{background:var(--green-bg);color:var(--green);border:1px solid #A8C8B8;}
.badge-txt{background:var(--sakura-pale);color:var(--sakura-deep);border:1px solid var(--sakura-light);}
.badge-json{background:var(--gold-light);color:var(--gold);border:1px solid #D9C090;}
.badge-gijiroku{background:var(--purple-bg);color:var(--purple);border:1px solid var(--purple-border);}
.meta-row{display:flex;align-items:center;gap:0.8rem;flex-wrap:wrap;margin-bottom:0.5rem;}
.meta-chip{font-size:0.74rem;padding:0.22rem 0.8rem;border-radius:999px;background:var(--sakura-pale);color:var(--sakura-deep);border:1px solid var(--sakura-light);font-weight:500;}
.sh{font-size:0.67rem;font-weight:700;color:var(--ink-soft);letter-spacing:0.16em;text-transform:uppercase;margin-bottom:0.9rem;padding-bottom:0.45rem;border-bottom:2px solid var(--border);}

/* ── FIX-3: Gijiroku structured preview card ──────────────────────────────── */
.gijiroku-card{background:#FDFAFF;border:1px solid var(--purple-border);border-radius:14px;overflow:hidden;margin-top:1rem;}
.gijiroku-header{background:linear-gradient(135deg,#7D4E8A 0%,#A06CB5 100%);padding:16px 20px;}
.gijiroku-header-title{font-size:0.58rem;color:rgba(255,255,255,0.7);letter-spacing:0.15em;text-transform:uppercase;margin-bottom:4px;}
.gijiroku-header-name{font-size:1rem;font-weight:700;color:#fff;font-family:'Noto Sans JP',sans-serif;}
.gijiroku-section-label{font-size:0.58rem;font-weight:700;color:var(--purple);letter-spacing:0.15em;text-transform:uppercase;margin-bottom:8px;padding-bottom:5px;border-bottom:1px solid var(--purple-border);}
.gijiroku-chip{display:inline-block;background:rgba(125,78,138,0.08);border:1px solid var(--purple-border);border-radius:999px;padding:3px 12px;font-size:0.72rem;color:var(--purple);margin:2px 4px 2px 0;}
.gijiroku-decision-row{display:flex;align-items:flex-start;gap:8px;margin-bottom:6px;}
.gijiroku-decision-num{background:#7D4E8A;color:#fff;border-radius:5px;padding:1px 7px;font-size:0.6rem;font-weight:700;flex-shrink:0;margin-top:2px;}
.gijiroku-table{width:100%;border-collapse:collapse;border:1px solid var(--border);border-radius:8px;overflow:hidden;}
.gijiroku-table th{padding:7px 10px;font-size:0.62rem;font-weight:700;color:var(--purple);background:var(--purple-bg);text-align:left;letter-spacing:0.08em;text-transform:uppercase;}
.gijiroku-table td{padding:7px 10px;font-size:0.78rem;color:var(--ink);border-top:1px solid var(--border);}
.gijiroku-table tr:hover td{background:rgba(125,78,138,0.03);}
.gijiroku-risk-box{text-align:center;padding:10px 14px;border-radius:8px;}

/* ── Top navbar ─────────────────────────────────────────────────────────────── */
.tai-navbar { position:sticky; top:0; z-index:999; display:flex; align-items:center; justify-content:space-between; background:rgba(250,246,242,0.96); backdrop-filter:blur(12px); -webkit-backdrop-filter:blur(12px); border-bottom:1px solid var(--border); padding:0 1rem; height:46px; margin-bottom:0.5rem; box-shadow:0 1px 8px rgba(60,36,22,0.06); }
.tai-navbar-brand { display:flex; align-items:center; gap:8px; font-size:0.9rem; font-weight:700; color:var(--ink); text-decoration:none; letter-spacing:-0.01em; flex-shrink:0; }
.tai-navbar-links { display:flex; align-items:center; gap:4px; }
.tai-navbar-link { display:inline-flex; align-items:center; gap:6px; padding:6px 14px; border-radius:8px; font-size:0.8rem; font-weight:500; color:var(--ink-soft); text-decoration:none; transition:background 0.15s,color 0.15s; white-space:nowrap; border:1px solid transparent; }
.tai-navbar-link:hover { background:var(--sakura-pale); color:var(--sakura-deep); border-color:var(--sakura-light); }
.tai-navbar-link.tai-nav-active { background:var(--sakura-bg); color:var(--sakura-deep); border-color:var(--sakura-light); font-weight:600; }
.tai-navbar-link.tai-nav-active .tai-nav-dot { display:inline-block; width:6px; height:6px; border-radius:50%; background:var(--sakura); flex-shrink:0; }
.tai-navbar-link .tai-nav-dot { display:none; }
@media(max-width:540px) { .tai-navbar-brand-text { display:none; } .tai-navbar-link { padding:6px 9px; font-size:0.74rem; } }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1.8rem 0.5rem 1.2rem;'>
      <div style='font-size:1.6rem;margin-bottom:0.4rem;'>🎙️</div>
      <div style='font-size:1rem;font-weight:600;color:#3D2B1F;'>TranscriptAI</div>
      <div style='font-size:0.62rem;color:#C4A99E;letter-spacing:0.14em;text-transform:uppercase;margin-top:0.2rem;'>
        Speech &amp; Meeting Intelligence
      </div>
    </div>
    <hr style='border:none;border-top:1px solid #EDE0D8;margin:0 0 1rem;'/>
    """, unsafe_allow_html=True)

    st.markdown("<div class='sh'>Navigation</div>", unsafe_allow_html=True)
    st.page_link("app.py", label="🎙️  Meeting Analysis", use_container_width=True)
    st.page_link("pages/Export_Documents.py", label="📄  Export Documents", use_container_width=True)
    st.markdown("<hr style='border:none;border-top:1px solid #EDE0D8;margin:1rem 0;'/>", unsafe_allow_html=True)

    has_result = "analysis_result" in st.session_state and st.session_state["analysis_result"] is not None
    if has_result:
        R2      = st.session_state["analysis_result"] or {}
        lang2   = st.session_state.get("detected_language") or "en"
        spk_cnt = len(R2.get("speakers") or [])
        act_cnt = len(R2.get("action_items") or [])
        st.markdown(
            f"<div style='background:var(--green-bg);border:1px solid #A8C8B8;border-radius:8px;"
            f"padding:0.7rem 0.9rem;font-size:0.78rem;color:var(--green);margin-bottom:1rem;'>"
            f"✓ Analysis ready<br>"
            f"<span style='color:var(--ink-soft)'>{spk_cnt} speakers · {act_cnt} actions · {str(lang2).upper()}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='background:var(--peach-bg);border:1px solid #E8C0A0;border-radius:8px;"
            "padding:0.7rem 0.9rem;font-size:0.78rem;color:var(--peach);margin-bottom:1rem;'>"
            "No analysis yet<br><span style='color:var(--ink-soft)'>Run analysis first</span></div>",
            unsafe_allow_html=True,
        )

    with st.expander("About exports"):
        st.markdown("""
**PPT** — Slide deck with titles, bullets, speaker notes

**議事録** — Formal Japanese business meeting minutes

**Markdown** — Clean `.md` for Notion, GitHub, Obsidian

**Plain Text** — Simple `.txt` for email or Slack

**JSON** — Full raw analysis data for developers
""")


# ── Top navbar ───────────────────────────────────────────────────────────────
st.markdown("""
<nav class="tai-navbar">
  <a class="tai-navbar-brand" href="/" target="_self">
    <span>🎙️</span>
    <span class="tai-navbar-brand-text">TranscriptAI</span>
  </a>
  <div class="tai-navbar-links">
    <a class="tai-navbar-link tai-nav-active" href="/" target="_self">
      <span class="tai-nav-dot"></span>🎙️ Meeting Analysis
    </a>
    <a class="tai-navbar-link" href="Export_Documents" target="_self">
      📄 Export Documents
    </a>
  </div>
</nav>
""", unsafe_allow_html=True)
# ── FIX-1: Reduced header padding ────────────────────────────────────────────
st.markdown("""
<div style='padding:0.8rem 0 1.2rem;'>
  <div style='font-size:0.62rem;color:#C8A898;letter-spacing:0.2em;text-transform:uppercase;margin-bottom:0.6rem;font-weight:500;'>
    Meeting Intelligence
  </div>
  <h1 style='font-size:2.1rem;font-weight:600;color:#3C2416;margin:0 0 0.6rem;letter-spacing:-0.025em;'>
    Export Documents
  </h1>
  <div style='font-size:0.88rem;color:#A87868;line-height:1.6;'>
    Turn your analyzed meeting into a presentation, report, or document — ready to share in seconds.
  </div>
</div>
<hr style='border:none;border-top:1px solid rgba(60,36,22,0.10);margin:0 0 1.4rem;'/>
""", unsafe_allow_html=True)

# ── Guard ─────────────────────────────────────────────────────────────────────
if "analysis_result" not in st.session_state or st.session_state["analysis_result"] is None:
    st.markdown("""
    <div style='background:var(--surface);border:1px solid var(--border);border-left:4px solid var(--sakura);
                border-radius:0 12px 12px 0;padding:1.8rem 2rem;max-width:560px;margin:2rem auto;text-align:center;'>
      <div style='font-size:2rem;margin-bottom:1rem;'>🎙️</div>
      <div style='font-size:1rem;font-weight:600;color:var(--ink);margin-bottom:0.5rem;'>No meeting analyzed yet</div>
      <div style='font-size:0.85rem;color:var(--ink-soft);line-height:1.65;'>
        Go to <strong>Meeting Analysis</strong>, paste your transcript, and click Analyze.
        Then come back here to export your results.
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Data & Security Fallbacks ─────────────────────────────────────────────────
R          = st.session_state["analysis_result"] or {}
lang       = st.session_state.get("detected_language") or "en"
transcript = st.session_state.get("current_transcript") or ""

meeting_title   = R.get("meeting_title") or "Meeting Summary"
speaker_names   = [s.get("name", "?") for s in (R.get("speakers") or []) if s]
action_count    = len(R.get("action_items") or [])
summary_bullets = R.get("summary") or []
full_summary    = R.get("full_summary") or ""
action_items    = R.get("action_items") or []
timestamp       = datetime.now().strftime("%Y-%m-%d")

chips = "".join(
    f"<span class='meta-chip'>{c}</span>"
    for c in [f"{len(speaker_names)} speakers", f"{action_count} action items", str(lang).upper(), timestamp]
)
st.markdown(f"<div class='meta-row'>{chips}</div>", unsafe_allow_html=True)
st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

# ── Row 1: PPT + Markdown ─────────────────────────────────────────────────────
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("""
    <div class='export-card'>
      <div class='export-card-icon'>📊</div>
      <div class='export-card-badge badge-ppt'>PowerPoint</div>
      <div class='export-card-title'>Slide Presentation</div>
      <div class='export-card-desc'>
        Auto-generates a slide deck from your meeting — titles, bullet points,
        and speaker notes for every key topic. Download as <strong>.pptx</strong>
        and open in PowerPoint or Google Slides.
      </div>
    </div>
    """, unsafe_allow_html=True)

    if not PPTX_AVAILABLE:
        st.error(f"PPT module not loaded: {_pptx_err_msg}")
    else:
        if st.button("Generate Presentation", key="gen_ppt", use_container_width=True):
            groq_key = os.getenv("GROQ_API_KEY", "").strip()
            if not groq_key:
                try: groq_key = st.secrets.get("GROQ_API_KEY", "")
                except Exception: groq_key = ""
            if not groq_key:
                st.error("GROQ_API_KEY not found. Add it to `.env` or HuggingFace Space secrets.")
            else:
                with st.spinner("Building slide deck · calling Groq · ~5s"):
                    try:
                        agent      = SlideArchitectAgent(groq_api_key=groq_key)
                        plan       = agent.plan(R, language=lang)
                        pptx_bytes = build_pptx(plan)
                        st.session_state["_pptx_ready"]  = pptx_bytes
                        st.session_state["_pptx_slides"] = len(plan.slides or [])
                        st.session_state["_pptx_title"]  = plan.meeting_title or "Meeting"
                    except Exception as e:
                        import traceback
                        st.error(f"Slide generation failed: {e}")
                        st.code(traceback.format_exc())

        if st.session_state.get("_pptx_ready"):
            n     = st.session_state.get("_pptx_slides", "?")
            title = st.session_state.get("_pptx_title", "meeting")
            safe  = str(title).replace(" ", "_")[:40]
            st.success(f"✓ {n} slides ready — {title}")
            st.download_button(
                label=f"⬇  Download .pptx  ({n} slides)",
                data=st.session_state["_pptx_ready"],
                file_name=f"{safe}_{timestamp}.pptx",
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                use_container_width=True,
            )

with col2:
    st.markdown("""
    <div class='export-card'>
      <div class='export-card-icon'>📝</div>
      <div class='export-card-badge badge-md'>Markdown</div>
      <div class='export-card-title'>Markdown Report</div>
      <div class='export-card-desc'>
        A clean, structured <strong>.md</strong> file with summary, action items,
        speaker sentiment, and insights. Perfect for Notion, GitHub, or Obsidian.
      </div>
    </div>
    """, unsafe_allow_html=True)

    md_lines = [f"# {meeting_title}", f"**Date:** {timestamp}  ", f"**Language:** {str(lang).upper()}  ",
                f"**Speakers:** {', '.join(speaker_names) if speaker_names else 'Unknown'}", "", "---", "", "## Summary"]
    if full_summary: md_lines += [full_summary, ""]
    if summary_bullets:
        md_lines.append("### Key Points")
        for b in summary_bullets: md_lines.append(f"- {b}")
        md_lines.append("")
    if action_items:
        md_lines.append("## Action Items")
        for item in action_items:
            if item and isinstance(item, dict):
                flag = " ⚠️" if item.get("hallucination_flag") else ""
                md_lines.append(f"- [ ] **{item.get('task','') or ''}**{flag}  ")
                md_lines.append(f"  Owner: {item.get('owner','') or 'TBD'} · Deadline: {item.get('deadline','') or 'TBD'}")
        md_lines.append("")
        
    sentiment = R.get("sentiment") or []
    if sentiment:
        md_lines.append("## Speaker Sentiment")
        for s in sentiment: 
            if s and isinstance(s, dict):
                score_str = str(s.get('score') or '').upper()
                md_lines.append(f"- **{s.get('speaker','') or ''}**: {score_str} — {s.get('label','') or ''}")
        md_lines.append("")
        
    soft = R.get("soft_rejections") or {}
    if soft and soft.get("total_signals", 0) > 0:
        md_lines += ["## Communication Signals", f"**Risk Level:** {soft.get('risk_level','NONE')}  ",
                     f"**Total Signals:** {soft.get('total_signals',0)}"]
        if soft.get("cultural_note"): md_lines.append(f"\n> {soft['cultural_note']}")
        md_lines.append("")
    md_lines += ["---", "*Generated by TranscriptAI · github.com/aiKunalBisht/Transcript-ai*"]
    md_content = "\n".join(md_lines)

    st.download_button(label="⬇  Download .md", data=md_content,
                       file_name=f"meeting_{timestamp}.md", mime="text/markdown", use_container_width=True)

# ── FIX-2: Row 2 = 議事録 (moved up from Row 3) ───────────────────────────────
st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)
col5, col6 = st.columns(2, gap="large")

with col5:
    st.markdown("""
    <div class='export-card' style='border-color:var(--purple-border);'>
      <div class='export-card-icon'>📋</div>
      <div class='export-card-badge badge-gijiroku'>議事録 · Gijiroku</div>
      <div class='export-card-title'>Japanese Business Meeting Minutes</div>
      <div class='export-card-desc'>
        Formats your analysis as a formal <strong>議事録</strong> — the standard
        Japanese enterprise meeting document. Includes 会議名, 出席者, 議題,
        決定事項, アクションアイテム, and 次回予定.
        Download as <strong>.md</strong> or <strong>.txt</strong>.
      </div>
    </div>
    """, unsafe_allow_html=True)

    if not GIJIROKU_AVAILABLE:
        st.error(f"議事録 module not loaded: {_g_err_msg}")
    else:
        with st.expander("⚙️ Options", expanded=False):
            g_recorder = st.text_input("記録者 (Recorder)", value="TranscriptAI", key="g_recorder")
            g_basho    = st.text_input("場所 (Location)", value="オンライン会議 / Online", key="g_basho")
            g_jikai    = st.text_input("次回予定 (Next meeting)", value="未定 / TBD", key="g_jikai")
            g_format   = st.radio("Export format", ["Markdown (.md)", "Plain Text (.txt)"], key="g_fmt", horizontal=True)

        if st.button("生成する Generate 議事録", key="gen_gijiroku", use_container_width=True):
            with st.spinner("議事録を生成中 · Formatting..."):
                try:
                    formatter = GijirokulFormatter()
                    plan_g = formatter.format(
                        analysis=R, recorder=g_recorder, basho=g_basho,
                        jikai_yotei=g_jikai, timestamp=datetime.now().strftime("%Y年%m月%d日 %H:%M"),
                    )
                    if "Markdown" in g_format:
                        content = render_markdown(plan_g)
                        ext, mime = "md", "text/markdown"
                    else:
                        content = render_text(plan_g)
                        ext, mime = "txt", "text/plain"

                    st.session_state["_gijiroku_content"] = content
                    st.session_state["_gijiroku_ext"]     = ext
                    st.session_state["_gijiroku_mime"]    = mime
                    st.session_state["_gijiroku_title"]   = plan_g.kaigi_mei or "meeting"
                    st.session_state["_gijiroku_plan"]    = plan_g
                    st.success("✓ 議事録が生成されました / Document ready")

                    # FIX-3: Structured HTML preview (not a single line)
                    _soft = R.get("soft_rejections") or {}
                    _risk = _soft.get("risk_level") or "NONE"
                    _risk_colors = {"HIGH":"#963030","MEDIUM":"#986820","LOW":"#BE4060","MINIMAL":"#A87868","NONE":"#2D7A55"}
                    _risk_bgs    = {"HIGH":"#FAF0F0","MEDIUM":"#FAF0E0","LOW":"#FEF6F8","MINIMAL":"#FDF0EA","NONE":"#EDF3EF"}
                    _rc = _risk_colors.get(str(_risk).upper(), "#2D7A55")
                    _rb = _risk_bgs.get(str(_risk).upper(), "#EDF3EF")

                    _attendee_chips = "".join(
                        f"<span class='gijiroku-chip'>{s}</span>"
                        for s in (plan_g.shussekisha or [])
                    )
                    _agenda_html = "".join(
                        f"<div class='gijiroku-decision-row'>"
                        f"<span class='gijiroku-decision-num'>{i:02d}</span>"
                        f"<span style='font-size:0.83rem;color:#3C2416;line-height:1.5;'>{item}</span>"
                        f"</div>"
                        for i, item in enumerate(plan_g.gidai or [], 1)
                    )
                    _decision_html = "".join(
                        f"<div class='gijiroku-decision-row'>"
                        f"<span class='gijiroku-decision-num'>{i:02d}</span>"
                        f"<span style='font-size:0.83rem;color:#3C2416;line-height:1.5;'>{d}</span>"
                        f"</div>"
                        for i, d in enumerate(plan_g.kettei_jiko or [], 1)
                    )
                    _action_rows = "".join(
                        f"<tr><td style='padding:7px 10px;font-size:0.75rem;color:#7A5040;border-top:1px solid #EFE2D8;'>{getattr(a, 'owner', 'TBD')}</td>"
                        f"<td style='padding:7px 10px;font-size:0.78rem;color:#3C2416;border-top:1px solid #EFE2D8;'>"
                        f"{getattr(a, 'task', '') or ''}{'<span style=\"color:#963030;margin-left:4px;\">⚠</span>' if getattr(a, 'flag', False) else ''}</td>"
                        f"<td style='padding:7px 10px;font-size:0.75rem;color:#A87868;border-top:1px solid #EFE2D8;'>{getattr(a, 'deadline', 'TBD')}</td></tr>"
                        for a in (plan_g.action_items or [])
                    )
                    _tokki_html = ""
                    if plan_g.tokki_jiko:
                        _tokki_html = (
                            f"<div style='margin-top:14px;padding:10px 14px;background:#FAF0F0;"
                            f"border-left:3px solid #963030;border-radius:0 8px 8px 0;font-size:0.78rem;color:#963030;'>"
                            f"⚠ 特記事項: {plan_g.tokki_jiko}</div>"
                        )

                    preview_html = f"""
<div class='gijiroku-card'>
  <div class='gijiroku-header' style='display:flex;justify-content:space-between;align-items:flex-start;'>
    <div>
      <div class='gijiroku-header-title'>議事録 · Japanese Business Meeting Minutes</div>
      <div class='gijiroku-header-name'>{plan_g.kaigi_mei or ''}</div>
    </div>
    <div style='text-align:right;'>
      <div style='font-size:0.72rem;color:rgba(255,255,255,0.8);'>{plan_g.nichiji or ''}</div>
      <div style='font-size:0.67rem;color:rgba(255,255,255,0.6);margin-top:3px;'>{plan_g.basho or ''}</div>
      <div style='font-size:0.67rem;color:rgba(255,255,255,0.6);margin-top:2px;'>記録者: {plan_g.kirokusha or ''}</div>
    </div>
  </div>
  <div style='padding:18px 20px;'>
    <div class='gijiroku-section-label'>出席者 Attendees</div>
    <div style='margin-bottom:16px;'>{_attendee_chips}</div>
    <div style='display:grid;grid-template-columns:1fr 110px;gap:16px;margin-bottom:16px;align-items:start;'>
      <div>
        <div class='gijiroku-section-label'>議題 Agenda</div>
        {_agenda_html}
      </div>
      <div>
        <div class='gijiroku-section-label'>リスク</div>
        <div class='gijiroku-risk-box' style='background:{_rb};border:1px solid {_rc}33;'>
          <div style='font-size:1.05rem;font-weight:800;color:{_rc};'>{_risk}</div>
          <div style='font-size:0.6rem;color:#A87868;margin-top:3px;'>{_soft.get("total_signals",0)} signals</div>
        </div>
      </div>
    </div>
    <div class='gijiroku-section-label'>決定事項 Decisions</div>
    <div style='margin-bottom:16px;'>{_decision_html}</div>
    <div class='gijiroku-section-label'>アクションアイテム</div>
    <table class='gijiroku-table' style='margin-bottom:0;'>
      <thead><tr>
        <th>担当者</th><th>タスク</th><th>期限</th>
      </tr></thead>
      <tbody>{_action_rows}</tbody>
    </table>
    {_tokki_html}
    <div style='margin-top:14px;display:flex;justify-content:space-between;align-items:center;padding-top:12px;border-top:1px solid var(--border);'>
      <div style='font-size:0.72rem;color:#A87868;'>次回予定: {plan_g.jikai_yotei or ''}</div>
      <div style='font-size:0.68rem;color:#C8A898;'>Generated by TranscriptAI · {plan_g.generated_at or ''}</div>
    </div>
  </div>
</div>"""
                    st.markdown(preview_html, unsafe_allow_html=True)

                except Exception as e:
                    import traceback
                    st.error(f"議事録生成エラー: {e}")
                    st.code(traceback.format_exc())

        if st.session_state.get("_gijiroku_content"):
            safe_title = str(st.session_state.get("_gijiroku_title", "meeting")).replace(" ", "_")[:40]
            ext  = st.session_state.get("_gijiroku_ext", "md")
            mime = st.session_state.get("_gijiroku_mime", "text/markdown")
            st.download_button(
                label=f"⬇  議事録をダウンロード  (.{ext})",
                data=st.session_state["_gijiroku_content"].encode("utf-8") if isinstance(st.session_state["_gijiroku_content"], str) else st.session_state["_gijiroku_content"],
                file_name=f"gijiroku_{safe_title}_{timestamp}.{ext}",
                mime=mime, use_container_width=True,
            )

with col6:
    st.markdown("""
    <div class='export-card' style='border-color:var(--purple-border);background:#FDFAFF;'>
      <div class='export-card-icon'>🗾</div>
      <div class='export-card-badge badge-gijiroku'>フォーマット解説</div>
      <div class='export-card-title'>About 議事録 Format</div>
      <div class='export-card-desc'>
        議事録 (gijiroku) is the standard formal meeting minutes format
        used in Japanese enterprise environments — Fujitsu, Hitachi, NTT Data,
        and all major corporations expect this structure for any formal meeting.
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
| フィールド | 内容 |
|-----------|------|
| 会議名 | Meeting name |
| 日時 | Date and time |
| 場所 | Location / platform |
| 出席者 | Attendees with roles |
| 議題 | Agenda items |
| 決定事項 | Decisions made |
| アクションアイテム | Who · What · By when |
| 次回予定 | Next meeting schedule |
| 記録者 | Recorder |
| 特記事項 | Notes (soft rejection risk) |
""")
    st.caption(
        "TranscriptAI is currently the only multilingual meeting AI "
        "outputting 議事録 format. This covers Japanese business requirements "
        "that Fujitsu Takane and similar enterprise tools are targeting."
    )

# ── Row 3: Plain Text + JSON (moved down from Row 2) ─────────────────────────
st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)
col3, col4 = st.columns(2, gap="large")

with col3:
    st.markdown("""
    <div class='export-card'>
      <div class='export-card-icon'>📋</div>
      <div class='export-card-badge badge-txt'>Plain Text</div>
      <div class='export-card-title'>Plain Text Summary</div>
      <div class='export-card-desc'>
        Simple <strong>.txt</strong> file. Works everywhere — email, Slack,
        any notes app. No formatting, just the important content.
      </div>
    </div>
    """, unsafe_allow_html=True)

    txt_lines = [f"MEETING SUMMARY — {timestamp}", "=" * 40, ""]
    if full_summary: txt_lines += ["OVERVIEW", full_summary, ""]
    if summary_bullets:
        txt_lines.append("KEY POINTS")
        for b in summary_bullets: txt_lines.append(f"  • {b}")
        txt_lines.append("")
    if action_items:
        txt_lines.append("ACTION ITEMS")
        for item in action_items:
            if item and isinstance(item, dict):
                txt_lines.append(f"  [ ] {item.get('task','') or ''} (Owner: {item.get('owner','') or 'TBD'} · Due: {item.get('deadline','') or 'TBD'})")
        txt_lines.append("")
    txt_lines.append("Generated by TranscriptAI · github.com/aiKunalBisht/Transcript-ai")

    st.download_button(label="⬇  Download .txt", data="\n".join(txt_lines),
                       file_name=f"meeting_{timestamp}.txt", mime="text/plain", use_container_width=True)

with col4:
    st.markdown("""
    <div class='export-card'>
      <div class='export-card-icon'>🗂️</div>
      <div class='export-card-badge badge-json'>Raw Data</div>
      <div class='export-card-title'>Full JSON Export</div>
      <div class='export-card-desc'>
        Complete raw analysis output as <strong>.json</strong>.
        All fields, scores, signals, and metadata.
        For developers or piping into other tools.
      </div>
    </div>
    """, unsafe_allow_html=True)

    export_data = {
        "exported_at": timestamp, "language": lang,
        "transcript_snippet": transcript[:200] + "..." if len(transcript) > 200 else transcript,
        "analysis": R,
    }
    st.download_button(
        label="⬇  Download .json",
        data=json.dumps(export_data, ensure_ascii=False, indent=2),
        file_name=f"meeting_{timestamp}.json", mime="application/json", use_container_width=True,
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.tai-footer-mini{margin-top:3rem;border-top:1px solid #EDE0D8;padding:1.5rem 0;text-align:center;font-size:0.75rem;color:#C8A898;}
.tai-footer-mini a{color:#D96080;text-decoration:none;font-weight:500;}
</style>
<div class='tai-footer-mini'>
  TranscriptAI by <a href='https://linkedin.com/in/kunalhere' target='_blank'>Kunal Bisht</a>
  &nbsp;·&nbsp;
  <a href='https://github.com/aiKunalBisht/Transcript-ai' target='_blank'>GitHub</a>
  &nbsp;·&nbsp;
  <a href='https://huggingface.co/spaces/KunalTheBeast/TranscriptAI' target='_blank'>Live Demo</a>
</div>
""", unsafe_allow_html=True)