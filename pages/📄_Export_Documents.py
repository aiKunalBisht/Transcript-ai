"""
pages/📄_Export_Documents.py — TranscriptAI Export Layer
Converts analyzed meeting results into downloadable documents.
Must be run after app.py has analyzed a transcript.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import json
from datetime import datetime

# ── Page config — MUST be first ─────────────────────────────────────────────
st.set_page_config(
    page_title="TranscriptAI · Export Documents",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Try importing builders ───────────────────────────────────────────────────
try:
    from agents.slide_architect import SlideArchitectAgent
    from exporters.pptx_builder import build_pptx
    PPTX_AVAILABLE = True
except ImportError:
    PPTX_AVAILABLE = False

try:
    from docx import Document as DocxDocument
    import io as _io
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# ── Same CSS as app.py (sakura palette) ─────────────────────────────────────
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
}

html, body, [class*="css"] {
    font-family: 'DM Sans', 'Noto Sans JP', sans-serif !important;
    color: var(--ink) !important;
    -webkit-font-smoothing: antialiased;
}

.stApp {
    background-color: var(--washi) !important;
    background-image:
        radial-gradient(circle at 92% 8%,  rgba(217,96,128,0.09) 0%, transparent 45%),
        radial-gradient(circle at 8%  92%, rgba(232,128,96,0.07) 0%, transparent 45%) !important;
}

[data-testid="stToolbar"],
[data-testid="stHeader"],
[data-testid="stDecoration"],
header[data-testid="stHeader"] {
    display: none !important;
}

[data-testid="stSidebar"] {
    background-color: #FDF8F5 !important;
    border-right: 1px solid var(--border) !important;
    box-shadow: 2px 0 20px rgba(60,36,22,0.06) !important;
}
[data-testid="stSidebar"] * { color: var(--ink) !important; }

.block-container {
    background: transparent !important;
    padding-top: 1rem !important;
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
    transition: all 0.2s !important;
    box-shadow: 0 2px 8px rgba(217,96,128,0.30) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, var(--sakura-deep) 0%, #A03050 100%) !important;
    box-shadow: 0 6px 20px rgba(217,96,128,0.40) !important;
    transform: translateY(-1px) !important;
}

[data-testid="stDownloadButton"] button {
    background-color: transparent !important;
    color: var(--sakura-deep) !important;
    border: 1.5px solid var(--sakura-light) !important;
    box-shadow: none !important;
    font-size: 1rem !important;
    padding: 0.8rem 1.8rem !important;
    border-radius: 10px !important;
    width: 100% !important;
    transition: all 0.2s !important;
}
[data-testid="stDownloadButton"] button:hover {
    background-color: var(--sakura-bg) !important;
    border-color: var(--sakura) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 16px rgba(217,96,128,0.15) !important;
}

[data-testid="stSpinner"] > div { border-top-color: var(--sakura) !important; }

.stAlert { border-radius: 8px !important; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, var(--sakura-light), var(--sakura));
    border-radius: 999px;
}

.export-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 1.2rem;
    transition: border-color 0.25s, box-shadow 0.25s, transform 0.2s;
    box-shadow: 0 1px 3px rgba(60,36,22,0.04);
}
.export-card:hover {
    border-color: var(--sakura-light);
    box-shadow: 0 6px 24px rgba(217,96,128,0.10);
    transform: translateY(-2px);
}
.export-card-icon {
    font-size: 2.2rem;
    margin-bottom: 0.8rem;
}
.export-card-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--ink);
    margin-bottom: 0.35rem;
}
.export-card-desc {
    font-size: 0.8rem;
    color: var(--ink-soft);
    line-height: 1.65;
    margin-bottom: 1.2rem;
}
.export-card-badge {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 999px;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.badge-ppt  { background: var(--peach-bg);  color: var(--peach); border: 1px solid #F0C0A0; }
.badge-md   { background: var(--green-bg);  color: var(--green); border: 1px solid #A8C8B8; }
.badge-txt  { background: var(--sakura-pale); color: var(--sakura-deep); border: 1px solid var(--sakura-light); }
.badge-json { background: var(--gold-light); color: var(--gold); border: 1px solid #D9C090; }
.badge-coming { background: var(--border); color: var(--ink-soft); border: 1px solid var(--border-mid); }

.meta-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    flex-wrap: wrap;
    margin-bottom: 0.5rem;
}
.meta-chip {
    font-size: 0.74rem;
    padding: 0.22rem 0.8rem;
    border-radius: 999px;
    background: var(--sakura-pale);
    color: var(--sakura-deep);
    border: 1px solid var(--sakura-light);
    font-weight: 500;
}
.sh {
    font-size: 0.67rem;
    font-weight: 700;
    color: var(--ink-soft);
    letter-spacing: 0.16em;
    text-transform: uppercase;
    margin-bottom: 0.9rem;
    padding-bottom: 0.45rem;
    border-bottom: 2px solid var(--border);
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar — matches app.py ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1.8rem 0.5rem 1.2rem;'>
      <div style='font-size:1.6rem; margin-bottom:0.4rem;'>🎙️</div>
      <div style='font-size:1rem; font-weight:600; color:#3D2B1F;'>TranscriptAI</div>
      <div style='font-size:0.62rem; color:#C4A99E; letter-spacing:0.14em;
                  text-transform:uppercase; margin-top:0.2rem;'>
        Speech &amp; Meeting Intelligence
      </div>
    </div>
    <hr style='border:none; border-top:1px solid #EDE0D8; margin:0 0 1rem;'/>
    """, unsafe_allow_html=True)

    st.markdown("<div class='sh'>Navigation</div>", unsafe_allow_html=True)
    st.page_link("app.py", label="🎙️  Meeting Analysis", use_container_width=True)
    st.page_link("pages/📄_Export_Documents.py", label="📄  Export Documents", use_container_width=True)

    st.markdown("<hr style='border:none; border-top:1px solid #EDE0D8; margin:1rem 0;'/>", unsafe_allow_html=True)

    # Show what's in session state
    has_result = "analysis_result" in st.session_state and st.session_state["analysis_result"] is not None
    if has_result:
        R = st.session_state["analysis_result"]
        lang = st.session_state.get("detected_language", "en")
        spk_count = len(R.get("speakers", []))
        act_count = len(R.get("action_items", []))
        st.markdown(
            f"<div style='background:var(--green-bg);border:1px solid #A8C8B8;"
            f"border-radius:8px;padding:0.7rem 0.9rem;font-size:0.78rem;color:var(--green);margin-bottom:1rem;'>"
            f"✓ Analysis ready<br>"
            f"<span style='color:var(--ink-soft)'>"
            f"{spk_count} speakers · {act_count} actions · {lang.upper()}"
            f"</span></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='background:var(--peach-bg);border:1px solid #E8C0A0;"
            "border-radius:8px;padding:0.7rem 0.9rem;font-size:0.78rem;color:var(--peach);margin-bottom:1rem;'>"
            "No analysis yet<br>"
            "<span style='color:var(--ink-soft)'>Run analysis first</span></div>",
            unsafe_allow_html=True,
        )

    with st.expander("About exports"):
        st.markdown("""
**PPT** — Slide deck with titles, bullets, speaker notes

**Markdown** — Clean `.md` summary file

**Plain Text** — Simple `.txt` for any editor

**JSON** — Full raw analysis data
""")

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding:2rem 0 1.6rem;'>
  <div style='font-size:0.62rem; color:#C8A898; letter-spacing:0.2em;
              text-transform:uppercase; margin-bottom:0.8rem; font-weight:500;'>
    Meeting Intelligence
  </div>
  <h1 style='font-size:2.1rem; font-weight:600; color:#3C2416;
             margin:0 0 0.7rem; letter-spacing:-0.025em;'>
    Export Documents
  </h1>
  <div style='font-size:0.88rem; color:#A87868; line-height:1.6;'>
    Turn your analyzed meeting into a presentation, report, or document
    — ready to share in seconds.
  </div>
</div>
<hr style='border:none; border-top:1px solid rgba(60,36,22,0.10); margin:0 0 1.8rem;'/>
""", unsafe_allow_html=True)

# ── Guard: no analysis yet ───────────────────────────────────────────────────
if "analysis_result" not in st.session_state or st.session_state["analysis_result"] is None:
    st.markdown("""
    <div style='background:var(--surface);border:1px solid var(--border);
                border-left:4px solid var(--sakura);border-radius:0 12px 12px 0;
                padding:1.8rem 2rem;max-width:560px;margin:2rem auto;text-align:center;'>
      <div style='font-size:2rem;margin-bottom:1rem;'>🎙️</div>
      <div style='font-size:1rem;font-weight:600;color:var(--ink);margin-bottom:0.5rem;'>
        No meeting analyzed yet
      </div>
      <div style='font-size:0.85rem;color:var(--ink-soft);line-height:1.65;'>
        Go to <strong>Meeting Analysis</strong>, paste your transcript,
        and click Analyze. Then come back here to export your results.
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── We have a result — grab it ───────────────────────────────────────────────
R    = st.session_state["analysis_result"]
lang = st.session_state.get("detected_language", "en")
transcript = st.session_state.get("current_transcript", "")

# Meeting metadata row
meeting_title   = R.get("meeting_title", "Meeting Summary")
speaker_names   = [s.get("name","?") for s in R.get("speakers", [])]
action_count    = len(R.get("action_items", []))
summary_bullets = R.get("summary", [])
full_summary    = R.get("full_summary", "")
action_items    = R.get("action_items", [])
timestamp       = datetime.now().strftime("%Y-%m-%d")

chips = "".join(
    f"<span class='meta-chip'>{c}</span>"
    for c in [
        f"{len(speaker_names)} speakers",
        f"{action_count} action items",
        lang.upper(),
        timestamp,
    ]
)
st.markdown(
    f"<div class='meta-row'>{chips}</div>",
    unsafe_allow_html=True,
)

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

# ── 4 Export Cards ───────────────────────────────────────────────────────────
col1, col2 = st.columns(2, gap="large")

# ── Card 1: PowerPoint ───────────────────────────────────────────────────────
with col1:
    st.markdown("""
    <div class='export-card'>
      <div class='export-card-icon'>📊</div>
      <div class='export-card-badge badge-ppt'>PowerPoint</div>
      <div class='export-card-title'>Slide Presentation</div>
      <div class='export-card-desc'>
        Auto-generates a slide deck from your meeting —
        titles, bullet points, and speaker notes for every key topic.
        Download as <strong>.pptx</strong> and open in PowerPoint or Google Slides.
      </div>
    </div>
    """, unsafe_allow_html=True)

    if PPTX_AVAILABLE:
        if st.button("Generate Presentation", key="gen_ppt", use_container_width=True):
            groq_key = os.getenv("GROQ_API_KEY", "")
            if not groq_key:
                try:
                    groq_key = st.secrets.get("GROQ_API_KEY", "")
                except Exception:
                    groq_key = ""
            if not groq_key:
                st.error("GROQ_API_KEY not found. Add it in HuggingFace Space secrets.")
            else:
                with st.spinner("Building slide deck · ~5s"):
                    try:
                        agent = SlideArchitectAgent(groq_api_key=groq_key)
                        plan  = agent.plan(R, language=lang)
                        pptx_bytes = build_pptx(plan)
                        st.session_state["_pptx_ready"] = pptx_bytes
                        st.session_state["_pptx_slides"] = len(plan.slides)
                    except Exception as e:
                        st.error(f"Could not generate slides: {e}")

        if st.session_state.get("_pptx_ready"):
            n = st.session_state.get("_pptx_slides", "?")
            st.success(f"✓ {n} slides ready")
            st.download_button(
                label=f"⬇  Download .pptx  ({n} slides)",
                data=st.session_state["_pptx_ready"],
                file_name=f"meeting_{timestamp}.pptx",
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                use_container_width=True,
            )
    else:
        st.info("Install `python-pptx` to enable this export:\n```\npip install python-pptx\n```")

# ── Card 2: Markdown ─────────────────────────────────────────────────────────
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

    # Build markdown content
    md_lines = [
        f"# {meeting_title}",
        f"**Date:** {timestamp}  ",
        f"**Language:** {lang.upper()}  ",
        f"**Speakers:** {', '.join(speaker_names) if speaker_names else 'Unknown'}",
        "",
        "---",
        "",
        "## Summary",
    ]
    if full_summary:
        md_lines.append(full_summary)
        md_lines.append("")
    if summary_bullets:
        md_lines.append("### Key Points")
        for b in summary_bullets:
            md_lines.append(f"- {b}")
        md_lines.append("")

    if action_items:
        md_lines.append("## Action Items")
        for item in action_items:
            owner    = item.get("owner", "TBD")
            deadline = item.get("deadline", "TBD")
            task     = item.get("task", "")
            flag     = " ⚠️" if item.get("hallucination_flag") else ""
            md_lines.append(f"- [ ] **{task}**{flag}  ")
            md_lines.append(f"  Owner: {owner} · Deadline: {deadline}")
        md_lines.append("")

    sentiment = R.get("sentiment", [])
    if sentiment:
        md_lines.append("## Speaker Sentiment")
        for s in sentiment:
            md_lines.append(f"- **{s.get('speaker','')}**: {s.get('score','').upper()} — {s.get('label','')}")
        md_lines.append("")

    soft = R.get("soft_rejections", {})
    if soft and soft.get("total_signals", 0) > 0:
        md_lines.append("## Communication Signals")
        md_lines.append(f"**Risk Level:** {soft.get('risk_level','NONE')}  ")
        md_lines.append(f"**Total Signals:** {soft.get('total_signals',0)}")
        if soft.get("cultural_note"):
            md_lines.append(f"\n> {soft['cultural_note']}")
        md_lines.append("")

    md_lines.append("---")
    md_lines.append("*Generated by TranscriptAI · github.com/aiKunalBisht/Transcript-ai*")

    md_content = "\n".join(md_lines)

    st.download_button(
        label="⬇  Download .md",
        data=md_content,
        file_name=f"meeting_{timestamp}.md",
        mime="text/markdown",
        use_container_width=True,
    )

# ── Row 2 ────────────────────────────────────────────────────────────────────
col3, col4 = st.columns(2, gap="large")

# ── Card 3: Plain Text ───────────────────────────────────────────────────────
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

    txt_lines = [
        f"MEETING SUMMARY — {timestamp}",
        "=" * 40,
        "",
    ]
    if full_summary:
        txt_lines += ["OVERVIEW", full_summary, ""]
    if summary_bullets:
        txt_lines.append("KEY POINTS")
        for b in summary_bullets:
            txt_lines.append(f"  • {b}")
        txt_lines.append("")
    if action_items:
        txt_lines.append("ACTION ITEMS")
        for item in action_items:
            txt_lines.append(
                f"  [ ] {item.get('task','')} "
                f"(Owner: {item.get('owner','TBD')} · Due: {item.get('deadline','TBD')})"
            )
        txt_lines.append("")
    txt_lines.append("Generated by TranscriptAI")

    txt_content = "\n".join(txt_lines)

    st.download_button(
        label="⬇  Download .txt",
        data=txt_content,
        file_name=f"meeting_{timestamp}.txt",
        mime="text/plain",
        use_container_width=True,
    )

# ── Card 4: JSON ─────────────────────────────────────────────────────────────
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
        "exported_at": timestamp,
        "language": lang,
        "transcript_snippet": transcript[:200] + "..." if len(transcript) > 200 else transcript,
        "analysis": R,
    }
    json_content = json.dumps(export_data, ensure_ascii=False, indent=2)

    st.download_button(
        label="⬇  Download .json",
        data=json_content,
        file_name=f"meeting_{timestamp}.json",
        mime="application/json",
        use_container_width=True,
    )

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.tai-footer-mini {
    margin-top: 3rem;
    border-top: 1px solid #EDE0D8;
    padding: 1.5rem 0;
    text-align: center;
    font-size: 0.75rem;
    color: #C8A898;
}
.tai-footer-mini a {
    color: #D96080;
    text-decoration: none;
    font-weight: 500;
}
</style>
<div class='tai-footer-mini'>
  TranscriptAI by <a href='https://linkedin.com/in/kunalhere' target='_blank'>Kunal Bisht</a>
  &nbsp;·&nbsp;
  <a href='https://github.com/aiKunalBisht/Transcript-ai' target='_blank'>GitHub</a>
  &nbsp;·&nbsp;
  <a href='https://huggingface.co/spaces/KunalTheBeast/TranscriptAI' target='_blank'>Live Demo</a>
</div>
""", unsafe_allow_html=True)