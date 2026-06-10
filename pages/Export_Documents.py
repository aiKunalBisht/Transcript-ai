"""
pages/Export_Documents.py — TranscriptAI Export Layer
"""
import sys, os
import pathlib

# Add project root to path so agents/ and exporters/ are importable
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

# ── CSS — full sakura palette ────────────────────────────────────────────────
st.markdown("""
<style>

[data-testid="stSidebarNav"],
[data-testid="stSidebarNavItems"],
[data-testid="stSidebarNavLink"],
section[data-testid="stSidebar"] > div:first-child > div > ul,
section[data-testid="stSidebar"] nav {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    overflow: hidden !important;
}

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=Noto+Sans+JP:wght@400;500&display=swap');
:root {
    --washi:#FAF6F2; --surface:#FFFEFB; --border:#EFE2D8; --border-mid:#E5D0C4;
    --ink:#3C2416; --ink-mid:#7A5040; --ink-soft:#A87868; --ink-faint:#C8A898;
    --sakura:#D96080; --sakura-deep:#BE4060; --sakura-bg:#FDEEF2;
    --sakura-pale:#FEF6F8; --sakura-light:#F2B0C0;
    --peach:#E88060; --peach-bg:#FDF0EA;
    --gold:#B87830; --gold-light:#F5E0C0;
    --green:#486858; --green-bg:#EDF3EF;
}
html,body,[class*="css"]{font-family:'DM Sans','Noto Sans JP',sans-serif!important;color:var(--ink)!important;-webkit-font-smoothing:antialiased;}
.stApp{background-color:var(--washi)!important;background-image:radial-gradient(circle at 92% 8%,rgba(217,96,128,0.09) 0%,transparent 45%),radial-gradient(circle at 8% 92%,rgba(232,128,96,0.07) 0%,transparent 45%)!important;}
[data-testid="stToolbar"],[data-testid="stHeader"],[data-testid="stDecoration"],header[data-testid="stHeader"]{display:none!important;}
[data-testid="stSidebar"]{background-color:#FDF8F5!important;border-right:1px solid var(--border)!important;box-shadow:2px 0 20px rgba(60,36,22,0.06)!important;}
[data-testid="stSidebar"] *{color:var(--ink)!important;}
.block-container{background:transparent!important;padding-top:1rem!important;}
.stButton>button{background:linear-gradient(135deg,var(--sakura) 0%,var(--sakura-deep) 100%)!important;color:#FFFDFB!important;border:none!important;border-radius:8px!important;font-family:'DM Sans',sans-serif!important;font-weight:600!important;font-size:0.9rem!important;padding:0.7rem 1.6rem!important;transition:all 0.2s!important;box-shadow:0 2px 8px rgba(217,96,128,0.30)!important;}
.stButton>button:hover{background:linear-gradient(135deg,var(--sakura-deep) 0%,#A03050 100%)!important;box-shadow:0 6px 20px rgba(217,96,128,0.40)!important;transform:translateY(-1px)!important;}
[data-testid="stDownloadButton"] button{background-color:transparent!important;color:var(--sakura-deep)!important;border:1.5px solid var(--sakura-light)!important;box-shadow:none!important;font-size:0.95rem!important;padding:0.75rem 1.6rem!important;border-radius:10px!important;width:100%!important;transition:all 0.2s!important;font-weight:500!important;}
[data-testid="stDownloadButton"] button:hover{background-color:var(--sakura-bg)!important;border-color:var(--sakura)!important;transform:translateY(-2px)!important;box-shadow:0 4px 16px rgba(217,96,128,0.15)!important;}
[data-testid="stSpinner"]>div{border-top-color:var(--sakura)!important;}
.stAlert{border-radius:8px!important;}
::-webkit-scrollbar{width:4px;}
::-webkit-scrollbar-thumb{background:linear-gradient(180deg,var(--sakura-light),var(--sakura));border-radius:999px;}
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
.meta-row{display:flex;align-items:center;gap:0.8rem;flex-wrap:wrap;margin-bottom:0.5rem;}
.meta-chip{font-size:0.74rem;padding:0.22rem 0.8rem;border-radius:999px;background:var(--sakura-pale);color:var(--sakura-deep);border:1px solid var(--sakura-light);font-weight:500;}
.sh{font-size:0.67rem;font-weight:700;color:var(--ink-soft);letter-spacing:0.16em;text-transform:uppercase;margin-bottom:0.9rem;padding-bottom:0.45rem;border-bottom:2px solid var(--border);}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
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
        R2       = st.session_state["analysis_result"]
        lang2    = st.session_state.get("detected_language", "en")
        spk_cnt  = len(R2.get("speakers", []))
        act_cnt  = len(R2.get("action_items", []))
        st.markdown(
            f"<div style='background:var(--green-bg);border:1px solid #A8C8B8;"
            f"border-radius:8px;padding:0.7rem 0.9rem;font-size:0.78rem;color:var(--green);margin-bottom:1rem;'>"
            f"✓ Analysis ready<br>"
            f"<span style='color:var(--ink-soft)'>{spk_cnt} speakers · {act_cnt} actions · {lang2.upper()}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='background:var(--peach-bg);border:1px solid #E8C0A0;"
            "border-radius:8px;padding:0.7rem 0.9rem;font-size:0.78rem;color:var(--peach);margin-bottom:1rem;'>"
            "No analysis yet<br><span style='color:var(--ink-soft)'>Run analysis first</span></div>",
            unsafe_allow_html=True,
        )

    with st.expander("About exports"):
        st.markdown("""
**PPT** — Slide deck with titles, bullets, speaker notes

**Markdown** — Clean `.md` for Notion, GitHub, Obsidian

**Plain Text** — Simple `.txt` for email or Slack

**JSON** — Full raw analysis data for developers
""")

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding:2rem 0 1.6rem;'>
  <div style='font-size:0.62rem;color:#C8A898;letter-spacing:0.2em;text-transform:uppercase;margin-bottom:0.8rem;font-weight:500;'>
    Meeting Intelligence
  </div>
  <h1 style='font-size:2.1rem;font-weight:600;color:#3C2416;margin:0 0 0.7rem;letter-spacing:-0.025em;'>
    Export Documents
  </h1>
  <div style='font-size:0.88rem;color:#A87868;line-height:1.6;'>
    Turn your analyzed meeting into a presentation, report, or document — ready to share in seconds.
  </div>
</div>
<hr style='border:none;border-top:1px solid rgba(60,36,22,0.10);margin:0 0 1.8rem;'/>
""", unsafe_allow_html=True)

# ── Guard ────────────────────────────────────────────────────────────────────
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

# ── Data ─────────────────────────────────────────────────────────────────────
R          = st.session_state["analysis_result"]
lang       = st.session_state.get("detected_language", "en")
transcript = st.session_state.get("current_transcript", "")

meeting_title   = R.get("meeting_title", "Meeting Summary")
speaker_names   = [s.get("name", "?") for s in R.get("speakers", [])]
action_count    = len(R.get("action_items", []))
summary_bullets = R.get("summary", [])
full_summary    = R.get("full_summary", "")
action_items    = R.get("action_items", [])
timestamp       = datetime.now().strftime("%Y-%m-%d")

chips = "".join(
    f"<span class='meta-chip'>{c}</span>"
    for c in [f"{len(speaker_names)} speakers", f"{action_count} action items", lang.upper(), timestamp]
)
st.markdown(f"<div class='meta-row'>{chips}</div>", unsafe_allow_html=True)
st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

# ── Row 1 ────────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2, gap="large")

# Card 1: PowerPoint
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
                try:
                    groq_key = st.secrets.get("GROQ_API_KEY", "")
                except Exception:
                    groq_key = ""
            if not groq_key:
                st.error(
                    "GROQ_API_KEY not found.  \n"
                    "**Local:** add `GROQ_API_KEY=your_key` to your `.env` file.  \n"
                    "**HuggingFace:** add it in Space Settings → Repository secrets."
                )
            else:
                with st.spinner("Building slide deck · calling Groq · ~5s"):
                    try:
                        agent      = SlideArchitectAgent(groq_api_key=groq_key)
                        plan       = agent.plan(R, language=lang)
                        pptx_bytes = build_pptx(plan)
                        st.session_state["_pptx_ready"]  = pptx_bytes
                        st.session_state["_pptx_slides"] = len(plan.slides)
                        st.session_state["_pptx_title"]  = plan.meeting_title
                    except Exception as e:
                        import traceback
                        st.error(f"Slide generation failed: {e}")
                        st.code(traceback.format_exc())

        if st.session_state.get("_pptx_ready"):
            n     = st.session_state.get("_pptx_slides", "?")
            title = st.session_state.get("_pptx_title", "meeting")
            safe  = title.replace(" ", "_")[:40]
            st.success(f"✓ {n} slides ready — {title}")
            st.download_button(
                label=f"⬇  Download .pptx  ({n} slides)",
                data=st.session_state["_pptx_ready"],
                file_name=f"{safe}_{timestamp}.pptx",
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                use_container_width=True,
            )

# Card 2: Markdown
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

    md_lines = [
        f"# {meeting_title}",
        f"**Date:** {timestamp}  ",
        f"**Language:** {lang.upper()}  ",
        f"**Speakers:** {', '.join(speaker_names) if speaker_names else 'Unknown'}",
        "", "---", "", "## Summary",
    ]
    if full_summary:
        md_lines += [full_summary, ""]
    if summary_bullets:
        md_lines.append("### Key Points")
        for b in summary_bullets:
            md_lines.append(f"- {b}")
        md_lines.append("")
    if action_items:
        md_lines.append("## Action Items")
        for item in action_items:
            flag = " ⚠️" if item.get("hallucination_flag") else ""
            md_lines.append(f"- [ ] **{item.get('task','')}**{flag}  ")
            md_lines.append(f"  Owner: {item.get('owner','TBD')} · Deadline: {item.get('deadline','TBD')}")
        md_lines.append("")
    sentiment = R.get("sentiment", [])
    if sentiment:
        md_lines.append("## Speaker Sentiment")
        for s in sentiment:
            md_lines.append(f"- **{s.get('speaker','')}**: {s.get('score','').upper()} — {s.get('label','')}")
        md_lines.append("")
    soft = R.get("soft_rejections", {})
    if soft and soft.get("total_signals", 0) > 0:
        md_lines += [
            "## Communication Signals",
            f"**Risk Level:** {soft.get('risk_level','NONE')}  ",
            f"**Total Signals:** {soft.get('total_signals',0)}",
        ]
        if soft.get("cultural_note"):
            md_lines.append(f"\n> {soft['cultural_note']}")
        md_lines.append("")
    md_lines += ["---", "*Generated by TranscriptAI · github.com/aiKunalBisht/Transcript-ai*"]
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

# Card 3: Plain Text
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
    txt_lines.append("Generated by TranscriptAI · github.com/aiKunalBisht/Transcript-ai")

    st.download_button(
        label="⬇  Download .txt",
        data="\n".join(txt_lines),
        file_name=f"meeting_{timestamp}.txt",
        mime="text/plain",
        use_container_width=True,
    )

# Card 4: JSON
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

    st.download_button(
        label="⬇  Download .json",
        data=json.dumps(export_data, ensure_ascii=False, indent=2),
        file_name=f"meeting_{timestamp}.json",
        mime="application/json",
        use_container_width=True,
    )

# ── 議事録 card — INSERT this as Row 3 in Export_Documents.py ────────────────
# Place this block AFTER the col3/col4 row and BEFORE the footer.
# Import at top of file (add alongside other imports):
#   from agents.gijiroku_formatter import GijirokulFormatter, render_markdown, render_text

try:
    from agents.gijiroku_formatter import GijirokulFormatter, render_markdown, render_text
    GIJIROKU_AVAILABLE = True
except Exception as _g_err:
    GIJIROKU_AVAILABLE = False
    _g_err_msg = str(_g_err)

# ── Row 3: 議事録 ─────────────────────────────────────────────────────────────
st.markdown("<div style='height:0.2rem'></div>", unsafe_allow_html=True)
col5, col6 = st.columns(2, gap="large")

with col5:
    st.markdown("""
    <div class='export-card' style='border-color:#D0B0C8;'>
      <div class='export-card-icon'>📋</div>
      <div class='export-card-badge' style='background:#F5EEF8;color:#7D4E8A;border:1px solid #D0B0C8;'>
        議事録 · Gijiroku
      </div>
      <div class='export-card-title'>Japanese Business Meeting Minutes</div>
      <div class='export-card-desc'>
        Formats your analysis as a formal <strong>議事録</strong> — the standard
        Japanese enterprise meeting document. Includes 会議名, 出席者, 議題,
        決定事項, アクションアイテム, and 次回予定.
        Download as <strong>.md</strong> or <strong>.txt</strong>.
        Compatible with Notion, Confluence, and email.
      </div>
    </div>
    """, unsafe_allow_html=True)

    if not GIJIROKU_AVAILABLE:
        st.error(f"議事録 module not loaded: {_g_err_msg}")
    else:
        with st.expander("⚙️ Options", expanded=False):
            g_recorder   = st.text_input("記録者 (Recorder)", value="TranscriptAI", key="g_recorder")
            g_basho      = st.text_input("場所 (Location)", value="オンライン会議 / Online", key="g_basho")
            g_jikai      = st.text_input("次回予定 (Next meeting)", value="未定 / TBD", key="g_jikai")
            g_format     = st.radio("Export format", ["Markdown (.md)", "Plain Text (.txt)"], key="g_fmt", horizontal=True)

        if st.button("生成する Generate 議事録", key="gen_gijiroku", use_container_width=True):
            with st.spinner("議事録を生成中 · Formatting..."):
                try:
                    formatter = GijirokulFormatter()
                    plan_g = formatter.format(
                        analysis=R,
                        recorder=g_recorder,
                        basho=g_basho,
                        jikai_yotei=g_jikai,
                        timestamp=datetime.now().strftime("%Y年%m月%d日 %H:%M"),
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
                    st.session_state["_gijiroku_title"]   = plan_g.kaigi_mei
                    st.success("✓ 議事録が生成されました / Document ready")

                    # Preview
                    with st.expander("プレビュー Preview", expanded=True):
                        if ext == "md":
                            st.markdown(content)
                        else:
                            st.code(content, language=None)

                except Exception as e:
                    import traceback
                    st.error(f"議事録生成エラー: {e}")
                    st.code(traceback.format_exc())

        if st.session_state.get("_gijiroku_content"):
            safe_title = st.session_state.get("_gijiroku_title", "meeting").replace(" ", "_")[:40]
            ext  = st.session_state.get("_gijiroku_ext", "md")
            mime = st.session_state.get("_gijiroku_mime", "text/markdown")
            st.download_button(
                label=f"⬇  議事録をダウンロード  (.{ext})",
                data=st.session_state["_gijiroku_content"].encode("utf-8"),
                file_name=f"gijiroku_{safe_title}_{timestamp}.{ext}",
                mime=mime,
                use_container_width=True,
            )

with col6:
    st.markdown("""
    <div class='export-card' style='border-color:#D0B0C8;background:#FDFAFF;'>
      <div class='export-card-icon'>🗾</div>
      <div class='export-card-badge' style='background:#F5EEF8;color:#7D4E8A;border:1px solid #D0B0C8;'>
        フォーマット解説
      </div>
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



# ── Footer ───────────────────────────────────────────────────────────────────
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