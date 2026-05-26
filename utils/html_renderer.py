from utils.utils import language_display_name

# ═══════════════════════════════════════════════════════════════
# RESULTS RENDERER v2 — single HTML blob for 10K user scale
# All result display is built as one string → one st.markdown call
# vs the old 50+ individual st.markdown calls (50x WebSocket msgs)
# ═══════════════════════════════════════════════════════════════


def _svg_donut(pct: int, color: str, size: int = 56) -> str:
    """SVG donut chart."""
    r = (size - 8) // 2
    circ = 2 * 3.14159 * r
    dash = circ * pct / 100
    return (
        f"<svg width='{size}' height='{size}' viewBox='0 0 {size} {size}'>"
        f"<circle cx='{size//2}' cy='{size//2}' r='{r}' fill='none' "
        f"stroke='rgba(0,0,0,0.08)' stroke-width='6'/>"
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
        f"<circle cx='60' cy='60' r='{r}' fill='none' stroke='rgba(0,0,0,0.08)' stroke-width='10'/>"
        f"<circle cx='60' cy='60' r='{r}' fill='none' stroke='{color}' stroke-width='10' "
        f"stroke-linecap='round' stroke-dasharray='{dash:.1f} {circ:.1f}' "
        f"transform='rotate(-90 60 60)' style='filter:drop-shadow(0 0 6px {color}88)'/>"
        f"<text x='50%' y='46%' text-anchor='middle' font-size='22' font-weight='800' "
        f"fill='#2C1810' font-family='Arial'>{score}</text>"
        f"<text x='50%' y='62%' text-anchor='middle' font-size='10' fill='#8B7060' "
        f"font-family='Arial'>/ 100</text></svg>"
        f"<div style='font-size:0.7rem;font-weight:600;color:{color};letter-spacing:0.1em;"
        f"text-transform:uppercase;margin-top:2px'>{label}</div></div>"
    )


def build_results_html(R: dict, language: str, features: dict,
                       pii_rep: dict | None) -> str:
    """
    Builds the ENTIRE results view as a single HTML string.
    One st.markdown() call = one WebSocket message = scales to 10K users.
    """
    COLORS = ["#E8829A", "#F4A07A", "#C9924A", "#5A7D6B", "#A8897C", "#7A5C50"]
    SENT_COLOR = {"positive": "#2D9E6B", "neutral": "#A8897C", "negative": "#C84040"}
    SENT_ICON = {"positive": "🌸", "neutral": "🌿", "negative": "🍂"}

    ji = R.get("japan_insights", {})
    speakers = sorted(R.get("speakers", []),
                      key=lambda s: s.get("talk_time_pct", 0), reverse=True)

    def _health():
        score = 0
        soft = R.get("soft_rejections", {})
        risk = soft.get("risk_level", "NONE")
        risk_pts = {"NONE": 25, "MINIMAL": 20, "LOW": 15, "MEDIUM": 8, "HIGH": 0}
        sents = R.get("sentiment", [])
        w = {"positive": 1.0, "neutral": 0.6, "negative": 0.1}
        s_pts = round((sum(w.get(s.get("score", "neutral").lower(), 0.5)
                           for s in sents) / len(sents) * 30) if sents else 15)
        items = R.get("action_items", [])
        if not items:
            a_pts = 10
        else:
            ver = [i for i in items if not i.get("hallucination_flag")]
            wo = sum(1 for i in ver if i.get("owner", "TBD") not in ("TBD", "Unknown", ""))
            wd = sum(1 for i in ver if i.get("deadline", "TBD") not in ("TBD", "N/A", ""))
            a_pts = round((wo + wd) / (2 * len(items)) * 25)
        r_pts = risk_pts.get(risk, 25)
        ver2 = R.get("verification", {})
        h_pts = round((1 - ver2.get("overall_hallucination_risk", 0)) * 20)
        score = min(s_pts + a_pts + r_pts + h_pts, 100)
        color = ("#2D9E6B" if score >= 80 else "#B87830" if score >= 60
                 else "#D96080" if score >= 40 else "#C84040")
        bd = [("Sentiment", s_pts, 30), ("Action Clarity", a_pts, 25),
              ("Comm Risk", r_pts, 25), ("AI Confidence", h_pts, 20)]
        bars = "".join(
            f"<div style='margin-bottom:8px'>"
            f"<div style='display:flex;justify-content:space-between;margin-bottom:3px'>"
            f"<span style='font-size:0.68rem;color:#5A4030'>{lb}</span>"
            f"<span style='font-size:0.68rem;color:{color};font-weight:600'>{pt}/{tot}</span></div>"
            f"<div style='height:5px;background:rgba(0,0,0,0.06);border-radius:999px'>"
            f"<div style='height:100%;width:{round(pt / tot * 100)}%;background:{color};"
            f"border-radius:999px;box-shadow:0 0 6px {color}66'></div></div></div>"
            for lb, pt, tot in bd
        )
        return score, color, bars

    score, hc, hbars = _health()

    spk_count = len(R.get("speakers", []))
    act_count = len(R.get("action_items", []))
    cs_val = ji.get("code_switch_count", "—") if features.get("show_code_switch") else "—"
    keigo_val = ji.get("keigo_level", "—").title() if features.get("show_japan_insights") else language_display_name(language).split(" ", 1)[-1]
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
            f"{n} item{'s' if n != 1 else ''} anonymized before analysis</div>"
        )

    full_sum = R.get("full_summary", "")
    bullets = R.get("summary", [])
    sum_html = ""
    if full_sum:
        sum_html += (
            f"<div class='tai-summary-box'>"
            f"<div class='tai-summary-label'>📋 Meeting Overview</div>"
            f"<p style='margin:0;line-height:1.9;font-size:0.93rem;color:#2C1810'>"
            f"{full_sum}</p></div>"
        )
    sum_html += f"<div class='tai-section-label'>{len(bullets)} Key Points</div>"
    sum_html += "".join(
        f"<div class='tai-bullet-card'>"
        f"<span class='tai-bullet-num'>{i:02d}</span>"
        f"<span style='color:#2C1810;font-size:0.9rem;line-height:1.65'>{b}</span>"
        f"</div>"
        for i, b in enumerate(bullets, 1)
    )

    items = R.get("action_items", [])
    v_count = sum(1 for i in items if not i.get("hallucination_flag"))
    f_count = len(items) - v_count
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
            "<div style='font-weight:600;color:#2C1810;font-size:0.9rem;margin-bottom:4px'>" +
            str(i.get("task", "")) + "</div>"
            "<div style='font-size:0.76rem;color:#6B5548'>"
            "Owner: <strong style='color:#3C2416'>" + str(i.get("owner", "TBD")) + "</strong>"
            " &nbsp;·&nbsp; Deadline: <strong style='color:#3C2416'>" + str(i.get("deadline", "TBD")) + "</strong>" +
            (f" &nbsp;·&nbsp; {i.get('confidence', 0):.0%} confidence" if i.get("confidence") else "") +
            (f"<div style='color:#C84040;font-size:0.72rem;margin-top:3px'>⚠ {i.get('flag_reason', '')}</div>" if i.get("flag_reason") else "") +
            "</div></div></div>"
        )
        for i in items
    ) if items else "<div style='color:#A8897C;font-size:0.85rem;padding:1rem 0'>No action items extracted.</div>"

    sent_html = "<div class='tai-section-label'>Speaker Sentiment</div>"
    sent_html += "".join(
        (
            "<div class='tai-sent-row'>"
            "<span style='font-size:1.2rem'>" + SENT_ICON.get(s.get("score", "neutral").lower(), "🌿") + "</span>"
            "<div style='flex:1'>"
            "<div style='font-weight:600;color:#2C1810;font-size:0.88rem'>" + str(s.get("speaker", "")) + "</div>"
            "<div style='font-size:0.75rem;color:#6B5548;font-style:italic;margin-top:1px'>" + str(s.get("label", "")) + "</div>"
            "</div>"
            "<span class='tai-sent-badge tai-sent-" + s.get("score", "neutral").lower() + "'>" +
            s.get("score", "neutral").upper() + "</span>"
            "</div>"
        )
        for s in R.get("sentiment", [])
    )

    spk_html = "<div class='tai-section-label'>Talk Time Distribution</div>"
    for idx2, spk in enumerate(speakers):
        nm = spk.get("name", f"Speaker {idx2 + 1}")
        pct = spk.get("talk_time_pct", 0)
        tone = spk.get("tone", "—")
        col = COLORS[idx2 % len(COLORS)]
        spk_html += (
            f"<div class='tai-spk-row'>"
            f"{_avatar(nm, col)}"
            f"<div style='flex:1;min-width:0'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:6px'>"
            f"<span style='font-weight:600;color:#2C1810;font-size:0.88rem'>{nm}</span>"
            f"<span style='font-size:0.75rem;color:{col};font-weight:600'>{pct}%</span></div>"
            f"<div style='height:6px;background:rgba(0,0,0,0.06);border-radius:999px'>"
            f"<div style='height:100%;width:{pct}%;background:{col};border-radius:999px;"
            f"box-shadow:0 0 8px {col}66;transition:width 0.8s cubic-bezier(0.4,0,0.2,1)'></div></div>"
            f"<div style='font-size:0.7rem;color:#8B7060;margin-top:4px'>{tone}</div>"
            f"</div>"
            f"{_svg_donut(pct, col, 52)}"
            f"</div>"
        )

    ins_html = ""
    if features.get("show_japan_insights"):
        keigo = ji.get("keigo_level", "—")
        k_src = ji.get("keigo_source", "llm")
        kc = {"high": "#E8829A", "medium": "#C9924A", "low": "#8B7060"}.get(keigo, "#A8897C")
        sigs = ji.get("nemawashi_signals", [])
        soft = R.get("soft_rejections", {})
        risk = soft.get("risk_level", "NONE") if soft else "NONE"
        rclr = {"HIGH": "#C84040", "MEDIUM": "#C9924A", "LOW": "#D96080", "MINIMAL": "#A8897C", "NONE": "#2D9E6B"}.get(risk, "#2D9E6B")
        cs_cnt = ji.get("code_switch_count", 0)

        ins_html += (
            "<div style='display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px'>"
            "<div class='tai-insight-chip'>"
            "<div style='font-size:0.6rem;color:#8B7060;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:2px'>Keigo Register</div>"
            f"<div style='font-size:1.1rem;font-weight:700;color:{kc}'>{keigo.upper()}</div>"
            f"<div style='font-size:0.62rem;color:#A8897C'>via {k_src}</div>"
            "</div>"
            "<div class='tai-insight-chip'>"
            "<div style='font-size:0.6rem;color:#8B7060;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:2px'>Rejection Risk</div>"
            f"<div style='font-size:1.1rem;font-weight:700;color:{rclr}'>{risk}</div>"
            f"<div style='font-size:0.62rem;color:#A8897C'>{soft.get('total_signals', 0)} signals</div>"
            "</div>"
            "<div class='tai-insight-chip'>"
            "<div style='font-size:0.6rem;color:#8B7060;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:2px'>Code Switches</div>"
            f"<div style='font-size:1.1rem;font-weight:700;color:#F4A07A'>{cs_cnt}</div>"
            "<div style='font-size:0.62rem;color:#A8897C'>language switches</div>"
            "</div>"
            "</div>"
        )

        if sigs:
            ins_html += f"<div class='tai-section-label'>Indirect Consensus Signals · {len(sigs)} detected</div>"
            ins_html += "".join(
                "<div class='tai-nemawashi-pill'>◆ " + s + "</div>" for s in sigs
            )

        if soft and soft.get("total_signals", 0) > 0:
            ins_html += "<div class='tai-section-label' style='margin-top:16px'>Soft Rejection Analysis</div>"
            for sig in soft.get("high_signals", []):
                ins_html += (
                    f"<div class='tai-sig-high'>"
                    f"<div style='font-weight:700;font-size:0.9rem'>🚨 {sig['phrase']}</div>"
                    f"<div style='font-size:0.76rem;color:#6B5548;margin-top:4px'>"
                    f"{sig['reading']} · {sig['speaker']} · {sig['confidence']:.0%}</div>"
                    f"<div style='font-size:0.75rem;color:#3C2416;margin-top:6px;line-height:1.5'>{sig['explanation']}</div>"
                    f"</div>"
                )
            for sig in soft.get("medium_signals", []):
                ins_html += (
                    f"<div class='tai-sig-med'>"
                    f"<div style='font-weight:700;font-size:0.9rem'>⚠ {sig['phrase']}</div>"
                    f"<div style='font-size:0.76rem;color:#6B5548;margin-top:4px'>"
                    f"{sig['reading']} · {sig['speaker']} · {sig['confidence']:.0%}</div>"
                    f"<div style='font-size:0.75rem;color:#3C2416;margin-top:6px;line-height:1.5'>{sig['explanation']}</div>"
                    f"</div>"
                )
            ins_html += f"<div style='font-size:0.73rem;color:#A8897C;font-style:italic;margin-top:8px'>{soft.get('cultural_note', '')}</div>"
    else:
        ins_html = "<div style='color:#A8897C;font-size:0.85rem;padding:1rem 0;line-height:1.7'>Cultural intelligence features apply to Japanese and Hindi transcripts.</div>"

    return f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Noto+Sans+JP:wght@400;500;700&display=swap" rel="stylesheet">
<style>
:root {{
  --glass:     rgba(60,36,22,0.03);
  --glass-b:   rgba(60,36,22,0.10);
  --glass-h:   rgba(60,36,22,0.14);
  --accent:    #E8829A;
  --accent2:   #F4A07A;
  --green:     #2D9E6B;
  --ink:       #2C1810;
  --ink-mid:   #5A4030;
  --ink-soft:  #8B7060;
  --r:         14px;
}}
.tai-results {{
  font-family: 'Inter','Noto Sans JP',sans-serif;
  color: var(--ink);
  padding: 24px 0 2rem;
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
  border-color: rgba(232,130,154,0.3);
  box-shadow: 0 8px 32px rgba(232,130,154,0.08);
}}
.tai-tiles {{
  display: grid;
  grid-template-columns: repeat(4,1fr);
  gap: 12px;
  margin-bottom: 16px;
}}
@media(max-width:768px) {{ .tai-tiles {{ grid-template-columns: repeat(2,1fr); }} }}
@media(max-width:380px) {{ .tai-tiles {{ grid-template-columns: repeat(2,1fr); }} }}
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
  border-color: rgba(232,130,154,0.35);
  box-shadow: 0 12px 32px rgba(232,130,154,0.12);
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
.tai-health-left  {{ padding: 24px 20px; display: flex; align-items: center; justify-content: center; border-right: 1px solid rgba(0,0,0,0.06); }}
.tai-health-right {{ padding: 20px 24px; }}
.tai-health-title {{ font-size: 0.6rem; font-weight: 700; color: var(--accent); letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 14px; }}
@media(max-width:600px) {{ .tai-health-left {{ border-right: none; border-bottom: 1px solid rgba(0,0,0,0.06); }} }}
.tai-tab-bar {{
  display: flex; gap: 4px;
  border-bottom: 1px solid rgba(0,0,0,0.08);
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
  border-bottom: 1px solid rgba(0,0,0,0.07);
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
.tai-action-card:hover {{ transform: translateX(4px); border-left-color: #C84060; }}
.tai-action-flagged {{ border-left-color: #C84040 !important; background: rgba(200,64,64,0.06) !important; color: #C84040 !important; }}
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
.tai-sent-positive {{ background: rgba(45,158,107,0.15); color: #2D9E6B; border: 1px solid rgba(45,158,107,0.3); }}
.tai-sent-neutral  {{ background: rgba(168,136,122,0.15); color: #8B7060; border: 1px solid rgba(168,136,122,0.3); }}
.tai-sent-negative {{ background: rgba(200,64,64,0.15);  color: #C84040; border: 1px solid rgba(200,64,64,0.3); }}
.tai-spk-row {{
  display: flex; align-items: center; gap: 14px;
  background: var(--glass);
  border: 1px solid var(--glass-b);
  border-radius: 10px; padding: 14px 16px; margin-bottom: 10px;
  transition: border-color 0.2s;
}}
.tai-spk-row:hover {{ border-color: rgba(232,130,154,0.25); }}
.tai-insight-chip {{
  background: var(--glass);
  border: 1px solid var(--glass-b);
  border-radius: 10px; padding: 12px 16px;
  min-width: 100px; flex: 1;
}}
.tai-nemawashi-pill {{
  display: inline-block;
  background: rgba(232,130,154,0.1);
  border: 1px solid rgba(232,130,154,0.25);
  border-radius: 999px; padding: 5px 14px;
  font-size: 0.82rem; color: #C45C74;
  font-family: 'Noto Sans JP', sans-serif;
  margin: 0 6px 8px 0;
}}
.tai-sig-high {{
  background: rgba(200,64,64,0.06); border-left: 3px solid #C84040;
  border-radius: 0 10px 10px 0; padding: 12px 16px; margin-bottom: 10px;
  color: #3C2416;
}}
.tai-sig-med {{
  background: rgba(201,146,74,0.06); border-left: 3px solid #C9924A;
  border-radius: 0 10px 10px 0; padding: 12px 16px; margin-bottom: 10px;
  color: #3C2416;
}}
.tai-pii-pill {{
  display: inline-flex; align-items: center; gap: 6px;
  background: rgba(45,158,107,0.12); border: 1px solid rgba(45,158,107,0.3);
  border-radius: 999px; padding: 5px 14px;
  font-size: 0.73rem; color: #2D9E6B; font-weight: 500; margin-bottom: 14px;
}}
.tai-panel {{
  background: var(--glass);
  border: 1px solid var(--glass-b);
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
  <button class="tai-tab" onclick="taiTab(this,'ins')">{features.get('insight_tab_label', '🌐 Insights')}</button>
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
