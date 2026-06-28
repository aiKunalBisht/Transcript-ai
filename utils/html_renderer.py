"""
utils/html_renderer.py  — TranscriptAI v3.1
============================================
v3.1 fixes:
  - Health score caps at 22 for explicit contract termination meetings
  - CRITICAL risk level added to all color maps
  - Termination detected banner in Insights tab (purple, distinct from soft-rejection)
  - Unlabeled transcript warning banner
  - Sentiment scoring concept updated: communicative register, not emotional valence
    (affects the label strings shown in the Sentiment tab subtitle)
"""
from utils.utils import language_display_name


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
    # Override label for very low scores (termination)
    if score <= 22:
        label = "Terminated"
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
            + ("" if not a.flag else "<span style='color:#963030'> ⚠</span>") + "</td>"
            f"<td style='padding:5px 10px;font-size:0.75rem;color:#A87868;border-bottom:1px solid #EFE2D8;'>{a.deadline}</td></tr>"
            for a in plan.action_items[:4]
        )

        soft = R.get("soft_rejections", {}) or {}
        risk = soft.get("risk_level", "NONE")
        risk_colors = {"CRITICAL":"#7C3AED","HIGH":"#963030","MEDIUM":"#986820","LOW":"#BE4060","MINIMAL":"#A87868","NONE":"#2D7A55"}
        risk_bgs    = {"CRITICAL":"#F5F3FF","HIGH":"#FAF0F0","MEDIUM":"#FAF0E0","LOW":"#FEF6F8","MINIMAL":"#FDF0EA","NONE":"#EDF3EF"}
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
            <th style='padding:6px 10px;font-size:0.6rem;font-weight:700;color:#7D4E8A;text-align:left;'>担当者 Owner</th>
            <th style='padding:6px 10px;font-size:0.6rem;font-weight:700;color:#7D4E8A;text-align:left;'>タスク Task</th>
            <th style='padding:6px 10px;font-size:0.6rem;font-weight:700;color:#7D4E8A;text-align:left;'>期限 Deadline</th>
          </tr>
        </thead>
        <tbody>{action_rows}</tbody>
      </table>
    </div>
    {tokki}
    <div style='margin-top:12px;font-size:0.68rem;color:#A87868;'>次回予定 · Next Meeting: {plan.jikai_yotei}</div>
  </div>
</div>"""
    except Exception:
        return ""


def build_results_html(R: dict, language: str, features: dict, pii_rep: dict | None) -> str:
    COLORS    = ["#E8829A","#F4A07A","#C9924A","#5A7D6B","#A8897C","#7A5C50"]
    SENT_ICON = {"positive":"🌸","neutral":"🌿","negative":"🍂"}

    ji       = R.get("japan_insights", {})
    speakers = sorted(R.get("speakers", []), key=lambda s: s.get("talk_time_pct", 0), reverse=True)
    soft     = R.get("soft_rejections", {}) or {}

    # ── Termination detection ─────────────────────────────────────────────────
    termination_detected = (
        soft.get("termination_detected", False) or
        R.get("meeting_type") == "contract_termination"
    )

    def _health():
        risk    = soft.get("risk_level", "NONE")
        risk_pts= {"NONE":25,"MINIMAL":20,"LOW":15,"MEDIUM":8,"HIGH":0,"CRITICAL":0}

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

        # Termination cap — never show "good" for a contract termination
        if termination_detected:
            score = min(score, 22)
            color = "#7C3AED"
            label = "Contract Terminated"
            bd = [("Sentiment",s_pts,30),("Clarity",a_pts,25),("Comm Risk",0,25),("AI Confidence",h_pts,20)]
        else:
            color = ("#2D9E6B" if score >= 80 else "#B87830" if score >= 60 else "#D96080" if score >= 40 else "#C84040")
            label = ("Productive Meeting" if score >= 80 else "Mostly Aligned" if score >= 60
                     else "Needs Follow-up" if score >= 40 else "High Risk")
            bd = [("Sentiment",s_pts,30),("Action Clarity",a_pts,25),("Comm Risk",r_pts,25),("AI Confidence",h_pts,20)]

        bars = "".join(
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

    # ── PII banner ────────────────────────────────────────────────────────────
    pii_html = ""
    if pii_rep and pii_rep.get("total_pii_found", 0) > 0:
        n = pii_rep["total_pii_found"]
        pii_html = (
            f"<div class='tai-pii-pill'>🔒 APPI — "
            f"{n} item{'s' if n!=1 else ''} anonymized before analysis</div>"
        )

    # ── Unlabeled transcript warning ──────────────────────────────────────────
    unlabeled_html = ""
    if R.get("_unlabeled_transcript"):
        unlabeled_html = (
            "<div style='background:#FFFBEB;border-left:4px solid #D97706;"
            "border-radius:0 10px 10px 0;padding:0.9rem 1.2rem;margin-bottom:1rem;"
            "font-size:0.82rem;color:#78350F;line-height:1.6;'>"
            "⚠ <strong>No speaker labels detected</strong> — each paragraph was assigned to a generic speaker. "
            "For best results, prefix each line with the speaker's name: "
            "<code style='background:#FEF3C7;border-radius:4px;padding:1px 5px;'>Name: their words here</code>"
            "</div>"
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

    gijiroku_preview = _build_gijiroku_preview(R, language) if features.get("show_japan_insights") else ""
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
    # Add note about sentiment scoring model when termination detected
    if termination_detected:
        sent_html += (
            "<div style='background:#F5F3FF;border-left:3px solid #7C3AED;"
            "border-radius:0 8px 8px 0;padding:0.7rem 1rem;margin-bottom:1rem;"
            "font-size:0.78rem;color:#4C1D95;line-height:1.6;'>"
            "Sentiment scored on <strong>communicative register</strong> — "
            "cooperative/deferential/gracious = neutral, not negative. "
            "Professional acceptance of a termination is not hostility."
            "</div>"
        )
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
        risk    = soft.get("risk_level","NONE") if soft else "NONE"
        risk_colors = {"CRITICAL":"#7C3AED","HIGH":"#963030","MEDIUM":"#986820","LOW":"#BE4060","MINIMAL":"#A87868","NONE":"#2D7A55"}
        rclr    = risk_colors.get(risk, "#2D7A55")
        cs_cnt  = ji.get("code_switch_count",0)

        # ── Termination detected banner ───────────────────────────────────────
        if termination_detected:
            term_sigs = soft.get("termination_signals", [])
            term_phrases = "".join(
                f"<div style='margin-bottom:8px;'>"
                f"<div style='font-size:0.82rem;font-weight:700;color:#5B21B6;"
                f"font-family:Noto Sans JP,sans-serif;'>⛔ {s['phrase']}</div>"
                f"<div style='font-size:0.72rem;color:#6B7280;margin-top:2px;'>"
                f"{s.get('english','')} · Speaker: {s.get('speaker','Unknown')}</div>"
                f"</div>"
                for s in term_sigs
            ) if term_sigs else (
                "<div style='font-size:0.82rem;color:#5B21B6;'>"
                "Contract termination language detected in transcript.</div>"
            )
            ins_html += (
                f"<div style='background:#F5F3FF;border:2px solid #7C3AED;"
                f"border-radius:12px;padding:1.2rem 1.4rem;margin-bottom:1.5rem;'>"
                f"<div style='font-size:0.7rem;font-weight:800;color:#7C3AED;"
                f"letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.8rem;'>"
                f"⛔ Explicit Contract Termination Detected</div>"
                f"{term_phrases}"
                f"<div style='font-size:0.78rem;color:#4C1D95;line-height:1.65;"
                f"border-top:1px solid #DDD6FE;padding-top:0.8rem;margin-top:0.6rem;'>"
                f"{soft.get('cultural_note', 'This is an explicit, irrevocable termination — not a soft refusal. The polite keigo delivery is cultural courtesy, not ambiguity.')}"
                f"</div>"
                f"</div>"
            )

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
            if not termination_detected:
                ins_html += f"<div style='font-size:0.73rem;color:#A87868;font-style:italic;margin-top:8px'>{soft.get('cultural_note','')}</div>"
    else:
        ins_html = "<div style='color:#A87868;font-size:0.85rem;padding:1rem 0;line-height:1.7'>Cultural intelligence features apply to Japanese and Hindi transcripts.</div>"

    insight_label = features.get('insight_tab_label', '🌐 Insights') or "インサイト"

    # 議事録 format banner
    gijiroku_format_banner = ""
    if features.get("show_japan_insights"):
        gijiroku_format_banner = (
            '<div style="margin-bottom:16px;border:1px solid #D0B0C8;border-radius:12px;overflow:hidden;">'
            +   '<div style="background:linear-gradient(135deg,#7D4E8A,#A06CB5);padding:10px 16px;display:flex;align-items:center;justify-content:space-between;">'
            +     '<div style="font-size:0.72rem;font-weight:700;color:#fff;letter-spacing:0.1em;text-transform:uppercase;">🗾 議事録 Format · Japanese Business Minutes</div>'
            +     '<div style="font-size:0.65rem;color:rgba(255,255,255,0.7);">Standard enterprise document structure</div>'
            +   '</div>'
            +   '<div style="padding:16px;background:#FDFAFF;display:grid;grid-template-columns:repeat(5, 1fr);gap:8px;text-align:center;align-items:center;">'
            +     ''.join(f"<div style='padding:6px 4px;'><div style='font-size:0.78rem;font-weight:700;color:#7D4E8A;font-family:Noto Sans JP,sans-serif;'>{ja}</div><div style='font-size:0.6rem;color:#A87868;margin-top:2px;'>{en}</div></div>" for ja, en in [("会議名","Meeting name"),("出席者","Attendees"),("議題","Agenda"),("決定事項","Decisions"),("アクション","Action items")])
            +   '</div>'
            + '</div>'
        )

    export_banner = """
    <div style='margin-top:20px; padding:18px; background:linear-gradient(135deg, rgba(125,78,138,0.04), rgba(160,108,181,0.06)); border:1px solid #D0B0C8; border-radius:12px; display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:10px;'>
        <div>
            <div style='font-size:0.9rem; font-weight:700; color:#7D4E8A; margin-bottom:4px;'>✨ Analysis Complete</div>
            <div style='font-size:0.75rem; color:#A87868;'>Your meeting intelligence is ready. Export full documents below.</div>
        </div>
    </div>
    """

    return (
        '<div class="tai-results">'
        + pii_html
        + unlabeled_html
        + '<div class="tai-tiles">' + tiles + '</div>'
        + gijiroku_format_banner
        + '<div class="tai-health">'
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
        +     '<label class="tai-tab-label" for="tai-radio-sum">📝 Summary</label>'
        +     '<label class="tai-tab-label" for="tai-radio-act">✅ Actions</label>'
        +     '<label class="tai-tab-label" for="tai-radio-sent">🌸 Sentiment</label>'
        +     '<label class="tai-tab-label" for="tai-radio-spk">🎤 Speakers</label>'
        +     '<label class="tai-tab-label" for="tai-radio-ins">' + insight_label + '</label>'
        +   '</div>'
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
    document.querySelectorAll('.tai-tab-label').forEach(function(lbl) {
      lbl.addEventListener('click', function(e) {
        e.preventDefault();
        e.stopPropagation();
        activateTab(lbl.getAttribute('for'));
      });
    });
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
    soft  = R.get("soft_rejections", {}) or {}
    termination_detected = (
        soft.get("termination_detected", False) or
        R.get("meeting_type") == "contract_termination"
    )

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

    risk  = soft.get("risk_level", "NONE")
    r_pts = {"NONE":25,"MINIMAL":20,"LOW":15,"MEDIUM":8,"HIGH":0,"CRITICAL":0}.get(risk, 25)
    h_pts = round((1 - R.get("verification",{}).get("overall_hallucination_risk", 0)) * 20)
    score = min(s_pts + a_pts + r_pts + h_pts, 100)

    if termination_detected:
        score = min(score, 22)
        return {"score": score, "label": "Contract Terminated",
                "color": "#7C3AED", "bg": "#F5F3FF", "border": "#C4B5FD"}

    if score >= 80:   label, color, bg, border = "Productive Meeting", "#486858", "#EDF3EF", "#A8C8B8"
    elif score >= 60: label, color, bg, border = "Mostly Aligned",    "#986820", "#FAF0E0", "#D9C090"
    elif score >= 40: label, color, bg, border = "Needs Follow-up",   "#C87030", "#FDF0EA", "#E8C090"
    else:             label, color, bg, border = "High Risk",         "#B04040", "#FAF0F0", "#E8A0A0"
    return {"score":score,"label":label,"color":color,"bg":bg,"border":border}