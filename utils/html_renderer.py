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
        f"<svg class='tai-viz-donut' width='{size}' height='{size}' viewBox='0 0 {size} {size}'>"
        f"<circle cx='{size//2}' cy='{size//2}' r='{r}' fill='none' stroke='#D9E0E7' stroke-width='6'/>"
        f"<circle cx='{size//2}' cy='{size//2}' r='{r}' fill='none' stroke='{color}' stroke-width='6' stroke-linecap='round' "
        f"stroke-dasharray='{dash:.1f} {circ:.1f}' transform='rotate(-90 {size//2} {size//2})'/>"
        f"<text x='50%' y='54%' text-anchor='middle' font-size='13' font-weight='800' fill='{color}' font-family='Arial'>{pct}%</text></svg>"
    )


def _avatar(name: str, color: str) -> str:
    initials = "".join(p[0].upper() for p in name.split()[:2]) or name[:2].upper()
    return (
        f"<div class='tai-viz-avatar' style='--avatar-color:{color}'>"
        f"{initials}</div>"
    )


def _health_ring(score: int, color: str) -> str:
    r, size = 54, 120
    circ = 2 * 3.14159 * r
    dash = circ * score / 100
    label = ("Excellent" if score >= 80 else "Good" if score >= 60 else "Fair" if score >= 40 else "At Risk")
    if score <= 22:
        label = "Terminated"
    return (
        f"<div class='tai-viz-health-ring'>"
        f"<svg width='{size}' height='{size}' viewBox='0 0 {size} {size}'>"
        f"<circle cx='60' cy='60' r='{r}' fill='none' stroke='#D9E0E7' stroke-width='10'/>"
        f"<circle cx='60' cy='60' r='{r}' fill='none' stroke='{color}' stroke-width='10' stroke-linecap='round' "
        f"stroke-dasharray='{dash:.1f} {circ:.1f}' transform='rotate(-90 60 60)'/>"
        f"<text x='50%' y='46%' text-anchor='middle' font-size='22' font-weight='800' fill='#17212B' font-family='Arial'>{score}</text>"
        f"<text x='50%' y='62%' text-anchor='middle' font-size='10' fill='#7A8694' font-family='Arial'>/ 100</text></svg>"
        f"<div class='tai-viz-health-label' style='color:{color}'>{label}</div></div>"
    )


def _analytics_css() -> str:
    return r'''<style>
    .tai-analytics-shell{--ax-bg:#F3F6F8;--ax-surface:#FFFFFF;--ax-ink:#17212B;--ax-muted:#667482;--ax-faint:#93A0AD;--ax-line:#DDE4EA;--ax-navy:#14222D;--ax-blue:#557A96;--ax-sakura:#B55478;--ax-sakura-bg:#FAEEF3;--ax-jp:#6E4C7A;--ax-jp-bg:#F6F1F8;--ax-green:#357A62;--ax-green-bg:#EDF6F2;--ax-amber:#9B6A27;--ax-amber-bg:#FBF5E9;--ax-red:#A64B4B;--ax-red-bg:#FBEEEE;--ax-purple:#6C4CA1;--ax-purple-bg:#F2EEFA;font-family:Inter,'Noto Sans JP',sans-serif;color:var(--ax-ink)}
    .tai-analytics-head{display:flex;justify-content:space-between;gap:16px;align-items:flex-start;margin-bottom:16px}
    .tai-analytics-kicker{font-size:.65rem;font-weight:800;letter-spacing:.14em;text-transform:uppercase;color:var(--ax-muted);margin-bottom:5px}
    .tai-analytics-title{font-size:1.35rem;font-weight:800;letter-spacing:-.02em;color:var(--ax-ink);margin:0}
    .tai-analytics-sub{font-size:.76rem;color:var(--ax-muted);margin-top:4px}
    .tai-analytics-status{display:inline-flex;align-items:center;gap:7px;background:var(--ax-surface);border:1px solid var(--ax-line);padding:7px 10px;border-radius:8px;font-size:.68rem;font-weight:700;color:var(--ax-green);white-space:nowrap}
    .tai-analytics-dot{width:7px;height:7px;border-radius:50%;background:currentColor}
    .tai-viz-outcome{display:flex;align-items:center;gap:14px;background:var(--outcome-bg,#F5F7F9);border:1px solid var(--outcome-border,#DDE4EA);padding:12px 14px;border-radius:10px;margin-bottom:14px}
    .tai-viz-outcome-icon{font-size:1.35rem;line-height:1}
    .tai-viz-label{font-size:.6rem;font-weight:800;letter-spacing:.12em;text-transform:uppercase;color:var(--outcome-color,var(--ax-blue));margin-bottom:2px}
    .tai-viz-outcome-title{font-size:.96rem;font-weight:800;color:var(--ax-ink)}
    .tai-viz-outcome-meaning{font-size:.72rem;color:var(--ax-muted);margin-top:2px;line-height:1.4}
    .tai-analytics-meta{display:flex;gap:7px;flex-wrap:wrap;margin-bottom:12px}
    .tai-viz-pill{display:inline-flex;align-items:center;gap:5px;border:1px solid var(--ax-line);background:var(--ax-surface);border-radius:7px;padding:5px 8px;font-size:.64rem;font-weight:700;color:var(--ax-muted)}
    .tai-viz-pill strong{color:var(--ax-ink)}
    .tai-viz-pill.jp{background:var(--ax-jp-bg);border-color:#D8C8DF;color:var(--ax-jp)}
    .tai-viz-pill.green{background:var(--ax-green-bg);border-color:#C8E0D6;color:var(--ax-green)}
    .tai-viz-pill.warn{background:var(--ax-amber-bg);border-color:#E8D5A6;color:var(--ax-amber)}
    .tai-viz-warning{display:flex;gap:9px;align-items:flex-start;background:var(--ax-amber-bg);border:1px solid #E8D5A6;border-left:3px solid var(--ax-amber);padding:10px 12px;border-radius:8px;margin-bottom:12px;color:#78571F;font-size:.72rem;line-height:1.55}
    .tai-viz-warning code{background:#F6EBCB;border-radius:4px;padding:1px 4px}
    .tai-viz-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-bottom:12px}
    .tai-viz-kpi{background:var(--ax-surface);border:1px solid var(--ax-line);border-radius:10px;padding:12px 13px;min-height:82px}
    .tai-viz-kpi-top{display:flex;justify-content:space-between;gap:8px;align-items:center}
    .tai-viz-kpi-label{font-size:.6rem;font-weight:800;text-transform:uppercase;letter-spacing:.08em;color:var(--ax-faint)}
    .tai-viz-kpi-icon{font-size:.85rem;opacity:.75}
    .tai-viz-kpi-value{font-size:1.3rem;font-weight:800;letter-spacing:-.03em;margin-top:8px;color:var(--ax-ink)}
    .tai-viz-kpi-foot{font-size:.62rem;color:var(--ax-muted);margin-top:3px}
    .tai-viz-main-grid{display:grid;grid-template-columns:minmax(0,1.65fr) minmax(270px,.9fr);gap:12px;margin-bottom:12px}
    .tai-viz-card{background:var(--ax-surface);border:1px solid var(--ax-line);border-radius:10px;padding:14px}
    .tai-viz-card-head{display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:12px}
    .tai-viz-card-title{font-size:.72rem;font-weight:800;letter-spacing:.06em;text-transform:uppercase;color:var(--ax-ink)}
    .tai-viz-card-caption{font-size:.62rem;color:var(--ax-faint)}
    .tai-viz-health-layout{display:grid;grid-template-columns:140px 1fr;gap:14px;align-items:center}
    .tai-viz-health-ring{text-align:center}
    .tai-viz-health-label{font-size:.68rem;font-weight:800;letter-spacing:.08em;text-transform:uppercase;margin-top:2px}
    .tai-viz-health-bars{display:grid;gap:9px}
    .tai-viz-health-bar-row{display:grid;grid-template-columns:110px 1fr 48px;gap:8px;align-items:center}
    .tai-viz-health-bar-label{font-size:.65rem;color:var(--ax-muted)}
    .tai-viz-health-track{height:6px;background:#E8EDF1;border-radius:99px;overflow:hidden}
    .tai-viz-health-fill{height:100%;border-radius:99px;background:var(--health-color,#557A96)}
    .tai-viz-health-score{font-size:.63rem;font-weight:800;color:var(--ax-muted);text-align:right}
    .tai-viz-jp-card{background:linear-gradient(180deg,#FBF8FC 0%,#F8F4FA 100%);border-color:#DCCFE1}
    .tai-viz-jp-badge{display:inline-flex;align-items:center;gap:5px;background:#EEE7F2;color:var(--ax-jp);border:1px solid #D5C5DD;border-radius:7px;padding:4px 7px;font-size:.59rem;font-weight:800;letter-spacing:.08em;text-transform:uppercase}
    .tai-viz-jp-stat-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:11px}
    .tai-viz-jp-stat{background:#fff;border:1px solid #E4DCE8;border-radius:8px;padding:9px}
    .tai-viz-jp-stat-label{font-size:.57rem;color:#8E7A97;text-transform:uppercase;letter-spacing:.06em;font-weight:800}
    .tai-viz-jp-stat-value{font-size:.95rem;font-weight:800;color:var(--ax-jp);margin-top:5px}
    .tai-viz-jp-stat-foot{font-size:.57rem;color:#9A8AA0;margin-top:2px}
    .tai-radio-tabs{display:contents}
    .tai-tab-bar{background:var(--ax-surface);border:1px solid var(--ax-line);border-radius:9px 9px 0 0;padding:3px;gap:2px}
    .tai-tab-label{padding:9px 12px!important;font-size:.68rem!important;border:0!important;border-radius:6px!important;margin:0!important}
    .tai-tab-label:hover{background:#F5F7F9!important}
    .tai-panel{border:1px solid var(--ax-line)!important;border-top:0!important;border-radius:0 0 10px 10px!important;padding:14px!important;box-shadow:none!important}
    .tai-viz-section{font-size:.62rem;font-weight:800;letter-spacing:.1em;text-transform:uppercase;color:var(--ax-muted);padding-bottom:7px;border-bottom:1px solid var(--ax-line);margin:3px 0 11px}
    .tai-viz-summary{background:#F7F9FA;border:1px solid var(--ax-line);border-radius:9px;padding:13px;margin-bottom:12px}
    .tai-viz-summary-title{font-size:.6rem;font-weight:800;text-transform:uppercase;letter-spacing:.1em;color:var(--ax-muted);margin-bottom:8px}
    .tai-viz-bilingual{display:grid;grid-template-columns:1fr 1fr;gap:10px}
    .tai-viz-lang-block{background:#fff;border:1px solid var(--ax-line);border-radius:8px;padding:10px}
    .tai-viz-lang-tag{font-size:.56rem;font-weight:800;letter-spacing:.08em;text-transform:uppercase;color:var(--ax-faint);margin-bottom:6px}
    .tai-viz-lang-tag.ja{color:var(--ax-jp)}
    .tai-viz-lang-tag.en{color:var(--ax-green)}
    .tai-viz-lang-text{font-size:.78rem;line-height:1.7;color:var(--ax-ink)}
    .tai-viz-keypoint{display:grid;grid-template-columns:30px 1fr;gap:9px;background:#fff;border:1px solid var(--ax-line);border-radius:8px;padding:10px;margin-bottom:7px}
    .tai-viz-keypoint-num{width:24px;height:24px;border-radius:6px;background:#EEF2F5;color:var(--ax-blue);display:flex;align-items:center;justify-content:center;font-size:.62rem;font-weight:800}
    .tai-viz-keypoint-text{font-size:.77rem;line-height:1.6;color:var(--ax-ink)}
    .tai-viz-action{display:grid;grid-template-columns:26px 1fr auto;gap:10px;align-items:start;border:1px solid var(--ax-line);border-left:3px solid var(--ax-blue);background:#fff;border-radius:0 8px 8px 0;padding:10px 11px;margin-bottom:7px}
    .tai-viz-action.flagged{border-left-color:var(--ax-red);background:#FEFAFA}
    .tai-viz-action-icon{font-size:.75rem;padding-top:1px;color:var(--ax-blue)}
    .tai-viz-action.flagged .tai-viz-action-icon{color:var(--ax-red)}
    .tai-viz-action-task{font-size:.78rem;font-weight:700;color:var(--ax-ink);line-height:1.45}
    .tai-viz-action-meta{font-size:.61rem;color:var(--ax-muted);margin-top:4px}
    .tai-viz-action-status{font-size:.58rem;font-weight:800;color:var(--ax-muted);background:#F2F5F7;border:1px solid var(--ax-line);padding:4px 6px;border-radius:6px;white-space:nowrap}
    .tai-viz-sent{display:grid;grid-template-columns:36px 1fr auto;gap:10px;align-items:center;border-bottom:1px solid var(--ax-line);padding:9px 0}
    .tai-viz-sent:last-child{border-bottom:0}
    .tai-viz-sent-icon{font-size:1rem;text-align:center}
    .tai-viz-sent-name{font-size:.75rem;font-weight:700;color:var(--ax-ink)}
    .tai-viz-sent-label{font-size:.61rem;color:var(--ax-muted);margin-top:2px}
    .tai-viz-badge{font-size:.57rem;font-weight:800;letter-spacing:.06em;padding:4px 7px;border-radius:6px}
    .tai-viz-badge.positive{background:var(--ax-green-bg);color:var(--ax-green)}
    .tai-viz-badge.neutral{background:#F2F5F7;color:#687583}
    .tai-viz-badge.negative{background:var(--ax-red-bg);color:var(--ax-red)}
    .tai-viz-speaker{display:grid;grid-template-columns:38px 1fr 54px;gap:10px;align-items:center;padding:9px 0;border-bottom:1px solid var(--ax-line)}
    .tai-viz-speaker:last-child{border-bottom:0}
    .tai-viz-avatar{width:34px;height:34px;border-radius:8px;display:flex;align-items:center;justify-content:center;background:color-mix(in srgb,var(--avatar-color) 12%,#fff);border:1px solid color-mix(in srgb,var(--avatar-color) 40%,#fff);color:var(--avatar-color);font-size:.66rem;font-weight:800}
    .tai-viz-speaker-name{font-size:.74rem;font-weight:700;color:var(--ax-ink)}
    .tai-viz-speaker-tone{font-size:.59rem;color:var(--ax-muted);margin-top:2px}
    .tai-viz-speaker-bar{height:6px;background:#E9EEF2;border-radius:99px;overflow:hidden;margin-top:6px}
    .tai-viz-speaker-fill{height:100%;border-radius:99px;background:var(--speaker-color)}
    .tai-viz-speaker-pct{font-size:.63rem;font-weight:800;color:var(--speaker-color);text-align:right}
    .tai-viz-insight-chips{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin-bottom:12px}
    .tai-viz-insight-chip{border:1px solid var(--ax-line);background:#fff;border-radius:8px;padding:10px}
    .tai-viz-insight-chip.jp{background:var(--ax-jp-bg);border-color:#DDD1E2}
    .tai-viz-insight-chip-label{font-size:.56rem;font-weight:800;letter-spacing:.08em;text-transform:uppercase;color:var(--ax-faint)}
    .tai-viz-insight-chip-value{font-size:.9rem;font-weight:800;color:var(--ax-ink);margin-top:5px}
    .tai-viz-insight-chip-foot{font-size:.56rem;color:var(--ax-muted);margin-top:2px}
    .tai-viz-signal{border:1px solid var(--ax-line);border-left:3px solid var(--ax-amber);background:#fff;border-radius:0 8px 8px 0;padding:10px 11px;margin-bottom:7px}
    .tai-viz-signal.high{border-left-color:var(--ax-red);background:#FFFBFB}
    .tai-viz-signal-phrase{font-size:.77rem;font-weight:800;color:var(--ax-ink)}
    .tai-viz-signal-meta{font-size:.6rem;color:var(--ax-muted);margin-top:3px}
    .tai-viz-signal-exp{font-size:.68rem;color:#55616D;line-height:1.55;margin-top:5px}
    .tai-viz-jp-format{border:1px solid #DCCFE1;background:#FBF9FC;border-radius:9px;overflow:hidden;margin-bottom:12px}
    .tai-viz-jp-format-head{background:var(--ax-navy);color:#fff;padding:9px 11px;display:flex;justify-content:space-between;gap:10px;align-items:center}
    .tai-viz-jp-format-title{font-size:.63rem;font-weight:800;letter-spacing:.1em;text-transform:uppercase}
    .tai-viz-jp-format-sub{font-size:.57rem;opacity:.7}
    .tai-viz-jp-format-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:0}
    .tai-viz-jp-format-cell{padding:9px 5px;text-align:center;border-right:1px solid #E5DDEA}
    .tai-viz-jp-format-cell:last-child{border-right:0}
    .tai-viz-jp-format-ja{font-family:'Noto Sans JP',sans-serif;font-size:.72rem;font-weight:700;color:var(--ax-jp)}
    .tai-viz-jp-format-en{font-size:.53rem;color:#917F99;margin-top:2px}
    .tai-viz-banner{display:flex;align-items:center;gap:10px;border:1px solid var(--ax-line);background:#F7F9FA;border-radius:8px;padding:10px 12px;margin-top:12px}
    .tai-viz-banner-title{font-size:.68rem;font-weight:800;color:var(--ax-ink)}
    .tai-viz-banner-sub{font-size:.61rem;color:var(--ax-muted);margin-top:2px}
    .tai-viz-gijiroku{margin-top:14px}
    .tai-viz-gijiroku-head{background:var(--ax-navy);color:#fff;padding:12px 14px;display:flex;justify-content:space-between;gap:10px}
    .tai-viz-gijiroku-label{font-size:.58rem;letter-spacing:.11em;text-transform:uppercase;opacity:.72}
    .tai-viz-gijiroku-title{font-family:'Noto Sans JP',sans-serif;font-size:.88rem;font-weight:700;margin-top:3px}
    .tai-viz-gijiroku-body{padding:13px;background:#FBF9FC}
    .tai-viz-gijiroku-section{font-size:.58rem;font-weight:800;letter-spacing:.09em;text-transform:uppercase;color:var(--ax-jp);margin:9px 0 6px}
    .tai-viz-attendee{display:inline-flex;background:#F0EAF3;border:1px solid #DDD0E3;border-radius:6px;padding:3px 7px;font-size:.57rem;color:var(--ax-jp);margin:2px 4px 2px 0}
    .tai-viz-table{width:100%;border-collapse:collapse;background:#fff;border:1px solid var(--ax-line);border-radius:7px;overflow:hidden}
    .tai-viz-table th{padding:7px 8px;background:#F2EEF4;color:var(--ax-jp);font-size:.56rem;text-align:left}
    .tai-viz-table td{padding:7px 8px;font-size:.62rem;color:var(--ax-ink);border-top:1px solid var(--ax-line)}
    .tai-viz-eval-summary{display:grid;grid-template-columns:auto 1fr;gap:12px;align-items:center;background:#F7F9FA;border:1px solid var(--ax-line);border-radius:9px;padding:14px;margin-bottom:12px}
    .tai-viz-eval-score{font-size:2rem;font-weight:800;letter-spacing:-.04em}
    .tai-viz-eval-label{font-size:.62rem;font-weight:800;letter-spacing:.1em;text-transform:uppercase}
    .tai-viz-eval-sub{font-size:.67rem;color:var(--ax-muted);margin-top:4px;line-height:1.45}
    .tai-viz-eval-card{border:1px solid var(--ax-line);border-radius:9px;padding:12px 13px;margin-bottom:8px;background:#fff}
    .tai-viz-eval-head{display:flex;justify-content:space-between;gap:12px;align-items:flex-start;margin-bottom:9px}
    .tai-viz-eval-name{font-size:.76rem;font-weight:800;color:var(--ax-ink)}
    .tai-viz-eval-meta{font-size:.58rem;color:var(--ax-muted);margin-top:3px}
    .tai-viz-eval-score-wrap{display:flex;gap:7px;align-items:center}
    .tai-viz-eval-score-mini{font-size:1rem;font-weight:800}
    .tai-viz-grade{font-size:.58rem;font-weight:800;color:#fff;border-radius:5px;padding:3px 7px}
    .tai-viz-metrics{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:6px}
    .tai-viz-metric{background:#F7F9FA;border:1px solid var(--ax-line);border-radius:6px;padding:7px 5px;text-align:center}
    .tai-viz-metric-label{font-size:.52rem;color:var(--ax-faint);text-transform:uppercase;letter-spacing:.06em}
    .tai-viz-metric-value{font-size:.72rem;font-weight:800;color:var(--ax-ink);margin-top:3px}
    @media(max-width:900px){.tai-viz-grid{grid-template-columns:repeat(2,1fr)}.tai-viz-main-grid{grid-template-columns:1fr}.tai-viz-metrics{grid-template-columns:repeat(3,1fr)}}
    @media(max-width:640px){.tai-analytics-head{flex-direction:column}.tai-viz-grid{grid-template-columns:repeat(2,1fr)}.tai-viz-health-layout,.tai-viz-bilingual{grid-template-columns:1fr}.tai-viz-insight-chips,.tai-viz-jp-stat-grid{grid-template-columns:1fr 1fr}.tai-viz-jp-format-grid{grid-template-columns:repeat(3,1fr)}.tai-viz-jp-format-cell:nth-child(3){border-right:0}.tai-viz-action{grid-template-columns:22px 1fr}.tai-viz-action-status{grid-column:2;justify-self:start}.tai-viz-metrics{grid-template-columns:repeat(2,1fr)}}
</style>'''

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
            f"<span class='tai-viz-attendee'>{_clean_val(s)}</span>"
            for s in plan.shussekisha[:6]
        )
        agenda_items = "".join(
            f"<div class='tai-viz-keypoint'><span class='tai-viz-keypoint-num'>{i:02d}</span><span class='tai-viz-keypoint-text'>{_clean_val(item)}</span></div>"
            for i, item in enumerate(plan.gidai[:3], 1)
        )
        action_rows = "".join(
            f"<tr><td>{a.owner}</td><td>{a.task}{'' if not a.flag else ' ⚠'}</td><td>{a.deadline}</td></tr>"
            for a in plan.action_items[:4]
        )

        soft = R.get("soft_rejections", {}) or {}
        risk = soft.get("risk_level", "NONE")
        risk_colors = {"CRITICAL":"#6C4CA1","HIGH":"#A64B4B","MEDIUM":"#9B6A27","LOW":"#B55478","MINIMAL":"#8D7265","NONE":"#357A62"}
        risk_clr = risk_colors.get(risk, "#357A62")

        tokki = ""
        if plan.tokki_jiko:
            tokki = f"<div class='tai-viz-warning' style='margin-top:10px;border-left-color:#A64B4B;color:#7E3D3D'>⚠ {plan.tokki_jiko}</div>"

        return f"""
<div class='tai-viz-gijiroku'>
  <div class='tai-viz-gijiroku-head'>
    <div><div class='tai-viz-gijiroku-label'>議事録 · Japanese Formal Business Minutes</div><div class='tai-viz-gijiroku-title'>{plan.kaigi_mei}</div></div>
    <div style='text-align:right;font-size:.58rem;opacity:.72'>{plan.nichiji}<br>{plan.basho}</div>
  </div>
  <div class='tai-viz-gijiroku-body'>
    <div class='tai-viz-gijiroku-section'>出席者 · Attendees</div>
    <div>{attendee_chips}</div>
    <div style='display:grid;grid-template-columns:1fr auto;gap:12px;align-items:start;'>
      <div><div class='tai-viz-gijiroku-section'>議題 · Agenda</div>{agenda_items}</div>
      <div><div class='tai-viz-gijiroku-section'>リスク · Risk</div><div class='tai-viz-pill' style='border-color:{risk_clr}55;color:{risk_clr};justify-content:center'>{risk} · {soft.get('total_signals',0)} signals</div></div>
    </div>
    <div class='tai-viz-gijiroku-section'>アクションアイテム · Action Items</div>
    <table class='tai-viz-table'><thead><tr><th>担当者 Owner</th><th>タスク Task</th><th>期限 Deadline</th></tr></thead><tbody>{action_rows}</tbody></table>
    {tokki}
    <div style='font-size:.58rem;color:#8B7C92;margin-top:9px'>次回予定 · Next Meeting: {plan.jikai_yotei}</div>
  </div>
</div>"""
    except Exception:
        return ""

def build_results_html(R: dict, language: str, features: dict, pii_rep: dict | None) -> str:
    COLORS    = ["#557A96", "#B55478", "#6E4C7A", "#357A62", "#9B6A27", "#7C8A96"]
    SENT_ICON = {"positive":"↑", "neutral":"•", "negative":"↓"}

    ji       = R.get("japan_insights", {})
    speakers = sorted(R.get("speakers", []), key=lambda s: s.get("talk_time_pct", 0), reverse=True)
    soft     = R.get("soft_rejections", {}) or {}

    termination_detected = (
        soft.get("termination_detected", False) or
        R.get("meeting_type") == "contract_termination"
    )
    approval_gate_detected = soft.get("approval_gate_detected", False)

    def _health():
        risk    = soft.get("risk_level", "NONE")
        risk_pts= {"NONE":25,"MINIMAL":20,"LOW":15,"MEDIUM":8,"HIGH":0,"CRITICAL":0}

        sents   = R.get("sentiment", [])
        w       = {"positive":1.0, "neutral":0.6, "negative":0.1}
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

        approval_gate_detected_health = soft.get('approval_gate_detected', False)

        if termination_detected:
            score = min(score, 22)
            color = "#6C4CA1"
            label = "Contract Terminated"
            bd = [("Sentiment",s_pts,30),("Clarity",a_pts,25),("Comm Risk",0,25),("AI Confidence",h_pts,20)]
        elif approval_gate_detected_health:
            score = min(score, 55)
            color = "#9B6A27"
            label = "Approval Pending"
            bd = [("Sentiment",s_pts,30),("Action Clarity",a_pts,25),("Comm Risk",r_pts,25),("AI Confidence",h_pts,20)]
        else:
            color = ("#357A62" if score >= 80 else "#9B6A27" if score >= 60 else "#B55478" if score >= 40 else "#A64B4B")
            label = ("Productive Meeting" if score >= 80 else "Mostly Aligned" if score >= 60 else "Needs Follow-up" if score >= 40 else "High Risk")
            bd = [("Sentiment",s_pts,30),("Action Clarity",a_pts,25),("Comm Risk",r_pts,25),("AI Confidence",h_pts,20)]

        bars = "".join(
            f"<div class='tai-viz-health-bar-row'><span class='tai-viz-health-bar-label'>{lb}</span><div class='tai-viz-health-track'><div class='tai-viz-health-fill' style='--health-color:{color};width:{round(pt/tot*100)}%'></div></div><span class='tai-viz-health-score'>{pt}/{tot}</span></div>"
            for lb, pt, tot in bd
        )
        return score, color, bars

    score, hc, hbars = _health()

    spk_count = len(R.get("speakers", []))
    act_count = len(R.get("action_items", []))
    cs_val    = ji.get("code_switch_count","—") if features.get("show_code_switch") else "—"
    keigo_val = ji.get("keigo_level","—").title() if features.get("show_japan_insights") else language_display_name(language).split(" ",1)[-1]
    keigo_lbl = "Formality" if features.get("show_japan_insights") else "Language"

    def _tile(val, lbl, icon, foot=""):
        return (
            f"<div class='tai-viz-kpi'><div class='tai-viz-kpi-top'><span class='tai-viz-kpi-label'>{lbl}</span><span class='tai-viz-kpi-icon'>{icon}</span></div>"
            f"<div class='tai-viz-kpi-value'>{val}</div><div class='tai-viz-kpi-foot'>{foot}</div></div>"
        )

    tiles = (
        _tile(spk_count, "Speakers", "◌", "detected participants") +
        _tile(act_count, "Action Items", "✓", "tasks extracted") +
        _tile(cs_val, "Code Switches", "⇄", "language transitions") +
        _tile(keigo_val, keigo_lbl, "JP", "Japanese register" if features.get("show_japan_insights") else "detected language")
    )

    pii_html = ""
    if pii_rep and pii_rep.get("total_pii_found", 0) > 0:
        n = pii_rep["total_pii_found"]
        pii_html = f"<div class='tai-viz-pill green' style='margin-bottom:10px'>✓ APPI · {n} item{'s' if n!=1 else ''} anonymized before analysis</div>"

    unlabeled_html = ""
    if R.get("_unlabeled_transcript"):
        unlabeled_html = (
            "<div class='tai-viz-warning'>⚠ <span><strong>No speaker labels detected</strong> — each paragraph was assigned to a generic speaker. "
            "For best results, prefix each line with the speaker's name: <code>Name: their words here</code></span></div>"
        )

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
                "<div class='tai-viz-summary'><div class='tai-viz-summary-title'>Meeting Overview · 会議概要</div>"
                "<div class='tai-viz-bilingual'>"
                f"<div class='tai-viz-lang-block'><div class='tai-viz-lang-tag ja'>日本語 · JA</div><div class='tai-viz-lang-text' style='font-family:\"Noto Sans JP\",sans-serif'>{full_sum}</div></div>"
                f"<div class='tai-viz-lang-block'><div class='tai-viz-lang-tag en'>English · EN</div><div class='tai-viz-lang-text'>{en_summary}</div></div>"
                "</div></div>"
            )
        elif is_japanese and not en_summary:
            sum_html += f"<div class='tai-viz-summary'><div class='tai-viz-summary-title'>Meeting Overview · 会議概要</div><div class='tai-viz-lang-block'><div class='tai-viz-lang-tag ja'>日本語 · JA</div><div class='tai-viz-lang-text' style='font-family:\"Noto Sans JP\",sans-serif'>{full_sum}</div></div></div>"
        else:
            sum_html += f"<div class='tai-viz-summary'><div class='tai-viz-summary-title'>Meeting Overview</div><div class='tai-viz-lang-text'>{full_sum}</div></div>"

    if bullets:
        sum_html += f"<div class='tai-viz-section'>{len(bullets)} Key Points</div>"
        for i, b in enumerate(bullets, 1):
            has_cjk = any('\u4e00' <= c <= '\u9fff' or '\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff' for c in str(b))
            font = "font-family:'Noto Sans JP',sans-serif;" if has_cjk else ""
            sum_html += f"<div class='tai-viz-keypoint'><span class='tai-viz-keypoint-num'>{i:02d}</span><span class='tai-viz-keypoint-text' style='{font}'>{b}</span></div>"
    elif not full_sum:
        sum_html += "<div style='color:#7A8694;font-size:.78rem;padding:10px 0'>No summary extracted. Try a longer transcript.</div>"

    gijiroku_preview = _build_gijiroku_preview(R, language) if features.get("show_japan_insights") else ""
    if gijiroku_preview:
        sum_html += gijiroku_preview

    items    = R.get("action_items", [])
    v_count  = sum(1 for i in items if not i.get("hallucination_flag"))
    f_count  = len(items) - v_count
    act_html = f"<div class='tai-viz-section'>{len(items)} Items · <span style='color:#357A62'>✓ {v_count} verified</span>" + (f" · <span style='color:#A64B4B'>⚑ {f_count} flagged</span>" if f_count else "") + "</div>"
    act_html += "".join(
        (
            "<div class='tai-viz-action" + (" flagged" if i.get("hallucination_flag") else "") + "'>"
            "<div class='tai-viz-action-icon'>" + ("⚑" if i.get("hallucination_flag") else "◆") + "</div>"
            "<div><div class='tai-viz-action-task'>" + str(i.get("task","")) + "</div>"
            "<div class='tai-viz-action-meta'>Owner: <strong>" + str(i.get("owner","TBD")) + "</strong> · Deadline: <strong>" + str(i.get("deadline","TBD")) + "</strong>" + (f" · {i.get('confidence',0):.0%} confidence" if i.get("confidence") else "") + (f"<div style='color:#A64B4B;margin-top:3px'>⚠ {i.get('flag_reason','')}</div>" if i.get("flag_reason") else "") + "</div></div>"
            "<div class='tai-viz-action-status'>" + ("FLAGGED" if i.get("hallucination_flag") else "EXTRACTED") + "</div></div>"
        ) for i in items
    ) if items else "<div style='color:#7A8694;font-size:.78rem;padding:10px 0'>No action items extracted.</div>"

    sent_html = "<div class='tai-viz-section'>Speaker Sentiment · Communicative Register</div>"
    if termination_detected:
        sent_html += "<div class='tai-viz-banner' style='background:#F5F1FB;border-color:#D9CEE9'><div>ⓘ</div><div><div class='tai-viz-banner-title'>Register-aware sentiment</div><div class='tai-viz-banner-sub'>Cooperative, deferential and gracious language is treated as neutral rather than negative.</div></div></div>"
    sent_html += "".join(
        f"<div class='tai-viz-sent'><span class='tai-viz-sent-icon'>{SENT_ICON.get(s.get('score','neutral').lower(),'•')}</span><div><div class='tai-viz-sent-name'>{s.get('speaker','')}</div><div class='tai-viz-sent-label'>{s.get('label','')}</div></div><span class='tai-viz-badge {s.get('score','neutral').lower()}'>{s.get('score','neutral').upper()}</span></div>"
        for s in R.get("sentiment", [])
    )

    spk_html = "<div class='tai-viz-section'>Talk Time Distribution</div>"
    for idx2, spk in enumerate(speakers):
        nm  = spk.get("name", f"Speaker {idx2+1}")
        pct = spk.get("talk_time_pct", 0)
        tone= spk.get("tone","—")
        col = COLORS[idx2 % len(COLORS)]
        spk_html += (
            f"<div class='tai-viz-speaker'><div>{_avatar(nm,col)}</div><div><div class='tai-viz-speaker-name'>{nm}</div><div class='tai-viz-speaker-tone'>{tone}</div><div class='tai-viz-speaker-bar'><div class='tai-viz-speaker-fill' style='--speaker-color:{col};width:{pct}%'></div></div></div><div style='text-align:right'>{_svg_donut(pct,col,48)}</div></div>"
        )

    ins_html = ""
    if features.get("show_japan_insights"):
        keigo   = ji.get("keigo_level","—")
        k_src   = ji.get("keigo_source","llm")
        kc      = {"high":"#B55478","medium":"#9B6A27","low":"#8D7265"}.get(keigo,"#6E4C7A")
        sigs    = ji.get("nemawashi_signals",[])
        risk    = soft.get("risk_level","NONE") if soft else "NONE"
        risk_colors = {"CRITICAL":"#6C4CA1","HIGH":"#A64B4B","MEDIUM":"#9B6A27","LOW":"#B55478","MINIMAL":"#8D7265","NONE":"#357A62"}
        rclr    = risk_colors.get(risk, "#357A62")
        cs_cnt  = ji.get("code_switch_count",0)

        if approval_gate_detected and not termination_detected:
            ag_sigs = soft.get("approval_gate_signals", [])
            has_tech_commercial = any("technical" in s.get("phrase","").lower() or "技術" in s.get("phrase","") for s in ag_sigs)
            has_personal_vs_org = any("personally" in s.get("phrase","").lower() or "board" in s.get("phrase","").lower() or "headquarters" in s.get("phrase","").lower() for s in ag_sigs)
            has_committee = any("committee" in s.get("phrase","").lower() or "委員会" in s.get("phrase","") or "稟議" in s.get("phrase","") for s in ag_sigs)
            status_chips = ""
            if has_tech_commercial:
                status_chips += "<span class='tai-viz-pill green'>✓ Technical Review Approved</span><span class='tai-viz-pill warn'>⌛ Commercial Approval Pending</span>"
            if has_personal_vs_org:
                status_chips += "<span class='tai-viz-pill'>👤 Personal Support Only</span><span class='tai-viz-pill warn'>⌛ Organizational Decision Pending</span>"
            if has_committee:
                status_chips += "<span class='tai-viz-pill jp'>🏛 Committee Review Required</span>"
            hierarchy_html = ""
            if has_tech_commercial:
                hierarchy_html = "<div class='tai-viz-banner' style='background:#FBF5E9;border-color:#E8D5A6'><div>↳</div><div><div class='tai-viz-banner-title'>Decision Authority Hierarchy</div><div class='tai-viz-banner-sub'><strong>Engineering</strong> → Technical recommendation · <strong>Procurement / 調達部</strong> → Commercial review · <strong>購買委員会</strong> → Final decision authority</div></div></div>"
            elif has_personal_vs_org:
                hierarchy_html = "<div class='tai-viz-banner' style='background:#FBF5E9;border-color:#E8D5A6'><div>↳</div><div><div class='tai-viz-banner-title'>Authority Clarification</div><div class='tai-viz-banner-sub'><strong>Meeting participant</strong> → Personal support · <strong>Board / HQ / Executive Committee</strong> → Actual decision authority</div></div></div>"
            ins_html += f"<div class='tai-viz-warning' style='border-left-color:#9B6A27;background:#FBF5E9;color:#775522'><strong>Approval Gate Detected</strong><span><div style='margin-top:5px'>{status_chips}</div>{hierarchy_html}<div style='margin-top:8px'>{soft.get('cultural_note','In Japanese organizations, technical and commercial approval are separate processes.')}</div></span></div>"

        if termination_detected:
            term_sigs = soft.get("termination_signals", [])
            term_phrases = "".join(f"<div class='tai-viz-signal' style='border-left-color:#6C4CA1;background:#FAF8FD'><div class='tai-viz-signal-phrase' style='color:#5B4386'>⛔ {s['phrase']}</div><div class='tai-viz-signal-meta'>{s.get('english','')} · Speaker: {s.get('speaker','Unknown')}</div></div>" for s in term_sigs) if term_sigs else "<div class='tai-viz-signal' style='border-left-color:#6C4CA1;background:#FAF8FD'><div class='tai-viz-signal-phrase'>Contract termination language detected in transcript.</div></div>"
            ins_html += f"<div class='tai-viz-warning' style='border-left-color:#6C4CA1;background:#F4F0FB;color:#57407A'><strong>Explicit Contract Termination Detected</strong><span>{term_phrases}<div style='margin-top:8px'>{soft.get('cultural_note', 'This is an explicit, irrevocable termination — not a soft refusal. The polite keigo delivery is cultural courtesy, not ambiguity.')}</div></span></div>"

        ins_html += (
            "<div class='tai-viz-insight-chips'>"
            f"<div class='tai-viz-insight-chip jp'><div class='tai-viz-insight-chip-label'>Keigo Register</div><div class='tai-viz-insight-chip-value' style='color:{kc}'>{keigo.upper()}</div><div class='tai-viz-insight-chip-foot'>via {k_src}</div></div>"
            f"<div class='tai-viz-insight-chip'><div class='tai-viz-insight-chip-label'>Rejection Risk</div><div class='tai-viz-insight-chip-value' style='color:{rclr}'>{risk}</div><div class='tai-viz-insight-chip-foot'>{soft.get('total_signals',0)} signals</div></div>"
            f"<div class='tai-viz-insight-chip'><div class='tai-viz-insight-chip-label'>Code Switches</div><div class='tai-viz-insight-chip-value'>{cs_cnt}</div><div class='tai-viz-insight-chip-foot'>language switches</div></div>"
            "</div>"
        )

        if sigs:
            ins_html += f"<div class='tai-viz-section'>Indirect Consensus Signals · {len(sigs)} detected</div>"
            ins_html += "".join(f"<div class='tai-viz-pill jp' style='margin:0 5px 5px 0'>◆ {s}</div>" for s in sigs)

        if soft and soft.get("total_signals",0) > 0:
            ins_html += "<div class='tai-viz-section' style='margin-top:13px'>Soft Rejection Analysis</div>"
            for sig in soft.get("high_signals",[]):
                ins_html += f"<div class='tai-viz-signal high'><div class='tai-viz-signal-phrase'>🚨 {sig['phrase']}</div><div class='tai-viz-signal-meta'>{sig['reading']} · {sig['speaker']} · {sig['confidence']:.0%}</div><div class='tai-viz-signal-exp'>{sig['explanation']}</div></div>"
            for sig in soft.get("medium_signals",[]):
                ins_html += f"<div class='tai-viz-signal'><div class='tai-viz-signal-phrase'>⚠ {sig['phrase']}</div><div class='tai-viz-signal-meta'>{sig['reading']} · {sig['speaker']} · {sig['confidence']:.0%}</div><div class='tai-viz-signal-exp'>{sig['explanation']}</div></div>"
            if not termination_detected:
                ins_html += f"<div style='font-size:.65rem;color:#7A8694;font-style:italic;margin-top:7px'>{soft.get('cultural_note','')}</div>"
    else:
        ins_html = "<div style='color:#7A8694;font-size:.78rem;padding:10px 0;line-height:1.7'>Cultural intelligence features apply to Japanese and Hindi transcripts.</div>"

    insight_label = features.get('insight_tab_label', '🌐 Insights') or "Insights"

    gijiroku_format_banner = ""
    if features.get("show_japan_insights"):
        cells = ''.join(f"<div class='tai-viz-jp-format-cell'><div class='tai-viz-jp-format-ja'>{ja}</div><div class='tai-viz-jp-format-en'>{en}</div></div>" for ja,en in [("会議名","Meeting name"),("出席者","Attendees"),("議題","Agenda"),("決定事項","Decisions"),("アクション","Action items")])
        gijiroku_format_banner = f"<div class='tai-viz-jp-format'><div class='tai-viz-jp-format-head'><div class='tai-viz-jp-format-title'>🗾 議事録 Format · Japanese Business Minutes</div><div class='tai-viz-jp-format-sub'>Enterprise document structure</div></div><div class='tai-viz-jp-format-grid'>{cells}</div></div>"

    export_banner = "<div class='tai-viz-banner'><div style='font-size:1rem'>✓</div><div><div class='tai-viz-banner-title'>Analysis complete</div><div class='tai-viz-banner-sub'>Meeting intelligence is ready for export.</div></div></div>"

    deal_outcome = R.get("deal_outcome", {}) or {}
    outcome_banner = ""
    try:
        from analysis.deal_outcome_detector import compute_meeting_outcome
        outcome = compute_meeting_outcome(soft, deal_outcome)
        outcome_banner = (
            '<div class="tai-viz-outcome" style="--outcome-color:' + outcome["color"] + ';--outcome-bg:' + outcome["color"] + '0D;--outcome-border:' + outcome["color"] + '33;">'
            + '<div class="tai-viz-outcome-icon">' + outcome["emoji"] + '</div>'
            + '<div><div class="tai-viz-label">Meeting Outcome</div><div class="tai-viz-outcome-title">' + outcome["label"] + '</div><div class="tai-viz-outcome-meaning">' + outcome["meaning"] + '</div></div></div>'
        )
    except Exception:
        pass

    return (
        '<div class="tai-results tai-analytics-shell">'
        + _analytics_css()
        + '<div class="tai-analytics-head"><div><div class="tai-analytics-kicker">Meeting Intelligence · Analysis Complete</div><h2 class="tai-analytics-title">Meeting Intelligence Overview</h2><div class="tai-analytics-sub">Structured signals across conversation, actions, language, risk and Japanese business communication.</div></div><div class="tai-analytics-status"><span class="tai-analytics-dot"></span> Live Analysis</div></div>'
        + outcome_banner
        + pii_html
        + unlabeled_html
        + '<div class="tai-viz-grid">' + tiles + '</div>'
        + '<div class="tai-viz-main-grid">'
        + '<div class="tai-viz-card"><div class="tai-viz-card-head"><div class="tai-viz-card-title">Meeting Health</div><div class="tai-viz-card-caption">Composite signal</div></div><div class="tai-viz-health-layout"><div>' + _health_ring(score, hc) + '</div><div class="tai-viz-health-bars">' + hbars + '</div></div></div>'
        + (f'<div class="tai-viz-card tai-viz-jp-card"><div class="tai-viz-card-head"><div class="tai-viz-card-title">Japanese Intelligence</div><span class="tai-viz-jp-badge">日本語 · JA</span></div><div style="font-size:.7rem;color:#77667D;line-height:1.55">Business-context signals are surfaced alongside standard meeting metrics.</div><div class="tai-viz-jp-stat-grid"><div class="tai-viz-jp-stat"><div class="tai-viz-jp-stat-label">Keigo</div><div class="tai-viz-jp-stat-value">{keigo_val}</div><div class="tai-viz-jp-stat-foot">formality</div></div><div class="tai-viz-jp-stat"><div class="tai-viz-jp-stat-label">Risk</div><div class="tai-viz-jp-stat-value" style="color:{hc}">{soft.get("risk_level","NONE")}</div><div class="tai-viz-jp-stat-foot">signals</div></div><div class="tai-viz-jp-stat"><div class="tai-viz-jp-stat-label">Switches</div><div class="tai-viz-jp-stat-value">{cs_val}</div><div class="tai-viz-jp-stat-foot">code switches</div></div></div></div>' if features.get("show_japan_insights") else '')
        + '</div>'
        + gijiroku_format_banner
        + '<div class="tai-radio-tabs">'
        + '<input type="radio" name="tai-tabs" id="tai-radio-sum" checked><input type="radio" name="tai-tabs" id="tai-radio-act"><input type="radio" name="tai-tabs" id="tai-radio-sent"><input type="radio" name="tai-tabs" id="tai-radio-spk"><input type="radio" name="tai-tabs" id="tai-radio-ins">'
        + '<div class="tai-tab-bar"><label class="tai-tab-label" for="tai-radio-sum">Overview</label><label class="tai-tab-label" for="tai-radio-act">Actions</label><label class="tai-tab-label" for="tai-radio-sent">Sentiment</label><label class="tai-tab-label" for="tai-radio-spk">Speakers</label><label class="tai-tab-label" for="tai-radio-ins">' + insight_label + '</label></div>'
        + '<div class="tai-panel"><div id="tai-sum" class="tai-tab-content">' + sum_html + '</div><div id="tai-act" class="tai-tab-content">' + act_html + '</div><div id="tai-sent" class="tai-tab-content">' + sent_html + '</div><div id="tai-spk" class="tai-tab-content">' + spk_html + '</div><div id="tai-ins" class="tai-tab-content">' + ins_html + '</div>' + export_banner + '</div></div>'
        + '''<script>
(function(){
  var TAB_KEY='tai-active-tab';var ids=['tai-radio-sum','tai-radio-act','tai-radio-sent','tai-radio-spk','tai-radio-ins'];var panels=['tai-sum','tai-act','tai-sent','tai-spk','tai-ins'];
  function activateTab(radioId){ids.forEach(function(id,idx){var radio=document.getElementById(id);var panel=document.getElementById(panels[idx]);if(radio&&panel){if(id===radioId){radio.checked=true;panel.style.display='block';}else{radio.checked=false;panel.style.display='none';}}});document.querySelectorAll('.tai-tab-label').forEach(function(lbl){var forId=lbl.getAttribute('for');if(forId===radioId){lbl.style.color='#17212B';lbl.style.fontWeight='800';lbl.style.background='#EEF2F5';}else{lbl.style.color='';lbl.style.fontWeight='';lbl.style.background='';}});try{sessionStorage.setItem(TAB_KEY,radioId);}catch(e){}}
  function init(){var saved=null;try{saved=sessionStorage.getItem(TAB_KEY);}catch(e){}activateTab(saved&&ids.indexOf(saved)!==-1?saved:'tai-radio-sum');document.querySelectorAll('.tai-tab-label').forEach(function(lbl){lbl.addEventListener('click',function(e){e.preventDefault();e.stopPropagation();activateTab(lbl.getAttribute('for'));});});ids.forEach(function(id){var radio=document.getElementById(id);if(radio)radio.addEventListener('change',function(){if(radio.checked)activateTab(id);});});}
  if(document.getElementById('tai-radio-sum'))init();else setTimeout(init,100);
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

# ════════════════════════════════════════════════════════════════════════════════
# EVALUATION PAGE — renders results from utils/evaluator.py's evaluate() against
# ════════════════════════════════════════════════════════════════════════════════

def _eval_grade_color(grade: str) -> str:
    return {
        "A": "#2D9E6B", "B": "#86A340", "C": "#B87830", "D": "#D96080", "F": "#C84040",
    }.get((grade or "").upper(), "#7A5040")



def _eval_grade_color(grade: str) -> str:
    return {"A": "#357A62", "B": "#6E8C43", "C": "#9B6A27", "D": "#B55478", "F": "#A64B4B"}.get((grade or "").upper(), "#667482")


def build_evaluation_html(reports: list, mlflow_logged: bool) -> str:
    """
    reports: list of {"tc_id", "tc_name", "provider", "duration_ms", "report": <evaluate() output>}
    mlflow_logged: whether MLFLOW_AVAILABLE was True for this run (evaluator.py
                   logs to http://127.0.0.1:5000 automatically when so).
    """
    if not reports:
        return _analytics_css() + "<div class='tai-analytics-shell'><div style='text-align:center;padding:2.5rem 1rem;color:#667482;font-size:.82rem'>No evaluation results — the run may have failed before producing any report.</div></div>"

    n = len(reports)
    avg_overall = round(sum(r["report"]["overall_score"] for r in reports) / n, 1)
    avg_color = ("#357A62" if avg_overall >= 80 else "#9B6A27" if avg_overall >= 60 else "#B55478" if avg_overall >= 40 else "#A64B4B")

    summary_html = (
        _analytics_css() + "<div class='tai-analytics-shell'>"
        "<div class='tai-analytics-head'><div><div class='tai-analytics-kicker'>MLOps · Evaluation Analytics</div><h2 class='tai-analytics-title'>Evaluation Overview</h2><div class='tai-analytics-sub'>Ground-truth quality signals across summary, actions, sentiment and Japanese communication.</div></div>"
        "<div class='tai-analytics-status' style='color:" + avg_color + "'><span class='tai-analytics-dot'></span> Suite Complete</div></div>"
        "<div class='tai-viz-eval-summary'>"
        "<div class='tai-viz-eval-score' style='color:" + avg_color + "'>" + f"{avg_overall}%" + "</div>"
        "<div><div class='tai-viz-eval-label' style='color:" + avg_color + "'>Average Overall Score · " + str(n) + " test case" + ("s" if n != 1 else "") + "</div><div class='tai-viz-eval-sub'>" + (
            "✓ Logged to MLflow — <a href='http://127.0.0.1:5000' target='_blank' style='color:" + avg_color + ";font-weight:800'>view run history →</a>" if mlflow_logged else
            "⚠ MLflow not detected on this run — results shown here only, not persisted."
        ) + "</div></div></div>"
    )

    cards_html = ""
    for r in reports:
        rep = r["report"]
        grade = rep.get("overall_grade", "—")
        gcolor = _eval_grade_color(grade)
        score = rep.get("overall_score", 0)

        metrics = [
            ("Semantic", f"{rep['summary'].get('semantic_score', 0):.0%}" if isinstance(rep['summary'].get('semantic_score'), float) else rep['summary'].get('semantic_score', '—')),
            ("Actions F1", f"{rep['action_items'].get('f1', 0):.0%}" if isinstance(rep['action_items'].get('f1'), float) else rep['action_items'].get('f1', '—')),
            ("Sentiment", f"{rep['sentiment'].get('soft_accuracy', 0):.0%}" if isinstance(rep['sentiment'].get('soft_accuracy'), float) else rep['sentiment'].get('soft_accuracy', '—')),
        ]
        if "japan_insights" in rep:
            ji = rep["japan_insights"]
            metrics.append(("Keigo", ji.get("keigo", {}).get("grade", "—")))
            nm = ji.get("nemawashi", {})
            metrics.append(("Nemawashi P/R", f"{nm.get('precision', 0):.0%}/{nm.get('recall', 0):.0%}" if isinstance(nm.get('precision'), float) else "—"))

        metric_chips = "".join(f"<div class='tai-viz-metric'><div class='tai-viz-metric-label'>{lbl}</div><div class='tai-viz-metric-value'>{val}</div></div>" for lbl,val in metrics)
        hallu = ""
        if "hallucination_bonus" in rep:
            hallu = f"<div class='tai-viz-eval-meta' style='margin-top:7px'>Hallucination risk: <strong>{rep.get('hallucination_risk','UNKNOWN')}</strong> · bonus +{rep.get('hallucination_bonus',0):.0%}</div>"
        cards_html += (
            "<div class='tai-viz-eval-card'><div class='tai-viz-eval-head'><div><div class='tai-viz-eval-name'>" + str(r.get("tc_name", r.get("tc_id", "Test case"))) + "</div>"
            "<div class='tai-viz-eval-meta'>" + str(r.get("tc_id", "")) + " · provider: " + str(r.get("provider", "unknown")) + " · " + f"{r.get('duration_ms',0):.0f}ms" + "</div></div>"
            "<div class='tai-viz-eval-score-wrap'><div class='tai-viz-eval-score-mini' style='color:" + gcolor + "'>" + str(score) + "%</div><div class='tai-viz-grade' style='background:" + gcolor + "'>" + str(grade) + "</div></div></div>"
            "<div class='tai-viz-metrics'>" + metric_chips + "</div>" + hallu + "</div>"
        )

    return summary_html + cards_html + "</div>"
