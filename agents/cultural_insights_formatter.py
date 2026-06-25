"""
agents/cultural_insights_formatter.py
Japanese Business Culture Insights — a dedicated export, separate from 議事録.

Deliberately scoped to the CULTURAL/BUSINESS layer only:
  - 根回し (nemawashi) / honne-tatemae indirect-communication signals
  - 稟議 (ringi-sho) decision approval status
  - Meeting dynamics: closing-summarizer pattern, senior-silence pivots,
    topic stalls / circle-back

Deliberately EXCLUDES raw NLP metrics (keigo_level, code_switch_count) —
those are linguistic measurements, not cultural/business insight, and
Kunal was explicit that this export should read as "the meeting and its
cultural nuances," not as an NLP report.

Rule-based. No LLM call. Mirrors gijiroku_formatter.py's structure
(dataclass plan + separate render functions) for consistency.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime


@dataclass
class ApprovalStep:
    approver: str
    role: str = ""
    status: str = "未承認"  # 未承認 (pending) / 承認済み (approved)


@dataclass
class CulturalInsightsPlan:
    meeting_title: str
    generated_at: str = ""
    language: str = "ja"

    # 根回し / nemawashi
    soft_rejection_risk: str = "NONE"
    soft_rejection_signals: List[dict] = field(default_factory=list)
    cultural_note: str = ""

    # 稟議 / ringi-sho
    key_decisions: List[str] = field(default_factory=list)
    approval_chain: List[ApprovalStep] = field(default_factory=list)
    kairan_jotai: str = "ドラフト — 関係者の確認待ち / Draft — pending circulation for review"

    # Meeting dynamics
    closing_summarizer: Optional[dict] = None
    senior_silence_pivots: List[dict] = field(default_factory=list)
    topic_stalls: List[dict] = field(default_factory=list)

    def __post_init__(self):
        if not self.generated_at:
            self.generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")


class CulturalInsightsFormatter:
    """Converts a TranscriptAI analysis_result → CulturalInsightsPlan."""

    def format(self, analysis: dict) -> CulturalInsightsPlan:
        soft = analysis.get("soft_rejections", {}) or {}
        dynamics = analysis.get("conversation_dynamics", {}) or {}
        role_hints = analysis.get("role_hints") or dynamics.get("role_hints") or {}

        ranked = sorted(
            ((name, h.get("role", ""), h.get("rank", 0)) for name, h in role_hints.items()),
            key=lambda x: -x[2],
        )
        approval_chain = [ApprovalStep(approver=name, role=role) for name, role, rank in ranked if rank > 0]
        if not approval_chain:
            approval_chain = [ApprovalStep(approver="未指定 / Unassigned", role="")]

        return CulturalInsightsPlan(
            meeting_title=analysis.get("meeting_title") or "会議",
            language=analysis.get("language", "ja"),
            soft_rejection_risk=soft.get("risk_level", "NONE"),
            soft_rejection_signals=soft.get("detected", []),
            cultural_note=soft.get("cultural_note", ""),
            key_decisions=analysis.get("key_decisions", []) or [],
            approval_chain=approval_chain,
            closing_summarizer=dynamics.get("closing_summarizer") or {"detected": False},
            senior_silence_pivots=dynamics.get("senior_silence_pivots", []),
            topic_stalls=dynamics.get("topic_stalls", []),
        )


# ─────────────────────────────────────────────────────────────
# Renderers
# ─────────────────────────────────────────────────────────────

_DIVIDER_MD  = "\n---\n"
_DIVIDER_TXT = "\n" + "─" * 50 + "\n"


def render_markdown(plan: CulturalInsightsPlan) -> str:
    lines = [
        "# 日本ビジネス文化インサイト",
        "## Japanese Business Culture Insights",
        "",
        f"**{plan.meeting_title}**  ·  {plan.generated_at}",
    ]

    # ── 根回し ──
    lines += [_DIVIDER_MD, "## 根回し — Indirect Communication Signals", ""]
    lines.append(f"**Risk level: {plan.soft_rejection_risk}**")
    if plan.soft_rejection_signals:
        lines.append("")
        for s in plan.soft_rejection_signals:
            lines.append(
                f"- **{s.get('speaker','Unknown')}**: \"{s.get('phrase','')}\" "
                f"({s.get('reading','')}) — {s.get('explanation','')}"
            )
        if plan.cultural_note:
            lines += ["", f"*{plan.cultural_note}*"]
    else:
        lines.append("\nNo indirect refusal or hedging signals were detected — "
                      "communication in this meeting read as direct.")

    # ── 稟議 ──
    lines += [_DIVIDER_MD, "## 稟議 — Decision & Approval Status", ""]
    if plan.key_decisions:
        lines.append("**決定事項 / Decisions:**")
        for d in plan.key_decisions:
            lines.append(f"- {d}")
    else:
        lines.append("No formal decisions were recorded — discussion remained exploratory.")
    lines += ["", f"*{plan.kairan_jotai}*", ""]
    lines += ["| 承認者 | 役職 | 状態 |", "|--------|------|------|"]
    for step in plan.approval_chain:
        lines.append(f"| {step.approver} | {step.role or '—'} | {step.status} |")

    # ── Meeting dynamics ──
    lines += [_DIVIDER_MD, "## 会議の力学 — Meeting Dynamics", ""]
    any_dynamics = False
    if plan.closing_summarizer and plan.closing_summarizer.get("detected"):
        any_dynamics = True
        lines.append(f"**Closing summarizer pattern:** {plan.closing_summarizer.get('explanation','')}")
    for pivot in plan.senior_silence_pivots:
        any_dynamics = True
        lines.append(f"- {pivot.get('explanation','')}")
    for stall in plan.topic_stalls:
        any_dynamics = True
        lines.append(f"- {stall.get('explanation','')}")
    if not any_dynamics:
        lines.append("No notable structural patterns (senior silence, topic stalls, "
                      "closing-summarizer) were detected in this meeting.")

    lines += [_DIVIDER_MD, "*Generated by TranscriptAI — github.com/aiKunalBisht/Transcript-ai*"]
    return "\n".join(lines)


def render_text(plan: CulturalInsightsPlan) -> str:
    lines = [
        "日本ビジネス文化インサイト",
        "Japanese Business Culture Insights",
        "=" * 50,
        f"{plan.meeting_title}  ·  {plan.generated_at}",
    ]

    lines += [_DIVIDER_TXT, "【根回し / Indirect Communication Signals】"]
    lines.append(f"  Risk level: {plan.soft_rejection_risk}")
    if plan.soft_rejection_signals:
        lines.append("")
        for s in plan.soft_rejection_signals:
            lines.append(f"  ・{s.get('speaker','Unknown')}: \"{s.get('phrase','')}\" "
                          f"({s.get('reading','')}) — {s.get('explanation','')}")
        if plan.cultural_note:
            lines += ["", f"  {plan.cultural_note}"]
    else:
        lines.append("  No indirect refusal or hedging signals were detected.")

    lines += [_DIVIDER_TXT, "【稟議 / Decision & Approval Status】"]
    if plan.key_decisions:
        for d in plan.key_decisions:
            lines.append(f"  ・{d}")
    else:
        lines.append("  No formal decisions were recorded.")
    lines += ["", f"  {plan.kairan_jotai}", ""]
    lines.append(f"  {'承認者':<15} {'役職':<12} 状態")
    lines.append(f"  {'─'*14} {'─'*11} {'─'*10}")
    for step in plan.approval_chain:
        lines.append(f"  {step.approver:<15} {(step.role or '—'):<12} {step.status}")

    lines += [_DIVIDER_TXT, "【会議の力学 / Meeting Dynamics】"]
    any_dynamics = False
    if plan.closing_summarizer and plan.closing_summarizer.get("detected"):
        any_dynamics = True
        lines.append(f"  {plan.closing_summarizer.get('explanation','')}")
    for pivot in plan.senior_silence_pivots:
        any_dynamics = True
        lines.append(f"  ・{pivot.get('explanation','')}")
    for stall in plan.topic_stalls:
        any_dynamics = True
        lines.append(f"  ・{stall.get('explanation','')}")
    if not any_dynamics:
        lines.append("  No notable structural patterns were detected in this meeting.")

    lines += [_DIVIDER_TXT, "Generated by TranscriptAI · github.com/aiKunalBisht/Transcript-ai"]
    return "\n".join(lines)


def format_cultural_insights(analysis: dict, as_markdown: bool = False) -> str:
    """Convenience wrapper, same pattern as gijiroku_formatter.format_gijiroku()."""
    plan = CulturalInsightsFormatter().format(analysis)
    return render_markdown(plan) if as_markdown else render_text(plan)