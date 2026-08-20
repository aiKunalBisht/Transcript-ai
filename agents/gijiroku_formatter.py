"""
agents/gijiroku_formatter.py
議事録 (Gijiroku) — Japanese Formal Business Meeting Minutes Formatter

Rule-based. No LLM call. Restructures analysis_result into the
standard Japanese business meeting minutes format.

Fix 4 (discussion content): Added 討議内容 (togi_naiyou) section with
fallback chain: result.get("discussion") → full_summary → summary bullets
→ placeholder. Previously the section showed a header but no content
because analysis.get("discussion") returns None (the LLM schema doesn't
produce this key), and there was no fallback.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime


def _get(d: dict, key: str, default):
    """dict.get() that also coalesces an explicit None to `default`."""
    v = d.get(key, default)
    return v if v is not None else default


@dataclass
class ActionItem:
    task: str
    owner: str
    deadline: str
    flag: bool = False


@dataclass
class ApprovalStep:
    approver: str
    role: str = ""
    status: str = "未承認"


@dataclass
class GijirokulPlan:
    """Structured 議事録 ready for any renderer."""
    kaigi_mei: str
    nichiji: str
    basho: str
    shussekisha: List[str]
    gidai: List[str]
    togi_naiyou: str            # 討議内容 — Fix 4: discussion content with fallback chain
    kettei_jiko: List[str]
    action_items: List[ActionItem]
    jikai_yotei: str
    kirokusha: str
    tokki_jiko: Optional[str]
    language: str = "ja"
    generated_at: str = ""
    approval_chain: List[ApprovalStep] = field(default_factory=list)
    kairan_jotai: str = "ドラフト — 関係者の確認待ち / Draft — pending circulation for review"

    def __post_init__(self):
        if not self.generated_at:
            self.generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")


class GijirokulFormatter:
    """Converts TranscriptAI analysis_result → GijirokulPlan. Zero API calls."""

    def format(
        self,
        analysis: dict,
        recorder: str = "TranscriptAI",
        basho: str = "オンライン会議 / Online",
        jikai_yotei: str = "未定 / TBD",
        timestamp: Optional[str] = None,
    ) -> GijirokulPlan:

        # ── 会議名 ──────────────────────────────────────────────
        kaigi_mei = (
            analysis.get("meeting_title")
            or analysis.get("title")
            or "会議議事録"
        )

        # ── 日時 ────────────────────────────────────────────────
        if timestamp:
            nichiji = timestamp
        else:
            now = datetime.now()
            nichiji = (
                f"{now.year}年{now.month:02d}月{now.day:02d}日 "
                f"{now.hour:02d}:{now.minute:02d}"
            )

        # ── 出席者 ──────────────────────────────────────────────
        raw_speakers = _get(analysis, "speakers", [])
        shussekisha = []
        for s in raw_speakers:
            if isinstance(s, dict):
                name = s.get("name") or "不明"
                role = s.get("role") or ""
                talk = s.get("talk_time_pct")
                entry = name
                if role:
                    entry += f"（{role}）"
                if talk:
                    entry += f" — 発言比率 {talk}%"
                shussekisha.append(entry)
            elif isinstance(s, str) and s:
                shussekisha.append(s)

        if not shussekisha:
            shussekisha = ["出席者情報なし / Not identified"]

        # ── 議題 ────────────────────────────────────────────────
        gidai = []
        bullets = _get(analysis, "summary", [])
        if isinstance(bullets, list):
            gidai = [b for b in bullets if isinstance(b, str) and len(b.strip()) > 4]
        elif isinstance(bullets, str) and bullets.strip():
            import re
            sentences = re.split(r'[。.]\s*', bullets.strip())
            gidai = [s.strip() for s in sentences if len(s.strip()) > 4]

        if not gidai:
            full = analysis.get("full_summary") or ""
            if full:
                gidai = [full[:120] + ("..." if len(full) > 120 else "")]
            else:
                gidai = ["議題情報なし / Agenda not extracted"]

        # ── 討議内容 ─────────────────────────────────────────────
        # Fix 4: fallback chain — discussion field (LLM extended output) →
        # full_summary → summary bullets joined → empty placeholder.
        # analysis.get("discussion") returns None when the LLM didn't produce
        # a dedicated discussion key (the common case); without this chain the
        # section rendered a header with no content underneath.
        discussion_content = (
            analysis.get("discussion")
            or analysis.get("full_summary")
            or "\n".join(f"・{b}" for b in _get(analysis, "summary", []))
            or "（討議内容なし）"
        )

        # ── 決定事項 ────────────────────────────────────────────
        raw_decisions = _get(analysis, "key_decisions", [])
        kettei_jiko: List[str] = []
        if isinstance(raw_decisions, list):
            kettei_jiko = [str(d) for d in raw_decisions if d]
        elif isinstance(raw_decisions, str):
            kettei_jiko = [raw_decisions] if raw_decisions else []

        if not kettei_jiko:
            kettei_jiko = ["明示的な決定事項なし / No explicit decisions recorded"]

        # ── アクションアイテム ──────────────────────────────────
        raw_actions = _get(analysis, "action_items", [])
        action_items = []
        for item in raw_actions:
            if isinstance(item, dict):
                task     = item.get("task") or ""
                owner    = item.get("owner") or "TBD"
                deadline = item.get("deadline") or "未定"
                flag     = bool(item.get("hallucination_flag", False))
                if task:
                    action_items.append(ActionItem(
                        task=task, owner=owner,
                        deadline=deadline, flag=flag
                    ))
            elif isinstance(item, str) and item:
                action_items.append(ActionItem(
                    task=item, owner="TBD", deadline="未定"
                ))

        if not action_items:
            action_items = [ActionItem(
                task="アクションアイテムなし / No action items recorded",
                owner="—", deadline="—"
            )]

        # ── 特記事項 ────────────────────────────────────────────
        tokki_jiko_parts = []
        soft = analysis.get("soft_rejections", {}) or {}
        risk = soft.get("risk_level") or "NONE"
        signals = soft.get("total_signals") or 0
        note = soft.get("cultural_note") or ""

        if risk in ("MEDIUM", "HIGH") and signals > 0:
            part = (
                f"【コミュニケーションリスク】ソフトリジェクションリスク: {risk} "
                f"（{signals}件検出）"
            )
            if note:
                part += f"\n　文化的注記: {note}"
            tokki_jiko_parts.append(part)

        dynamics = analysis.get("conversation_dynamics", {}) or {}
        closing = dynamics.get("closing_summarizer", {}) or {}
        if closing.get("detected"):
            tokki_jiko_parts.append(
                f"【発言順序の注記】{closing.get('explanation') or ''}"
            )
        for stall in (dynamics.get("topic_stalls") or []):
            tokki_jiko_parts.append(f"【議題の保留と再提起】{stall.get('explanation') or ''}")
        for pivot in (dynamics.get("senior_silence_pivots") or []):
            tokki_jiko_parts.append(f"【発言パターンの注記】{pivot.get('explanation') or ''}")

        tokki_jiko = "\n\n".join(tokki_jiko_parts) if tokki_jiko_parts else None

        # ── 承認状況 ────────────────────────────────────────────
        role_hints = analysis.get("role_hints") or dynamics.get("role_hints") or {}
        ranked = sorted(
            (
                (name, (h.get("role") or "") if isinstance(h, dict) else "",
                 (h.get("rank") or 0) if isinstance(h, dict) else 0)
                for name, h in role_hints.items()
            ),
            key=lambda x: -x[2],
        )
        approval_chain = [
            ApprovalStep(approver=name, role=role)
            for name, role, rank in ranked
            if rank > 0
        ]
        if not approval_chain:
            approval_chain = [ApprovalStep(approver="未指定 / Unassigned", role="")]

        return GijirokulPlan(
            kaigi_mei=kaigi_mei,
            nichiji=nichiji,
            basho=basho,
            shussekisha=shussekisha,
            gidai=gidai,
            togi_naiyou=discussion_content,
            kettei_jiko=kettei_jiko,
            action_items=action_items,
            jikai_yotei=jikai_yotei,
            kirokusha=recorder,
            tokki_jiko=tokki_jiko,
            language=analysis.get("language") or "ja",
            approval_chain=approval_chain,
        )


# ─────────────────────────────────────────────────────────────
# Renderers
# ─────────────────────────────────────────────────────────────

_DIVIDER_MD  = "\n---\n"
_DIVIDER_TXT = "\n" + "─" * 50 + "\n"


def render_markdown(plan: GijirokulPlan) -> str:
    """Renders GijirokulPlan as clean Markdown 議事録."""
    lines = []

    lines += [
        "# 議事録",
        "",
        "| 項目 | 内容 |",
        "|------|------|",
        f"| **会議名** | {plan.kaigi_mei} |",
        f"| **日時** | {plan.nichiji} |",
        f"| **場所** | {plan.basho} |",
        f"| **記録者** | {plan.kirokusha} |",
        f"| **作成日時** | {plan.generated_at} |",
        "",
    ]

    lines += [_DIVIDER_MD, "## 出席者", ""]
    for s in plan.shussekisha:
        lines.append(f"- {s}")
    lines.append("")

    lines += [_DIVIDER_MD, "## 議題", ""]
    for i, item in enumerate(plan.gidai, 1):
        lines.append(f"{i}. {item}")
    lines.append("")

    # Fix 4: 討議内容 section — populated via fallback chain in format()
    lines += [_DIVIDER_MD, "## 討議内容", ""]
    for line in plan.togi_naiyou.split("\n"):
        if line.strip():
            lines.append(line)
    lines.append("")

    lines += [_DIVIDER_MD, "## 決定事項", ""]
    for i, d in enumerate(plan.kettei_jiko, 1):
        lines.append(f"{i}. {d}")
    lines.append("")

    lines += [_DIVIDER_MD, "## アクションアイテム", ""]
    lines += [
        "| # | 担当者 | 内容 | 期限 |",
        "|---|--------|------|------|",
    ]
    for i, a in enumerate(plan.action_items, 1):
        flag = " ⚠️" if a.flag else ""
        lines.append(f"| {i} | {a.owner} | {a.task}{flag} | {a.deadline} |")
    lines.append("")

    lines += [_DIVIDER_MD, "## 次回予定", "", plan.jikai_yotei, ""]

    if plan.tokki_jiko:
        lines += [_DIVIDER_MD, "## 特記事項", "", plan.tokki_jiko, ""]

    lines += [_DIVIDER_MD, "## 承認状況", "", f"*{plan.kairan_jotai}*", ""]
    lines += [
        "| 承認者 | 役職 | 状態 |",
        "|--------|------|------|",
    ]
    for step in plan.approval_chain:
        lines.append(f"| {step.approver} | {step.role or '—'} | {step.status} |")
    lines.append("")
    lines.append(
        "*この議事録は確定版ではありません。関係者の確認・承認を経て最終版となります。*\n"
        "*This record is not final — minutes circulate for review and approval before "
        "being treated as confirmed.*"
    )

    lines += [
        _DIVIDER_MD,
        "*Generated by TranscriptAI — github.com/aiKunalBisht/Transcript-ai*",
        f"*{plan.generated_at}*",
    ]

    return "\n".join(lines)


def render_text(plan: GijirokulPlan) -> str:
    """Renders GijirokulPlan as plain-text 議事録 (email/Slack safe)."""
    lines = []

    def _s(v):
        return v if v is not None else ""

    lines += [
        "議　事　録",
        "=" * 50,
        f"会議名　: {plan.kaigi_mei}",
        f"日　時　: {plan.nichiji}",
        f"場　所　: {plan.basho}",
        f"記録者　: {plan.kirokusha}",
        f"作成日時: {plan.generated_at}",
        _DIVIDER_TXT,
        "【出席者】",
    ]
    for s in plan.shussekisha:
        lines.append(f"  ・{s}")

    lines += [_DIVIDER_TXT, "【議題】"]
    for i, item in enumerate(plan.gidai, 1):
        lines.append(f"  {i}. {item}")

    # Fix 4: 討議内容 section — populated via fallback chain in format()
    lines += [_DIVIDER_TXT, "【討議内容】"]
    for line in plan.togi_naiyou.split("\n"):
        if line.strip():
            lines.append(f"  {line}")

    lines += [_DIVIDER_TXT, "【決定事項】"]
    for i, d in enumerate(plan.kettei_jiko, 1):
        lines.append(f"  {i}. {d}")

    lines += [_DIVIDER_TXT, "【アクションアイテム】"]
    lines.append(f"  {'担当者':<15} {'期限':<12} 内容")
    lines.append(f"  {'─'*14} {'─'*11} {'─'*30}")
    for a in plan.action_items:
        flag = " [要確認]" if a.flag else ""
        lines.append(f"  {_s(a.owner):<15} {_s(a.deadline):<12} {_s(a.task)}{flag}")

    lines += [_DIVIDER_TXT, "【次回予定】", f"  {plan.jikai_yotei}"]

    if plan.tokki_jiko:
        lines += [_DIVIDER_TXT, "【特記事項】", f"  {plan.tokki_jiko}"]

    lines += [_DIVIDER_TXT, "【承認状況】", f"  {plan.kairan_jotai}", ""]
    lines.append(f"  {'承認者':<15} {'役職':<12} 状態")
    lines.append(f"  {'─'*14} {'─'*11} {'─'*10}")
    for step in plan.approval_chain:
        lines.append(f"  {_s(step.approver):<15} {_s(step.role or '—'):<12} {step.status}")
    lines.append("")
    lines.append("  ※ この議事録は確定版ではなく、関係者の確認・承認を経て最終版となります。")
    lines.append("  ※ Not final — circulates for review and approval before being confirmed.")

    lines += [
        _DIVIDER_TXT,
        "Generated by TranscriptAI · github.com/aiKunalBisht/Transcript-ai",
    ]

    return "\n".join(lines)


def format_gijiroku(analysis: dict, as_markdown: bool = False, **kwargs) -> str:
    """
    Convenience wrapper — this is what main.py actually imports.
    analysis: the full analyze_transcript() result dict
    as_markdown: False → render_text(); True → render_markdown()
    """
    plan = GijirokulFormatter().format(analysis, **kwargs)
    return render_markdown(plan) if as_markdown else render_text(plan)