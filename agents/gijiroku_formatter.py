"""
agents/gijiroku_formatter.py
議事録 (Gijiroku) — Japanese Formal Business Meeting Minutes Formatter

Rule-based. No LLM call. Restructures analysis_result into the
standard Japanese business meeting minutes format.

Fields mapped from analysis_result:
  meeting_title     → 会議名
  timestamp         → 日時
  speakers          → 出席者
  key_decisions     → 決定事項
  action_items      → アクションアイテム (owner, task, deadline)
  summary/bullets   → 議題 (agenda items inferred from summary)
  soft_rejections   → appended as note under 特記事項 if risk>=MEDIUM
  recorder          → 記録者 (passed in or defaults to TranscriptAI)

Output: plain dict ready for both .txt and .md rendering.
Caller decides the format — this class only structures the data.

FIX (v2): analysis.get(key, default) only falls back to `default` when the
key is MISSING — not when it's present with an explicit null. The analyzer's
LLM step occasionally returns null instead of [] for an empty list field
(more likely on harder/garbled transcripts), and _validate_and_fill()'s
setdefault() doesn't catch that either, since the key already exists. That
previously left raw_decisions/raw_speakers/raw_actions/bullets as None,
which either skipped an assignment entirely (NameError on kettei_jiko) or
tried to iterate None directly (TypeError). Every extraction below now uses
the same `analysis.get(key, default) or default` guard that soft_rejections/
conversation_dynamics/role_hints already used further down in this file.

FIX (v3): the 日時 (nichiji) line used to build its value with
datetime.now().strftime("%Y年%m月%d日 %H:%M") — i.e. literal Japanese
characters embedded directly inside the strftime format string. On Windows,
CPython's strftime hands that format string to the CRT, which has to
round-trip it through the process's active locale codepage before calling
the underlying C function. On any machine whose active locale isn't a
Japanese codepage, that round-trip can't encode 年/月/日 and raises:
    UnicodeEncodeError: 'locale' codec can't encode character '\u5e74'
    in position 2: encoding error
(position 2 is exactly where 年 sits in "%Y年%m月%d日 %H:%M" — confirmed
against the real error text, not guessed). This is a known, widely
reported Windows-only CRT quirk; Linux's glibc doesn't round-trip through
a codepage this way, so it won't reproduce in most container deployments —
which is exactly why it can stay hidden until someone runs the app locally
on a non-Japanese-locale Windows machine. The fix below never lets a CJK
literal anywhere near strftime: only the numeric %Y/%m/%d/%H/%M-equivalent
fields (pure ASCII, always safe) are produced via formatting; the
年/月/日 literals are spliced in afterward with a plain f-string.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime


def _get(d: dict, key: str, default):
    """dict.get() that also coalesces an explicit None to `default` — the
    one-line fix for the whole class of bug this file hit."""
    v = d.get(key, default)
    return v if v is not None else default


@dataclass
class ActionItem:
    task: str
    owner: str
    deadline: str
    flag: bool = False          # hallucination_flag from pipeline


@dataclass
class ApprovalStep:
    """
    One stamp in a ringi-sho-style approval chain. The ringi-sho (稟議書) is
    the actual paper-trail-of-record for a decision in many Japanese
    companies — a written proposal that picks up a hanko (stamp) from each
    manager in turn, often a better audit trail than the meeting minutes
    themselves.

    approver/role come from speaker_normalizer.extract_role_hints() — rule-
    based, derived from the raw transcript label, not from the LLM. If no
    role hints were detected in the transcript, this falls back to a single
    "unassigned" placeholder rather than guessing a chain.
    """
    approver: str
    role: str = ""
    status: str = "未承認"   # 未承認 (pending) / 承認済み (approved)


@dataclass
class GijirokulPlan:
    """Structured 議事録 ready for any renderer."""
    kaigi_mei: str              # 会議名 — meeting name
    nichiji: str                # 日時 — date/time
    basho: str                  # 場所 — location/platform
    shussekisha: List[str]      # 出席者 — attendees with roles
    gidai: List[str]            # 議題 — agenda items
    kettei_jiko: List[str]      # 決定事項 — decisions made
    action_items: List[ActionItem]   # アクションアイテム
    jikai_yotei: str            # 次回予定 — next meeting
    kirokusha: str              # 記録者 — recorder
    tokki_jiko: Optional[str]   # 特記事項 — special notes (soft rejection warnings)
    language: str = "ja"        # source language of transcript
    generated_at: str = ""
    approval_chain: List[ApprovalStep] = field(default_factory=list)  # 承認状況
    kairan_jotai: str = "ドラフト — 関係者の確認待ち / Draft — pending circulation for review"

    def __post_init__(self):
        if not self.generated_at:
            # ASCII-only format string ("-", ":", " ") — never touches the
            # locale codepage, so this one was never at risk. Left as-is.
            self.generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")


class GijirokulFormatter:
    """
    Converts TranscriptAI analysis_result → GijirokulPlan.
    Pure rule-based mapping. Zero API calls.
    """

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
        # FIX (v3): do NOT pass 年/月/日 as literal characters inside a
        # strftime format string — see module docstring for why. Only the
        # numeric fields go through Python's own :02d zero-padding (pure
        # Python, not the CRT); the CJK literals are spliced in afterward.
        if timestamp:
            nichiji = timestamp
        else:
            now = datetime.now()
            nichiji = (
                f"{now.year}年{now.month:02d}月{now.day:02d}日 "
                f"{now.hour:02d}:{now.minute:02d}"
            )

        # ── 出席者 ──────────────────────────────────────────────
        # analysis.speakers can be:
        #   list of str: ["Kenji", "Client"]
        #   list of dict: [{"name": "Kenji", "role": "Engineer"}]
        # FIX: guard against an explicit `null` from the LLM, not just a
        # missing key — `.get(key, [])` alone does NOT catch that case.
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
        # Use key_decisions as starting point, fall back to summary bullets
        gidai = []
        bullets = _get(analysis, "summary", [])
        if isinstance(bullets, list):
            gidai = [b for b in bullets if isinstance(b, str) and len(b.strip()) > 4]
        elif isinstance(bullets, str) and bullets.strip():
            # summary is a single string — split at sentence boundaries
            import re
            sentences = re.split(r'[。.]\s*', bullets.strip())
            gidai = [s.strip() for s in sentences if len(s.strip()) > 4]

        # fallback: use full_summary first sentence
        if not gidai:
            full = analysis.get("full_summary") or ""
            if full:
                gidai = [full[:120] + ("..." if len(full) > 120 else "")]
            else:
                gidai = ["議題情報なし / Agenda not extracted"]

        # ── 決定事項 ────────────────────────────────────────────
        # FIX: this is the exact spot that crashed — if raw_decisions was
        # None, neither branch below fired and kettei_jiko was referenced
        # before assignment (NameError). `_get(...)` makes that impossible.
        raw_decisions = _get(analysis, "key_decisions", [])
        kettei_jiko: List[str] = []
        if isinstance(raw_decisions, list):
            kettei_jiko = [str(d) for d in raw_decisions if d]
        elif isinstance(raw_decisions, str):
            kettei_jiko = [raw_decisions] if raw_decisions else []

        if not kettei_jiko:
            kettei_jiko = ["明示的な決定事項なし / No explicit decisions recorded"]

        # ── アクションアイテム ──────────────────────────────────
        # FIX: same guard — `for item in raw_actions` would raise TypeError
        # if action_items came back as an explicit null.
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

        # ── 特記事項 (soft rejection risk + conversation dynamics) ──
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

        # ── 承認状況 (ringi-sho style approval chain) ───────────
        # Seniority-ranked from role_hints (rule-based, see speaker_normalizer
        # .extract_role_hints) — never guessed when no roles are detected.
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

        # ── Assemble ────────────────────────────────────────────
        return GijirokulPlan(
            kaigi_mei=kaigi_mei,
            nichiji=nichiji,
            basho=basho,
            shussekisha=shussekisha,
            gidai=gidai,
            kettei_jiko=kettei_jiko,
            action_items=action_items,
            jikai_yotei=jikai_yotei,
            kirokusha=recorder,
            tokki_jiko=tokki_jiko,
            language=analysis.get("language") or "ja",
            approval_chain=approval_chain,
        )


# ─────────────────────────────────────────────────────────────
# Renderers — plain text and markdown
# ─────────────────────────────────────────────────────────────

_DIVIDER_MD  = "\n---\n"
_DIVIDER_TXT = "\n" + "─" * 50 + "\n"


def render_markdown(plan: GijirokulPlan) -> str:
    """Renders GijirokulPlan as clean Markdown 議事録."""
    lines = []

    lines += [
        f"# 議事録",
        f"",
        f"| 項目 | 内容 |",
        f"|------|------|",
        f"| **会議名** | {plan.kaigi_mei} |",
        f"| **日時** | {plan.nichiji} |",
        f"| **場所** | {plan.basho} |",
        f"| **記録者** | {plan.kirokusha} |",
        f"| **作成日時** | {plan.generated_at} |",
        f"",
    ]

    lines += [_DIVIDER_MD, "## 出席者", ""]
    for s in plan.shussekisha:
        lines.append(f"- {s}")
    lines.append("")

    lines += [_DIVIDER_MD, "## 議題", ""]
    for i, item in enumerate(plan.gidai, 1):
        lines.append(f"{i}. {item}")
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
        f"*Generated by TranscriptAI — github.com/aiKunalBisht/Transcript-ai*",
        f"*{plan.generated_at}*",
    ]

    return "\n".join(lines)


def render_text(plan: GijirokulPlan) -> str:
    """Renders GijirokulPlan as plain-text 議事録 (email/Slack safe)."""
    lines = []

    # FIX: format-spec width (e.g. "{x:<15}") raises TypeError on None —
    # plain f-string interpolation ({x}) does not, since it implicitly
    # calls str(). Coalescing here makes render_text() safe even if some
    # upstream field slips through as None.
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
    GijirokulFormatter/render_text/render_markdown above are the building
    blocks; main.py expects a single function: dict in, finished string out.

    analysis: the full analyze_transcript() result dict
    as_markdown: False (default) -> render_text(); True -> render_markdown()
    kwargs: forwarded to GijirokulFormatter().format() (recorder, basho,
            jikai_yotei, timestamp) if a caller ever wants to override them
    """
    plan = GijirokulFormatter().format(analysis, **kwargs)
    return render_markdown(plan) if as_markdown else render_text(plan)