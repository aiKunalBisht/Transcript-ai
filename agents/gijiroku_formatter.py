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
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime


@dataclass
class ActionItem:
    task: str
    owner: str
    deadline: str
    flag: bool = False          # hallucination_flag from pipeline


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

    def __post_init__(self):
        if not self.generated_at:
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
        if timestamp:
            nichiji = timestamp
        else:
            nichiji = datetime.now().strftime("%Y年%m月%d日 %H:%M")

        # ── 出席者 ──────────────────────────────────────────────
        # analysis.speakers can be:
        #   list of str: ["Kenji", "Client"]
        #   list of dict: [{"name": "Kenji", "role": "Engineer"}]
        raw_speakers = analysis.get("speakers", [])
        shussekisha = []
        for s in raw_speakers:
            if isinstance(s, dict):
                name = s.get("name", "不明")
                role = s.get("role", "")
                talk = s.get("talk_time_pct", "")
                entry = name
                if role:
                    entry += f"（{role}）"
                if talk:
                    entry += f" — 発言比率 {talk}%"
                shussekisha.append(entry)
            elif isinstance(s, str):
                shussekisha.append(s)

        if not shussekisha:
            shussekisha = ["出席者情報なし / Not identified"]

        # ── 議題 ────────────────────────────────────────────────
        # Use key_decisions as starting point, fall back to summary bullets
        gidai = []
        bullets = analysis.get("summary", [])
        if isinstance(bullets, list):
            gidai = [b for b in bullets if b and len(b.strip()) > 4]
        elif isinstance(bullets, str) and bullets.strip():
            # summary is a single string — split at sentence boundaries
            import re
            sentences = re.split(r'[。.]\s*', bullets.strip())
            gidai = [s.strip() for s in sentences if len(s.strip()) > 4]

        # fallback: use full_summary first sentence
        if not gidai:
            full = analysis.get("full_summary", "")
            if full:
                gidai = [full[:120] + ("..." if len(full) > 120 else "")]
            else:
                gidai = ["議題情報なし / Agenda not extracted"]

        # ── 決定事項 ────────────────────────────────────────────
        raw_decisions = analysis.get("key_decisions", [])
        if isinstance(raw_decisions, list):
            kettei_jiko = [str(d) for d in raw_decisions if d]
        elif isinstance(raw_decisions, str):
            kettei_jiko = [raw_decisions] if raw_decisions else []

        if not kettei_jiko:
            kettei_jiko = ["明示的な決定事項なし / No explicit decisions recorded"]

        # ── アクションアイテム ──────────────────────────────────
        raw_actions = analysis.get("action_items", [])
        action_items = []
        for item in raw_actions:
            if isinstance(item, dict):
                task     = item.get("task", "")
                owner    = item.get("owner", "TBD")
                deadline = item.get("deadline", "未定")
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

        # ── 特記事項 (soft rejection risk) ──────────────────────
        tokki_jiko = None
        soft = analysis.get("soft_rejections", {}) or {}
        risk = soft.get("risk_level", "NONE")
        signals = soft.get("total_signals", 0)
        note = soft.get("cultural_note", "")

        if risk in ("MEDIUM", "HIGH") and signals > 0:
            tokki_jiko = (
                f"【コミュニケーションリスク】ソフトリジェクションリスク: {risk} "
                f"（{signals}件検出）\n"
            )
            if note:
                tokki_jiko += f"　文化的注記: {note}"

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
            language=analysis.get("language", "ja"),
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

    lines += [
        _DIVIDER_MD,
        f"*Generated by TranscriptAI — github.com/aiKunalBisht/Transcript-ai*",
        f"*{plan.generated_at}*",
    ]

    return "\n".join(lines)


def render_text(plan: GijirokulPlan) -> str:
    """Renders GijirokulPlan as plain-text 議事録 (email/Slack safe)."""
    lines = []

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
        lines.append(f"  {a.owner:<15} {a.deadline:<12} {a.task}{flag}")

    lines += [_DIVIDER_TXT, "【次回予定】", f"  {plan.jikai_yotei}"]

    if plan.tokki_jiko:
        lines += [_DIVIDER_TXT, "【特記事項】", f"  {plan.tokki_jiko}"]

    lines += [
        _DIVIDER_TXT,
        "Generated by TranscriptAI · github.com/aiKunalBisht/Transcript-ai",
    ]

    return "\n".join(lines)