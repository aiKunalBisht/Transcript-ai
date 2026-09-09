# agents/slide_architect.py
# Transforms a TranscriptAI analysis result into an ordered slide plan.
#
# No LLM call required for standard operation.
# groq_api_key accepted for a future polish pass (slide text summarisation).
#
# ── Data structure ────────────────────────────────────────────────────────────
# SlidePlan (dict):
#   meeting_title : str
#   language      : str          — "en" | "ja" | "hi" | "mixed"
#   generated_at  : str          — ISO-8601 UTC
#   slides        : list[SlideDefinition]
#
# SlideDefinition — fields depend on type:
#   cover        : title, subtitle, date, language
#   overview     : title, bullets[str], full_summary
#   action_items : title, items[{task,owner,deadline,urgency_tier,flagged}]
#   speakers     : title, speakers[{name,pct,tone,role}]
#   sentiment    : title, entries[{speaker,score,note}]
#   risk         : title, risk_level, total_signals, signals[], cultural_note, termination
#   closing      : title, takeaways[str]
#
# ── Algorithm ─────────────────────────────────────────────────────────────────
# O(N) — single pass over each list field (action_items, speakers, sentiment).
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Optional


class SlideArchitectAgent:
    """
    Plans a slide deck from an analysis result dict.

    Accepts groq_api_key for a future LLM enhancement pass;
    currently not consumed — rule-based mapping only.
    """

    _MAX_BULLETS      = 6
    _MAX_ACTION_ITEMS = 7   # 7 data rows + 1 header row ≤ slide height
    _MAX_SPEAKERS     = 8
    _MAX_SENTIMENT    = 6   # 2 rows of 3 cards

    def __init__(self, groq_api_key: Optional[str] = None):
        self._groq_api_key = groq_api_key   # reserved

    # ── Public API ─────────────────────────────────────────────────────────────

    def plan(self, result: dict, lang: str = "en") -> dict:
        """
        Transform analysis result → ordered slide definitions.

        Slide order:
          1. Cover        (always)
          2. Overview     (when summary bullets exist)
          3. Action Items (when action_items is non-empty)
          4. Speakers     (when speakers is non-empty)
          5. Sentiment    (when sentiment is non-empty)
          6. Risk         (only when risk_level is HIGH, CRITICAL, or MEDIUM)
          7. Closing      (always)

        DSA: O(N) — N = max(action_items, speakers, sentiment) items
        """
        slides: list[dict] = []

        # 1. Cover ─────────────────────────────────────────────────────────────
        title    = (result.get("meeting_title") or "Meeting Analysis").strip()
        subtitle = self._build_subtitle(result, lang)
        slides.append({
            "type":     "cover",
            "title":    title,
            "subtitle": subtitle,
            "date":     datetime.now(timezone.utc).strftime("%B %d, %Y"),
            "language": lang,
        })

        # 2. Overview ──────────────────────────────────────────────────────────
        bullets = self._extract_bullets(result)
        if bullets:
            slides.append({
                "type":         "overview",
                "title":        "Meeting Overview",
                "bullets":      bullets[: self._MAX_BULLETS],
                "full_summary": (result.get("full_summary") or "").strip(),
            })

        # 3. Action Items ──────────────────────────────────────────────────────
        raw_items = result.get("action_items") or []
        if raw_items:
            action_items = [
                {
                    # Prefer clean description (Task-1 field) over raw task text
                    "task":         (i.get("description") or i.get("task") or "").strip(),
                    "owner":        (i.get("owner") or "TBD").strip(),
                    "deadline":     (i.get("deadline") or "N/A").strip(),
                    "urgency_tier": i.get("urgency_tier") or "unknown",
                    "flagged":      bool(i.get("hallucination_flag")),
                }
                for i in raw_items[: self._MAX_ACTION_ITEMS]
                if i.get("task") or i.get("description")
            ]
            if action_items:
                slides.append({
                    "type":  "action_items",
                    "title": "Action Items",
                    "items": action_items,
                })

        # 4. Speakers ──────────────────────────────────────────────────────────
        speakers = result.get("speakers") or []
        if speakers:
            slides.append({
                "type":     "speakers",
                "title":    "Speaker Breakdown",
                "speakers": [
                    {
                        "name": (s.get("name") or "Unknown").strip(),
                        "pct":  int(s.get("talk_time_pct") or 0),
                        "tone": (s.get("tone") or "").strip(),
                        "role": (s.get("role") or "").strip(),
                    }
                    for s in speakers[: self._MAX_SPEAKERS]
                ],
            })

        # 5. Sentiment ─────────────────────────────────────────────────────────
        sentiment = result.get("sentiment") or []
        if sentiment:
            slides.append({
                "type":    "sentiment",
                "title":   "Communication Sentiment",
                "entries": [
                    {
                        "speaker": (s.get("speaker") or "Unknown").strip(),
                        "score":   (s.get("score") or "neutral").upper().strip(),
                        "note": (
                            s.get("note") or
                            s.get("communicative_function") or
                            ""
                        ).strip(),
                    }
                    for s in sentiment[: self._MAX_SENTIMENT]
                ],
            })

        # 6. Risk (only when meaningful) ───────────────────────────────────────
        soft_rej   = result.get("soft_rejections") or {}
        risk_level = (soft_rej.get("risk_level") or "NONE").upper()
        if risk_level in ("HIGH", "CRITICAL", "MEDIUM"):
            slides.append({
                "type":          "risk",
                "title":         "Soft Rejection & Trust Risk",
                "risk_level":    risk_level,
                "total_signals": int(soft_rej.get("total_signals") or 0),
                "signals":       (soft_rej.get("signals") or [])[:5],
                "cultural_note": (soft_rej.get("cultural_note") or "").strip(),
                "termination":   bool(soft_rej.get("termination_detected")),
            })

        # 7. Closing ───────────────────────────────────────────────────────────
        slides.append({
            "type":      "closing",
            "title":     "Key Takeaways",
            "takeaways": self._extract_takeaways(result, bullets, soft_rej),
        })

        return {
            "meeting_title": title,
            "language":      lang,
            "generated_at":  datetime.now(timezone.utc).isoformat(),
            "slides":        slides,
        }

    # ── Private helpers ────────────────────────────────────────────────────────

    def _build_subtitle(self, result: dict, lang: str) -> str:
        lang_labels = {
            "ja":    "Japanese Business Meeting · 日本語",
            "en":    "English Meeting Analysis",
            "hi":    "Hindi Meeting Analysis · हिंदी",
            "mixed": "Multilingual Meeting Analysis",
        }
        base    = lang_labels.get(lang, "Meeting Analysis")
        verdict = (
            result.get("deal_outcome") or
            result.get("meeting_outcome") or
            result.get("verdict")
        )
        if verdict:
            return f"{base}  ·  {verdict}"
        return base

    def _extract_bullets(self, result: dict) -> list[str]:
        """
        Pull summary bullets from result.
        Prefers result["summary"] (list) over result["full_summary"] (str).
        O(N) where N = number of summary items.
        """
        summary = result.get("summary") or []

        if isinstance(summary, list):
            cleaned = []
            for b in summary:
                text = str(b).strip().lstrip("•-– ").strip()
                if text:
                    cleaned.append(text)
            return cleaned

        if isinstance(summary, str) and summary.strip():
            return [
                line.strip().lstrip("•-– ").strip()
                for line in summary.splitlines()
                if line.strip()
            ]

        # Fallback: sentence-split full_summary
        full = (result.get("full_summary") or "").strip()
        if full:
            sentences = re.split(r"(?<=[.。!?])\s+", full)
            return [s.strip() for s in sentences[:6] if len(s.strip()) > 10]

        return []

    def _extract_takeaways(
        self,
        result: dict,
        bullets: list[str],
        soft_rej: dict,
    ) -> list[str]:
        """
        Build closing takeaways:
          - last 2 summary bullets (the most conclusive points)
          - action item owner list
          - risk signal if present
        O(N) where N = number of action items.
        """
        takeaways: list[str] = []

        if bullets:
            takeaways.extend(bullets[-2:])

        items = result.get("action_items") or []
        owners = sorted({
            i.get("owner") for i in items
            if i.get("owner") and i.get("owner") not in ("TBD", "Unknown", "")
        })
        if owners:
            takeaways.append(
                f"Action items assigned to: {', '.join(owners)}"
            )

        risk_level = (soft_rej.get("risk_level") or "NONE").upper()
        if risk_level not in ("NONE", "MINIMAL", ""):
            n = soft_rej.get("total_signals", 0)
            takeaways.append(
                f"Risk level: {risk_level} — {n} signal{'s' if n != 1 else ''} detected"
            )

        return takeaways or ["Review the full analysis for detailed insights."]


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    mock_result = {
        "meeting_title": "Q3 Budget Review — Acme Corp",
        "full_summary":  "The team reviewed Q3 budget overruns. Action items were assigned.",
        "summary": [
            "Q3 budget exceeded by 12% due to engineering hires.",
            "Client raised concerns about delivery timeline.",
            "Next review scheduled for end of quarter.",
        ],
        "action_items": [
            {"task": "Send revised budget report", "description": "Send revised budget report",
             "owner": "Alice", "deadline": "by Friday", "urgency_tier": "this_week"},
            {"task": "上司に相談して書面でご回答します", "description": "上司に相談して書面でご回答します",
             "owner": "Kenji", "deadline": "within 2 hours", "urgency_tier": "immediate"},
        ],
        "speakers": [
            {"name": "Alice",  "talk_time_pct": 45, "tone": "assertive"},
            {"name": "Kenji",  "talk_time_pct": 35, "tone": "deferential"},
            {"name": "Client", "talk_time_pct": 20, "tone": "concerned"},
        ],
        "sentiment": [
            {"speaker": "Alice",  "score": "professional", "note": "Direct and solution-focused"},
            {"speaker": "Kenji",  "score": "defensive",    "note": "Escalation + relationship preservation"},
            {"speaker": "Client", "score": "concerned",    "note": "Trust fragile but recoverable"},
        ],
        "soft_rejections": {
            "risk_level": "HIGH",
            "total_signals": 3,
            "signals": [
                {"phrase": "上司に相談します", "english": "I need to consult my manager", "speaker": "Kenji"},
            ],
            "cultural_note": "The escalation phrase signals authority deferral, not nemawashi.",
        },
        "_detected_language": "ja",
    }

    agent = SlideArchitectAgent(groq_api_key=None)
    plan  = agent.plan(mock_result, lang="ja")

    print(f"Slides generated: {len(plan['slides'])}")
    for s in plan["slides"]:
        print(f"  [{s['type']:14s}]", end=" ")
        if s["type"] == "action_items":
            print(f"items={len(s['items'])}, "
                  f"urgencies={[i['urgency_tier'] for i in s['items']]}")
        elif s["type"] == "speakers":
            print(f"speakers={[sp['name'] for sp in s['speakers']]}")
        elif s["type"] == "closing":
            print(f"takeaways={len(s['takeaways'])}")
        else:
            print()

    print("\nFull plan (JSON):")
    print(json.dumps(plan, ensure_ascii=False, indent=2))