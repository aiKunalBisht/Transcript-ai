"""
agents/slide_architect.py
Converts TranscriptAI analysis result into a validated slide plan.
"""
import os, json, requests
from dotenv import load_dotenv
import pathlib
load_dotenv(dotenv_path=pathlib.Path(__file__).resolve().parent.parent / ".env")

from pydantic import BaseModel, Field
from typing import List


class Slide(BaseModel):
    slide_number: int
    title: str
    bullets: List[str]
    speaker_notes: str
    language: str = "en"
    estimated_duration_seconds: int = 60


class PresentationPlan(BaseModel):
    meeting_title: str
    total_slides: int
    language: str
    executive_summary: str
    slides: List[Slide]


class SlideArchitectAgent:
    def __init__(self, groq_api_key: str):
        self.api_key = groq_api_key
        self.model   = "llama-3.3-70b-versatile"

    def _call_groq(self, prompt: str) -> dict:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.model,
                "temperature": 0.2,
                "max_tokens": 2500,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        resp.raise_for_status()
        return json.loads(resp.json()["choices"][0]["message"]["content"])

    def plan(self, analysis_result: dict, language: str = "en") -> PresentationPlan:
        summary_bullets = analysis_result.get("summary", []) or []
        summary      = analysis_result.get("full_summary") or ""
        if not summary:
            summary = " ".join(summary_bullets) if summary_bullets else "Meeting completed."

        action_items = analysis_result.get("action_items", [])
        decisions    = analysis_result.get("key_decisions", [])
        sentiment    = analysis_result.get("sentiment", [])
        soft         = analysis_result.get("soft_rejections", {}) or {}
        soft_risk    = soft.get("risk_level", "NONE")
        soft_signals = soft.get("total_signals", 0)
        speakers     = analysis_result.get("speakers", [])

        # Build rich context for the LLM
        action_details = []
        for item in action_items:
            task  = item.get("task", "")
            owner = item.get("owner", "TBD")
            due   = item.get("deadline", "TBD")
            if task:
                action_details.append(f"{task} (Owner: {owner}, Due: {due})")

        sentiment_summary = []
        for s in sentiment:
            sentiment_summary.append(
                f"{s.get('speaker','?')}: {s.get('score','neutral')} — {s.get('label','')}"
            )

        prompt = f"""You are a senior presentation architect. Convert this meeting analysis into a polished, professional slide deck.

MEETING SUMMARY:
{summary}

ACTION ITEMS (with owners and deadlines):
{json.dumps(action_details)}

KEY DECISIONS:
{json.dumps(decisions)}

SPEAKER SENTIMENT:
{json.dumps(sentiment_summary)}

SOFT REJECTION RISK: {soft_risk} ({soft_signals} signals detected)
LANGUAGE: {language}

CRITICAL BULLET POINT RULES — READ CAREFULLY:
- Every bullet must be a COMPLETE, INFORMATIVE PHRASE of 6 to 12 words
- Include NAMES of speakers, owners, deadlines where known
- NEVER write fragments like "Fix Needed", "Minor Bug", "No Risks", "None", "On Track"
- BAD: "Backend Issue" | GOOD: "Vikram identified a backend bug blocking the sprint deadline"
- BAD: "Direct Communication" | GOOD: "Sharma Sir emphasized direct escalation for all blockers"
- BAD: "Fix Needed" | GOOD: "Priya committed to resolving the bug by tomorrow morning"
- Each bullet must give a reader NEW information they could act on

Return ONLY this exact JSON structure, no explanation, no markdown:
{{
  "meeting_title": "descriptive title under 8 words",
  "total_slides": 5,
  "language": "{language}",
  "executive_summary": "one complete sentence summarising the meeting outcome",
  "slides": [
    {{
      "slide_number": 1,
      "title": "slide title",
      "bullets": [
        "complete informative phrase with 6-12 words naming actors",
        "another complete informative phrase with specific detail",
        "third complete phrase with outcome or decision"
      ],
      "speaker_notes": "Two to three full sentences a presenter would actually say.",
      "language": "{language}",
      "estimated_duration_seconds": 60
    }}
  ]
}}

SLIDE STRUCTURE RULES:
- Slide 1: Title/Overview — summarise the meeting purpose and outcome
- Middle slides: One theme each — discussion points, decisions, speaker sentiment/risk, team updates
- Last slide: Next Steps — every action item as a complete bullet with owner and deadline
- If soft rejection risk is MEDIUM or HIGH: include a dedicated "Unresolved Items" slide
- MINIMUM 5 slides, up to 7 — 4 slides is NOT enough, do not under-produce
- Each slide: exactly 3 to 4 bullets
- speaker_notes: 2-3 full sentences, conversational, presenter-ready"""

        try:
            raw  = self._call_groq(prompt)
            plan = PresentationPlan(**raw)
            # Post-process: replace any remaining fragments
            for slide in plan.slides:
                slide.bullets = [
                    b for b in slide.bullets
                    if b and len(b.split()) >= 4
                ] or ["See full transcript for complete details"]
            # Enforce the 5-slide minimum even if the model under-produced —
            # better to use the real-data fallback than ship a thin deck.
            if len(plan.slides) < 5:
                raise ValueError(f"Model returned only {len(plan.slides)} slides, need >= 5")
            plan.total_slides = len(plan.slides)
            return plan
        except Exception as e:
            return self._build_fallback(
                summary, summary_bullets, action_items, action_details,
                decisions, sentiment_summary, soft_risk, soft_signals, language
            )

    def _build_fallback(self, summary: str, summary_bullets: list, action_items: list,
                        action_details: list, decisions: list, sentiment_summary: list,
                        soft_risk: str, soft_signals: int, language: str) -> PresentationPlan:
        """
        Used when the Groq call fails OR under-produces. Builds 5 real slides
        from data the pipeline already extracted — no vague filler like
        "Full details available in the transcript analysis". If a given
        section genuinely has no data, says so honestly instead of padding.
        """
        # Slide 1 — Overview: real summary bullets if we have them
        overview_bullets = (summary_bullets[:3] if summary_bullets
                             else [summary[:100]] if summary
                             else ["Meeting analysis is complete."])

        # Slide 2 — Key Decisions: real decisions, or an honest "none" note
        decision_bullets = (
            [f"Decided: {d}" for d in decisions[:4]] if decisions
            else ["No formal decisions were recorded — discussion remained exploratory."]
        )

        # Slide 3 — Speaker Sentiment & Communication Signals
        sentiment_bullets = list(sentiment_summary[:3])
        if soft_risk in ("MEDIUM", "HIGH") and soft_signals:
            sentiment_bullets.append(
                f"Soft rejection risk: {soft_risk} ({soft_signals} signals) — follow up explicitly to confirm status."
            )
        if not sentiment_bullets:
            sentiment_bullets = ["Speaker sentiment data was not available for this transcript."]

        # Slide 4 — Discussion Highlights: remaining summary bullets, or a
        # second pass over full_summary if there weren't enough bullets
        remaining = summary_bullets[3:6]
        discussion_bullets = remaining if remaining else (
            [summary[100:220]] if summary and len(summary) > 100
            else ["Full discussion is captured in the transcript analysis."]
        )

        # Slide 5 — Next Steps & Action Items: real action items, or honest "none"
        action_bullets = action_details[:4] if action_details else [
            "No specific action items were identified for this meeting."
        ]

        exec_summary = summary.strip()
        if len(exec_summary) > 150:
            cutoff = exec_summary.rfind(" ", 0, 150)
            exec_summary = exec_summary[:cutoff if cutoff > 0 else 150] + "…"
        if not exec_summary:
            exec_summary = "Meeting analysis complete."

        slides = [
            Slide(
                slide_number=1,
                title="Meeting Overview",
                bullets=overview_bullets,
                speaker_notes="This slide covers what the meeting was about and how it concluded.",
                language=language,
                estimated_duration_seconds=45,
            ),
            Slide(
                slide_number=2,
                title="Key Decisions",
                bullets=decision_bullets,
                speaker_notes="These are the decisions that came out of the discussion, or a note that none were formally made.",
                language=language,
                estimated_duration_seconds=50,
            ),
            Slide(
                slide_number=3,
                title="Speaker Sentiment & Communication Signals",
                bullets=sentiment_bullets,
                speaker_notes="This covers how each speaker came across, plus any indirect or soft-rejection signals worth following up on.",
                language=language,
                estimated_duration_seconds=50,
            ),
            Slide(
                slide_number=4,
                title="Discussion Highlights",
                bullets=discussion_bullets,
                speaker_notes="A closer look at the main points raised during the discussion.",
                language=language,
                estimated_duration_seconds=50,
            ),
            Slide(
                slide_number=5,
                title="Next Steps & Action Items",
                bullets=action_bullets,
                speaker_notes="These are the action items identified from the meeting — please follow up accordingly.",
                language=language,
                estimated_duration_seconds=45,
            ),
        ]

        return PresentationPlan(
            meeting_title="Meeting Summary",
            total_slides=len(slides),
            language=language,
            executive_summary=exec_summary,
            slides=slides,
        )