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
        summary      = analysis_result.get("full_summary") or ""
        if not summary:
            bullets_list = analysis_result.get("summary", [])
            summary = " ".join(bullets_list) if bullets_list else "Meeting completed."

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
- Middle slides: One theme each — discussion points, decisions, risks, team updates
- Last slide: Next Steps — every action item as a complete bullet with owner and deadline
- If soft rejection risk is MEDIUM or HIGH: include an "Unresolved Items" slide
- 4 to 6 slides total
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
            return plan
        except Exception as e:
            return self._build_fallback(summary, action_items, action_details, language)

    def _build_fallback(self, summary: str, action_items: list,
                        action_details: list, language: str) -> PresentationPlan:
        action_bullets = action_details[:4] if action_details else [
            "Review full transcript for action items"
        ]

        return PresentationPlan(
            meeting_title="Meeting Summary",
            total_slides=3,
            language=language,
            executive_summary=summary[:120] if summary else "Meeting analysis complete.",
            slides=[
                Slide(
                    slide_number=1,
                    title="Meeting Overview",
                    bullets=[
                        summary[:80] if len(summary) > 8 else "Meeting analysis is complete",
                        "Full details available in the transcript analysis",
                        "Review action items on the final slide",
                    ],
                    speaker_notes="This slide provides an overview of the meeting discussed today.",
                    language=language,
                    estimated_duration_seconds=45,
                ),
                Slide(
                    slide_number=2,
                    title="Key Discussion Points",
                    bullets=[
                        "Full transcript analysis is available in the JSON export",
                        "Speaker sentiment and talk time captured in the analysis",
                        "Communication signals and risk level assessed by pipeline",
                    ],
                    speaker_notes="The key discussion points have been captured in the transcript analysis.",
                    language=language,
                    estimated_duration_seconds=60,
                ),
                Slide(
                    slide_number=3,
                    title="Next Steps & Action Items",
                    bullets=action_bullets,
                    speaker_notes="These are the action items identified from the meeting. Please follow up accordingly.",
                    language=language,
                    estimated_duration_seconds=45,
                ),
            ],
        )