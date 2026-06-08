"""
agents/slide_architect.py
Converts TranscriptAI analysis result into a validated slide plan.
"""
import os, json, requests
from dotenv import load_dotenv
import pathlib
load_dotenv(dotenv_path=pathlib.Path(__file__).resolve().parent.parent / ".env")

from pydantic import BaseModel, Field
from typing import List, Optional


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
                "temperature": 0.1,
                "max_tokens": 2000,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return json.loads(content)

    def plan(self, analysis_result: dict, language: str = "en") -> PresentationPlan:
        summary      = analysis_result.get("full_summary") or ""
        if not summary:
            bullets_list = analysis_result.get("summary", [])
            summary = " ".join(bullets_list) if bullets_list else "Meeting completed."

        action_items = analysis_result.get("action_items", [])
        decisions    = analysis_result.get("key_decisions", [])
        soft         = analysis_result.get("soft_rejections", {}) or {}
        soft_risk    = soft.get("risk_level", "NONE")
        soft_signals = soft.get("total_signals", 0)

        prompt = f"""You are a presentation architect. Convert this meeting analysis into a slide deck.
Return ONLY a valid JSON object. No explanation, no markdown, no code fences.

Meeting Summary: {summary}
Action Items: {json.dumps([i.get("task", "") for i in action_items])}
Key Decisions: {json.dumps(decisions)}
Soft Rejection Risk: {soft_risk} ({soft_signals} signals)
Language: {language}

Return exactly this JSON structure:
{{
  "meeting_title": "short title for the meeting",
  "total_slides": 5,
  "language": "{language}",
  "executive_summary": "one sentence summary of the entire meeting",
  "slides": [
    {{
      "slide_number": 1,
      "title": "Meeting Overview",
      "bullets": ["first key point", "second key point", "third key point"],
      "speaker_notes": "Two or three full sentences explaining this slide.",
      "language": "{language}",
      "estimated_duration_seconds": 60
    }}
  ]
}}

Rules:
- First slide must be the overview/title slide
- Last slide must be Next Steps or Action Items
- If soft rejection risk is MEDIUM or HIGH, include an Unresolved Items slide
- Each slide must have 2 to 5 bullet points
- Each bullet point must be under 12 words
- Total slides must be between 4 and 7
- speaker_notes must be 2-3 full sentences"""

        try:
            raw  = self._call_groq(prompt)
            plan = PresentationPlan(**raw)
            # Validate bullets are lists not empty
            for slide in plan.slides:
                if not slide.bullets:
                    slide.bullets = ["See full transcript for details"]
            return plan
        except Exception as e:
            # Build fallback from whatever summary we have
            return self._build_fallback(summary, action_items, language)

    def _build_fallback(self, summary: str, action_items: list, language: str) -> PresentationPlan:
        action_bullets = [i.get("task", "") for i in action_items[:4]] or ["Review transcript"]

        return PresentationPlan(
            meeting_title="Meeting Summary",
            total_slides=3,
            language=language,
            executive_summary=summary[:120] if summary else "Meeting analysis complete.",
            slides=[
                Slide(
                    slide_number=1,
                    title="Meeting Overview",
                    bullets=[summary[:80]] if summary else ["Meeting analysis complete"],
                    speaker_notes="This slide provides an overview of the meeting discussed.",
                    language=language,
                    estimated_duration_seconds=45,
                ),
                Slide(
                    slide_number=2,
                    title="Key Discussion Points",
                    bullets=["Review full transcript for details", "Analysis results available in JSON export"],
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
        