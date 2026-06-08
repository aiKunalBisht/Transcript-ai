"""
agents/slide_architect.py
Converts TranscriptAI analysis result into a validated slide plan.
"""
import os, json, requests
from pydantic import BaseModel, Field
from typing import List, Literal


class Slide(BaseModel):
    slide_number: int
    title: str
    bullets: List[str] = Field(..., min_length=2, max_length=5)
    speaker_notes: str
    language: str = "en"
    estimated_duration_seconds: int = Field(default=60, ge=20, le=300)


class PresentationPlan(BaseModel):
    meeting_title: str
    total_slides: int
    language: str
    executive_summary: str
    slides: List[Slide]


class SlideArchitectAgent:
    """
    Converts analysis result dict → PresentationPlan.
    Flow: call_llm → validate_pydantic → fallback_if_invalid
    """

    def __init__(self, groq_api_key: str):
        self.api_key = groq_api_key
        self.model   = "llama-3.3-70b-versatile"

    def _call_groq(self, prompt: str) -> dict:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
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
        return json.loads(resp.json()["choices"][0]["message"]["content"])

    def plan(self, analysis_result: dict, language: str = "en") -> PresentationPlan:
        summary        = analysis_result.get("full_summary") or " ".join(analysis_result.get("summary", []))
        action_items   = analysis_result.get("action_items", [])
        decisions      = analysis_result.get("key_decisions", [])
        soft           = analysis_result.get("soft_rejections", {})
        soft_risk      = soft.get("risk_level", "NONE") if soft else "NONE"
        soft_signals   = soft.get("total_signals", 0) if soft else 0

        prompt = f"""
You are a presentation architect. Convert this meeting analysis into slides.
Return ONLY valid JSON. No explanation, no markdown fences.

Meeting Summary: {summary}
Action Items: {json.dumps([i.get("task","") for i in action_items])}
Key Decisions: {json.dumps(decisions)}
Soft Rejection Risk: {soft_risk} ({soft_signals} signals detected)
Output Language: {language}

Return this exact JSON structure:
{{
  "meeting_title": "short descriptive title",
  "total_slides": <integer 4 to 8>,
  "language": "{language}",
  "executive_summary": "one sentence overview for title slide",
  "slides": [
    {{
      "slide_number": 1,
      "title": "slide title",
      "bullets": ["point one", "point two", "point three"],
      "speaker_notes": "what the presenter says here, full sentences",
      "language": "{language}",
      "estimated_duration_seconds": 60
    }}
  ]
}}

Rules:
- Slide 1 = title/overview slide always
- Last slide = Next Steps / Action Items always
- If soft_rejection risk is MEDIUM or HIGH, add an "Unresolved Items" slide
- Bullets max 10 words each, concise
- speaker_notes must be 2-3 full sentences the presenter would actually say
- 4 to 8 slides total, no more
"""
        try:
            raw  = self._call_groq(prompt)
            # Pydantic validation
            plan = PresentationPlan(**raw)
            return plan
        except Exception:
            return self._fallback_plan(summary, language)

    def _fallback_plan(self, summary: str, language: str) -> PresentationPlan:
        """Guaranteed safe plan if LLM output fails."""
        return PresentationPlan(
            meeting_title="Meeting Summary",
            total_slides=3,
            language=language,
            executive_summary=summary[:120] if summary else "Meeting analysis complete.",
            slides=[
                Slide(
                    slide_number=1,
                    title="Meeting Overview",
                    bullets=["Analysis complete", "See full report below"],
                    speaker_notes="This slide provides an overview of the meeting. Please refer to the full analysis for details.",
                    language=language,
                    estimated_duration_seconds=45,
                ),
                Slide(
                    slide_number=2,
                    title="Key Points",
                    bullets=["Review transcript for details", "Action items logged"],
                    speaker_notes="The key discussion points have been captured. Please review the transcript for full context.",
                    language=language,
                    estimated_duration_seconds=60,
                ),
                Slide(
                    slide_number=3,
                    title="Next Steps",
                    bullets=["Follow up on action items", "Schedule next meeting"],
                    speaker_notes="The team should follow up on all action items discussed. Schedule the next meeting accordingly.",
                    language=language,
                    estimated_duration_seconds=45,
                ),
            ],
        )