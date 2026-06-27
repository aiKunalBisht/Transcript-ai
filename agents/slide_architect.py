"""
agents/slide_architect.py — v2
Converts TranscriptAI analysis result into a validated slide plan.

v2 CHANGE OF APPROACH vs v1:
v1 asked the LLM to look at the whole analysis_result and freestyle a slide
deck. That meant the most sensitive content in the deck — "what did the
Japanese side actually mean" and "what should we be worried about" — was
left to the model to paraphrase or invent, with no grounding in the
detectors that were specifically built and calibrated to answer those two
questions (soft_rejection_detector.py, conversation_dynamics.py).

v2 splits the deck into:
  - DETERMINISTIC slides (said_vs_meant, risk_watch) — built directly from
    soft_rejections / conversation_dynamics output in this file, zero LLM
    involvement. These are the slides someone will actually be held to, so
    they're exactly as trustworthy as the detectors that produced their
    inputs — no more, no less.
  - NARRATIVE slides (bottom_line, decisions/commitments, next_steps) —
    still LLM-written, because turning "what happened" into "what we tell
    the room" genuinely benefits from synthesis. The LLM is given the
    deterministic facts as its only inputs and told not to invent beyond
    them.
  - The cover slide is assembled from both: title/context from the LLM,
    status_flag computed deterministically from the same detectors that
    drive the risk slide, so the headline badge can never disagree with
    the risk slide underneath it.

A slide is only included if it has something to say. A clean, direct
meeting with no soft-rejection signals and no conversation_dynamics flags
produces a 4-slide deck (cover, bottom line, decisions, next steps) — not
a padded deck with an empty "Risks" slide for show.
"""
import os, json, requests
from dotenv import load_dotenv
import pathlib
load_dotenv(dotenv_path=pathlib.Path(__file__).resolve().parent.parent / ".env")

from pydantic import BaseModel, Field
from typing import List, Optional


# ── Slide content models ──────────────────────────────────────────────────────

class SaidVsMeantItem(BaseModel):
    speaker:  str = "Unknown"
    said:     str                 # the original phrase (often Japanese)
    reading:  str = ""            # plain-language reading of that phrase
    meant:    str                 # what it actually signals — from the detector
    severity: str = "MEDIUM"      # HIGH | MEDIUM | LOW


class CommitmentItem(BaseModel):
    side:     str = "Unclear"     # "Japan side" | "Our team" | "Joint" | "Unclear"
    text:     str = ""
    owner:    str = ""
    deadline: str = ""


class WatchItem(BaseModel):
    flag:     str = ""
    detail:   str = ""
    severity: str = "MEDIUM"      # HIGH | MEDIUM | LOW


class Slide(BaseModel):
    slide_number: int
    slide_type: str = "content"   # cover|bottom_line|said_vs_meant|decisions|risk_watch|closing
    title: str
    bullets: List[str] = Field(default_factory=list)
    said_vs_meant: List[SaidVsMeantItem] = Field(default_factory=list)
    commitments: List[CommitmentItem] = Field(default_factory=list)
    watch_items: List[WatchItem] = Field(default_factory=list)
    speaker_notes: str = ""
    language: str = "en"
    estimated_duration_seconds: int = 60


class PresentationPlan(BaseModel):
    meeting_title: str
    status_flag: str = "ON_TRACK"   # ON_TRACK | WATCH | ESCALATE
    meeting_context: str = ""
    total_slides: int
    language: str
    executive_summary: str
    slides: List[Slide]


class _NarrativePlan(BaseModel):
    """The narrow slice of the deck the LLM is actually responsible for."""
    meeting_title: str = "Meeting Summary"
    meeting_context: str = ""
    executive_summary: str = "Meeting analysis complete."
    bottom_line_bullets: List[str] = Field(default_factory=list)
    commitments: List[CommitmentItem] = Field(default_factory=list)
    next_steps_bullets: List[str] = Field(default_factory=list)


def _filter_bullets(bullets: list, min_words: int = 4, fallback: str = None) -> list:
    """Drops vague fragments ('Fix Needed', 'On Track') the same way v1 did —
    kept as a safety net even though the prompt already asks for full sentences."""
    filtered = [b for b in bullets if isinstance(b, str) and b.strip() and len(b.split()) >= min_words]
    if filtered:
        return filtered
    return [fallback] if fallback else []


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
                "max_tokens": 1800,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        resp.raise_for_status()
        return json.loads(resp.json()["choices"][0]["message"]["content"])

    # ── Deterministic slide builders — no LLM, built straight from detector output ──

    @staticmethod
    def _infer_japan_side_speakers(soft_rejections: dict) -> set:
        """
        Heuristic, not certainty: soft_rejection_detector only matches Japanese
        phrases, so any speaker it attributes a signal to is almost certainly
        the Japan-side party in the room. This is a v1 heuristic in the same
        spirit as conversation_dynamics.py's — useful as a default label, not
        proof of nationality. Returns an empty set (not a guess) when nothing
        was detected, and the narrative prompt is told explicitly to fall back
        to per-speaker labeling rather than invent a side in that case.
        """
        names = set()
        for sig in (soft_rejections or {}).get("detected", []):
            spk = sig.get("speaker")
            if spk and spk != "Unknown":
                names.add(spk)
        return names

    @staticmethod
    def _severity_rank(sev: str) -> int:
        return {"HIGH": 3, "MEDIUM": 2, "LOW": 1}.get((sev or "").upper(), 0)

    def _build_said_vs_meant_slide(self, analysis_result: dict, slide_number: int) -> Optional[Slide]:
        soft = analysis_result.get("soft_rejections", {}) or {}
        detected = soft.get("detected", [])
        if not detected:
            return None

        ranked = sorted(
            detected,
            key=lambda s: (self._severity_rank(s.get("severity", "")), s.get("confidence", 0)),
            reverse=True,
        )
        items = [
            SaidVsMeantItem(
                speaker=sig.get("speaker", "Unknown"),
                said=sig.get("phrase", ""),
                reading=sig.get("reading", ""),
                meant=sig.get("explanation", ""),
                severity=(sig.get("severity") or "MEDIUM").upper(),
            )
            for sig in ranked[:4]
        ]
        bullets = [f'{i.speaker} said "{i.reading}" — reads as: {i.meant}' for i in items[:3]]

        return Slide(
            slide_number=slide_number,
            slide_type="said_vs_meant",
            title="What They Said vs. What It Meant",
            bullets=bullets,
            said_vs_meant=items,
            speaker_notes=(
                "These are the indirect phrases used in the meeting and what they "
                "actually signal in Japanese business communication — this is a "
                "read on intent, not a literal translation."
            ),
            estimated_duration_seconds=90,
        )

    def _build_risk_watch_slide(self, analysis_result: dict, slide_number: int) -> Optional[Slide]:
        soft     = analysis_result.get("soft_rejections", {}) or {}
        dynamics = analysis_result.get("conversation_dynamics", {}) or {}
        watch_items: List[WatchItem] = []

        risk_level = (soft.get("risk_level") or "NONE").upper()
        if risk_level not in ("NONE", "MINIMAL"):
            watch_items.append(WatchItem(
                flag=f"Soft rejection risk: {risk_level}",
                detail=soft.get("risk_summary", ""),
                severity="HIGH" if risk_level == "HIGH" else "MEDIUM",
            ))

        for stall in dynamics.get("topic_stalls", []):
            watch_items.append(WatchItem(
                flag=f"Topic stalled — {stall.get('stalled_speaker', 'unknown speaker')}",
                detail=stall.get("explanation", ""),
                severity="MEDIUM",
            ))

        for pivot in dynamics.get("senior_silence_pivots", []):
            watch_items.append(WatchItem(
                flag=f"{pivot.get('senior_speaker', 'A senior speaker')} went quiet",
                detail=pivot.get("explanation", ""),
                severity="MEDIUM",
            ))

        closer = dynamics.get("closing_summarizer", {}) or {}
        if closer.get("detected"):
            watch_items.append(WatchItem(
                flag=f"{closer.get('speaker', 'A quiet speaker')} controlled the close",
                detail=closer.get("explanation", ""),
                severity="LOW",
            ))

        if not watch_items:
            return None  # nothing to flag — no slide, rather than an empty one for show

        # Highest severity first
        watch_items.sort(key=lambda w: self._severity_rank(w.severity), reverse=True)
        bullets = [f"{w.flag}: {w.detail}" for w in watch_items[:4]]

        return Slide(
            slide_number=slide_number,
            slide_type="risk_watch",
            title="Risk & Watch Items",
            bullets=bullets,
            watch_items=watch_items[:4],
            speaker_notes=(
                "These are language and structural signals worth a deliberate "
                "follow-up — not necessarily a crisis, but not nothing either."
            ),
            estimated_duration_seconds=75,
        )

    @staticmethod
    def _compute_status_flag(analysis_result: dict) -> str:
        soft     = analysis_result.get("soft_rejections", {}) or {}
        risk     = (soft.get("risk_level") or "NONE").upper()
        dynamics = analysis_result.get("conversation_dynamics", {}) or {}
        dyn_events = (
            len(dynamics.get("topic_stalls", []))
            + len(dynamics.get("senior_silence_pivots", []))
        )
        if risk == "HIGH":
            return "ESCALATE"
        if risk == "MEDIUM" or dyn_events >= 2:
            return "WATCH"
        if risk == "LOW" or dyn_events >= 1:
            return "WATCH"
        return "ON_TRACK"

    # ── Narrative prompt (the only part still written by the LLM) ────────────

    def _build_narrative_prompt(self, summary, action_details, decisions,
                                 sentiment_summary, speaker_names, japan_side,
                                 status_flag, language) -> str:
        japan_side_note = (
            f"Speakers identified as the Japan-side party, based on language "
            f"patterns in their lines: {', '.join(sorted(japan_side))}."
            if japan_side else
            "Could not confidently identify which speaker(s) are on the Japan "
            "side from language patterns. Do not guess nationality — label "
            "commitments by speaker name, or as 'Unclear', instead."
        )

        return f"""You are briefing a room of Indian colleagues who were NOT in this meeting. Only 2-3 of their colleagues were actually in the room with the Japanese client or HQ contact, speaking mostly in Japanese. Your job is to translate this into a business briefing the rest of the team can act on.

MEETING SUMMARY:
{summary}

ALL SPEAKERS DETECTED: {', '.join(speaker_names) if speaker_names else 'Not detected'}
{japan_side_note}

ACTION ITEMS (owner, deadline):
{json.dumps(action_details)}

KEY DECISIONS:
{json.dumps(decisions)}

SPEAKER SENTIMENT:
{json.dumps(sentiment_summary)}

OVERALL STATUS (already computed from language-risk and conversation-pattern detectors — do not contradict this, and do not soften or dramatize it): {status_flag}

Write the following. Every bullet must be a COMPLETE, INFORMATIVE SENTENCE a presenter could read aloud — never a fragment like "Budget discussed" or "On track". Use real names and dates from the data above; never write a placeholder like "[Name]". Do not invent any decision, commitment, or outcome that isn't present in the data above.

1. meeting_title: under 8 words, specific to this meeting's actual subject — not "Team Meeting" or "Client Sync".
2. meeting_context: ONE sentence describing who was in the room and why this meeting mattered, for someone who wasn't there.
3. executive_summary: ONE sentence on what happened and where things stand. Must be consistent with OVERALL STATUS — don't sound upbeat if status is ESCALATE, don't sound alarming if status is ON_TRACK.
4. bottom_line_bullets: 2-4 sentences covering what was discussed and the real takeaway.
5. commitments: a list of {{"side": "Japan side" | "Our team" | "Joint" | "Unclear", "text": "complete sentence describing the commitment", "owner": "name or empty string", "deadline": "date or empty string"}}, based only on the key decisions and action items given above.
6. next_steps_bullets: 3-5 sentences framed as "what we do in response" — not a restated task list — each naming an owner where known.

Return ONLY this exact JSON structure, no explanation, no markdown:
{{
  "meeting_title": "...",
  "meeting_context": "...",
  "executive_summary": "...",
  "bottom_line_bullets": ["...", "..."],
  "commitments": [{{"side": "...", "text": "...", "owner": "...", "deadline": "..."}}],
  "next_steps_bullets": ["...", "..."]
}}"""

    def _fallback_narrative(self, summary, summary_bullets, action_details,
                             decisions, japan_side) -> _NarrativePlan:
        """Used only if the Groq call fails — built from the same real data,
        no vague filler, honest 'none recorded' notes where data is missing."""
        bottom_line = summary_bullets[:3] if summary_bullets else (
            [summary[:140]] if summary else ["Meeting analysis is complete."]
        )
        commitments = [CommitmentItem(side="Unclear", text=f"Decided: {d}") for d in decisions[:4]]
        if not commitments:
            commitments = [CommitmentItem(side="Unclear", text="No formal decisions were recorded in this transcript.")]
        next_steps = action_details[:5] if action_details else [
            "No specific action items were identified for this meeting."
        ]
        exec_summary = (summary or "").strip()
        if len(exec_summary) > 150:
            cutoff = exec_summary.rfind(" ", 0, 150)
            exec_summary = exec_summary[:cutoff if cutoff > 0 else 150] + "…"
        if not exec_summary:
            exec_summary = "Meeting analysis complete."
        context = (
            f"Discussion involving {', '.join(sorted(japan_side))} on the Japan side."
            if japan_side else "Meeting context could not be automatically determined."
        )
        return _NarrativePlan(
            meeting_title="Meeting Summary",
            meeting_context=context,
            executive_summary=exec_summary,
            bottom_line_bullets=bottom_line,
            commitments=commitments,
            next_steps_bullets=next_steps,
        )

    # ── Public API ─────────────────────────────────────────────────────────

    def plan(self, analysis_result: dict, language: str = "en") -> PresentationPlan:
        summary_bullets = analysis_result.get("summary", []) or []
        summary = analysis_result.get("full_summary") or ""
        if not summary:
            summary = " ".join(summary_bullets) if summary_bullets else "Meeting completed."

        action_items = analysis_result.get("action_items", [])
        decisions    = analysis_result.get("key_decisions", [])
        sentiment    = analysis_result.get("sentiment", [])
        speakers     = analysis_result.get("speakers", [])
        soft         = analysis_result.get("soft_rejections", {}) or {}

        speaker_names = [s.get("name", "") for s in speakers if s.get("name")]
        japan_side    = self._infer_japan_side_speakers(soft)
        status_flag   = self._compute_status_flag(analysis_result)

        said_vs_meant_slide = self._build_said_vs_meant_slide(analysis_result, slide_number=0)
        risk_watch_slide    = self._build_risk_watch_slide(analysis_result, slide_number=0)

        action_details = []
        for item in action_items:
            task  = item.get("task", "")
            owner = item.get("owner", "TBD")
            due   = item.get("deadline", "TBD")
            if task:
                action_details.append(f"{task} (Owner: {owner}, Due: {due})")

        sentiment_summary = [
            f"{s.get('speaker', '?')}: {s.get('score', 'neutral')} — {s.get('label', '')}"
            for s in sentiment
        ]

        try:
            prompt = self._build_narrative_prompt(
                summary, action_details, decisions, sentiment_summary,
                speaker_names, japan_side, status_flag, language,
            )
            raw = self._call_groq(prompt)
            narrative = _NarrativePlan(**raw)
        except Exception:
            narrative = self._fallback_narrative(
                summary, summary_bullets, action_details, decisions, japan_side
            )

        bottom_line_bullets = _filter_bullets(narrative.bottom_line_bullets, fallback=summary[:140] or "Meeting analysis is complete.")
        next_steps_bullets  = _filter_bullets(
            narrative.next_steps_bullets,
            fallback=(action_details[0] if action_details else "No specific action items were identified for this meeting."),
        )
        commitments = [c for c in narrative.commitments if c.text.strip()] or [
            CommitmentItem(side="Unclear", text="No formal decisions were recorded in this transcript.")
        ]

        slides: List[Slide] = []
        n = 1

        slides.append(Slide(
            slide_number=n, slide_type="cover", title=narrative.meeting_title,
            speaker_notes=narrative.meeting_context, language=language,
            estimated_duration_seconds=30,
        )); n += 1

        slides.append(Slide(
            slide_number=n, slide_type="bottom_line", title="Bottom Line",
            bullets=bottom_line_bullets, speaker_notes=narrative.executive_summary,
            language=language, estimated_duration_seconds=60,
        )); n += 1

        if said_vs_meant_slide:
            said_vs_meant_slide.slide_number = n
            slides.append(said_vs_meant_slide); n += 1

        slides.append(Slide(
            slide_number=n, slide_type="decisions", title="Decisions & Commitments",
            bullets=[c.text for c in commitments[:4]], commitments=commitments,
            speaker_notes="What each side actually committed to, kept separate so nothing gets misattributed later.",
            language=language, estimated_duration_seconds=75,
        )); n += 1

        if risk_watch_slide:
            risk_watch_slide.slide_number = n
            slides.append(risk_watch_slide); n += 1

        slides.append(Slide(
            slide_number=n, slide_type="closing", title="Next Steps",
            bullets=next_steps_bullets,
            speaker_notes="What we do in response — not just a restated task list.",
            language=language, estimated_duration_seconds=45,
        ))

        return PresentationPlan(
            meeting_title=narrative.meeting_title,
            status_flag=status_flag,
            meeting_context=narrative.meeting_context,
            total_slides=len(slides),
            language=language,
            executive_summary=narrative.executive_summary,
            slides=slides,
        )