"""
analysis/nemawashi_sequence.py
===============================
Nemawashi sequence detector — three-stage ordered pattern across utterances.

Nemawashi (根回し) is NOT a single phrase. It is a pre-decision consensus
process that unfolds in sequence across multiple speaker turns:
  Stage 1 — Internal process invoked (社内確認 / 調整 language)
  Stage 2 — Authority escalated upward (経営陣 / 上位承認 language)
  Stage 3 — Timeline made indefinite (申し上げにくい / こちらからご連絡 language)

All three stages must appear in this order from the same speaker.
Detection returns a standalone result written to japan_insights["nemawashi"]
and result["nemawashi_sequence"] — NOT merged into approval_gate_signals.

DSA:
  _detect_sequence: O(U × M) — U utterances × M markers per stage (≤ 5)
  Total: sub-millisecond for any realistic transcript length.
"""

from __future__ import annotations
import re


# ══════════════════════════════════════════════════════════════════════════════
# STAGE MARKER SETS
# Each stage is a list of Japanese keywords.
# ANY keyword match advances the sequence to the next stage.
# ══════════════════════════════════════════════════════════════════════════════

_STAGE_1_INTERNAL_PROCESS = [
    "社内",       # internal (company-side)
    "調整",       # coordination / adjustment
    "確認",       # confirmation / check
    "社内調整",   # internal coordination (compound)
    "検討",       # deliberation / consideration
]

_STAGE_2_AUTHORITY_ESCALATION = [
    "経営陣",     # management / executives
    "上位",       # superior / upper level
    "承認",       # approval / authorization
    "関係者",     # stakeholders
    "役員",       # board / directors
    "本社",       # headquarters
]

_STAGE_3_TIMELINE_DISSOLVED = [
    "申し上げにくい",      # difficult to say (evasion phrase)
    "時間がかかります",    # it will take time (timeline dissolved)
    "ご連絡いたします",    # we will contact you (reverses contact ownership)
    "こちらから",          # from our side (exit signal)
    "追ってご連絡",        # will follow up (indefinite)
    "いずれ",              # eventually / at some point
]

NEMAWASHI_STAGES = [
    _STAGE_1_INTERNAL_PROCESS,
    _STAGE_2_AUTHORITY_ESCALATION,
    _STAGE_3_TIMELINE_DISSOLVED,
]

# Confidence tiers based on how many stages were detected
_CONFIDENCE_MAP = {3: 0.91, 2: 0.55, 1: 0.20, 0: 0.0}


# ══════════════════════════════════════════════════════════════════════════════
# CORE DETECTOR
# ══════════════════════════════════════════════════════════════════════════════

def _detect_sequence(utterances: list[dict]) -> dict:
    """
    Ordered sequence detector across speaker utterances.

    Args:
        utterances: list of {"speaker": str, "text": str}

    Returns:
        {
            "nemawashi_sequence_detected": bool,
            "confidence": float,
            "stages_completed": int,
            "sequence_positions": list[int],  # utterance indices
            "matched_markers": list[str],      # which marker triggered each stage
            "primary_speaker": str | None,
            "evidence_phrases": list[str],     # phrases that triggered detection
            "description": str,
        }
    DSA: O(U × M) — linear scan with stage bookmarking.
    """
    stage            = 0
    sequence_positions: list[int]  = []
    matched_markers:    list[str]  = []
    evidence_phrases:   list[str]  = []
    primary_speaker:    str | None = None

    for idx, utt in enumerate(utterances):
        text    = utt.get("text", "") or utt.get("utterance", "")
        speaker = utt.get("speaker", "Unknown")

        if stage >= len(NEMAWASHI_STAGES):
            break

        for marker in NEMAWASHI_STAGES[stage]:
            if marker in text:
                # First stage sets the speaker — all stages must be same speaker
                if stage == 0:
                    primary_speaker = speaker
                elif speaker != primary_speaker:
                    continue  # wrong speaker — skip, don't advance

                sequence_positions.append(idx)
                matched_markers.append(marker)

                # Extract surrounding phrase for evidence (±20 chars)
                pos   = text.index(marker)
                start = max(0, pos - 20)
                end   = min(len(text), pos + len(marker) + 20)
                evidence_phrases.append(text[start:end].strip())

                stage += 1
                break   # advance one stage per utterance

    stages_completed = stage
    detected         = stages_completed == 3
    confidence       = _CONFIDENCE_MAP.get(stages_completed, 0.0)

    if detected:
        description = (
            f"Three-stage nemawashi sequence detected across utterances "
            f"{sequence_positions} — internal process invoked, authority escalated "
            f"upward, timeline dissolved. Speaker: {primary_speaker}."
        )
    elif stages_completed == 2:
        description = (
            f"Partial nemawashi pattern: stages 1-2 detected "
            f"(internal + authority escalation) but no timeline dissolution found."
        )
    elif stages_completed == 1:
        description = (
            f"Single nemawashi signal: stage 1 only (internal process language). "
            f"Insufficient for full sequence confirmation."
        )
    else:
        description = "No nemawashi sequence detected."

    return {
        "nemawashi_sequence_detected": detected,
        "confidence":                  confidence,
        "stages_completed":            stages_completed,
        "sequence_positions":          sequence_positions,
        "matched_markers":             matched_markers,
        "primary_speaker":             primary_speaker,
        "evidence_phrases":            evidence_phrases,
        "description":                 description,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API — called by analyze_transcript Stage 12d
# ══════════════════════════════════════════════════════════════════════════════

def detect_nemawashi_sequence(fn_results: list[dict]) -> dict:
    """
    Entry point called from Stage 12d with output of detect_functions().

    detect_functions() returns a list of dicts with keys:
        speaker, communicative_function, utterance, confidence, ...

    We normalize to the {"speaker": str, "text": str} format _detect_sequence expects.
    DSA: O(U) normalization + O(U × M) detection.
    """
    utterances = []
    for item in fn_results:
        utterances.append({
            "speaker": item.get("speaker", "Unknown"),
            "text":    item.get("utterance", item.get("text", "")),
        })
    return _detect_sequence(utterances)


def detect_nemawashi_from_raw(transcript: str) -> dict:
    """
    Convenience: detect from raw 'Speaker: text' transcript string.
    Used when detect_functions() is not available.
    DSA: O(lines) parse + O(U × M) detection.
    """
    utterances = []
    pattern    = re.compile(r"^([^\:\n]+):\s*(.+)$", re.MULTILINE)
    for m in pattern.finditer(transcript):
        utterances.append({
            "speaker": m.group(1).strip(),
            "text":    m.group(2).strip(),
        })
    return _detect_sequence(utterances)


# ══════════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    _TEST_TRANSCRIPT = """
Arjun: When do you think we can finalise the contract?
田中: 少々難しい状況でございます。社内で確認が必要でございます。
Arjun: Could you give us a rough timeline?
田中: 調整プロセスを経る必要がございまして、経営陣とさらに協議が必要でございます。
Arjun: Should we set up a follow-up call?
田中: 何とも申し上げにくい状況です。こちらからご連絡いたします。
"""

    result = detect_nemawashi_from_raw(_TEST_TRANSCRIPT)
    print("=== Nemawashi Sequence Self-Test ===")
    print(f"Detected:         {result['nemawashi_sequence_detected']}")
    print(f"Confidence:       {result['confidence']:.2f}")
    print(f"Stages completed: {result['stages_completed']}/3")
    print(f"Primary speaker:  {result['primary_speaker']}")
    print(f"Markers matched:  {result['matched_markers']}")
    print(f"Evidence:         {result['evidence_phrases']}")
    print(f"Description:      {result['description']}")
    print()

    # Negative test
    _NO_NEMA = "Alice: Let's schedule a call. Bob: Sounds good, see you Friday."
    neg = detect_nemawashi_from_raw(_NO_NEMA)
    print(f"Negative test — detected: {neg['nemawashi_sequence_detected']} "
          f"(expected: False)  {'OK' if not neg['nemawashi_sequence_detected'] else 'FAIL'}")