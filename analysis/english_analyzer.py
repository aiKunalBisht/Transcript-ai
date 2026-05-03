# english_analyzer.py
# English Communication Intelligence Layer
#
# Detects communication patterns unique to English business speech:
# - Hedging & non-commitment
# - Power imbalance signals
# - Passive aggression
# - Commitment strength (will vs will try vs will see)
# - Escalation signals
# - Corporate soft rejection ("circle back", "take under advisement")

import re

# ── PATTERN LIBRARY ───────────────────────────────────────────────────────────

HEDGING_PATTERNS = [
    {"phrase": "we'll circle back",       "reading": "Deferral — no concrete next step",           "confidence": 0.80, "severity": "MEDIUM"},
    {"phrase": "circle back on this",     "reading": "Deferral — no concrete next step",           "confidence": 0.80, "severity": "MEDIUM"},
    {"phrase": "take it under advisement","reading": "Corporate soft rejection — unlikely to act",  "confidence": 0.85, "severity": "HIGH"},
    {"phrase": "take that under advisement","reading": "Corporate soft rejection",                  "confidence": 0.85, "severity": "HIGH"},
    {"phrase": "we'll see",               "reading": "Non-commitment — outcome uncertain",          "confidence": 0.65, "severity": "MEDIUM"},
    {"phrase": "let me see what I can do","reading": "Weak commitment — no promise made",           "confidence": 0.60, "severity": "MEDIUM"},
    {"phrase": "i'll try",                "reading": "Soft commitment — outcome not guaranteed",    "confidence": 0.55, "severity": "LOW"},
    {"phrase": "i'll try to",             "reading": "Soft commitment — outcome not guaranteed",    "confidence": 0.55, "severity": "LOW"},
    {"phrase": "we might be able to",     "reading": "Uncertainty — no commitment",                 "confidence": 0.65, "severity": "MEDIUM"},
    {"phrase": "it depends",              "reading": "Conditional — no direct answer",              "confidence": 0.50, "severity": "LOW"},
    {"phrase": "we'll look into it",      "reading": "Deferral — vague follow-up",                  "confidence": 0.70, "severity": "MEDIUM"},
    {"phrase": "let me look into that",   "reading": "Deferral — no commitment to outcome",         "confidence": 0.65, "severity": "MEDIUM"},
    {"phrase": "we'll get back to you",   "reading": "Deferral — timeline unspecified",             "confidence": 0.70, "severity": "MEDIUM"},
    {"phrase": "i'll get back to you",    "reading": "Deferral — timeline unspecified",             "confidence": 0.65, "severity": "MEDIUM"},
    {"phrase": "possibly",                "reading": "Low confidence — not a commitment",           "confidence": 0.45, "severity": "LOW"},
    {"phrase": "we'll see what happens",  "reading": "Non-commitment — passive outcome framing",    "confidence": 0.70, "severity": "MEDIUM"},
    {"phrase": "at this point in time",   "reading": "Corporate filler — avoids direct answer",     "confidence": 0.55, "severity": "LOW"},
    {"phrase": "going forward",           "reading": "Vague future framing — no specific plan",     "confidence": 0.40, "severity": "LOW"},
    {"phrase": "touch base",              "reading": "Vague follow-up — no commitment",             "confidence": 0.45, "severity": "LOW"},
    {"phrase": "let's revisit",           "reading": "Deferral — postponing decision",              "confidence": 0.60, "severity": "MEDIUM"},
    {"phrase": "something to consider",   "reading": "Soft suggestion — avoiding direct position",  "confidence": 0.50, "severity": "LOW"},
    {"phrase": "food for thought",        "reading": "Avoiding commitment — deflecting to thinking","confidence": 0.55, "severity": "LOW"},
]

POWER_IMBALANCE_PATTERNS = [
    {"phrase": "as i told you",           "reading": "Dominance assertion — implying listener failed to listen", "confidence": 0.80, "severity": "HIGH"},
    {"phrase": "as i said before",        "reading": "Mild dominance — repetition frustration",    "confidence": 0.70, "severity": "MEDIUM"},
    {"phrase": "let me be clear",         "reading": "Authority assertion — signals prior miscommunication", "confidence": 0.75, "severity": "MEDIUM"},
    {"phrase": "you need to understand",  "reading": "Condescending framing — power imbalance",    "confidence": 0.85, "severity": "HIGH"},
    {"phrase": "you have to",             "reading": "Direct command — authority over recipient",   "confidence": 0.70, "severity": "MEDIUM"},
    {"phrase": "i expect",                "reading": "Authority framing — setting unilateral expectation","confidence": 0.65, "severity": "MEDIUM"},
    {"phrase": "non-negotiable",          "reading": "Hard boundary — no room for discussion",     "confidence": 0.90, "severity": "HIGH"},
    {"phrase": "this is unacceptable",    "reading": "Strong dissatisfaction — relationship at risk","confidence": 0.90, "severity": "HIGH"},
    {"phrase": "i demand",                "reading": "Aggressive demand — escalation likely",       "confidence": 0.95, "severity": "HIGH"},
    {"phrase": "make no mistake",         "reading": "Authority signal — issuing a warning",        "confidence": 0.80, "severity": "HIGH"},
    {"phrase": "bottom line",             "reading": "Closing down discussion — asserting final position","confidence": 0.60, "severity": "MEDIUM"},
]

PASSIVE_AGGRESSION_PATTERNS = [
    {"phrase": "fine",                    "reading": "Hidden disagreement — compliance without agreement","confidence": 0.55, "severity": "LOW"},
    {"phrase": "whatever works for you",  "reading": "Passive resignation — disengaged from outcome","confidence": 0.65, "severity": "MEDIUM"},
    {"phrase": "if that's what you want", "reading": "Passive disagreement — distancing from decision","confidence": 0.70, "severity": "MEDIUM"},
    {"phrase": "sure, sure",              "reading": "Dismissive agreement — not genuinely engaged","confidence": 0.60, "severity": "LOW"},
    {"phrase": "i thought we agreed",     "reading": "Veiled accusation — implying bad faith",     "confidence": 0.75, "severity": "MEDIUM"},
    {"phrase": "with all due respect",    "reading": "Signals disagreement following — diplomatic friction","confidence": 0.70, "severity": "MEDIUM"},
    {"phrase": "no offense but",          "reading": "Pre-emptive disagreement — criticism incoming","confidence": 0.75, "severity": "MEDIUM"},
    {"phrase": "not to be difficult",     "reading": "Signals difficulty ahead — self-aware friction","confidence": 0.65, "severity": "LOW"},
]

ESCALATION_PATTERNS = [
    {"phrase": "i'm going to have to escalate","reading": "Formal escalation threat — management involved","confidence": 0.90, "severity": "HIGH"},
    {"phrase": "i'll have to escalate",   "reading": "Escalation threat",                           "confidence": 0.90, "severity": "HIGH"},
    {"phrase": "this is the second time", "reading": "Pattern of failure — relationship strained",  "confidence": 0.85, "severity": "HIGH"},
    {"phrase": "this keeps happening",    "reading": "Recurring issue — frustration escalating",    "confidence": 0.80, "severity": "HIGH"},
    {"phrase": "reconsider the contract", "reading": "Contract at risk — high business impact",     "confidence": 0.95, "severity": "HIGH"},
    {"phrase": "involve legal",           "reading": "Legal escalation signal",                     "confidence": 0.95, "severity": "HIGH"},
    {"phrase": "take this further",       "reading": "Escalation intent",                           "confidence": 0.85, "severity": "HIGH"},
    {"phrase": "speak to your manager",   "reading": "Bypassing contact — management escalation",  "confidence": 0.80, "severity": "HIGH"},
    {"phrase": "compensation",            "reading": "Financial remedy being demanded",             "confidence": 0.70, "severity": "MEDIUM"},
    {"phrase": "this is the last time",   "reading": "Final warning — relationship near breaking",  "confidence": 0.85, "severity": "HIGH"},
]

# Commitment strength — detected differently (look for I will vs I'll try etc.)
WEAK_COMMITMENT_MARKERS = [
    "i'll try", "i will try", "i'll attempt", "i'll see if",
    "i'll do my best", "i'll make an effort", "hopefully",
    "if possible", "if i can", "time permitting",
]

STRONG_COMMITMENT_MARKERS = [
    "i will", "i'll make sure", "i'll ensure", "i commit",
    "guaranteed", "absolutely", "count on me", "i'll get it done",
    "will be ready", "will be done", "will deliver",
]

# ── HELPERS ───────────────────────────────────────────────────────────────────
def _find_speaker(transcript: str, phrase: str) -> str:
    idx = transcript.lower().find(phrase.lower())
    if idx == -1:
        return "Unknown"
    preceding = transcript[:idx]
    matches = re.findall(r"([^\s\[\]:：\n]{2,30}?)\s*[:：]", preceding)
    speakers = [m.strip() for m in matches if not re.match(r"^\d+$", m.strip())]
    return speakers[-1] if speakers else "Unknown"

def _extract_context(transcript: str, phrase: str, window: int = 100) -> str:
    idx = transcript.lower().find(phrase.lower())
    if idx == -1:
        return ""
    start = max(0, idx - window)
    end   = min(len(transcript), idx + len(phrase) + window)
    return transcript[start:end].strip()

# ── MAIN DETECTOR ─────────────────────────────────────────────────────────────
def detect_english_patterns(transcript: str) -> dict:
    text_lower = transcript.lower()
    detected = []

    all_patterns = (
        [(p, "HEDGING")           for p in HEDGING_PATTERNS] +
        [(p, "POWER_IMBALANCE")   for p in POWER_IMBALANCE_PATTERNS] +
        [(p, "PASSIVE_AGGRESSION")for p in PASSIVE_AGGRESSION_PATTERNS] +
        [(p, "ESCALATION")        for p in ESCALATION_PATTERNS]
    )

    for pattern, category in all_patterns:
        phrase = pattern["phrase"]
        if phrase.lower() in text_lower:
            speaker = _find_speaker(transcript, phrase)
            context = _extract_context(transcript, phrase)
            detected.append({
                "phrase":      phrase,
                "reading":     pattern["reading"],
                "category":    category,
                "confidence":  pattern["confidence"],
                "severity":    pattern["severity"],
                "speaker":     speaker,
                "context":     context[:120],
            })

    # Commitment strength analysis
    weak_count   = sum(1 for m in WEAK_COMMITMENT_MARKERS   if m in text_lower)
    strong_count = sum(1 for m in STRONG_COMMITMENT_MARKERS if m in text_lower)

    commitment_score = "STRONG" if strong_count > weak_count else (
        "WEAK" if weak_count > 0 else "MODERATE"
    )

    # Risk assessment
    high   = [d for d in detected if d["severity"] == "HIGH"]
    medium = [d for d in detected if d["severity"] == "MEDIUM"]
    low    = [d for d in detected if d["severity"] == "LOW"]

    escalation = [d for d in detected if d["category"] == "ESCALATION"]
    power      = [d for d in detected if d["category"] == "POWER_IMBALANCE"]
    hedging    = [d for d in detected if d["category"] == "HEDGING"]
    passive    = [d for d in detected if d["category"] == "PASSIVE_AGGRESSION"]

    if escalation or len(high) >= 2:
        risk_level   = "HIGH"
        risk_summary = "Escalation signals or strong power imbalance detected. Relationship may be at risk."
    elif len(high) >= 1 or len(medium) >= 3:
        risk_level   = "MEDIUM"
        risk_summary = "Notable communication friction detected. Follow up to ensure alignment."
    elif len(medium) >= 1 or len(low) >= 3:
        risk_level   = "LOW"
        risk_summary = "Minor hedging or uncertainty detected. Communication is mostly on track."
    elif detected:
        risk_level   = "MINIMAL"
        risk_summary = "Minimal signals. Communication appears direct and productive."
    else:
        risk_level   = "NONE"
        risk_summary = "No friction signals detected. Direct and professional communication."

    return {
        "detected":          detected,
        "high_signals":      high,
        "medium_signals":    medium,
        "low_signals":       low,
        "total_signals":     len(detected),
        "escalation_signals":escalation,
        "power_signals":     power,
        "hedging_signals":   hedging,
        "passive_signals":   passive,
        "commitment_score":  commitment_score,
        "weak_commitments":  weak_count,
        "strong_commitments":strong_count,
        "risk_level":        risk_level,
        "risk_summary":      risk_summary,
        "cultural_note":     "In English business communication, directness is valued. Hedging and vague language often signals avoidance rather than politeness.",
    }

if __name__ == "__main__":
    import json
    sample = """
    Client: This is unacceptable. This is the second time the deadline was missed.
    Kenji: I understand your concern. Let me see what I can do. We'll circle back on this.
    Client: I'm going to have to escalate this. We may need to reconsider the contract.
    Kenji: With all due respect, we'll take that under advisement. I'll try to resolve it.
    """
    result = detect_english_patterns(sample)
    print(f"Risk: {result['risk_level']} — {result['risk_summary']}")
    for d in result["detected"]:
        print(f"  [{d['severity']}] {d['phrase']} ({d['category']}) — {d['reading']}")