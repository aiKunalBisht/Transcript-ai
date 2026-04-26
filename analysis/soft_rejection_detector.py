# soft_rejection_detector.py
# Detects indirect rejection and hesitation patterns in Japanese business speech
#
# In Japanese business culture, direct "No" is extremely rare.
# Instead, speakers use indirect patterns that signal rejection/hesitation:
#
#   検討します       → "We will consider it"  → likely means NO
#   難しいかもしれません → "It may be difficult"  → likely means NO
#   前向きに検討      → "Positive consideration" → maybe, leaning NO
#   確認してみます    → "I will check"          → deferring, uncertain
#
# This detector flags these patterns and explains their cultural meaning.
# Runs as post-processing on the transcript — adds soft_rejections to result.

import re


# ── PATTERN DICTIONARY ────────────────────────────────────────────────────────
# Each entry: phrase → (english meaning, true intent, confidence, severity)
# severity: HIGH = almost certainly No, MEDIUM = uncertain, LOW = slight hesitation

SOFT_REJECTION_PATTERNS = [
    # HIGH confidence rejections
    {
        "phrase":   "難しいかもしれません",
        "reading":  "It may be difficult",
        "intent":   "REJECTION",
        "confidence": 0.90,
        "severity": "HIGH",
        "explanation": "Classic soft rejection. Directly saying 'No' is culturally avoided. This almost always means the request will not be fulfilled."
    },
    {
        "phrase":   "難しい状況です",
        "reading":  "The situation is difficult",
        "intent":   "REJECTION",
        "confidence": 0.88,
        "severity": "HIGH",
        "explanation": "Situational difficulty framing — indirect way of declining."
    },
    {
        "phrase":   "ちょっと難しい",
        "reading":  "A little difficult",
        "intent":   "REJECTION",
        "confidence": 0.85,
        "severity": "HIGH",
        "explanation": "'A little' (ちょっと) is used to soften what is actually a firm refusal."
    },
    {
        "phrase":   "対応しかねます",
        "reading":  "We are unable to accommodate",
        "intent":   "REJECTION",
        "confidence": 0.95,
        "severity": "HIGH",
        "explanation": "One of the most direct soft rejections. Formal and definitive."
    },
    {
        "phrase":   "いたしかねます",
        "reading":  "We are unable to do that",
        "intent":   "REJECTION",
        "confidence": 0.95,
        "severity": "HIGH",
        "explanation": "Formal polite rejection. Very definitive despite polite framing."
    },
    # MEDIUM confidence — likely deferral or rejection
    {
        "phrase":   "検討します",
        "reading":  "We will consider it",
        "intent":   "LIKELY_REJECTION",
        "confidence": 0.72,
        "severity": "MEDIUM",
        "explanation": "In Japanese business, 'We will consider it' without a specific timeline almost always means No. If followed by a deadline, it may be genuine."
    },
    {
        "phrase":   "検討いたします",
        "reading":  "We will humbly consider it",
        "intent":   "LIKELY_REJECTION",
        "confidence": 0.72,
        "severity": "MEDIUM",
        "explanation": "Formal version of 検討します. Same implication — likely a polite No."
    },
    {
        "phrase":   "前向きに検討",
        "reading":  "Positive consideration",
        "intent":   "UNCERTAIN",
        "confidence": 0.55,
        "severity": "MEDIUM",
        "explanation": "'Positive consideration' sounds optimistic but is often used when the speaker cannot commit. Outcome genuinely uncertain."
    },
    {
        "phrase":   "善処します",
        "reading":  "I will handle it appropriately",
        "intent":   "LIKELY_REJECTION",
        "confidence": 0.68,
        "severity": "MEDIUM",
        "explanation": "Vague commitment with no concrete action. Often used to close a topic without agreeing."
    },
    {
        "phrase":   "確認してみます",
        "reading":  "I will try to confirm",
        "intent":   "UNCERTAIN",
        "confidence": 0.50,
        "severity": "MEDIUM",
        "explanation": "Genuine uncertainty or deferral. May indicate the speaker needs approval from a superior."
    },
    {
        "phrase":   "上司に相談",
        "reading":  "Will consult with my superior",
        "intent":   "UNCERTAIN",
        "confidence": 0.50,
        "severity": "MEDIUM",
        "explanation": "Escalation to superior — may be genuine or a delaying tactic. Watch for follow-up."
    },
    {
        "phrase":   "社内で確認",
        "reading":  "Will confirm internally",
        "intent":   "UNCERTAIN",
        "confidence": 0.48,
        "severity": "MEDIUM",
        "explanation": "Internal confirmation pending. Genuine uncertainty — decision not yet made."
    },
    # LOW confidence — mild hesitation signals
    {
        "phrase":   "少し懸念",
        "reading":  "A little concerned",
        "intent":   "HESITATION",
        "confidence": 0.40,
        "severity": "LOW",
        "explanation": "Signals discomfort or disagreement expressed indirectly. Worth probing further."
    },
    {
        "phrase":   "懸念がございます",
        "reading":  "There are concerns",
        "intent":   "HESITATION",
        "confidence": 0.45,
        "severity": "LOW",
        "explanation": "Formal expression of concern. The speaker disagrees but won't say so directly."
    },
    {
        "phrase":   "少し時間をいただけますか",
        "reading":  "Could I have a little time",
        "intent":   "UNCERTAIN",
        "confidence": 0.42,
        "severity": "LOW",
        "explanation": "Requesting delay. May indicate reluctance or need for internal approval."
    },
    {
        "phrase":   "そうですね",
        "reading":  "That's right / I see",
        "intent":   "HESITATION",
        "confidence": 0.25,
        "severity": "LOW",
        "explanation": "Ambiguous. Can be genuine agreement OR a filler to avoid disagreement. Context determines meaning."
    },
]

# Build fast lookup dict
_PHRASE_MAP = {p["phrase"]: p for p in SOFT_REJECTION_PATTERNS}


# ── CONTEXT EXTRACTOR ─────────────────────────────────────────────────────────
def _extract_context(transcript: str, phrase: str, window: int = 80) -> str:
    """Extract surrounding text for a detected phrase."""
    idx = transcript.find(phrase)
    if idx == -1:
        return ""
    start = max(0, idx - window)
    end   = min(len(transcript), idx + len(phrase) + window)
    context = transcript[start:end].strip()
    # Replace the phrase with a highlight marker
    return context.replace(phrase, f"【{phrase}】")


def _find_speaker(transcript: str, phrase: str) -> str:
    """Try to identify which speaker used the phrase."""
    idx = transcript.find(phrase)
    if idx == -1:
        return "Unknown"
    # Look backwards for the nearest speaker label
    preceding = transcript[:idx]
    # Match "SpeakerName:" or "[00:00] SpeakerName:"
    matches = re.findall(r"([^\s\[\]:：\n]+)\s*[:：]", preceding)
    # Filter out timestamps
    speakers = [m for m in matches if not re.match(r"^\d+$", m)]
    return speakers[-1] if speakers else "Unknown"


# ── MAIN DETECTOR ─────────────────────────────────────────────────────────────
def detect_soft_rejections(transcript: str) -> dict:
    """
    Scans transcript for soft rejection and hesitation patterns.

    Returns:
        dict with detected signals, risk assessment, and cultural explanations
    """
    detected = []
    rejection_count  = 0
    hesitation_count = 0
    uncertain_count  = 0

    for pattern in SOFT_REJECTION_PATTERNS:
        phrase = pattern["phrase"]
        if phrase in transcript:
            speaker = _find_speaker(transcript, phrase)
            context = _extract_context(transcript, phrase)

            signal = {
                "phrase":      phrase,
                "reading":     pattern["reading"],
                "intent":      pattern["intent"],
                "confidence":  pattern["confidence"],
                "severity":    pattern["severity"],
                "speaker":     speaker,
                "context":     context,
                "explanation": pattern["explanation"]
            }
            detected.append(signal)

            if pattern["intent"] == "REJECTION":
                rejection_count += 1
            elif pattern["intent"] == "LIKELY_REJECTION":
                rejection_count += 1
            elif pattern["intent"] == "HESITATION":
                hesitation_count += 1
            elif pattern["intent"] == "UNCERTAIN":
                uncertain_count += 1

    # Overall risk assessment
    total = len(detected)
    if rejection_count >= 2 or (rejection_count >= 1 and total >= 3):
        risk_level = "HIGH"
        risk_summary = "Multiple rejection signals detected. The meeting likely ended without agreement."
    elif rejection_count >= 1 or uncertain_count >= 2:
        risk_level = "MEDIUM"
        risk_summary = "Soft rejection or significant uncertainty detected. Follow up explicitly to confirm status."
    elif hesitation_count >= 2 or uncertain_count >= 1:
        risk_level = "LOW"
        risk_summary = "Some hesitation detected. The other party may have reservations not expressed directly."
    elif total > 0:
        risk_level = "MINIMAL"
        risk_summary = "Minor hesitation signals only. Likely proceeding normally."
    else:
        risk_level = "NONE"
        risk_summary = "No soft rejection patterns detected. Communication appears direct and positive."

    # Separate by severity for UI
    high_signals   = [s for s in detected if s["severity"] == "HIGH"]
    medium_signals = [s for s in detected if s["severity"] == "MEDIUM"]
    low_signals    = [s for s in detected if s["severity"] == "LOW"]

    return {
        "detected":        detected,
        "high_signals":    high_signals,
        "medium_signals":  medium_signals,
        "low_signals":     low_signals,
        "total_signals":   total,
        "rejection_count": rejection_count,
        "hesitation_count": hesitation_count,
        "uncertain_count": uncertain_count,
        "risk_level":      risk_level,
        "risk_summary":    risk_summary,
        "cultural_note":   (
            "In Japanese business culture, direct refusal is avoided. "
            "These patterns indicate the speaker's true intent through "
            "indirect language. Always follow up in writing to confirm."
        )
    }


# ── QUICK TEST ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    sample = """
    田中: Q3の提案についてご検討いただけますか？
    鈴木: そうですね、検討いたします。ただ、予算の面では少し懸念がございます。
    田中: 来週までにご回答いただけますか？
    鈴木: 難しいかもしれません。社内で確認してから、前向きに検討したいと思います。
    田中: 承知しました。では来月はいかがでしょうか？
    鈴木: 善処します。上司に相談してみます。
    """

    result = detect_soft_rejections(sample)

    print(f"Risk Level: {result['risk_level']}")
    print(f"Summary: {result['risk_summary']}")
    print(f"Total signals: {result['total_signals']}")
    print()

    for signal in result["detected"]:
        icon = {"HIGH": "🚨", "MEDIUM": "⚠️", "LOW": "💡"}.get(signal["severity"], "•")
        print(f"{icon} [{signal['severity']}] {signal['phrase']} → {signal['reading']}")
        print(f"   Speaker: {signal['speaker']}")
        print(f"   Intent: {signal['intent']} (confidence: {signal['confidence']:.0%})")
        print(f"   {signal['explanation']}")
        print()

    print(json.dumps(result, indent=2, ensure_ascii=False))