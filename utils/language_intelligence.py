# language_intelligence.py
# Language-aware feature routing for TranscriptAI
#
# Purpose: Different languages have different business communication patterns.
# This module determines which analysis features are relevant for each language
# and provides language-specific insight patterns.
#
# Supported:
#   Japanese (ja)  → Full Japan layer: keigo, nemawashi, soft rejection
#   Hindi (hi)     → Hindi business patterns: indirect refusal, formality signals
#   English (en)   → Clean analysis: summary, actions, sentiment only
#   Mixed (mixed)  → Smart combination based on dominant language

# ── LANGUAGE FEATURE FLAGS ────────────────────────────────────────────────────

LANGUAGE_FEATURES = {
    "ja": {
        "show_japan_insights":       True,
        "show_keigo":                True,
        "show_nemawashi":            True,
        "show_soft_rejection":       True,
        "show_code_switch":          False,   # no switch in pure JA
        "show_hindi_insights":       False,
        "insight_tab_label":         "🇯🇵 Japan Insights",
        "insight_tab_enabled":       True,
    },
    "mixed": {
        "show_japan_insights":       True,
        "show_keigo":                True,
        "show_nemawashi":            True,
        "show_soft_rejection":       True,
        "show_code_switch":          True,
        "show_hindi_insights":       False,
        "insight_tab_label":         "🇯🇵 Japan Insights",
        "insight_tab_enabled":       True,
    },
    "en": {
        "show_japan_insights":       False,
        "show_keigo":                False,
        "show_nemawashi":            False,
        "show_soft_rejection":       False,
        "show_code_switch":          False,
        "show_hindi_insights":       False,
        "insight_tab_label":         None,
        "insight_tab_enabled":       False,
    },
    "hi": {
        "show_japan_insights":       False,
        "show_keigo":                False,
        "show_nemawashi":            False,
        "show_soft_rejection":       False,
        "show_code_switch":          False,
        "show_hindi_insights":       True,
        "insight_tab_label":         "🇮🇳 Hindi Insights",
        "insight_tab_enabled":       True,
    },
}


def get_features(language: str) -> dict:
    """Returns feature flags for a detected language."""
    return LANGUAGE_FEATURES.get(language, LANGUAGE_FEATURES["en"])


# ── HINDI BUSINESS PATTERNS ───────────────────────────────────────────────────
# Hindi business communication has its own indirect patterns.
# These are different from Japanese nemawashi but serve a similar social function.

HINDI_INDIRECT_PATTERNS = [
    {
        "phrase":      "देखते हैं",
        "reading":     "Let's see / We'll see",
        "intent":      "UNCERTAIN",
        "confidence":  0.65,
        "severity":    "MEDIUM",
        "explanation": (
            "Common deferral phrase. Often means the speaker has reservations "
            "but won't say so directly. Follow up with a specific timeline."
        )
    },
    {
        "phrase":      "कोशिश करेंगे",
        "reading":     "We will try",
        "intent":      "UNCERTAIN",
        "confidence":  0.60,
        "severity":    "MEDIUM",
        "explanation": (
            "'Will try' without a commitment date is a soft hedge. "
            "In formal business Hindi this often signals low priority."
        )
    },
    {
        "phrase":      "सोचना पड़ेगा",
        "reading":     "Will have to think about it",
        "intent":      "LIKELY_REJECTION",
        "confidence":  0.70,
        "severity":    "MEDIUM",
        "explanation": (
            "Polite way to avoid a direct No. Similar to Japanese 検討します. "
            "Rarely followed by action unless a deadline is attached."
        )
    },
    {
        "phrase":      "बाद में बात करते हैं",
        "reading":     "Let's talk later",
        "intent":      "DEFERRAL",
        "confidence":  0.55,
        "severity":    "LOW",
        "explanation": (
            "Topic deferral. May be genuine scheduling or avoidance. "
            "Watch for whether follow-up actually happens."
        )
    },
    {
        "phrase":      "थोड़ा मुश्किल है",
        "reading":     "It's a little difficult",
        "intent":      "REJECTION",
        "confidence":  0.80,
        "severity":    "HIGH",
        "explanation": (
            "In Hindi business context, 'a little difficult' is a strong "
            "negative signal. Direct rejection is avoided out of politeness."
        )
    },
    {
        "phrase":      "ऊपर से approval लेना होगा",
        "reading":     "Will need approval from above",
        "intent":      "UNCERTAIN",
        "confidence":  0.50,
        "severity":    "MEDIUM",
        "explanation": (
            "Escalation to senior management — may be genuine or a polite "
            "way to delay. Common in hierarchical Indian business culture."
        )
    },
    {
        "phrase":      "हम विचार करेंगे",
        "reading":     "We will consider",
        "intent":      "LIKELY_REJECTION",
        "confidence":  0.68,
        "severity":    "MEDIUM",
        "explanation": (
            "Formal Hindi equivalent of 'we will consider it'. "
            "Without a specific timeline, this rarely leads to action."
        )
    },
    {
        "phrase":      "अभी नहीं",
        "reading":     "Not right now",
        "intent":      "REJECTION",
        "confidence":  0.85,
        "severity":    "HIGH",
        "explanation": (
            "Clearer rejection. 'Not right now' in formal business often "
            "means not at all, framed as a timing issue."
        )
    },
]

# Hindi formality markers (rough equivalent of keigo)
HINDI_FORMAL_MARKERS = [
    "आपका", "आपकी", "आपके", "जी", "श्रीमान", "महोदय",
    "सहयोग", "कृपया", "धन्यवाद", "शुक्रिया", "नमस्ते",
]

HINDI_CASUAL_MARKERS = [
    "यार", "भाई", "बोलो", "सुनो", "देखो",
]


def detect_hindi_patterns(transcript: str) -> dict:
    """
    Detects indirect communication patterns in Hindi business speech.
    Returns structured output matching the soft_rejections schema.
    """
    detected = []
    rejection_count = 0
    uncertain_count = 0

    for pattern in HINDI_INDIRECT_PATTERNS:
        if pattern["phrase"] in transcript:
            # Find speaker context
            import re
            idx = transcript.find(pattern["phrase"])
            start = max(0, idx - 100)
            end   = min(len(transcript), idx + len(pattern["phrase"]) + 100)
            context = transcript[start:end].replace(
                pattern["phrase"], f"【{pattern['phrase']}】"
            )

            # Find speaker
            speaker = "Unknown"
            preceding = transcript[:idx]
            speakers = re.findall(r"([^\n:：]+)\s*[:：]", preceding)
            if speakers:
                speaker = speakers[-1].strip()

            detected.append({
                **pattern,
                "speaker": speaker,
                "context": context,
            })

            if pattern["intent"] in ("REJECTION", "LIKELY_REJECTION"):
                rejection_count += 1
            else:
                uncertain_count += 1

    # Formality level
    formal_count   = sum(1 for m in HINDI_FORMAL_MARKERS  if m in transcript)
    casual_count   = sum(1 for m in HINDI_CASUAL_MARKERS  if m in transcript)
    formality = "high" if formal_count >= 3 else "medium" if formal_count >= 1 else "low"

    # Risk level
    if rejection_count >= 2:
        risk_level   = "HIGH"
        risk_summary = "Multiple indirect rejection signals detected."
    elif rejection_count >= 1 or uncertain_count >= 2:
        risk_level   = "MEDIUM"
        risk_summary = "Indirect refusal or significant uncertainty detected. Follow up explicitly."
    elif uncertain_count >= 1:
        risk_level   = "LOW"
        risk_summary = "Some hesitation signals present."
    elif detected:
        risk_level   = "MINIMAL"
        risk_summary = "Minor indirect signals only."
    else:
        risk_level   = "NONE"
        risk_summary = "No indirect communication patterns detected."

    return {
        "language":        "hindi",
        "detected":        detected,
        "total_signals":   len(detected),
        "rejection_count": rejection_count,
        "uncertain_count": uncertain_count,
        "formality_level": formality,
        "risk_level":      risk_level,
        "risk_summary":    risk_summary,
        "cultural_note": (
            "Hindi business communication uses indirect refusal similar to Japanese nemawashi. "
            "Direct 'No' is uncommon in formal settings. Watch for hedging language and escalation phrases."
        )
    }


# ── PROMPT BUILDER — language-aware ───────────────────────────────────────────

def build_language_aware_prompt_suffix(language: str) -> str:
    """
    Returns additional prompt instructions based on detected language.
    Injected into analyzer.py build_prompt() to give language-specific guidance.
    """
    if language == "ja":
        return (
            "This is a Japanese business meeting. Pay special attention to:\n"
            "- Indirect refusal patterns (検討します, 難しいかもしれません)\n"
            "- Honorific speech levels (敬語, 謙譲語, 丁寧語)\n"
            "- Consensus-building phrases (根回し)\n"
            "Sentiment for Japanese speakers: neutral IS the professional standard. "
            "Only mark positive if there is explicit enthusiasm."
        )
    elif language == "hi":
        return (
            "This is a Hindi business conversation. Pay attention to:\n"
            "- Indirect refusal (थोड़ा मुश्किल है, देखते हैं, सोचना पड़ेगा)\n"
            "- Formality markers (आपका, जी, श्रीमान)\n"
            "- Hierarchical escalation (ऊपर से approval)\n"
            "Extract action items and sentiment as clearly as possible."
        )
    elif language == "mixed":
        return (
            "This transcript mixes multiple languages. "
            "Identify each speaker's primary language and apply appropriate cultural context. "
            "For Japanese speakers: watch for indirect patterns. "
            "For English speakers: be direct in sentiment assessment."
        )
    else:
        return (
            "This is an English business meeting. "
            "Focus on clarity: extract concrete action items with owners and deadlines. "
            "Sentiment should reflect tone directly — no cultural hedging needed."
        )


if __name__ == "__main__":
    import json

    # Test Hindi detection
    sample_hi = """
    Rahul: नमस्ते, आज की meeting में हम Q3 budget discuss करेंगे।
    Priya: जी, मैंने report तैयार किया है।
    Rahul: Budget increase के बारे में क्या सोचते हैं?
    Priya: थोड़ा मुश्किल है। हम देखते हैं। ऊपर से approval लेना होगा।
    Rahul: Okay, तो कब तक बता सकते हैं?
    Priya: हम विचार करेंगे। बाद में बात करते हैं।
    """

    result = detect_hindi_patterns(sample_hi)
    print(f"Risk: {result['risk_level']}")
    print(f"Signals: {result['total_signals']}")
    for s in result["detected"]:
        print(f"  {s['severity']}: {s['phrase']} → {s['reading']} ({s['confidence']:.0%})")

    print("\nFeature flags per language:")
    for lang in ["en", "ja", "hi", "mixed"]:
        flags = get_features(lang)
        active = [k for k, v in flags.items() if v is True]
        print(f"  {lang}: {active}")